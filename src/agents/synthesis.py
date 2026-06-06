"""
Synthesis Agent
===============

Reads all enriched papers and synthesises:
  - key_themes: recurring themes across the literature
  - research_gaps: what the field has not yet addressed
  - recommended_future_work: concrete directions for future research
  - suggested_reading_order: a prioritised reading list with rationale
  - summary_paragraph: a prose overview of the field

For large paper sets (>50), does two-pass synthesis to stay within
context limits: summarises halves separately, then combines.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from src.core.state import PipelineState
from src.llm.providers import LLMProvider

TWO_PASS_THRESHOLD = 50

SYSTEM_PROMPT = """You are a senior research scientist writing a literature review synthesis.
Your output will help researchers quickly understand the landscape of a field.
You MUST respond with valid JSON only — no prose, no markdown fences."""

SYNTHESIS_TEMPLATE = """You have analysed {n} academic papers on the topic: "{topic}"

Here are the enriched paper summaries:
{papers_json}

Return a JSON object with exactly these fields:
- "key_themes": list of 4-8 strings describing the main themes across the literature
- "research_gaps": list of 3-6 strings describing what the field has NOT yet addressed
- "recommended_future_work": list of 3-5 strings with concrete future research directions
- "suggested_reading_order": list of objects, each with:
    - "paperId": str
    - "title": str
    - "reason": str (why this paper should be read at this position)
  Include 8-12 papers in the suggested order, prioritising high-relevance and high-citation papers.
- "summary_paragraph": 3-5 sentence prose overview of the field's current state

Return the JSON object only."""

PARTIAL_SYNTHESIS_TEMPLATE = """Synthesise these {n} papers (partial batch {batch_num} of 2) on topic: "{topic}"

{papers_json}

Return a JSON object with:
- "key_themes": list of strings
- "research_gaps": list of strings
- "notable_papers": list of {{paperId, title, reason}} for top 5 papers from this batch

Return JSON only."""

COMBINE_TEMPLATE = """Combine these two partial literature review syntheses into one final synthesis.
Topic: "{topic}"

Partial synthesis 1:
{partial1}

Partial synthesis 2:
{partial2}

Return a JSON object with exactly these fields:
- "key_themes": list of 4-8 strings (merged and deduplicated)
- "research_gaps": list of 3-6 strings
- "recommended_future_work": list of 3-5 strings
- "suggested_reading_order": list of {{paperId, title, reason}} for 8-12 top papers total
- "summary_paragraph": 3-5 sentence prose overview

Return JSON only."""


def run(state: PipelineState, provider: LLMProvider) -> PipelineState:
    """
    Run the Synthesis agent on enriched papers.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state. Reads state.papers_enriched and state.topic.
    provider : LLMProvider
        The configured LLM provider.

    Returns
    -------
    PipelineState
        Updated state with synthesis populated.
    """
    papers = state.papers_enriched
    if not papers:
        raise ValueError("[Synthesis] papers_enriched is empty — run Enrichment first.")

    logger.info(f"[Synthesis] Synthesising {len(papers)} enriched papers.")

    if len(papers) > TWO_PASS_THRESHOLD:
        logger.info(f"[Synthesis] Paper count > {TWO_PASS_THRESHOLD} — using two-pass synthesis.")
        synthesis = _two_pass_synthesis(papers, state.topic, provider)
    else:
        synthesis = _single_pass_synthesis(papers, state.topic, provider)

    if not synthesis:
        raise ValueError("[Synthesis] Failed to produce a valid synthesis response.")

    state.synthesis = synthesis
    logger.info(
        f"[Synthesis] Done. "
        f"Themes={len(synthesis.get('key_themes', []))}, "
        f"Gaps={len(synthesis.get('research_gaps', []))}, "
        f"Reading order={len(synthesis.get('suggested_reading_order', []))} papers."
    )
    return state


def _single_pass_synthesis(papers: list[dict], topic: str, provider: LLMProvider) -> dict:
    """Synthesise all papers in a single LLM call."""
    papers_summary = _papers_to_prompt_format(papers)
    prompt = SYNTHESIS_TEMPLATE.format(
        n=len(papers),
        topic=topic,
        papers_json=json.dumps(papers_summary, indent=2, ensure_ascii=False),
    )

    response = provider.call(
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    return _parse_synthesis(response.get("content", ""))


def _two_pass_synthesis(papers: list[dict], topic: str, provider: LLMProvider) -> dict:
    """Split papers into two halves, synthesise each, then combine."""
    mid = len(papers) // 2
    halves = [papers[:mid], papers[mid:]]
    partials: list[str] = []

    for i, half in enumerate(halves, 1):
        logger.info(f"[Synthesis] Two-pass: processing batch {i}/2 ({len(half)} papers)")
        papers_summary = _papers_to_prompt_format(half)
        prompt = PARTIAL_SYNTHESIS_TEMPLATE.format(
            n=len(half),
            batch_num=i,
            topic=topic,
            papers_json=json.dumps(papers_summary, indent=2, ensure_ascii=False),
        )
        response = provider.call(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        partials.append(response.get("content", "{}"))

    logger.info("[Synthesis] Two-pass: combining partial syntheses.")
    combine_prompt = COMBINE_TEMPLATE.format(
        topic=topic,
        partial1=partials[0],
        partial2=partials[1],
    )
    combine_response = provider.call(
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": combine_prompt}],
        tools=[],
    )
    return _parse_synthesis(combine_response.get("content", ""))


def _papers_to_prompt_format(papers: list[dict]) -> list[dict]:
    """Return a minimal, token-efficient representation of each paper."""
    return [
        {
            "paperId": p.get("paperId", ""),
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citationCount": p.get("citationCount", 0),
            "relevance_score": p.get("relevance_score", 3),
            "methodology": p.get("methodology", "other"),
            "contribution_type": p.get("contribution_type", "other"),
            "one_line_summary": p.get("one_line_summary", ""),
        }
        for p in papers
    ]


def _parse_synthesis(raw: str) -> dict:
    """
    Extract and validate the JSON synthesis object from LLM output.

    Parameters
    ----------
    raw : str
        Raw LLM text.

    Returns
    -------
    dict
        Parsed synthesis dict, or empty dict if parsing failed.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        logger.error(f"[Synthesis] No JSON object found in response: {raw[:300]!r}")
        return {}

    json_str = cleaned[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error(f"[Synthesis] JSON parse error: {exc}. Input: {json_str[:300]!r}")
        return {}

    if not isinstance(data, dict):
        logger.error("[Synthesis] Expected dict, got: %s", type(data))
        return {}

    # Ensure required keys have default values
    data.setdefault("key_themes", [])
    data.setdefault("research_gaps", [])
    data.setdefault("recommended_future_work", [])
    data.setdefault("suggested_reading_order", [])
    data.setdefault("summary_paragraph", "")
    return data
