"""
Enrichment Agent
================

Takes raw papers (title + abstract) and asks the LLM to assign:
  - relevance_score (1–5)
  - methodology (empirical / survey / formal analysis / case study / other)
  - contribution_type (attack / defense / protocol / framework / tool / other)
  - priority_read (bool)
  - one_line_summary (str)

Processes papers in batches of `batch_size` to stay within context limits.
Uses direct structured JSON output — no tool calling needed.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from src.core.state import PipelineState
from src.llm.providers import LLMProvider
from src.observability.langfuse_tracing import trace_span

SYSTEM_PROMPT = """You are a research analyst reviewing academic papers.
For each paper provided, you will output structured metadata to help prioritise reading.

You MUST respond with valid JSON only — no prose, no markdown fences, no explanations."""

BATCH_USER_TEMPLATE = """Analyse the following {n} academic papers and return a JSON array with one object per paper.

Each object must have exactly these fields:
- "paperId": the paper's ID (copy from input)
- "relevance_score": integer 1-5 (5 = highly relevant to the research topic)
- "methodology": one of ["empirical", "survey", "formal analysis", "case study", "simulation", "theoretical", "other"]
- "contribution_type": one of ["attack", "defense", "protocol", "framework", "tool", "dataset", "survey", "other"]
- "priority_read": boolean — true if this is a must-read paper
- "one_line_summary": single sentence summarising the paper's main contribution

Research topic: {topic}

Papers:
{papers_json}

Return the JSON array only."""


def run(
    state: PipelineState,
    provider: LLMProvider,
    batch_size: int = 8,
) -> PipelineState:
    """
    Run the Enrichment agent on all raw papers in state.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state. Reads state.papers_raw and state.topic.
    provider : LLMProvider
        The configured LLM provider.
    batch_size : int
        Number of papers to enrich per LLM call (default 8).

    Returns
    -------
    PipelineState
        Updated state with papers_enriched populated.
    """
    papers = state.papers_raw
    if not papers:
        raise ValueError("[Enrichment] papers_raw is empty — run Discovery first.")

    total = len(papers)
    batches = _make_batches(papers, batch_size)
    logger.info(f"[Enrichment] Enriching {total} papers in {len(batches)} batch(es) of {batch_size}.")

    # Build a lookup from paperId → enrichment fields
    enrichment_map: dict[str, dict] = {}

    for batch_idx, batch in enumerate(batches, 1):
        logger.info(f"[Enrichment] Batch {batch_idx}/{len(batches)} ({len(batch)} papers)")

        with trace_span(
            f"enrichment-batch-{batch_idx}",
            input_data={"batch_idx": batch_idx, "batch_size": len(batch)},
        ) as batch_span:
            batch_results = _enrich_batch(
                batch=batch,
                batch_idx=batch_idx,
                total_batches=len(batches),
                topic=state.topic,
                provider=provider,
            )
            if batch_span is not None:
                batch_span.update(output={"parsed_count": len(batch_results)})

        for result in batch_results:
            pid = result.get("paperId")
            if pid:
                enrichment_map[pid] = result

        logger.debug(f"[Enrichment] Batch {batch_idx}: parsed {len(batch_results)} results")

    # Merge enrichment fields back into the paper dicts
    enriched_papers: list[dict] = []
    for paper in papers:
        pid = paper.get("paperId")
        enriched = dict(paper)
        if pid and pid in enrichment_map:
            fields = enrichment_map[pid]
            enriched["relevance_score"] = fields.get("relevance_score", 3)
            enriched["methodology"] = fields.get("methodology", "other")
            enriched["contribution_type"] = fields.get("contribution_type", "other")
            enriched["priority_read"] = fields.get("priority_read", False)
            enriched["one_line_summary"] = fields.get("one_line_summary", "")
        else:
            # Fallback defaults for papers the LLM didn't return
            enriched["relevance_score"] = 3
            enriched["methodology"] = "other"
            enriched["contribution_type"] = "other"
            enriched["priority_read"] = False
            enriched["one_line_summary"] = ""
            state.add_error(f"[Enrichment] No enrichment returned for paperId={pid!r}")

        enriched_papers.append(enriched)

    state.papers_enriched = enriched_papers
    priority_count = sum(1 for p in enriched_papers if p.get("priority_read"))
    logger.info(f"[Enrichment] Done. {len(enriched_papers)} papers enriched, {priority_count} flagged as priority.")
    return state


def _enrich_batch(
    *,
    batch: list[dict],
    batch_idx: int,
    total_batches: int,
    topic: str,
    provider: LLMProvider,
) -> list[dict]:
    """Run a single enrichment batch LLM call and parse the response."""
    # Build a minimal representation of each paper for the prompt
    papers_for_prompt = [
        {
            "paperId": p.get("paperId", f"unknown_{i}"),
            "title": p.get("title", ""),
            "abstract": (p.get("abstract") or "")[:800],  # truncate to save tokens
            "year": p.get("year"),
            "citationCount": p.get("citationCount", 0),
        }
        for i, p in enumerate(batch)
    ]

    prompt = BATCH_USER_TEMPLATE.format(
        n=len(batch),
        topic=topic,
        papers_json=json.dumps(papers_for_prompt, indent=2, ensure_ascii=False),
    )

    response = provider.call(
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )

    raw = response.get("content", "")
    return _parse_enrichment(raw, batch)


def _make_batches(papers: list[dict], batch_size: int) -> list[list[dict]]:
    """Split a flat list into sub-lists of at most batch_size."""
    return [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]


def _parse_enrichment(raw: str, batch: list[dict]) -> list[dict]:
    """
    Parse the LLM's JSON response for a batch of papers.

    Falls back to defaults for malformed JSON so one bad batch doesn't
    crash the entire enrichment run.

    Parameters
    ----------
    raw : str
        Raw LLM text output.
    batch : list[dict]
        Original paper dicts (used to fill in fallback defaults).

    Returns
    -------
    list[dict]
        List of enrichment dicts with paperId and enrichment fields.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        logger.error(f"[Enrichment] No JSON array found in batch response: {raw[:300]!r}")
        return _fallback_enrichment(batch)

    json_str = cleaned[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error(f"[Enrichment] JSON parse error: {exc}. Input: {json_str[:300]!r}")
        return _fallback_enrichment(batch)

    if not isinstance(data, list):
        logger.error("[Enrichment] Expected list, got: %s", type(data))
        return _fallback_enrichment(batch)

    return data


def _fallback_enrichment(batch: list[dict]) -> list[dict]:
    """Return safe default enrichment for all papers in a failed batch."""
    return [
        {
            "paperId": p.get("paperId", ""),
            "relevance_score": 3,
            "methodology": "other",
            "contribution_type": "other",
            "priority_read": False,
            "one_line_summary": "(enrichment failed)",
        }
        for p in batch
    ]
