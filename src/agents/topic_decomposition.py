"""
Topic Decomposition Agent
=========================

Takes a broad research topic and breaks it into 4-6 focused keyword
clusters, each covering a distinct sub-theme of the topic space.

No tools required — this is a pure LLM call with structured JSON output.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from src.core.state import PipelineState
from src.llm.providers import LLMProvider

SYSTEM_PROMPT = """You are a research librarian specialising in academic literature search.
Your task is to decompose a research topic into distinct keyword clusters for database searching.

Each cluster should cover a unique sub-theme. Good clusters are:
- Non-overlapping (each targets a different facet of the topic)
- Specific enough to retrieve relevant papers on Semantic Scholar
- Broad enough to not miss important work

You MUST respond with valid JSON only — no prose, no markdown fences."""

USER_TEMPLATE = """Decompose this research topic into 4-6 keyword clusters for Semantic Scholar searches.

Topic: {topic}

Return a JSON array where each element has exactly these fields:
- "theme": short label for the sub-theme (e.g. "OAuth2 Authorization Flows")
- "keywords": list of 3-5 search terms or short phrases to use as queries
- "description": one sentence explaining what papers this cluster will find

Example format:
[
  {{
    "theme": "Token Security Attacks",
    "keywords": ["JWT token hijacking", "bearer token theft", "session token replay"],
    "description": "Papers on how authentication tokens are attacked and exploited."
  }}
]

Respond with the JSON array only."""


def run(state: PipelineState, provider: LLMProvider) -> PipelineState:
    """
    Run the Topic Decomposition agent and update state.keyword_clusters.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state. Reads state.topic.
    provider : LLMProvider
        The configured LLM provider.

    Returns
    -------
    PipelineState
        Updated state with keyword_clusters populated.
    """
    logger.info(f"[TopicDecomposition] Decomposing topic: {state.topic!r}")

    user_message = USER_TEMPLATE.format(topic=state.topic)

    response = provider.call(
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        tools=[],
    )

    raw_content: str = response.get("content", "")
    clusters = _parse_clusters(raw_content)

    if not clusters:
        raise ValueError(
            f"[TopicDecomposition] Failed to parse keyword clusters from LLM response.\n"
            f"Raw response:\n{raw_content}"
        )

    state.keyword_clusters = clusters
    logger.info(f"[TopicDecomposition] Generated {len(clusters)} keyword clusters:")
    for i, c in enumerate(clusters, 1):
        logger.info(f"  {i}. {c['theme']} — keywords: {c['keywords']}")

    return state


def _parse_clusters(raw: str) -> list[dict]:
    """
    Extract and validate the JSON cluster array from the LLM response.

    Handles cases where the model wraps the JSON in markdown code fences.

    Parameters
    ----------
    raw : str
        Raw LLM output.

    Returns
    -------
    list[dict]
        Validated list of cluster dicts, or empty list if parsing failed.
    """
    # Strip markdown fences if present (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Find the first '[' to handle any leading prose
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        logger.error(f"[TopicDecomposition] No JSON array found in response: {raw[:200]!r}")
        return []

    json_str = cleaned[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error(f"[TopicDecomposition] JSON parse error: {exc}\nInput: {json_str[:300]!r}")
        return []

    if not isinstance(data, list):
        logger.error("[TopicDecomposition] Expected JSON array, got: %s", type(data))
        return []

    validated: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "theme" not in item or "keywords" not in item:
            logger.warning(f"[TopicDecomposition] Skipping malformed cluster: {item}")
            continue
        # Ensure description is present
        if "description" not in item:
            item["description"] = ""
        # Ensure keywords is a list
        if isinstance(item["keywords"], str):
            item["keywords"] = [item["keywords"]]
        validated.append(item)

    return validated
