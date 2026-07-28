"""
Semantic Scholar API Tools
==========================

Wraps the free Semantic Scholar Academic Graph API.
No auth required for basic use; supply SEMANTIC_SCHOLAR_API_KEY for
higher rate limits (10 req/s vs 1 req/s without key).

API docs: https://api.semanticscholar.org/graph/v1
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.tool import Tool
from src.observability.langfuse_tracing import trace_span

BASE_URL = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "authors",
    "fieldsOfStudy",
])


def _get_headers() -> dict[str, str]:
    """Return request headers, including the API key if available."""
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _handle_rate_limit(response: requests.Response) -> None:
    """If we hit a 429, sleep 5 s before the retry machinery takes over."""
    if response.status_code == 429:
        logger.warning("Semantic Scholar rate limit hit (429). Sleeping 5s before retry.")
        time.sleep(5)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_papers(query: str, limit: int = 20) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query string.

    Parameters
    ----------
    query : str
        Free-text search query (e.g. "OAuth2 token security attacks").
    limit : int
        Maximum number of results to return (default 20, max 100).

    Returns
    -------
    list[dict]
        Each dict contains: paperId, title, abstract, year, citationCount,
        authors (list of {authorId, name}), fieldsOfStudy (list of str).
    """
    limit = min(limit, 100)
    params = {
        "query": query,
        "limit": limit,
        "fields": PAPER_FIELDS,
    }

    logger.debug(f"Semantic Scholar search: query={query!r}, limit={limit}")
    with trace_span(
        "semantic-scholar-search",
        input_data={"query": query, "limit": limit},
        metadata={"api": "semantic_scholar"},
    ) as span:
        response = requests.get(
            f"{BASE_URL}/paper/search",
            params=params,
            headers=_get_headers(),
            timeout=20,
        )

        _handle_rate_limit(response)
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        papers: list[dict] = data.get("data", [])
        if span is not None:
            span.update(output={"result_count": len(papers)})

    # Normalise authors to a simple list of name strings for easier handling
    for paper in papers:
        if "authors" in paper and isinstance(paper["authors"], list):
            paper["authors"] = [a.get("name", "") for a in paper["authors"]]
        if paper.get("abstract") is None:
            paper["abstract"] = ""
        if paper.get("fieldsOfStudy") is None:
            paper["fieldsOfStudy"] = []

    logger.debug(f"Semantic Scholar returned {len(papers)} papers for query={query!r}")
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_paper_details(paper_id: str) -> dict:
    """
    Fetch full metadata for a single paper by its Semantic Scholar paperId.

    Parameters
    ----------
    paper_id : str
        The Semantic Scholar paperId (e.g. "204e3073870fae3d05bcbc2f6a8e263d9b72e776").

    Returns
    -------
    dict
        Paper metadata (same fields as search_papers results).

    Raises
    ------
    requests.HTTPError
        If the paper is not found or the API returns an error.
    """
    logger.debug(f"Semantic Scholar get_paper_details: paper_id={paper_id!r}")
    with trace_span(
        "semantic-scholar-get-paper",
        input_data={"paper_id": paper_id},
        metadata={"api": "semantic_scholar"},
    ) as span:
        response = requests.get(
            f"{BASE_URL}/paper/{paper_id}",
            params={"fields": PAPER_FIELDS},
            headers=_get_headers(),
            timeout=20,
        )

        _handle_rate_limit(response)
        response.raise_for_status()

        paper: dict = response.json()
        if span is not None:
            span.update(output={"title": paper.get("title", "")})
    if "authors" in paper and isinstance(paper["authors"], list):
        paper["authors"] = [a.get("name", "") for a in paper["authors"]]
    if paper.get("abstract") is None:
        paper["abstract"] = ""
    if paper.get("fieldsOfStudy") is None:
        paper["fieldsOfStudy"] = []

    return paper


# ---------------------------------------------------------------------------
# Tool wrappers (for agent tool-calling)
# ---------------------------------------------------------------------------

def _search_papers_tool(query: str, limit: int = 20) -> str:
    """Thin wrapper that serialises the result to JSON for the agent loop."""
    import json
    results = search_papers(query=query, limit=limit)
    return json.dumps(results, ensure_ascii=False)


def _get_paper_details_tool(paper_id: str) -> str:
    """Thin wrapper that serialises the result to JSON for the agent loop."""
    import json
    result = get_paper_details(paper_id=paper_id)
    return json.dumps(result, ensure_ascii=False)


SEARCH_PAPERS_TOOL = Tool(
    name="search_papers",
    description=(
        "Search Semantic Scholar for academic papers. "
        "Returns a JSON list of papers with title, abstract, year, citation count, authors, and fields of study."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string (e.g. 'OAuth2 token replay attacks')",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of papers to return (default 20, max 100)",
            },
        },
        "required": ["query"],
    },
    func=_search_papers_tool,
)

GET_PAPER_DETAILS_TOOL = Tool(
    name="get_paper_details",
    description=(
        "Fetch full metadata for a single paper using its Semantic Scholar paperId. "
        "Use this when you need details (abstract, citation count) for a specific paper."
    ),
    parameters={
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "The Semantic Scholar paperId string",
            },
        },
        "required": ["paper_id"],
    },
    func=_get_paper_details_tool,
)
