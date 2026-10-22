"""
Discovery Agent
===============

Searches keyword clusters in rounds, then selects a balanced, deduplicated
corpus capped at max_total_papers to control downstream LLM cost. Every
observed cluster/query source is retained on selected papers.
"""

from __future__ import annotations

import json
from collections import deque

from loguru import logger

from src.core.agent import run_agent_loop
from src.core.errors import safe_exception_summary
from src.core.state import PipelineState
from src.llm.providers import LLMProvider
from src.tools.semantic_scholar import SEARCH_PAPERS_TOOL, search_papers

SYSTEM_PROMPT = """You are a research discovery agent. Your job is to find relevant academic papers
on Semantic Scholar for a given set of keyword clusters.

For each cluster you are given, call the search_papers tool with each of the cluster's keywords.
After collecting all results, respond with a JSON summary indicating how many papers you found.

Be thorough — run a search for every keyword in every cluster.
"""


def run(
    state: PipelineState,
    provider: LLMProvider,
    results_per_query: int = 20,
    max_total_papers: int = 80,
    max_iterations: int = 10,
) -> PipelineState:
    """
    Run the Discovery agent and update state.papers_raw.

    Calls the search API once per keyword, one keyword per cluster per round.
    Fetching candidates is independent of the selected-paper cap so later
    clusters and duplicate provenance cannot be hidden by an early full cap.
    Selection takes one unique paper per cluster per round: this reserves an
    initial floor(cap / clusters) share for productive clusters, then gives
    unused slots to clusters with remaining candidates. Ties favor input order.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state. Reads state.keyword_clusters.
    provider : LLMProvider
        The configured LLM provider (not used for direct search mode).
    results_per_query : int
        How many results to request per keyword search.
    max_total_papers : int
        Hard cap on total unique papers collected.

    Returns
    -------
    PipelineState
        Updated state with papers_raw populated.
    """
    if not state.keyword_clusters:
        raise ValueError("[Discovery] keyword_clusters is empty — run TopicDecomposition first.")
    for name, value in (
        ("results_per_query", results_per_query),
        ("max_total_papers", max_total_papers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    logger.info(f"[Discovery] Starting search over {len(state.keyword_clusters)} clusters.")

    candidates: dict[str, dict] = {}
    cluster_candidates: list[dict[str, None]] = [{} for _ in state.keyword_clusters]
    fetch_limit = min(results_per_query, 100)  # Semantic Scholar's per-query maximum.
    for round_index in range(max(len(c.keywords) for c in state.keyword_clusters)):
        for cluster_index, cluster in enumerate(state.keyword_clusters):
            if round_index >= len(cluster.keywords):
                continue
            keyword = cluster.keywords[round_index]
            try:
                papers = search_papers(query=keyword, limit=fetch_limit)
            except Exception as exc:
                msg = (
                    f"[Discovery] Search failed for keyword {keyword!r}: "
                    f"{safe_exception_summary(exc)}"
                )
                logger.warning(msg)
                state.add_error(msg)
                continue

            for paper in papers:
                paper_id = paper.get("paperId")
                if not paper_id or not paper.get("title"):
                    continue
                if paper_id not in candidates:
                    candidates[paper_id] = {
                        **paper,
                        "_source_cluster": cluster.theme,
                        "_discovery_sources": [],
                    }
                source = {"cluster": cluster.theme, "query": keyword}
                sources = candidates[paper_id]["_discovery_sources"]
                if source not in sources:
                    sources.append(source)
                cluster_candidates[cluster_index][paper_id] = None

    queues = [deque(ids) for ids in cluster_candidates]
    selected: set[str] = set()
    all_papers: list[dict] = []
    while len(all_papers) < max_total_papers:
        previous_count = len(all_papers)
        for queue in queues:
            # Shared papers occupy one global slot, regardless of source count.
            while queue and queue[0] in selected:
                queue.popleft()
            if queue:
                paper_id = queue.popleft()
                selected.add(paper_id)
                all_papers.append(candidates[paper_id])
            if len(all_papers) >= max_total_papers:
                break
        if len(all_papers) == previous_count:
            break

    logger.info(f"[Discovery] Collected {len(all_papers)} unique papers.")
    state.papers_raw = all_papers
    return state


def run_with_agent_loop(
    state: PipelineState,
    provider: LLMProvider,
    max_iterations: int = 15,
) -> PipelineState:
    """
    Alternative: run Discovery via the agent loop (LLM drives the searches).

    Useful for exploring topics where the LLM can adaptively refine queries
    based on early results. Slower and uses more tokens than the direct mode.

    Parameters
    ----------
    state : PipelineState
    provider : LLMProvider
    max_iterations : int

    Returns
    -------
    PipelineState
    """
    clusters_json = json.dumps(
        [cluster.model_dump() for cluster in state.keyword_clusters], indent=2
    )
    user_message = (
        f"Search for papers across these keyword clusters:\n\n{clusters_json}\n\n"
        "Use the search_papers tool for each keyword. "
        "When done, respond with: DONE"
    )

    run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_PAPERS_TOOL],
        initial_messages=[{"role": "user", "content": user_message}],
        provider=provider,
        max_iterations=max_iterations,
    )

    # Note: in this mode the agent loop calls the real search_papers function
    # via the tool, but results are not automatically captured into state.
    # This mode is provided as a reference; the direct mode (run()) is preferred.
    logger.warning("[Discovery] run_with_agent_loop: paper results are not captured into state in this mode.")
    return state
