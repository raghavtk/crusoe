"""Offline scoring of saved direct-search and decomposed-discovery corpora."""

from __future__ import annotations


def score_retrieval(
    papers: list[dict], relevant_by_subtopic: dict[str, list[str]], *, top_k: int = 10,
) -> dict:
    """Score unique paper IDs against independent, manually curated judgments.

    Input order is the ranking being evaluated (retrieval order by default;
    pass curator-sorted records to measure curator ranking). Unjudged papers
    count as nonrelevant, so precision is relative to the supplied gold set.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(relevant_by_subtopic, dict) or not relevant_by_subtopic:
        raise ValueError("relevant_by_subtopic must be a nonempty mapping")
    for name, ids in relevant_by_subtopic.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("curated subtopic names must be nonempty strings")
        if (
            not isinstance(ids, list) or not ids
            or any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids)
        ):
            raise ValueError("each curated subtopic must contain a nonempty list of paper IDs")
    if not isinstance(papers, list) or any(
        not isinstance(paper, dict)
        or not isinstance(paper.get("paperId"), str)
        or not paper["paperId"].strip()
        for paper in papers
    ):
        raise ValueError("papers must be a list of records with nonempty string paperId values")
    gold = set().union(*relevant_by_subtopic.values())
    ranked_ids = list(dict.fromkeys(p["paperId"] for p in papers))
    retrieved = set(ranked_ids)
    top = ranked_ids[:top_k]
    covered = [name for name, ids in relevant_by_subtopic.items() if retrieved.intersection(ids)]
    return {
        "unique_papers": len(ranked_ids),
        "gold_papers": len(gold),
        "subtopics": len(relevant_by_subtopic),
        "relevant_papers": len(retrieved & gold),
        "relevant_paper_recall": len(retrieved & gold) / len(gold),
        # Missing ranks count as misses rather than inflating a short result list.
        "precision_at_k": len(set(top) & gold) / top_k,
        "top_k": top_k,
        "subtopic_coverage": len(covered) / len(relevant_by_subtopic),
        "covered_subtopics": covered,
    }


def compare_retrieval(case: dict, *, top_k: int = 10) -> dict:
    """Compare saved results only when request-slot and corpus budgets match.

    A result slot is one requested result, including duplicates/empty slots.
    Equal selected corpus sizes alone do not control retrieval effort.
    """
    direct = case["direct"]
    decomposed = case["decomposed"]
    budgets = []
    for arm in (direct, decomposed):
        limits = arm["request_limits"]
        if not limits or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100
                             for n in limits):
            raise ValueError("request_limits must record actual positive API limits (at most 100)")
        if arm.get("errors"):
            raise ValueError("failed searches invalidate the fixed-budget comparison")
        budgets.append(sum(limits))
    if budgets[0] != budgets[1]:
        raise ValueError("direct and decomposed retrieval slot budgets must match")
    cap = case["max_total_papers"]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("max_total_papers must be a positive integer")
    for arm in (direct, decomposed):
        if len(arm["papers"]) > cap:
            raise ValueError("saved corpus exceeds max_total_papers")
        if len(arm["papers"]) > sum(arm["request_limits"]):
            raise ValueError("saved corpus exceeds requested result slots")
    return {
        "topic": case["topic"],
        "retrieval_slot_budget": budgets[0],
        "max_total_papers": cap,
        "direct_requests": len(direct["request_limits"]),
        "decomposed_requests": len(decomposed["request_limits"]),
        "direct": score_retrieval(direct["papers"], case["relevant_by_subtopic"], top_k=top_k),
        "decomposed": score_retrieval(decomposed["papers"], case["relevant_by_subtopic"], top_k=top_k),
    }
