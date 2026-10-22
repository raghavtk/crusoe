"""Discovery contracts at the search-service boundary; no network or LLM calls."""

from collections import Counter
from copy import deepcopy

import pytest

from src.agents import discovery
from src.core.state import KeywordCluster, PipelineState


def clusters(count=4):
    return [KeywordCluster(theme=f"Theme {i}", keywords=[f"{i}:{j}" for j in range(3)],
                           description=f"Subtopic {i}") for i in range(count)]


def paper(identifier):
    return {"paperId": identifier, "title": f"Paper {identifier}"}


def run(monkeypatch, responses, *, count=4, cap=40, limit=20):
    calls = []

    def search(*, query, limit):
        calls.append((query, limit))
        response = responses.get(query, [])
        if isinstance(response, Exception):
            raise response
        return response[:limit]

    monkeypatch.setattr(discovery, "search_papers", search)
    state = PipelineState(topic="Topic", keyword_clusters=clusters(count))
    result = discovery.run(state, None, results_per_query=limit, max_total_papers=cap)
    assert result is state
    return state, calls


def test_four_productive_clusters_share_default_cap_and_search_in_rounds(monkeypatch):
    responses = {f"{i}:{j}": [paper(f"{i}:{j}:{k}") for k in range(20)]
                 for i in range(4) for j in range(3)}
    state, calls = run(monkeypatch, responses)
    assert len(state.papers_raw) == 40
    assert Counter(p["_source_cluster"] for p in state.papers_raw) == {
        f"Theme {i}": 10 for i in range(4)
    }
    # Query order is an externally visible fairness/request-budget contract.
    assert calls == [(f"{i}:{j}", 20) for j in range(3) for i in range(4)]


def test_unused_quota_is_redistributed_after_all_keywords_get_a_chance(monkeypatch):
    responses = {"0:0": [paper(f"a{k}") for k in range(20)],
                 "1:2": [paper("late")], "2:0": RuntimeError("offline"),
                 "3:1": [paper(f"d{k}") for k in range(20)]}
    state, calls = run(monkeypatch, responses, cap=10)
    assert len(state.papers_raw) == 10
    assert "late" in {p["paperId"] for p in state.papers_raw}
    assert Counter(p["_source_cluster"] for p in state.papers_raw) == {
        "Theme 0": 5, "Theme 1": 1, "Theme 3": 4,
    }
    assert len(calls) == 12
    assert len(state.errors) == 1
    assert "2:0" in state.errors[0]


def test_duplicate_provenance_is_complete_unique_and_checkpointed(monkeypatch, tmp_path):
    shared = paper("shared")
    responses = {"0:0": [shared, shared, {"paperId": "no-title"}, {"title": "no-id"}],
                 "0:1": [shared], "1:0": [shared, paper("other")],
                 "1:2": [shared]}
    original = deepcopy(responses)
    state, _ = run(monkeypatch, responses, count=2, cap=2)
    assert [p["paperId"] for p in state.papers_raw] == ["shared", "other"]
    assert state.papers_raw[0]["_source_cluster"] == "Theme 0"
    assert state.papers_raw[0]["_discovery_sources"] == [
        {"cluster": "Theme 0", "query": "0:0"},
        {"cluster": "Theme 1", "query": "1:0"},
        {"cluster": "Theme 0", "query": "0:1"},
        {"cluster": "Theme 1", "query": "1:2"},
    ]
    assert responses == original
    path = tmp_path / "checkpoint.json"
    state.save(path)
    assert PipelineState.load(path).papers_raw == state.papers_raw


@pytest.mark.parametrize("cap", [1, 3, 5, 9])
def test_small_and_indivisible_caps_never_overflow(monkeypatch, cap):
    responses = {f"{i}:0": [paper(f"{i}-{k}") for k in range(20)] for i in range(4)}
    state, calls = run(monkeypatch, responses, cap=cap)
    assert len(state.papers_raw) == cap
    assert len({p["paperId"] for p in state.papers_raw}) == cap
    assert len(calls) == 12  # Even a cap smaller than cluster count searches every cluster.
    counts = [sum(p["_source_cluster"] == f"Theme {i}" for p in state.papers_raw)
              for i in range(4)]
    assert max(counts) - min(counts) <= 1


def test_empty_invalid_and_duplicate_results_do_not_consume_capacity(monkeypatch):
    state, _ = run(monkeypatch, {"0:0": [{}, {"paperId": "bad"}, paper("only")],
                               "1:0": [paper("only")]}, cap=40)
    assert [p["paperId"] for p in state.papers_raw] == ["only"]


def test_all_failed_searches_finish_with_empty_corpus_and_report_each_failure(monkeypatch):
    state, calls = run(monkeypatch, {f"{i}:{j}": RuntimeError("offline")
                                     for i in range(4) for j in range(3)})
    assert state.papers_raw == []
    assert len(state.errors) == len(calls) == 12


def test_unequal_keyword_counts_and_service_limit(monkeypatch):
    state = PipelineState(keyword_clusters=clusters(2))
    state.keyword_clusters[1].keywords.extend(["extra query", "final query"])
    calls = []

    def search(*, query, limit):
        calls.append((query, limit))
        return [paper(query)]

    monkeypatch.setattr(discovery, "search_papers", search)
    discovery.run(state, None, results_per_query=200, max_total_papers=10)
    assert calls == [(query, 100) for query in
                     ["0:0", "1:0", "0:1", "1:1", "0:2", "1:2", "extra query", "final query"]]
    assert len(state.papers_raw) == 8


@pytest.mark.parametrize("option,value", [("results_per_query", 0), ("results_per_query", -1),
                                        ("results_per_query", True), ("results_per_query", 1.5),
                                        ("max_total_papers", 0), ("max_total_papers", -1),
                                        ("max_total_papers", False), ("max_total_papers", 1.5)])
def test_invalid_limits_fail_before_search(monkeypatch, option, value):
    def unexpected(**kwargs):
        pytest.fail("Invalid configuration must not contact the search service")

    monkeypatch.setattr(discovery, "search_papers", unexpected)
    with pytest.raises(ValueError, match=option):
        discovery.run(PipelineState(keyword_clusters=clusters()), None, **{option: value})
