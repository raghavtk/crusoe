"""Offline contract and ranking tests for the Paper Curator agent."""

from __future__ import annotations

import json

import pytest

from src.agents import paper_curator
from src.agents.paper_curator import CuratorValidationError
from src.core.state import PipelineState


class FakeProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def call(self, system_prompt: str, messages: list[dict], tools: list[object]) -> dict:
        self.calls.append({"system_prompt": system_prompt, "messages": messages, "tools": tools})
        return {"content": self.responses.pop(0)}


def _papers() -> list[dict]:
    return [
        {"paperId": "p1", "title": "Central paper", "abstract": "Detailed evidence.", "year": 2025, "citationCount": 100},
        {"paperId": "p2", "title": "Background paper", "abstract": "Some context.", "year": 2020, "citationCount": 10},
    ]


def _assessment(paper_id: str, *, relevance: int = 4, confidence: float = 0.8) -> dict:
    return {
        "paperId": paper_id,
        "relevance_score": relevance,
        "relevance_rationale": "The abstract directly addresses the research topic.",
        "confidence_score": confidence,
        "methodology": "empirical",
        "contribution_type": "framework",
        "one_line_summary": "The paper presents and evaluates a relevant research framework.",
    }


def test_make_batches_and_invalid_batch_sizes() -> None:
    assert [len(batch) for batch in paper_curator._make_batches([{}, {}, {}], 2)] == [2, 1]
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            paper_curator._make_batches([], invalid)  # type: ignore[arg-type]


def test_run_rejects_empty_input_without_calling_provider() -> None:
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="papers_raw is empty"):
        paper_curator.run(PipelineState(topic="topic"), provider)  # type: ignore[arg-type]
    assert provider.calls == []


def test_valid_response_is_curated_ranked_and_normalised() -> None:
    response = [_assessment("p1", relevance=5, confidence=1.0), _assessment("p2", relevance=2)]
    response[0]["one_line_summary"] = "  A   strong contribution to the field. "
    state = PipelineState(topic="topic", papers_raw=_papers())

    paper_curator.run(state, FakeProvider([json.dumps(response)]))  # type: ignore[arg-type]

    assert [paper["paperId"] for paper in state.papers_curated] == ["p1", "p2"]
    assert state.papers_curated[0]["one_line_summary"] == "A strong contribution to the field."
    assert state.papers_curated[0]["reading_priority_score"] == 100.0
    assert state.papers_curated[0]["reading_priority"] == "high"
    assert all(paper["assessment_status"] == "success" for paper in state.papers_curated)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        json.dumps({"paperId": "p1"}),
        json.dumps([_assessment("p1"), _assessment("p1")]),
        json.dumps([_assessment("p1"), _assessment("unknown")]),
        json.dumps([_assessment("p1")]),
        json.dumps([{**_assessment("p1"), "relevance_score": 6}, _assessment("p2")]),
        json.dumps([{**_assessment("p1"), "methodology": "laboratory"}, _assessment("p2")]),
        json.dumps([{**_assessment("p1"), "one_line_summary": " "}, _assessment("p2")]),
        json.dumps([{**_assessment("p1"), "one_line_summary": "First sentence. Second sentence."}, _assessment("p2")]),
    ],
)
def test_parser_rejects_malformed_or_incomplete_batches(response: str) -> None:
    with pytest.raises(CuratorValidationError):
        paper_curator._parse_assessments(response, _papers())


def test_invalid_first_response_is_repaired_once() -> None:
    valid = json.dumps([_assessment("p1"), _assessment("p2")])
    provider = FakeProvider(["bad json", valid])
    state = PipelineState(topic="topic", papers_raw=_papers())

    paper_curator.run(state, provider)  # type: ignore[arg-type]

    assert len(provider.calls) == 2
    assert "Validation errors:" in provider.calls[1]["messages"][0]["content"]
    assert not state.errors
    assert all(p["assessment_status"] == "success" for p in state.papers_curated)


def test_two_invalid_responses_produce_explicit_failures() -> None:
    state = PipelineState(topic="topic", papers_raw=_papers())
    paper_curator.run(state, FakeProvider(["bad", "still bad"]))  # type: ignore[arg-type]

    assert len(state.errors) == 1
    assert all(p["assessment_status"] == "failed" for p in state.papers_curated)
    assert all(p["reading_priority_score"] == 0.0 for p in state.papers_curated)
    assert all(p["reading_priority"] == "low" for p in state.papers_curated)


def test_priority_ties_preserve_discovery_order_and_missing_metadata_is_neutral() -> None:
    papers = [
        {"paperId": "first", "title": "First", "abstract": "Evidence."},
        {"paperId": "second", "title": "Second", "abstract": "Evidence."},
    ]
    response = json.dumps([_assessment("first"), _assessment("second")])
    state = PipelineState(topic="topic", papers_raw=papers)

    paper_curator.run(state, FakeProvider([response]))  # type: ignore[arg-type]

    assert [p["paperId"] for p in state.papers_curated] == ["first", "second"]
    assert state.papers_curated[0]["reading_priority_score"] == 69.5
    assert state.papers_curated[0]["reading_priority"] == "medium"


def test_priority_score_boundaries() -> None:
    papers = [
        {**_papers()[0], "assessment_status": "success", "relevance_score": 5, "confidence_score": 1.0},
        {**_papers()[1], "assessment_status": "success", "relevance_score": 1, "confidence_score": 0.0},
    ]
    paper_curator._apply_priority_scores(papers)
    assert papers[0]["reading_priority_score"] == 100.0
    assert papers[1]["reading_priority_score"] == 0.0
