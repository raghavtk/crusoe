"""Offline integration coverage for the curator stage in the orchestrator."""

from __future__ import annotations

import json

import pytest

from src.agents import orchestrator
from src.core.state import KeywordCluster, PipelineState


class FakeProvider:
    def call(self, system_prompt: str, messages: list[dict], tools: list[object]) -> dict:
        paper = {
            "paperId": "paper-1",
            "relevance_score": 5,
            "relevance_rationale": "The abstract directly investigates the requested research topic.",
            "confidence_score": 0.9,
            "methodology": "empirical",
            "contribution_type": "tool",
            "one_line_summary": "The paper introduces and evaluates a tool for the research problem.",
        }
        return {"content": json.dumps([paper])}


def test_orchestrator_runs_real_curator_with_fake_provider(monkeypatch, tmp_path) -> None:
    def fake_topic(state: PipelineState, provider: object) -> PipelineState:
        state.keyword_clusters = [KeywordCluster(theme="Theme", keywords=["one", "two", "three"], description="Description")]
        return state

    def fake_discovery(state: PipelineState, provider: object, **kwargs: object) -> PipelineState:
        state.papers_raw = [{"paperId": "paper-1", "title": "Paper", "abstract": "Strong evidence", "year": 2025, "citationCount": 5}]
        return state

    def fake_synthesis(state: PipelineState, provider: object) -> PipelineState:
        assert state.papers_curated[0]["assessment_status"] == "success"
        state.synthesis = {"key_themes": ["Theme"]}
        return state

    monkeypatch.setattr(orchestrator.topic_decomposition, "run", fake_topic)
    monkeypatch.setattr(orchestrator.discovery, "run", fake_discovery)
    monkeypatch.setattr(orchestrator.synthesis, "run", fake_synthesis)
    monkeypatch.setattr(orchestrator, "write_to_google_sheets", lambda *args, **kwargs: "https://example.test/sheet")
    config = {
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "semantic_scholar": {},
        "paper_curator": {"batch_size": 1},
        "google_sheets": {},
    }

    state = orchestrator.run_pipeline("topic", FakeProvider(), config)

    assert len(state.papers_curated) == 1
    assert state.sheet_url == "https://example.test/sheet"


def test_orchestrator_rejects_removed_enrichment_config(tmp_path) -> None:
    config = {
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "enrichment": {"batch_size": 8},
    }
    with pytest.raises(ValueError, match="paper_curator"):
        orchestrator.run_pipeline("topic", FakeProvider(), config)
