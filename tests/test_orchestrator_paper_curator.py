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

    def fake_synthesis(state: PipelineState, provider: object, *, batch_size: int) -> PipelineState:
        assert batch_size == 20
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


def test_orchestrator_passes_configured_synthesis_batch_size(monkeypatch, tmp_path) -> None:
    def fake_topic(state: PipelineState, provider: object) -> PipelineState:
        state.keyword_clusters = [KeywordCluster(theme="Theme", keywords=["one", "two", "three"], description="Description")]
        return state

    def fake_discovery(state: PipelineState, provider: object, **kwargs: object) -> PipelineState:
        state.papers_raw = [{"paperId": "paper-1", "title": "Paper"}]
        return state

    def fake_curator(state: PipelineState, provider: object, *, batch_size: int) -> PipelineState:
        state.papers_curated = [{"paperId": "paper-1", "assessment_status": "success"}]
        return state

    received_batch_sizes: list[int] = []

    def fake_synthesis(state: PipelineState, provider: object, *, batch_size: int) -> PipelineState:
        received_batch_sizes.append(batch_size)
        state.synthesis = {"key_themes": ["Theme"]}
        return state

    monkeypatch.setattr(orchestrator.topic_decomposition, "run", fake_topic)
    monkeypatch.setattr(orchestrator.discovery, "run", fake_discovery)
    monkeypatch.setattr(orchestrator.paper_curator, "run", fake_curator)
    monkeypatch.setattr(orchestrator.synthesis, "run", fake_synthesis)
    monkeypatch.setattr(orchestrator, "write_to_google_sheets", lambda *args, **kwargs: "https://example.test/sheet")
    config = {
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "semantic_scholar": {},
        "paper_curator": {"batch_size": 1},
        "synthesis": {"batch_size": 7},
        "google_sheets": {},
    }

    orchestrator.run_pipeline("topic", FakeProvider(), config)

    assert received_batch_sizes == [7]


def test_orchestrator_rejects_removed_enrichment_config(tmp_path) -> None:
    config = {
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "enrichment": {"batch_size": 8},
    }
    with pytest.raises(ValueError, match="paper_curator"):
        orchestrator.run_pipeline("topic", FakeProvider(), config)


def test_sheets_writer_renders_evidence_grounded_landscape() -> None:
    class ValuesResource:
        def __init__(self) -> None:
            self.rows: list[list[str]] | None = None

        def clear(self, **kwargs: object) -> "ValuesResource":
            return self

        def update(self, *, body: dict[str, list[list[str]]], **kwargs: object) -> "ValuesResource":
            self.rows = body["values"]
            return self

        def execute(self) -> dict[str, object]:
            return {}

    class SpreadsheetsResource:
        def __init__(self, values: ValuesResource) -> None:
            self._values = values

        def values(self) -> ValuesResource:
            return self._values

    class FakeSheetsService:
        def __init__(self) -> None:
            self.values_resource = ValuesResource()

        def spreadsheets(self) -> SpreadsheetsResource:
            return SpreadsheetsResource(self.values_resource)

    synthesis = {
        "summary_paragraph": "Summary",
        "key_themes": ["Theme"],
        "research_gaps": ["Gap"],
        "recommended_future_work": ["Future work"],
        "suggested_reading_order": [{"paperId": "paper-1", "title": "Paper One", "reason": "Start here"}],
        "landscape": {
            "themes": [{"name": "Theme", "explanation": "Evidence", "supporting_paper_ids": ["paper-1"], "confidence": 0.9}],
            "gaps": [{"name": "Gap", "explanation": "Missing evidence", "supporting_paper_ids": ["paper-2"], "confidence": 0.7}],
            "future_work": [{"recommendation": "Test broader settings", "rationale": "Current coverage is narrow", "supporting_paper_ids": ["paper-1"], "confidence": 0.8}],
            "methodology_patterns": [{"methodology": "empirical", "observation": "Common", "representative_paper_ids": ["paper-1"]}],
            "disagreements": [{"question": "Question", "positions": [{"position": "For", "supporting_paper_ids": ["paper-1"]}, {"position": "Against", "supporting_paper_ids": ["paper-2"]}], "interpretation": "Mixed"}],
            "shared_limitations": [{"limitation": "Small samples", "supporting_paper_ids": ["paper-1", "paper-2"]}],
        },
    }
    service = FakeSheetsService()

    orchestrator._write_synthesis_tab(service, "sheet-id", synthesis)

    rows = service.values_resource.rows
    assert rows is not None
    headers = [row[0] for row in rows if row]
    assert "THEME EVIDENCE" in headers
    assert "GAP EVIDENCE" in headers
    assert "FUTURE WORK EVIDENCE" in headers
    assert "METHODOLOGY LANDSCAPE" in headers
    assert "DISAGREEMENTS" in headers
    assert "SHARED LIMITATIONS" in headers
    assert any("paper-1; paper-2" in row for row in rows)


@pytest.mark.parametrize("synthesis_config", [{"batch_size": 0}, {"batch_size": False}, []])
def test_orchestrator_validates_synthesis_config_before_work(
    synthesis_config: object, tmp_path
) -> None:
    config = {
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "semantic_scholar": {},
        "paper_curator": {"batch_size": 1},
        "synthesis": synthesis_config,
    }
    with pytest.raises(ValueError, match="synthesis|positive integer"):
        orchestrator.run_pipeline("topic", FakeProvider(), config)


def test_resume_rejects_truthy_malformed_synthesis(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    PipelineState(topic="topic", synthesis={"unexpected": True}).save(checkpoint)
    config = {
        "pipeline": {"checkpoint_path": str(checkpoint), "max_agent_iterations": 2},
        "semantic_scholar": {},
        "paper_curator": {"batch_size": 1},
        "synthesis": {"batch_size": 20},
    }
    with pytest.raises(ValueError, match="invalid legacy synthesis"):
        orchestrator.run_pipeline("topic", FakeProvider(), config, resume=True)


def test_request_estimate_for_free_tier_profile() -> None:
    config = {
        "llm": {"max_requests_per_run": 20, "transient_503_retries": 0},
        "semantic_scholar": {"max_total_papers": 40},
        "paper_curator": {"batch_size": 8},
        "synthesis": {"batch_size": 20},
    }

    estimate = orchestrator.estimate_llm_requests(config)

    assert estimate.clean == 9
    assert estimate.validation_ceiling == 18
    assert estimate.transport_ceiling == 18
    assert estimate.hard_cap == 20


def test_request_estimate_includes_one_503_retry() -> None:
    config = {
        "llm": {"max_requests_per_run": 20, "transient_503_retries": 1},
        "semantic_scholar": {"max_total_papers": 40},
        "paper_curator": {"batch_size": 8},
        "synthesis": {"batch_size": 20},
    }

    assert orchestrator.estimate_llm_requests(config).transport_ceiling == 36


def test_request_estimate_uses_remaining_checkpoint_work() -> None:
    state = PipelineState(
        topic="topic",
        keyword_clusters=[
            KeywordCluster(theme="Theme", keywords=["one", "two", "three"], description="Description")
        ],
        papers_raw=[{"paperId": str(index)} for index in range(9)],
    )
    config = {
        "llm": {"max_requests_per_run": 20},
        "semantic_scholar": {"max_total_papers": 40},
        "paper_curator": {"batch_size": 8},
        "synthesis": {"batch_size": 20},
    }

    estimate = orchestrator.estimate_llm_requests(config, state)

    assert estimate.clean == 3  # two curator batches plus one synthesis call


def test_pipeline_rejects_clean_plan_above_hard_cap(tmp_path) -> None:
    config = {
        "llm": {"max_requests_per_run": 8},
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "semantic_scholar": {"max_total_papers": 40},
        "paper_curator": {"batch_size": 8},
        "synthesis": {"batch_size": 20},
    }

    with pytest.raises(ValueError, match="requires 9 LLM requests"):
        orchestrator.run_pipeline("topic", FakeProvider(), config)
