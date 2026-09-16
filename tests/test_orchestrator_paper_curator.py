"""Offline integration coverage for the curator stage in the orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

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

    def fake_synthesis(
        state: PipelineState, provider: object, *, batch_size: int, **kwargs: object
    ) -> PipelineState:
        assert batch_size == 20
        assert state.papers_curated[0]["assessment_status"] == "success"
        state.synthesis = {"key_themes": ["Theme"]}
        return state

    monkeypatch.setattr(orchestrator.topic_decomposition, "run", fake_topic)
    monkeypatch.setattr(orchestrator.discovery, "run", fake_discovery)
    monkeypatch.setattr(orchestrator.synthesis, "run", fake_synthesis)
    def fake_export(state: PipelineState, *args: object, **kwargs: object) -> None:
        state.sheet_url = "https://example.test/sheet"
        state.output_results["google_sheets"] = {
            "status": "success", "location": state.sheet_url,
        }
    monkeypatch.setattr(orchestrator, "_export_spreadsheet", fake_export)
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

    def fake_synthesis(
        state: PipelineState, provider: object, *, batch_size: int, **kwargs: object
    ) -> PipelineState:
        received_batch_sizes.append(batch_size)
        state.synthesis = {"key_themes": ["Theme"]}
        return state

    monkeypatch.setattr(orchestrator.topic_decomposition, "run", fake_topic)
    monkeypatch.setattr(orchestrator.discovery, "run", fake_discovery)
    monkeypatch.setattr(orchestrator.paper_curator, "run", fake_curator)
    monkeypatch.setattr(orchestrator.synthesis, "run", fake_synthesis)
    monkeypatch.setattr(orchestrator, "_export_spreadsheet", lambda *args, **kwargs: None)
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
    workbook = orchestrator.project_state(PipelineState(topic="topic", synthesis=synthesis))

    assert [sheet.name for sheet in workbook.sheets] == [
        "Summary", "Papers", "Themes", "Gaps", "Future Work", "Methods",
        "Disagreements", "Reading Order",
    ]
    assert workbook.sheets[2].rows[0][2].value == "paper-1"
    assert workbook.sheets[3].rows[0][3].value == "paper-2"
    assert workbook.sheets[6].rows[0][2].value == "paper-1; paper-2"


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


def test_request_estimate_credits_validated_synthesis_maps() -> None:
    from tests.test_synthesis import FakeProvider as SynthesisProvider
    from tests.test_synthesis import _batch, _paper

    papers = [_paper(i) for i in range(1, 22)]
    state = PipelineState(
        topic="topic",
        keyword_clusters=[
            KeywordCluster(theme="Theme", keywords=["one", "two", "three"], description="Description")
        ],
        papers_curated=papers,
    )
    provider = SynthesisProvider([
        _batch([f"p{i}" for i in range(1, 12)]),
        _batch([f"p{i}" for i in range(12, 22)]),
        RuntimeError("stop before reducer result"),
    ])
    with pytest.raises(RuntimeError):
        orchestrator.synthesis.run(
            state,
            provider,  # type: ignore[arg-type]
            provider_identity="gemini:gemini-3.6-flash:temperature=None",
        )
    config = {
        "llm": {"provider": "gemini", "gemini": {"model": "gemini-3.6-flash"}},
        "semantic_scholar": {"max_total_papers": 40},
        "paper_curator": {"batch_size": 8},
        "synthesis": {"batch_size": 20},
    }

    assert orchestrator.estimate_llm_requests(config, state).clean == 1


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


def test_pipeline_rejects_discovery_limit_above_synthesis_capacity_before_calls(tmp_path) -> None:
    provider = FakeProvider()
    config = {
        "pipeline": {"checkpoint_path": str(tmp_path / "checkpoint.json"), "max_agent_iterations": 2},
        "semantic_scholar": {"max_total_papers": 81},
        "paper_curator": {"batch_size": 8},
        "synthesis": {"batch_size": 20},
    }

    with pytest.raises(ValueError, match="synthesis limit of 80"):
        orchestrator.run_pipeline("topic", provider, config)


def test_xlsx_export_records_result_and_skips_matching_resume(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    state = PipelineState(
        topic="stable topic",
        papers_curated=[{"paperId": "p1", "title": "Paper"}],
        synthesis={"summary_paragraph": "Summary"},
    )
    output = {"backend": "xlsx", "xlsx": {"directory": "outputs"}}

    orchestrator._export_spreadsheet(
        state,
        output,
        checkpoint_path=str(checkpoint),
        config_dir=tmp_path,
        resume=False,
    )

    result = state.output_results["xlsx"]
    assert result["status"] == "success"
    assert result["location"] == result["destination_id"]
    assert Path(result["location"]).exists()
    assert PipelineState.load(checkpoint).output_results == state.output_results

    monkeypatch.setattr(
        orchestrator.XlsxWriter,
        "write",
        lambda *args, **kwargs: pytest.fail("matching export should be skipped"),
    )
    orchestrator._export_spreadsheet(
        state,
        output,
        checkpoint_path=str(checkpoint),
        config_dir=tmp_path,
        resume=True,
    )


def test_google_destination_is_checkpointed_before_failed_write(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    state = PipelineState(
        topic="topic",
        papers_curated=[{"paperId": "p1", "title": "Paper"}],
        synthesis={"summary_paragraph": "Summary"},
    )

    class FailingWriter:
        seen_destinations: list[str | None] = []

        @classmethod
        def from_config(cls, settings: dict, config_dir: Path) -> "FailingWriter":
            return cls()

        def ensure_destination(self, workbook: object, destination_id: str | None = None) -> str:
            self.seen_destinations.append(destination_id)
            return destination_id or "created-sheet-id"

        def write(
            self, workbook: object, destination_id: str, payload_sha256: str
        ) -> object:
            saved = PipelineState.load(checkpoint)
            assert saved.output_results["google_sheets"]["destination_id"] == destination_id
            raise RuntimeError("remote write failed")

    monkeypatch.setattr(orchestrator, "GoogleSheetsWriter", FailingWriter)
    orchestrator._export_spreadsheet(
        state,
        {"backend": "google_sheets", "google_sheets": {}},
        checkpoint_path=str(checkpoint),
        config_dir=tmp_path,
        resume=False,
    )

    result = state.output_results["google_sheets"]
    assert result["status"] == "failed"
    assert result["destination_id"] == "created-sheet-id"
    assert result["error"] == "RuntimeError"
    assert state.errors[-1].startswith("[Spreadsheet export:google_sheets]")

    resumed = PipelineState.load(checkpoint)
    orchestrator._export_spreadsheet(
        resumed,
        {"backend": "google_sheets", "google_sheets": {}},
        checkpoint_path=str(checkpoint),
        config_dir=tmp_path,
        resume=True,
    )
    assert FailingWriter.seen_destinations == [None, "created-sheet-id"]
    assert len([
        error for error in resumed.errors
        if error.startswith("[Spreadsheet export:google_sheets]")
    ]) == 1


def test_resume_with_all_agents_checkpointed_accepts_no_provider(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    PipelineState(
        topic="topic",
        keyword_clusters=[
            KeywordCluster(
                theme="Theme", keywords=["one", "two", "three"], description="Description"
            )
        ],
        papers_raw=[{"paperId": "p1"}],
        papers_curated=[{"paperId": "p1", "assessment_status": "success"}],
        synthesis={"legacy": True},
    ).save(checkpoint)
    monkeypatch.setattr(orchestrator.synthesis, "validate_checkpoint_synthesis", lambda *args: None)
    monkeypatch.setattr(orchestrator, "_export_spreadsheet", lambda *args, **kwargs: None)
    config = {
        "pipeline": {"checkpoint_path": str(checkpoint), "max_agent_iterations": 2},
        "semantic_scholar": {},
        "paper_curator": {"batch_size": 1},
        "synthesis": {"batch_size": 20},
        "output": {"backend": "xlsx", "xlsx": {"directory": "outputs"}},
    }

    result = orchestrator.run_pipeline("", None, config, resume=True)

    assert result.topic == "topic"
