"""Offline checks for deterministic synthesis evaluation support."""

from __future__ import annotations

import json

from src.evaluation.synthesis import MeasuredProvider, fixed_map_reduce_corpus, save_evaluation_artifact


class Delegate:
    def call(self, system_prompt: str, messages: list[dict], tools: list) -> dict:
        return {"content": "ok"}


def test_map_reduce_corpus_is_fixed_unique_and_priority_ordered() -> None:
    first = fixed_map_reduce_corpus()
    second = fixed_map_reduce_corpus()
    assert first == second
    assert len(first) == 21
    assert len({paper["paperId"] for paper in first}) == 21
    assert all(paper["assessment_status"] == "success" for paper in first)
    assert [paper["reading_priority_score"] for paper in first] == sorted(
        (paper["reading_priority_score"] for paper in first), reverse=True
    )


def test_measured_provider_enforces_ceiling_and_records_latency() -> None:
    measured = MeasuredProvider(Delegate(), max_calls=1)  # type: ignore[arg-type]
    assert measured.call("system", [{"role": "user", "content": "prompt"}], []) == {"content": "ok"}
    assert measured.call_count == 1
    assert len(measured.call_latencies_seconds) == 1


def test_artifact_contains_metrics_hash_and_blank_rubric(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    papers = fixed_map_reduce_corpus()[:3]
    measured = MeasuredProvider(Delegate(), max_calls=2)  # type: ignore[arg-type]
    measured.call("system", [{"role": "user", "content": "prompt"}], [])
    synthesis = {
        "key_themes": ["Theme"],
        "research_gaps": ["Gap"],
        "suggested_reading_order": [{"paperId": paper["paperId"]} for paper in papers],
    }

    path = save_evaluation_artifact(
        provider="test", model="fixed", scenario="offline", papers=papers,
        synthesis=synthesis, measured=measured, expected_base_calls=1,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["input_sha256"]) == 64
    assert payload["metrics"]["call_count"] == 1
    assert payload["metrics"]["repair_count"] == 0
    assert payload["metrics"]["theme_count"] == 1
    assert payload["human_rubric"]["grounding"] is None
