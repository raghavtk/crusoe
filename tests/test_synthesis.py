"""Offline contract, batching, and grounding tests for synthesis."""

from __future__ import annotations

import json

import pytest

from src.agents import synthesis
from src.agents.synthesis import SynthesisValidationError
from src.core.state import PipelineState


class FakeProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def call(self, system_prompt: str, messages: list[dict], tools: list[object]) -> dict:
        self.calls.append({"system_prompt": system_prompt, "messages": messages, "tools": tools})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"content": response}


def _paper(index: int, *, status: str = "success", abstract: str = "Evidence.") -> dict:
    return {
        "paperId": f"p{index}", "title": f"Paper {index}", "abstract": abstract,
        "year": 2020 + index % 6, "citationCount": index, "authors": [f"Author {n}" for n in range(7)],
        "fieldsOfStudy": ["Computer Science"], "relevance_score": 5,
        "relevance_rationale": "Directly relevant evidence.", "confidence_score": 0.9,
        "methodology": "empirical", "contribution_type": "framework",
        "one_line_summary": "The paper presents relevant evidence.",
        "reading_priority_score": 100 - index, "reading_priority": "high",
        "assessment_status": status,
    }


def _landscape(ids: list[str]) -> dict:
    first = ids[0]
    return {
        "themes": [{"name": "Main theme", "explanation": "Evidence supports this theme.", "supporting_paper_ids": [first], "confidence": 0.9}],
        "gaps": [{"name": "Main gap", "explanation": "Existing work leaves this unresolved.", "supporting_paper_ids": [first], "confidence": 0.7}],
        "future_work": [{"recommendation": "Evaluate broader settings", "rationale": "Current evaluation is narrow.", "supporting_paper_ids": [first], "confidence": 0.8}],
        "methodology_patterns": [{"methodology": "empirical", "observation": "Empirical evaluation is common.", "representative_paper_ids": [first]}],
        "disagreements": [],
        "shared_limitations": [{"limitation": "Evaluation scope is limited.", "supporting_paper_ids": [first]}],
    }


def _final(ids: list[str]) -> str:
    return json.dumps({
        "summary_paragraph": "The literature establishes a coherent foundation. Important gaps remain. Broader evaluation is needed.",
        "landscape": _landscape(ids),
        "suggested_reading_order": [
            {"paperId": paper_id, "title": f"Paper {paper_id[1:]}", "reason": "Read for its evidence."}
            for paper_id in ids[:12]
        ],
    })


def _batch(ids: list[str]) -> str:
    landscape = _landscape(ids)
    return json.dumps({
        **landscape,
        "notable_papers": [{"paperId": ids[0], "title": f"Paper {ids[0][1:]}", "reason": "Notable evidence."}],
    })


def test_empty_and_all_failed_inputs_do_not_call_provider() -> None:
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="papers_curated is empty"):
        synthesis.run(PipelineState(topic="topic"), provider)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no successfully assessed"):
        synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(1, status="failed")]), provider)  # type: ignore[arg-type]
    assert not provider.calls


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_fails_before_provider(batch_size: object) -> None:
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="positive integer"):
        synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(1)]), provider, batch_size=batch_size)  # type: ignore[arg-type]
    assert not provider.calls


def test_duplicate_or_blank_eligible_ids_fail_before_provider() -> None:
    provider = FakeProvider([])
    duplicate = [_paper(1), _paper(1)]
    with pytest.raises(ValueError, match="unique"):
        synthesis.run(PipelineState(topic="topic", papers_curated=duplicate), provider)  # type: ignore[arg-type]
    blank = _paper(1)
    blank["paperId"] = " "
    with pytest.raises(ValueError, match="nonblank"):
        synthesis.run(PipelineState(topic="topic", papers_curated=[blank]), provider)  # type: ignore[arg-type]


def test_compaction_normalizes_caps_and_marks_abstracts() -> None:
    abstract = "word " * 170 + "Complete sentence. " + "tail " * 100
    paper = _paper(1, abstract=abstract)
    paper["authors"] = [{"name": f" Person  {i} "} for i in range(8)]
    compact = synthesis._papers_to_prompt_format([paper])[0]
    assert set(compact) == {
        "paperId", "title", "abstract_excerpt", "abstract_excerpt_truncated", "year",
        "citationCount", "authors", "fieldsOfStudy", "relevance_score", "relevance_rationale",
        "confidence_score", "methodology", "contribution_type", "one_line_summary",
        "reading_priority_score", "reading_priority",
    }
    assert compact["authors"] == [f"Person {i}" for i in range(5)]
    assert compact["abstract_excerpt_truncated"] is True
    assert len(compact["abstract_excerpt"]) <= 1200
    assert compact["abstract_excerpt"].endswith(".")


def test_abstract_hard_cut_and_missing_value() -> None:
    excerpt, truncated = synthesis._abstract_excerpt("x" * 1300)
    assert (len(excerpt), truncated) == (1200, True)
    assert synthesis._abstract_excerpt(None) == ("", False)


@pytest.mark.parametrize(
    "count,batch_size,expected_sizes",
    [(21, 20, [11, 10]), (40, 20, [20, 20]), (41, 20, [14, 14, 13]), (80, 20, [20, 20, 20, 20])],
)
def test_batches_are_deterministically_balanced(
    count: int, batch_size: int, expected_sizes: list[int]
) -> None:
    batches = synthesis._make_batches([{"paperId": str(i)} for i in range(count)], batch_size)
    assert [len(batch) for batch in batches] == expected_sizes
    assert [paper["paperId"] for batch in batches for paper in batch] == [str(i) for i in range(count)]


def test_single_pass_derives_legacy_fields_and_excludes_failed_papers() -> None:
    provider = FakeProvider([_final(["p1"])])
    state = PipelineState(topic="topic", papers_curated=[_paper(1), _paper(2, status="failed")])
    synthesis.run(state, provider)  # type: ignore[arg-type]
    assert state.synthesis["key_themes"] == ["Main theme"]
    assert state.synthesis["research_gaps"] == ["Main gap"]
    assert state.synthesis["recommended_future_work"] == ["Evaluate broader settings"]
    prompt = provider.calls[0]["messages"][0]["content"]
    assert '"paperId":"p1"' in prompt
    assert '"paperId":"p2"' not in prompt


def test_unsupported_categories_may_be_empty_without_invention() -> None:
    payload = json.loads(_final(["p1"]))
    for category in (
        "gaps", "future_work", "methodology_patterns", "disagreements", "shared_limitations"
    ):
        payload["landscape"][category] = []
    state = PipelineState(topic="topic", papers_curated=[_paper(1)])

    synthesis.run(state, FakeProvider([json.dumps(payload)]))  # type: ignore[arg-type]

    assert state.synthesis["research_gaps"] == []
    assert state.synthesis["recommended_future_work"] == []
    assert state.synthesis["landscape"]["methodology_patterns"] == []
    assert "instead of inventing an unsupported" in synthesis.FINAL_SCHEMA


@pytest.mark.parametrize("count,expected_calls", [(1, 1), (20, 1), (21, 3), (40, 3), (41, 4), (80, 5)])
def test_adaptive_batch_call_counts(count: int, expected_calls: int) -> None:
    ids = [f"p{i}" for i in range(1, count + 1)]
    responses = [_final(ids)] if count <= 20 else [
        *[_batch(ids[i:i + 20]) for i in range(0, count, 20)], _final(ids)
    ]
    provider = FakeProvider(responses)
    synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(i) for i in range(1, count + 1)]), provider, batch_size=20)  # type: ignore[arg-type]
    assert len(provider.calls) == expected_calls


def test_reducer_receives_validated_objects_and_preserves_batch_order() -> None:
    ids = [f"p{i}" for i in range(1, 22)]
    provider = FakeProvider([_batch(ids[:20]), _batch(ids[20:]), _final(ids)])
    synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(i) for i in range(1, 22)]), provider)  # type: ignore[arg-type]
    reducer_prompt = provider.calls[-1]["messages"][0]["content"]
    assert "VALIDATED BATCH ANALYSES" in reducer_prompt
    assert reducer_prompt.index('"paperId":"p1"') < reducer_prompt.index('"paperId":"p21"')
    assert "SOURCE EVIDENCE PACKETS" in reducer_prompt
    assert '"abstract_excerpt":"Evidence."' in reducer_prompt


def test_map_batches_resume_from_checkpoint_without_repaying(tmp_path) -> None:
    papers = [_paper(i) for i in range(1, 22)]
    checkpoint = tmp_path / "checkpoint.json"
    first = PipelineState(topic="topic", papers_curated=papers)
    first_provider = FakeProvider([_batch([f"p{i}" for i in range(1, 12)]), RuntimeError("stop")])

    with pytest.raises(RuntimeError, match="stop"):
        synthesis.run(
            first,
            first_provider,  # type: ignore[arg-type]
            checkpoint_callback=lambda current: current.save(checkpoint),
            provider_identity="gemini:test",
        )

    resumed = PipelineState.load(checkpoint)
    second_provider = FakeProvider([
        _batch([f"p{i}" for i in range(12, 22)]),
        _final([f"p{i}" for i in range(1, 22)]),
    ])
    synthesis.run(
        resumed,
        second_provider,  # type: ignore[arg-type]
        checkpoint_callback=lambda current: current.save(checkpoint),
        provider_identity="gemini:test",
    )

    assert len(first_provider.calls) == 2
    assert len(second_provider.calls) == 2  # missing map plus reducer; map one was reused
    assert resumed.synthesis_work["status"] == "complete"
    assert all(batch["status"] == "validated" for batch in resumed.synthesis_work["batches"])
    assert '"paperId":"p1"' not in second_provider.calls[0]["messages"][0]["content"]


def test_cached_maps_leave_only_the_reducer_call(tmp_path) -> None:
    papers = [_paper(i) for i in range(1, 22)]
    checkpoint = tmp_path / "checkpoint.json"
    state = PipelineState(topic="topic", papers_curated=papers)
    provider = FakeProvider([
        _batch([f"p{i}" for i in range(1, 12)]),
        _batch([f"p{i}" for i in range(12, 22)]),
        RuntimeError("reducer stopped"),
    ])
    with pytest.raises(RuntimeError, match="reducer stopped"):
        synthesis.run(
            state,
            provider,  # type: ignore[arg-type]
            checkpoint_callback=lambda current: current.save(checkpoint),
            provider_identity="gemini:test",
        )

    resumed = PipelineState.load(checkpoint)
    assert synthesis.remaining_clean_calls(
        resumed, batch_size=20, provider_identity="gemini:test"
    ) == 1
    reducer_only = FakeProvider([_final([f"p{i}" for i in range(1, 22)])])
    synthesis.run(resumed, reducer_only, provider_identity="gemini:test")  # type: ignore[arg-type]
    assert len(reducer_only.calls) == 1
    assert "VALIDATED BATCH ANALYSES" in reducer_only.calls[0]["messages"][0]["content"]


def test_stale_work_is_recomputed_and_corrupt_work_fails_before_call() -> None:
    papers = [_paper(i) for i in range(1, 22)]
    stale = PipelineState(topic="topic", papers_curated=papers)
    initial = FakeProvider([
        _batch([f"p{i}" for i in range(1, 12)]), RuntimeError("stop")
    ])
    with pytest.raises(RuntimeError):
        synthesis.run(stale, initial, provider_identity="gemini:first")  # type: ignore[arg-type]
    refreshed = FakeProvider([
        _batch([f"p{i}" for i in range(1, 12)]),
        _batch([f"p{i}" for i in range(12, 22)]),
        _final([f"p{i}" for i in range(1, 22)]),
    ])
    synthesis.run(stale, refreshed, provider_identity="gemini:changed")  # type: ignore[arg-type]
    assert len(refreshed.calls) == 3

    corrupt = PipelineState(topic="topic", papers_curated=papers, synthesis_work={"bad": True})
    unused = FakeProvider([])
    with pytest.raises(SynthesisValidationError, match="corrupt synthesis_work"):
        synthesis.run(corrupt, unused, provider_identity="gemini:test")  # type: ignore[arg-type]
    assert not unused.calls


def test_old_work_version_is_stale_but_current_checksum_tampering_fails() -> None:
    papers = [_paper(i) for i in range(1, 22)]
    state = PipelineState(topic="topic", papers_curated=papers)
    interrupted = FakeProvider([
        _batch([f"p{i}" for i in range(1, 12)]), RuntimeError("stop")
    ])
    with pytest.raises(RuntimeError):
        synthesis.run(state, interrupted, provider_identity="gemini:test")  # type: ignore[arg-type]

    old = PipelineState(
        topic="topic",
        papers_curated=papers,
        synthesis_work={**state.synthesis_work, "version": 0},
    )
    recomputed = FakeProvider([
        _batch([f"p{i}" for i in range(1, 12)]),
        _batch([f"p{i}" for i in range(12, 22)]),
        _final([f"p{i}" for i in range(1, 22)]),
    ])
    synthesis.run(old, recomputed, provider_identity="gemini:test")  # type: ignore[arg-type]
    assert len(recomputed.calls) == 3

    tampered_work = json.loads(json.dumps(state.synthesis_work))
    tampered_work["batches"][0]["result"]["themes"][0]["name"] = "Altered valid theme"
    tampered = PipelineState(topic="topic", papers_curated=papers, synthesis_work=tampered_work)
    unused = FakeProvider([])
    with pytest.raises(SynthesisValidationError, match="checksum"):
        synthesis.run(tampered, unused, provider_identity="gemini:test")  # type: ignore[arg-type]
    assert not unused.calls


def test_validated_final_is_resumed_without_another_provider_call(tmp_path) -> None:
    papers = [_paper(i) for i in range(1, 22)]
    checkpoint = tmp_path / "checkpoint.json"
    state = PipelineState(topic="topic", papers_curated=papers)
    provider = FakeProvider([
        _batch([f"p{i}" for i in range(1, 12)]),
        _batch([f"p{i}" for i in range(12, 22)]),
        _final([f"p{i}" for i in range(1, 22)]),
    ])

    def fail_after_saving_validated_final(current: PipelineState) -> None:
        current.save(checkpoint)
        if current.synthesis_work.get("status") == "final_validated":
            raise OSError("simulated publication interruption")

    with pytest.raises(OSError, match="publication interruption"):
        synthesis.run(
            state,
            provider,  # type: ignore[arg-type]
            checkpoint_callback=fail_after_saving_validated_final,
            provider_identity="gemini:test",
        )

    resumed = PipelineState.load(checkpoint)
    assert not resumed.synthesis
    assert synthesis.remaining_clean_calls(
        resumed, batch_size=20, provider_identity="gemini:test"
    ) == 0
    no_calls = FakeProvider([])
    synthesis.run(resumed, no_calls, provider_identity="gemini:test")  # type: ignore[arg-type]
    assert not no_calls.calls
    assert resumed.synthesis_work["status"] == "complete"
    assert resumed.synthesis["key_themes"] == ["Main theme"]


def test_reducer_budget_is_checked_before_any_map_call(monkeypatch) -> None:
    papers = [_paper(i, abstract="x" * 1_200) for i in range(1, 22)]
    monkeypatch.setattr(synthesis, "MAX_REDUCER_PROMPT_CHARS", 1)
    provider = FakeProvider([])
    with pytest.raises(SynthesisValidationError, match="context budget"):
        synthesis.run(
            PipelineState(topic="topic", papers_curated=papers),
            provider,  # type: ignore[arg-type]
            provider_identity="gemini:test",
        )
    assert not provider.calls


def test_oversized_topic_fails_before_any_provider_call() -> None:
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="topic must not exceed"):
        synthesis.run(
            PipelineState(
                topic="x" * (synthesis.MAX_TOPIC_CHARS + 1),
                papers_curated=[_paper(1)],
            ),
            provider,  # type: ignore[arg-type]
        )
    assert not provider.calls


@pytest.mark.parametrize("bad", ["not json", "{}", "prefix {}", json.dumps({"summary_paragraph": "x", "landscape": {}})])
def test_invalid_response_is_repaired_once(bad: str) -> None:
    provider = FakeProvider([bad, _final(["p1"])])
    state = PipelineState(topic="topic", papers_curated=[_paper(1)])
    synthesis.run(state, provider)  # type: ignore[arg-type]
    assert len(provider.calls) == 2
    assert "Validation errors:" in provider.calls[1]["messages"][0]["content"]


def test_double_invalid_is_fatal_and_preserves_existing_state() -> None:
    provider = FakeProvider(["bad", "still bad"])
    state = PipelineState(topic="topic", papers_curated=[_paper(1)], synthesis={"old": True})
    with pytest.raises(SynthesisValidationError, match="failed after repair"):
        synthesis.run(state, provider)  # type: ignore[arg-type]
    assert state.synthesis == {"old": True}


def test_provider_exception_preserves_existing_state() -> None:
    state = PipelineState(topic="topic", papers_curated=[_paper(1)], synthesis={"old": True})
    with pytest.raises(RuntimeError, match="provider failed"):
        synthesis.run(state, FakeProvider([RuntimeError("provider failed")]))  # type: ignore[arg-type]
    assert state.synthesis == {"old": True}


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "title", "extra", "confidence"])
def test_reference_and_contract_violations_trigger_repair(mutation: str) -> None:
    payload = json.loads(_final(["p1", "p2"]))
    if mutation == "unknown":
        payload["landscape"]["themes"][0]["supporting_paper_ids"] = ["unknown"]
    elif mutation == "duplicate":
        payload["suggested_reading_order"][1] = payload["suggested_reading_order"][0]
    elif mutation == "title":
        payload["suggested_reading_order"][0]["title"] = "Wrong"
    elif mutation == "extra":
        payload["unexpected"] = True
    else:
        payload["landscape"]["themes"][0]["confidence"] = 2.0
    provider = FakeProvider([json.dumps(payload), _final(["p1", "p2"])])
    synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(1), _paper(2)]), provider)  # type: ignore[arg-type]
    assert len(provider.calls) == 2


def test_map_rejects_cross_batch_reference_before_reducer() -> None:
    ids = [f"p{i}" for i in range(1, 22)]
    bad = json.loads(_batch(ids[:20]))
    bad["themes"][0]["supporting_paper_ids"] = ["p21"]
    provider = FakeProvider([json.dumps(bad), json.dumps(bad)])
    with pytest.raises(SynthesisValidationError, match="map batch 1 failed"):
        synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(i) for i in range(1, 22)]), provider)  # type: ignore[arg-type]
    assert len(provider.calls) == 2


def test_whole_json_fence_is_allowed_but_surrounding_prose_is_not() -> None:
    titles = {"p1": "Paper 1"}
    assert isinstance(synthesis._parse_and_validate(f"```json\n{_final(['p1'])}\n```", synthesis.FinalDraft, titles, 1), synthesis.FinalDraft)
    with pytest.raises(SynthesisValidationError):
        synthesis._parse_and_validate(f"Here: {_final(['p1'])}", synthesis.FinalDraft, titles, 1)


def test_reducer_cannot_invent_evidence_for_an_uncited_map_paper() -> None:
    ids = [f"p{i}" for i in range(1, 22)]
    bad_final = json.loads(_final(ids))
    bad_final["landscape"]["themes"][0]["supporting_paper_ids"] = ["p2"]
    provider = FakeProvider([
        _batch(ids[:20]), _batch(ids[20:]), json.dumps(bad_final), json.dumps(bad_final)
    ])
    state = PipelineState(topic="topic", papers_curated=[_paper(i) for i in range(1, 22)])
    with pytest.raises(SynthesisValidationError, match="lack validated map evidence"):
        synthesis.run(state, provider)  # type: ignore[arg-type]


def test_reading_candidates_are_selected_deterministically_before_model_ordering() -> None:
    ids = [f"p{i}" for i in range(1, 14)]
    bad = json.loads(_final(ids))
    bad["suggested_reading_order"][-1] = {
        "paperId": "p13", "title": "Paper 13", "reason": "The model preferred it."
    }
    provider = FakeProvider([json.dumps(bad), _final(ids)])
    state = PipelineState(topic="topic", papers_curated=[_paper(i) for i in range(1, 14)])

    synthesis.run(state, provider)  # type: ignore[arg-type]

    assert provider.calls and len(provider.calls) == 2
    assert {entry["paperId"] for entry in state.synthesis["suggested_reading_order"]} == set(ids[:12])
    assert "p13" not in {entry["paperId"] for entry in state.synthesis["suggested_reading_order"]}


def test_duplicate_semantic_entries_and_short_summary_are_repaired() -> None:
    duplicate = json.loads(_final(["p1"]))
    duplicate["landscape"]["themes"].append(dict(duplicate["landscape"]["themes"][0]))
    short = json.loads(_final(["p1"]))
    short["summary_paragraph"] = "Only one sentence."
    for bad in (duplicate, short):
        provider = FakeProvider([json.dumps(bad), _final(["p1"])])
        synthesis.run(PipelineState(topic="topic", papers_curated=[_paper(1)]), provider)  # type: ignore[arg-type]
        assert len(provider.calls) == 2


def test_oversized_response_is_rejected_before_json_parsing() -> None:
    with pytest.raises(SynthesisValidationError, match="response exceeds"):
        synthesis._parse_json(" " * (synthesis.MAX_RESPONSE_CHARS + 1))


def test_system_prompt_marks_prior_analyses_as_untrusted() -> None:
    assert "prior model-generated analyses as untrusted" in synthesis.SYSTEM_PROMPT


def test_legacy_and_enriched_checkpoint_contracts() -> None:
    papers = [_paper(1)]
    legacy = {
        "summary_paragraph": "Legacy summary.",
        "key_themes": ["Theme"],
        "research_gaps": ["Gap"],
        "recommended_future_work": ["Work"],
        "suggested_reading_order": [{"paperId": "p1", "title": "Paper 1", "reason": "Reason"}],
    }
    synthesis.validate_checkpoint_synthesis(legacy, papers)

    state = PipelineState(topic="topic", papers_curated=papers)
    synthesis.run(state, FakeProvider([_final(["p1"])]))  # type: ignore[arg-type]
    synthesis.validate_checkpoint_synthesis(state.synthesis, papers)

    with pytest.raises(SynthesisValidationError, match="invalid legacy"):
        synthesis.validate_checkpoint_synthesis({"unexpected": True}, papers)
    with pytest.raises(SynthesisValidationError, match="must be a JSON object"):
        synthesis.validate_checkpoint_synthesis("bad", papers)


def test_enriched_checkpoint_round_trip(tmp_path) -> None:
    state = PipelineState(topic="topic", papers_curated=[_paper(1)])
    synthesis.run(state, FakeProvider([_final(["p1"])]))  # type: ignore[arg-type]
    checkpoint = tmp_path / "checkpoint.json"
    state.save(checkpoint)
    loaded = PipelineState.load(checkpoint)
    synthesis.validate_checkpoint_synthesis(loaded.synthesis, loaded.papers_curated)
    assert loaded.synthesis == state.synthesis
    assert loaded.synthesis_work == state.synthesis_work


def test_legacy_checkpoint_without_synthesis_work_loads(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.json"
    checkpoint.write_text(json.dumps({"topic": "legacy"}), encoding="utf-8")
    assert PipelineState.load(checkpoint).synthesis_work == {}
