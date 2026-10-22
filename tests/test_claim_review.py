"""Human review contracts using exact stored abstracts and private artifacts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.core.state import PipelineState
from src.evaluation.claim_review import (
    build_snapshot, load_review, new_review, record_judgment, review_counts, save_review, review_lock,
)


def state():
    return PipelineState(topic="Retrieval", papers_curated=[{
        "paperId": "p1", "title": "A real source", "abstract": "First sentence.\nExact source text.",
        "one_line_summary": "MODEL INTERPRETATION, NOT SOURCE TEXT", "assessment_status": "success",
    }], synthesis={"summary_paragraph": "An uncited summary.", "landscape": {
        "themes": [{"name": "Retrieval", "explanation": "Retrieval helps.",
                    "supporting_paper_ids": ["p1"], "confidence": 0.8}],
        "gaps": [], "future_work": [], "methodology_patterns": [],
        "shared_limitations": [], "disagreements": [],
    }})


def test_review_pairs_claim_with_exact_source_not_curator_interpretation():
    original = state()
    before = deepcopy(original.to_dict())
    snapshot = build_snapshot(original)
    claim = snapshot.claims[0]
    assert claim.text == "Retrieval helps."
    assert claim.label == "Retrieval"
    assert claim.evidence[0].abstract == "First sentence.\nExact source text."
    assert claim.evidence[0].paper_id == "p1"
    assert "MODEL INTERPRETATION" not in snapshot.model_dump_json()
    assert "summary_paragraph" in snapshot.uncited_sections
    assert original.to_dict() == before


def test_every_evidence_bearing_section_has_review_items():
    original = state()
    landscape = original.synthesis["landscape"]
    landscape["gaps"] = [{"name": "Gap", "explanation": "A gap.", "supporting_paper_ids": ["p1"], "confidence": 0.7}]
    landscape["future_work"] = [{"recommendation": "Broaden scope", "rationale": "Narrow evaluation.", "supporting_paper_ids": ["p1"], "confidence": 0.6}]
    landscape["methodology_patterns"] = [{"methodology": "empirical", "observation": "Mostly empirical.", "representative_paper_ids": ["p1"]}]
    landscape["shared_limitations"] = [{"limitation": "Small samples.", "supporting_paper_ids": ["p1"]}]
    landscape["disagreements"] = [{"question": "Does it help?", "positions": [
        {"position": "It helps.", "supporting_paper_ids": ["p1"]},
        {"position": "It can hurt.", "supporting_paper_ids": ["missing"]},
    ], "interpretation": "Settings differ."}]
    snapshot = build_snapshot(original)
    texts = {c.text for c in snapshot.claims}
    assert texts == {"Retrieval helps.", "A gap.", "Broaden scope", "Narrow evaluation.",
                     "Mostly empirical.", "Small samples.", "It helps.", "It can hurt.", "Settings differ."}
    assert len({c.claim_id for c in snapshot.claims}) == 9
    missing = next(c for c in snapshot.claims if c.text == "It can hurt.")
    assert missing.evidence[0].missing is True
    assert missing.evidence[0].abstract == ""


@pytest.mark.parametrize("status", ["supported", "unsupported", "uncertain"])
def test_judgments_round_trip_without_changing_pipeline(tmp_path, status):
    snapshot = build_snapshot(state())
    review = new_review(snapshot)
    updated = record_judgment(review, snapshot.claims[0].claim_id, status, note="Human rationale", reviewer="Raghav")
    path = tmp_path / "review.json"
    save_review(updated, path)
    loaded = load_review(path, snapshot)
    judgment = loaded.judgments[snapshot.claims[0].claim_id]
    assert judgment.status == status
    assert judgment.note == "Human rationale"
    assert judgment.reviewer == "Raghav"
    assert review.judgments == {}


@pytest.mark.parametrize("change", ["claim", "abstract", "citation", "topic"])
def test_stale_feedback_is_rejected_and_original_file_preserved(tmp_path, change):
    original = state()
    snapshot = build_snapshot(original)
    path = tmp_path / "review.json"
    save_review(record_judgment(new_review(snapshot), snapshot.claims[0].claim_id, "supported"), path)
    before = path.read_bytes()
    if change == "claim":
        original.synthesis["landscape"]["themes"][0]["explanation"] = "A different claim."
    elif change == "abstract":
        original.papers_curated[0]["abstract"] = "Different evidence."
    elif change == "citation":
        original.synthesis["landscape"]["themes"][0]["supporting_paper_ids"] = ["another"]
    else:
        original.topic = "Another topic"
    with pytest.raises(ValueError, match="changed"):
        load_review(path, build_snapshot(original))
    assert path.read_bytes() == before


def test_corrupt_and_unknown_claim_artifacts_are_rejected(tmp_path):
    snapshot = build_snapshot(state())
    path = tmp_path / "review.json"
    review = record_judgment(new_review(snapshot), snapshot.claims[0].claim_id, "supported")
    save_review(review, path)
    data = json.loads(path.read_text())
    data["snapshot"]["claims"][0]["text"] = "Tampered claim."
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_review(path, snapshot)
    save_review(review, path)
    data = json.loads(path.read_text())
    data["judgments"]["unknown"] = data["judgments"].pop(snapshot.claims[0].claim_id)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown claim"):
        load_review(path, snapshot)
    path.write_text("{broken")
    with pytest.raises(ValueError):
        load_review(path, snapshot)


def test_failed_atomic_replace_preserves_last_saved_judgment(tmp_path, monkeypatch):
    snapshot = build_snapshot(state())
    path = tmp_path / "review.json"
    saved = record_judgment(new_review(snapshot), snapshot.claims[0].claim_id, "supported")
    save_review(saved, path)
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr("src.evaluation.claim_review.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_review(record_judgment(saved, snapshot.claims[0].claim_id, "unsupported"), path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["review.json"]


def test_lock_excludes_second_writer_and_releases_after_exception(tmp_path):
    path = tmp_path / "review.json"
    with pytest.raises(RuntimeError):
        with review_lock(path):
            with pytest.raises(ValueError, match="already open"):
                with review_lock(path):
                    pytest.fail("second writer acquired the lock")
            raise RuntimeError("review interrupted")
    with review_lock(path):
        pass
    assert list(tmp_path.iterdir()) == []


def test_legacy_synthesis_has_no_invented_citations():
    original = state()
    original.synthesis.pop("landscape")
    with pytest.raises(ValueError, match="landscape"):
        build_snapshot(original)


@pytest.mark.parametrize("version", [None, True, 2, "1"])
def test_missing_or_unsupported_artifact_version_is_rejected(tmp_path, version):
    snapshot = build_snapshot(state())
    path = tmp_path / "review.json"
    save_review(new_review(snapshot), path)
    data = json.loads(path.read_text())
    if version is None:
        data.pop("version")
    else:
        data["version"] = version
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_review(path, snapshot)


def test_irrelevant_metadata_and_paper_order_do_not_invalidate_human_work(tmp_path):
    original = state()
    original.papers_curated.append({"paperId": "uncited", "title": "Other", "abstract": "Other text."})
    snapshot = build_snapshot(original)
    path = tmp_path / "review.json"
    save_review(record_judgment(new_review(snapshot), snapshot.claims[0].claim_id, "supported"), path)
    original.papers_curated.reverse()
    original.papers_curated[0]["abstract"] = "Uncited text changed."
    original.papers_curated[1]["citationCount"] = 123
    original.synthesis_work = {"internal": "cache metadata changed"}
    assert load_review(path, build_snapshot(original)).judgments[snapshot.claims[0].claim_id].status == "supported"


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_claims.py"


def test_cli_review_resume_and_eof_preserve_decisions_without_touching_checkpoint(tmp_path):
    checkpoint = tmp_path / "session.json"
    state().save(checkpoint)
    before = checkpoint.read_bytes()
    output = tmp_path / "reviews" / "human.json"
    command = [sys.executable, str(SCRIPT), str(checkpoint), "--output", str(output), "--reviewer", "Raghav"]
    result = subprocess.run(command, input="1\nThe source supports it.\n", text=True, capture_output=True,
                            cwd=tmp_path, timeout=20, check=True)
    assert "Retrieval helps." in result.stdout
    assert "Exact source text." in result.stdout
    assert "MODEL INTERPRETATION" not in result.stdout
    snapshot = build_snapshot(state())
    assert load_review(output, snapshot).judgments[snapshot.claims[0].claim_id].status == "supported"
    resumed = subprocess.run(command, input="", text=True, capture_output=True, cwd=tmp_path, timeout=20, check=True)
    assert "1/1 reviewed" in resumed.stdout
    assert checkpoint.read_bytes() == before
    assert not output.with_suffix(".json.lock").exists()


def test_cli_refuses_checkpoint_as_output(tmp_path):
    checkpoint = tmp_path / "session.json"
    state().save(checkpoint)
    before = checkpoint.read_bytes()
    result = subprocess.run([sys.executable, str(SCRIPT), str(checkpoint), "--output", str(checkpoint)],
                            input="", capture_output=True, text=True, cwd=tmp_path, timeout=20)
    assert result.returncode != 0
    assert checkpoint.read_bytes() == before


def test_interrupted_cli_reuses_exactly_completed_human_work(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    original = state()
    original.synthesis["landscape"]["themes"] += [
        {"name": "Second", "explanation": "Second claim.", "supporting_paper_ids": ["p1"], "confidence": 0.7},
        {"name": "Third", "explanation": "Third claim.", "supporting_paper_ids": ["p1"], "confidence": 0.6},
    ]
    checkpoint = tmp_path / "session.json"
    original.save(checkpoint)
    output = tmp_path / "judgments.json"
    command = [sys.executable, str(SCRIPT), str(checkpoint), "--output", str(output)]
    first = subprocess.run(command, input="1\nSource agrees.\n3\nNeeds more context.\n", capture_output=True,
                           text=True, cwd=tmp_path, timeout=20, check=True)
    assert "Stopped" in first.stdout
    assert "2/3 reviewed" in first.stdout
    snapshot = build_snapshot(original)
    partial = load_review(output, snapshot)
    assert review_counts(partial) == {"supported": 1, "unsupported": 0, "uncertain": 1, "unreviewed": 1}
    resumed = subprocess.run(command, input="2\nNot in abstract.\n", capture_output=True,
                             text=True, cwd=tmp_path, timeout=20, check=True)
    assert "Claim 1/3" not in resumed.stdout
    assert "Claim 2/3" not in resumed.stdout
    assert "Third claim." in resumed.stdout
    completed = load_review(output, snapshot)
    assert review_counts(completed) == {"supported": 1, "unsupported": 1, "uncertain": 1, "unreviewed": 0}
    assert all(completed.judgments[key] == value for key, value in partial.judgments.items())
    summary = subprocess.run(command + ["--summary"], input="", capture_output=True,
                             text=True, cwd=tmp_path, timeout=20, check=True)
    assert "3/3 reviewed" in summary.stdout
    assert "1 supported / 1 unsupported / 1 uncertain / 0 unreviewed" in summary.stdout


def test_revisit_can_change_a_judgment_and_source_controls_are_literal(tmp_path):
    original = state()
    original.papers_curated[0]["abstract"] = "Untrusted \x1b[2Jsource text."
    checkpoint = tmp_path / "session.json"
    original.save(checkpoint)
    output = tmp_path / "judgments.json"
    command = [sys.executable, str(SCRIPT), str(checkpoint), "--output", str(output)]
    first = subprocess.run(command, input="1\n\n", capture_output=True, text=True, timeout=20, check=True)
    assert "\x1b" not in first.stdout
    assert "\\u001b[2Jsource text." in first.stdout
    subprocess.run(command + ["--revisit"], input="3\nNeeds verification.\n", capture_output=True,
                   text=True, timeout=20, check=True)
    review = load_review(output, build_snapshot(original))
    assert review_counts(review)["uncertain"] == 1
    assert review_counts(review)["supported"] == 0


def test_missing_abstract_is_explicit_and_invalid_feedback_is_not_saved(tmp_path):
    original = state()
    original.papers_curated[0]["abstract"] = None
    snapshot = build_snapshot(original)
    assert snapshot.claims[0].evidence[0].abstract == ""
    review = new_review(snapshot)
    with pytest.raises(ValueError):
        record_judgment(review, snapshot.claims[0].claim_id, "probably supported")
    with pytest.raises(ValueError, match="unknown claim"):
        record_judgment(review, "unknown", "supported")
    assert review.judgments == {}


def test_cli_preserves_unicode_and_refuses_stale_or_malformed_saved_feedback(tmp_path):
    original = state()
    original.papers_curated[0]["abstract"] = "Evidence about β-catenin and 日本語."
    checkpoint = tmp_path / "session.json"
    original.save(checkpoint)
    output = tmp_path / "judgments.json"
    command = [sys.executable, str(SCRIPT), str(checkpoint), "--output", str(output)]
    result = subprocess.run(command, input="nonsense\n1\n\n", capture_output=True, text=True,
                            cwd=tmp_path, timeout=20, check=True)
    assert "Enter 1, 2, 3, s, or q." in result.stdout
    saved = load_review(output, build_snapshot(original))
    assert saved.snapshot.claims[0].evidence[0].abstract == "Evidence about β-catenin and 日本語."
    before = output.read_bytes()
    original.papers_curated[0]["abstract"] = "Changed evidence."
    original.save(checkpoint)
    stale = subprocess.run(command, input="1\n\n", capture_output=True, text=True, timeout=20)
    assert stale.returncode != 0
    assert "changed" in stale.stderr
    assert output.read_bytes() == before
    output.write_text("{broken", encoding="utf-8")
    malformed = subprocess.run(command, input="", capture_output=True, text=True, timeout=20)
    assert malformed.returncode != 0
    assert "invalid checkpoint or review artifact schema" in malformed.stderr
    assert output.read_text() == "{broken"
