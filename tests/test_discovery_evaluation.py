"""Metric expectations are hand-calculated from independent relevance labels."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.discovery import compare_retrieval, score_retrieval


def case():
    return {
        "topic": "Example", "max_total_papers": 4,
        "relevant_by_subtopic": {"alpha": ["a", "b"], "beta": ["c"]},
        "direct": {"request_limits": [12], "papers": [{"paperId": "a"}, {"paperId": "noise"}]},
        "decomposed": {"request_limits": [2] * 6,
                       "papers": [{"paperId": "c"}, {"paperId": "b"}, {"paperId": "a"}]},
    }


def test_known_recall_precision_and_coverage():
    result = compare_retrieval(case(), top_k=2)
    assert result["retrieval_slot_budget"] == 12
    assert result["direct"]["gold_papers"] == 3
    assert result["direct"]["subtopics"] == 2
    assert result["direct"]["relevant_paper_recall"] == pytest.approx(1 / 3)
    assert result["direct"]["precision_at_k"] == 0.5
    assert result["direct"]["subtopic_coverage"] == 0.5
    assert result["decomposed"]["relevant_paper_recall"] == 1
    assert result["decomposed"]["precision_at_k"] == 1
    assert result["decomposed"]["subtopic_coverage"] == 1


def test_duplicates_short_results_and_overlapping_judgments():
    result = score_retrieval([{"paperId": "a"}, {"paperId": "a"}],
                             {"alpha": ["a", "b"], "beta": ["a", "c"]}, top_k=4)
    assert result["unique_papers"] == 1
    assert result["relevant_paper_recall"] == pytest.approx(1 / 3)
    assert result["precision_at_k"] == 0.25
    assert result["subtopic_coverage"] == 1
    empty = score_retrieval([], {"alpha": ["a"]})
    assert empty["relevant_paper_recall"] == empty["precision_at_k"] == empty["subtopic_coverage"] == 0


@pytest.mark.parametrize("judgments", [
    {"subtopic": "paper-id"}, {"subtopic": []}, {"subtopic": [None]},
    {"subtopic": [""]}, {"subtopic": [" "]}, {"": ["paper-id"]}, {}, [],
])
def test_malformed_judgments_cannot_silently_produce_metrics(judgments):
    with pytest.raises(ValueError):
        score_retrieval([{"paperId": "paper-id"}], judgments)


@pytest.mark.parametrize("papers", [[{}], [{"paperId": None}], [{"paperId": ""}],
                                  [{"paperId": " "}], ["paper-id"], "paper-id"])
def test_invalid_retrieved_paper_ids_are_rejected(papers):
    with pytest.raises(ValueError, match="paperId"):
        score_retrieval(papers, {"subtopic": ["paper-id"]})


@pytest.mark.parametrize("change", ["budget", "errors", "cap", "limits"])
def test_invalid_comparisons_are_rejected(change):
    data = deepcopy(case())
    if change == "budget":
        data["direct"]["request_limits"] = [10]
    elif change == "errors":
        data["decomposed"]["errors"] = ["search failed"]
    elif change == "cap":
        data["max_total_papers"] = 1
    else:
        data["direct"]["request_limits"] = [0]
    with pytest.raises(ValueError):
        compare_retrieval(data)


def test_cli_scores_multiple_topics_from_private_files(tmp_path):
    cases = [case(), {**case(), "topic": "Second example"}]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_discovery.py"
    result = subprocess.run([sys.executable, str(script), str(path), "--top-k", "2"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15, check=True)
    output = json.loads(result.stdout)
    assert [item["topic"] for item in output] == ["Example", "Second example"]
    assert output[1]["direct"]["precision_at_k"] == 0.5
