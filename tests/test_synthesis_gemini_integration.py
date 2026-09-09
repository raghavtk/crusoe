"""Opt-in, fixed-corpus behavioral test against the real Gemini API."""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv

from src.agents import synthesis
from src.core.state import PipelineState
from src.evaluation.synthesis import (
    MeasuredProvider,
    fixed_map_reduce_corpus,
    save_evaluation_artifact,
)
from src.llm.providers import GeminiProvider


FIXED_CURATED_PAPERS = [
    {
        "paperId": "fixed-survey",
        "title": "A Survey of Retrieval-Augmented Generation",
        "abstract": (
            "This survey organizes retrieval-augmented generation systems by retrieval, "
            "augmentation, and generation design choices. It compares evaluation practices "
            "and identifies recurring weaknesses in evidence attribution and benchmark coverage."
        ),
        "year": 2024,
        "citationCount": 240,
        "authors": ["A. Researcher", "B. Scholar"],
        "fieldsOfStudy": ["Computer Science"],
        "relevance_score": 5,
        "relevance_rationale": "It provides the organizing taxonomy for the requested topic.",
        "confidence_score": 0.95,
        "methodology": "survey",
        "contribution_type": "survey",
        "one_line_summary": "The paper surveys retrieval-augmented generation architectures and evaluations.",
        "reading_priority_score": 96.0,
        "reading_priority": "high",
        "assessment_status": "success",
    },
    {
        "paperId": "fixed-evaluation",
        "title": "Evaluating Attribution in Retrieval-Augmented Language Models",
        "abstract": (
            "This empirical study evaluates whether generated claims are supported by retrieved "
            "documents. Results show that answer accuracy and citation correctness can diverge, "
            "and that common aggregate metrics hide claim-level attribution failures."
        ),
        "year": 2025,
        "citationCount": 85,
        "authors": ["C. Evaluator"],
        "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"],
        "relevance_score": 5,
        "relevance_rationale": "It directly tests evidence attribution quality in the target systems.",
        "confidence_score": 0.9,
        "methodology": "empirical",
        "contribution_type": "framework",
        "one_line_summary": "The paper demonstrates gaps between answer accuracy and citation correctness.",
        "reading_priority_score": 91.0,
        "reading_priority": "high",
        "assessment_status": "success",
    },
    {
        "paperId": "fixed-defense",
        "title": "Robust Retrieval Under Adversarial Knowledge Sources",
        "abstract": (
            "This paper studies retrieval when indexed documents contain misleading or adversarial "
            "content. A filtering and source-consistency defense improves robustness in simulation, "
            "although evaluation is limited to synthetic attacks and a small number of domains."
        ),
        "year": 2025,
        "citationCount": 30,
        "authors": ["D. Defender", "E. Analyst"],
        "fieldsOfStudy": ["Computer Science", "Security"],
        "relevance_score": 4,
        "relevance_rationale": "It covers an important robustness risk in retrieval-grounded systems.",
        "confidence_score": 0.85,
        "methodology": "simulation",
        "contribution_type": "defense",
        "one_line_summary": "The paper proposes a defense against adversarial retrieval sources.",
        "reading_priority_score": 82.0,
        "reading_priority": "high",
        "assessment_status": "success",
    },
]


@pytest.mark.integration
def test_fixed_corpus_synthesis_with_gemini() -> None:
    """Exercise one real synthesis, permitting only one validation repair."""
    if os.getenv("RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("set RUN_GEMINI_INTEGRATION=1 to authorize real Gemini calls")
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_EVAL_MODEL", "gemini-3.6-flash")
    provider = MeasuredProvider(GeminiProvider(model=model, temperature=None), max_calls=2)
    state = PipelineState(
        topic="Evidence quality and robustness in retrieval-augmented generation",
        papers_curated=[dict(paper) for paper in FIXED_CURATED_PAPERS],
    )

    synthesis.run(state, provider, batch_size=20)

    assert provider.call_count in (1, 2)
    assert len(state.synthesis["suggested_reading_order"]) == 3
    assert state.synthesis["key_themes"] == [
        item["name"] for item in state.synthesis["landscape"]["themes"]
    ]
    known_ids = {paper["paperId"] for paper in FIXED_CURATED_PAPERS}
    assert {
        entry["paperId"] for entry in state.synthesis["suggested_reading_order"]
    } == known_ids

    artifact = save_evaluation_artifact(
        provider="gemini",
        model=model,
        scenario="fixed-3-paper",
        papers=FIXED_CURATED_PAPERS,
        synthesis=state.synthesis,
        measured=provider,
        expected_base_calls=1,
    )

    print("\nValidated Gemini synthesis for the fixed corpus:")
    print(json.dumps(state.synthesis, indent=2, ensure_ascii=False))
    print(f"\nEvaluation artifact: {artifact.resolve()}")


@pytest.mark.integration
def test_fixed_map_reduce_corpus_with_gemini() -> None:
    """Exercise two 20-paper map partitions and the validated reducer."""
    if os.getenv("RUN_GEMINI_MAP_REDUCE_INTEGRATION") != "1":
        pytest.skip("set RUN_GEMINI_MAP_REDUCE_INTEGRATION=1 to authorize map-reduce calls")
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_EVAL_MODEL", "gemini-3.6-flash")
    papers = fixed_map_reduce_corpus()
    provider = MeasuredProvider(GeminiProvider(model=model, temperature=None), max_calls=6)
    state = PipelineState(
        topic="Design, evaluation, robustness, and deployment of retrieval-augmented generation",
        papers_curated=papers,
    )

    synthesis.run(state, provider, batch_size=20)

    assert 3 <= provider.call_count <= 6
    assert len(state.synthesis["suggested_reading_order"]) == 12
    known_ids = {paper["paperId"] for paper in papers}
    cited_ids = {
        paper_id
        for theme in state.synthesis["landscape"]["themes"]
        for paper_id in theme["supporting_paper_ids"]
    }
    assert cited_ids <= known_ids
    artifact = save_evaluation_artifact(
        provider="gemini",
        model=model,
        scenario="fixed-21-paper-map-reduce",
        papers=papers,
        synthesis=state.synthesis,
        measured=provider,
        expected_base_calls=3,
    )
    print("\nValidated Gemini map-reduce metrics:")
    print(json.dumps({
        "artifact": str(artifact.resolve()),
        "call_count": provider.call_count,
        "latencies_seconds": provider.call_latencies_seconds,
        "themes": state.synthesis["key_themes"],
        "gaps": state.synthesis["research_gaps"],
        "reading_order": state.synthesis["suggested_reading_order"],
    }, indent=2, ensure_ascii=False))
