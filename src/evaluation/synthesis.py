"""Fixed corpora, call measurement, and ignored artifacts for synthesis evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.providers import LLMProvider


MAP_REDUCE_TOPICS = [
    ("rag-survey", "Survey of Retrieval-Augmented Generation Systems", "survey", "survey", "organizes retrieval, augmentation, generation, and evaluation choices across RAG systems"),
    ("dense-retrieval", "Dense Retrieval for Knowledge-Intensive Tasks", "empirical", "framework", "compares dense retrieval with lexical baselines across knowledge-intensive benchmarks"),
    ("hybrid-retrieval", "Hybrid Sparse and Dense Retrieval", "empirical", "framework", "finds that hybrid retrieval improves coverage but adds tuning and latency tradeoffs"),
    ("reranking", "Cross-Encoder Reranking for RAG", "empirical", "framework", "evaluates reranking gains and shows sensitivity to candidate-set quality"),
    ("chunking", "Document Chunking Strategies for Retrieval", "empirical", "framework", "shows that chunk size changes recall, context coherence, and downstream answer quality"),
    ("query-rewrite", "Query Rewriting for Conversational Retrieval", "empirical", "framework", "tests query rewriting for ambiguous multi-turn questions and reports error propagation"),
    ("multi-hop", "Multi-Hop Retrieval and Evidence Composition", "empirical", "framework", "studies iterative retrieval for questions requiring evidence from multiple documents"),
    ("citation", "Claim-Level Citation Evaluation", "empirical", "framework", "demonstrates that answer accuracy and citation correctness can diverge at claim level"),
    ("faithfulness", "Measuring Faithfulness in Grounded Generation", "empirical", "framework", "compares automated faithfulness metrics and finds inconsistent agreement with human judgments"),
    ("human-eval", "Human Evaluation Protocols for RAG", "survey", "framework", "reviews evaluator instructions and identifies low reproducibility across human studies"),
    ("benchmark", "A Benchmark for End-to-End RAG Evaluation", "empirical", "dataset", "introduces an end-to-end benchmark spanning retrieval relevance and generated-answer support"),
    ("domain-shift", "RAG Under Domain Shift", "empirical", "framework", "finds substantial degradation when retrievers and generators move to unfamiliar domains"),
    ("freshness", "Temporal Freshness in Retrieval-Augmented Models", "empirical", "framework", "studies stale indexes and shows that freshness policies affect factual reliability"),
    ("poisoning", "Knowledge-Base Poisoning Attacks on RAG", "empirical", "attack", "demonstrates that a small number of malicious documents can steer generated answers"),
    ("prompt-injection", "Indirect Prompt Injection Through Retrieved Documents", "case study", "attack", "documents instruction-following attacks embedded inside retrieved content"),
    ("filtering", "Source Filtering for Robust Retrieval", "simulation", "defense", "tests filtering defenses against synthetic malicious sources across limited domains"),
    ("provenance", "Provenance-Aware Retrieval and Generation", "empirical", "framework", "uses source provenance to improve evidence selection and user-visible attribution"),
    ("uncertainty", "Uncertainty Estimation for Retrieval-Grounded Answers", "empirical", "framework", "evaluates confidence calibration when retrieved evidence is missing or contradictory"),
    ("contradiction", "Reasoning Over Contradictory Retrieved Evidence", "empirical", "framework", "compares strategies for detecting and presenting conflicts among retrieved sources"),
    ("latency", "Latency and Cost Tradeoffs in Production RAG", "case study", "framework", "measures retrieval depth, reranking, context size, latency, and serving cost"),
    ("privacy", "Privacy Risks in Retrieval-Augmented Generation", "formal analysis", "attack", "analyzes leakage risks when private indexed documents are exposed through generated answers"),
]


def fixed_map_reduce_corpus() -> list[dict[str, Any]]:
    """Return the same 21 curated records on every evaluation run."""
    papers: list[dict[str, Any]] = []
    for index, (slug, title, methodology, contribution, finding) in enumerate(MAP_REDUCE_TOPICS, 1):
        papers.append({
            "paperId": f"eval-{index:02d}-{slug}",
            "title": title,
            "abstract": (
                f"This study {finding}. The evaluation reports both useful results and bounded "
                "limitations, providing evidence for a controlled synthesis evaluation."
            ),
            "year": 2020 + index % 6,
            "citationCount": 260 - index * 7,
            "authors": [f"Evaluation Author {index}", f"Collaborator {index}"],
            "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"],
            "relevance_score": 5 if index <= 18 else 4,
            "relevance_rationale": "The abstract directly addresses RAG design, evaluation, robustness, or deployment.",
            "confidence_score": 0.9,
            "methodology": methodology,
            "contribution_type": contribution,
            "one_line_summary": f"The paper {finding}.",
            "reading_priority_score": round(98.0 - index * 1.7, 1),
            "reading_priority": "high" if index <= 16 else "medium",
            "assessment_status": "success",
        })
    return papers


class MeasuredProvider(LLMProvider):
    """Measure a delegate without changing its inputs or outputs."""

    def __init__(self, delegate: LLMProvider, max_calls: int) -> None:
        self.delegate = delegate
        self.max_calls = max_calls
        self.call_count = 0
        self.call_latencies_seconds: list[float] = []

    def call(self, system_prompt: str, messages: list[dict], tools: list) -> dict:
        self.call_count += 1
        if self.call_count > self.max_calls:
            raise AssertionError(f"live evaluation exceeded its {self.max_calls}-call ceiling")
        started = time.perf_counter()
        try:
            return self.delegate.call(system_prompt, messages, tools)
        finally:
            self.call_latencies_seconds.append(round(time.perf_counter() - started, 3))


def save_evaluation_artifact(
    *,
    provider: str,
    model: str,
    scenario: str,
    papers: list[dict[str, Any]],
    synthesis: dict[str, Any],
    measured: MeasuredProvider,
    expected_base_calls: int,
) -> Path:
    """Write a timestamped, ignored JSON record for human evaluation."""
    now = datetime.now(timezone.utc)
    canonical_input = json.dumps(papers, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = {
        "recorded_at_utc": now.isoformat(),
        "provider": provider,
        "model": model,
        "scenario": scenario,
        "input_sha256": hashlib.sha256(canonical_input.encode("utf-8")).hexdigest(),
        "paper_count": len(papers),
        "metrics": {
            "call_count": measured.call_count,
            "repair_count": max(0, measured.call_count - expected_base_calls),
            "call_latencies_seconds": measured.call_latencies_seconds,
            "total_latency_seconds": round(sum(measured.call_latencies_seconds), 3),
            "theme_count": len(synthesis.get("key_themes", [])),
            "gap_count": len(synthesis.get("research_gaps", [])),
            "reading_order_count": len(synthesis.get("suggested_reading_order", [])),
        },
        "human_rubric": {
            "grounding": None,
            "theme_coherence": None,
            "gap_quality": None,
            "method_and_disagreement_fidelity": None,
            "reading_order_usefulness": None,
            "review_notes": "",
        },
        "synthesis": synthesis,
    }
    directory = Path("data/evaluations")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{provider}_{scenario}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
