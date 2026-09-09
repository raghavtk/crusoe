"""Opt-in provider-parity test against the real Cerebras API."""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv

from src.agents import synthesis
from src.core.state import PipelineState
from src.evaluation.synthesis import MeasuredProvider, fixed_map_reduce_corpus, save_evaluation_artifact
from src.llm.providers import CerebrasProvider


@pytest.mark.integration
def test_fixed_corpus_synthesis_with_cerebras() -> None:
    """Run the same deterministic three-record slice through Cerebras."""
    if os.getenv("RUN_CEREBRAS_INTEGRATION") != "1":
        pytest.skip("set RUN_CEREBRAS_INTEGRATION=1 to authorize real Cerebras calls")
    load_dotenv()
    if not os.getenv("CEREBRAS_API_KEY"):
        pytest.skip("CEREBRAS_API_KEY is not configured")

    model = os.getenv("CEREBRAS_EVAL_MODEL", "gpt-oss-120b")
    papers = fixed_map_reduce_corpus()[:3]
    provider = MeasuredProvider(CerebrasProvider(model=model, temperature=0.5), max_calls=2)
    state = PipelineState(
        topic="Retrieval design choices in retrieval-augmented generation",
        papers_curated=papers,
    )

    synthesis.run(state, provider, batch_size=20)

    assert provider.call_count in (1, 2)
    assert len(state.synthesis["suggested_reading_order"]) == 3
    assert {
        entry["paperId"] for entry in state.synthesis["suggested_reading_order"]
    } == {paper["paperId"] for paper in papers}
    artifact = save_evaluation_artifact(
        provider="cerebras",
        model=model,
        scenario="fixed-3-paper",
        papers=papers,
        synthesis=state.synthesis,
        measured=provider,
        expected_base_calls=1,
    )
    print("\nValidated Cerebras provider-parity metrics:")
    print(json.dumps({
        "artifact": str(artifact.resolve()),
        "call_count": provider.call_count,
        "latencies_seconds": provider.call_latencies_seconds,
        "themes": state.synthesis["key_themes"],
        "gaps": state.synthesis["research_gaps"],
        "reading_order": state.synthesis["suggested_reading_order"],
    }, indent=2, ensure_ascii=False))
