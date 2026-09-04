"""
Orchestrator
============

Runs agents 1–4 in sequence, checkpointing state after each one.
On restart (--resume), it skips already-completed stages.
At the end, exports a normalized workbook to XLSX or Google Sheets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.agents import (
    discovery,
    paper_curator,
    synthesis,
    topic_decomposition,
)
from src.core.state import PipelineState
from src.core.errors import safe_exception_summary
from src.llm.providers import LLMProvider, apply_request_budget
from src.observability.langfuse_tracing import flush_traces, trace_agent, trace_span
from src.output import (
    GoogleSheetsWriter,
    OutputResult,
    SpreadsheetWriter,
    XlsxWriter,
    payload_fingerprint,
    project_state,
    validate_output_config,
    xlsx_destination,
)


@dataclass(frozen=True)
class LLMRequestEstimate:
    """Deterministic request plan for the remaining pipeline work."""

    clean: int
    validation_ceiling: int
    transport_ceiling: int
    hard_cap: int | None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _synthesis_provider_identity(llm_config: dict) -> str:
    """Build a credential-free semantic identity for synthesis artifacts."""
    provider_name = str(llm_config.get("provider", "gemini")).strip().lower()
    defaults = {"gemini": "gemini-3.6-flash", "cerebras": "gpt-oss-120b"}
    provider_config = llm_config.get(provider_name, {})
    if not isinstance(provider_config, dict):
        raise ValueError(f"llm.{provider_name} configuration must be a mapping")
    model = provider_config.get("model", defaults.get(provider_name, "unspecified"))
    temperature_defaults: dict[str, float | None] = {"gemini": None, "cerebras": 0.5}
    temperature = provider_config.get("temperature", temperature_defaults.get(provider_name))
    return f"{provider_name}:{model}:temperature={temperature!r}"


def estimate_llm_requests(
    config: dict,
    state: PipelineState | None = None,
) -> LLMRequestEstimate:
    """Estimate clean and failure-path physical requests for remaining stages.

    Discovery is deterministic and makes no LLM calls. Topic decomposition,
    every curator batch, and every synthesis unit permit one schema repair.
    Transport retries multiply those semantic attempts, but the hard cap always
    remains authoritative at runtime.
    """
    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict):
        raise ValueError("llm configuration must be a mapping")
    configured_cap = llm_config.get("max_requests_per_run")
    hard_cap = (
        _positive_int(configured_cap, "llm.max_requests_per_run")
        if configured_cap is not None
        else None
    )
    retries = llm_config.get("transient_503_retries", 0)
    if isinstance(retries, bool) or retries not in (0, 1):
        raise ValueError("llm.transient_503_retries must be 0 or 1")

    ss_config = config.get("semantic_scholar", {})
    if not isinstance(ss_config, dict):
        raise ValueError("semantic_scholar configuration must be a mapping")
    maximum_papers = _positive_int(
        ss_config.get("max_total_papers", 80), "semantic_scholar.max_total_papers"
    )
    if maximum_papers > synthesis.MAX_ELIGIBLE_PAPERS:
        raise ValueError(
            "semantic_scholar.max_total_papers exceeds the synthesis limit of "
            f"{synthesis.MAX_ELIGIBLE_PAPERS}"
        )
    curator_config = config.get("paper_curator", {})
    if not isinstance(curator_config, dict):
        raise ValueError("paper_curator configuration must be a mapping")
    curator_batch = _positive_int(
        curator_config.get("batch_size", 8), "paper_curator.batch_size"
    )
    synthesis_config = config.get("synthesis", {})
    if not isinstance(synthesis_config, dict):
        raise ValueError("synthesis configuration must be a mapping")
    synthesis_batch = synthesis_config.get("batch_size", 20)
    synthesis.validate_batch_size(synthesis_batch)
    synthesis_identity = _synthesis_provider_identity(llm_config)

    current = state or PipelineState()
    clean = 0
    if not current.has_clusters:
        clean += 1

    if current.has_raw_papers:
        paper_count = len(current.papers_raw)
    elif current.has_curated_papers:
        paper_count = len(current.papers_curated)
    else:
        paper_count = maximum_papers

    if not current.has_curated_papers:
        clean += (paper_count + curator_batch - 1) // curator_batch

    if not current.has_synthesis:
        if current.has_curated_papers:
            eligible_count = sum(
                paper.get("assessment_status") == "success"
                for paper in current.papers_curated
            )
        else:
            eligible_count = paper_count
        if eligible_count:
            if current.has_curated_papers:
                synthesis_units = synthesis.remaining_clean_calls(
                    current,
                    batch_size=synthesis_batch,
                    provider_identity=synthesis_identity,
                )
            else:
                synthesis_units = 1
                if eligible_count > synthesis_batch:
                    synthesis_units = (eligible_count + synthesis_batch - 1) // synthesis_batch + 1
            clean += synthesis_units

    validation_ceiling = clean * 2
    return LLMRequestEstimate(
        clean=clean,
        validation_ceiling=validation_ceiling,
        transport_ceiling=validation_ceiling * (retries + 1),
        hard_cap=hard_cap,
    )


def run_pipeline(
    topic: str,
    provider: LLMProvider | None,
    config: dict,
    resume: bool = False,
    config_path: str = "config.yaml",
) -> PipelineState:
    """
    Execute the full Crusoe pipeline end-to-end.

    Parameters
    ----------
    topic : str
        Research topic entered by the user.
    provider : LLMProvider
        Configured LLM provider.
    config : dict
        Full config.yaml contents.
    resume : bool
        If True, load state from checkpoint and skip completed stages.

    Returns
    -------
    PipelineState
        Final state after all agents and spreadsheet export.
    """
    checkpoint_path: str = config["pipeline"]["checkpoint_path"]
    max_iterations: int = config["pipeline"]["max_agent_iterations"]
    ss_config: dict = config.get("semantic_scholar", {})
    results_per_query: int = ss_config.get("results_per_query", 20)
    max_total_papers: int = ss_config.get("max_total_papers", 80)
    if "enrichment" in config:
        raise ValueError(
            "The 'enrichment' configuration was removed; rename it to 'paper_curator'."
        )
    curator_batch_size: int = config.get("paper_curator", {}).get("batch_size", 8)
    synthesis_config = config.get("synthesis", {})
    if not isinstance(synthesis_config, dict):
        raise ValueError("The 'synthesis' configuration must be a mapping.")
    synthesis_batch_size: int = synthesis_config.get("batch_size", 20)
    synthesis.validate_batch_size(synthesis_batch_size)
    synthesis_identity = _synthesis_provider_identity(config.get("llm", {}))
    output_config = validate_output_config(config)

    # ── Load or initialise state ─────────────────────────────────────────────
    if resume and Path(checkpoint_path).exists():
        logger.info(f"[Orchestrator] Resuming from checkpoint: {checkpoint_path}")
        state = PipelineState.load(checkpoint_path)
        if state.synthesis:
            synthesis.validate_checkpoint_synthesis(state.synthesis, state.papers_curated)
        if state.topic != topic and topic:
            logger.warning(
                f"[Orchestrator] Topic mismatch: checkpoint has {state.topic!r}, "
                f"flag has {topic!r}. Using checkpoint topic."
            )
    else:
        state = PipelineState(topic=topic)

    request_estimate = estimate_llm_requests(config, state)
    if (
        request_estimate.hard_cap is not None
        and request_estimate.clean > request_estimate.hard_cap
    ):
        raise ValueError(
            "The remaining clean pipeline plan requires "
            f"{request_estimate.clean} LLM requests, exceeding the configured hard cap of "
            f"{request_estimate.hard_cap}. Reduce the paper limit or increase the cap."
        )
    logger.info(
        "[Orchestrator] LLM request plan: clean={}, validation ceiling={}, "
        "transport ceiling={}, hard cap={}.",
        request_estimate.clean,
        request_estimate.validation_ceiling,
        request_estimate.transport_ceiling,
        request_estimate.hard_cap,
    )
    if request_estimate.clean and provider is None:
        raise ValueError("an LLM provider is required while agent stages remain")
    if provider is not None:
        provider = apply_request_budget(provider, config.get("llm", {}))

    # ── Stage 1: Topic Decomposition ─────────────────────────────────────────
    if not state.has_clusters:
        logger.info("[Orchestrator] Running Topic Decomposition agent...")
        with trace_agent("topic-decomposition", input_data={"topic": state.topic}) as span:
            state = topic_decomposition.run(state, provider)  # type: ignore[arg-type]
            if span is not None:
                span.update(output={"cluster_count": len(state.keyword_clusters)})
        state.save(checkpoint_path)
        flush_traces()
        logger.info(f"[Orchestrator] ✓ Topic Decomposition — {len(state.keyword_clusters)} clusters")
    else:
        logger.info(f"[Orchestrator] ↩ Skipping Topic Decomposition (checkpoint: {len(state.keyword_clusters)} clusters)")

    # ── Stage 2: Discovery ───────────────────────────────────────────────────
    if not state.has_raw_papers:
        logger.info("[Orchestrator] Running Discovery agent...")
        with trace_agent(
            "discovery",
            input_data={"cluster_count": len(state.keyword_clusters)},
        ) as span:
            state = discovery.run(
                state,
                provider,  # type: ignore[arg-type]
                results_per_query=results_per_query,
                max_total_papers=max_total_papers,
                max_iterations=max_iterations,
            )
            if span is not None:
                span.update(output={"paper_count": len(state.papers_raw)})
        state.save(checkpoint_path)
        flush_traces()
        logger.info(f"[Orchestrator] ✓ Discovery — {len(state.papers_raw)} papers found")
    else:
        logger.info(f"[Orchestrator] ↩ Skipping Discovery (checkpoint: {len(state.papers_raw)} papers)")

    # ── Stage 3: Paper Curator ───────────────────────────────────────────────
    if not state.has_curated_papers:
        logger.info("[Orchestrator] Running Paper Curator agent...")
        with trace_agent(
            "paper-curator",
            input_data={"paper_count": len(state.papers_raw), "batch_size": curator_batch_size},
        ) as span:
            state = paper_curator.run(state, provider, batch_size=curator_batch_size)  # type: ignore[arg-type]
            if span is not None:
                span.update(output={"curated_count": len(state.papers_curated)})
        state.save(checkpoint_path)
        flush_traces()
        n_batches = (len(state.papers_curated) + curator_batch_size - 1) // curator_batch_size
        logger.info(
            f"[Orchestrator] ✓ Paper Curator — {len(state.papers_curated)} papers curated ({n_batches} batches)"
        )
    else:
        logger.info(f"[Orchestrator] ↩ Skipping Paper Curator (checkpoint: {len(state.papers_curated)} papers)")

    # ── Stage 4: Synthesis ───────────────────────────────────────────────────
    if not state.has_synthesis:
        logger.info("[Orchestrator] Running Synthesis agent...")
        with trace_agent(
            "synthesis",
            input_data={
                "curated_count": len(state.papers_curated),
                "batch_size": synthesis_batch_size,
            },
        ) as span:
            state = synthesis.run(
                state,
                provider,  # type: ignore[arg-type]
                batch_size=synthesis_batch_size,
                checkpoint_callback=lambda current: current.save(checkpoint_path),
                provider_identity=synthesis_identity,
            )
            if span is not None:
                span.update(output={"theme_count": len(state.synthesis.get("key_themes", []))})
        state.save(checkpoint_path)
        flush_traces()
        logger.info("[Orchestrator] ✓ Synthesis — complete")
    else:
        logger.info("[Orchestrator] ↩ Skipping Synthesis (checkpoint: already done)")

    # ── Stage 5: Spreadsheet export ─────────────────────────────────────────
    _export_spreadsheet(
        state,
        output_config,
        checkpoint_path=checkpoint_path,
        config_dir=Path(config_path).resolve().parent,
        resume=resume,
    )

    if state.errors:
        logger.warning(f"[Orchestrator] Pipeline completed with {len(state.errors)} non-fatal error(s):")
        for err in state.errors:
            logger.warning(f"  - {err}")

    return state


# ---------------------------------------------------------------------------
# Spreadsheet export
# ---------------------------------------------------------------------------

_EXPORT_ERROR_PREFIX = "[Spreadsheet export:"


def _export_spreadsheet(
    state: PipelineState,
    output_config: dict[str, Any],
    *,
    checkpoint_path: str,
    config_dir: Path,
    resume: bool,
) -> None:
    """Write the selected backend, checkpointing destination and outcome."""
    backend = str(output_config["backend"])
    workbook = project_state(state)
    fingerprint = payload_fingerprint(workbook)
    previous = state.output_results.get(backend, {})

    writer: SpreadsheetWriter
    if backend == "xlsx":
        destination = str(
            xlsx_destination(
                state.topic,
                {"output": output_config},
                config_dir,
            ).resolve()
        )
        writer = XlsxWriter(destination)
    else:
        destination = (
            str(previous.get("destination_id"))
            if resume and previous.get("destination_id")
            else ""
        )
        writer = GoogleSheetsWriter.from_config(
            output_config.get("google_sheets", {}), config_dir
        )

    if (
        previous.get("status") == "success"
        and previous.get("payload_sha256") == fingerprint
        and previous.get("destination_id") == destination
    ):
        logger.info(
            "[Orchestrator] ↩ Skipping Spreadsheet export "
            "(checkpoint: matching {} output)",
            backend,
        )
        return

    # Clear only prior export errors; unrelated agent errors remain intact.
    state.errors = [
        error for error in state.errors
        if not error.startswith(f"{_EXPORT_ERROR_PREFIX}{backend}]")
    ]
    try:
        with trace_span(
            "spreadsheet-export",
            input_data={"topic": state.topic, "backend": backend},
        ):
            destination = writer.ensure_destination(workbook, destination or None)
            # Persist the destination before writing so a partial remote
            # failure cannot create a duplicate on resume.
            state.output_results[backend] = OutputResult(
                status="failed",
                location=None,
                destination_id=destination,
                payload_sha256=fingerprint,
                error="Export did not complete.",
            ).to_dict()
            state.save(checkpoint_path)
            result = writer.write(workbook, destination, fingerprint)

        state.output_results[backend] = result.to_dict()
        location = result.location or ""
        if backend == "google_sheets":
            state.sheet_url = location
        state.save(checkpoint_path)
        flush_traces()
        logger.info(
            "[Orchestrator] ✓ Spreadsheet export — {} written to: {}",
            backend,
            location,
        )
    except Exception as exc:
        summary = safe_exception_summary(exc)
        message = f"{_EXPORT_ERROR_PREFIX}{backend}] {summary}"
        state.output_results[backend] = OutputResult(
            status="failed",
            location=None,
            destination_id=destination,
            payload_sha256=fingerprint,
            error=summary,
        ).to_dict()
        state.add_error(message)
        state.save(checkpoint_path)
        logger.error("[Orchestrator] Spreadsheet export failed: {}", summary)
        flush_traces()
