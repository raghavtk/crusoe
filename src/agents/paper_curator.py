"""Assess, classify, summarise, and rank discovered academic papers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, field_validator

from src.core.state import PipelineState
from src.llm.providers import LLMProvider
from src.observability.langfuse_tracing import trace_span


Methodology = Literal[
    "empirical", "survey", "formal analysis", "case study", "simulation", "theoretical", "other"
]
ContributionType = Literal[
    "attack", "defense", "protocol", "framework", "tool", "dataset", "survey", "other"
]


class PaperAssessment(BaseModel):
    """Strict hand-off contract for one LLM-produced paper assessment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    paperId: StrictStr
    relevance_score: StrictInt = Field(ge=1, le=5)
    relevance_rationale: StrictStr = Field(min_length=10, max_length=300)
    confidence_score: float = Field(ge=0.0, le=1.0, strict=True)
    methodology: Methodology
    contribution_type: ContributionType
    one_line_summary: StrictStr = Field(min_length=10, max_length=300)

    @field_validator("paperId", "relevance_rationale", "one_line_summary")
    @classmethod
    def _normalise_text(cls, value: str) -> str:
        normalised = " ".join(value.split())
        if not normalised:
            raise ValueError("must not be blank")
        return normalised

    @field_validator("one_line_summary")
    @classmethod
    def _require_one_sentence(cls, value: str) -> str:
        """Require one terminal sentence; generated summaries should spell out abbreviations."""
        if value[-1] not in ".!?":
            raise ValueError("must end with sentence punctuation")
        if re.search(r"[.!?]\s+\S", value):
            raise ValueError("must contain exactly one sentence")
        return value


class CuratorValidationError(ValueError):
    """Raised when an LLM batch does not satisfy the curator contract."""


@dataclass(frozen=True)
class BatchOutcome:
    assessments: list[PaperAssessment]
    retry_used: bool
    validation_failures: int
    error: str | None = None


SYSTEM_PROMPT = """You are a rigorous academic paper curator. Assess only the evidence in each
title and abstract. Topical word overlap is not enough to establish research relevance.
When an abstract is missing or weak, lower confidence and avoid unsupported claims.
Return valid JSON only: no prose, markdown fences, or explanations outside the JSON."""

BATCH_USER_TEMPLATE = """Assess these {n} papers for the research topic: {topic}

Return one JSON array item for every input paper, in the same order, with exactly:
- "paperId": copied exactly from the input
- "relevance_score": integer 1-5
  1 = unrelated; 2 = tangential; 3 = useful supporting context;
  4 = directly relevant; 5 = central or essential to the topic
- "relevance_rationale": concise explanation grounded in the title or abstract
- "confidence_score": number from 0.0-1.0 reflecting evidence sufficiency
- "methodology": one of ["empirical", "survey", "formal analysis", "case study", "simulation", "theoretical", "other"]
- "contribution_type": one of ["attack", "defense", "protocol", "framework", "tool", "dataset", "survey", "other"]
- "one_line_summary": one concise sentence describing the paper's main contribution

Papers:
{papers_json}

Return the JSON array only."""

REPAIR_TEMPLATE = """The response to the complete request above failed validation.

Validation errors:
{errors}

Return a complete corrected JSON array for all of these papers, exactly once each:
{papers_json}

Return a corrected JSON array only."""


def run(state: PipelineState, provider: LLMProvider, batch_size: int = 8) -> PipelineState:
    """Curate every discovered paper and store a deterministic ranked result."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("[Paper Curator] batch_size must be a positive integer.")
    if not state.papers_raw:
        raise ValueError("[Paper Curator] papers_raw is empty — run Discovery first.")

    papers = state.papers_raw
    batches = _make_batches(papers, batch_size)
    logger.info(
        f"[Paper Curator] Curating {len(papers)} papers in {len(batches)} batch(es) of {batch_size}."
    )
    assessment_map: dict[str, PaperAssessment] = {}
    failed_ids: set[str] = set()

    for batch_idx, batch in enumerate(batches, 1):
        logger.info(f"[Paper Curator] Batch {batch_idx}/{len(batches)} ({len(batch)} papers)")
        with trace_span(
            f"paper-curator-batch-{batch_idx}",
            input_data={"batch_idx": batch_idx, "batch_size": len(batch)},
        ) as span:
            outcome = _curate_batch(batch=batch, topic=state.topic, provider=provider)
            if span is not None:
                span.update(output={
                    "validated_count": len(outcome.assessments),
                    "validation_failures": outcome.validation_failures,
                    "retry_used": outcome.retry_used,
                    "status": "failed" if outcome.error else "success",
                })

        if outcome.error:
            ids = [str(p.get("paperId", "")) for p in batch]
            failed_ids.update(ids)
            state.add_error(
                f"[Paper Curator] Batch {batch_idx} failed after repair attempt: {outcome.error}"
            )
            logger.error(state.errors[-1])
        else:
            assessment_map.update({item.paperId: item for item in outcome.assessments})

    curated: list[dict] = []
    for discovery_index, paper in enumerate(papers):
        paper_id = str(paper.get("paperId", ""))
        result = dict(paper)
        assessment = assessment_map.get(paper_id)
        if assessment is not None:
            result.update(assessment.model_dump())
            result["assessment_status"] = "success"
        else:
            result.update({
                "relevance_score": None,
                "relevance_rationale": "Assessment unavailable after validation failure.",
                "confidence_score": 0.0,
                "methodology": "other",
                "contribution_type": "other",
                "one_line_summary": "Assessment unavailable after validation failure.",
                "assessment_status": "failed",
            })
            failed_ids.add(paper_id)
        result["_discovery_index"] = discovery_index
        curated.append(result)

    _apply_priority_scores(curated)
    curated.sort(key=lambda p: (-p["reading_priority_score"], p["_discovery_index"]))
    for paper in curated:
        paper.pop("_discovery_index", None)

    state.papers_curated = curated
    logger.info(
        f"[Paper Curator] Done. {len(curated) - len(failed_ids)} assessed, "
        f"{len(failed_ids)} failed, {sum(p['reading_priority'] == 'high' for p in curated)} high priority."
    )
    return state


def _curate_batch(*, batch: list[dict], topic: str, provider: LLMProvider) -> BatchOutcome:
    prompt_papers = _papers_for_prompt(batch)
    initial_prompt = BATCH_USER_TEMPLATE.format(
        n=len(batch), topic=topic, papers_json=json.dumps(prompt_papers, indent=2, ensure_ascii=False)
    )
    raw = provider.call(
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": initial_prompt}],
        tools=[],
    ).get("content", "")
    try:
        return BatchOutcome(_parse_assessments(raw, batch), False, 0)
    except CuratorValidationError as first_error:
        logger.warning(f"[Paper Curator] Batch validation failed; requesting repair: {first_error}")
        repair_prompt = REPAIR_TEMPLATE.format(
            errors=str(first_error),
            papers_json=json.dumps(prompt_papers, indent=2, ensure_ascii=False),
        )
        repaired = provider.call(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{initial_prompt}\n\n{repair_prompt}"}],
            tools=[],
        ).get("content", "")
        try:
            return BatchOutcome(_parse_assessments(repaired, batch), True, 1)
        except CuratorValidationError as second_error:
            return BatchOutcome([], True, 2, str(second_error))


def _papers_for_prompt(batch: list[dict]) -> list[dict]:
    return [
        {
            "paperId": p.get("paperId", f"unknown_{i}"),
            "title": p.get("title", ""),
            "abstract": (p.get("abstract") or "")[:1200],
            "year": p.get("year"),
            "citationCount": p.get("citationCount", 0),
        }
        for i, p in enumerate(batch)
    ]


def _make_batches(papers: list[dict], batch_size: int) -> list[list[dict]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]


def _parse_assessments(raw: str, batch: list[dict]) -> list[PaperAssessment]:
    cleaned = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CuratorValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise CuratorValidationError("response must be a JSON array")

    assessments: list[PaperAssessment] = []
    errors: list[str] = []
    for index, item in enumerate(data):
        try:
            assessments.append(PaperAssessment.model_validate(item))
        except ValidationError as exc:
            errors.append(f"item {index}: {exc}")
    if errors:
        raise CuratorValidationError("; ".join(errors))

    expected = [str(p.get("paperId", "")) for p in batch]
    actual = [item.paperId for item in assessments]
    duplicates = sorted({pid for pid in actual if actual.count(pid) > 1})
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    if duplicates or missing or unknown or len(actual) != len(expected):
        raise CuratorValidationError(
            f"paperId mismatch: duplicates={duplicates}, missing={missing}, unknown={unknown}"
        )
    return assessments


def _percentile_values(values: list[float]) -> list[float]:
    """Return average-rank percentiles, assigning 0.5 when no comparison exists."""
    if len(values) <= 1 or len(set(values)) == 1:
        return [0.5] * len(values)
    sorted_values = sorted(values)
    denominator = len(values) - 1
    output: list[float] = []
    for value in values:
        positions = [i for i, candidate in enumerate(sorted_values) if candidate == value]
        output.append((positions[0] + positions[-1]) / 2 / denominator)
    return output


def _apply_priority_scores(papers: list[dict]) -> None:
    citations = [float(max(0, p.get("citationCount") or 0)) for p in papers]
    known_years = [float(p["year"]) for p in papers if isinstance(p.get("year"), int)]
    year_percentiles = _percentile_values(known_years)
    year_by_value = dict(zip(known_years, year_percentiles))
    citation_percentiles = _percentile_values(citations)

    for paper, citation_percentile in zip(papers, citation_percentiles):
        if paper.get("assessment_status") != "success":
            score = 0.0
        else:
            relevance = (paper["relevance_score"] - 1) / 4
            confidence = paper["confidence_score"]
            year = paper.get("year")
            recency = year_by_value.get(float(year), 0.5) if isinstance(year, int) else 0.5
            score = round(
                100 * (0.60 * relevance + 0.15 * confidence + 0.15 * citation_percentile + 0.10 * recency),
                1,
            )
        paper["reading_priority_score"] = score
        paper["reading_priority"] = "high" if score >= 70 else "medium" if score >= 40 else "low"
