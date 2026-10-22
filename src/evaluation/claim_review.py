"""Human judgments over immutable synthesis claims and cited source abstracts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from src.agents.synthesis import Landscape
from src.core.state import PipelineState


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceAbstract(ReviewModel):
    paper_id: StrictStr
    title: StrictStr
    abstract: StrictStr
    missing: bool = False


class ReviewClaim(ReviewModel):
    claim_id: StrictStr
    path: StrictStr
    label: StrictStr
    text: StrictStr
    evidence: list[SourceAbstract]
    model_confidence: float | None = None


class ReviewSnapshot(ReviewModel):
    topic: StrictStr
    synthesis_sha256: StrictStr
    claims: list[ReviewClaim]
    uncited_sections: list[StrictStr]


class Judgment(ReviewModel):
    status: Literal["supported", "unsupported", "uncertain"]
    note: StrictStr = ""
    reviewer: StrictStr = ""
    reviewed_at: StrictStr


class ClaimReview(ReviewModel):
    version: Literal[1]
    snapshot: ReviewSnapshot
    snapshot_sha256: StrictStr
    judgments: dict[StrictStr, Judgment] = Field(default_factory=dict)

    @field_validator("version", mode="before")
    @classmethod
    def _supported_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported claim-review version")
        return value


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_snapshot(state: PipelineState) -> ReviewSnapshot:
    """Expose cited context without inventing claim-specific supporting spans.

    Abstracts are the full, exact stored source text, not curator summaries or
    reconstructed model inputs. Legacy uncited summaries cannot be aligned.
    """
    if "landscape" not in state.synthesis:
        raise ValueError("Claim review requires a synthesis landscape with paper citations")
    landscape = Landscape.model_validate(state.synthesis["landscape"]).model_dump()
    papers = {}
    for paper in state.papers_curated:
        identifier = paper.get("paperId")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in papers:
            raise ValueError("Curated paper IDs must be nonblank and unique")
        papers[identifier] = paper

    claims = []

    def append(path, label, text, ids, confidence=None):
        sources = []
        for identifier in dict.fromkeys(ids):
            paper = papers.get(identifier)
            sources.append(SourceAbstract(
                paper_id=identifier,
                title=paper.get("title", "") if paper else "",
                abstract=(paper.get("abstract") or "") if paper else "",
                missing=paper is None,
            ))
        claims.append(ReviewClaim(
            claim_id=_digest({"path": path, "label": label, "text": text, "ids": ids}),
            path=path, label=label, text=text, evidence=sources, model_confidence=confidence,
        ))

    for section in ("themes", "gaps"):
        for index, entry in enumerate(landscape[section]):
            append(f"landscape.{section}.{index}.explanation", entry["name"], entry["explanation"],
                   entry["supporting_paper_ids"], entry["confidence"])
    for index, entry in enumerate(landscape["future_work"]):
        for field in ("recommendation", "rationale"):
            append(f"landscape.future_work.{index}.{field}", "Future work: " + entry["recommendation"],
                   entry[field], entry["supporting_paper_ids"], entry["confidence"])
    for index, entry in enumerate(landscape["methodology_patterns"]):
        append(f"landscape.methodology_patterns.{index}.observation", entry["methodology"],
               entry["observation"], entry["representative_paper_ids"])
    for index, entry in enumerate(landscape["shared_limitations"]):
        append(f"landscape.shared_limitations.{index}.limitation", "Shared limitation",
               entry["limitation"], entry["supporting_paper_ids"])
    for index, entry in enumerate(landscape["disagreements"]):
        ids = []
        for position_index, position in enumerate(entry["positions"]):
            append(f"landscape.disagreements.{index}.positions.{position_index}.position",
                   entry["question"], position["position"], position["supporting_paper_ids"])
            ids.extend(position["supporting_paper_ids"])
        append(f"landscape.disagreements.{index}.interpretation", entry["question"],
               entry["interpretation"], list(dict.fromkeys(ids)))

    return ReviewSnapshot(
        topic=state.topic, synthesis_sha256=_digest(state.synthesis), claims=claims,
        uncited_sections=[name for name in (
            "summary_paragraph", "key_themes", "research_gaps", "recommended_future_work",
            "suggested_reading_order",
        ) if state.synthesis.get(name)],
    )


def new_review(snapshot: ReviewSnapshot) -> ClaimReview:
    return ClaimReview(version=1, snapshot=snapshot, snapshot_sha256=_digest(snapshot.model_dump()))


def _validate_review(review: ClaimReview) -> None:
    if review.snapshot_sha256 != _digest(review.snapshot.model_dump()):
        raise ValueError("Review snapshot checksum does not match its contents")
    ids = [claim.claim_id for claim in review.snapshot.claims]
    if len(ids) != len(set(ids)):
        raise ValueError("Review contains duplicate claim IDs")
    if set(review.judgments) - set(ids):
        raise ValueError("Review contains an unknown claim judgment")


def load_review(path: str | Path, snapshot: ReviewSnapshot) -> ClaimReview:
    review = ClaimReview.model_validate_json(Path(path).read_text(encoding="utf-8"))
    _validate_review(review)
    if review.snapshot_sha256 != _digest(snapshot.model_dump()):
        raise ValueError("Synthesis or cited evidence changed; choose a new review output file")
    return review


def record_judgment(
    review: ClaimReview, claim_id: str, status: str, *, note: str = "", reviewer: str = "",
) -> ClaimReview:
    _validate_review(review)
    if claim_id not in {claim.claim_id for claim in review.snapshot.claims}:
        raise ValueError("Cannot judge an unknown claim")
    judgment = Judgment(status=status, note=note, reviewer=reviewer,
                        reviewed_at=datetime.now(timezone.utc).isoformat())
    return review.model_copy(update={"judgments": {**review.judgments, claim_id: judgment}})


def review_counts(review: ClaimReview) -> dict[str, int]:
    """Count human labels. Unreviewed items are distinct from uncertain judgments."""
    _validate_review(review)
    return {
        **{status: sum(item.status == status for item in review.judgments.values())
           for status in ("supported", "unsupported", "uncertain")},
        "unreviewed": len(review.snapshot.claims) - len(review.judgments),
    }


def save_review(review: ClaimReview, path: str | Path) -> None:
    """Replace the artifact atomically. Interactive writers hold review_lock."""
    _validate_review(review)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".claim-review-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(review.model_dump_json(indent=2))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def review_lock(path: str | Path):
    """Exclude concurrent CLI writers. A killed process leaves a fail-closed lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Review is already open; verify its owner stopped before removing the lock") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)
