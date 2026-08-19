"""
Pipeline State
==============

PipelineState is the single shared data object that flows through all
five Crusoe agents. Each agent reads from it and writes its output back.

Checkpoint / Resume
-------------------
Call state.save(path) after every agent completes. If the pipeline crashes,
call PipelineState.load(path) on restart to pick up from the last successful
agent without re-running earlier stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class KeywordCluster(BaseModel):
    """A validated search theme produced by the Topic Decomposition agent.

    The model is deliberately strict because it is the hand-off contract
    between Topic Decomposition and Discovery.  Keeping it here makes the
    contract available to every pipeline stage without creating an agent-to-
    agent dependency.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    theme: StrictStr
    keywords: list[StrictStr] = Field(min_length=3, max_length=5)
    description: StrictStr

    @field_validator("theme", "description")
    @classmethod
    def _normalise_required_text(cls, value: str) -> str:
        """Collapse whitespace and reject empty text fields."""
        normalised = " ".join(value.split())
        if not normalised:
            raise ValueError("must not be blank")
        return normalised

    @field_validator("keywords")
    @classmethod
    def _normalise_keywords(cls, values: list[str]) -> list[str]:
        """Normalise keywords and require case-insensitive uniqueness."""
        normalised = [" ".join(value.split()) for value in values]
        if any(not value for value in normalised):
            raise ValueError("keywords must not be blank")

        seen: set[str] = set()
        duplicates: list[str] = []
        for keyword in normalised:
            key = keyword.casefold()
            if key in seen:
                duplicates.append(keyword)
            seen.add(key)
        if duplicates:
            raise ValueError(
                "keywords must be case-insensitively unique: "
                + ", ".join(repr(keyword) for keyword in duplicates)
            )
        return normalised


@dataclass
class PipelineState:
    """
    Mutable state container that is threaded through the entire pipeline.

    Attributes
    ----------
    topic : str
        The original research topic string entered by the user.
    keyword_clusters : list[KeywordCluster]
        Validated output of the Topic Decomposition agent. Each cluster has a
        theme, 3--5 unique keywords, and a description.
    papers_raw : list[dict]
        Output of the Discovery agent.
        Each dict has: paperId, title, abstract, year, citationCount, authors, fieldsOfStudy
    papers_curated : list[dict]
        Output of the Paper Curator agent. Same as papers_raw plus its validated
        assessment and deterministic reading-priority fields.
    synthesis : dict
        Output of the Synthesis agent.
        Keys: key_themes, research_gaps, recommended_future_work,
              suggested_reading_order, summary_paragraph
    sheet_url : str | None
        Google Sheets URL written by the Orchestrator.
    errors : list[str]
        Non-fatal errors accumulated during the run (e.g. a single failed
        API call). Fatal errors should raise exceptions instead.
    """

    topic: str = ""
    keyword_clusters: list[KeywordCluster] = field(default_factory=list)
    papers_raw: list[dict] = field(default_factory=list)
    papers_curated: list[dict] = field(default_factory=list)
    synthesis: dict = field(default_factory=dict)
    sheet_url: str | None = None
    errors: list[str] = field(default_factory=list)

    # ── Serialisation helpers ────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Coerce legacy checkpoint dictionaries into typed clusters.

        Checkpoints intentionally retain their historic JSON object shape;
        only their in-memory representation becomes typed.
        """
        self.keyword_clusters = [
            cluster
            if isinstance(cluster, KeywordCluster)
            else KeywordCluster.model_validate(cluster)
            for cluster in self.keyword_clusters
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of the state."""
        return {
            "topic": self.topic,
            "keyword_clusters": [cluster.model_dump() for cluster in self.keyword_clusters],
            "papers_raw": self.papers_raw,
            "papers_curated": self.papers_curated,
            "synthesis": self.synthesis,
            "sheet_url": self.sheet_url,
            "errors": self.errors,
        }

    def save(self, path: str | Path) -> None:
        """
        Serialise state to JSON and write to disk.

        Parameters
        ----------
        path : str | Path
            File path for the checkpoint (e.g. "data/session_checkpoint.json").
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PipelineState":
        """
        Load a checkpoint from disk and return a PipelineState instance.

        Parameters
        ----------
        path : str | Path
            Path to a previously saved checkpoint JSON file.

        Returns
        -------
        PipelineState

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        data: dict = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def has_clusters(self) -> bool:
        """True if topic decomposition has already been completed."""
        return bool(self.keyword_clusters)

    @property
    def has_raw_papers(self) -> bool:
        """True if discovery has already been completed."""
        return bool(self.papers_raw)

    @property
    def has_curated_papers(self) -> bool:
        """True if paper curation has already been completed."""
        return bool(self.papers_curated)

    @property
    def has_synthesis(self) -> bool:
        """True if synthesis has already been completed."""
        return bool(self.synthesis)

    def add_error(self, message: str) -> None:
        """Append a non-fatal error message to the errors list."""
        self.errors.append(message)

    def __repr__(self) -> str:
        return (
            f"PipelineState(topic={self.topic!r}, "
            f"clusters={len(self.keyword_clusters)}, "
            f"papers_raw={len(self.papers_raw)}, "
            f"papers_curated={len(self.papers_curated)}, "
            f"synthesis={'yes' if self.synthesis else 'no'}, "
            f"sheet_url={self.sheet_url!r})"
        )
