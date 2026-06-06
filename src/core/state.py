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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineState:
    """
    Mutable state container that is threaded through the entire pipeline.

    Attributes
    ----------
    topic : str
        The original research topic string entered by the user.
    keyword_clusters : list[dict]
        Output of the Topic Decomposition agent.
        Each dict has: {"theme": str, "keywords": list[str], "description": str}
    papers_raw : list[dict]
        Output of the Discovery agent.
        Each dict has: paperId, title, abstract, year, citationCount, authors, fieldsOfStudy
    papers_enriched : list[dict]
        Output of the Enrichment agent. Same as papers_raw plus:
        relevance_score, methodology, contribution_type, priority_read, one_line_summary
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
    keyword_clusters: list[dict] = field(default_factory=list)
    papers_raw: list[dict] = field(default_factory=list)
    papers_enriched: list[dict] = field(default_factory=list)
    synthesis: dict = field(default_factory=dict)
    sheet_url: str | None = None
    errors: list[str] = field(default_factory=list)

    # ── Serialisation helpers ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of the state."""
        return asdict(self)

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
    def has_enriched_papers(self) -> bool:
        """True if enrichment has already been completed."""
        return bool(self.papers_enriched)

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
            f"papers_enriched={len(self.papers_enriched)}, "
            f"synthesis={'yes' if self.synthesis else 'no'}, "
            f"sheet_url={self.sheet_url!r})"
        )
