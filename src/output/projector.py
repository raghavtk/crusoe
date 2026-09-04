"""Projection of pipeline state into a stable, normalized workbook."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from src.output.model import Cell, Workbook, Worksheet

SHEET_ORDER = (
    "Summary", "Papers", "Themes", "Gaps", "Future Work", "Methods",
    "Disagreements", "Reading Order",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(_text(item) for item in value)
    return str(value)


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _ids(value: Any) -> str:
    return "; ".join(_text(item) for item in _items(value))


def _authors(value: Any) -> str:
    names: list[str] = []
    for author in _items(value):
        if isinstance(author, dict):
            names.append(_text(author.get("name") or author.get("authorId")))
        else:
            names.append(_text(author))
    return "; ".join(name for name in names if name)


def _source(paper: dict[str, Any]) -> str:
    for key in ("url", "paperUrl", "doi", "externalIds"):
        value = paper.get(key)
        if isinstance(value, str) and value:
            return value if value.startswith("http") else f"https://doi.org/{value}" if key == "doi" else value
        if key == "externalIds" and isinstance(value, dict) and value.get("DOI"):
            return f"https://doi.org/{value['DOI']}"
    paper_id = _text(paper.get("paperId"))
    return f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""


def _evidence_rows(
    entries: Any, columns: tuple[str, ...], mapping: Iterable[tuple[str, str]],
) -> tuple[tuple[Cell, ...], ...]:
    rows = []
    for entry in _items(entries):
        if not isinstance(entry, dict):
            # Legacy synthesis represents many items as plain strings.
            rows.append((Cell(_text(entry)),) + tuple(Cell("") for _ in columns[1:]))
            continue
        values = []
        for key, kind in mapping:
            value = entry.get(key, "")
            if kind == "ids":
                value = _ids(value)
            elif kind == "text":
                value = _text(value)
            values.append(Cell(value))
        rows.append(tuple(values))
    return tuple(rows)


def project_state(state: Any) -> Workbook:
    """Return the canonical workbook for a completed or legacy state.

    It intentionally tolerates incomplete/older synthesis payloads so a
    checkpoint can still be inspected and exported.
    """
    synthesis = getattr(state, "synthesis", {}) or {}
    if not isinstance(synthesis, dict):
        synthesis = {}
    landscape = synthesis.get("landscape", {})
    if not isinstance(landscape, dict):
        landscape = {}

    summary = Worksheet(
        "Summary", ("Topic", "Summary"),
        ((Cell(_text(getattr(state, "topic", ""))), Cell(_text(synthesis.get("summary_paragraph", "")))),),
    )
    papers_rows = []
    for paper in getattr(state, "papers_curated", []) or []:
        if not isinstance(paper, dict):
            continue
        source = _source(paper)
        papers_rows.append((
            Cell(_text(paper.get("paperId"))), Cell(_text(paper.get("title"))), Cell(source, source or None),
            Cell(_authors(paper.get("authors"))), Cell(_text(paper.get("fieldsOfStudy"))),
            Cell(paper.get("year", "")), Cell(paper.get("citationCount", "")),
            Cell(_text(paper.get("assessment_status"))), Cell(_text(paper.get("methodology"))),
            Cell(_text(paper.get("contribution_type"))), Cell(paper.get("relevance_score", "")),
            Cell(paper.get("confidence_score", "")), Cell(paper.get("reading_priority_score", "")),
            Cell(_text(paper.get("reading_priority"))), Cell(_text(paper.get("relevance_rationale"))),
            Cell(_text(paper.get("one_line_summary"))), Cell(_text(paper.get("abstract"))),
        ))
    papers = Worksheet("Papers", (
        "Paper ID", "Title", "Source URL / DOI", "Authors", "Fields of study", "Year", "Citations", "Assessment status",
        "Methodology", "Contribution", "Relevance score", "Confidence", "Priority score", "Priority",
        "Rationale", "Summary", "Abstract",
    ), tuple(papers_rows))

    themes = Worksheet("Themes", ("Name", "Explanation", "Supporting paper IDs", "Confidence"), _evidence_rows(
        landscape.get("themes", synthesis.get("key_themes", [])), ("Name", "Explanation", "Supporting paper IDs", "Confidence"),
        (("name", "text"), ("explanation", "text"), ("supporting_paper_ids", "ids"), ("confidence", "number"))))
    gap_rows = [
        (Cell("Gap"), *row)
        for row in _evidence_rows(
            landscape.get("gaps", synthesis.get("research_gaps", [])),
            ("Name", "Explanation", "Supporting paper IDs", "Confidence"),
            (("name", "text"), ("explanation", "text"), ("supporting_paper_ids", "ids"), ("confidence", "number")),
        )
    ]
    for limitation in _items(landscape.get("shared_limitations", [])):
        if isinstance(limitation, dict):
            gap_rows.append((
                Cell("Shared limitation"),
                Cell(""),
                Cell(_text(limitation.get("limitation"))),
                Cell(_ids(limitation.get("supporting_paper_ids"))),
                Cell(""),
            ))
    gaps = Worksheet(
        "Gaps",
        ("Type", "Name", "Explanation", "Supporting paper IDs", "Confidence"),
        tuple(gap_rows),
    )
    future = Worksheet("Future Work", ("Recommendation", "Rationale", "Supporting paper IDs", "Confidence"), _evidence_rows(
        landscape.get("future_work", synthesis.get("recommended_future_work", [])), ("Recommendation", "Rationale", "Supporting paper IDs", "Confidence"),
        (("recommendation", "text"), ("rationale", "text"), ("supporting_paper_ids", "ids"), ("confidence", "number"))))
    methods = Worksheet("Methods", ("Methodology", "Observation", "Representative paper IDs"), _evidence_rows(
        landscape.get("methodology_patterns", []), ("Methodology", "Observation", "Representative paper IDs"),
        (("methodology", "text"), ("observation", "text"), ("representative_paper_ids", "ids"))))

    disagreement_rows = []
    for entry in _items(landscape.get("disagreements", [])):
        if not isinstance(entry, dict):
            continue
        positions = _items(entry.get("positions"))
        position_text = "; ".join(_text(p.get("position")) for p in positions if isinstance(p, dict))
        support = _ids([paper_id for p in positions if isinstance(p, dict) for paper_id in _items(p.get("supporting_paper_ids"))])
        disagreement_rows.append((Cell(_text(entry.get("question"))), Cell(position_text), Cell(support), Cell(_text(entry.get("interpretation")))))
    disagreements = Worksheet("Disagreements", ("Question", "Positions", "Supporting paper IDs", "Interpretation"), tuple(disagreement_rows))

    reading_rows = []
    for index, entry in enumerate(_items(synthesis.get("suggested_reading_order", [])), 1):
        if isinstance(entry, dict):
            reading_rows.append((Cell(index), Cell(_text(entry.get("paperId"))), Cell(_text(entry.get("title"))), Cell(_text(entry.get("reason")))))
        else:
            reading_rows.append((Cell(index), Cell(""), Cell(_text(entry)), Cell("")))
    reading = Worksheet("Reading Order", ("Order", "Paper ID", "Title", "Reason"), tuple(reading_rows))
    return Workbook(_text(getattr(state, "topic", "Crusoe research")), (summary, papers, themes, gaps, future, methods, disagreements, reading))


def payload_fingerprint(workbook: Workbook) -> str:
    """Fingerprint data (including links), not backend formatting."""
    payload = {
        "title": workbook.title,
        "sheets": [{"name": sheet.name, "headers": sheet.headers,
                    "rows": [[{"value": cell.value, "hyperlink": cell.hyperlink} for cell in row] for row in sheet.rows]}
                   for sheet in workbook.sheets],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
