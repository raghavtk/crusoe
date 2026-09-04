"""Backend-neutral spreadsheet data structures.

The model deliberately contains only values and presentation hints.  This
makes the potentially complicated synthesis-to-spreadsheet transformation
deterministic and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Cell:
    value: Any = ""
    hyperlink: str | None = None


@dataclass(frozen=True)
class Worksheet:
    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...] = ()

    @property
    def values(self) -> list[list[Any]]:
        return [[cell.value for cell in row] for row in (self.headers_as_cells, *self.rows)]

    @property
    def headers_as_cells(self) -> tuple[Cell, ...]:
        return tuple(Cell(header) for header in self.headers)


@dataclass(frozen=True)
class Workbook:
    title: str
    sheets: tuple[Worksheet, ...]


@dataclass(frozen=True)
class OutputResult:
    status: str
    location: str | None
    destination_id: str | None
    payload_sha256: str
    error: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "location": self.location,
            "destination_id": self.destination_id,
            "payload_sha256": self.payload_sha256,
            "error": self.error,
            "completed_at": self.completed_at,
        }


class SpreadsheetWriter(Protocol):
    """Common contract implemented by every spreadsheet backend."""

    def ensure_destination(
        self, workbook: Workbook, destination_id: str | None = None
    ) -> str: ...

    def write(
        self, workbook: Workbook, destination_id: str, payload_sha256: str
    ) -> OutputResult: ...
