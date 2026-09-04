"""Pluggable spreadsheet export interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .google_sheets import GoogleSheetsWriter
from .model import Cell, OutputResult, SpreadsheetWriter, Workbook, Worksheet
from .projector import SHEET_ORDER, payload_fingerprint, project_state
from .xlsx import XlsxWriter, write_xlsx, xlsx_destination


def validate_output_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize export configuration before pipeline work."""
    output = config.get("output")
    if output is None:
        return {"backend": "xlsx", "xlsx": {"directory": "data/outputs"}}
    if not isinstance(output, dict):
        raise ValueError("output configuration must be a mapping")
    backend = output.get("backend", "xlsx")
    if backend not in {"xlsx", "google_sheets"}:
        raise ValueError("output.backend must be 'xlsx' or 'google_sheets'")
    normalized = dict(output)
    normalized["backend"] = backend
    if backend == "xlsx":
        section = normalized.get("xlsx", {})
        if not isinstance(section, dict):
            raise ValueError("output.xlsx must be a mapping")
        directory = section.get("directory", "data/outputs")
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("output.xlsx.directory must be a non-empty string")
        normalized["xlsx"] = {"directory": directory}
    else:
        section = normalized.get("google_sheets", {})
        if not isinstance(section, dict):
            raise ValueError("output.google_sheets must be a mapping")
        for key in ("credentials_file", "token_file"):
            if key in section and (
                not isinstance(section[key], str) or not section[key].strip()
            ):
                raise ValueError(
                    f"output.google_sheets.{key} must be a non-empty string"
                )
        normalized["google_sheets"] = section
    return normalized


__all__ = ["Cell", "GoogleSheetsWriter", "OutputResult", "SpreadsheetWriter", "SHEET_ORDER", "Workbook", "Worksheet", "XlsxWriter", "payload_fingerprint", "project_state", "validate_output_config", "write_xlsx", "xlsx_destination"]
