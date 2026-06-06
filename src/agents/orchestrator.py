"""
Orchestrator
============

Runs agents 1–4 in sequence, checkpointing state after each one.
On restart (--resume), it skips already-completed stages.
At the end, writes results to Google Sheets.

Google Sheets Output
--------------------
Two tabs are created/updated:
  - "Papers"    : one row per enriched paper
  - "Synthesis" : key themes, gaps, future work, reading order

Requires credentials.json (OAuth 2.0) in the project root.
On first run, opens a browser for consent. Token is cached in token.json.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.agents import (
    discovery,
    enrichment,
    synthesis,
    topic_decomposition,
)
from src.core.state import PipelineState
from src.llm.providers import LLMProvider


def run_pipeline(
    topic: str,
    provider: LLMProvider,
    config: dict,
    resume: bool = False,
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
        Final state after all agents and Google Sheets write.
    """
    checkpoint_path: str = config["pipeline"]["checkpoint_path"]
    max_iterations: int = config["pipeline"]["max_agent_iterations"]
    ss_config: dict = config.get("semantic_scholar", {})
    results_per_query: int = ss_config.get("results_per_query", 20)
    max_total_papers: int = ss_config.get("max_total_papers", 80)
    batch_size: int = config.get("enrichment", {}).get("batch_size", 8)

    # ── Load or initialise state ─────────────────────────────────────────────
    if resume and Path(checkpoint_path).exists():
        logger.info(f"[Orchestrator] Resuming from checkpoint: {checkpoint_path}")
        state = PipelineState.load(checkpoint_path)
        if state.topic != topic and topic:
            logger.warning(
                f"[Orchestrator] Topic mismatch: checkpoint has {state.topic!r}, "
                f"flag has {topic!r}. Using checkpoint topic."
            )
    else:
        state = PipelineState(topic=topic)

    # ── Stage 1: Topic Decomposition ─────────────────────────────────────────
    if not state.has_clusters:
        logger.info("[Orchestrator] Running Topic Decomposition agent...")
        state = topic_decomposition.run(state, provider)
        state.save(checkpoint_path)
        logger.info(f"[Orchestrator] ✓ Topic Decomposition — {len(state.keyword_clusters)} clusters")
    else:
        logger.info(f"[Orchestrator] ↩ Skipping Topic Decomposition (checkpoint: {len(state.keyword_clusters)} clusters)")

    # ── Stage 2: Discovery ───────────────────────────────────────────────────
    if not state.has_raw_papers:
        logger.info("[Orchestrator] Running Discovery agent...")
        state = discovery.run(
            state,
            provider,
            results_per_query=results_per_query,
            max_total_papers=max_total_papers,
            max_iterations=max_iterations,
        )
        state.save(checkpoint_path)
        logger.info(f"[Orchestrator] ✓ Discovery — {len(state.papers_raw)} papers found")
    else:
        logger.info(f"[Orchestrator] ↩ Skipping Discovery (checkpoint: {len(state.papers_raw)} papers)")

    # ── Stage 3: Enrichment ──────────────────────────────────────────────────
    if not state.has_enriched_papers:
        logger.info("[Orchestrator] Running Enrichment agent...")
        state = enrichment.run(state, provider, batch_size=batch_size)
        state.save(checkpoint_path)
        n_batches = (len(state.papers_enriched) + batch_size - 1) // batch_size
        logger.info(
            f"[Orchestrator] ✓ Enrichment — {len(state.papers_enriched)} papers enriched ({n_batches} batches)"
        )
    else:
        logger.info(f"[Orchestrator] ↩ Skipping Enrichment (checkpoint: {len(state.papers_enriched)} papers)")

    # ── Stage 4: Synthesis ───────────────────────────────────────────────────
    if not state.has_synthesis:
        logger.info("[Orchestrator] Running Synthesis agent...")
        state = synthesis.run(state, provider)
        state.save(checkpoint_path)
        logger.info("[Orchestrator] ✓ Synthesis — complete")
    else:
        logger.info("[Orchestrator] ↩ Skipping Synthesis (checkpoint: already done)")

    # ── Stage 5: Google Sheets ───────────────────────────────────────────────
    sheets_config: dict = config.get("google_sheets", {})
    try:
        sheet_url = write_to_google_sheets(state, sheets_config, config_path="config.yaml")
        state.sheet_url = sheet_url
        state.save(checkpoint_path)
        logger.info(f"[Orchestrator] ✓ Google Sheets — written to: {sheet_url}")
    except Exception as exc:
        msg = f"[Orchestrator] Google Sheets write failed: {exc}"
        logger.error(msg)
        state.add_error(msg)

    if state.errors:
        logger.warning(f"[Orchestrator] Pipeline completed with {len(state.errors)} non-fatal error(s):")
        for err in state.errors:
            logger.warning(f"  - {err}")

    return state


# ---------------------------------------------------------------------------
# Google Sheets writer
# ---------------------------------------------------------------------------

def write_to_google_sheets(
    state: PipelineState,
    sheets_config: dict,
    config_path: str = "config.yaml",
) -> str:
    """
    Write enriched papers and synthesis to Google Sheets.

    Creates the spreadsheet if sheet_id is blank in config.yaml, then
    writes back the new sheet_id so future runs reuse the same sheet.

    Parameters
    ----------
    state : PipelineState
        Pipeline state with papers_enriched and synthesis populated.
    sheets_config : dict
        The google_sheets section of config.yaml.
    config_path : str
        Path to config.yaml (for updating sheet_id after creation).

    Returns
    -------
    str
        URL of the Google Sheet.
    """
    from google.oauth2.credentials import Credentials  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]

    creds_file: str = sheets_config.get("credentials_file", "credentials.json")
    sheet_id: str = sheets_config.get("sheet_id", "")
    token_file = "token.json"

    # ── Authenticate ─────────────────────────────────────────────────────────
    creds: Credentials | None = None
    if Path(token_file).exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request  # type: ignore
            creds.refresh(Request())
        else:
            if not Path(creds_file).exists():
                raise FileNotFoundError(
                    f"Google OAuth credentials file not found: {creds_file!r}. "
                    "Download it from Google Cloud Console and place it in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(token_file).write_text(creds.to_json())

    sheets_service = build("sheets", "v4", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # ── Create spreadsheet if needed ─────────────────────────────────────────
    if not sheet_id:
        spreadsheet_title = f"Crusoe — {state.topic}"
        spreadsheet = sheets_service.spreadsheets().create(
            body={
                "properties": {"title": spreadsheet_title},
                "sheets": [
                    {"properties": {"title": "Papers"}},
                    {"properties": {"title": "Synthesis"}},
                ],
            }
        ).execute()
        sheet_id = spreadsheet["spreadsheetId"]
        logger.info(f"[Sheets] Created new spreadsheet: {sheet_id}")

        # Persist the new sheet_id back to config.yaml
        _update_config_sheet_id(config_path, sheet_id)
    else:
        logger.info(f"[Sheets] Using existing spreadsheet: {sheet_id}")
        _ensure_tabs_exist(sheets_service, sheet_id, ["Papers", "Synthesis"])

    # ── Write Papers tab ─────────────────────────────────────────────────────
    _write_papers_tab(sheets_service, sheet_id, state.papers_enriched)

    # ── Write Synthesis tab ──────────────────────────────────────────────────
    _write_synthesis_tab(sheets_service, sheet_id, state.synthesis)

    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    return sheet_url


def _write_papers_tab(service: Any, sheet_id: str, papers: list[dict]) -> None:
    """Write all enriched papers to the 'Papers' tab."""
    HEADERS = [
        "Title", "Year", "Authors", "Citations", "Relevance",
        "Methodology", "Contribution", "Priority", "Summary", "Abstract",
    ]

    rows: list[list[Any]] = [HEADERS]
    for p in papers:
        authors = p.get("authors", [])
        if isinstance(authors, list):
            authors_str = "; ".join(str(a) for a in authors[:5])
        else:
            authors_str = str(authors)

        abstract = (p.get("abstract") or "")[:300]

        rows.append([
            p.get("title", ""),
            p.get("year", ""),
            authors_str,
            p.get("citationCount", 0),
            p.get("relevance_score", ""),
            p.get("methodology", ""),
            p.get("contribution_type", ""),
            "Yes" if p.get("priority_read") else "No",
            p.get("one_line_summary", ""),
            abstract,
        ])

    _clear_and_write(service, sheet_id, "Papers", rows)
    _freeze_header_row(service, sheet_id, "Papers")
    logger.info(f"[Sheets] Papers tab: wrote {len(rows) - 1} paper rows.")


def _write_synthesis_tab(service: Any, sheet_id: str, synthesis: dict) -> None:
    """Write the synthesis output to the 'Synthesis' tab."""
    rows: list[list[str]] = []

    rows.append(["SUMMARY"])
    rows.append([synthesis.get("summary_paragraph", "")])
    rows.append([])

    rows.append(["KEY THEMES"])
    for theme in synthesis.get("key_themes", []):
        rows.append([theme])
    rows.append([])

    rows.append(["RESEARCH GAPS"])
    for gap in synthesis.get("research_gaps", []):
        rows.append([gap])
    rows.append([])

    rows.append(["RECOMMENDED FUTURE WORK"])
    for work in synthesis.get("recommended_future_work", []):
        rows.append([work])
    rows.append([])

    rows.append(["SUGGESTED READING ORDER", "", "Reason"])
    for i, entry in enumerate(synthesis.get("suggested_reading_order", []), 1):
        rows.append([
            f"{i}. {entry.get('title', '')}",
            entry.get("paperId", ""),
            entry.get("reason", ""),
        ])

    _clear_and_write(service, sheet_id, "Synthesis", rows)
    logger.info(f"[Sheets] Synthesis tab: wrote {len(rows)} rows.")


def _clear_and_write(service: Any, sheet_id: str, tab_name: str, rows: list[list]) -> None:
    """Clear a tab and write new data."""
    range_str = f"{tab_name}!A1"
    service.spreadsheets().values().clear(
        spreadsheetId=sheet_id,
        range=range_str,
    ).execute()
    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=range_str,
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()


def _freeze_header_row(service: Any, sheet_id: str, tab_name: str) -> None:
    """Freeze the first row of a tab."""
    sheet_meta = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    tab_id = None
    for s in sheet_meta.get("sheets", []):
        if s["properties"]["title"] == tab_name:
            tab_id = s["properties"]["sheetId"]
            break
    if tab_id is None:
        return

    service.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id,
        body={
            "requests": [{
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": tab_id,
                        "gridProperties": {"frozenRowCount": 1},
                    },
                    "fields": "gridProperties.frozenRowCount",
                }
            }]
        },
    ).execute()


def _ensure_tabs_exist(service: Any, sheet_id: str, tab_names: list[str]) -> None:
    """Create tabs that don't already exist in the spreadsheet."""
    meta = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    existing = {s["properties"]["title"] for s in meta.get("sheets", [])}
    requests = []
    for name in tab_names:
        if name not in existing:
            requests.append({"addSheet": {"properties": {"title": name}}})
    if requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=sheet_id,
            body={"requests": requests},
        ).execute()


def _update_config_sheet_id(config_path: str, sheet_id: str) -> None:
    """Persist the newly created sheet_id back into config.yaml."""
    try:
        p = Path(config_path)
        cfg = yaml.safe_load(p.read_text()) or {}
        cfg.setdefault("google_sheets", {})["sheet_id"] = sheet_id
        p.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        logger.info(f"[Sheets] Saved sheet_id={sheet_id!r} to {config_path}")
    except Exception as exc:
        logger.warning(f"[Sheets] Could not update {config_path} with sheet_id: {exc}")
