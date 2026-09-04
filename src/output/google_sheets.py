"""Google Sheets adapter; service injection keeps it fully offline-testable."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable

from src.output.model import OutputResult, Workbook

GOOGLE_CELL_LIMIT = 50_000


def _google_value(value: Any) -> Any:
    """Return a Sheets-safe value with explicit cell-limit truncation."""
    if not isinstance(value, str) or len(value) <= GOOGLE_CELL_LIMIT:
        return value
    suffix = "… [truncated: Google Sheets cell limit reached]"
    return value[: GOOGLE_CELL_LIMIT - len(suffix)] + suffix


class GoogleSheetsWriter:
    def __init__(self, service: Any | None = None, service_factory: Callable[[], Any] | None = None) -> None:
        self._service = service
        self._service_factory = service_factory

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service
        if self._service_factory is not None:
            self._service = self._service_factory()
            return self._service
        raise RuntimeError("GoogleSheetsWriter requires a service or service_factory")

    @classmethod
    def from_config(cls, settings: dict[str, Any], config_dir: str | Path) -> "GoogleSheetsWriter":
        """Build a lazily authenticated writer using config-relative paths."""
        base = Path(config_dir)
        credentials_file = Path(settings.get("credentials_file", "credentials.json"))
        token_file = Path(settings.get("token_file", "token.json"))
        if not credentials_file.is_absolute():
            credentials_file = base / credentials_file
        if not token_file.is_absolute():
            token_file = base / token_file

        def service_factory() -> Any:
            from google.oauth2.credentials import Credentials  # type: ignore
            from google.auth.transport.requests import Request  # type: ignore
            from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
            from googleapiclient.discovery import build  # type: ignore

            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            credentials: Any | None = None
            if token_file.exists():
                credentials = Credentials.from_authorized_user_file(str(token_file), scopes)
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                else:
                    if not credentials_file.exists():
                        raise FileNotFoundError(f"Google OAuth credentials file not found: {credentials_file}")
                    credentials = InstalledAppFlow.from_client_secrets_file(str(credentials_file), scopes).run_local_server(port=0)
                token_file.parent.mkdir(parents=True, exist_ok=True)
                token_file.write_text(credentials.to_json(), encoding="utf-8")
            return build("sheets", "v4", credentials=credentials)

        return cls(service_factory=service_factory)

    def ensure_destination(self, workbook: Workbook, destination_id: str | None = None) -> str:
        service = self._get_service()
        if destination_id:
            self._ensure_tabs(service, destination_id, [sheet.name for sheet in workbook.sheets])
            return destination_id
        response = service.spreadsheets().create(body={"properties": {"title": f"Crusoe — {workbook.title}"}, "sheets": [{"properties": {"title": sheet.name}} for sheet in workbook.sheets]}).execute()
        return str(response["spreadsheetId"])

    def write(
        self, workbook: Workbook, destination_id: str, payload_sha256: str = ""
    ) -> OutputResult:
        service = self._get_service()
        self._ensure_tabs(service, destination_id, [sheet.name for sheet in workbook.sheets])
        for sheet in workbook.sheets:
            # An unbounded sheet range clears obsolete data from previous larger exports.
            service.spreadsheets().values().clear(spreadsheetId=destination_id, range=f"'{sheet.name}'").execute()
            values = [[_google_value(value) for value in row] for row in sheet.values]
            service.spreadsheets().values().update(spreadsheetId=destination_id, range=f"'{sheet.name}'!A1", valueInputOption="RAW", body={"values": values}).execute()
        self._format(service, destination_id, workbook)
        return OutputResult(
            status="success",
            location=f"https://docs.google.com/spreadsheets/d/{destination_id}",
            destination_id=destination_id,
            payload_sha256=payload_sha256,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _ensure_tabs(service: Any, destination_id: str, names: list[str]) -> None:
        metadata = service.spreadsheets().get(spreadsheetId=destination_id).execute()
        existing = {item.get("properties", {}).get("title") for item in metadata.get("sheets", [])}
        requests = [{"addSheet": {"properties": {"title": name}}} for name in names if name not in existing]
        if requests:
            service.spreadsheets().batchUpdate(spreadsheetId=destination_id, body={"requests": requests}).execute()

    @staticmethod
    def _format(service: Any, destination_id: str, workbook: Workbook) -> None:
        metadata = service.spreadsheets().get(spreadsheetId=destination_id).execute()
        ids = {item.get("properties", {}).get("title"): item.get("properties", {}).get("sheetId") for item in metadata.get("sheets", [])}
        requests = []
        for sheet in workbook.sheets:
            sheet_id = ids.get(sheet.name)
            if sheet_id is None:
                continue
            sheet_requests = [
                {"updateSheetProperties": {"properties": {"sheetId": sheet_id, "gridProperties": {"frozenRowCount": 1}}, "fields": "gridProperties.frozenRowCount"}},
                {"repeatCell": {"range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1}, "cell": {"userEnteredFormat": {"backgroundColor": {"red": 0.12, "green": 0.28, "blue": 0.46}, "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}, "wrapStrategy": "WRAP", "verticalAlignment": "TOP"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,wrapStrategy,verticalAlignment)"}},
                {"setBasicFilter": {"filter": {"range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": max(1, len(sheet.rows) + 1), "startColumnIndex": 0, "endColumnIndex": len(sheet.headers)}}}},
                {"autoResizeDimensions": {"dimensions": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": len(sheet.headers)}}},
            ]
            if sheet.rows:
                sheet_requests.insert(2, {"repeatCell": {"range": {"sheetId": sheet_id, "startRowIndex": 1, "endRowIndex": len(sheet.rows) + 1}, "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}}, "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)"}})
            requests.extend(sheet_requests)
        if requests:
            service.spreadsheets().batchUpdate(spreadsheetId=destination_id, body={"requests": requests}).execute()
