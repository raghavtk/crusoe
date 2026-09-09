"""Safe summaries for exceptions originating at external-service boundaries."""

from __future__ import annotations


def safe_exception_summary(exc: Exception) -> str:
    """Return allow-listed metadata without rendering messages, responses, or locals."""
    details: list[str] = []
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            details.append(f"HTTP {value}")
            break
    status = getattr(exc, "status", None)
    if (
        isinstance(status, str)
        and status
        and len(status) <= 64
        and all(character.isupper() or character.isdigit() or character == "_" for character in status)
    ):
        details.append(status)
    suffix = f" ({' '.join(details)})" if details else ""
    return f"{type(exc).__name__}{suffix}"
