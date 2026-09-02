"""Offline tests for Langfuse tracing lifecycle behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from src.observability import langfuse_tracing
from src.core.errors import safe_exception_summary


class FakeLangfuse:
    """Minimal Langfuse replacement for lifecycle tests."""

    def __init__(self, **_: object) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_init_resets_shutdown_guard_for_reinitialised_client(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeLangfuse))
    monkeypatch.setattr(langfuse_tracing, "_shutdown_done", True)

    assert langfuse_tracing.init_langfuse() is True
    first_client = langfuse_tracing._client
    langfuse_tracing.shutdown_traces()

    assert first_client.shutdown_calls == 1
    assert langfuse_tracing.init_langfuse() is True
    second_client = langfuse_tracing._client
    langfuse_tracing.shutdown_traces()

    assert second_client.shutdown_calls == 1


def test_langfuse_error_summary_does_not_copy_provider_secrets() -> None:
    class ProviderError(RuntimeError):
        code = 429
        status = "RESOURCE_EXHAUSTED"

    secret = "x-goog-api-key: should-never-be-exported"
    summary = safe_exception_summary(ProviderError(secret))

    assert summary == "ProviderError (HTTP 429 RESOURCE_EXHAUSTED)"
    assert secret not in summary
