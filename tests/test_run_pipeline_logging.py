from __future__ import annotations

from io import StringIO

from loguru import logger
from rich.console import Console

from scripts import run_pipeline


class FakeProviderError(RuntimeError):
    status_code = 429
    status = "RESOURCE_EXHAUSTED"


def test_pipeline_failure_does_not_expose_exception_secrets(tmp_path, monkeypatch):
    fake_key = "AIza-fake-secret-that-must-never-be-logged"
    exception = FakeProviderError(
        f"request failed: headers={{'x-goog-api-key': '{fake_key}'}}"
    )
    terminal = StringIO()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(
        run_pipeline,
        "console",
        Console(file=terminal, force_terminal=False, color_system=None),
    )

    try:
        run_pipeline.setup_logging("INFO")
        run_pipeline._report_pipeline_failure(exception)
        logger.complete()

        log_text = (tmp_path / "data" / "crusoe.log").read_text()
        combined_output = terminal.getvalue() + log_text
        assert fake_key not in combined_output
        assert "x-goog-api-key" not in combined_output
        assert "FakeProviderError (HTTP 429 RESOURCE_EXHAUSTED)" in combined_output
    finally:
        logger.remove()
        logger.add(lambda _: None)


def test_logging_sinks_disable_diagnostic_tracebacks(monkeypatch):
    added_sinks = []

    monkeypatch.setattr(run_pipeline.logger, "remove", lambda *args: None)
    monkeypatch.setattr(
        run_pipeline.logger,
        "add",
        lambda sink, **kwargs: added_sinks.append((sink, kwargs)),
    )

    run_pipeline.setup_logging("INFO")

    assert len(added_sinks) == 2
    assert all(options["backtrace"] is False for _, options in added_sinks)
    assert all(options["diagnose"] is False for _, options in added_sinks)


def test_exception_summary_does_not_copy_arbitrary_status_text():
    class UnsafeStatusError(RuntimeError):
        code = 503
        status = "request used x-goog-api-key: secret"

    summary = run_pipeline.safe_exception_summary(UnsafeStatusError("another secret"))

    assert summary == "UnsafeStatusError (HTTP 503)"
    assert "secret" not in summary
