from __future__ import annotations

import pytest

from src.llm.providers import (
    BudgetedProvider,
    CerebrasProvider,
    GeminiProvider,
    LLMProvider,
    LLMRequestBudgetExceeded,
    apply_request_budget,
)


class StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("untrusted provider details")


class SequenceProvider(LLMProvider):
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def call(self, system_prompt: str, messages: list[dict], tools: list) -> dict:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def test_429_is_never_retried() -> None:
    delegate = SequenceProvider([StatusError(429), {"content": "must not run"}])
    provider = BudgetedProvider(delegate, max_requests=5, transient_503_retries=1)

    with pytest.raises(StatusError):
        provider.call("system", [], [])

    assert delegate.calls == 1
    assert provider.budget.used == 1


def test_503_gets_at_most_one_configured_retry(monkeypatch) -> None:
    delegate = SequenceProvider([StatusError(503), {"content": "ok"}])
    provider = BudgetedProvider(
        delegate, max_requests=5, transient_503_retries=1, retry_delay_seconds=1
    )
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm.providers.time.sleep", sleeps.append)

    assert provider.call("system", [], []) == {"content": "ok"}
    assert delegate.calls == 2
    assert provider.budget.used == 2
    assert sleeps == [1.0]


def test_503_is_not_retried_by_default() -> None:
    delegate = SequenceProvider([StatusError(503)])
    provider = BudgetedProvider(delegate, max_requests=5)

    with pytest.raises(StatusError):
        provider.call("system", [], [])

    assert delegate.calls == 1


def test_budget_stops_before_network_request() -> None:
    delegate = SequenceProvider([{"content": "one"}, {"content": "must not run"}])
    provider = BudgetedProvider(delegate, max_requests=1)

    assert provider.call("system", [], []) == {"content": "one"}
    with pytest.raises(LLMRequestBudgetExceeded, match="no request was sent"):
        provider.call("system", [], [])

    assert delegate.calls == 1
    assert provider.budget.used == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_requests": 0},
        {"max_requests": True},
        {"max_requests": 5, "transient_503_retries": 2},
        {"max_requests": 5, "transient_503_retries": True},
        {"max_requests": 5, "retry_delay_seconds": -1},
    ],
)
def test_invalid_budget_configuration_fails(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BudgetedProvider(SequenceProvider([]), **kwargs)


def test_apply_request_budget_is_optional_and_does_not_double_wrap() -> None:
    delegate = SequenceProvider([])

    assert apply_request_budget(delegate, {}) is delegate
    wrapped = apply_request_budget(delegate, {"max_requests_per_run": 3})
    assert isinstance(wrapped, BudgetedProvider)
    assert apply_request_budget(wrapped, {"max_requests_per_run": 3}) is wrapped


def test_gemini_constructor_disables_sdk_retries(monkeypatch) -> None:
    from google import genai

    captured: dict = {}
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: captured.update(kwargs) or object())

    GeminiProvider()

    retry_options = captured["http_options"].retry_options
    assert retry_options.attempts == 1
    assert retry_options.http_status_codes == [503]


def test_cerebras_constructor_disables_sdk_retries(monkeypatch) -> None:
    from cerebras.cloud import sdk

    captured: dict = {}
    monkeypatch.setenv("CEREBRAS_API_KEY", "fake-key")
    monkeypatch.setattr(sdk, "Cerebras", lambda **kwargs: captured.update(kwargs) or object())

    CerebrasProvider()

    assert captured["max_retries"] == 0
