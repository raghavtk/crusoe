"""
Langfuse tracing for Crusoe
===========================

Centralises Langfuse client setup, span helpers, and flush behaviour.

Environment variables (required when enabled):
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_HOST          (optional, defaults to https://us.cloud.langfuse.com)

Flush strategy
--------------
Short-lived CLI scripts buffer traces in the background. We:
  1. Set flush_at=1 and flush_interval=1 by default (send after each event).
  2. Call flush_traces() after each pipeline stage and each LLM call.
  3. Call shutdown_traces() when the CLI exits.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Generator

from loguru import logger

_client: Any = None
_enabled: bool = False
_shutdown_done: bool = False
_config: dict[str, Any] = {}


def init_langfuse(config: dict[str, Any] | None = None) -> bool:
    """
    Initialise the Langfuse client from config.yaml and environment variables.

    Returns True when tracing is active, False when disabled or keys are missing.
    """
    global _client, _enabled, _config, _shutdown_done

    cfg = (config or {}).get("langfuse", {})
    _config = cfg
    _shutdown_done = False

    if not cfg.get("enabled", True):
        logger.info("[Langfuse] Tracing disabled via config (langfuse.enabled=false).")
        _enabled = False
        return False

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        logger.warning(
            "[Langfuse] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled."
        )
        _enabled = False
        return False

    from langfuse import Langfuse

    host = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com").strip()
    _client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        flush_at=cfg.get("flush_at", 1),
        flush_interval=cfg.get("flush_interval", 1),
    )
    _enabled = True
    logger.info(f"[Langfuse] Tracing enabled (host={host}, flush_at={cfg.get('flush_at', 1)}).")
    return True


def is_tracing_enabled() -> bool:
    """Return whether Langfuse tracing is active."""
    return _enabled and _client is not None


def _maybe_flush_after_llm() -> None:
    """Flush buffered events after an LLM call when configured."""
    if _config.get("flush_after_llm_call", True):
        flush_traces()


def flush_traces() -> None:
    """Send all buffered trace events to Langfuse."""
    if not is_tracing_enabled():
        return
    try:
        _client.flush()
        logger.debug("[Langfuse] Flushed pending traces.")
    except Exception as exc:
        logger.warning(f"[Langfuse] Flush failed: {exc}")


def shutdown_traces() -> None:
    """Flush and shut down the Langfuse client (call on CLI exit)."""
    global _shutdown_done
    if not is_tracing_enabled() or _shutdown_done:
        return
    try:
        _client.shutdown()
        _shutdown_done = True
        logger.debug("[Langfuse] Client shut down.")
    except Exception as exc:
        logger.warning(f"[Langfuse] Shutdown failed: {exc}")


@contextmanager
def trace_pipeline(
    *,
    topic: str,
    provider: str,
    resume: bool,
) -> Generator[Any, None, None]:
    """Top-level trace for a full Crusoe pipeline run."""
    if not is_tracing_enabled():
        yield None
        return

    from langfuse import propagate_attributes

    with _client.start_as_current_observation(
        as_type="span",
        name="crusoe-pipeline",
        input={"topic": topic, "provider": provider, "resume": resume},
        metadata={"app": "crusoe"},
    ) as span:
        with propagate_attributes(
            session_id=topic[:128] or "crusoe-session",
            tags=["crusoe", provider],
        ):
            yield span
            if span is not None:
                span.update(metadata={"resume": resume, "provider": provider})


@contextmanager
def trace_agent(
    name: str,
    *,
    input_data: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Span for one pipeline agent stage (topic, discovery, enrichment, etc.)."""
    if not is_tracing_enabled():
        yield None
        return

    with _client.start_as_current_observation(
        as_type="span",
        name=name,
        input=input_data or {},
        metadata={"agent": name},
    ) as span:
        yield span


@contextmanager
def trace_span(
    name: str,
    *,
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Generic child span (tool calls, API requests, agent-loop iterations)."""
    if not is_tracing_enabled():
        yield None
        return

    with _client.start_as_current_observation(
        as_type="span",
        name=name,
        input=input_data,
        metadata=metadata or {},
    ) as span:
        yield span


def trace_llm_call(
    *,
    name: str,
    model: str,
    provider: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[Any],
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute an LLM call inside a Langfuse generation observation.

    Parameters
    ----------
    fn : Callable
        Zero-arg callable that performs the actual provider API call.
    """
    if not is_tracing_enabled():
        return fn()

    tool_names = [getattr(t, "name", str(t)) for t in tools]
    llm_input = {
        "system_prompt": system_prompt[:500],
        "message_count": len(messages),
        "messages": _truncate_messages(messages),
        "tools": tool_names,
    }

    with _client.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=llm_input,
        metadata={"provider": provider, "tool_count": len(tools)},
    ) as generation:
        try:
            response = fn()
            generation.update(
                output={
                    "stop_reason": response.get("stop_reason"),
                    "content_preview": (response.get("content") or "")[:1000],
                    "tool_calls": response.get("tool_calls", []),
                },
            )
            return response
        except Exception as exc:
            generation.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            _maybe_flush_after_llm()


def _truncate_messages(messages: list[dict[str, Any]], max_chars: int = 2000) -> list[dict[str, Any]]:
    """Truncate message content for Langfuse input payloads."""
    truncated: list[dict[str, Any]] = []
    for msg in messages:
        entry = {"role": msg.get("role")}
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > max_chars:
            entry["content"] = content[:max_chars] + "…"
        else:
            entry["content"] = content
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        if msg.get("name"):
            entry["name"] = msg["name"]
        truncated.append(entry)
    return truncated
