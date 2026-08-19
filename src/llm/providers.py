"""
LLM Provider Adapters
=====================

Provides a unified interface to multiple LLM backends.
The agent loop speaks the internal normalised format; each provider
class translates to/from the backend's native API format.

Internal normalised format
--------------------------
Input messages (list of dicts):
  {"role": "user",      "content": "..."}
  {"role": "assistant", "content": "...", "tool_calls": [...]}
  {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "..."}

Output response (dict):
  {
    "stop_reason": "end_turn" | "tool_use",
    "content":    str,        # text portion of the response
    "tool_calls": [           # only present when stop_reason == "tool_use"
      {"id": str, "name": str, "args": dict}
    ]
  }
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.observability.langfuse_tracing import trace_llm_call

if TYPE_CHECKING:
    from src.core.tool import Tool


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base for all LLM provider adapters."""

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list["Tool"],
    ) -> dict:
        """
        Send a request to the LLM and return a normalised response dict.

        Parameters
        ----------
        system_prompt : str
            High-level instructions for the model.
        messages : list[dict]
            Conversation history in the internal normalised format.
        tools : list[Tool]
            Available tools the model may call.

        Returns
        -------
        dict
            Normalised response: {"stop_reason", "content", "tool_calls"}.
        """


# ---------------------------------------------------------------------------
# Gemini Provider
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    Google Gemini adapter (google-generativeai SDK).

    Translates the internal message format ↔ Gemini's Content/Part format.
    Tool schemas are converted to Gemini function_declarations.
    """

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.5) -> None:
        import google.generativeai as genai  # type: ignore

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model
        self._temperature = temperature
        logger.info(f"GeminiProvider initialised: model={model}, temperature={temperature}")

    def call(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list["Tool"],
    ) -> dict:
        """Call Gemini and return a normalised response."""
        from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool  # type: ignore

        # Convert our Tool objects → Gemini function_declarations
        gemini_tools = []
        if tools:
            declarations = [
                FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
                for t in tools
            ]
            gemini_tools = [GeminiTool(function_declarations=declarations)]

        # Convert internal messages → Gemini Content objects
        gemini_history = self._to_gemini_messages(messages)

        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
            tools=gemini_tools or None,
            generation_config=self._genai.GenerationConfig(temperature=self._temperature),
        )

        # Gemini's chat API: history is everything except the last message.
        # The last message is sent via chat.send_message().
        if not gemini_history:
            logger.warning("GeminiProvider: no messages to send.")
            return {"stop_reason": "end_turn", "content": "", "tool_calls": []}

        history_part = gemini_history[:-1]
        last_message = gemini_history[-1]

        def _invoke() -> dict:
            chat = model.start_chat(history=history_part)
            response = chat.send_message(last_message)
            return self._normalise_response(response)

        return trace_llm_call(
            name="gemini-completion",
            model=self._model_name,
            provider="gemini",
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            fn=_invoke,
        )

    def _to_gemini_messages(self, messages: list[dict]) -> list[Any]:
        """Convert internal messages to Gemini Content objects."""
        from google.generativeai.types import ContentDict  # type: ignore

        gemini_msgs: list[Any] = []
        for msg in messages:
            role = msg["role"]

            if role == "user":
                gemini_msgs.append(ContentDict(role="user", parts=[msg["content"]]))

            elif role == "assistant":
                parts: list[Any] = []
                if msg.get("content"):
                    parts.append(msg["content"])
                for tc in msg.get("tool_calls", []):
                    from google.generativeai.types import protos  # type: ignore
                    parts.append(
                        protos.Part(
                            function_call=protos.FunctionCall(
                                name=tc["name"],
                                args=tc.get("args", {}),
                            )
                        )
                    )
                gemini_msgs.append(ContentDict(role="model", parts=parts))

            elif role == "tool":
                from google.generativeai.types import protos  # type: ignore
                content_val = msg["content"]
                # Gemini expects tool results as a dict, not a raw string
                try:
                    result_dict = json.loads(content_val) if isinstance(content_val, str) else content_val
                    if not isinstance(result_dict, dict):
                        result_dict = {"result": content_val}
                except (json.JSONDecodeError, TypeError):
                    result_dict = {"result": content_val}

                gemini_msgs.append(
                    ContentDict(
                        role="user",
                        parts=[
                            protos.Part(
                                function_response=protos.FunctionResponse(
                                    name=msg["name"],
                                    response=result_dict,
                                )
                            )
                        ],
                    )
                )

        return gemini_msgs

    def _normalise_response(self, response: Any) -> dict:
        """Translate a Gemini response object into the internal normalised format."""
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason

        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            elif hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                tool_calls.append({
                    "id": f"gemini_{fc.name}_{len(tool_calls)}",
                    "name": fc.name,
                    "args": dict(fc.args),
                })

        stop_reason = "tool_use" if tool_calls else "end_turn"

        # Gemini STOP = normal end, SAFETY = content filtered (treat as end_turn)
        if finish_reason.name not in ("STOP", "MAX_TOKENS") and not tool_calls:
            logger.warning(f"Gemini finish_reason={finish_reason.name}; treating as end_turn")

        return {
            "stop_reason": stop_reason,
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
        }


# ---------------------------------------------------------------------------
# Cerebras Provider
# ---------------------------------------------------------------------------

class CerebrasProvider(LLMProvider):
    """
    Cerebras Cloud SDK adapter.

    Cerebras uses an OpenAI-compatible chat completions interface, so
    tool schemas follow the OpenAI function-calling format.
    """

    def __init__(self, model: str = "llama-3.3-70b", temperature: float = 0.5) -> None:
        from cerebras.cloud.sdk import Cerebras  # type: ignore

        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise EnvironmentError("CEREBRAS_API_KEY environment variable is not set.")

        self._client = Cerebras(api_key=api_key)
        self._model = model
        self._temperature = temperature
        logger.info(f"CerebrasProvider initialised: model={model}, temperature={temperature}")

    def call(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list["Tool"],
    ) -> dict:
        """Call Cerebras and return a normalised response."""
        oai_messages = self._to_openai_messages(system_prompt, messages)
        oai_tools = [t.to_openai_schema() for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "temperature": self._temperature,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        def _invoke() -> dict:
            response = self._client.chat.completions.create(**kwargs)
            return self._normalise_response(response)

        return trace_llm_call(
            name="cerebras-completion",
            model=self._model,
            provider="cerebras",
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            fn=_invoke,
        )

    def _to_openai_messages(self, system_prompt: str, messages: list[dict]) -> list[dict]:
        """Prepend system prompt and convert internal messages to OpenAI format."""
        oai: list[dict] = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            role = msg["role"]

            if role == "user":
                oai.append({"role": "user", "content": msg["content"]})

            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
                if msg.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ]
                oai.append(entry)

            elif role == "tool":
                oai.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                })

        return oai

    def _normalise_response(self, response: Any) -> dict:
        """Translate a Cerebras/OpenAI response into the internal normalised format."""
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        content: str = message.content or ""
        tool_calls: list[dict] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })

        stop_reason = "tool_use" if tool_calls else "end_turn"
        logger.debug(f"CerebrasProvider: finish_reason={finish_reason}, stop_reason={stop_reason}")

        return {
            "stop_reason": stop_reason,
            "content": content,
            "tool_calls": tool_calls,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(config: dict) -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    Parameters
    ----------
    config : dict
        The 'llm' section of config.yaml.

    Returns
    -------
    LLMProvider
    """
    provider_name: str = config.get("provider", "gemini").lower()

    if provider_name == "gemini":
        cfg = config.get("gemini", {})
        return GeminiProvider(
            model=cfg.get("model", "gemini-2.0-flash"),
            temperature=cfg.get("temperature", 0.5),
        )
    elif provider_name == "cerebras":
        cfg = config.get("cerebras", {})
        return CerebrasProvider(
            model=cfg.get("model", "llama-3.3-70b"),
            temperature=cfg.get("temperature", 0.5),
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            "Set llm.provider to 'gemini' or 'cerebras' in config.yaml."
        )
