"""
Crusoe Agent Loop
=================

This module implements the core "agent loop" — the heartbeat of every AI agent.
Understanding this loop is the key to understanding how all modern LLM agents work.

What is an Agent Loop?
----------------------
A "raw" LLM call is stateless: you send messages, you get one response, done.
An *agent* wraps that call in a loop so that the model can:
  1. Decide to call a tool (a Python function)
  2. Receive the result of that tool
  3. Keep thinking/calling more tools
  4. Eventually produce a final answer when it has enough information

ASCII Flow Diagram
------------------

  ┌─────────────────────────────────────────────────────────┐
  │                    AGENT LOOP START                     │
  │                                                         │
  │   messages = [system_prompt] + initial_messages         │
  │                     │                                   │
  │                     ▼                                   │
  │            ┌─────────────────┐                          │
  │            │  Call LLM with  │◄──────────────────┐      │
  │            │ messages + tools│                   │      │
  │            └────────┬────────┘                   │      │
  │                     │                            │      │
  │          ┌──────────▼──────────┐                 │      │
  │          │  What did LLM do?   │                 │      │
  │          └──┬──────────────┬───┘                 │      │
  │             │              │                     │      │
  │    stop=="end_turn"  stop=="tool_use"             │      │
  │             │              │                     │      │
  │             ▼              ▼                     │      │
  │        ┌────────┐   ┌────────────┐               │      │
  │        │ RETURN │   │ Execute    │               │      │
  │        │ final  │   │ tool(args) │               │      │
  │        │ answer │   └─────┬──────┘               │      │
  │        └────────┘         │                     │      │
  │                      ┌────▼──────────────────┐   │      │
  │                      │ Append tool_result to │   │      │
  │                      │ messages list         │───┘      │
  │                      └───────────────────────┘          │
  │                                                         │
  │   (repeat until end_turn or max_iterations exceeded)    │
  └─────────────────────────────────────────────────────────┘

Key Concepts
------------
- messages: The entire conversation history, including tool calls and results.
  This is how the LLM "remembers" what tools it has already called.
- stop_reason: Either "tool_use" (model wants to call a tool) or "end_turn"
  (model is done and has a final answer).
- tool_result: After we execute a tool, we append the result as a special
  message so the model can see what the tool returned.
- max_iterations: A safety valve. Without it, a buggy prompt could loop forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.core.tool import Tool
    from src.llm.providers import LLMProvider


class MaxIterationsExceeded(Exception):
    """Raised when the agent loop hits the iteration cap without ending."""


def run_agent_loop(
    system_prompt: str,
    tools: list["Tool"],
    initial_messages: list[dict],
    provider: "LLMProvider",
    max_iterations: int = 10,
) -> str:
    """
    Run the core agent loop until the LLM signals end_turn or max_iterations is hit.

    Parameters
    ----------
    system_prompt:
        The instruction that shapes the agent's personality and task.
        This becomes the first "system" message every loop iteration sees.
    tools:
        Python-callable tools the LLM can invoke. Each Tool carries its
        JSON schema so the LLM knows what arguments to provide.
    initial_messages:
        Seed messages (usually just the user's request).
    provider:
        The LLM backend (Gemini or Cerebras). The loop is provider-agnostic;
        it only speaks the normalised internal format that providers.py defines.
    max_iterations:
        Hard cap on loop iterations. Prevents infinite loops on bad prompts.

    Returns
    -------
    str
        The final text answer produced by the LLM.

    Raises
    ------
    MaxIterationsExceeded
        If the loop runs more than max_iterations times without stopping.
    ValueError
        If the LLM returns an unrecognised stop reason.
    """

    # ── Step 1: Build the initial message list ──────────────────────────────
    # We keep the full conversation history in `messages`. The system prompt
    # is prepended on every call inside the provider, so we don't include it
    # here — we just hand the system_prompt string to the provider separately.
    messages: list[dict] = list(initial_messages)

    # ── Step 2: Enter the loop ───────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        logger.debug(f"[Agent] Iteration {iteration}/{max_iterations} — sending {len(messages)} messages to LLM")

        # ── Step 3: Call the LLM ─────────────────────────────────────────────
        # The provider returns a normalised dict:
        #   {
        #     "stop_reason": "end_turn" | "tool_use",
        #     "content":     str  (the text response, if end_turn),
        #     "tool_calls":  [{"name": str, "id": str, "args": dict}]  (if tool_use)
        #   }
        response = provider.call(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )

        stop_reason: str = response["stop_reason"]
        logger.debug(f"[Agent] Iteration {iteration} — stop_reason={stop_reason!r}")

        # ── Step 4a: End of reasoning — return the final answer ───────────────
        if stop_reason == "end_turn":
            final_content: str = response.get("content", "")
            logger.info(f"[Agent] Finished after {iteration} iteration(s). Response length: {len(final_content)} chars")
            return final_content

        # ── Step 4b: Model wants to call one or more tools ────────────────────
        elif stop_reason == "tool_use":
            tool_calls: list[dict] = response.get("tool_calls", [])

            if not tool_calls:
                # The model said tool_use but provided no calls — treat as done.
                logger.warning("[Agent] stop_reason=tool_use but no tool_calls found; treating as end_turn")
                return response.get("content", "")

            # Append the assistant's tool-call message to history so the model
            # remembers that it made this call.
            messages.append({
                "role": "assistant",
                "tool_calls": tool_calls,
                # Some providers also return partial text alongside tool calls.
                "content": response.get("content", ""),
            })

            # Execute each tool call and collect results.
            for tc in tool_calls:
                tool_name: str = tc["name"]
                tool_id: str = tc["id"]
                tool_args: dict[str, Any] = tc.get("args", {})

                logger.info(f"[Agent] Iteration {iteration} — calling tool '{tool_name}' with args: {tool_args}")

                # Find the matching Tool object by name.
                matched_tool = _find_tool(tool_name, tools)

                if matched_tool is None:
                    # Unknown tool — tell the model so it can recover.
                    result_content = f"Error: tool '{tool_name}' not found."
                    logger.error(f"[Agent] Unknown tool requested: {tool_name!r}")
                else:
                    # Execute the tool. Wrap in try/except so one bad tool call
                    # doesn't crash the whole agent — the model gets the error
                    # message and can try a different approach.
                    try:
                        result_content = matched_tool.func(**tool_args)
                        # Ensure result is always a string for the message history.
                        if not isinstance(result_content, str):
                            import json
                            result_content = json.dumps(result_content, ensure_ascii=False)
                        logger.debug(f"[Agent] Tool '{tool_name}' returned {len(result_content)} chars")
                    except Exception as exc:
                        result_content = f"Error executing tool '{tool_name}': {exc}"
                        logger.error(f"[Agent] Tool '{tool_name}' raised: {exc}")

                # Append the tool result to history so the model can see it
                # on the next iteration.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result_content,
                })

            # Loop again — the model will now see the tool results and decide
            # whether to call more tools or produce a final answer.
            continue

        else:
            # Defensive: the provider returned an unexpected stop reason.
            raise ValueError(
                f"[Agent] Unrecognised stop_reason from LLM: {stop_reason!r}. "
                "Expected 'end_turn' or 'tool_use'."
            )

    # ── Step 5: Safety valve ─────────────────────────────────────────────────
    # If we exit the for-loop normally, the model never said "end_turn".
    raise MaxIterationsExceeded(
        f"Agent loop exceeded {max_iterations} iterations without finishing. "
        "This usually means the LLM is stuck in a tool-calling cycle. "
        "Try increasing max_iterations or simplifying the prompt/tools."
    )


def _find_tool(name: str, tools: list["Tool"]) -> "Tool | None":
    """Return the Tool with the given name, or None if not found."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None
