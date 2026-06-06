"""
Tool Schema and Execution Wrapper
==================================

Defines the Tool dataclass that wraps a Python callable with enough
metadata for an LLM to understand what the function does and what
arguments it expects.

Two schema formats are supported:
  - Gemini  (function_declarations format)
  - OpenAI / Cerebras  (tools list with type: "function")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """
    A named, typed wrapper around a Python callable for use in agent tool loops.

    Attributes
    ----------
    name : str
        Unique identifier the LLM uses to call this tool.
    description : str
        Plain-English explanation of what the tool does. This is the most
        important field — the LLM decides *when* to call a tool based on it.
    parameters : dict
        JSON Schema describing the tool's arguments (type "object" with
        "properties" and "required" keys).
    func : Callable
        The actual Python function to execute when the tool is called.
    """

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    # Metadata fields (optional, not sent to the LLM)
    tags: list[str] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the underlying function with keyword arguments."""
        return self.func(**kwargs)

    def to_gemini_schema(self) -> dict:
        """
        Return a Gemini-compatible function declaration dict.

        Gemini expects:
          {
            "name": "...",
            "description": "...",
            "parameters": { <JSON Schema> }
          }
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_schema(self) -> dict:
        """
        Return an OpenAI-compatible tool dict (also used by Cerebras).

        OpenAI expects:
          {
            "type": "function",
            "function": {
              "name": "...",
              "description": "...",
              "parameters": { <JSON Schema> }
            }
          }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def make_tool(
    name: str,
    description: str,
    parameters: dict,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    Decorator factory that turns a plain Python function into a Tool.

    Usage
    -----
    @make_tool(
        name="search_papers",
        description="Search Semantic Scholar for academic papers.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        },
    )
    def search_papers(query: str, limit: int = 20) -> list[dict]:
        ...
    """
    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            tags=tags or [],
        )
    return decorator
