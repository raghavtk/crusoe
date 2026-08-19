"""
Topic Decomposition Agent
=========================

Takes a broad research topic and breaks it into 4-6 focused keyword
clusters, each covering a distinct sub-theme of the topic space.

No tools required — this is a pure LLM call with structured JSON output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from loguru import logger
from pydantic import ValidationError

from src.core.state import KeywordCluster, PipelineState
from src.llm.providers import LLMProvider

SYSTEM_PROMPT = """You are a research librarian specialising in academic literature search.
Your task is to decompose a research topic into distinct keyword clusters for database searching.

Each cluster should cover a unique sub-theme. Good clusters are:
- Non-overlapping (each targets a different facet of the topic)
- Specific enough to retrieve relevant papers on Semantic Scholar
- Broad enough to not miss important work

You MUST respond with valid JSON only — no prose or explanations."""

USER_TEMPLATE = """Decompose this research topic into {min_clusters}-{max_clusters} keyword clusters for Semantic Scholar searches.

Topic: {topic}

Return a JSON array where each element has exactly these fields:
- "theme": short label for the sub-theme (e.g. "OAuth2 Authorization Flows")
- "keywords": list of {min_keywords}-{max_keywords} search terms or short phrases to use as queries
- "description": one sentence explaining what papers this cluster will find

Example format:
[
  {{
    "theme": "Token Security Attacks",
    "keywords": ["JWT token hijacking", "bearer token theft", "session token replay"],
    "description": "Papers on how authentication tokens are attacked and exploited."
  }}
]

Respond with the JSON array only."""

RETRY_USER_TEMPLATE = """Your previous response did not satisfy the required schema.

Topic: {topic}

Validation errors:
{errors}

Return a complete replacement JSON array only. It must contain {min_clusters}-{max_clusters}
clusters, each with exactly "theme", "keywords", and "description". Every cluster
must have {min_keywords}-{max_keywords} unique keyword strings. Do not include prose
or Markdown outside the JSON array."""

_JSON_FENCE_RE = re.compile(
    r"^[ \t]*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*$",
    re.IGNORECASE,
)


class TopicDecompositionError(ValueError):
    """Raised when an LLM response cannot satisfy the decomposition contract."""

    def __init__(self, errors: Sequence[str], *, attempts: int) -> None:
        self.errors = list(errors)
        self.attempts = attempts
        details = "; ".join(self.errors)
        super().__init__(
            f"[TopicDecomposition] Could not produce valid keyword clusters after "
            f"{attempts} attempt(s): {details}"
        )


def run(
    state: PipelineState,
    provider: LLMProvider,
    min_clusters: int = 4,
    max_clusters: int = 6,
    min_keywords: int = 3,
    max_keywords: int = 5,
    max_attempts: int = 2,
) -> PipelineState:
    """
    Run the Topic Decomposition agent and update state.keyword_clusters.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state. Reads state.topic.
    provider : LLMProvider
        The configured LLM provider.

    Returns
    -------
    PipelineState
        Updated state with keyword_clusters populated.
    """
    _validate_limits(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_keywords=min_keywords,
        max_keywords=max_keywords,
        max_attempts=max_attempts,
    )
    if not isinstance(state.topic, str):
        raise TopicDecompositionError(["topic must be a string"], attempts=0)
    topic = " ".join(state.topic.split())
    if not topic:
        raise TopicDecompositionError(["topic must not be blank"], attempts=0)

    logger.info(f"[TopicDecomposition] Decomposing topic: {topic!r}")
    message = USER_TEMPLATE.format(
        topic=topic,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_keywords=min_keywords,
        max_keywords=max_keywords,
    )
    last_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        response = provider.call(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
            tools=[],
        )
        raw_content = response.get("content", "")

        try:
            clusters = _parse_clusters(
                raw_content,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                min_keywords=min_keywords,
                max_keywords=max_keywords,
            )
        except TopicDecompositionError as exc:
            last_errors = exc.errors
            logger.warning(
                "[TopicDecomposition] Attempt {}/{} failed validation: {}",
                attempt,
                max_attempts,
                "; ".join(last_errors),
            )
            if attempt == max_attempts:
                raise TopicDecompositionError(last_errors, attempts=attempt) from exc
            message = RETRY_USER_TEMPLATE.format(
                topic=topic,
                errors="\n".join(f"- {error}" for error in last_errors),
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                min_keywords=min_keywords,
                max_keywords=max_keywords,
            )
            continue

        # Mutate only after the entire response has passed validation.
        state.keyword_clusters = clusters
        break

    logger.info(f"[TopicDecomposition] Generated {len(clusters)} keyword clusters:")
    for i, c in enumerate(clusters, 1):
        logger.info(f"  {i}. {c.theme} — keywords: {c.keywords}")

    return state


def _validate_limits(
    *,
    min_clusters: int,
    max_clusters: int,
    min_keywords: int,
    max_keywords: int,
    max_attempts: int,
) -> None:
    """Validate caller-supplied bounds against the shared cluster contract."""
    if min_clusters < 1 or min_clusters > max_clusters:
        raise ValueError("min_clusters must be at least 1 and no greater than max_clusters")
    if min_keywords < 3 or max_keywords > 5 or min_keywords > max_keywords:
        raise ValueError("keyword limits must be within the shared 3-5 keyword contract")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")


def _parse_clusters(
    raw: object,
    *,
    min_clusters: int = 4,
    max_clusters: int = 6,
    min_keywords: int = 3,
    max_keywords: int = 5,
) -> list[KeywordCluster]:
    """
    Extract and validate the JSON cluster array from the LLM response.

    Handles cases where the model wraps the JSON in markdown code fences.

    Parameters
    ----------
    raw : object
        Raw LLM output. A string is required; other values are rejected with
        TopicDecompositionError.

    Raises
    ------
    TopicDecompositionError
        If the response is not exactly one valid JSON array matching the
        shared KeywordCluster contract.
    """
    if not isinstance(raw, str):
        raise TopicDecompositionError(["LLM response content must be a string"], attempts=1)

    cleaned = raw.strip()
    fence = _JSON_FENCE_RE.fullmatch(cleaned)
    if fence is not None:
        cleaned = fence.group("body").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TopicDecompositionError([f"response must be a single valid JSON array: {exc.msg}"], attempts=1) from exc

    if not isinstance(data, list):
        raise TopicDecompositionError(["response root must be a JSON array"], attempts=1)
    if not min_clusters <= len(data) <= max_clusters:
        raise TopicDecompositionError(
            [f"expected {min_clusters}-{max_clusters} clusters, got {len(data)}"], attempts=1
        )

    clusters: list[KeywordCluster] = []
    errors: list[str] = []
    for index, item in enumerate(data, start=1):
        try:
            cluster = KeywordCluster.model_validate(item)
        except ValidationError as exc:
            errors.extend(_format_pydantic_errors(index, exc))
            continue
        if not min_keywords <= len(cluster.keywords) <= max_keywords:
            errors.append(
                f"cluster {index} ({cluster.theme!r}) must have "
                f"{min_keywords}-{max_keywords} keywords, got {len(cluster.keywords)}"
            )
        clusters.append(cluster)

    if errors:
        raise TopicDecompositionError(errors, attempts=1)

    _validate_global_uniqueness(clusters)
    return clusters


def _format_pydantic_errors(index: int, exc: ValidationError) -> list[str]:
    """Convert Pydantic errors to concise feedback an LLM can repair."""
    formatted: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        formatted.append(f"cluster {index}.{location}: {error['msg']}")
    return formatted


def _validate_global_uniqueness(clusters: list[KeywordCluster]) -> None:
    """Require themes and keywords to be unique across the full response."""
    errors: list[str] = []
    themes: dict[str, str] = {}
    keywords: dict[str, tuple[str, str]] = {}

    for cluster in clusters:
        theme_key = cluster.theme.casefold()
        if theme_key in themes:
            errors.append(
                f"theme {cluster.theme!r} duplicates {themes[theme_key]!r} after normalisation"
            )
        else:
            themes[theme_key] = cluster.theme

        for keyword in cluster.keywords:
            keyword_key = keyword.casefold()
            if keyword_key in keywords:
                previous_keyword, previous_theme = keywords[keyword_key]
                errors.append(
                    f"keyword {keyword!r} in theme {cluster.theme!r} duplicates "
                    f"{previous_keyword!r} in theme {previous_theme!r}"
                )
            else:
                keywords[keyword_key] = (keyword, cluster.theme)

    if errors:
        raise TopicDecompositionError(errors, attempts=1)
