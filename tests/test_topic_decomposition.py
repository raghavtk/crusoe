"""Offline contract tests for the Topic Decomposition hand-off."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.agents import discovery, topic_decomposition
from src.agents.topic_decomposition import TopicDecompositionError
from src.core.state import KeywordCluster, PipelineState


def _clusters() -> list[dict]:
    """Return a complete, globally unique response fixture."""
    return [
        {
            "theme": "Threat Models",
            "keywords": ["token theft", "session replay", "credential phishing"],
            "description": "Papers that describe attacks against credentials.",
        },
        {
            "theme": "Protocol Design",
            "keywords": ["OAuth authorization", "OIDC protocol", "PKCE extension"],
            "description": "Papers about modern authorization protocol design.",
        },
        {
            "theme": "Implementation Defenses",
            "keywords": ["token rotation", "secure cookies", "browser storage"],
            "description": "Papers about implementing authentication safely.",
        },
        {
            "theme": "Formal Analysis",
            "keywords": ["protocol verification", "symbolic analysis", "security proof"],
            "description": "Papers that analyse authentication formally.",
        },
    ]


class FakeProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def call(self, system_prompt: str, messages: list[dict], tools: list[object]) -> dict:
        self.calls.append({"system_prompt": system_prompt, "messages": messages, "tools": tools})
        return {"content": self.responses.pop(0)}


def test_keyword_cluster_is_strict_and_normalises_whitespace() -> None:
    cluster = KeywordCluster.model_validate(
        {
            "theme": "  Threat   Models  ",
            "keywords": [" token   theft ", "session replay", "credential phishing"],
            "description": "  Papers   about attacks.  ",
        }
    )

    assert cluster.theme == "Threat Models"
    assert cluster.keywords == ["token theft", "session replay", "credential phishing"]
    assert cluster.description == "Papers about attacks."


@pytest.mark.parametrize(
    "cluster",
    [
        {"theme": "Theme", "keywords": ["one", "two", "three"], "description": "Desc", "extra": "no"},
        {"theme": "Theme", "keywords": "one", "description": "Desc"},
        {"theme": "Theme", "keywords": ["one", "two", 3], "description": "Desc"},
        {"theme": " ", "keywords": ["one", "two", "three"], "description": "Desc"},
        {"theme": "Theme", "keywords": ["one", "ONE", "three"], "description": "Desc"},
    ],
)
def test_keyword_cluster_rejects_invalid_shape(cluster: dict) -> None:
    with pytest.raises(ValidationError):
        KeywordCluster.model_validate(cluster)


def test_run_accepts_single_fenced_json_array_and_normalises_values() -> None:
    response = _clusters()
    response[0]["theme"] = "  Threat  Models "
    response[0]["keywords"][0] = " token   theft "
    provider = FakeProvider([f"```json\n{json.dumps(response)}\n```"])
    state = PipelineState(topic="  authentication   tokens ")

    result = topic_decomposition.run(state, provider)  # type: ignore[arg-type]

    assert result is state
    assert len(state.keyword_clusters) == 4
    assert state.keyword_clusters[0].theme == "Threat Models"
    assert state.keyword_clusters[0].keywords[0] == "token theft"
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "Here is the JSON: " + json.dumps(_clusters()),
        json.dumps(_clusters()) + " Thanks!",
        "```json\n" + json.dumps(_clusters()) + "\n```\nMore prose",
        "{\"theme\": \"not an array\"}",
        "not JSON",
    ],
)
def test_parse_rejects_prose_and_non_array_responses(raw: str) -> None:
    with pytest.raises(TopicDecompositionError):
        topic_decomposition._parse_clusters(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda clusters: clusters.pop(),
        lambda clusters: clusters[0].update(extra="nope"),
        lambda clusters: clusters[0].update(description=""),
        lambda clusters: clusters[0].update(keywords=["one", "two", 3]),
        lambda clusters: clusters[1].update(theme="threat models"),
        lambda clusters: clusters[1].update(keywords=["TOKEN THEFT", "different", "another"]),
    ],
)
def test_parse_rejects_invalid_counts_fields_types_and_duplicates(mutate) -> None:
    response = _clusters()
    mutate(response)

    with pytest.raises(TopicDecompositionError):
        topic_decomposition._parse_clusters(json.dumps(response))


def test_invalid_first_response_is_repaired_once() -> None:
    invalid = _clusters()
    invalid[0]["keywords"] = ["only", "two"]
    provider = FakeProvider([json.dumps(invalid), json.dumps(_clusters())])
    state = PipelineState(topic="authentication tokens")

    topic_decomposition.run(state, provider)  # type: ignore[arg-type]

    assert len(provider.calls) == 2
    retry_prompt = provider.calls[1]["messages"][0]["content"]
    assert "Validation errors:" in retry_prompt
    assert "expected" in retry_prompt or "at least" in retry_prompt
    assert len(state.keyword_clusters) == 4


def test_custom_limits_are_reflected_in_initial_prompt() -> None:
    response = _clusters()
    for index, cluster in enumerate(response):
        cluster["keywords"].append(f"additional query {index}")
    provider = FakeProvider([json.dumps(response)])

    topic_decomposition.run(
        PipelineState(topic="authentication tokens"),
        provider,  # type: ignore[arg-type]
        min_clusters=3,
        max_clusters=4,
        min_keywords=4,
        max_keywords=5,
    )

    prompt = provider.calls[0]["messages"][0]["content"]
    assert "3-4 keyword clusters" in prompt
    assert "list of 4-5 search terms" in prompt


def test_failure_after_two_attempts_is_atomic() -> None:
    existing = KeywordCluster(
        theme="Existing",
        keywords=["existing one", "existing two", "existing three"],
        description="Existing cluster.",
    )
    provider = FakeProvider(["[]", "[]"])
    state = PipelineState(topic="authentication tokens", keyword_clusters=[existing])

    with pytest.raises(TopicDecompositionError) as exc_info:
        topic_decomposition.run(state, provider)  # type: ignore[arg-type]

    assert exc_info.value.attempts == 2
    assert len(provider.calls) == 2
    assert state.keyword_clusters == [existing]


def test_blank_topic_fails_without_provider_call() -> None:
    provider = FakeProvider([json.dumps(_clusters())])

    with pytest.raises(TopicDecompositionError) as exc_info:
        topic_decomposition.run(PipelineState(topic=" \t\n "), provider)  # type: ignore[arg-type]

    assert exc_info.value.attempts == 0
    assert provider.calls == []


def test_non_string_topic_fails_without_provider_call() -> None:
    provider = FakeProvider([json.dumps(_clusters())])
    state = PipelineState(topic=123)  # type: ignore[arg-type]

    with pytest.raises(TopicDecompositionError):
        topic_decomposition.run(state, provider)  # type: ignore[arg-type]

    assert provider.calls == []


def test_checkpoint_round_trip_and_legacy_dictionary_loading(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    state = PipelineState(topic="authentication tokens", keyword_clusters=_clusters())
    state.save(path)

    wire_data = json.loads(path.read_text())
    assert isinstance(wire_data["keyword_clusters"][0], dict)
    loaded = PipelineState.load(path)
    assert isinstance(loaded.keyword_clusters[0], KeywordCluster)
    assert loaded.to_dict() == state.to_dict()

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps({"topic": "legacy", "keyword_clusters": _clusters()}))
    legacy = PipelineState.load(legacy_path)
    assert isinstance(legacy.keyword_clusters[0], KeywordCluster)


def test_discovery_uses_typed_cluster_keywords_unchanged(monkeypatch) -> None:
    state = PipelineState(topic="topic", keyword_clusters=[_clusters()[0]])
    submitted: list[str] = []

    def fake_search_papers(*, query: str, limit: int) -> list[dict]:
        submitted.append(query)
        return [{"paperId": query, "title": query}]

    monkeypatch.setattr(discovery, "search_papers", fake_search_papers)

    discovery.run(state, provider=None, results_per_query=1, max_total_papers=10)  # type: ignore[arg-type]

    assert submitted == state.keyword_clusters[0].keywords
    assert {paper["_source_cluster"] for paper in state.papers_raw} == {"Threat Models"}
