# Crusoe — Architecture Reference

## Overview

Crusoe is a sequential multi-agent pipeline. Each agent is a Python module
with a `run(state, provider)` function. State flows forward; no agent
communicates with another directly.

```
scripts/run_pipeline.py
        │
        ▼
src/agents/orchestrator.py          ← coordinates all agents + Google Sheets
        │
        ├── src/agents/topic_decomposition.py   [LLM: structured JSON output]
        ├── src/agents/discovery.py             [Semantic Scholar API]
        ├── src/agents/paper_curator.py          [LLM: validated batched assessments]
        ├── src/agents/synthesis.py             [LLM: one/two-pass JSON output]
        │
        └── Google Sheets (google-api-python-client)
```

## Layer-by-Layer

### `src/core/`

| File | Purpose |
|------|---------|
| `agent.py` | The reusable agent loop. Accepts messages + tools, loops until `end_turn`. |
| `tool.py` | `Tool` dataclass wrapping a Python callable with JSON Schema. |
| `state.py` | `PipelineState` dataclass and the strict `KeywordCluster` hand-off model. Checkpoints remain JSON-serialisable. |

### `src/llm/`

| File | Purpose |
|------|---------|
| `providers.py` | `GeminiProvider` and `CerebrasProvider`. Both implement `LLMProvider.call()`. |

The internal normalised message format:
```python
# User turn
{"role": "user", "content": "..."}

# Assistant turn (with optional tool calls)
{"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "args"}]}

# Tool result
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

### `src/tools/`

| File | Purpose |
|------|---------|
| `semantic_scholar.py` | `search_papers()` and `get_paper_details()` with tenacity retry logic. |

Also exports `SEARCH_PAPERS_TOOL` and `GET_PAPER_DETAILS_TOOL` as `Tool` instances
ready to pass to the agent loop.

### `src/agents/`

| Agent | LLM calls | Tools used | Input → Output |
|-------|-----------|------------|----------------|
| `topic_decomposition` | 1-2 | None | topic → validated keyword_clusters |
| `discovery` | 0 (direct API) | search_papers | keyword_clusters → papers_raw |
| `paper_curator` | N/batch_size (plus one repair attempt when invalid) | None | papers_raw → papers_curated |
| `synthesis` | 1 or 3 (two-pass) | None | papers_curated → synthesis |
| `orchestrator` | — | All above | topic → Google Sheet |

## Data Flow

```
PipelineState.topic
        │
        ▼
PipelineState.keyword_clusters    [4-6 KeywordCluster values: theme, 3-5 keywords, description]
        │
        ▼
PipelineState.papers_raw          [≤80 dicts: paperId, title, abstract, ...]
        │
        ▼
PipelineState.papers_curated      [same + validated assessment and reading priority]
        │
        ▼
PipelineState.synthesis           [key_themes, research_gaps, reading_order, ...]
        │
        ▼
PipelineState.sheet_url           [Google Sheets URL]
```

## Checkpoint / Resume

After each stage, the orchestrator calls `state.save(checkpoint_path)`.
On `--resume`, the orchestrator loads the checkpoint and skips any stage
whose output fields are already populated.

```python
# Save
state.save("data/session_checkpoint.json")

# Load
state = PipelineState.load("data/session_checkpoint.json")
```

`KeywordCluster` values are stored as ordinary JSON objects in a checkpoint,
so existing checkpoints with `theme`, `keywords`, and `description` dictionaries
continue to load. On load, those dictionaries are validated and converted into
the typed contract used by Discovery.

## Topic Decomposition Validation

Topic Decomposition rejects blank topics and accepts exactly one JSON array
(optionally in a single `json` Markdown fence). The array must contain 4--6
clusters, each with exactly `theme`, `keywords`, and `description`; text is
whitespace-normalised, keyword lists contain 3--5 unique strings, and themes
and keywords cannot repeat across clusters. If the first LLM response is
invalid, the agent sends one corrective retry with the validation errors. A
second invalid response raises `TopicDecompositionError` without changing the
previous state.

## Rate Limits and Cost Controls

| Concern | Mitigation |
|---------|-----------|
| Semantic Scholar 429 | `tenacity` retry with 5s sleep |
| LLM token cost (paper curator) | Batch size 8, abstract truncated to 1200 chars |
| LLM token cost (synthesis) | Two-pass for >50 papers |
| Infinite agent loops | `max_iterations` hard cap, raises `MaxIterationsExceeded` |
| Paper explosion | `max_total_papers` cap (default 80) |

## Configuration Reference

See `config.yaml` for all tunable parameters. Key settings:

```yaml
llm.provider          # "gemini" | "cerebras"
semantic_scholar.max_total_papers   # default 80
paper_curator.batch_size             # default 8
pipeline.max_agent_iterations       # default 10
google_sheets.sheet_id              # blank = auto-create
```

## Adding a New Tool

1. Define a Python function in `src/tools/`
2. Wrap it in a `Tool` instance with a JSON Schema for `parameters`
3. Pass the `Tool` in the `tools=[]` list when calling `run_agent_loop()`

```python
from src.core.tool import Tool

my_tool = Tool(
    name="my_tool",
    description="Does something useful.",
    parameters={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "The input value"},
        },
        "required": ["input"],
    },
    func=lambda input: f"result: {input}",
)
```

## Adding a New Agent

1. Create `src/agents/my_agent.py`
2. Implement `def run(state: PipelineState, provider: LLMProvider) -> PipelineState:`
3. Add the stage to `orchestrator.py` with checkpoint save
4. Add a `has_X` property to `PipelineState` for resume logic
