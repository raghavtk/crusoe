# Crusoe — Architecture Reference

## Overview

Crusoe is a sequential multi-agent pipeline. Each agent is a Python module
with a `run(state, provider)` function (the batch-capable agents also accept
their configured `batch_size`). State flows forward; no agent
communicates with another directly.

```
scripts/run_pipeline.py
        │
        ▼
src/agents/orchestrator.py          ← coordinates all agents + spreadsheet export
        │
        ├── src/agents/topic_decomposition.py   [LLM: structured JSON output]
        ├── src/agents/discovery.py             [Semantic Scholar API]
        ├── src/agents/paper_curator.py          [LLM: validated batched assessments]
        ├── src/agents/synthesis.py             [LLM: validated adaptive map-reduce output]
        │
        └── src/output/ (normalized workbook model → XLSX or Google Sheets)
```

## Layer-by-Layer

### `src/core/`

| File | Purpose |
|------|---------|
| `agent.py` | The reusable agent loop. Accepts messages + tools, loops until `end_turn`. |
| `tool.py` | `Tool` dataclass wrapping a Python callable with JSON Schema. |
| `state.py` | `PipelineState` dataclass, output-result checkpoint records, and the strict `KeywordCluster` hand-off model. Checkpoints remain JSON-serialisable. |

### `src/output/`

The output package projects completed `PipelineState` into a backend-neutral workbook model.
The XLSX and Google Sheets writers consume that same model, producing the same Summary, Papers,
Themes, Gaps, Future Work, Methods, Disagreements, and Reading Order tabs.

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
| `synthesis` | 1, or N map calls + 1 reducer | None | papers_curated → evidence-grounded synthesis |
| `orchestrator` | — | All above | topic → spreadsheet export |

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
PipelineState.synthesis           [legacy fields + evidence-grounded landscape]
PipelineState.synthesis_work      [versioned validated map-batch artifacts]
        │
        ▼
PipelineState.output_results      [latest export result for each backend]
PipelineState.sheet_url           [latest successful Google Sheets URL; legacy compatibility]
```

Each output result records the status, destination, backend identifier, error, payload fingerprint,
and completion time. Resume skips an export only when a successful result matches the selected
backend, destination identity, and current payload fingerprint; otherwise it retries only the
export. Legacy checkpoints containing only `sheet_url` continue to load.

Synthesis uses the model only for semantic judgment. Code deterministically filters failed
assessments, compacts abstracts, creates balanced batches (avoiding a 20+1 singleton partition),
selects the top 12 reading candidates from curator priority, validates evidence provenance,
derives legacy fields, and writes metrics. Unsupported gaps, future work, methodology patterns,
disagreements, and shared limitations remain empty rather than being invented. The model forms
themes and chooses a pedagogical order only within the fixed candidate set.

For collections larger than `synthesis.batch_size`, every strictly validated map result is stored
in `synthesis_work` immediately. Its fingerprint covers the normalized topic, compact paper
records in priority order, batch size, prompt version, and provider/model/generation identity. A resumed run
reuses only work with the same fingerprint, so changing evidence or model semantics recomputes it.
The reducer receives both validated map analyses and bounded source-evidence packets containing
the cited abstract excerpts and clearly separated curator assessments. `state.synthesis` is not
replaced until the final reducer output passes the complete contract. Stored map and final results
carry deterministic checksums to detect accidental edits. A validated final result is persisted
before publication, so a save interruption can resume without another provider call. The reducer's
worst-case bounded input is checked before map calls begin.

## Checkpoint / Resume

After each stage, the orchestrator calls `state.save(checkpoint_path)`. During a large synthesis,
it also saves after each validated map batch, so `--resume` does not repay for completed Gemini or
Cerebras map calls. The preflight request estimate credits matching validated batches and still
budgets one reducer call until the final synthesis is published.
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
| LLM token cost (synthesis) | Adaptive batches of 20 with a reducer for larger collections |
| Infinite agent loops | `max_iterations` hard cap, raises `MaxIterationsExceeded` |
| Paper explosion | `max_total_papers` cap (default 40 in the free-tier profile) |

## Configuration Reference

See `config.yaml` for all tunable parameters. Key settings:

```yaml
llm.provider          # "gemini" | "cerebras"
semantic_scholar.max_total_papers   # default 40 (free-tier profile)
paper_curator.batch_size             # default 8
synthesis.batch_size                 # default 20
llm.max_requests_per_run             # default 20; physical request hard cap
llm.transient_503_retries            # default 0; may be 0 or 1; never retries 429
pipeline.max_agent_iterations       # default 10
output.backend                      # "xlsx" (default) | "google_sheets"
output.xlsx.directory               # default data/outputs
output.google_sheets.credentials_file
output.google_sheets.token_file
```

The legacy top-level `google_sheets` configuration is deprecated. A Google spreadsheet ID is
checkpoint-owned so an interrupted or resumed run reuses the same remote spreadsheet without
rewriting `config.yaml`. XLSX exports use a stable topic-slug-plus-short-hash filename and atomically
replace the Crusoe-owned workbook. Google Sheets replaces only Crusoe-managed tabs and preserves
unrelated tabs.

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
   (add an optional `batch_size` keyword argument when the agent processes batches).
3. Add the stage to `orchestrator.py` with checkpoint save
4. Add a `has_X` property to `PipelineState` for resume logic
