# Crusoe — Multi-Agent Literature Review Pipeline

> *"I was now landed, and safe on shore, and began to look up and take a survey of myself, and what I had about me."*
> — Robinson Crusoe

**Crusoe** explores and maps an unknown research landscape. Give it a topic; it returns a structured literature review, curated and ranked paper list, and synthesized insights — exported to an Excel workbook by default or, optionally, to Google Sheets.

## What It Does

```
Topic (string)
    │
    ▼
[Topic Decomposition Agent]  →  4-6 keyword clusters
    │
    ▼
[Discovery Agent]            →  up to 80 papers via Semantic Scholar
    │
    ▼
[Paper Curator Agent]        →  validated assessments and ranked reading priorities
    │
    ▼
[Synthesis Agent]            →  evidence-grounded themes, gaps, future work, methodology, reading order, summary
    │
    ▼
[Orchestrator]               →  Spreadsheet with Papers + Synthesis tabs
```

## Quick Start

```bash
# Clone the repository, or in WSL open the existing Windows checkout:
# cd /mnt/c/Users/<windows-user>/path/to/crusoe

# Create and activate the Python 3.11 environment
conda env create -f environment.yml
conda activate crusoe

# Add provider credentials (GEMINI_API_KEY is enough for the default provider)
cp .env.example .env
# Edit .env without committing it

# Run the offline test suite
python -m pytest -q

# Run Crusoe
python scripts/run_pipeline.py --topic "authentication tokens in web security"
```

A full pipeline run writes an `.xlsx` workbook under `data/outputs/` by default and checkpoints
progress at `data/session_checkpoint.json`. The stable filename combines a topic slug with a short
hash, so reruns of the same topic target the same workbook. Google OAuth is needed only when the
optional Google Sheets backend is selected; see [Google Sheets Setup](#google-sheets-setup).

Useful alternatives:

```bash
python scripts/run_pipeline.py --resume                         # continue the last checkpoint
python scripts/run_pipeline.py --topic "..." --provider cerebras  # use CEREBRAS_API_KEY
```

## Configuration

All settings live in `config.yaml`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `llm.provider` | `"gemini"` | `"gemini"` or `"cerebras"` (`gemini-3.6-flash` / `gpt-oss-120b`) |
| `llm.max_requests_per_run` | `20` | Hard cap on physical LLM requests in one run |
| `llm.transient_503_retries` | `0` | Optional single retry for HTTP 503; HTTP 429 is never retried |
| `semantic_scholar.max_total_papers` | `40` | Free-tier-oriented cap on papers collected |
| `paper_curator.batch_size` | `8` | Papers per LLM assessment batch |
| `synthesis.batch_size` | `20` | Curated papers per evidence-synthesis batch |
| `output.backend` | `"xlsx"` | `"xlsx"` or `"google_sheets"` |
| `output.xlsx.directory` | `"data/outputs"` | Directory for generated Excel workbooks |
| `output.google_sheets.credentials_file` | `"credentials.json"` | OAuth client credentials for Google Sheets |
| `output.google_sheets.token_file` | `"token.json"` | Cached Google OAuth token |
| `langfuse.enabled` | `true` | Set `false` to disable tracing |
| `langfuse.flush_at` | `1` | Send traces after each event |

Before a run, Crusoe prints the clean request plan, the schema-repair and transport-retry ceilings,
and the enforced hard cap. The default 40-paper profile plans about 9 clean Gemini requests and at
most 18 when every schema response needs repair. SDK-managed retries are disabled so every physical
request is visible to Crusoe's counter.

## Langfuse Tracing

Crusoe sends traces to [Langfuse](https://langfuse.com) for every pipeline run:

- **Pipeline trace** — one root span per `--topic` run (session = topic)
- **Agent spans** — topic decomposition, discovery, paper curator, synthesis, spreadsheet export
- **LLM generations** — every Gemini/Cerebras call (with flush after each call)
- **Tool/API spans** — Semantic Scholar searches, agent-loop tool calls

Add your keys to `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com   # optional
```

Traces are flushed after each pipeline stage and on CLI exit so short runs don't lose data.
Set `langfuse.enabled: false` in `config.yaml` to disable without removing keys.

## Google Sheets Setup

Google Sheets is optional. Set `output.backend: "google_sheets"`, then:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Sheets API**
3. Create OAuth 2.0 credentials → download as `credentials.json` in project root
4. First run will open a browser for OAuth consent

The created spreadsheet ID is stored in the checkpoint, not written back into `config.yaml`, so
resume can reuse the destination without rewriting configuration. The old top-level
`google_sheets` configuration is deprecated; use `output.google_sheets`.

## Project Structure

```
crusoe/
├── src/
│   ├── core/          # Agent loop, tool wrapper, pipeline state
│   ├── agents/        # The 5 specialized agents
│   ├── output/        # Workbook model plus XLSX and Google Sheets writers
│   ├── tools/         # Semantic Scholar API wrapper
│   └── llm/           # Gemini and Cerebras provider adapters
├── scripts/           # CLI entry point
├── data/              # Checkpoints saved here
└── docs/              # Architecture and learning guide
```

See `docs/ARCHITECTURE.md` for a deep-dive on each component.
See `docs/LEARNING_GUIDE.md` to understand how agent loops work.
See `docs/SYNTHESIS_EVALUATION.md` for deterministic gates and the human review rubric.

## Testing Synthesis

The normal suite is offline and never spends API quota:

```bash
python -m pytest -q
```

Live evaluations are explicit. They use fixed synthetic records, enforce call ceilings, and save
ignored artifacts under `data/evaluations/`:

```bash
# Linux, macOS, or WSL
RUN_GEMINI_INTEGRATION=1 python -m pytest -q -s \
  tests/test_synthesis_gemini_integration.py -k fixed_corpus

RUN_GEMINI_MAP_REDUCE_INTEGRATION=1 python -m pytest -q -s \
  tests/test_synthesis_gemini_integration.py -k map_reduce

RUN_CEREBRAS_INTEGRATION=1 python -m pytest -q -s \
  tests/test_synthesis_cerebras_integration.py
```

On PowerShell, set the corresponding variable first—for example,
`$env:RUN_GEMINI_INTEGRATION = "1"`—then run the `python -m pytest ...` command.

Set `GEMINI_EVAL_MODEL=gemini-3.7-flash` to compare 3.7 explicitly. The project defaults to 3.6
because it has been more available in live evaluation.

## LLM Providers

- **Primary**: Google Gemini (`gemini-3.6-flash`) — set `GEMINI_API_KEY`; 3.7 can be selected explicitly for evaluation
- **Fallback**: Cerebras (`gpt-oss-120b`) — set `CEREBRAS_API_KEY`

## License

MIT
