# Crusoe — Multi-Agent Literature Review Pipeline

> *"I was now landed, and safe on shore, and began to look up and take a survey of myself, and what I had about me."*
> — Robinson Crusoe

**Crusoe** explores and maps an unknown research landscape. Give it a topic; it returns a structured literature review, enriched paper list, and synthesized insights — all written to a Google Sheet.

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
[Enrichment Agent]           →  relevance scores, methodology tags, summaries
    │
    ▼
[Synthesis Agent]            →  themes, gaps, reading order, summary
    │
    ▼
[Orchestrator]               →  Google Sheet with Papers + Synthesis tabs
```

## Quick Start

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate crusoe

# 2. Copy and fill in your API keys
cp .env.example .env
# Edit .env with your GEMINI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, etc.

# 3. Run the pipeline
python scripts/run_pipeline.py --topic "authentication tokens in web security"

# Resume from a checkpoint after a crash
python scripts/run_pipeline.py --resume

# Use Cerebras instead of Gemini
python scripts/run_pipeline.py --topic "..." --provider cerebras
```

## Configuration

All settings live in `config.yaml`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `llm.provider` | `"gemini"` | `"gemini"` or `"cerebras"` |
| `semantic_scholar.max_total_papers` | `80` | Cap on papers collected |
| `enrichment.batch_size` | `8` | Papers per LLM enrichment batch |
| `google_sheets.sheet_id` | `""` | Leave blank to auto-create |
| `langfuse.enabled` | `true` | Set `false` to disable tracing |
| `langfuse.flush_at` | `1` | Send traces after each event |

## Langfuse Tracing

Crusoe sends traces to [Langfuse](https://langfuse.com) for every pipeline run:

- **Pipeline trace** — one root span per `--topic` run (session = topic)
- **Agent spans** — topic decomposition, discovery, enrichment, synthesis, Google Sheets
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

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Sheets API** and **Google Drive API**
3. Create OAuth 2.0 credentials → download as `credentials.json` in project root
4. First run will open a browser for OAuth consent
5. Set `google_sheets.sheet_id` in `config.yaml` (or leave blank to auto-create)

## Project Structure

```
crusoe/
├── src/
│   ├── core/          # Agent loop, tool wrapper, pipeline state
│   ├── agents/        # The 5 specialized agents
│   ├── tools/         # Semantic Scholar API wrapper
│   └── llm/           # Gemini and Cerebras provider adapters
├── scripts/           # CLI entry point
├── data/              # Checkpoints saved here
└── docs/              # Architecture and learning guide
```

See `docs/ARCHITECTURE.md` for a deep-dive on each component.
See `docs/LEARNING_GUIDE.md` to understand how agent loops work.

## LLM Providers

- **Primary**: Google Gemini (`gemini-2.0-flash`) — free tier, set `GEMINI_API_KEY`
- **Fallback**: Cerebras (`llama-3.3-70b`) — free tier, fast inference, set `CEREBRAS_API_KEY`

## License

MIT
