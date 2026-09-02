# Crusoe — Multi-Agent Literature Review Pipeline

> *"I was now landed, and safe on shore, and began to look up and take a survey of myself, and what I had about me."*
> — Robinson Crusoe

**Crusoe** explores and maps an unknown research landscape. Give it a topic; it returns a structured literature review, curated and ranked paper list, and synthesized insights — all written to a Google Sheet.

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
| `llm.provider` | `"gemini"` | `"gemini"` or `"cerebras"` (`gemini-3.6-flash` / `gpt-oss-120b`) |
| `semantic_scholar.max_total_papers` | `80` | Cap on papers collected |
| `paper_curator.batch_size` | `8` | Papers per LLM assessment batch |
| `synthesis.batch_size` | `20` | Curated papers per evidence-synthesis batch |
| `google_sheets.sheet_id` | `""` | Leave blank to auto-create |
| `langfuse.enabled` | `true` | Set `false` to disable tracing |
| `langfuse.flush_at` | `1` | Send traces after each event |

## Langfuse Tracing

Crusoe sends traces to [Langfuse](https://langfuse.com) for every pipeline run:

- **Pipeline trace** — one root span per `--topic` run (session = topic)
- **Agent spans** — topic decomposition, discovery, paper curator, synthesis, Google Sheets
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
See `docs/SYNTHESIS_EVALUATION.md` for deterministic gates and the human review rubric.

## Testing Synthesis

The normal suite is offline and never spends API quota:

```powershell
python -m pytest -q
```

Live evaluations are explicit. They use fixed synthetic records, enforce call ceilings, and save
ignored artifacts under `data/evaluations/`:

```powershell
$env:RUN_GEMINI_INTEGRATION = "1"             # 3 papers; 1 call + at most 1 repair
python -m pytest -q -s tests/test_synthesis_gemini_integration.py -k fixed_corpus

$env:RUN_GEMINI_MAP_REDUCE_INTEGRATION = "1"  # 21 papers; 3 base calls + repairs
python -m pytest -q -s tests/test_synthesis_gemini_integration.py -k map_reduce

$env:RUN_CEREBRAS_INTEGRATION = "1"
python -m pytest -q -s tests/test_synthesis_cerebras_integration.py
```

Set `GEMINI_EVAL_MODEL=gemini-3.7-flash` to compare 3.7 explicitly. The project defaults to 3.6
because it has been more available in live evaluation.

## LLM Providers

- **Primary**: Google Gemini (`gemini-3.6-flash`) — set `GEMINI_API_KEY`; 3.7 can be selected explicitly for evaluation
- **Fallback**: Cerebras (`gpt-oss-120b`) — set `CEREBRAS_API_KEY`

## License

MIT
