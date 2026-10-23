# Crusoe: Multi-Agent Literature Review Pipeline

> *"I was now landed, and safe on shore, and began to look up and take a survey of myself, and what I had about me."*
>
> Robinson Crusoe

Crusoe explores and maps an unknown research landscape. Give it a topic to get a
ranked paper list, a cited synthesis, and an Excel workbook. You can inspect
synthesis statements beside their cited abstracts
and record your own support judgments. Google Sheets export is optional.

## How it works

The orchestrator runs four research stages in sequence and checkpoints progress:

1. **Topic decomposition** turns the topic into 4–6 validated keyword clusters.
2. **Discovery** searches Semantic Scholar across those clusters and selects up to
   40 unique papers by default, balancing cluster contributions.
3. **Paper curator** assesses papers in batches and computes reading priorities.
4. **Synthesis** combines successful assessments into cited themes, gaps, future
   work, methodology observations, disagreements, limitations, and a reading order.

The export stage writes eight tabs: Summary, Papers, Themes, Gaps, Future Work,
Methods, Disagreements, and Reading Order. A separate offline review command lets
you mark cited synthesis statements as supported, unsupported, or uncertain.

Discovery searches every configured keyword in cluster rounds before selecting the
corpus. It redistributes unused slots and preserves every observed cluster/query
source for selected papers. The paper cap bounds downstream LLM work, not retrieval
requests. Four clusters with three keywords each make 12 logical searches at the
default 20 results per query.

Crusoe maps a bounded set of research papers. It does not establish exhaustive
coverage, and a valid citation does not by itself prove that a statement is supported.
The [pipeline walkthrough](docs/PIPELINE_WALKTHROUGH.md) explains these boundaries
and the recent discovery and human-review changes.

## Quick start

Use Python 3.11 and Conda. From the repository root:

```bash
conda env create -f environment.yml
conda activate crusoe
```

For first-time setup, copy the example environment file:

```bash
cp .env.example .env
```

On PowerShell, you can use `Copy-Item .env.example .env`. Set `GEMINI_API_KEY` in
the local `.env` file. This is enough for the default Gemini provider and XLSX
output; Google OAuth and tracing credentials are optional.

```bash
python -m pytest -q
python scripts/run_pipeline.py --topic "authentication tokens in web security" --run-dir data/runs/my-run
```

The run directory contains `checkpoint.json`, `crusoe.log`, and an `outputs/`
directory for Excel workbooks. Omit `--run-dir` to create a unique directory under
`data/runs/`; the CLI prints its checkpoint path. Independent runs of the same
topic therefore keep separate files. Workbooks retain a topic slug and short hash.

## Run and resume

```bash
# Continue the same run
python scripts/run_pipeline.py --resume --run-dir data/runs/my-run

# Resume a checkpoint created before run directories were introduced
python scripts/run_pipeline.py --resume --checkpoint data/session_checkpoint.json

# Select Cerebras explicitly; requires CEREBRAS_API_KEY
python scripts/run_pipeline.py --topic "retrieval evaluation" --provider cerebras

# Use another configuration file
python scripts/run_pipeline.py --topic "retrieval evaluation" --config config.yaml
```

Gemini is the default provider. Cerebras is an explicit alternative, not an
automatic fallback. The configured models are `gemini-3.6-flash` and `gpt-oss-120b`.

Resume reuses successful discovery queries, strictly validated curator batches,
and matching synthesis work. Cached units must match their content fingerprint
and pass validation before reuse or request-budget credit. Changed discovery or
curator inputs invalidate dependent results. Checkpoints are replaced atomically,
and a lock excludes concurrent writers to the same run. Deleted or modified XLSX
files are regenerated instead of trusting an old success record.

Older explicit `pipeline.checkpoint_path` configurations still work. Legacy completed
stages without manifests remain loadable but lack the new input-identity guarantees.
See the [recovery guide](docs/RECOVERY_HANDOFF.md) for migration, lock recovery,
interruption examples, and the distinction between this work and human claim review.

## Review synthesis claims

After synthesis completes, inspect its cited landscape statements alongside the
full stored source abstracts:

```bash
python scripts/review_claims.py data/runs/my-run/checkpoint.json --reviewer Raghav
```

Enter `1` for supported, `2` for unsupported, `3` for uncertain, `s` to skip, or
`q` to stop. You can add a note before confirming each judgment. Review makes no
API calls and leaves the pipeline checkpoint and model output unchanged.

Judgments save after each confirmation beside the checkpoint, here as
`data/runs/my-run/checkpoint.claim-review.json`.
Running the command again skips completed items. Use `--output` for a separate
reviewer or run, `--revisit` to revise judgments, or `--summary` to see label counts:

```bash
python scripts/review_claims.py data/runs/my-run/checkpoint.json --summary
python scripts/review_claims.py data/runs/my-run/checkpoint.json --revisit
```

Review artifacts use atomic replacement and a single-writer lock. Changed claims
or cited evidence require a new artifact. Abstracts are source context, not
automatically selected supporting quotations. They can contain more text than the
1,200-character excerpts synthesis saw. Summary prose without claim-level citations
and legacy checkpoints without a cited landscape are outside this workflow.
See [claim review](docs/CLAIM_REVIEW.md) for scope, persistence, and evaluation guidance.

## Configuration and request budgets

Edit [config.yaml](config.yaml) to choose providers, retrieval limits, batching,
output, and tracing. The main defaults are:

- `semantic_scholar.results_per_query: 20` and `max_total_papers: 40`.
- `paper_curator.batch_size: 8` and `synthesis.batch_size: 20`.
- `llm.max_requests_per_run: 20` and `transient_503_retries: 0`.
- `output.backend: "xlsx"` and `pipeline.runs_directory: "data/runs"`. CLI run
  directories override the XLSX location to keep exports inside their own run.

Before running, Crusoe prints the clean LLM request plan, repair/retry ceilings,
and hard cap. A full 40-paper profile plans nine clean requests: one decomposition,
five curator batches, two synthesis maps, and one reducer. Schema repairs can
increase that to eighteen before any configured transport retries.

The hard cap counts physical LLM attempts. SDK retries are disabled; HTTP 429 is
never retried, and one HTTP 503 retry can be enabled. Semantic Scholar searches
are separate from this LLM request budget. Human claim review uses neither budget.

## Optional integrations

### Google Sheets

Set `output.backend: "google_sheets"`, then:

1. Enable the Google Sheets API in your [Google Cloud project](https://console.cloud.google.com/).
2. Create OAuth client credentials for a desktop application and save the downloaded
   JSON as `credentials.json` in the project root.
3. Run the pipeline and complete the browser consent flow. The cached token is
   stored in `token.json`.

Credential and token paths can be configured under `output.google_sheets`. The
spreadsheet ID is stored in the checkpoint so resume can reuse its destination.
Use the nested `output.google_sheets` configuration; the old top-level section is
deprecated.

### Langfuse tracing

To enable tracing, add these entries to `.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

With credentials present and `langfuse.enabled: true`, Crusoe records the pipeline,
stage spans, LLM generations, and tool/API calls. It flushes after stages and at
CLI exit. Missing credentials disable tracing; set `langfuse.enabled: false` to
disable it explicitly.

## Tests and evaluation

The default test suite runs offline without spending API quota:

```bash
python -m pytest -q
```

Live synthesis evaluations are explicit opt-ins. They use fixed synthetic inputs,
enforce call ceilings, and save ignored artifacts under `data/evaluations/`:

```bash
# Linux, macOS, or WSL
RUN_GEMINI_INTEGRATION=1 python -m pytest -q -s tests/test_synthesis_gemini_integration.py -k fixed_corpus
RUN_GEMINI_MAP_REDUCE_INTEGRATION=1 python -m pytest -q -s tests/test_synthesis_gemini_integration.py -k map_reduce
RUN_CEREBRAS_INTEGRATION=1 python -m pytest -q -s tests/test_synthesis_cerebras_integration.py
```

On PowerShell, set the variable first, for example
`$env:RUN_GEMINI_INTEGRATION = "1"`, then run the corresponding pytest command.
`GEMINI_EVAL_MODEL` can select a different Gemini model for those evaluations.

See [synthesis evaluation](docs/SYNTHESIS_EVALUATION.md) for deterministic checks
and the human rubric, and [discovery evaluation](docs/DISCOVERY_EVALUATION.md) for
fixed-budget recall, precision, and subtopic coverage. Passing offline tests does
not establish real-world relevance or claim support.

## Credentials

Keep provider secrets in the local, ignored `.env` file. To rotate the Gemini key,
create a replacement in [Google AI Studio](https://aistudio.google.com/app/apikey),
update `GEMINI_API_KEY` locally and in deployments, and restart affected processes.
Verify the replacement before revoking the old key, then review API usage for
unexpected activity. Apply the same sequence to other configured providers.

For a certificate-trust incident, update affected clients and trust stores before
creating or using replacement credentials. See [Google's key-management guidance](https://ai.google.dev/gemini-api/docs/api-key).

## Project guide

- `src/agents/` contains decomposition, discovery, curation, synthesis, and orchestration.
- `src/core/` contains pipeline state, the generic agent loop, and tool contracts.
- `src/llm/` contains provider adapters and physical-request budgeting.
- `src/tools/` contains the Semantic Scholar wrapper.
- `src/output/` contains workbook projection and XLSX/Google Sheets writers.
- `src/evaluation/` contains evaluation helpers and human-review artifacts.
- `scripts/` contains pipeline, discovery evaluation, and claim-review commands.
- `tests/` contains offline tests and explicitly enabled live evaluations.

Start with the [pipeline walkthrough](docs/PIPELINE_WALKTHROUGH.md) for current
behavior, the [architecture](docs/ARCHITECTURE.md) for component details, or the
[learning guide](docs/LEARNING_GUIDE.md) for the generic agent loop. Design decisions
and remaining limitations are recorded in [tradeoffs.md](tradeoffs.md).

## License

MIT
