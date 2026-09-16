# Crusoe Decision Tradeoffs

This document records the benefits, costs, rejected alternatives, and reconsideration triggers behind Crusoe's technical and procedural decisions. It is descriptive rather than normative: a decision that is correct for the current small, local-first project may become wrong as scale, reliability requirements, contributors, providers, or product goals change.

The file is intentionally tracked. The `!tradeoffs.md` rule in `.gitignore` makes that policy explicit even though no broader Markdown rule currently ignores it. Private scratch notes remain under the ignored `.crusoe/` directory; durable reasoning belongs here and should change in the same pull request as the decision it explains.

## How to maintain this document

- Update the relevant entry whenever behavior, defaults, schemas, dependencies, or development procedure changes.
- Describe both the upside and the cost. “No meaningful downside” is usually a sign that the analysis is incomplete.
- Distinguish a deliberate choice from an implementation constraint or temporary expedient.
- Preserve superseded reasoning, marking it as superseded and linking to the replacement, when that history would prevent the same debate from recurring.
- Revisit a decision when its stated trigger occurs rather than treating this file as permanent architecture law.

## 1. Product and system boundary decisions

### 1.1 A sequential multi-stage pipeline

**Decision.** Crusoe runs topic decomposition, discovery, paper curation, synthesis, and export in a fixed order through one orchestrator.

**Benefits.** The data dependency is obvious, each handoff can be inspected, checkpoints align with meaningful milestones, and failures have a clear owner. Sequential execution also keeps API pressure and concurrent mutation of shared state low.

**Costs and risks.** Independent searches and LLM batches do not run concurrently, so wall-clock latency is higher. The orchestrator accumulates integration responsibilities and can become a central change bottleneck. A fixed order makes feedback loops—such as synthesis requesting more discovery—awkward.

**Alternatives.** A DAG/workflow engine would offer parallelism, richer retries, and scheduling; an event-driven agent network would permit feedback; a single large prompt would reduce code. Each alternative adds operational machinery or reduces determinism and inspectability.

**Revisit when.** Runtime becomes a product constraint, stages need independent scaling, multiple users run jobs concurrently, or later stages routinely need to send work backward.

### 1.2 Specialized agents instead of one general agent

**Decision.** Semantic responsibilities are split among narrowly scoped agent modules while deterministic code owns orchestration and validation.

**Benefits.** Prompts are easier to reason about, outputs have stage-specific contracts, failures are localized, and individual stages can be evaluated or replaced. Deterministic boundaries reduce the chance that an LLM silently changes control flow.

**Costs and risks.** Every boundary creates schemas, adapters, tests, and possible information loss. Several calls cost more and take longer than one call. Cross-stage optimization is limited because each agent sees a curated slice of context.

**Alternatives.** One end-to-end agent is simpler but harder to constrain and resume. Autonomous agents communicating directly are flexible but introduce nondeterministic routing, duplicate work, and harder debugging.

**Revisit when.** A stable model can reliably produce the entire artifact within one context window, or a workflow engine provides typed multi-agent communication without sacrificing traceability.

### 1.3 Literature mapping rather than exhaustive systematic review

**Decision.** Crusoe produces a useful research landscape from bounded discovery and model-assisted assessment; it does not claim exhaustive coverage or formal systematic-review compliance.

**Benefits.** Results arrive within practical API, cost, and time limits. The tool is useful for exploration and prioritization without requiring a full review protocol.

**Costs and risks.** Search coverage, database coverage, screening reproducibility, and bias controls are insufficient for high-stakes systematic review. Users may over-interpret polished synthesis as comprehensive evidence.

**Alternatives.** PRISMA-style workflows, multiple bibliographic databases, duplicate independent screening, and registered protocols provide stronger evidence guarantees at much greater operational and human cost.

**Revisit when.** Crusoe is used for clinical, legal, regulatory, or publication-grade evidence claims.

## 2. State, contracts, and resumability

### 2.1 One mutable `PipelineState`

**Decision.** All stages read and update a shared dataclass that is JSON-serializable.

**Benefits.** Data flow is concrete, checkpointing is straightforward, the CLI has one result object, and legacy JSON is human-readable. A local file is enough; no database is required.

**Costs and risks.** The state object is a broad coupling surface. In-place mutation makes ownership less explicit and complicates concurrent execution. Many payloads remain untyped dictionaries, so errors can move downstream before detection. Large abstracts and synthesis work can make checkpoints bulky.

**Alternatives.** Immutable stage outputs improve provenance; fully typed models strengthen validation; a database or artifact store supports scale and queries. Those choices require migrations, serialization policy, and more infrastructure.

**Revisit when.** Multiple processes touch a run, checkpoint size becomes material, or schema mistakes repeatedly escape stage validation.

### 2.2 JSON checkpoint after each completed stage

**Decision.** The orchestrator persists the full state after durable milestones and after individual synthesis map batches.

**Benefits.** Expensive API work survives interruption, checkpoints can be inspected manually, and resume needs no service dependency. Fine-grained synthesis saves avoid repaying completed map calls.

**Costs and risks.** Most checkpoint writes are not atomic, so process or disk failure during `write_text` can corrupt the file. A single checkpoint path supports one active run safely; concurrent runs can overwrite each other. Rewriting the full state amplifies I/O.

**Alternatives.** Atomic temporary-file replacement, append-only events, SQLite, or an object store improve durability and concurrency but add recovery logic and operational complexity.

**Revisit when.** Concurrent jobs are supported, corrupted checkpoints occur, or checkpoints become too large for cheap full rewrites.

### 2.3 Presence-based stage completion

**Decision.** Non-empty state fields such as `keyword_clusters`, `papers_raw`, `papers_curated`, and `synthesis` mark stages complete.

**Benefits.** Resume logic is simple and backward-compatible with old checkpoint shapes. There is no separate status machine to drift from the payload.

**Costs and risks.** A legitimate empty result cannot be distinguished from “not run.” Presence alone does not prove that data matches current configuration, prompts, provider, or upstream inputs. Synthesis adds stronger validation and fingerprints, but earlier stages do not have equivalent identities.

**Alternatives.** Explicit per-stage status, input fingerprints, schema versions, and completion timestamps are more robust but increase migration and invalidation complexity.

**Revisit when.** Empty discovery must be resumable, configurations change frequently between resumes, or stale upstream artifacts cause correctness issues.

### 2.4 Versioned and fingerprinted synthesis work

**Decision.** Partial map-reduce artifacts carry checksums and an identity derived from topic, eligible papers, batching, prompt version, and provider/model generation settings.

**Benefits.** Resume reuses only semantically compatible work, completed calls are not repeated, and accidental checkpoint edits are detectable. The final synthesis is published only after complete validation.

**Costs and risks.** Fingerprint composition becomes a compatibility contract: omitting a semantic input risks stale reuse, while adding an irrelevant input causes needless recomputation. Provider behavior can change behind the same model name. Stored map artifacts increase state size and code complexity.

**Alternatives.** Always recompute is simple but costly; trust any checkpoint is cheap but unsafe; content-addressed immutable artifacts are stronger but need storage management.

**Revisit when.** Providers expose immutable model revisions, prompts become externally versioned, or synthesis artifacts move into a dedicated store.

### 2.5 Backward-compatible checkpoint loading

**Decision.** Legacy keyword-cluster dictionaries and `sheet_url` checkpoints remain loadable, and `sheet_url` mirrors the latest successful Google Sheets URL.

**Benefits.** Existing users do not lose resumability after upgrades. Incremental migration allows the output redesign to ship without a flag day.

**Costs and risks.** Duplicate representations can disagree, compatibility branches expand the test matrix, and old weakly validated data remains accepted. Deprecations can persist indefinitely without a removal policy.

**Alternatives.** A one-time migration tool simplifies runtime code but adds an upgrade step and failure mode. Breaking old checkpoints is clean but wastes previous paid work.

**Revisit when.** A major version permits breaking changes or telemetry shows legacy checkpoint use has ended.

## 3. Discovery and evidence selection

### 3.1 LLM-generated keyword clusters with strict validation

**Decision.** Topic decomposition asks the model for 4–6 clusters, each with 3–5 unique keywords, and permits one schema-repair attempt.

**Benefits.** The model supplies semantic breadth while code enforces predictable downstream shape. Whitespace normalization and case-insensitive uniqueness reduce redundant searches. One repair recovers common formatting errors without opening an unbounded loop.

**Costs and risks.** Fixed counts can be too broad for narrow topics and too shallow for broad ones. Strict rejection may discard semantically good output for a minor shape error. A single repair may fail under transient model behavior, while additional repairs would increase cost.

**Alternatives.** Dynamic cluster counts, deterministic keyword extraction, or permissive coercion trade consistency for adaptability. Human-approved search plans improve quality but remove unattended execution.

**Revisit when.** Evaluation shows systematic recall gaps or the fixed cluster shape dominates search cost without improving coverage.

### 3.2 Semantic Scholar as the sole discovery source

**Decision.** Discovery calls Semantic Scholar directly rather than asking an LLM to browse or aggregating multiple databases.

**Benefits.** The API returns structured metadata, stable paper IDs, authors, citations, abstracts, and URLs. Direct calls are cheaper and more reproducible than LLM-mediated discovery.

**Costs and risks.** Coverage and metadata quality inherit one provider's corpus, indexing latency, outages, rate limits, and field conventions. Some disciplines and paywalled content may be underrepresented. Citation count is not quality.

**Alternatives.** Crossref, OpenAlex, PubMed, arXiv, domain databases, or federated search improve recall but require deduplication, source precedence, licensing review, and normalized identifiers.

**Revisit when.** Users target poorly covered disciplines, missing-paper reports become common, or availability requirements exceed a single external dependency.

### 3.3 Direct API discovery rather than the generic agent tool loop

**Decision.** Discovery is deterministic Python over validated clusters even though Semantic Scholar functions are also exposed as generic tools.

**Benefits.** Query count, result caps, deduplication, and retry behavior are visible and testable. The model cannot improvise costly or irrelevant searches.

**Costs and risks.** Search adaptation is limited; the pipeline cannot inspect early results and refine weak queries. Two interfaces to the search functions can drift.

**Alternatives.** A tool-using discovery agent is adaptive but less predictable. A deterministic multi-pass algorithm could add refinement without handing control to an LLM.

**Revisit when.** Fixed queries routinely return poor evidence and a bounded refinement strategy demonstrates measurable recall gains.

### 3.4 Hard caps and deduplication

**Decision.** Results are capped globally, with a default of 40 papers for the free-tier profile, and duplicates are collapsed by stable identifiers.

**Benefits.** Runtime, prompt size, request count, and spreadsheet size stay bounded. Stable IDs support evidence references and deterministic fingerprints.

**Costs and risks.** Ordering and provider ranking decide which evidence survives the cap, which can bias synthesis. Papers without stable IDs are harder to handle. Forty papers are rarely enough for exhaustive coverage.

**Alternatives.** Adaptive stopping, stratified quotas per theme/year/source, or relevance-first incremental retrieval provide better coverage control but require stronger metrics.

**Revisit when.** Corpus diversity matters more than free-tier cost or evaluation reveals cap-induced theme loss.

### 3.5 Batched paper curation

**Decision.** Papers are assessed in batches of eight by default, and failed assessments are represented rather than silently treated as valid.

**Benefits.** Batching amortizes prompt overhead and keeps calls predictable. Per-paper validated fields make ranking and later filtering deterministic. Explicit failure status preserves provenance.

**Costs and risks.** Papers in a batch influence one another; changing batch boundaries can change scores. Larger batches risk context pressure, while smaller batches cost more. Model scores are ordinal judgments, not calibrated measurements.

**Alternatives.** One paper per call isolates assessments but is expensive. All papers in one prompt enables global comparison but can exceed context and reduce attention. Embedding or rules-based ranking is cheaper but less semantically rich.

**Revisit when.** Ranking stability across batch partitions is poor or model/context economics change materially.

### 3.6 Deterministic reading-priority calculation

**Decision.** Code derives reading priority from validated curator fields rather than letting the final model freely rank every paper.

**Benefits.** The candidate set is auditable, stable, and bounded; the synthesis model only arranges a fixed top set pedagogically. This separates evidence scoring from narrative presentation.

**Costs and risks.** A hand-designed formula encodes product assumptions and may overvalue citations, recency, or curator scores. Determinism can make a consistently biased ranking look authoritative.

**Alternatives.** Pairwise LLM ranking, learning-to-rank, user-selected weights, or diverse submodular selection may improve relevance but need evaluation data and more compute.

**Revisit when.** User feedback or labeled relevance data can support calibration, personalization, or diversity-aware ranking.

## 4. Synthesis quality and cost controls

### 4.1 Adaptive direct versus map-reduce synthesis

**Decision.** Small eligible collections use one synthesis call; larger collections use balanced map batches plus a reducer.

**Benefits.** Small runs avoid unnecessary abstraction and cost. Large runs remain within context bounds, can checkpoint maps, and distribute attention across papers. Balanced partitioning avoids a weak singleton final batch.

**Costs and risks.** Map-reduce can lose cross-batch relationships and amplify early summarization errors. Results change at the batch-size threshold. The reducer consumes interpretations plus bounded source packets rather than every full abstract.

**Alternatives.** One huge-context call maximizes cross-paper visibility but costs more and may attend unevenly. Hierarchical or iterative synthesis handles very large corpora but adds calls and compounded error.

**Revisit when.** Context windows and prices make direct synthesis practical, or evaluation shows cross-batch disagreements are consistently missed.

### 4.2 Evidence-grounded structured synthesis

**Decision.** Themes, gaps, future work, methods, disagreements, and limitations use explicit supporting-paper references and strict schemas.

**Benefits.** Claims can be traced to papers, output maps cleanly into filterable spreadsheet columns, and validation can reject invented identifiers. Empty unsupported sections are preferable to fabricated completeness.

**Costs and risks.** Citation presence does not prove that a source supports a claim. Strict schemas constrain nuance and can encourage the model to force ambiguous findings into available fields. Empty sections may disappoint users even when caution is correct.

**Alternatives.** Free-form narrative is richer but harder to audit and export. Sentence-level evidence spans or quotation alignment improve grounding but require more tokens and validation.

**Revisit when.** Claim-level entailment evaluation becomes available or users need publication-quality narrative rather than a research map.

### 4.3 One schema-repair attempt

**Decision.** Invalid semantic outputs generally receive one corrective retry.

**Benefits.** Common JSON/schema failures are recoverable, while costs and physical calls remain bounded and visible.

**Costs and risks.** The repair prompt consumes budget and may still fail. Treating all validation failures alike ignores whether an error is trivially repairable or indicates semantic misunderstanding.

**Alternatives.** Local coercion is cheap but can hide meaning changes; multiple adaptive repairs improve success rates but weaken cost predictability; native constrained decoding would be preferable where providers support the complete schema reliably.

**Revisit when.** Failure telemetry identifies safe deterministic repairs or provider-native structured output becomes dependable.

### 4.4 Explicit request estimation and hard budget

**Decision.** Crusoe computes clean, validation-ceiling, and transport-ceiling request counts before running and enforces a physical-call cap.

**Benefits.** Users see potential cost before execution, hidden SDK retries are avoided, and runaway repair/retry loops are prevented. Resume credits completed work.

**Costs and risks.** Request count is only a proxy for token cost, latency, and monetary cost. Estimates depend on accurate stage logic and may become stale as prompts or providers change. A hard cap can terminate useful work late in a run.

**Alternatives.** Token or dollar budgets are more meaningful but require provider pricing and usage accounting. Soft warnings are friendlier but do not guarantee bounds.

**Revisit when.** Providers expose reliable preflight token counts and usage/pricing APIs, or users need per-run monetary budgets.

### 4.5 Retry only bounded HTTP 503 failures and never 429

**Decision.** Transport retries are configurable as zero or one for transient 503 errors; rate-limit 429 errors are not retried by the LLM layer.

**Benefits.** Physical calls remain accountable and a quota error cannot trigger a long retry storm. A single 503 retry covers a common transient failure.

**Costs and risks.** Some recoverable 429 responses include a useful retry-after window, and one 503 retry may be insufficient. Conservative retrying lowers completion rates during brief provider incidents.

**Alternatives.** Exponential backoff with jitter and retry-after support improves resilience but makes runtime and cost less predictable. Queueing the run for later is operationally cleaner but needs a scheduler.

**Revisit when.** Crusoe gains background jobs, provider SLAs require stronger resilience, or retry telemetry supports a safe policy.

### 4.6 Gemini default with a Cerebras adapter

**Decision.** The provider interface supports Gemini and Cerebras, with Gemini 3.6 Flash as the documented default and provider-specific defaults isolated behind adapters.

**Benefits.** Users have a fallback, tests can use a common interface, and most pipeline code is provider-neutral. A fast model keeps exploratory runs practical.

**Costs and risks.** Provider APIs, tool formats, safety behavior, schemas, and retry semantics are not truly identical. Default model names age quickly, and reproducibility is limited when providers update aliases. Supporting two adapters doubles integration surface.

**Alternatives.** One provider reduces maintenance but increases lock-in. A third-party abstraction broadens choice but introduces another dependency and may obscure physical calls.

**Revisit when.** One provider repeatedly fails quality gates, immutable versions become available, or adapter maintenance exceeds the value of fallback choice.

## 5. Spreadsheet output architecture

### 5.1 A normalized workbook model between state and backends

**Decision.** A pure projector converts `PipelineState` into backend-neutral workbooks, worksheets, rows, and cells; adapters render that model.

**Benefits.** XLSX and Google Sheets receive the same logical content and tab order. Projection can be tested without I/O, new backends have a narrow contract, and synthesis logic stays independent of presentation APIs.

**Costs and risks.** The common model targets the intersection of backend capabilities, so backend-specific features require metadata or escape hatches. There is another representation to maintain and fingerprint. “Parity” can hide meaningful differences in cell limits or hyperlink behavior.

**Alternatives.** Backend-specific projection is simpler initially but drifts. A richer document intermediate representation supports more formats but is premature for two spreadsheet targets.

**Revisit when.** A third substantially different backend is added, charts/formulas become required, or backend-specific UX cannot fit cleanly in the common model.

### 5.2 Exactly one selected backend per invocation

**Decision.** `output.backend` selects either `xlsx` or `google_sheets`.

**Benefits.** Success and retry semantics are unambiguous, users do not accidentally create remote resources, and a failure in an unused backend cannot spoil a run.

**Costs and risks.** Producing both formats requires two invocations. Backend switching creates another output result and can surprise users who expected mirroring.

**Alternatives.** A list of sinks supports fan-out but needs partial-success policy, ordering, and independent credentials. Separate export commands decouple execution further but add CLI complexity.

**Revisit when.** Routine workflows require simultaneous local archival and remote collaboration.

### 5.3 XLSX as the default

**Decision.** Local Excel workbooks are the documented default.

**Benefits.** Fresh runs need no Google account, OAuth consent, network write, or remote resource creation. Files are portable, inspectable offline, and easy to archive.

**Costs and risks.** Collaboration and live sharing are manual. Local files can be lost, duplicated, or opened with differing spreadsheet software. `openpyxl` adds a dependency and XLSX is a complex binary format.

**Alternatives.** Google Sheets is collaboration-first but operationally heavier. CSV is universal but cannot preserve eight tabs, styles, types, and hyperlinks in one artifact.

**Revisit when.** The primary usage becomes team collaboration rather than local research, or a hosted Crusoe service owns authentication.

### 5.4 Stable topic-slug and short-hash XLSX filenames

**Decision.** The destination filename is derived deterministically from a readable topic slug plus a short topic hash.

**Benefits.** The path is stable across retries, readable to humans, resistant to most slug collisions, and does not depend on timestamps.

**Costs and risks.** Same-topic runs overwrite the same artifact even if configuration or evidence changes. A short hash has a small collision probability. Topic text leaks into local filenames and path-length or reserved-name rules require sanitization.

**Alternatives.** Full payload hashes guarantee content identity but are less readable and change every run. Timestamps preserve history but break idempotence. User-supplied paths provide control but add validation and overwrite risk.

**Revisit when.** Users need run history, multiple configurations per topic, stronger privacy, or multi-user storage.

### 5.5 Atomic XLSX replacement

**Decision.** The writer saves a temporary workbook and replaces the stable destination only after successful serialization.

**Benefits.** A failed write does not leave a partially written final workbook, and reruns remove stale rows by replacing the whole managed artifact.

**Costs and risks.** Atomicity depends on temporary and destination paths sharing a filesystem and on platform replace semantics. Replacement can fail if Excel holds the file open. Whole-file rewrites cost more for large workbooks.

**Alternatives.** In-place updates preserve manual edits but risk stale data and partial failure. Versioned filenames avoid locks and retain history but sacrifice a stable destination.

**Revisit when.** Workbooks become large, manual edits must survive, or Windows file-lock failures become frequent.

### 5.6 Fresh Google spreadsheet creation with checkpoint-scoped reuse

**Decision.** A fresh run creates a spreadsheet, checkpoints its ID immediately, and only resume reuses that destination.

**Benefits.** Separate research runs do not silently overwrite each other. Checkpointing before tab writes prevents orphan multiplication after a mid-export failure. Configuration remains immutable.

**Costs and risks.** Repeated fresh runs create spreadsheet clutter. If checkpointing fails after remote creation, an orphan can still exist. Reusing a checkpoint after the user deletes or revokes access to the sheet requires failure handling.

**Alternatives.** A configured fixed sheet is predictable but couples unrelated runs and requires mutation policy. Searching Drive by title is ambiguous and needs broader scope. Idempotency metadata could be stronger but adds remote state.

**Revisit when.** A hosted run registry exists or users need named, reusable destinations independent of checkpoints.

### 5.7 Sheets-only OAuth scope and no Drive client

**Decision.** Google export requests only the spreadsheet scope and does not use Drive APIs.

**Benefits.** Least privilege reduces consent surface, security impact, dependency behavior, and credential exposure. Spreadsheet creation is available through the Sheets API.

**Costs and risks.** Crusoe cannot search, move, permission, organize, or clean up spreadsheets in Drive. Orphan cleanup and destination discovery remain manual.

**Alternatives.** Drive scope enables lifecycle management but asks users for broader access. A service account simplifies automation but creates sharing and credential-distribution issues.

**Revisit when.** Folder placement, sharing, discovery, or cleanup becomes a core supported workflow.

### 5.8 Preserve unrelated Google tabs; replace reserved Crusoe tabs

**Decision.** The adapter owns a fixed set of tab names, clears and rewrites their complete managed ranges, and leaves other tabs untouched.

**Benefits.** Users can add notes or analysis tabs without losing them. Clearing avoids stale cells when new output shrinks. Ownership is understandable.

**Costs and risks.** A user tab with a reserved name is treated as Crusoe-owned and overwritten. Formulas elsewhere that reference managed ranges may break as layout evolves. Full-range clearing can be expensive.

**Alternatives.** Recreate the entire spreadsheet for strongest consistency, or use hidden metadata/developer properties to identify ownership. Both add disruption or API complexity.

**Revisit when.** Users customize managed tabs, schema evolution breaks dependent analysis, or large sheets make full clearing slow.

### 5.9 Eight normalized, filterable tabs

**Decision.** Output uses Summary, Papers, Themes, Gaps, Future Work, Methods, Disagreements, and Reading Order in deterministic order.

**Benefits.** Each concept has a focused, filterable schema. Stable names and order support user habits, tests, and downstream automation. Evidence references remain explicit columns instead of being buried in prose.

**Costs and risks.** Wide Papers rows and multiple tabs can overwhelm casual users. Some concepts overlap, particularly gaps and shared limitations. Stable names become an API and constrain localization or redesign.

**Alternatives.** One denormalized sheet is easier to scan but duplicates data. A relational database is more queryable but less approachable. A dashboard is friendlier but outside the portable-output goal.

**Revisit when.** Usability studies favor a different information architecture or downstream consumers need a formal machine-readable schema.

### 5.10 Rich paper metadata and separate synthesis columns

**Decision.** Paper IDs, full author names, URLs/DOIs, bibliographic metadata, scores, assessment fields, summaries, and abstracts are retained; synthesis claims and references remain separately filterable.

**Benefits.** Users can audit rankings, follow sources, filter numerically, and distinguish provider metadata from model judgment. Full authors and abstracts avoid unnecessary information loss.

**Costs and risks.** Workbooks become wide and large, abstracts may carry copyrighted text or sensitive query context, and field availability varies. Scores can imply precision beyond model reliability.

**Alternatives.** Minimal exports are easier to read but force users back to checkpoints or APIs. A normalized paper/author relation is cleaner for machines but awkward in spreadsheets.

**Revisit when.** File size, licensing, privacy, or user comprehension outweighs completeness.

### 5.11 Preserve text until backend limits, then visibly truncate

**Decision.** Text is not arbitrarily shortened by the projector; adapters truncate only at Excel or Google cell limits and append a marker.

**Benefits.** Backend parity preserves maximum information, and visible markers prevent silent loss. Synthesis prompts may compact inputs without forcing the export to discard source abstracts.

**Costs and risks.** Near-limit cells are unwieldy, slow to render, and hard to scan. The marker consumes part of the limit. Different backend limits mean byte-for-byte output cannot be identical.

**Alternatives.** A product-level configurable limit improves readability but discards data earlier. Storing long text in attachments or comments complicates portability.

**Revisit when.** Workbook performance degrades or users consistently prefer concise excerpts with links to full text.

### 5.12 Formula-injection prevention without changing display

**Decision.** Google writes values with `RAW`; XLSX forces dangerous prefix strings to the literal string type while preserving their visible text.

**Benefits.** Untrusted titles and abstracts cannot execute as spreadsheet formulas, and users still see the original value. Safety is enforced in adapters where spreadsheet semantics are known.

**Costs and risks.** Legitimate formulas cannot be intentionally exported through the current model. Spreadsheet applications differ in how they interpret exotic prefixes, so tests must cover the supported path.

**Alternatives.** Prefixing an apostrophe is simple but changes displayed or copied values in some clients. Sanitizing upstream corrupts source text. Allowing formulas requires explicit trusted cell types and a stricter boundary.

**Revisit when.** Crusoe intentionally generates formulas or additional spreadsheet clients expose different injection behavior.

### 5.13 Presentation formatting in adapters

**Decision.** Writers apply frozen headers, filters, wrapping, top alignment, bounded widths, numeric types, styles, and hyperlinks.

**Benefits.** The artifact is useful immediately and preserves numeric filtering and clickable provenance. Adapter-level formatting maps the common intent to each API.

**Costs and risks.** Formatting code is verbose, backend parity is approximate, API requests increase, and style assertions can be brittle. Bounded widths trade readability of long content for manageable sheets. Today XLSX constrains widths, while Google Sheets uses automatic resizing without a post-resize maximum. The workbook model carries hyperlink metadata and XLSX applies it, but Google currently writes the cell value with `RAW` rather than translating that metadata into a Sheets hyperlink. URL/DOI text remains available, but visual and behavioral parity is incomplete.

**Alternatives.** Raw data-only output is simpler and faster but feels unfinished. Templates offer richer design but introduce versioning and compatibility concerns.

**Revisit when.** Styling maintenance dominates adapter work, a stable template/design system is introduced, or users require bounded Google columns and clickable-link parity. Treat the latter two as known follow-up work rather than properties already guaranteed by the common model.

## 6. Failure, security, and observability

### 6.1 Invalid configuration is fatal; operational export failure is non-fatal

**Decision.** Preflight rejects unsupported or malformed configuration, while runtime spreadsheet failures preserve completed synthesis and record a retryable failed result.

**Benefits.** User-fixable mistakes fail before spending LLM quota. Valuable research is not discarded because a local file is locked or Google is unavailable. `--resume` can perform export only without initializing an LLM.

**Costs and risks.** A process can exit “complete” without its expected artifact unless the user notices the warning. Deciding whether an error is configuration or operational is sometimes ambiguous.

**Alternatives.** Make all export failures fatal for simple automation semantics, or always return success with warnings for resilience. Either extreme loses useful nuance.

**Revisit when.** Crusoe gains machine-consumed exit codes, job status APIs, or delivery SLAs.

### 6.2 Output result matching by backend, destination, and payload hash

**Decision.** Resume skips export only if a successful record matches the selected backend, stable destination, and normalized workbook fingerprint.

**Benefits.** Idempotent resumes avoid needless writes while changed evidence or backend choice triggers regeneration. Failed results are replaced by success and stale general errors are removed.

**Costs and risks.** A hash proves projected payload equality, not that the external destination still contains it; users may edit or delete output after success. Destination identity semantics differ between a path and spreadsheet ID.

**Alternatives.** Always rewrite guarantees convergence but costs time and remote requests. Reading and hashing the destination detects drift but is expensive and complicated by formatting.

**Revisit when.** External edits are common or delivery verification becomes more important than cheap resume.

### 6.3 Sanitized errors and credential-neutral tracing

**Decision.** Logs, CLI summaries, checkpoints, traces, and output results record safe error classes/summaries and semantic provider identity, not raw credentials or exception locals.

**Benefits.** Debug artifacts are safer to share, secrets are less likely to leak, and provider identity remains useful for cache invalidation.

**Costs and risks.** Sanitization removes details needed to diagnose unusual failures. A safe class name may be too vague, and future messages/metadata can accidentally reintroduce sensitive content.

**Alternatives.** Full local debug logs aid diagnosis but create a high-value secret store. Structured allowlisted diagnostic codes offer a stronger compromise but require taxonomy work.

**Revisit when.** Supportability suffers, at which point add opt-in local diagnostics with explicit redaction and retention rules.

### 6.4 Langfuse enabled by configuration and flushed frequently

**Decision.** Crusoe can trace pipeline, agent, LLM, tool, and export work to Langfuse, with frequent flushes suitable for short CLI runs.

**Benefits.** Latency, call count, failures, and stage behavior become observable; frequent flushes reduce lost telemetry on exit.

**Costs and risks.** Telemetry adds network latency, a third-party dependency, cost, and privacy considerations for research topics or model inputs. Frequent flushing is less efficient than batching.

**Alternatives.** Local structured logs avoid remote disclosure but are harder to aggregate. OpenTelemetry is more vendor-neutral but adds setup. No tracing is simplest but makes model pipelines opaque.

**Revisit when.** Privacy requirements tighten, hosted operation needs standardized telemetry, or trace volume/cost increases.

### 6.5 Local OAuth credential and token files resolved from config location

**Decision.** Google credential and token paths are relative to the selected configuration file and ignored by Git.

**Benefits.** Alternate configurations are portable as a unit, secrets are not committed, and users can isolate environments.

**Costs and risks.** Local token files remain sensitive at rest, relative-path resolution can surprise users, and interactive OAuth is awkward in headless environments. Gitignore is a guardrail, not secret management.

**Alternatives.** OS keychains, environment-provided JSON, workload identity, or service accounts improve deployment but complicate local setup and cross-platform support.

**Revisit when.** Crusoe runs in CI, containers, multi-user machines, or a hosted service.

## 7. Configuration, dependency, and interface choices

### 7.1 One YAML configuration plus CLI overrides

**Decision.** Defaults and tunables live in `config.yaml`; the CLI currently overrides provider and accepts an alternate config path.

**Benefits.** Configuration is readable, reviewable, and easy to copy. Nested backend/provider settings are clearer than many environment variables. Secrets remain in `.env` or OAuth files.

**Costs and risks.** YAML has typing surprises and comments are not preserved by round-trip mutation, which is why runtime no longer writes sheet IDs into it. CLI override coverage is uneven. Config schema is validated across modules rather than centrally.

**Alternatives.** Typed settings models provide consistent validation; TOML has simpler scalar rules; exhaustive CLI flags improve discoverability but become unwieldy.

**Revisit when.** Configuration errors are frequent, migrations multiply, or Crusoe becomes a packaged application.

### 7.2 Paths relative to the selected config file for output credentials

**Decision.** Google auth files are anchored to the config directory, while existing pipeline paths retain their current process-relative semantics.

**Benefits.** A custom configuration can travel with its auth references and does not depend on the invocation directory for those files.

**Costs and risks.** Mixed path bases are inconsistent and can confuse users. Moving a config changes resolved destinations. Absolute paths reduce portability.

**Alternatives.** Resolve every path against config location for consistency, or every path against project root/current working directory for familiarity.

**Revisit when.** The broader configuration system is migrated; choose and document one uniform path policy then.

### 7.3 Python 3.11 and a Conda environment file

**Decision.** Development is pinned to Python 3.11 through `environment.yml`, while Python dependencies use minimum versions.

**Benefits.** Python 3.11 is mature and fast, Conda handles interpreter setup across Windows/WSL, and minimum bounds allow security and compatibility updates.

**Costs and risks.** Minimum-only dependencies do not guarantee reproducible environments; future major releases can break the project. Conda is heavier than `venv`/`pip` and not every contributor uses it. There is no packaged project metadata or lock file.

**Alternatives.** `pyproject.toml` plus a lock tool improves packaging and reproducibility; exact pins improve stability but require active upgrades.

**Revisit when.** CI/release builds require reproducibility, dependency conflicts appear, or the project is distributed as a package.

### 7.4 Synchronous APIs and CLI execution

**Decision.** Provider, discovery, export, and orchestration interfaces are synchronous and invoked by one CLI process.

**Benefits.** Control flow and exception handling are straightforward, debugging is easy, and libraries fit conventional Python usage.

**Costs and risks.** Network wait time cannot be overlapped, cancellation is coarse, and long jobs tie up the terminal. Scaling to many jobs would require process-level orchestration.

**Alternatives.** Async I/O improves concurrency but complicates adapters and tests. A job queue provides durable background work but adds services and state.

**Revisit when.** Latency or concurrent throughput becomes more important than local simplicity.

## 8. Testing and evaluation decisions

### 8.1 Offline tests are the default suite

**Decision.** Normal `pytest` runs use fakes and deterministic fixtures; live Gemini, Cerebras, and Google integration tests are explicit opt-ins.

**Benefits.** Tests are fast, free, reproducible, and safe to run on every change without credentials. Failure usually indicates code rather than provider drift.

**Costs and risks.** Mocks can diverge from real APIs, especially for Google batch formatting and provider response shapes. Passing offline tests does not prove authentication, quotas, or live model quality.

**Alternatives.** Live tests on every CI run detect drift but cost money, expose secrets, and are flaky. Contract recordings sit between the two but age and may contain sensitive data.

**Revisit when.** CI has secure credentials and a budget for scheduled canaries, or external API drift causes production failures.

### 8.2 Read-back tests for XLSX and request-shape tests for Google

**Decision.** XLSX tests reopen generated files and inspect values/styles/types, while Google tests fake the API and inspect clear, RAW-write, and formatting requests.

**Benefits.** Local serialization is verified end-to-end. Remote behavior can be tested without creating spreadsheets or requiring OAuth.

**Costs and risks.** An `openpyxl` read-back does not guarantee rendering in every Excel-compatible application. Fake Google services validate intended request shape, not API acceptance or UI appearance.

**Alternatives.** Golden-file or visual tests cover appearance but are brittle. Periodic live smoke tests provide confidence at operational cost.

**Revisit when.** Users report rendering differences or Google changes validation semantics.

### 8.3 Fixed synthetic corpora for synthesis evaluation

**Decision.** Live model evaluations use stable synthetic papers, explicit call ceilings, saved ignored artifacts, deterministic gates, and a human rubric.

**Benefits.** Provider/model comparisons have consistent inputs, costs are bounded, and artifacts support qualitative review without contaminating the repository.

**Costs and risks.** Synthetic records are cleaner and more balanced than real literature. A small fixed corpus can be overfit by prompts or fail to expose domain-specific weaknesses. Human scores are subjective.

**Alternatives.** A versioned benchmark of real papers is more representative but raises licensing, size, and maintenance issues. Fully automated metrics are scalable but weak proxies for synthesis quality.

**Revisit when.** Enough user cases exist to build an anonymized representative evaluation set.

### 8.4 Tests assert compatibility and failure semantics, not only happy paths

**Decision.** The suite covers legacy checkpoints, destination checkpointing before failure, export-only resume, skip fingerprints, stale error removal, formula safety, limits, and unrelated-tab preservation.

**Benefits.** The most expensive regressions—duplicate remote files, repeated LLM calls, silent stale data, and spreadsheet injection—are protected explicitly.

**Costs and risks.** Tests mirror implementation details and can make refactoring slower. Edge-case breadth increases maintenance and runtime.

**Alternatives.** Fewer end-to-end tests are easier to maintain but leave recovery contracts implicit. Property-based/state-machine testing could cover more combinations with additional complexity.

**Revisit when.** The state machine grows enough that example tests no longer cover important sequences.

## 9. Source-control and delivery decisions

### 9.1 Feature branch isolation and stacked-then-retargeted PR workflow

**Decision.** Spreadsheet work began on `codex/issue-8-spreadsheet-writer` from the then-unmerged synthesis branch, was initially intended as a stacked PR, then rebased onto `main` after synthesis merged and opened against `main`.

**Benefits.** Spreadsheet work did not pollute the synthesis branch. Development could proceed without waiting, and the final PR contains only issue-specific changes against the now-merged dependency.

**Costs and risks.** Stacked work requires careful base selection, retargeting, rebase conflict resolution, and repeated tests. Rebasing rewrites commit identity and can complicate collaboration if others pulled the branch.

**Alternatives.** Wait for the dependency PR to merge, which is simpler but slower; merge the dependency branch, which preserves history but leaves noisier ancestry; use a formal stacked-PR tool, which adds workflow dependency.

**Revisit when.** Multiple contributors frequently stack changes or dependency chains become deeper than one PR.

### 9.2 One implementation commit followed by focused documentation updates

**Decision.** The feature was delivered as a cohesive implementation commit, with this durable tradeoff record added to the same PR.

**Benefits.** The branch is easy to rebase and the feature can be reviewed as one conceptual unit. Documentation travels with the behavior before merge.

**Costs and risks.** A large commit is harder to review incrementally, bisect internally, or partially revert. Later documentation commits can make the branch less perfectly atomic.

**Alternatives.** Layered commits by model/projector/adapters/integration/tests aid review but require disciplined dependency ordering. Multiple PRs shrink diffs but delay an end-to-end usable feature.

**Revisit when.** PR size repeatedly slows review or multiple maintainers work on separate layers.

### 9.3 Generated artifacts, secrets, checkpoints, logs, and evaluations are ignored

**Decision.** Git tracks source, documentation, configuration defaults, tests, and an empty data-directory marker, but excludes runtime outputs and local/private material.

**Benefits.** Secrets and tokens are less likely to leak, binary churn stays out of reviews, checkpoints do not expose research topics or abstracts, and repositories remain small.

**Costs and risks.** Gitignore cannot protect secrets already force-added or stored under unexpected names. Generated examples and live evaluation evidence are unavailable to reviewers unless shared separately. Ignored local artifacts can be lost.

**Alternatives.** Commit sanitized fixtures/golden outputs, use Git LFS, or store artifacts in CI/object storage. Each needs retention and privacy policy.

**Revisit when.** Reproducible releases need versioned example outputs or team evaluation requires a shared artifact store.

### 9.4 `tradeoffs.md` is an explicit gitignore exception

**Decision.** Root `tradeoffs.md` is tracked and `.gitignore` contains `!tradeoffs.md` to state that durable decision reasoning is not private scratch material.

**Benefits.** The policy is visible beside ignore rules, future broad Markdown or documentation ignores are less likely to hide the file, and decision changes are reviewable.

**Costs and risks.** The negation is currently redundant and may confuse readers because no preceding pattern ignores the file. A negation cannot re-include a file whose parent directory is itself ignored without also re-including the parent.

**Alternatives.** Track the file without a gitignore entry, which is technically sufficient but does not satisfy an explicit ignore-policy marker. Place it under `docs/`, which improves organization but weakens root discoverability.

**Revisit when.** Repository documentation conventions settle on a dedicated ADR directory or the explicit exception proves more confusing than helpful.

### 9.5 Documentation beside code rather than an external wiki

**Decision.** README, architecture, learning, evaluation, and tradeoff documentation live in Git.

**Benefits.** Documentation versions with code, works offline, is reviewable in PRs, and cannot silently describe a different release if maintained properly.

**Costs and risks.** Detailed documents can become stale, increase review burden, and duplicate facts. Markdown lacks enforcement that code changes update corresponding reasoning.

**Alternatives.** Wikis are easy to edit but drift from releases. Formal ADR files give better decision granularity but add ceremony. Generated reference docs reduce duplication but do not capture rationale.

**Revisit when.** Staleness becomes frequent; add PR templates or checks linking architectural changes to documentation.

## 10. Deliberate current non-goals

### 10.1 No dashboards, charts, spreadsheet templates, or generated formulas

**Tradeoff.** Omitting presentation-heavy features keeps adapters safe, portable, and focused on faithful data delivery, but users must perform their own visualization and the output is less immediately executive-friendly.

### 10.2 No mandatory live Google test

**Tradeoff.** Contributors can work without accounts or credentials, but remote API compatibility is established by mocks and manual/optional runs rather than continuously proven.

### 10.3 No multi-backend fan-out in one run

**Tradeoff.** Retry and status semantics stay simple, but users cannot atomically request both a local archive and collaborative sheet.

### 10.4 No Drive lifecycle management

**Tradeoff.** OAuth remains least-privileged, but Crusoe cannot organize, share, discover, or delete created spreadsheets.

### 10.5 No synthesis prompt/schema redesign in the spreadsheet feature

**Tradeoff.** The output work integrates with the merged synthesis contract without reopening model-quality scope, but spreadsheet information architecture remains constrained by that upstream schema.

### 10.6 No database, server, scheduler, or queue

**Tradeoff.** Local operation has minimal infrastructure and clear ownership, but concurrency, durable job management, team access, and automated retries are limited.

## 11. Known asymmetries and implementation constraints

These points prevent the high-level architecture from being mistaken for a stronger guarantee than the current code provides.

### 11.1 Resume identity is not equally strong for every stage

Synthesis work and spreadsheet payloads are fingerprinted, but discovery and curation primarily use presence-based completion. Changing discovery limits, curator batching, or related configuration while resuming can reuse older upstream results. The current choice prioritizes backward compatibility and simple checkpoints over complete run provenance.

### 11.2 Checkpoint durability is weaker than XLSX durability

XLSX uses sibling temporary files and replacement; `PipelineState.save()` writes checkpoint JSON directly and has no file lock. An interrupted or concurrent checkpoint write can therefore corrupt or race even though workbook replacement is protected. The current local, single-run assumption keeps implementation small but must not be read as transactional durability.

### 11.3 Google managed-tab ownership is name-based

Reserved names identify managed tabs; there is no hidden ownership marker, schema version, or remote revision check. A colliding user tab can be overwritten, and concurrent edits to managed tabs are not detected. Conversely, unrelated and obsolete/default tabs are preserved rather than cleaned up.

### 11.4 Spreadsheet parity is semantic, not pixel-identical

Both adapters consume the same values and tab schemas, subject to different cell limits. Styling, hyperlink behavior, width calculation, and client rendering differ. Tests prove XLSX read-back and Google request construction, not identical presentation in Excel, LibreOffice, and the live Google Sheets UI.

### 11.5 Operational success is multi-dimensional

An upstream research run may be complete even when export fails; a spreadsheet result may be marked successful even if a user later edits or deletes the destination. Callers must inspect the selected backend's `output_results` record and should not interpret a process completion message, legacy `sheet_url`, or stored payload hash as live delivery verification.

### 11.6 Gitignore is hygiene, not data protection

Ignored credentials, checkpoints, logs, outputs, and evaluations can still be read locally, synchronized by OneDrive, included in backups, sent to providers, or emitted to enabled tracing. Secret management, data retention, and telemetry policy remain responsibilities beyond Git configuration.

## Decision review checklist

Use this checklist for future changes:

1. What user or engineering problem does the decision solve?
2. Which invariant becomes stronger, and which flexibility is lost?
3. What is the failure mode, and is it fatal, retryable, or silently degraded?
4. What external service, credential, data, cost, privacy, or lock-in does it introduce?
5. How does it affect checkpoints and older runs?
6. Is behavior deterministic across retries, provider changes, and backend changes?
7. What happens with empty, malformed, very large, Unicode, multiline, or adversarial input?
8. What must be tested offline, and what can only be established live?
9. Does the default remain safe for a new local user?
10. What measurable condition should cause the project to reconsider the choice?
