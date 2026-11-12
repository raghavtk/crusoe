# Human review of synthesis claims

PR #20 adds an offline review step after synthesis. It pairs each evidence-bearing
landscape statement with the full stored abstracts of the papers cited by that
statement. You decide whether the claim is supported, unsupported, or uncertain.
This does not call a provider, fetch new papers, edit the pipeline checkpoint, or
rewrite model output.

For a legacy checkpoint, the commands below work unchanged. For a new isolated run,
substitute its `data/runs/RUN_ID/checkpoint.json` path.

```bash
python scripts/review_claims.py data/session_checkpoint.json --reviewer Raghav
```

For each item, the terminal displays its section, claim text, model confidence
where available, cited paper IDs/titles, and abstracts. Enter `1` for supported,
`2` for unsupported, `3` for uncertain, `s` to skip, or `q` to stop. A judgment
is confirmed after the optional note prompt. Each confirmed judgment is saved
immediately. EOF or Ctrl+C preserves prior confirmed judgments. An unfinished
choice/note is not saved.

The default artifact is `data/session_checkpoint.claim-review.json`. Use a distinct
path per run or reviewer if you need independent assessments:

```bash
python scripts/review_claims.py data/session_checkpoint.json --output data/evaluations/review-raghav.json --reviewer Raghav
python scripts/review_claims.py data/session_checkpoint.json --output data/evaluations/review-raghav.json --summary
python scripts/review_claims.py data/session_checkpoint.json --output data/evaluations/review-raghav.json --revisit
```

Running the same command again resumes at unreviewed items. `--revisit` includes
already reviewed items and lets you replace their current judgments. The artifact
keeps the latest judgment per claim, not an append-only revision history. `--summary`
reports supported, unsupported, uncertain, and unreviewed counts without prompting.
These are human-label counts, not automatic accuracy measurements.

## What you are judging

The existing synthesis contract cites paper IDs. It does not store a selected
quotation aligned to each claim. The reviewer therefore labels abstracts as
**cited source context**, not as proven supporting quotations. It shows exact full
stored abstract text and never substitutes the curator's generated summary.
Missing source records and missing abstracts are explicit.

Synthesis sees normalized excerpts capped at 1,200 characters. The reviewer can
see text beyond that cap. A review can judge support in the stored abstract; it
cannot reconstruct exactly what an older provider saw or establish full-paper
support. A claim may contain multiple statements. Mark it supported only if the
material assertions are supported under your review policy; use the note for
partial support or qualifications. Absence from an abstract is often uncertain,
not proof that a claim is false.

Review items include theme and gap explanations, future-work recommendations and
rationales separately, methodology observations, shared limitations, disagreement
positions, and disagreement interpretations. Each position keeps its own citations.
An interpretation receives the union of its positions' citations. This is derived
context, not an independently generated citation alignment.

Summary prose and the compatibility theme/gap/future-work lists lack claim-level
citations. The CLI names them as outside this alignment workflow. Reading-order
reasons are also outside scope because they are recommendations about reading,
not landscape evidence claims. Older checkpoints without a `landscape` fail with
a clear message rather than receiving invented citations.

## Data ownership and issue #15

The review JSON owns a versioned snapshot of the topic, review items, exact cited
source text, and a SHA-256 of the complete synthesis. It includes a snapshot
fingerprint plus per-claim judgments, reviewer names, notes, and UTC timestamps.
It can be inspected independently as an evaluation artifact. It does not train a
model or automatically feed annotations into another synthesis run.

Changes to the topic, synthesis, citation lists, or displayed source evidence
invalidate resume. The CLI preserves the original review and requires a different
output path. It does not silently carry a supported label onto changed evidence.
Malformed artifacts, checksum mismatches, and unknown claim judgments fail closed.
The checksum detects accidental corruption; it is not an authenticity signature.

Writes use a sibling temporary file, flush/fsync, then atomic replacement. Failure
before replacement preserves the last artifact. The CLI holds an exclusive
`<output>.lock` throughout a review session to prevent two interactive writers
from losing each other's edits. Normal exit, EOF, and Ctrl+C release it. A killed
process can leave a lock behind. Verify the owning process has stopped before
manually removing that specific lock; the CLI never guesses that a lock is stale.
This is local single-writer coordination, not distributed transactional storage.

CLI persistence never uses `PipelineState.save()` and never targets the checkpoint
itself. Terminal control characters in source data are escaped before display.

Issue #15 separately adds curator/discovery partial-call reuse, atomic pipeline
checkpoints, run isolation, prompt-boundary hardening, and verification of missing
or modified local export files. See [the recovery guide](RECOVERY_HANDOFF.md).
Review annotations remain outside generated workbooks so export cannot erase them.

## Design choice

The current application is a Python CLI with spreadsheet export. A terminal review
command provides a usable human evaluation loop without adding a server, browser
session, or paid generation step. Editable spreadsheet annotations would need an
import/merge contract to survive export recovery. A web interface could improve
navigation later; it should reuse the snapshot/judgment contract rather than own a
second copy of the feedback. Fine-grained quoted-span selection is also a possible
extension. The first version deliberately preserves the existing synthesis schema
and cache identity.

Tests drive the CLI with private checkpoint/review files, exercise all three
labels, interrupt after two confirmed judgments, and verify that resume asks only
for the remaining claim. Filesystem tests cover failed replacement, artifact
corruption, stale source data, and concurrent-writer exclusion. No live API quota
or model judgment is needed to validate these behaviors.
