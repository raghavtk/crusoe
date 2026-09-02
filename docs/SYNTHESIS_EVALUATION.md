# Synthesis Evaluation Guide

Automated validation proves that an output is structurally safe and traceable. It does not prove
that the literature review is academically correct. Use this rubric on saved live-test artifacts.

## Deterministic Gates

An output fails before human scoring if it contains an unknown/duplicate paper ID, mismatched
title, failed-curator citation, unsupported reducer citation, malformed schema, contradictory
legacy fields, too many calls, or an unhandled provider error.

## Human Rubric

Score each category from 1 (poor) to 5 (excellent):

- **Grounding:** Do cited abstracts and curator summaries actually support the associated claim?
- **Theme coherence:** Do themes combine related evidence without becoming vague or duplicative?
- **Gap quality:** Are gaps genuine limitations exposed by the corpus rather than generic wishes?
- **Method and disagreement fidelity:** Are methods classified sensibly, and are disagreements
  represented only when the supplied evidence supports competing positions?
- **Reading-order usefulness:** Does the sequence help a new reader build understanding, rather
  than merely repeat deterministic priority order?

Record one short justification per score and flag any claim that needs full-paper verification.
An initial merge target is no deterministic gate failures, no category below 3, and an average of
at least 4 across two independent reviewers.

## Repeated Evaluations

Do not compare exact prose. Compare stable properties: cited paper sets, major themes, identified
gaps, reading-order structure, repair rate, latency, and rubric scores. Live evaluation artifacts
are written under ignored `data/evaluations/` so potentially sensitive abstracts and model output
are not committed accidentally.
