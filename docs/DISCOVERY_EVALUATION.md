# Discovery selection and evaluation

Discovery searches the first keyword of every cluster, then the second keyword
of every cluster, and so on. Each configured keyword is searched once, including
after enough candidates exist to fill the selected corpus. Failed queries are
recorded as nonfatal state errors; the remaining queries still run.

The request budget is `sum(len(cluster.keywords))` logical searches, each requesting
`min(results_per_query, 100)` results. The transport wrapper may retry a logical
search up to three times. The paper cap limits the selected corpus and downstream
LLM work; it does not limit candidate retrieval. Four clusters with three keywords
at 20 results per query use 12 logical searches and 240 requested result slots,
even with a 40-paper cap.

Within each cluster, candidate order follows query round and service rank.
Selection takes one unselected paper from each cluster per round. With disjoint
productive clusters, every cluster receives an initial `floor(cap / clusters)`
share. Exhausted clusters surrender unused slots to the remaining clusters.
Shared paper IDs occupy one global slot. Input cluster order breaks ties for
remainder slots. A cap smaller than the cluster count still searches all clusters,
but cannot guarantee a distinct selected paper for each one. Output order is
balanced retrieval order, not a relevance score; Paper Curator ranks it later.

Selected papers retain `_source_cluster` (the first observed theme) for compatibility
and `_discovery_sources`, an ordered list of unique `{"cluster": theme, "query": keyword}`
pairs. This includes duplicate discoveries in later queries, even after enough
candidates exist to fill the cap. Checkpoints and curated records retain these
fields. Sources outside the configured searches cannot be recorded.

## Compare retrieval quality

The regression suite establishes selection fairness and provenance, not real-world
recall. Prepare several topics with different subtopics and manually curate relevant
Semantic Scholar paper IDs **before examining the two retrieval outputs**. Record
criteria, paper titles/links, reviewer, judgment date, and subtopic assignments
alongside saved cases. Include difficult and overlapping subtopics.

For each topic:

1. Freeze the decomposition and API responses from the same time window.
2. Fix the requested-result-slot budget and selected-paper cap for both arms.
   Count duplicates and empty slots against retrieval effort. For example, four
   clusters with three keywords at four results per query use 48 slots; a direct
   topic query requesting 48 results uses the same budget. Both arms can select
   up to 40 unique valid papers. This fits the API's 100-result maximum without
   pagination. Compare several fixed budgets before drawing broader conclusions.
3. For direct search, preserve service order, remove invalid records, deduplicate
   by paper ID, and truncate to the cap. For decomposition, use `discovery.run`.
   Save each corpus in output order and the actual request limits. Record failed
   searches; the scorer rejects failed or unequal-budget comparisons.
4. To evaluate curator ranking, curate both corpora with identical settings and
   save curator order. Label the ranking in accompanying notes. Do not mix service
   ranking in one arm with curator ranking in the other.
5. Run the offline scorer:

   ```bash
   python scripts/evaluate_discovery.py data/evaluations/discovery-cases.json --top-k 10
   ```

Input is a JSON array, one object per topic. This miniature example shows the
schema only; its IDs and results are synthetic, not a research benchmark:

```json
[
  {
    "topic": "Example topic",
    "max_total_papers": 4,
    "relevant_by_subtopic": {"Subtopic A": ["a", "b"], "Subtopic B": ["c"]},
    "direct": {
      "request_limits": [12],
      "papers": [{"paperId": "a"}, {"paperId": "noise"}],
      "errors": []
    },
    "decomposed": {
      "request_limits": [2, 2, 2, 2, 2, 2],
      "papers": [{"paperId": "c"}, {"paperId": "b"}, {"paperId": "a"}],
      "errors": []
    }
  }
]
```

Metrics are relevant-paper recall (unique relevant retrieved / unique gold papers),
precision at K (relevant in the first K unique results / K, treating absent ranks
as misses), and subtopic coverage (subtopics with a relevant retrieved paper /
curated subtopics). A gold paper can belong to multiple subtopics but contributes
once to recall. Unjudged papers count as nonrelevant, so precision is relative to
the supplied judgments. Report per-topic results and gold set sizes; small curated
samples do not establish population-level quality.

No live quality comparison or independently judged benchmark is bundled with this
fix. Offline tests use explicit synthetic judgments to verify arithmetic and budget
enforcement; they are not evidence of relevance gains on real topics.
