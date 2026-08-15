# Reviewer revision round 5

This round addresses the remaining measurement concerns without regenerating
the locked forward candidates.

- `analyze_locked_cost_breadth.py` surfaces the already defined core
  equivalence metric, primary generation latency and token cost, and distinct
  gold coverage under exact, core, component-identical, and thresholded
  bipartite matching.
- `audit_registered_normal_form.py` validates the study-registered
  benchmark-normal-form checker against all canonical test golds and seven
  explicitly counted negative perturbation families.
- `run_eligible_reranker.py` constructs a new all-TAF pool from direct-basic,
  direct-rich, and staged-rich candidates. A candidate must be both renderer
  executable and study-registered-normal-form compliant. The full and
  leave-self-out variants use the same current reranker snapshots and require
  exactly `min(5, pool size)` returned IDs.
- `analyze_eligible_reranker.py` reports pool coverage, fixed RRF and LLM
  ordering, returned-prefix completeness, and candidate-source ancestry.
- `replay_forensic_record.py` independently recomputes the headline forensic
  ranking and step-6 containment summaries from the released retained record.
  It is deliberately labelled a replay, not a reconstruction of the missing
  original constructor source.

The new reranking protocol is recorded before inference in
`design/eligible_reranking_protocol.json`. It is a post-review sensitivity
campaign on the previously locked development sample, not an untouched final
evaluation set.

All four reranker cells completed 150/150 case rows. Full pools are empty for
one case; the Mistral and Qwen leave-self-out pools are empty for four and
three cases. Every nonempty response parsed without error and returned exactly
the required prefix. In the full eligible pool, Mistral and Qwen differ from
RRF at Hit@1 by -4.73 and -2.70 percentage points. The corresponding
leave-self-out differences are -0.02 and +0.03 points; all four finite-
population-adjusted intervals include zero.
