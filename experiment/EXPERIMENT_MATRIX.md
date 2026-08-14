# Reviewer experiment matrix

| Reviewer concern | Required evidence | Status | Canonical artifact |
|---|---|---:|---|
| Central result cannot be reconstructed | Reproduce legacy heuristic, learned, and oracle metrics | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Six reasoning fields/provenance unclear | Define every field and audit origin/use | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Step 6 may contain the answer | Measure set equality and gold containment on every split | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Relative contribution of six fields unclear | Retrain and evaluate heuristic/learned/oracle after removing each answer field | Complete | `out/a/step_ablation/summary.csv` |
| Candidate-construction stages hidden | Counts before expansion, after expansion, after deduplication, and cap-15 | Complete | `outputs/legacy_audit/legacy_pool_stages.csv` |
| Claimed validation stage unclear | Source audit plus explicit implemented/not-implemented finding | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Algorithm/Table 1 inconsistent | Distinguish complete pool, cap-15, heuristic top-5, learned top-5 | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Heuristic under-specified | Exact formula, weights, normalization, and tie breaks | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Need complete-pool oracle and coverage@K | Fixed-pool Hit/Recall/Precision/F1 curves through K=15 | Complete | `outputs/legacy_audit/legacy_audit_summary.json` |
| Need uncertainty for ranking gain | Paired bootstrap intervals and exact McNemar tests | Complete | `out/a/legacy_pair/paired_summary.csv` |
| Need a full worked example | Query, schema, six fields, candidate stages, scores, and gold flags | Complete | `out/w/e.json` |
| Direct-IR syntax/scoring/reproducibility unclear | Audit saved predictions, parser, configurations, and checkpoint provenance | Complete; branch excluded | `outputs/direct_ir/audit.json` |
| Binary labels ignore near-valid charts | Component-level graded match beside exact match | Analyzer complete; forward outputs running | `analyze_forward.py` |
| Need forward-predicted trace from query+schema | Direct versus six-stage forward condition without benchmark steps | Running | `outputs/campaign_status.json` |
| Single model family insufficient | Qwen, Llama, Gemma, Mistral comparison; DeepSeek/Phi smoke exclusions | Running | `design/smoke_decisions.json` |
| Stability unknown | Three temperature-0.2 seeds and pairwise rank/set stability | Queued in campaign | `run_campaign.py`, `analyze_stability.py` |
| Deployable reranking unknown | Valid pooled candidates, fixed RRF, LLM reranker, complete-pool oracle | Runner/analyzer complete; queued after generation | `run_forward_reranker.py`, `analyze_reranker.py` |
| Expert-method evidence | Real expert files only; legacy synthetic ratings excluded | Deliberately deferred by user | `README.md` |

“Complete” here means the machine artifact exists and has passed local checks.
No manuscript claim is final until the forward campaign and reranker analyses finish.
