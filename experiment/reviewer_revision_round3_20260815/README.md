# Reviewer revision round 3 (2026-08-15)

This directory contains the evidence added in response to the third major-review round.

## Locked reranking extension

`design/locked_reranking_protocol.json` was written before reranking outcomes were generated. It transfers the unchanged historical RRF and LLM-reranking protocol to the 150-case locked development holdout. Direct-only, staged-only, and union pools use validator-accepted candidates from the same four generator families. Qwen and Mistral rerankers use temperature 0 and seed 77.

All six locked runs contain 150 rows. Empty valid pools occur in 5 direct, 20 staged, and 1 union case; no other inference error occurs. Design-weighted complete-pool oracles are 14.99%, 29.36%, and 32.08%. Qwen improves Hit@1 over RRF by 4.07 points on the direct pool and 4.09 on the staged pool, but falls by 10.20 on the union. Mistral changes by +2.72, -1.34, and -9.51 points. Both union intervals exclude zero in the negative direction.

## New measurement analyses

The locked-holdout analysis now reports top-one validity, valid-candidate fraction, validity at ranks 3 and 5, and separate mark, channel, field, operation, and filter fidelity. Design-weighted uncertainty uses registered stratum inclusion probabilities and finite-population-adjusted replicate deviations for the fixed 720-record inference population.

## Cross-release lineage audit

`audit_cross_release_overlap.py` compares nvBench v1 with all committed nvBench 2.0 splits using database IDs, normalized queries, database--schema pairs, and database--analytic signatures derived from VQL/specification components. The output establishes that the 105-case nvBench-v1 analysis is a same-lineage cross-release consistency check, not independent external validation.

## Public-workflow source audit

`audit_public_nvbench2_workflow.py` reads the immutable public repository commit through `git show`. It verifies that the checked SFT inference path formats per-instance prompts from query and schema fields with an empty output placeholder. Gold answers are consumed only by post-prediction evaluation. This is a source-level positive control for the provenance protocol, not a reproduction of the published model scores.
