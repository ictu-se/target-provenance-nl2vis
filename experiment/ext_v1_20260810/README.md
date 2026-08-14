# nvBench v1 external transfer set

This directory contains the deterministic 105-record, seven-family external
transfer set used by the main manuscript. It is a single-gold test of forward
chart construction, not a multi-intent ambiguity benchmark.

- Source: local public nvBench v1 release (`NVBench.json` plus SQLite databases)
- Sampling seed: `20260810`
- Allocation: 15 records per released chart family
- Input: first released NL paraphrase plus schema fields, examples, and counts
- Gold conversion: VQL/visualization object to the manuscript's canonical grammar
- Exclusions: transformations or ordering not faithfully representable by that grammar

`selection_manifest.json` records the exact counts and scope. `test.json`
contains the source IDs, SQL, hardness, schema, and converted target used by the
runner. Outputs are generated outside the long manuscript path to avoid the
legacy Windows path-length limit, then copied back with their analysis manifests.

The earlier preflight with omitted sort semantics is explicitly excluded and
retained at `_runs/p01_extv1/aborted_sort_omission/` for auditability.
