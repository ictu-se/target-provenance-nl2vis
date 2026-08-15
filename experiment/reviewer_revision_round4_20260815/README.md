# Reviewer revision round 4

This round addresses the reviewer request to separate JSON parseability,
Vega-Lite renderer execution, and registered benchmark-grammar compliance. It
also audits the retained locked reranker records to document complete-pool
exposure and output handling. No candidate generation or model rerun was
performed; both analyses use previously locked outputs.

`audit_renderability.py` executes every retained candidate from the 150-record
development holdout through `vl-convert-python` 1.9.0. The renderer receives
the source table (at most 200 rows) and only renderer-required metadata. Marks,
fields, aggregates, transforms, filters, and candidate order are unchanged.
The analysis covers 1,800 model-condition rows and 4,703 candidates.

`audit_reranker_protocol.py` verifies that every nonempty reranker prompt saw
the complete candidate pool. Pools reach 16 direct, 11 staged, and 21 union
candidates. The five-item limit applies only to the returned prefix. All
nonempty responses parse and contain no unknown or duplicate ID; omitted IDs
are not backfilled.

The round does not estimate a causal effect of target exposure. The historical
internal prototype and the later forward generators are different mechanisms,
and the original prototype repository is not part of the public artifact. The
manuscript therefore treats the 96.14% result only as a controlled forensic
diagnostic and makes the locked metric-decomposition result central.
