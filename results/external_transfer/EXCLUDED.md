# Excluded external-adapter campaign

The files below this directory preserve an earlier nvBench-v1 conversion and
Qwen direct-versus-staged campaign for audit history only. The converter could
silently omit unsupported limits, binning, ordering, and distinct aggregation.
Consequently, none of its estimates is used in the manuscript, abstract,
figures, or conclusions.

The admissible external evidence is the strict-v3 campaign under
`results/reviewer_revision_round2_20260814/external_v3/`. Its adapter either
represents a released operation explicitly or excludes the source record with a
recorded reason, and it includes a seeded 21-case manual conversion audit.
