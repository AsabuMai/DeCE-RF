# Experiments

This directory now keeps only current experiment packs at the top level.

## Current Pack

The active support-v3 pack is:

```text
support_v3_2026-05-11/
```

Key files:

```text
support_v3_refinement_metrics_clip.csv
mug_candidate_compare_metrics_clip.csv
removal_controller_compare_metrics_clip.csv
README.md
```

## Archived Summaries

Older flat experiment summaries and metrics were moved to:

```text
archive_legacy_2026-05-11/
```

They are retained for traceability but should not be used as the current
project entry point.

## Output Runs

Raw generated images and per-run records live under:

```text
outputs/
```

Only use a run as evidence if the folder contains:

```text
result.png
stats.json
metadata.json
command.txt
```
