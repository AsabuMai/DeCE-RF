# Stage 2.5 Integrity Precheck

Date: 2026-05-10

Scope: archived baseline evidence package after the 2026-05-10 external-baseline run.

This is a precheck for entering academic-pipeline Stage 2.5. It is not the final
paper integrity report. The full Stage 2.5 check should run after the paper
draft is assembled.

## Artifact Coverage

Baseline manifest:

```text
experiments/archive_legacy_2026-05-11/baseline_parity_manifest.csv
```

Status:

```text
72 manifest rows
48 complete rows
24 failed rows with concrete failure reasons
```

Evidence package:

```text
experiments/archive_legacy_2026-05-11/baseline_summary.csv
experiments/archive_legacy_2026-05-11/baseline_summary.md
experiments/archive_legacy_2026-05-11/baseline_visual_score_template.csv
experiments/archive_legacy_2026-05-11/baseline_visual_scores_seed10_12.csv
```

Paper figure candidates:

```text
The old paper-figure candidates are not retained as active artifacts; regenerate
figures from the current `experiments/support_v3_2026-06-02/` package if needed.
```

## Integrity Notes

- FlowEdit and SplitFlow are SD3-family qualitative baselines.
- FireFlow and RF-Solver-Edit are FLUX-dev qualitative baselines, not
  SD3-matched runs.
- RF-Solver-Edit public image-editing script does not expose seed control.
- ReFlex is recorded as failed under available 24GB GPU and public-code
  constraints.
- SteerFlow is recorded as failed because no public runnable code was found as
  of 2026-05-10.
- `experiments/archive_legacy_2026-05-11/baseline_visual_scores_seed10_12.csv` is an internal visual
  audit, not a user study.

## Gate Result

Baseline evidence package is ready to be incorporated into the Stage 2 draft.

Full Stage 2.5 integrity verification is still pending because it must check the
assembled manuscript, including citation context, claim wording, figure/table
references, and all limitations.
