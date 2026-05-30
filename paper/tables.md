# Table Plan

Current source of truth: `paper/results.md`, core-5 DeCE-RF matrix, plus the
final Core-6 design decision to add one recolor task after a seed-10 gate.

## Main Comparison

Rows:

```text
current: 5 core tasks x 4 paper-facing methods x seeds 10,11,12 = 60 complete runs
target: 6 core tasks x 4 paper-facing methods x seeds 10,11,12 = 72 runs
```

Core tasks:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
red_chair_blue or red_office_chair_to_blue_office_chair
```

Paper-facing methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

Current artifacts:

```text
experiments/support_v3_2026-05-11/core5_seed10_12_fixed_mask_metrics.csv
experiments/support_v3_2026-05-11/core5_seed10_12_fixed_mask_metrics.json
experiments/support_v3_2026-05-11/core5_fixed_mask_audit_summary.csv
experiments/support_v3_2026-05-11/core5_visual_audit_filled.csv
experiments/support_v3_2026-05-11/core5_visual_audit_summary.md
```

Headline columns:

- outside-mask L1 / RMSE
- inside-mask change
- source SSIM
- DINO/source similarity
- CLIP target-source delta, labeled carefully for removal
- internal visual audit: edit success, preservation, locality, artifact, overall

`support_v3_fixed` should not appear as a headline main-table method. It belongs
in the component ablation table.

## Ablation Table

Rows:

```text
original core-4 tasks x support_v3_fixed x seeds 10,11,12 = 12 complete runs
```

Current use:

```text
support_v3_fixed vs DeCE-RF
```

Interpretation:

- isolates feedback-updated displacement weights from fixed displacement
  weights;
- supports component evidence, not the headline claim;
- should be completed for any promoted final task set before submission.

## Expansion Tables

The only planned main-table expansion beyond core-5 is one recolor task gated
by seed-10 visual inspection:

```text
preferred: red_chair_blue
fallback: red_office_chair_to_blue_office_chair
```

Weak replacement/removal candidates should move to a limitation/stress table
rather than being tuned into the main table.

## Extension Probe Tables

Report these separately from the main Core-6 comparison:

```text
laptop_remove_sticker: high-confidence completion-prior removal extension
whiteboard_probe_red_star_sticker: non-glyph replacement target-formation probe
```

Do not aggregate their metrics into the base DeCE-RF main-table mean unless the
method column explicitly names the extra route.

## Limitation / Diagnostic Table

Rows:

```text
whiteboard_remove_yellow_letter
dog_remove_tennis_ball
dog_replace_tennis_ball_star
fridge_remove_yellow_magnet
fridge_remove_peach_magnet
whiteboard_probe_blank / blue T / red A
```

Purpose:

```text
show accurate support can still fail when completion, occluded host synthesis,
precise glyph control, or replacement target formation is the bottleneck.
```

## External Baseline Table

Existing baseline artifacts are older core-4 evidence and should be reported as
availability/qualitative context unless rerun under the core-5 protocol.

Artifacts:

```text
experiments/baseline_parity_manifest.csv
experiments/baseline_summary.csv
experiments/baseline_summary.md
experiments/baseline_visual_scores_seed10_12.csv
paper/stage2_5_integrity_precheck.md
```

Recommended disclosure columns:

- runnable status
- complete rows
- failed rows and concrete reason
- model family / backbone
- seed-matching caveat
- whether the baseline used text-only, automatic masks, manual masks, or
  same-support masks
