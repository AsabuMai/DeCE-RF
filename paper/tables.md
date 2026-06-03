# Table Plan

Current source of truth: `paper/wacv_experiment_design.md` for experiment scope and claim boundary, and `paper/results.md` for the current strict Core-6 / E2.2 evidence readout. Archived server results are diagnostic only.

## Main Comparison

Rows:

```text
revised strict Phase 1: 6 task instances x 4 paper-facing methods x seeds 10,11,12 = 72 complete runs
```

Server task instances:

```text
cat_crown
bowl_apple_inside
tshirt_star
red_chair_blue
pillow_vertical_fabric_strip
backpack_remove_toy_charm
```

Mapping into the updated Core-6 taxonomy:

```text
T1 attached accessory: cat_crown canonical; dog_sunglasses diagnostic only
T2 container-constrained insertion: bowl_apple_inside canonical
T3 surface decal: tshirt_star canonical; mug_heart diagnostic only
T4 object-level recolor: red_chair_blue canonical
T5 surface material strip editing: pillow_vertical_fabric_strip canonical; DeCE-RF passes human review with a perspective-aligned blue silk strip
T6 exposed-object removal: backpack_remove_toy_charm canonical
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
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.md
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.md
```

Headline columns:

- outside-mask L1 / RMSE
- inside-mask change
- source SSIM
- DINO/source similarity
- CLIP target-source delta, labeled carefully for removal and recolor
- internal visual audit: edit success, preservation, locality, artifact, overall

`support_v3_fixed` should not appear as a headline main-table method. It belongs in the component ablation table.

## Ablation Table

Rows:

```text
6 server task instances x support_v3_fixed x seeds 10,11,12 = 18 complete
fixed-control runs; compare against the matching DeCE-RF rows in
`core6_fixed_control_metrics.csv`
```

Current use:

```text
support_v3_fixed vs DeCE-RF
```

Interpretation:

- isolates feedback-updated displacement weights from fixed displacement weights;
- supports component evidence, not the headline claim;
- is not yet the final E4 evidence for the revised strict Core-6; use it as a
  component baseline until the matched controller/Pareto runs are complete.

## Expansion Tables

The completed server evidence table includes one recolor task gated by seed-10
visual inspection and a corrected fixed evaluation mask:

```text
selected server expansion: red_chair_blue
updated T2: bowl_apple_inside, now included in strict Phase 1
updated T5: pillow_vertical_fabric_strip, now included in strict Phase 1 after replacing the earlier pillow_blue_stripes probe
```

Weak replacement/removal candidates should move to a limitation/stress table
rather than being tuned into the main table. `backpack_remove_toy_charm` remains
usable as T6 target-removal evidence with a local zipper/fabric preservation
caveat.

## Extension Probe Tables

Report these separately from the main comparison:

```text
laptop_remove_sticker: high-confidence completion-prior removal extension
whiteboard_probe_red_star_sticker: non-glyph replacement target-formation probe
```

Do not aggregate their metrics into the base DeCE-RF main-table mean unless the method column explicitly names the extra route.

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
show accurate support can still fail when completion, occluded host synthesis, precise glyph control, or replacement target formation is the bottleneck.
```

## Same-Support Removal Diagnostic

A removal-only diagnostic has been generated for `backpack_remove_toy_charm`
using Telea and Navier-Stokes OpenCV inpainting with the DeCE-RF support mask.
Report it separately from the main comparison because it receives the same
support mask and only applies to removal/fill cases.

Artifact status: this is a legacy diagnostic row. The active repository no
longer tracks summary CSV/JSON files for this run; generated images remain under
ignored generated same-support inpaint output directories and should
not be cited as active paper evidence unless regenerated and summarized.

Readout: same-support inpainting gives lower outside drift on the backpack case but produces visible fill artifacts around the strap/zipper region, while DeCE-RF removes the target charm but locally smooths the occluded zipper/fabric.

## E2 Fairness Tables

Baseline gap audit and external-baseline protocol:

```text
experiments/support_v3_2026-06-02/e2_baseline_download_registry.csv
experiments/support_v3_2026-06-02/e2_baseline_runnable_validation.csv
experiments/support_v3_2026-06-02/e2_baseline_audit.md
```

E2 is no longer a flat external-baseline leaderboard. It is a layered fairness
comparison:

```text
E2.1 backbone calibration
E2.2 same-backbone SD3 algorithm comparison
E2.3 native preservation-aware RF comparison
E2.4 support-matched diagnostic
```

### Table 2a: Same-Backbone SD3 Algorithm Comparison

This is the main E2 algorithmic table.

Current completed artifacts:

```text
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

Rows:

```text
direct_target-SD3
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
Fixed DeCE-SD3, when strict fixed cache is joined
DeCE-RF-SD3
```

Optional row only if genuinely same-backbone:

```text
OT-RF/OTIP-SD3 or RF-Edit-SD3
```

Required columns:

```text
method | backbone | input condition | support used for control | edit success |
preserve fidelity | excess preserve error | leakage/locality | NFE/runtime | caveat
```

### Table 2b: Calibration And Native Preservation-Aware RF Context

This table has two blocks.

Calibration block:

```text
SD3 reconstruction/direct-target floors
FLUX reconstruction/direct-target floors if FLUX access and adapters are valid
```

Native preservation-aware block:

```text
RF-Solver-Edit / RF-Edit
ReFlex
FireFlow
stable-flow
OT-RF / OTIP-style
DVRF / Delta Velocity RF
```

Rows that are blocked by FLUX.1-dev access or adapter gaps should remain in the
status table with exact failure reasons. Do not silently drop them and do not
replace them with unrelated non-RF baselines.

Caption requirement:

```text
Backbones and input conditions differ across native rows. This table evaluates
whether off-the-shelf preservation-aware RF editors solve the localized
edit-preserve tasks in practice; algorithm-level conclusions are drawn from the
same-backbone SD3 comparison.
```

### Table S-E2: Support-Matched Diagnostic

This supplement table answers whether DeCE-RF is only winning because it has a
support mask.

Rows:

```text
direct_target + same M_edit diagnostic
FlowEdit + same M_edit diagnostic if wrapper is stable
optional native preserve-aware + same M_edit if method supports it
Fixed DeCE
DeCE-RF
```

Use only a binary edit support for baseline diagnostic rows. Do not give baseline
rows DeCE-RF's `M_core`, `M_contact`, `M_preserve`, feedback weights, or
projection. Output blending must be labeled as diagnostic, not as a fair main
baseline.

### Legacy Baseline Artifacts

Legacy baseline artifacts are historical audit material only:

```text
experiments/archive_legacy_2026-05-11/baseline_parity_manifest.csv
experiments/archive_legacy_2026-05-11/baseline_summary.csv
experiments/archive_legacy_2026-05-11/baseline_summary.md
paper/archive_old_core6_20260602/old_stage2_5_integrity_precheck.md
```

Do not use legacy Core-4 or old Core-6 artifacts as active paper evidence unless
they are rerun under the current strict protocol.
