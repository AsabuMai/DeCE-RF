# Table Plan

Current source of truth: `paper/wacv_experiment_design.md` for the 2026-06-05
revised experiment design and claim boundary, `paper/results.md` for the
synchronized readout, and
`experiments/support_v3_2026-06-02/e2_evidence_lock_2026-06-04.md` for the
locked E2.1-E5 artifact package. Archived server results are diagnostic only.

## Main Comparison

Rows:

```text
revised strict Phase 1: 6 task instances x 4 paper-facing methods x seeds
10,11,12 = 72 complete runs for the previous blue-strip T5 package; rerun or
separately lock the revised same-color material-panel T5 before using it as
headline evidence
```

Server task instances:

```text
cat_crown
bowl_apple_inside
tshirt_star
red_chair_blue
pillow_same_color_corduroy_panel
backpack_remove_toy_charm
```

Mapping into the updated Core-6 taxonomy:

```text
T1 attached accessory: cat_crown canonical; dog_sunglasses diagnostic only
T2 container-constrained insertion: bowl_apple_inside canonical
T3 surface decal: tshirt_star canonical; mug_heart diagnostic only
T4 object-level recolor: red_chair_blue canonical
T5 localized same-color material replacement: pillow_same_color_corduroy_panel canonical; pillow_vertical_fabric_strip diagnostic only
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
experiments/support_v3_2026-06-02/eval_masks/
experiments/support_v3_2026-06-02/normalized_512/normalized_512_manifest.csv
```

Headline columns:

- outside-mask L1 / RMSE
- inside-mask change
- source SSIM
- DINO/source similarity
- CLIP target-source delta, labeled carefully for removal and recolor
- internal visual audit: edit success, preservation, locality, artifact, overall

`support_v3_fixed` should not appear as a headline E1 main-table method. Its
stable paper identity is Fixed DeCE (same-backbone component control): in E2.2
it is the preservation-control row, and in E4 it is the fixed-displacement
component ablation.

## Table 1: Strict Core-6 Main Results

Purpose:

```text
show the main localized edit-preserve effect under fixed evaluation masks
```

Rows:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

Required caption point:

```text
The fixed evaluation mask is shared across methods and is not each method's own
support mask. The table reports a controlled diagnostic suite, not a large-scale
benchmark.
```

## Table 2a: Same-Backbone SD3 Algorithm Comparison

This is the main E2 algorithmic table.

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed10_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed11_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed12_grid.png
```

Rows:

```text
direct_target-SD3
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
Fixed DeCE-SD3 (same-backbone component control)
DeCE-RF-SD3
```

Current locked E2.2 readout covers the core analyzed rows:

```text
6 tasks x 5 core SD3 rows x 3 seeds = 90 analyzed rows
rows: direct_target, FlowEdit, FlowAlign, SplitFlow, DeCE-RF
```

The 2026-06-05 design adds Fixed DeCE-SD3 as the same-backbone
preservation-control row:

```text
+ 6 tasks x 1 row x 3 seeds = 18 rows
total planned/joined E2.2 table: 108 rows
```

Optional same-backbone row only if genuinely verified:

```text
OT-RF/OTIP-SD3 or RF-Edit-SD3
```

Required columns:

```text
method | backbone | input condition | support used for control | edit success |
preserve fidelity | excess preserve error | leakage/locality | NFE/runtime | caveat
```

Caption requirement:

```text
Algorithm-level conclusions are drawn only from same-backbone SD3 rows under
the same source images, prompts, seeds, and fixed evaluation masks.
```

## Table 2b: Native FLUX Contextual Comparison

This table is contextual, not the algorithmic leaderboard.

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e2_native_flux_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_native_flux_fixed_mask_metrics_with_context.csv
experiments/support_v3_2026-06-02/e2_native_flux_contextual_table.csv
experiments/support_v3_2026-06-02/e2_native_flux_contextual_table.md
experiments/support_v3_2026-06-02/e2_native_flux_normalized_512_manifest.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_filled.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_contextual_conclusion.md
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_grids/
```

Native preservation-aware block:

```text
RF-Solver-Edit / RF-Edit
ReFlex
FireFlow
stable-flow, if the strict adapter is complete
OT-RF / OTIP-style, if registered/runnable
DVRF / Delta Velocity RF, if registered/runnable
```

Rows that remain blocked by access, repo, environment, or adapter gaps should
stay in a status/caveat block with exact failure reasons. Do not silently drop
them and do not replace them with unrelated non-RF baselines.

Caption requirement:

```text
Backbones and input conditions differ across native rows. This table evaluates
whether off-the-shelf preservation-aware RF editors solve the localized
edit-preserve tasks in practice; algorithm-level conclusions are drawn from the
same-backbone SD3 comparison.
```

## Table S-E2: Support-Matched Diagnostic

This supplement table answers whether DeCE-RF is only winning because it has a
support mask.

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e2_support_matched_diagnostic_manifest.csv
experiments/support_v3_2026-06-02/e2_support_matched_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_filled.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_conclusion.md
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_grids/
```

Locked diagnostic rows:

```text
direct_target_raw
direct_target_mask_blend
flowedit_mask_blend
support_v3_controller_rmsgap
```

The 2026-06-05 design treats these as useful but not sufficient by themselves.
If time permits, strengthen E2.4 with inference-time same-support rows:

```text
direct_target + same M_edit gating
FlowEdit raw
FlowEdit + same M_edit gating, if wrapper is stable
Fixed DeCE
DeCE-RF
```

Interpretation:

```text
Binary localization/output blending improves preservation metrics by
construction but does not recover target correctness or boundary coherence.
Therefore, localization alone is insufficient to explain the DeCE-RF result.
```

Use only a binary edit support for baseline diagnostic rows. Do not give
baseline rows DeCE-RF's `M_core`, `M_contact`, `M_preserve`, feedback weights,
or projection. Output blending must be labeled as diagnostic, not as a fair main
baseline.

## Table 3 / Figure 4: Support Geometry Ablation

Purpose:

```text
show that support geometry is an explicit experimental object, not a hidden
hand-picked mask
```

Boundary:

```text
E3 fixes the controller and changes support geometry.
E4 fixes support and changes controller or stress axis.
```

Current locked scope:

```text
3 tasks x 6 support geometry variants x 3 seeds = 54 support-map rows
tasks: cat_crown, tshirt_star, backpack_remove_toy_charm
variants: attention_only, clean_disagreement, velocity_disagreement,
grounding_sam, generic_support, operation_conditioned_support
```

The 2026-06-05 design recommends a downstream runnable compact ablation when
compute allows:

```text
3 representative tasks x 6 support variants x 2 seeds = 36 outputs
Phase 2 breadth: 6 categories x 2 examples x 6 variants x 2 seeds = 144 outputs
```

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_mask_metrics.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_summary.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_by_task_summary.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_correlation.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_summary.md
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_tshirt_star_seed10_figure4_panel.png
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_seed10_task_sheet.png
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_complete_2026-06-04.md
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_complete_2026-06-04.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry.sha256
```

Row interpretation:

```text
attention_only, clean_disagreement, velocity_disagreement, grounding_sam =
support-map diagnostics only

generic_support, operation_conditioned_support =
runnable downstream rows with support-quality and edit/preserve metrics
```

Paper-safe claim:

```text
Operation-conditioned support improves fixed-mask overlap and downstream edit
behavior relative to weak generic support, while Grounding/SAM alone tends to
over-cover the object/host region. This supports the claim that DeCE-RF's
support geometry is not merely generic segmentation or raw attention evidence.
```

## Table 4 / Figure 5: Controller And Robustness Ablation

This section includes the component anchor and the final E4 controller/stress
package.

Boundary:

```text
E4 fixes operation-conditioned support and compares Fixed DeCE, DeCE-RF full,
and selected controller/stress variants. Support perturbation is interpreted as
controller robustness stress, not as a new support-geometry comparison.
```

Component anchor artifacts:

```text
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.csv
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact_rows.csv
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.md
```

Current locked controller/stress scope:

```text
tasks: cat_crown, tshirt_star, pillow_same_color_corduroy_panel
base rows: support_v3_fixed, support_v3_controller_rmsgap x seeds 10/11/12 = 18
stress rows: support_v3_fixed, support_v3_controller_rmsgap x edit multipliers
0.50/0.75/1.00/1.25/1.50/2.00 x seed10 x 3 tasks = 36 metric rows
```

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e4_controller_base_metrics.csv
experiments/support_v3_2026-06-02/e4_edit_strength_metrics.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_base_summary.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_edit_strength_summary.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_stats.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_summary.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_figure5_edit_strength_pareto.png
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_tshirt_star_seed10.png
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_summary.md
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_complete_2026-06-04.md
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_complete_2026-06-04.csv
experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation.sha256
```

Paper-facing identity:

```text
Fixed DeCE (same-backbone component control) = support_v3_fixed
DeCE-RF = support_v3_controller_rmsgap
```

Interpretation:

```text
Fixed DeCE displacement is a component ablation. It is not an external baseline
and not an E2.4 support-only row.
```

Paper-safe claim:

```text
Under the same SD3 implementation, fixed evaluation masks, and declared E4
tasks, DeCE-RF's feedback/projection controller provides conservative
stabilization evidence over fixed DeCE displacement. The stress curve should be
reported as an edit-preserve tradeoff using local edit L1 as an edit-pressure
proxy, not as a standalone semantic success score.
```

The revised design treats E4 as Pareto/stress evidence rather than a single
fixed-vs-feedback mean. Recommended Phase 2 stress axes:

```text
edit strength / guidance scale sweep
support perturbation sweep: erode, dilate, shift, noisy support
feedback/projection lambda or update-frequency sweep
```

Only write a stronger controller claim if the Pareto curve shows lower outside
drift at matched edit success, or higher edit success under a matched preserve
budget.

## Table / Figure 6: Boundary And Extension Cases

Purpose:

```text
document scope boundary: where the base method stops working and which
extension routes are separate from the base DeCE-RF mean
```

Scope:

```text
selected outputs: 36/36 complete
positive extension routes: high-confidence completion prior; replacement target route
failure labels: semantic glyph hallucination; cluttered-surface damage;
removal completion failure; replacement ambiguity
```

Locked artifacts:

```text
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_selected_manifest.csv
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_failure_taxonomy.csv
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_boundary_extension_summary.md
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_figure6_boundary_extension_seed10.png
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_gated_completion_protocol.json
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_whiteboard_red_star_protocol.json
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_boundary_extension_complete_2026-06-04.md
experiments/support_v3_2026-06-02/e5_boundary_extension/e5_boundary_extension.sha256
```

Paper-safe claim:

```text
E5 documents extension routes and scope boundaries. It supports Figure 6 and
the limitations paragraph, but the extension routes are named separately and
are not aggregated into the base DeCE-RF mean.
```

## Same-Support Removal Diagnostic

A removal-only diagnostic was generated for `backpack_remove_toy_charm` using
Telea and Navier-Stokes OpenCV inpainting with the DeCE-RF support mask. Report
it separately from the main comparison because it receives the same support mask
and only applies to removal/fill cases.

Artifact status: this is a legacy diagnostic row. The active repository no
longer tracks summary CSV/JSON files for this run; generated images remain
under ignored generated same-support inpaint output directories and should not
be cited as active paper evidence unless regenerated and summarized.

Readout: same-support inpainting gives lower outside drift on the backpack case
but produces visible fill artifacts around the strap/zipper region, while
DeCE-RF removes the target charm but locally smooths the occluded zipper/fabric.

## Legacy Baseline Artifacts

Legacy baseline artifacts are historical audit material only:

```text
experiments/archive_legacy_2026-05-11/baseline_parity_manifest.csv
experiments/archive_legacy_2026-05-11/baseline_summary.csv
experiments/archive_legacy_2026-05-11/baseline_summary.md
paper/archive_old_core6_20260602/old_stage2_5_integrity_precheck.md
```

Do not use legacy Core-4 or old Core-6 artifacts as active paper evidence unless
they are rerun under the current strict protocol.

## Supplement: Audit And Mask Sensitivity

The 2026-06-05 design requires stronger evaluation transparency.

Blind internal audit:

```text
3 raters
randomized method order
source and target instruction visible
method name hidden
1-5 ratings for edit_correct, relation_correct, source_preservation, locality,
artifact_severity, overall, and controlled failure_type labels
```

Mask sensitivity:

```text
eroded evaluation mask
base evaluation mask
dilated evaluation mask
```

Operational mask-freeze protocol:

```text
define intended edit region from source image + target instruction
create base eval mask before inspecting method outputs
record mask metadata/hash
derive eroded and dilated variants
evaluate all methods/seeds with the same mask set
```

Main paper should report whether rankings are stable under mask sensitivity.
Full numbers belong in supplement. If preservation ranking is highly sensitive
to mask boundary choice, outside-mask metrics should be interpreted together
with visual audit rather than as standalone strong evidence.

## Phase 2 Breadth Plan

The strict Core-6 canonical set supports a controlled diagnostic suite. The
2026-06-05 design treats Phase 2 breadth as a supplement/robustness path, not a
prerequisite for the main controlled-suite claim. It recommends expanding
breadth before making broader claims:

```text
minimum: 6 categories x 2 source examples x 4 E1 methods x 3 seeds = 144 outputs
preferred: 6 categories x 3 source examples x 4 E1 methods x 3 seeds = 216 outputs
```

Priority is to add new source images for already-supported operations rather
than new relations:

```text
second accessory insertion
second surface decal
second local recolor
second exposed small-object removal
second inside-container insertion, after checking container-interior support
second material/surface-strip example
```
