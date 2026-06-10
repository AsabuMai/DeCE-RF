# Current Results

Date: evidence lock synchronized on 2026-06-04.

Scope: current WACV evidence package under the revised strict Core-6 taxonomy
and the locked E2.1-E5 experiment narrative. Older 2026-06-01 server-evidence
tables are archived under `paper/archive_old_core6_20260602/` and should be
used only as supplementary diagnostics.

Primary lock file:

```text
experiments/support_v3_2026-06-02/e2_evidence_lock_2026-06-04.md
```

Revised design source:

```text
paper/wacv_experiment_design.md
```

## Active Strict Core-6

The current strict task set is:

```text
T1 attached accessory: cat_crown
T2 container-constrained insertion: bowl_apple_inside
T3 surface decal: tshirt_star
T4 local recolor: red_chair_blue
T5 localized same-color material replacement: pillow_same_color_corduroy_panel
T6 exposed-object removal: backpack_remove_toy_charm
```

The diagnostic rows `dog_sunglasses`, `mug_heart`, `pillow_blue_stripes`, and
the previous `pillow_vertical_fabric_strip` blue-strip probe are no longer
canonical strict Core-6 rows.

## Protocol

Paper-facing E1 methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

Runner names:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

`support_v3_fixed` is retained as Fixed DeCE displacement. It should not appear
as a headline E1 main-table method, but the 2026-06-05 design uses it as an E4
component/controller ablation and as the E2.2 same-backbone SD3
preservation-control row when joined with the strict cache.

All headline preservation metrics use fixed per-task evaluation masks shared
across methods and seeds. The evaluation mask is not each method's own support
mask.

## Locked Evidence Package

E1 strict Core-6 metrics and visual audit:

```text
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.md
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.md
experiments/support_v3_2026-06-02/eval_masks/
experiments/support_v3_2026-06-02/normalized_512/normalized_512_manifest.csv
```

E2.2 same-backbone SD3 algorithm comparison:

```text
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

E2.3 native FLUX contextual comparison:

```text
experiments/support_v3_2026-06-02/e2_native_flux_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_native_flux_fixed_mask_metrics_with_context.csv
experiments/support_v3_2026-06-02/e2_native_flux_contextual_table.csv
experiments/support_v3_2026-06-02/e2_native_flux_contextual_table.md
experiments/support_v3_2026-06-02/e2_native_flux_normalized_512_manifest.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_contextual_conclusion.md
```

E2.4 support-matched diagnostic:

```text
experiments/support_v3_2026-06-02/e2_support_matched_diagnostic_manifest.csv
experiments/support_v3_2026-06-02/e2_support_matched_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.csv
experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_summary.csv
experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_conclusion.md
```

E3-E5 mechanism, controller, and boundary packages:

```text
experiments/support_v3_2026-06-02/e3_support_geometry/
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.md
experiments/support_v3_2026-06-02/e4_controller_ablation/
experiments/support_v3_2026-06-02/e5_boundary_extension/
```

## Main Readout

The previous strict Phase 1 matrix is complete for the older T5 blue-strip
probe:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72 complete runs
```

The quick visual audit marks the DeCE-RF row as usable for the previous
six-task package:

- `cat_crown`: clear crown, stable cat and background.
- `bowl_apple_inside`: apple centered inside bowl; stable scene.
- `tshirt_star`: large red star centered on shirt; stable folds, pose, jeans,
  and background.
- `red_chair_blue`: blue chair; stable room.
- `pillow_vertical_fabric_strip`: vertical blue silk strip follows pillow
  perspective with clean top/bottom boundaries across seeds 10/11/12; this is
  now diagnostic because the canonical T5 has been revised to same-color
  material replacement.
- `backpack_remove_toy_charm`: toy removed; surface/background stable, with the
  usual removal-completion caveat.

The current canonical T5 is `pillow_same_color_corduroy_panel`. It should be
evaluated as localized same-color material replacement, not as a blue strip,
decal, or recolor row.

Direct target guidance remains an aggressive baseline: it may produce target
semantics, but often changes source identity, crop/layout, or background.
Generic support control remains a strong preservation baseline, but often
over-preserves and misses the intended edit.

## E2 Readout

E2 is locked as a backbone-controlled and preservation-aware fairness
experiment, not a single RF-baseline leaderboard.

### E2.1 Protocol And Calibration Lock

E2.1 fixes the source set, fixed evaluation masks, normalized 512 display/eval
copies, strict readout, and human visual-audit protocol. SD3 reconstruction and
direct-target floors are taken from the strict E1 rows. Native FLUX rows are
reported with their own backbone/context columns rather than merged into a
same-backbone leaderboard.

### E2.2 Same-Backbone SD3 Evidence

Completed strict SD3 rows:

```text
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
DeCE-RF-SD3
```

Joined with E1 rows:

```text
direct_target-SD3
base_only-SD3, for calibration/reconstruction floor
```

FlowEdit, FlowAlign, and SplitFlow are runnable under the revised strict
same-backbone SD3 protocol. This supports only the narrow claim that DeCE-RF
improves localized edit-preserve behavior over runnable SD3 RF-native rows
under fixed evaluation masks. It is not a claim over all RF or FLUX editors.

The revised 2026-06-05 design also treats Fixed DeCE-SD3
(`support_v3_fixed`) as the preservation-control row for Table 2a. Its stable
paper identity is Fixed DeCE (same-backbone component control): operation-
conditioned support plus fixed displacement without feedback. It is not an
external baseline and not an E1 headline method.

### E2.3 Native FLUX Context

Native FLUX preservation-aware RF rows are now locked as contextual evidence.
They test whether off-the-shelf native implementations solve the same localized
edit-preserve setting in practice, but backbone and interface differences mean
they do not replace the same-backbone E2.2 algorithmic claim.

Paper-facing interpretation:

```text
Native FLUX preservation-aware RF editors are useful contextual baselines, but
backbone differences mean E2.3 does not replace the same-backbone E2.2
algorithmic claim.
```

### E2.4 Support-Matched Diagnostic

Locked rows:

```text
direct_target_raw
direct_target_mask_blend
flowedit_mask_blend
support_v3_controller_rmsgap
```

This diagnostic answers whether binary localization alone explains the
DeCE-RF result. The locked readout is that localization/output blending improves
preservation metrics by construction but does not recover target correctness or
boundary coherence. Therefore localization alone is insufficient to explain the
DeCE-RF result.

The 2026-06-05 design considers the current blending rows diagnostic but not the
strongest possible same-support evidence. If more runs are made, prioritize
inference-time same-`M_edit` gating rows for direct target and FlowEdit, plus
Fixed DeCE and DeCE-RF on the same compact task subset.

### Safe E2 Wording

Use:

```text
Under the same SD3 backbone, same prompts/source images, and fixed evaluation
masks, DeCE-RF improves localized edit-preserve behavior over runnable RF-native
SD3 editing baselines in the reduced strict comparison.
```

Use for native rows:

```text
Native preservation-aware RF editors are reported as implementation-context
baselines because their public routes use different backbones or interfaces.
```

Do not write:

```text
DeCE-RF beats all RF baselines.
DeCE-RF beats FLUX.
SD3-DeCE is directly superior to ReFlex-FLUX or RF-Edit-FLUX as an algorithm.
```

## Component And Mechanism Evidence

E3 treats the support mask as an explicit experimental object:

```text
3 tasks x 6 support geometry variants x 3 seeds = 54 support-map rows
tasks: cat_crown, tshirt_star, backpack_remove_toy_charm
variants: attention_only, clean_disagreement, velocity_disagreement,
grounding_sam, generic_support, operation_conditioned_support
```

Paper-safe E3 claim:

```text
Operation-conditioned support improves fixed-mask overlap and downstream edit
behavior relative to weak generic support, while Grounding/SAM alone tends to
over-cover the object/host region. Attention, clean-disagreement,
velocity-disagreement, and Grounding/SAM rows are support-map diagnostics, not
full editing baselines.
```

E4 is complete for the declared SD3 controller-ablation subset:

```text
tasks: cat_crown, tshirt_star, pillow_same_color_corduroy_panel
base rows: support_v3_fixed, support_v3_controller_rmsgap x seeds 10/11/12 = 18
stress rows: support_v3_fixed, support_v3_controller_rmsgap x edit multipliers
0.50/0.75/1.00/1.25/1.50/2.00 x seed10 x 3 tasks = 36 metric rows
```

`support_v3_fixed` should be described as Fixed DeCE displacement. It isolates
decoupled clean-estimate displacement with operation-conditioned support but
without the feedback/projection controller.

Paper-safe E4 claim:

```text
Under the same SD3 implementation, fixed evaluation masks, and declared E4
tasks, DeCE-RF's feedback/projection controller provides conservative
stabilization evidence over fixed DeCE displacement. The stress curve should be
reported as an edit-preserve tradeoff using local edit L1 as an edit-pressure
proxy, not as a standalone semantic success score.
```

## Boundary And Extension Evidence

E5 is complete for the selected boundary/extension package:

```text
selected outputs: 36/36 complete
positive extension routes: high-confidence completion prior; replacement target route
failure labels: semantic glyph hallucination; cluttered-surface damage;
removal completion failure; replacement ambiguity
```

Paper-safe E5 claim:

```text
E5 documents extension routes and scope boundaries. It supports Figure 6 and
the limitations paragraph, but the extension routes are named separately and
are not aggregated into the base DeCE-RF mean.
```

## Claim Boundary

The current evidence supports:

```text
DeCE-RF improves localized edit-preserve control under reasonable support
across insertion, surface editing, local recolor, and simple exposed removal
within a controlled Core-6 diagnostic suite.
```

The current evidence does not support:

```text
broad arbitrary removal/replacement
occluded-object removal requiring substantial host completion
precise glyph replacement
state-of-the-art general-purpose image editing
cross-backbone DeCE-RF transfer to FLUX
algorithm-level superiority over native FLUX editors
```

E2.5 cross-backbone DeCE-RF transfer is deferred. All algorithm-level
conclusions should be drawn from same-backbone SD3 comparisons; native FLUX rows
are contextual evidence for off-the-shelf RF editors.

## Next Drafting Steps

1. Use the locked E1/E2/E3/E4/E5 artifacts as the paper evidence package.
2. Build main Table 1, Table 2a/2b, and supplement E2/E3/E4/E5 tables from the
   locked artifact lists.
3. In Table 2a, include Fixed DeCE-SD3 as the same-backbone preservation-control
   row when the strict fixed cache is joined.
4. Update the experiment section around the evidence chain:
   main effect -> RF-specific comparison -> mechanism ablation -> boundary cases.
5. Keep old server-evidence rows only as supplement/diagnostic material.
6. Keep extension routes out of the base DeCE-RF aggregate unless the method
   column explicitly names the extra route.
7. Treat Phase 2 breadth, blind internal audit, erode/base/dilate mask
   sensitivity, and stronger E2.4 inference-time same-support diagnostics as
   supplement/robustness paths before making broader claims, not as prerequisites
   for the current controlled Core-6 readout.
