# E2.1-E5 Evidence Lock

Date: 2026-06-04

Status: E2.1-E2.4, E3, E4, and E5 are locked for paper drafting. E2.5 is deferred as an optional future/cross-backbone probe.

## Scope

This lock covers the evidence package used for the current WACV experiment narrative:

| Layer | Status | Role in paper | Claim boundary |
| --- | --- | --- | --- |
| E2.1 protocol/readout lock | Locked | fixes source set, masks, normalization, strict readout, and visual audit protocol | protocol/evaluation evidence only |
| E2.2 same-backbone SD3 comparison | Locked | primary algorithm-level comparison | SD3-only same-backbone claim |
| E2.3 native FLUX contextual comparison | Locked | practical/native external RF context | contextual only; not the E2.2 algorithmic claim |
| E2.4 support-matched diagnostic | Locked | tests whether binary localization alone explains the gain | diagnostic only; post-hoc blending is not a fair main baseline |
| E3 support geometry ablation | Locked | support geometry evidence and Figure 4 package | support-map diagnostic plus runnable generic-vs-operation rows |
| E4 controller/robustness ablation | Locked | fixed-vs-feedback controller evidence and Figure 5 package | SD3 controller evidence; not an external-baseline or cross-backbone claim |
| E5 boundary/extension cases | Locked | Figure 6 extension/failure package and limitation taxonomy | boundary/extension evidence only; not main quantitative claim |
| E2.5 cross-backbone transfer probe | Deferred | optional future work | no current cross-backbone DeCE-RF claim |

## Locked Artifacts

### E2.1 Protocol / Readout Lock

Use these artifacts to document fixed evaluation masks, normalized display/eval copies, strict readout, and human visual audit:

```text
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/eval_masks/
experiments/support_v3_2026-06-02/normalized_512/normalized_512_manifest.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.md
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.md
```

### E2.2 Same-Backbone SD3 Algorithm Comparison

Use these artifacts for the same-backbone SD3 algorithm-level evidence:

```text
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed10_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed11_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed12_grid.png
```

Paper-safe claim:

```text
Under the same SD3 backbone, same prompts/source images, and fixed evaluation
masks, DeCE-RF improves localized edit-preserve behavior over runnable RF-native
SD3 editing baselines in the reduced strict comparison.
```

### E2.3 Native FLUX Contextual Comparison

Use these artifacts for native/off-the-shelf FLUX contextual evidence:

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

Paper-safe claim:

```text
Native FLUX preservation-aware RF editors are useful contextual baselines, but
backbone differences mean E2.3 does not replace the same-backbone E2.2
algorithmic claim.
```

### E2.4 Support-Matched Diagnostic

Use these artifacts for the localization/support diagnostic:

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

Locked E2.4 rows:

```text
direct_target_raw
direct_target_mask_blend
flowedit_mask_blend
support_v3_controller_rmsgap
```

Paper-safe claim:

```text
Binary localization/output blending improves preservation metrics by
construction but does not recover target correctness or boundary coherence.
Therefore, localization alone is insufficient to explain the DeCE-RF result.
```

### E3 Support Geometry Ablation

E3 is complete for the current paper-drafting package.

Scope:

```text
3 tasks x 6 support geometry variants x 3 seeds = 54 support-map rows
tasks: cat_crown, tshirt_star, backpack_remove_toy_charm
variants: attention_only, clean_disagreement, velocity_disagreement,
grounding_sam, generic_support, operation_conditioned_support
```

Locked E3 artifacts:

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

Paper-safe E3 claim:

```text
E3 treats the support mask as an explicit experimental object. Operation-
conditioned support improves fixed-mask overlap and downstream edit behavior
relative to weak generic support, while Grounding/SAM alone tends to over-cover
the object/host region. Attention, clean-disagreement, velocity-disagreement,
and Grounding/SAM rows are support-map diagnostics, not full editing baselines.
```

Deferred optional E3 extensions:

```text
manual upper-bound support
support shrink/dilate perturbation
broader 6-task or 12-case expansion
```

### E4 Component Ablation Anchor

This is not part of E2.4, but it preserves the `support_v3_fixed` evidence under the correct paper identity:

```text
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.csv
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact_rows.csv
experiments/support_v3_2026-06-02/e4_fixed_dece_component_ablation_compact.md
```

Paper-facing identity:

```text
Fixed DeCE displacement = support_v3_fixed
DeCE-RF = support_v3_controller_rmsgap
```

Boundary:

```text
Fixed DeCE displacement is a component ablation. It is not an external baseline
and not an E2.4 support-only row.
```

### E4 Controller And Robustness Ablation

E4 is complete for the declared SD3 controller-ablation subset.

Scope:

```text
tasks: cat_crown, tshirt_star, pillow_vertical_fabric_strip
base rows: support_v3_fixed, support_v3_controller_rmsgap x seeds 10/11/12 = 18
stress rows: support_v3_fixed, support_v3_controller_rmsgap x edit multipliers 0.50/0.75/1.00/1.25/1.50/2.00 x seed10 x 3 tasks = 36 metric rows
```

Locked E4 artifacts:

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

Paper-safe E4 claim:

```text
Under the same SD3 implementation, fixed evaluation masks, and declared E4
tasks, DeCE-RF's feedback/projection controller provides conservative
stabilization evidence over fixed DeCE displacement. The stress curve should be
reported as an edit-preserve tradeoff using local edit L1 as an edit-pressure
proxy, not as a standalone semantic success score.
```

### E5 Boundary, Extension, And Failure Cases

E5 is complete for the selected boundary/extension package.

Scope:

```text
selected outputs: 36/36 complete
positive extension routes: high-confidence completion prior; replacement target route
failure labels: semantic glyph hallucination; cluttered-surface damage; removal completion failure; replacement ambiguity
```

Locked E5 artifacts:

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

Paper-safe E5 claim:

```text
E5 documents extension routes and scope boundaries. It supports Figure 6 and
the limitations paragraph, but the extension routes are named separately and
are not aggregated into the base DeCE-RF mean.
```

## Deferred E2.5

E2.5 is intentionally skipped for this evidence lock.

Reason:

```text
The current DeCE-RF implementation is SD3-specific. A FLUX implementation would
require separate support extraction, scheduler/controller calibration, and
validation. Running it now would introduce cross-backbone confounds rather than
strengthen the locked same-backbone claim.
```

Paper-safe wording:

```text
All algorithm-level conclusions are drawn from same-backbone SD3 comparisons.
Native FLUX rows are used only as contextual evidence for off-the-shelf RF
editors. Full cross-backbone DeCE-RF transfer is left to future work.
```

## Next Step

Proceed to paper evidence packaging or optional later experiments:

```text
1. Main tables and figure list
2. Experiment section draft
3. Supplement table/figure mapping
4. Optional E6/full supplement packaging
```
