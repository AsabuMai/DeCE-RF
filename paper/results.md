# Current Results

Date: revised strict Phase 1 reconciled on 2026-06-03.

Scope: current WACV Phase 1 evidence under the revised strict Core-6 taxonomy.
Older 2026-06-01 server-evidence tables are archived under
`paper/archive_old_core6_20260602/` and should be used only as supplementary
diagnostics.

## Active Strict Core-6

The current strict task set is:

```text
T1 attached accessory: cat_crown
T2 container-constrained insertion: bowl_apple_inside
T3 surface decal: tshirt_star
T4 local recolor: red_chair_blue
T5 surface material strip: pillow_vertical_fabric_strip
T6 exposed-object removal: backpack_remove_toy_charm
```

The diagnostic rows `dog_sunglasses`, `mug_heart`, and `pillow_blue_stripes`
are no longer canonical strict Core-6 rows.

## Protocol

Paper-facing methods:

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

`support_v3_fixed` is retained as a component ablation/control only. It should
not appear as a headline main-table method.

All headline preservation metrics use fixed per-task evaluation masks shared
across methods and seeds. The evaluation mask is not each method's own support
mask.

## Current Artifacts

Strict Core-6 metrics:

```text
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.md
```

Strict Core-6 visual audit:

```text
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.md
experiments/support_v3_2026-06-02/visual_audit/
```

Fixed evaluation masks:

```text
experiments/support_v3_2026-06-02/eval_masks/
```

E2.2 same-backbone SD3 RF comparison:

```text
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

RF baseline audit:

```text
experiments/support_v3_2026-06-02/e2_baseline_download_registry.csv
experiments/support_v3_2026-06-02/e2_baseline_runnable_validation.csv
experiments/support_v3_2026-06-02/e2_baseline_audit.md
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
```

## Main Readout

The revised strict Phase 1 matrix is complete:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72 complete runs
```

The quick visual audit marks the DeCE-RF row as usable for all six strict tasks:

- `cat_crown`: clear crown, stable cat and background.
- `bowl_apple_inside`: apple centered inside bowl; stable scene.
- `tshirt_star`: large red star centered on shirt; stable folds, pose, jeans,
  and background.
- `red_chair_blue`: blue chair; stable room.
- `pillow_vertical_fabric_strip`: vertical blue silk strip follows pillow
  perspective with clean top/bottom boundaries across seeds 10/11/12.
- `backpack_remove_toy_charm`: toy removed; surface/background stable, with the
  usual removal-completion caveat.

Direct target guidance remains an aggressive baseline: it may produce target
semantics, but often changes source identity, crop/layout, or background.
Generic support control remains a strong preservation baseline, but often
over-preserves and misses the intended edit.

## E2 Readout

E2 is now a backbone-controlled and preservation-aware fairness experiment, not
a single RF-baseline leaderboard.

### Completed E2.2 Same-Backbone SD3 Evidence

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

Artifacts:

```text
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

FlowEdit, FlowAlign, and SplitFlow are runnable under the revised strict
same-backbone SD3 protocol. The current readout supports only the narrow claim
that DeCE-RF improves localized edit-preserve behavior over these SD3 RF-native
rows under fixed evaluation masks. It is not a claim over all RF or FLUX editors.

### Needed E2 Upgrades

E2.1 backbone calibration:

```text
SD3 reconstruction/direct-target floors: reuse E1 base_only/direct_target.
FLUX reconstruction/direct-target floors: run only after FLUX access and adapters are valid.
```

E2.2 preservation-control upgrade:

```text
Join strict Fixed DeCE rows as an SD3 preservation-control row.
Attempt OT-RF/OTIP-SD3 or RF-Edit-SD3 only if a real SD3 adapter is verified.
```

E2.3 native preservation-aware RF context:

```text
rf_solver_edit = RF-Solver-Edit / RF-Edit, FLUX.1-dev route currently access-blocked
reflex = ReFlex, FLUX.1-dev route currently access-blocked
stable_flow = stable-flow, adapter pending
fireflow = FireFlow, FLUX.1-dev route currently access-blocked
ot_rf_otip = OT-RF / OTIP-style, repo/backbone/adapter pending
dvrf = DVRF / Delta Velocity RF, repo/backbone/adapter pending
```

E2.4 support-matched diagnostic:

```text
Run compact rows for direct_target + same M_edit, FlowEdit + same M_edit if stable,
Fixed DeCE, and DeCE-RF on cat_crown, tshirt_star, and backpack_remove_toy_charm.
```

### Safe Wording

Use:

```text
Under the same SD3 backbone and fixed evaluation masks, DeCE-RF improves the
localized edit-preserve tradeoff over completed SD3 RF-native baselines.
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

## Component Evidence

`support_v3_fixed` isolates decoupled clean-estimate displacement with
operation-conditioned support but without feedback-updated control. The
fixed-vs-feedback gap is modest, so this supports component evidence rather
than a headline claim. E4 should be strengthened with stress/Pareto curves
before making a strong controller-robustness claim.

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
large standalone gains from feedback control without stress/Pareto evidence
```

## Next Evidence Needed

1. Keep the revised strict Core-6 metrics and visual audit as the active E1
   evidence package.
2. Use old server-evidence rows only as supplement/diagnostic material.
3. Keep weak replacement and difficult completion cases out of the main table;
   use them only in limitation or extension-probe sections.
4. Report `laptop_remove_sticker` as a high-confidence completion-prior
   extension probe and `whiteboard_probe_red_star_sticker` as a non-glyph
   replacement probe, not as base DeCE-RF rows.
5. Upgrade E2 with calibration, one native preservation-aware RF status/metric row if runnable,
   and a compact support-matched diagnostic before claiming that existing strong RF editors
   cannot replace DeCE-RF in practice; keep the main algorithmic claim tied to E2.2 same-backbone SD3 evidence.
6. Strengthen E4 with controller stress/Pareto evidence if feedback control
   becomes reviewer-critical.
