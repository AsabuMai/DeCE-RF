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

E2-A SD3-matched RF comparison:

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

E2 should now be treated as a separated backbone-matched and native-backbone
RF comparison: E2-A is the SD3-matched algorithmic comparison, while E2-B is a
native-backbone RF / FLUX contextual comparison.

Completed E2-A evidence:

```text
Native / target-mode RF baselines: FlowEdit, FlowAlign, SplitFlow
Tasks: strict Core-6
Seeds: 10, 11, 12
Status: complete SD3-matched RF comparison
```

FlowEdit, FlowAlign, and SplitFlow are runnable under the revised strict
target-mode protocol. DeCE-RF has lower outside-mask change and stronger
source-preservation metrics under the fixed task masks, but this result should
be written narrowly as E2-A rather than as the full RF-baseline answer.

Pending E2-B contextual upgrade:

```text
Native-backbone contextual candidates:
rf_solver_edit = RF-Solver-Edit / RF-Edit, FLUX.1-dev
reflex = ReFlex, FLUX.1-dev
stable_flow = stable-flow, FLUX.1-dev
fireflow = FireFlow, FLUX.1-dev
ot_rf_otip = OT-RF / OTIP-style, backbone TBD
dvrf = DVRF / Delta Velocity RF, backbone TBD
```

At least one native-backbone RF / FLUX baseline should become runnable before
making the stronger contextual statement that existing RF editors do not close
the localized edit-preserve gap. The main algorithmic claim remains SD3-matched.

Minimum E2-B contextual target:

```text
1 native-backbone RF / FLUX baseline x 6 tasks x 3 seeds = 18 contextual outputs
```

Resource-saving fallback:

```text
1 native-backbone RF / FLUX baseline x 6 tasks x 2 seeds = 12 contextual outputs
```

Do not write:

```text
DeCE-RF beats all RF baselines.
DeCE-RF beats FLUX.
```

Use this safer wording:

```text
E2-A shows, under a matched SD3 backbone, whether RF-native target-mode editing
baselines solve localized edit-preserve control. E2-B is needed to contextualize
DeCE-RF against native-backbone RF / FLUX editors without treating those rows as
pure algorithmic controls.
```

RF-Solver-Edit (`rf_solver_edit`), ReFlex (`reflex`), FireFlow (`fireflow`),
and stable-flow (`stable_flow`) are native-backbone FLUX rows currently blocked
by gated FLUX.1-dev access or adapter gaps. OT-RF / OTIP (`ot_rf_otip`) and DVRF
(`dvrf`) are registered planned E2-B candidates that still need repo
verification, environment creation, smoke tests, and Core-6 adapters. If those
blockers remain, report them explicitly in the E2 audit rather than silently
omitting the baselines.

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
5. Upgrade E2 with one native-backbone RF / FLUX contextual baseline before
   claiming that existing strong RF editors cannot replace DeCE-RF in practice;
   keep the main algorithmic claim tied to SD3-matched E2-A.
6. Strengthen E4 with controller stress/Pareto evidence if feedback control
   becomes reviewer-critical.
