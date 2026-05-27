# DeCE-RF Experiment Plan

Working title:

```text
Decoupled Clean-Estimate Control for Localized Rectified Flow Image Editing
```

Method name:

```text
DeCE-RF
```

## Goal

The experiments should support one conservative claim:

```text
Localized Rectified Flow editing benefits from decoupling edit and
reconstruction corrections, localizing them through an operation-conditioned
support interface, and adapting their strengths with clean-estimate feedback.
```

Do not frame the paper as a fully general automatic image editor. The expected
paper claim is an edit-preserve control claim under reasonable support.

## Claims To Validate

Claim 1: edit/reconstruction decoupling improves preservation over direct
target guidance while preserving target-directed edit behavior.

Claim 2: operation-conditioned support gives better control geometry than weak
generic support and should approach manual/external support when the operation
and relation are well specified.

Claim 3: clean-estimate feedback control improves or stabilizes the
edit-preserve tradeoff relative to fixed edit/preserve weights, especially
under stronger edit scales or support perturbations.

## Primary Tasks

Use tasks that cover different operation conditions:

| Task | Operation | Relation | Paper role |
| --- | --- | --- | --- |
| `cat_crown` | `add_object` | `above_host` | compact accessory insertion and main stress case |
| `dog_sunglasses` | `add_object` | `on_face` | strong positive control and edit-strength tradeoff |
| `mug_heart` | `add_decal` | `on_surface` | surface/decal localization |
| `backpack_remove_toy_charm` | `remove_object` | `remove_source_object` | limitation and preservation/removal probe |

Optional expansion tasks after the primary matrix is stable:

| Task | Operation | Role |
| --- | --- | --- |
| `tshirt_star` | `add_decal` | second surface/decal case |
| `tote_leaf` | `add_decal` | logo/decal case with clear host surface |
| `rabbit_sunglasses` | `add_object` | difficult profile/face relation |
| `backpack_replace_patch_blue` | `replace` | replacement limitation |

## Primary Methods

Keep method names aligned with the existing runner where possible.

| Method | Runner name | Purpose |
| --- | --- | --- |
| Source/base only | `base_only` | no edit baseline |
| Direct target | `direct_target` | coupled target-velocity baseline |
| Generic support | `adaptive_full_generic_support` or closest active path | weak automatic support baseline |
| Fixed operation support | `support_v3_fixed` | decoupled edit/rec with operation-conditioned support, no feedback |
| DeCE-RF | `support_v3_controller_rmsgap` | full method: decoupled support-conditioned clean-estimate feedback |
| Manual/external support | `manual_support` where available | upper-bound diagnostic for support quality |

Paper-facing names should not expose `support_v3` as the conceptual method.
Use:

```text
operation-conditioned support
fixed decoupled control
DeCE-RF feedback control
```

## Seeds

Primary seeds:

```text
10, 11, 12
```

Use seed 10 for debugging and visual gate checks. Expand to 10/11/12 only after
the seed-10 output is visually acceptable and all required artifacts are saved.

## Main Matrix

Primary controlled matrix:

```text
4 tasks x 5 methods x 3 seeds = 60 runs
```

Recommended method set:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_fixed
support_v3_controller_rmsgap
```

Add `manual_support` only where a reliable manual/external support path exists,
and report it as an upper-bound diagnostic rather than a main automatic method.

## Ablations

### A1: Decoupling

Purpose: isolate edit/reconstruction decomposition.

Compare:

```text
direct_target
support_v3_fixed with rec disabled
support_v3_fixed with trajectory preserve disabled
support_v3_fixed
```

Evidence:

```text
outside-mask drift
DINO/source similarity
manual preservation score
qualitative overlays
```

### A2: Support Interface

Purpose: show why operation-conditioned support is a method component, not a
task-specific trick.

Compare:

```text
generic support
attention-only or clean-only support
operation-conditioned support
manual/external support upper bound
```

Evidence:

```text
M_edit visualization
mask leakage / area / localization notes
edit success under the same controller
```

### A3: Clean-Estimate Feedback

Purpose: isolate RMSGAP / clean-estimate feedback.

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
support_v3_controller_rmsgap under edit-strength sweep
support_v3_controller_rmsgap under support perturbations
```

Evidence:

```text
edit-progress curves
preserve-drift curves
adaptive edit/preserve weights
Pareto plot: edit score vs outside drift
```

### A4: Limits

Purpose: keep the claim honest.

Include:

```text
backpack_remove_toy_charm
backpack_replace_patch_blue
rabbit_sunglasses
```

Report failure type:

```text
support failure
target formation failure
removal weakness
replacement ambiguity
over-preservation
```

## Metrics

Automatic metrics:

```text
outside-mask L1/RMSE
inside-mask change
luma SSIM
DINO source similarity
CLIP target score
CLIP target-source delta
runtime
peak GPU memory
support area / leakage / overlap where masks exist
```

Trajectory/controller metrics:

```text
adaptive_edit_progress
adaptive_edit_target_gap_rms
adaptive_preserve_drift
adaptive_edit_weight
adaptive_preserve_weight
adaptive_projection_norm
```

Manual visual audit:

```text
edit success: 1-5
source preservation: 1-5
locality: 1-5
artifact severity: 1-5
overall: 1-5
failure flag and short note
```

Manual scores must be labeled as an internal visual audit, not a user study.

## Figures

Main qualitative figure:

```text
source | target text | direct target | generic support | fixed operation support | DeCE-RF
```

Support-interface figure:

```text
attention evidence | clean disagreement | velocity disagreement | relation/surface prior | selected M_edit/M_preserve
```

Controller figure:

```text
fixed vs DeCE-RF result
edit-progress curve
preserve-drift curve
adaptive weights
```

Failure figure:

```text
successful support case
bad support case
removal/replacement limitation
```

## Tables

Table 1: main quantitative matrix, averaged over tasks and seeds.

Table 2: ablation by contribution:

```text
decoupling
support interface
feedback control
```

Table 3: external baseline availability and qualitative comparison, with model
family and seed-matching caveats.

Table 4: failure taxonomy counts.

## External Baselines

Use existing baseline records where already available:

```text
FlowEdit
SplitFlow
FireFlow
RF-Solver-Edit
```

Report ReFlex and SteerFlow as failed/unavailable only if the failure metadata
is kept in the manifest. Do not claim superiority over baselines that could not
be run under matched conditions.

## Server Run Order

When the lab server is available:

1. Run a seed-10 visual gate for the primary tasks and method set.
2. Expand accepted rows to seeds 10/11/12.
3. Generate metrics with `scripts/evaluate_paper_metrics.py`.
4. Build comparison grids.
5. Run contribution ablations.
6. Run stress/perturbation sweeps for feedback-control evidence.
7. Update manual visual audit and failure flags.

Suggested runner pattern:

```bash
TASKS="cat_crown dog_sunglasses mug_heart backpack_remove_toy_charm" \
METHODS="base_only direct_target adaptive_full_generic_support support_v3_fixed support_v3_controller_rmsgap" \
SEEDS="10 11 12" \
bash scripts/run_pretty_matrix.sh
```

Adjust `ROOT`, `PYTHON`, model cache, and GPU device variables on the server.

## Local 12GB Role

Use this workstation only for smoke tests and small probes.

Known local-safe settings:

```text
512px, 28 steps, n_max 24, --low-vram, edit_field_mode=rf_diff, text guidance off
384px, 16 steps, n_max 12, text guidance on, no --low-vram
```

Do not use local 12GB results as final paper numbers unless the configuration
is explicitly labeled as low-VRAM diagnostic.

## Acceptance Criteria

Minimum workshop/arXiv evidence:

```text
4 tasks x 5 methods x 3 seeds complete
main qualitative grid
one support-interface figure
one feedback-controller curve figure
failure taxonomy
clear limitation statement
```

Stronger conference evidence:

```text
more tasks across operation types
multi-seed ablations
matched external baselines
manual visual audit with consistent scoring rubric
support perturbation robustness
edit-preserve Pareto plots
```
