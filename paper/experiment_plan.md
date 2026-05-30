# DeCE-RF Experiment Plan

Working title:

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

Method name:

```text
DeCE-RF
```

## Goal

The experiments should support one conservative main claim:

```text
Localized Rectified Flow editing benefits from decoupled clean-estimate
edit-preserve displacement control: edit displacement toward the target clean
estimate, preserve displacement toward the source latent, operation-conditioned
control geometry, and feedback-updated displacement weights.
```

Do not frame the paper as a fully general automatic image editor. The expected
paper claim is an edit-preserve control claim for localized add/accessory,
decal, exposed-object removal, and one localized attribute/recolor task under
reasonable support. Completion-prior removal and replacement target formation
are operation-conditioned extension probes, not the main controller claim.

## Claims To Validate

Claim 1: decomposing the desired clean displacement into edit and preserve
components improves preservation over direct target guidance while preserving
target-directed edit behavior.

Claim 2: operation-conditioned support gives better control geometry than weak
generic support and should approach manual/external support when the operation
and relation are well specified.

Claim 3: clean-estimate feedback improves or stabilizes the edit-preserve
tradeoff relative to fixed displacement weights, especially under stronger edit
scales or support perturbations.

## Primary Tasks

Use tasks that cover different operation conditions:

| Task | Operation | Relation | Paper role |
| --- | --- | --- | --- |
| `cat_crown` | `add_object` | `above_host` | compact accessory insertion and main stress case |
| `dog_sunglasses` | `add_object` | `on_face` | strong positive control and edit-strength tradeoff |
| `mug_heart` | `add_decal` | `on_surface` | surface/decal localization |
| `tshirt_star` | `add_decal` | `on_clothing_surface` | second surface/decal case |
| `backpack_remove_toy_charm` | `remove_object` | `remove_source_object` | removal and preservation probe |
| `red_chair_blue` or `red_office_chair_to_blue_office_chair` | `recolor` | `object_attribute` | localized attribute/color-change probe |

Current completed main matrix:

```text
5 tasks x 4 paper-facing methods x 3 seeds = 60 runs
```

The completed core-5 task set is:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
```

This core-5 result is valid but thin for a full submission. The preferred
main-matrix expansion is a controlled core-6 by adding exactly one recolor /
attribute-editing task after a seed-10 visual gate.

Core-6 recolor candidates:

| Candidate task | Operation | Role | Gate status |
| --- | --- | --- | --- |
| `red_chair_blue` | `recolor` | localized chair-fabric color change in an indoor scene | preferred seed-10 gate |
| `red_office_chair_to_blue_office_chair` | `recolor` | localized office-chair color change with simple background | fallback seed-10 gate |

Only promote one recolor candidate into the main matrix if seed 10 passes
visually under the fixed protocol. Otherwise, keep the main benchmark as
core-5 and report recolor as a failed generalization probe.

## Operation-Conditioned Extension Probes

These probes should be reported separately from the main Core-6 benchmark,
because they use additional operation-conditioned routes beyond the main
DeCE-RF controller.

| Probe | Route | Role |
| --- | --- | --- |
| `laptop_remove_sticker` | high-confidence completion clean-delta gate | planar-surface removal where the completion prior is reliable |
| `whiteboard_probe_red_star_sticker` | replacement target-formation route | non-glyph replacement in a semantic letter field |

Paper role:

```text
Use these as evidence that DeCE-RF-style control can be extended when the
operation provides a trustworthy prior or a strong target shape/color cue. Do
not silently mix these routes into the main Core-6 method table.
```

Removal stress tasks excluded from the main matrix:

| Task | Reason |
| --- | --- |
| `dog_remove_tennis_ball` | occluded object removal requires mouth/host completion and leaves residual artifacts |
| `fridge_remove_yellow_magnet` | support mask is accurate, but stronger removal damages neighboring magnets |
| `fridge_remove_peach_magnet` | default edit is too conservative; stronger removal damages nearby objects |
| `whiteboard_remove_yellow_letter` | support mask is accurate, but the letter is transformed into another letter rather than removed |

Optional expansion tasks after the primary matrix is stable:

| Task | Operation | Role |
| --- | --- | --- |
| `tote_leaf` | `add_decal` | logo/decal case with clear host surface |
| `rabbit_sunglasses` | `add_object` | difficult profile/face relation |
| `backpack_replace_patch_blue` | `replace` | replacement limitation |
| `dog_replace_tennis_ball_star` | `replace` | partial replacement limitation; target pressure fires but the object is not clean |

Weak replacement and stylization tasks should stay out of the main claim. Use
them only as qualitative limits or future-work examples.

## Primary Methods

Keep method names aligned with the existing runner where possible.

| Method | Runner name | Purpose |
| --- | --- | --- |
| Source image | n/a | visual reference only; not a method row |
| RF reconstruction / base reconstruction | `base_only` | reconstruction-only RF pass; may drift from the source |
| Direct target | `direct_target` | coupled target-velocity baseline |
| Generic support | `adaptive_full_generic_support` or closest active path | weak automatic support baseline |
| DeCE-RF | `support_v3_controller_rmsgap` | full method: operation-conditioned geometry with feedback-updated displacement weights |
| Manual/external support | `manual_support` where available | upper-bound diagnostic for support quality |

Use `support_v3_fixed` as an ablation-only internal control, not as a headline
baseline:

| Method | Runner name | Purpose |
| --- | --- | --- |
| Fixed displacement weights | `support_v3_fixed` | decoupled clean displacement with operation-conditioned geometry, no feedback |

Paper-facing names should not expose `support_v3` as the conceptual method.
Use:

```text
operation-conditioned support
fixed displacement weights
DeCE-RF feedback-updated control
```

## Seeds

Primary seeds:

```text
10, 11, 12
```

Use seed 10 for debugging and visual gate checks. Expand to 10/11/12 only after
the seed-10 output is visually acceptable and all required artifacts are saved.

## Main Matrix

Completed paper-facing internal matrix:

```text
5 tasks x 4 methods x 3 seeds = 60 runs
```

Recommended method set:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

Target expansion after the recolor gate passes:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72 runs
6 tasks x support_v3_fixed x 3 seeds = 18 ablation-cache runs
total internal execution bundle = 90 runs
```

The fixed-DeCE runs are reused for the component ablation and feedback stress
analysis. Do not present them as the main external comparison.

Add `manual_support` only where a reliable manual/external support path exists,
and report it as an upper-bound diagnostic rather than a main automatic method.

Extension-probe matrices:

```text
laptop high-confidence completion removal:
6 removal-diagnostic tasks already run for reliability analysis; report only
laptop as the positive extension and the rest as gate-off diagnostics.

whiteboard replacement probe:
5 variants x 3 methods x 3 seeds = 45 complete runs; report red star sticker
as the positive non-glyph replacement probe and blank/T/A as diagnostics.
```

If time allows, add the RF recolor comparison:

```text
1 recolor task x 4 RF baselines x 3 seeds = 12 runs
```

## Ablations

### A1: Decoupled Clean-Estimate Displacement

Purpose: isolate the benefit of decomposing the desired clean displacement into
edit and preserve components.

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

### A2: Operation-Conditioned Control Geometry

Purpose: show why operation-conditioned geometry is a method component, not a
task-specific mask trick.

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

### A3: Feedback-Updated Displacement Weights

Purpose: isolate RMSGAP / clean-estimate feedback as online displacement-weight
updates. This is the correct comparison point for `support_v3_fixed` versus
DeCE-RF; do not make this small gap the headline result.

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
support_v3_fixed under edit-strength/support perturbation
support_v3_controller_rmsgap under edit-strength/support perturbation
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

### Fixed Evaluation Mask Protocol

For each task, build one task-level evaluation mask and reuse it for every
method and every seed. The fixed mask is used for `inside_mask_l1`,
`outside_mask_l1`, luma SSIM, DINO/source similarity, CLIP-local scores where
applicable, and the composite `edit_score`.

Mask source priority:

```text
manual mask
source-target diff mask
corrected DeCE/support mask frozen as an evaluation mask
```

Do not use each method's own predicted support mask for main preservation
metrics. Method masks can still be reported as support diagnostics.

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
source | target text | direct target | generic support | DeCE-RF | support overlay
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
TASKS="cat_crown dog_sunglasses mug_heart tshirt_star backpack_remove_toy_charm dog_remove_tennis_ball" \
METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
SEEDS="10 11 12" \
bash scripts/run_pretty_matrix.sh
```

Run the ablation cache separately or append `support_v3_fixed` to `METHODS`
when compute is available:

```bash
TASKS="cat_crown dog_sunglasses mug_heart tshirt_star backpack_remove_toy_charm dog_remove_tennis_ball" \
METHODS="support_v3_fixed" \
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
4 tasks x 4 paper-facing methods x 3 seeds complete
fixed-DeCE ablation cache for at least 3 tasks
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
