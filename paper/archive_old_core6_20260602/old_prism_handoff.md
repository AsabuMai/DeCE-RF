# Prism Handoff: DeCE-RF Paper Context

## Working Title

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

## Method Name

```text
DeCE-RF
```

Expanded form:

```text
Decoupled Clean-Estimate Edit-Preserve Control for Rectified Flow
```

## One-Sentence Thesis

Localized Rectified Flow image editing can be formulated as decoupled
clean-estimate displacement control: keep the source-conditioned trajectory,
decompose the desired clean displacement into edit and preserve components,
estimate their spatial geometry, and adapt their weights with clean-estimate
feedback.

## Core Story

The paper should not be framed as a general automatic image-editing method.
Instead, it should be framed as a control framework for localized RF editing.

Problem:

```text
Direct target-prompt velocity couples target editing and source preservation.
This often moves both the intended edit region and source content that should
remain fixed.
```

DeCE-RF addresses this through one decoupled clean-estimate displacement:

```text
1. What to optimize:
   edit displacement toward target clean estimates and preserve displacement
   toward source clean latents.

2. Where to optimize:
   use an operation-conditioned control geometry estimator to produce spatial
   weights.

3. How strongly to optimize:
   use clean-estimate feedback to update edit and preserve displacement weights.
```

## Method Skeleton

The RF editing dynamics are written as:

```text
v_DeCE = v_src - t^-1 Delta_0
Delta_0 = Delta_edit + Delta_pres
```

with the clean displacement components:

```text
Delta_edit = lambda_e(t) M_e * (x0_hat_tar - x0_hat_src)
Delta_pres = lambda_p(t) M_p * (x_s - x0_hat_src)
```

The implementation maps these clean displacements into `v_tar - v_src` and
`(x0_hat_src - x_s) / t`, gated by operation-conditioned control geometry.

The clean estimate is:

```text
x0_hat = x_t - t * v_t
```

Clean-space displacements are converted back into RF velocity corrections using
the linear RF path convention.

## Operation-Conditioned Support Interface

Paper-facing name:

```text
operation-conditioned control geometry estimator
```

Avoid describing this as a "support-v3 heuristic" in the paper.

The support interface consumes:

```text
target/new-token attention
host-token attention
removed/source-token attention
clean-estimate disagreement
RF velocity disagreement
optional grounding or segmentation
operation relation such as above_host, on_face, on_surface, inside
```

It outputs control geometry:

```text
M_edit
M_core
M_contact
M_preserve
```

Implementation name:

```text
operation_support_v3.py
```

## Clean-Estimate Feedback Control

The controller measures:

```text
edit progress
preservation drift
target gap in clean-estimate space
```

and dynamically adjusts:

```text
adaptive_edit_weight
adaptive_preserve_weight
projection / conflict suppression
```

Implementation mainline:

```text
support_v3_controller_rmsgap
```

Paper-facing name:

```text
DeCE-RF feedback control
```

## Claims

Use conservative claims:

```text
DeCE-RF improves the edit-preserve tradeoff under reasonable support.
```

```text
Operation-conditioned support provides a control interface for localized RF
editing, but support quality remains a bottleneck.
```

Avoid broad claims:

```text
We solve fully automatic general image editing.
```

## Primary Experiment Plan

Primary tasks:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
```

Primary methods:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

`support_v3_fixed` is an ablation/control row, not a headline main-table method.

Primary seeds:

```text
10, 11, 12
```

Main matrix:

```text
5 tasks x 4 paper-facing methods x 3 seeds = 60 runs
```

## Ablation Mapping

Contribution 1: decoupled clean-estimate displacement

```text
direct_target
support_v3_fixed without preserve term
support_v3_fixed without trajectory preserve
support_v3_fixed
```

Contribution 2: operation-conditioned control geometry

```text
attention-only or generic support
operation-conditioned support
manual/external support upper bound
```

Contribution 3: feedback-updated displacement weights

```text
support_v3_fixed
support_v3_controller_rmsgap
edit-strength sweep
support perturbation sweep
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

Controller metrics:

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
edit success
source preservation
locality
artifact severity
overall
failure flag
short note
```

## Current Local-GPU Notes

The local 12GB GPU can run smoke tests but should not define final paper
numbers unless explicitly labeled as low-VRAM diagnostic.

Known local-safe settings:

```text
512px, 28 steps, n_max 24, --low-vram, edit_field_mode=rf_diff, text guidance off
384px, 16 steps, n_max 12, text guidance on, no --low-vram
```

## Files To Give Prism

Recommended files:

```text
paper/archive_old_core6_20260602/old_core5_draft.md
paper/outline.md
paper/wacv_experiment_design.md
paper/archive_old_core6_20260602/old_core6_server_results.md
paper/tables.md
paper/figures.md
paper/limitations.md
docs/operation_conditioned_support_interface.md
docs/rmsgap_mainline_mechanism_audit.md
```

Use this handoff file as the top-level context prompt for Prism.
