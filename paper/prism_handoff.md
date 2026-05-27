# Prism Handoff: DeCE-RF Paper Context

## Working Title

```text
DeCE-RF: Decoupled Clean-Estimate Control for Localized Rectified Flow Image Editing
```

## Method Name

```text
DeCE-RF
```

Expanded form:

```text
Decoupled Clean-Estimate Control for Rectified Flow
```

## One-Sentence Thesis

Localized Rectified Flow image editing benefits from decoupling edit and
reconstruction corrections, localizing them through an operation-conditioned
support interface, and adapting their strengths with clean-estimate feedback.

## Core Story

The paper should not be framed as a general automatic image-editing method.
Instead, it should be framed as a control framework for localized RF editing.

Problem:

```text
Direct target-prompt velocity couples target editing and source preservation.
This often moves both the intended edit region and source content that should
remain fixed.
```

DeCE-RF addresses this with three linked components:

```text
1. Decouple what to optimize:
   separate edit and reconstruction/preservation velocity corrections.

2. Localize where to optimize:
   use an operation-conditioned support interface to produce control geometry.

3. Adapt how strongly to optimize:
   use clean-estimate feedback to balance edit progress and preservation drift.
```

## Method Skeleton

The RF editing dynamics are decomposed as:

```text
v_total = v_src + u_rec + u_edit
```

where:

```text
v_src
```

is the source-conditioned RF base velocity;

```text
u_rec
```

is the reconstruction/preservation correction over `M_preserve`;

```text
u_edit
```

is the target-seeking edit correction over `M_edit` and `M_core`.

The clean estimate is:

```text
x0_hat = x_t - t * v_t
```

Clean-space displacements are converted back into RF velocity corrections using
the linear RF path convention.

## Operation-Conditioned Support Interface

Paper-facing name:

```text
operation-conditioned support interface
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
backpack_remove_toy_charm
```

Primary methods:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_fixed
support_v3_controller_rmsgap
```

Primary seeds:

```text
10, 11, 12
```

Main matrix:

```text
4 tasks x 5 methods x 3 seeds = 60 runs
```

## Ablation Mapping

Contribution 1: edit/reconstruction decoupling

```text
direct_target
fixed decoupled control without rec
fixed decoupled control without trajectory preserve
fixed decoupled control
```

Contribution 2: operation-conditioned support

```text
attention-only or generic support
operation-conditioned support
manual/external support upper bound
```

Contribution 3: clean-estimate feedback

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
paper/draft.md
paper/outline.md
paper/experiment_plan.md
paper/results.md
paper/tables.md
paper/figures.md
paper/limitations.md
docs/operation_conditioned_support_interface.md
docs/rmsgap_mainline_mechanism_audit.md
```

Use this handoff file as the top-level context prompt for Prism.
