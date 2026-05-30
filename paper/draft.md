# DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing

Working title:

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

Method name:

```text
DeCE-RF
```

Expanded form:

```text
Decoupled Clean-Estimate Edit-Preserve Control for Rectified Flow
```

Working thesis:

```text
Localized RF image editing can be formulated as decoupled clean-estimate
displacement control over a source-conditioned trajectory: decompose the desired
clean displacement into edit and preserve components, estimate their spatial
geometry from operation-conditioned evidence, and adapt their weights with
clean-estimate feedback.
```

This draft should be populated only from complete core-5 runs and the fixed-mask
metric artifacts listed in `paper/results.md`.

## Introduction

Rectified-flow image editing has a useful target-driven velocity field, but a
direct target field can move both the intended edit region and source content
that should remain fixed. This project studies a decoupled clean-estimate
displacement control formulation that keeps the source-conditioned RF velocity
as the base dynamics and maps an edit-preserve clean displacement back to RF
velocity at each ODE step.

The paper claim is intentionally narrow: localized RF editing benefits from
casting edit and preservation as two spatial components of the same clean
displacement. The support interface estimates where edit and preserve
displacements should act, and clean-estimate feedback adjusts their weights
online. The method is not presented as a fully general automatic editor;
failures with poor support remain part of the evidence.

## Related Work

The final related-work section should cover RF/ODE editing methods and
source-reference controls, including h-Edit, FlowEdit, RF-Solver, FireFlow,
ReFlex, and PnP/P2P-style attention or feature reuse. External baselines should
be reported only when they are run under matched image, prompt, resolution,
seed, and mask conditions.

## Method

The method writes controlled RF editing dynamics as:

```text
v_DeCE = v_src - t^-1 Delta_0
Delta_0 = Delta_edit + Delta_pres
```

Here `v_src` is the source-conditioned RF velocity and `Delta_0` is the desired
clean displacement. `Delta_edit` moves edit regions toward the target clean
estimate, while `Delta_pres` pulls preserve regions back toward the source
latent. The implementation maps these clean displacements into the familiar RF
velocity directions `v_tar - v_src` and `(x0_hat_src - x_s) / t`.

The preserve branch uses the linear RF clean estimate

```text
x0_hat = x_t - t * v_theta(x_t, t)
```

and maps clean-space displacement back to a velocity correction with the
linear-path convention. The edit branch combines target-source clean
displacement, clean-space anchor terms, region terms, optional text/image reward
gradients, surface-reference displacements, and experimental source-reference
Q/K/V injection. These terms are described as edit-side displacement
instantiations unless a branch explicitly computes an autograd gradient.

The spatial interface is operation-conditioned rather than object-specific. It
does not encode rules such as "if sunglasses, use this box." Instead, each task
is reduced to an operation type and relation:

```text
add_object, add_decal, remove_object, replace, recolor
```

For each operation, the support interface fuses evidence maps:

```text
target/new-token attention
host-token attention
removed/source-token attention
clean-estimate disagreement
RF velocity disagreement
optional grounding or segmentation
operation relation such as above_host, on_face, on_surface, inside
```

The output of this interface is not the final edit itself. It is the control
geometry:

```text
M_edit, M_core, M_contact, M_preserve
```

`M_edit` and `M_core` gate edit displacement, `M_contact` allows weak boundary
blending, and `M_preserve` gates preserve displacement. This keeps the
support module as a principled interface between operation semantics and
clean displacement control, not as the central claim.

Finally, the clean-estimate controller measures edit progress and preservation
drift along the trajectory. When the edit-region clean estimate remains far
from the target change, it increases the edit weight. When the preserve-region
clean estimate drifts from the source, it increases the reconstruction weight
and projects away conflicting edit components. Thus the final method is:

```text
decouple what to optimize,
localize where each correction acts,
adapt how strongly each correction is applied.
```

## Experiments

The current completed matrix is the core-5 DeCE-RF matrix documented in
`paper/results.md`. The main tasks are:

- `cat_crown`: compact above-host accessory insertion.
- `dog_sunglasses`: face/accessory insertion positive control.
- `mug_heart`: rigid surface decal placement.
- `tshirt_star`: clothing-surface decal placement.
- `backpack_remove_toy_charm`: exposed-object removal and preservation probe.

Each task is evaluated with:

- RF reconstruction / base reconstruction.
- Direct target guidance.
- Generic support control.
- DeCE-RF.

The required seeds are 10, 11, and 12. Each run must have `result.png`,
`stats.json`, `metadata.json`, `command.txt`, and any generated masks.

Metrics are generated by `scripts/evaluate_paper_metrics.py` and include
trajectory summaries, source L1/RMSE, luma SSIM, mask-outside drift, mask-inside
change, runtime, peak GPU memory when available, optional CLIP edit alignment,
optional DINO source similarity, and manual failure flags.

The current 60-row core-5 matrix has complete fixed-mask metrics and internal
visual audit annotations. DeCE-RF is strongest on localized add/decal tasks
(`cat_crown`, `mug_heart`, `tshirt_star`), remains competitive on
`dog_sunglasses`, and succeeds visually on the exposed-object removal task
`backpack_remove_toy_charm` despite weak global CLIP removal scores. The
defensible claim is an edit-preserve control result under reasonable support,
not broad arbitrary editing.

The current fixed-control ablation cache covers the original core-4 task set.
It should be used as component evidence for feedback-updated control, not as a
headline main-table method. Completing `support_v3_fixed` for any promoted
final task set remains the next ablation cleanup item.

## Failure Analysis

Known failure modes should remain visible:

- object replacement can become a target-texture or face-overlay failure rather
  than a true object rewrite;
- local accessory insertion can trade placement quality against source
  preservation;
- poor spatial support can produce mask artifacts or suppress the intended
  edit;
- exposed-object removal is easier than occluded removal requiring host or
  background completion;
- accurate support can still leave residual labels, marks, or transformed
  objects when local target formation is weak;
- replacement and recolor remain candidate expansion/stress tasks rather than
  part of the current core-5 claim.

## Limitations

- The method depends on reliable spatial support for local edits.
- The implementation is SD3-specific.
- External baselines require matched conditions before they can support a
  paper claim; current FireFlow/ReFlex artifacts are exploratory only.
- CLIP/DINO metrics are populated, but LPIPS/VLM-style perceptual or
  instruction-following metrics are not yet included.
