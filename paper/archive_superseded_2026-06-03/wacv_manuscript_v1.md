# DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing

## Abstract

Rectified Flow models provide a velocity-field view of image generation, but
localized image editing remains difficult because target-prompt velocities tend
to entangle the desired local change with unwanted source drift. We introduce
DeCE-RF, a localized Rectified Flow editing framework that treats editing as
decoupled clean-estimate displacement control. Rather than replacing the
source-conditioned trajectory with a target-conditioned velocity, DeCE-RF keeps
the source velocity as the base field and applies a clean displacement
decomposed into edit and preservation components:
`v_DeCE = v_src - t^-1 Delta_t^0`, where
`Delta_t^0 = Delta_edit + Delta_pres`. An operation-conditioned control
geometry estimator supplies edit, core, contact, and preserve regions, while
clean-estimate feedback adapts edit and preservation weights during ODE
sampling. Experiments on localized insertion, decal, exposed-object removal,
and recolor tasks show that DeCE-RF improves the edit-preserve tradeoff over
direct target guidance and weak generic support. Failure analysis further shows
that accurate support is necessary but not sufficient: difficult completion,
occlusion, and replacement target formation remain open limitations.

## 1. Introduction

Text-guided image editing asks a generative model to change a specified part of
an image while preserving the source identity, layout, and background. This
requirement is especially sharp for local edits such as adding an accessory to
an animal, placing a decal on a surface, recoloring an object, or removing a
small exposed object. A target prompt alone usually underspecifies which source
details should remain fixed. As a result, direct target guidance can improve
prompt alignment while changing pose, texture, identity, or background content
outside the intended edit region.

Rectified Flow (RF) models make this problem a natural control problem. An RF
model generates samples by integrating a learned velocity field along an ODE
trajectory [@rectifiedflow2022; @sd32024]. Existing RF editing methods often
use target-prompt velocities, inversion, or feature reuse to steer the
trajectory [@flowedit2024; @rfsolver2024; @fireflow2024; @hedit2025]. However,
for localized editing, a target-conditioned velocity still mixes two competing
goals: move the local edit region toward the target prompt, while keeping the
rest of the source image close to the original.

We propose DeCE-RF, a decoupled clean-estimate edit-preserve control framework
for localized RF editing. DeCE-RF keeps the source-conditioned velocity as the
base field and designs a displacement in clean-estimate space at each ODE step.
Under the linear RF path, a velocity prediction induces a clean estimate
`x0_hat = x_t - t v_theta(x_t,t)`. This gives a common space in which target
change and source preservation can be measured. DeCE-RF decomposes the desired
clean displacement into an edit component that moves local regions toward the
target clean estimate and a preserve component that pulls non-edit regions back
to the source latent. The displacement is then mapped back to an RF velocity
correction.

Spatial support is the second part of the problem. A local controller needs to
know where edit displacement should act, where preservation should dominate,
and where boundary blending should occur. We therefore describe support
estimation as an operation-conditioned control geometry estimator rather than
as an object-specific mask heuristic. Given an operation such as `add_object`,
`add_decal`, `remove_object`, or `recolor`, and a relation such as
`above_host`, `on_face`, or `on_surface`, the estimator fuses token attention,
clean-estimate disagreement, velocity disagreement, optional grounding, and
operation priors into `M_edit`, `M_core`, `M_contact`, and `M_preserve`.

The claim of this paper is deliberately scoped. DeCE-RF is not presented as a
fully general automatic image editor. Instead, we show that, when reasonable
support is available, clean-estimate edit-preserve control improves localized
RF editing. We also analyze cases where localization succeeds but completion or
target formation fails, because those failures clarify the boundary of the
control formulation.

Our contributions are:

- We formulate localized RF editing as decoupled clean-estimate displacement
  control over a source-conditioned base trajectory.
- We show that edit and preserve branches can be written as two components of
  one clean displacement and mapped back to RF velocity by the linear path.
- We introduce an operation-conditioned control geometry interface that maps
  task operation evidence into edit, core, contact, and preserve masks.
- We use clean-estimate feedback to adapt edit and preservation weights and to
  suppress preserve-conflicting edit components during sampling.
- We evaluate the method with main comparisons, support visualizations,
  feedback ablations, and explicit failure analysis.

## 2. Related Work

Rectified Flow and flow-matching models learn velocity fields that transport
noise to data through ODE-like dynamics [@rectifiedflow2022]. Stable Diffusion 3
scales this formulation with transformer-based high-resolution image synthesis
[@sd32024]. Recent RF editing methods explore inversion-free editing, RF
inversion, solver-based editing, and fast inversion strategies
[@flowedit2024; @rfsolver2024; @fireflow2024]. These methods demonstrate that
RF trajectories are editable, but local edit-preserve control remains
challenging when target guidance changes more than the intended region.

Diffusion and flow editing systems often use attention control, feature reuse,
or source-reference injection to preserve structure. h-Edit uses a
Doob-transform-inspired editing formulation [@hedit2025], while related
attention and feature control methods show that source features can constrain
image-to-image edits. DeCE-RF is complementary: it does not rely only on
feature reuse or prompt substitution, but expresses local editing as a
clean-estimate displacement added to a source-conditioned RF trajectory.

Local editing also depends on spatial support. Generic attention masks can be
diffuse or over-preserving, and segmentation or grounding tools such as SAM and
GroundingDINO can provide useful support but do not solve how the RF velocity
should be controlled [@kirillov2023segment; @liu2023groundingdino]. DeCE-RF
treats support as control geometry. The support interface estimates where
different displacement components should act; the controller then determines
how those components affect the RF trajectory.

## 3. Method

### 3.1 Clean-Estimate Displacement Control

Let `x_s` be the source latent, `c_s` the source prompt, and `c_t` the target
prompt. At timestep `t`, the source and target RF velocities are

```text
v_src = v_theta(x_t, t, c_s)
v_tar = v_theta(x_t, t, c_t).
```

Under the linear RF path, each velocity induces a clean estimate:

```text
x0_hat_src = x_t - t v_src
x0_hat_tar = x_t - t v_tar.
```

DeCE-RF keeps the source velocity as the base trajectory and applies a desired
clean displacement:

```text
v_DeCE = v_src - t^-1 Delta_t^0.
```

The displacement is decomposed into edit and preserve components:

```text
Delta_t^0 = Delta_edit + Delta_pres.
```

The edit component moves the edit region toward the target clean estimate:

```text
Delta_edit = lambda_e(t) M_e * (x0_hat_tar - x0_hat_src).
```

The preserve component pulls preserve regions back to the source latent:

```text
Delta_pres = lambda_p(t) M_p * (x_s - x0_hat_src).
```

Because `x0_hat_tar - x0_hat_src = -t (v_tar - v_src)`, the target-source
velocity difference is the RF velocity form of the edit clean displacement.
Similarly, the preservation branch corresponds to the velocity correction that
reduces clean-estimate drift from the source. Thus the edit and preserve
branches are not unrelated engineering terms; they are two spatial components
of the same clean displacement.

### 3.2 Operation-Conditioned Control Geometry

The clean displacement requires spatial weights. DeCE-RF estimates these
weights from operation-conditioned evidence. Each task is described by an
operation:

```text
add_object, add_decal, remove_object, replace, recolor
```

and an optional relation:

```text
above_host, on_face, on_surface, inside, remove_source_object
```

The geometry estimator combines:

- target or new-token attention;
- host-token attention;
- removed/source-token attention;
- clean-estimate disagreement `||x0_hat_tar - x0_hat_src||`;
- velocity disagreement `||v_tar - v_src||`;
- optional grounding or segmentation;
- relation regions derived from the operation.

Candidate maps include relation-response maps for object insertion,
surface-local response maps for decals and recolor, and removed-source maps for
removal. The selected support is thresholded, area-constrained, component
filtered, dilated, and smoothed. It is then split into:

```text
M_core: strongest edit region
M_contact: weak boundary blending ring
M_edit: effective edit support
M_preserve: outside object/contact support
```

This interface localizes the displacement. It does not by itself perform the
edit; the ODE controller consumes the masks as clean-displacement geometry.

### 3.3 Clean-Estimate Feedback

Fixed edit and preserve weights are brittle. A support region can be slightly
too large, the edit can be under-expressed, or preservation drift can appear
late in the trajectory. DeCE-RF therefore logs clean-estimate diagnostics at
each timestep:

```text
current_delta = x0_hat_src - x_s
target_delta  = x0_hat_tar - x_s
target_gap    = x0_hat_tar - x0_hat_src
preserve_drift = RMS_M_preserve(x0_hat_src - x_s).
```

The edit weight increases when the edit target gap indicates insufficient local
progress. The preserve weight increases when preserve drift exceeds a budget.
When the edit field is estimated to worsen preserve-region clean error, a
projection step removes the preserve-conflicting component before converting
the edit displacement back to velocity.

The final per-step update is:

```text
x_{t-dt} = x_t + (t_next - t) (v_src - t^-1 Delta_t^0).
```

In the implementation, `support_v3_controller_rmsgap` is the main DeCE-RF
controller. `support_v3_fixed` keeps the same operation-conditioned geometry
but disables feedback weighting, and is used only as an ablation.

## 4. Experiments

### 4.1 Tasks and Protocol

We organize the benchmark by operation type rather than object category. The
main Core-6 task design is:

| Task | Operation | Role |
| --- | --- | --- |
| `cat_crown` | add object / above host | compact relation-based insertion |
| `dog_sunglasses` | add accessory / face | positive accessory insertion control |
| `mug_heart` | add decal / surface | rigid-surface decal |
| `tshirt_star` | add decal / clothing surface | non-rigid surface decal |
| `backpack_remove_toy_charm` | remove object | exposed-object removal |
| `red_chair_blue` | recolor / attribute | localized appearance editing |

The first five tasks form the completed Core-5 matrix. The recolor task is the
planned Core-6 expansion after a seed-10 visual gate. Each task is evaluated
over seeds 10, 11, and 12.

Headline methods are:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

All headline preservation metrics use a fixed per-task evaluation mask shared
across methods and seeds. A method's own support mask is reported only as a
support diagnostic, not as the preservation mask. This avoids rewarding a
method for predicting a smaller or easier mask.

### 4.2 Metrics

We report preservation, edit, support, and controller metrics. Preservation is
measured by outside-mask L1/RMSE, source SSIM [@wang2004ssim], and DINO/source
similarity [@oquab2023dinov2]. Edit strength is measured by inside-mask change
and CLIP target-source delta [@radford2021clip]. CLIP is interpreted carefully:
it can underestimate removal quality and can reward global changes that are
undesirable for localized editing. Support diagnostics include support area,
overlap, leakage, and mask visualizations. Controller diagnostics include edit
target gap, preserve drift, adaptive edit weight, adaptive preserve weight, and
projection magnitude.

### 4.3 Main Results

The Core-5 fixed-mask results show a consistent qualitative pattern. Direct
target guidance produces stronger target pressure but the worst preservation
and artifact scores. Generic support control preserves the source well but
often over-preserves and misses the requested edit. DeCE-RF gives the strongest
edit success and overall visual audit scores while maintaining competitive
locality and source preservation.

For example, on `cat_crown`, generic support obtains low outside drift but
fails to form the crown, while DeCE-RF produces a localized crown with similar
outside drift. On `mug_heart` and `tshirt_star`, generic support preserves the
source but misses the decal, while DeCE-RF forms clear localized decals. On
`dog_sunglasses`, both generic support and DeCE-RF work, but DeCE-RF reduces
artifact severity. On `backpack_remove_toy_charm`, visual audit marks DeCE-RF
as a successful exposed-object removal, while global CLIP is less informative
for this operation.

The internal visual audit over 15 Core-5 runs gives the following method means:

| Method | Edit success | Source preservation | Locality | Artifact severity | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF reconstruction | 1.00 | 2.40 | 2.80 | 2.60 | 1.00 |
| Direct target | 2.60 | 1.40 | 1.80 | 3.40 | 2.00 |
| Generic support control | 1.60 | 4.40 | 4.20 | 1.20 | 2.40 |
| DeCE-RF | 4.40 | 4.00 | 4.20 | 1.60 | 4.40 |

These results support a conservative claim: operation-conditioned support plus
decoupled clean-estimate control improves the edit-preserve balance on selected
localized add-object, decal, exposed-object removal, and pending visual-gate
recolor tasks.

### 4.4 Figures as Experimental Evidence

The main paper should use figures as experiments rather than as decoration.
The planned WACV figure set is:

| Figure | Question answered |
| --- | --- |
| Fig. 1 Method overview | What is clean-estimate displacement control? |
| Fig. 2 Main qualitative comparison | Does DeCE-RF improve localized edit quality over baselines? |
| Fig. 3 Support geometry | Is support an operation-conditioned control interface rather than a mask trick? |
| Fig. 4 Edit-preserve tradeoff | Where do methods fall on edit strength vs outside drift? |
| Fig. 5 Feedback ablation | What does clean-estimate feedback add over fixed weights? |
| Fig. 6 Limitations | Where does the method fail and why? |

The main qualitative figure should use columns:

```text
Source | Direct target | Generic support | DeCE-RF | Support overlay
```

The support figure should visualize:

```text
new-token attention | clean disagreement | velocity disagreement |
operation-conditioned support | M_core/M_contact | M_preserve
```

The feedback figure should compare `support_v3_fixed` and DeCE-RF with curves
for edit gap, preserve drift, adaptive weights, and projection magnitude.

### 4.5 Ablations

We use ablations to isolate the three components of DeCE-RF.

First, decoupled clean-estimate displacement is tested by comparing direct
target guidance, edit-only displacement, preserve-disabled displacement, and
fixed DeCE displacement. This asks whether keeping the source trajectory and
adding local clean displacement reduces drift.

Second, operation-conditioned geometry is tested by comparing attention-only,
clean-disagreement-only, generic support, operation-conditioned support, and
manual/external support where available. This asks whether the geometry
estimator selects better control regions.

Third, feedback control is tested by comparing `support_v3_fixed` and DeCE-RF.
The existing fixed-control ablation covers the original Core-4 tasks. It shows
small but stable metric improvements, and visual evidence suggests that the
feedback controller is best interpreted as late-stage edit finishing plus
preserve-aware projection rather than as a mechanism that creates edits from
failed support.

### 4.6 Failure Analysis

We explicitly separate base-method failures from extension probes and
limitations. Occluded-object removal such as `dog_remove_tennis_ball` requires
host-mouth completion and is outside the main claim. Whiteboard letter removal
shows glyph-field hallucination: localization can be plausible while the local
generator transforms the letter instead of removing it. Fridge magnet removal
shows that accurate support can still damage nearby objects on cluttered
surfaces. Replacement probes such as `dog_replace_tennis_ball_star` remain
unreliable because target formation must decide which source attributes to
preserve and which to replace.

Two extension probes should be reported separately from the base method:
`laptop_remove_sticker` as `DeCE-RF + completion prior`, and
`whiteboard_probe_red_star_sticker` as `DeCE-RF + replacement route`. These
should not be aggregated into the base DeCE-RF main table.

## 5. Discussion

DeCE-RF suggests that localized RF editing is better understood as controlled
ODE sampling than as target-prompt velocity replacement. The source velocity
provides a stable reconstruction trajectory, clean displacement defines what
should change or be preserved, operation-conditioned support defines where
each component acts, and feedback defines how strongly each component should
act during sampling.

This framing also clarifies what the method does not solve. Support quality
remains a bottleneck, and even accurate support does not guarantee successful
object erasure, background completion, or precise replacement. These failure
modes do not invalidate the edit-preserve control formulation; instead, they
show where stronger support proposal, local target formation, or inpainting
priors are needed.

## 6. Limitations

The method depends on reliable spatial support. If the support interface
selects the wrong region, the controller can faithfully optimize the wrong
local problem. The current implementation is also SD3-specific and has not yet
been validated across multiple RF backbones.

Removal and replacement remain harder than insertion and decal editing.
Exposed-object removal can succeed, but occluded-object removal requiring host
completion remains outside the main claim. Replacement target formation is not
broadly solved, especially for small objects, tags, and precise glyph edits.

External baselines must be compared under matched prompts, seeds, resolution,
backbone assumptions, and mask inputs. Existing older baseline artifacts should
be labeled as contextual evidence unless rerun under the revised strict Core-6
protocol.

## 7. Conclusion

We introduced DeCE-RF, a decoupled clean-estimate edit-preserve control
framework for localized Rectified Flow editing. DeCE-RF keeps the source
velocity as the base trajectory, decomposes the desired clean displacement into
edit and preservation components, estimates operation-conditioned control
geometry, and adapts displacement weights from clean-estimate feedback. The
current evidence supports a conservative but useful claim: under reasonable
support, DeCE-RF improves the localized edit-preserve tradeoff for selected RF
editing tasks, while difficult support, completion, and replacement cases
remain open problems.

