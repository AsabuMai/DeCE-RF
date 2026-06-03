# DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing

## Abstract

Rectified Flow models offer a velocity-field view of image generation, but
localized image editing with such models remains difficult because
target-conditioned velocities tend to couple the desired local change with
unwanted source drift. This paper introduces DeCE-RF, a decoupled
clean-estimate edit-preserve control framework for localized Rectified Flow
editing. Instead of replacing the source-conditioned trajectory with a
target-conditioned velocity, DeCE-RF keeps the source velocity as the base
field and designs a clean-estimate displacement at each ODE step. The
controlled velocity is written as
`v_DeCE = v_src - t^-1 Delta_t^0`, where the clean displacement
`Delta_t^0` is decomposed into an edit component that moves localized regions
toward the target clean estimate and a preservation component that pulls
non-edit regions back toward the source latent. An operation-conditioned
control geometry estimator supplies edit, core, contact, and preserve regions,
while clean-estimate feedback adapts the edit and preservation weights during
sampling. The resulting framework separates three questions that are often
entangled in local editing: what clean displacement should be applied, where it
should act, and how strongly its components should be weighted along the
trajectory. The method is intentionally scoped as a localized RF control
framework rather than a fully general automatic image editor; difficult support,
completion, and replacement cases remain explicit limitations.

## 1. Introduction

Text-guided image editing asks a generative model to change a specified aspect
of an input image while preserving the source identity, layout, and background.
This edit-preserve requirement is especially demanding for local edits: adding
an accessory to an animal should not alter the animal's pose; placing a decal
on a mug or shirt should not regenerate the object; recoloring a chair should
not move its geometry; removing a small object should not damage the surrounding
surface. A target prompt by itself rarely specifies all of these preservation
constraints. Consequently, direct target guidance can improve semantic
alignment while changing source content outside the intended edit region.

Rectified Flow (RF) models make this problem particularly natural to study as
a control problem. RF generation is expressed through a learned velocity field
integrated along an ODE trajectory [@rectifiedflow2022; @sd32024]. Existing RF
editing methods show that these trajectories can be steered through inversion,
target guidance, solver choices, or feature reuse [@flowedit2024;
@rfsolver2024; @fireflow2024; @hedit2025]. However, for localized editing, a
target-conditioned velocity remains a blunt primitive: it pushes the sample
toward the target prompt without explicitly separating the local edit from the
source content that should remain fixed.

This paper proposes DeCE-RF, a decoupled clean-estimate edit-preserve control
framework for localized RF editing. The key design choice is to keep the
source-conditioned velocity as the base trajectory and express editing as a
clean-estimate displacement applied on top of that trajectory. Under the RF
linear path, a velocity prediction induces a clean estimate,
`x0_hat = x_t - t v_theta(x_t,t)`. This clean estimate provides a common space
in which target-directed change and source preservation can be measured. Rather
than treating editing and reconstruction as unrelated guidance terms, DeCE-RF
casts them as two spatial components of one clean displacement.

The clean-estimate displacement is decomposed into an edit component and a
preservation component. The edit component moves a localized region from the
source clean estimate toward the target clean estimate. The preservation
component pulls non-edit regions from the current source-conditioned clean
estimate back toward the original source latent. Because RF clean estimates are
linearly related to velocity, this desired clean displacement can be mapped
back to a velocity correction with a simple `-1/t` factor. The resulting
controlled velocity takes the compact form:

```text
v_DeCE = v_src - t^-1 Delta_t^0,
Delta_t^0 = Delta_edit + Delta_pres.
```

This formulation separates what should change from what should be preserved,
but it does not by itself solve where the components should act. A localized
controller also needs spatial geometry: an edit region, a stronger core, a
soft contact or transition region, and a preserve region. DeCE-RF therefore
uses an operation-conditioned control geometry estimator. The estimator
receives operation-level information such as `add_object`, `add_decal`,
`remove_object`, or `recolor`, plus relations such as `above_host`, `on_face`,
or `on_surface`. It fuses token attention, clean-estimate disagreement,
velocity disagreement, optional grounding or segmentation, and relation priors
into `M_edit`, `M_core`, `M_contact`, and `M_preserve`. In the paper, this
module should be understood as a geometry interface for clean displacement
control, not as the central contribution or as a collection of object-specific
mask rules.

The final part of the framework is feedback. A fixed edit strength can
under-edit when the target signal is weak, while a fixed preservation strength
can either over-constrain target formation or allow drift to accumulate. Since
RF sampling produces clean estimates at every timestep, DeCE-RF measures edit
progress and preservation drift during the ODE trajectory. These measurements
adapt the edit and preservation weights online and allow the controller to
project away edit components that are estimated to worsen preserve-region
clean error.

The scope of this paper is deliberately conservative. DeCE-RF is not presented
as a universal automatic image editor. Its claim is that localized RF editing
benefits from formulating edit and preservation as decoupled clean-estimate
displacement components over a source-conditioned trajectory, provided that
reasonable spatial support is available. Cases where support is wrong, where
local target formation is weak, or where object removal requires difficult
completion are treated as limitations rather than hidden behind a broad success
claim.

The contributions are:

- A clean-estimate displacement formulation for localized Rectified Flow
  editing that keeps the source-conditioned velocity as the base trajectory.
- A decomposition of the desired clean displacement into edit and preservation
  components, both mapped back to RF velocity through the linear path.
- An operation-conditioned control geometry interface that estimates edit,
  core, contact, and preserve regions from semantic and RF response evidence.
- A clean-estimate feedback controller that adapts edit and preservation
  weights and suppresses preserve-conflicting edit components during sampling.
- A scoped experimental framing that separates main localized-edit evidence
  from support failures, completion limitations, and extension probes.

## 2. Related Work

Rectified Flow and flow-matching models define generation through velocity
fields that transport noise to data along continuous trajectories
[@rectifiedflow2022]. Stable Diffusion 3 scales this paradigm with
transformer-based high-resolution image synthesis [@sd32024]. This velocity
view is attractive for editing because it exposes the sampling dynamics as an
object of control. Instead of editing only by prompt replacement, one can ask
which velocity field should be integrated and how it should be modified along
the trajectory.

Recent RF editing methods explore several ways of steering these dynamics.
FlowEdit studies inversion-free text-based editing with pretrained flow models
[@flowedit2024]. RF-Solver-Edit and related inversion-based methods investigate
how solver and inversion choices affect editing fidelity [@rfsolver2024].
FireFlow focuses on fast RF inversion for semantic editing [@fireflow2024].
h-Edit develops a flexible diffusion-based editing framework through a
Doob-transform perspective [@hedit2025]. These methods establish that RF and
ODE trajectories can support real-image editing, but local edit-preserve
control remains a distinct problem: a target velocity can still move source
content that should remain fixed.

DeCE-RF differs by making the edit-preserve decomposition explicit in
clean-estimate space. Rather than choosing between source and target velocities
as whole-field alternatives, the method keeps the source velocity as the base
and adds a spatially gated clean displacement. This makes preservation a
first-class component of the same control equation as editing. The method is
therefore complementary to inversion and solver improvements: better inversion
or RF backbones can provide a stronger trajectory, while DeCE-RF specifies how
local edit and preservation corrections should be organized on that trajectory.

Local image editing also depends on spatial support. Attention maps can
provide useful localization signals, but they are often diffuse, prompt
dependent, and poorly aligned with operation boundaries. Grounding and
segmentation tools such as GroundingDINO and Segment Anything can provide
stronger external evidence [@liu2023groundingdino; @kirillov2023segment], but
a segmentation mask alone does not specify how to balance editing and
preservation across the RF trajectory. DeCE-RF treats support as control
geometry: the support module outputs regions for edit, core, contact, and
preservation, and the controller uses those regions to construct clean
displacement components.

Evaluation of local editing also requires care. CLIP-based metrics
[@radford2021clip] can measure text-image alignment but can reward global
changes that are undesirable in local editing. DINO-style representation
similarities [@oquab2023dinov2] and SSIM [@wang2004ssim] provide preservation
signals, but they can understate successful local edits when the intended edit
is visually salient. DeCE-RF is therefore framed around edit-preserve
tradeoffs, fixed evaluation masks, support diagnostics, and qualitative
failure analysis rather than a single scalar metric.

## 3. Method

### 3.1 Problem Setting

Given a source image `I_s`, a source prompt `c_s`, and a target prompt `c_t`,
the goal is to generate an edited image that satisfies the target prompt only
in the intended local region while preserving the source elsewhere. The source
image is encoded into a latent `x_s`. The method first obtains an editable
latent trajectory through source-conditioned RF inversion, then integrates a
controlled reverse ODE from the noisy latent toward the edited output.

At each timestep, the model can be queried under the source condition and the
target condition:

```text
v_src = v_theta(x_t, t, c_s),
v_tar = v_theta(x_t, t, c_t).
```

A direct target method uses `v_tar` or a target-heavy mixture as the editing
field. This can increase target alignment, but it does not encode the locality
constraint. DeCE-RF instead keeps `v_src` as the base field and adds a local
clean-estimate displacement correction.

### 3.2 Rectified Flow Clean Estimates

We use the linear RF path convention:

```text
x_t = (1 - t) x_0 + t x_1,
```

where `x_0` is a clean latent and `x_1` is a noise latent. Given a velocity
prediction at timestep `t`, the corresponding clean estimate is:

```text
x0_hat(x_t,t,c) = x_t - t v_theta(x_t,t,c).
```

For source and target prompts:

```text
x0_hat_src = x_t - t v_src,
x0_hat_tar = x_t - t v_tar.
```

These clean estimates make the local editing objective explicit. If the target
condition changes a local object, `x0_hat_tar - x0_hat_src` indicates the clean
displacement implied by the target prompt. If the source-conditioned trajectory
drifts away from the source latent, `x_s - x0_hat_src` indicates the clean
displacement needed for preservation.

The same estimates also provide spatial evidence. Clean-estimate disagreement
is:

```text
D_clean(i,j) = ||x0_hat_tar(:,i,j) - x0_hat_src(:,i,j)||_2.
```

Velocity disagreement is:

```text
D_vel(i,j) = ||v_tar(:,i,j) - v_src(:,i,j)||_2.
```

These maps are used by the support interface because they show where the
target condition changes RF dynamics relative to the source condition.

### 3.3 Decoupled Clean Displacement

The central control object in DeCE-RF is a desired clean displacement
`Delta_t^0`. The controlled velocity is:

```text
v_DeCE = v_src - t^-1 Delta_t^0.
```

This follows directly from the clean-estimate relation. If a velocity
correction `u_t` is added to the source velocity, then:

```text
x0_hat(v_src + u_t) = x_t - t (v_src + u_t)
                    = x0_hat_src - t u_t.
```

To induce a desired clean displacement
`x0_hat(v_src + u_t) = x0_hat_src + Delta_t^0`, the correction must satisfy:

```text
u_t = -t^-1 Delta_t^0.
```

DeCE-RF decomposes this clean displacement into edit and preservation
components:

```text
Delta_t^0 = Delta_edit + Delta_pres.
```

The edit component is:

```text
Delta_edit = lambda_e(t) M_e * (x0_hat_tar - x0_hat_src).
```

It moves the edit support toward the target clean estimate. Since
`x0_hat_tar - x0_hat_src = -t (v_tar - v_src)`, the familiar target-source
velocity direction is the velocity form of this clean edit displacement.

The preservation component is:

```text
Delta_pres = lambda_p(t) M_p * (x_s - x0_hat_src).
```

It pulls preserve regions back toward the source latent. Its corresponding
velocity correction is proportional to `(x0_hat_src - x_s) / t`, gated by the
preserve mask. Preservation is therefore not a post-hoc penalty; it is a
clean-displacement component in the same control equation as editing.

This decomposition yields a concise interpretation of the method. The source
velocity provides the base trajectory, the edit displacement introduces target
change where allowed, and the preservation displacement cancels source drift
where the image should remain stable.

### 3.4 Operation-Conditioned Control Geometry

The displacement equation requires masks. DeCE-RF estimates them through an
operation-conditioned support interface. Each task is described by an operation
label:

```text
add_object, add_decal, remove_object, replace, recolor
```

and, when useful, a relation label:

```text
above_host, on_face, on_surface, inside, remove_source_object.
```

The support interface avoids object-specific rules. A crown above a cat and a
badge above another host should both be handled as `add_object` with an
`above_host` relation; sunglasses and similar accessories are `add_object`
with an `on_face` relation; decals and surface marks are `add_decal` with an
`on_surface` relation. The operation and relation specify the kind of control
geometry needed, not a template for a particular object.

The estimator fuses several evidence maps:

```text
A_new       target/new-token attention
A_host      host-token attention
A_removed   removed/source-token attention
D_clean     clean-estimate disagreement
D_vel       RF velocity disagreement
G           optional grounded or segmented mask
R           operation relation map
```

Candidate support maps are normalized products or mixtures of these signals.
For example, an above-host insertion can use a relation-response candidate,
a decal can use a surface-local response candidate, and a removal task can use
removed-source or segmentation evidence. The selected candidate is
thresholded, constrained by area, filtered by connected components, dilated,
and smoothed.

The output is a layered control geometry:

```text
M_core       strongest edit region
M_contact    soft transition ring
M_edit       effective edit support
M_preserve   preservation region
```

`M_core` receives the strongest edit displacement. `M_contact` permits weak
boundary blending. `M_preserve` gates the preservation displacement. This
layering is important because a local edit often requires a small object or
surface region to change while nearby structure remains stable.

### 3.5 Clean-Estimate Feedback Control

Fixed displacement weights can be too rigid. If the edit signal is weak, a
fixed edit weight may under-edit. If the preserve region begins to drift, a
fixed preserve weight may not respond. If the edit field conflicts with
preservation, simply increasing both terms can worsen artifacts. DeCE-RF uses
clean-estimate feedback to adapt the component weights during sampling.

At each timestep, the controller measures:

```text
current_delta = x0_hat_src - x_s,
target_delta  = x0_hat_tar - x_s,
target_gap    = x0_hat_tar - x0_hat_src.
```

In the edit region, target-gap and progress diagnostics estimate whether the
local target change is being expressed. In the preserve region, the controller
measures:

```text
preserve_drift = RMS_M_preserve(x0_hat_src - x_s).
```

The edit weight increases when the edit deficit is active:

```text
w_edit = clip(1 + k_e deficit_e, w_e_min, w_e_max).
```

The preserve weight increases when preserve drift exceeds a budget:

```text
w_preserve = clip(1 + k_p max(0, preserve_drift - tau_p),
                  w_p_min, w_p_max).
```

The displacement components are then scaled by these weights before being
converted to velocity. In addition, DeCE-RF estimates whether the edit
displacement would increase preserve-region clean error. When such a conflict
is detected, a projection step removes the destructive component from the edit
displacement. The feedback controller should therefore be understood as
late-stage edit finishing plus preserve-aware conflict suppression, not as a
replacement for correct support or strong target formation.

### 3.6 ODE Integration and Diagnostics

The controlled reverse update is:

```text
x_{t-dt} = x_t + (t_next - t) v_DeCE.
```

The implementation logs per-step diagnostics: edit target gap, edit progress,
preserve drift, adaptive edit weight, adaptive preserve weight, projection
norm, edit guidance norm, reconstruction guidance norm, and total velocity
norm. These diagnostics are important for the paper because the method is a
controller, not only a final-image generator. The intended evidence is not
only that the final image is better, but that the controller behaves
interpretable along the trajectory.

### 3.7 Algorithm

```text
Input:
  source image I_s
  source prompt c_s
  target prompt c_t
  operation o and relation rho

1. Encode I_s into source latent x_s.
2. Invert x_s with the source-conditioned RF ODE to obtain z_T.
3. Estimate support geometry:
     compute source and target velocities;
     compute clean and velocity disagreement maps;
     collect token attention and optional grounding;
     build operation-conditioned support candidates;
     select and postprocess M_core, M_contact, M_edit, M_preserve.
4. For each reverse ODE timestep:
     compute v_src and v_tar;
     compute x0_hat_src and x0_hat_tar;
     construct Delta_edit on M_edit / M_core;
     construct Delta_pres on M_preserve;
     measure edit gap and preserve drift;
     update edit and preserve weights;
     project away preserve-conflicting edit displacement when needed;
     update x_t using v_src - t^-1 (Delta_edit + Delta_pres).
5. Decode the final latent to the edited image.
```

## 4. Experiment Section Placeholder

The experimental results section is intentionally not drafted here because the
final runs are still in progress. The section should be filled only after the
Core-6 matrix, fixed-mask metrics, visual audit, and selected figure assets are
available. The planned structure is:

```text
4.1 Tasks and protocol
4.2 Metrics and fixed evaluation masks
4.3 Main comparison
4.4 Support geometry analysis
4.5 Feedback ablation
4.6 Failure analysis
```

The main paper-facing methods should remain:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

`support_v3_fixed` should be reported only in the component ablation, not as a
headline main-table method. Extension probes such as completion-prior removal
or replacement routes should be labeled as extensions and kept separate from
the base DeCE-RF mean.

## 5. Discussion

DeCE-RF reframes localized RF editing as controlled ODE sampling. The source
velocity provides a stable base trajectory, the clean displacement specifies
what should change and what should be preserved, the operation-conditioned
geometry specifies where these components act, and feedback specifies how
strongly each component should act over time. This separation is useful because
local editing failures often arise from different causes. A method can have a
reasonable edit field but poor support, accurate support but weak target
formation, or strong target pressure but unacceptable source drift.

The clean-estimate view makes these causes easier to distinguish. If
`x0_hat_tar - x0_hat_src` is diffuse, the target condition itself may not give
a reliable local edit direction. If support covers the wrong region, the
controller may optimize the wrong local problem. If preserve drift grows in
`M_preserve`, the reconstruction branch or projection must respond. These
diagnostics are harder to express when editing is treated only as target-prompt
velocity replacement.

The operation-conditioned support interface also clarifies the role of masks.
The paper should not claim that a particular support heuristic is universally
optimal. Instead, support is a modular estimate of control geometry. Better
grounding, segmentation, learned support proposal, or user-provided masks could
replace this module while preserving the DeCE-RF control formulation. This
modularity helps keep the paper's claim narrow and defensible.

The feedback controller should also be interpreted carefully. It is not a
magic mechanism that can create a target object when the support is wrong or
the target response is absent. Its defensible role is to adjust the
edit-preserve balance when a reasonable local edit field exists. In that sense,
feedback is best presented as a closed-loop stabilizer and finishing mechanism
for clean-estimate displacement control.

## 6. Limitations

The main limitation is support quality. DeCE-RF assumes that the support
geometry is at least approximately correct. If the edit mask misses the target
region, includes too much source content, or fails to capture the intended
relation, the controller can preserve the source or edit the wrong place. This
is a limitation of automatic local editing generally, but it is especially
visible in DeCE-RF because support is treated as first-class control geometry.

A second limitation is local target formation. Some edits require generating a
new local object with precise shape, pose, scale, and contact. If the target RF
response is weak or spatially diffuse, increasing the edit weight may not be
enough. The method can preserve the source well while failing to synthesize the
requested object. This limitation is expected in small-object replacement,
precise glyph editing, and ambiguous local insertion.

Removal is also harder than insertion or decal editing. Exposed-object removal
can sometimes be handled as a local suppression problem, but occluded-object
removal requires the model to reconstruct hidden host or background content.
Accurate localization does not guarantee plausible completion. For this reason,
completion-prior routes should be reported as extensions rather than as the
base DeCE-RF method.

Replacement remains only partially addressed. A replacement edit must decide
which source attributes to retain, which target attributes to introduce, and
how to handle local geometry changes. Current replacement probes should be
treated as diagnostic or extension cases unless the final experiments show
stable target formation under matched conditions.

The current implementation is SD3-specific. The formulation should transfer to
other RF or flow-matching backbones, but this requires separate validation.
The method also introduces hyperparameters for support thresholding, area
constraints, contact blending, edit weights, preserve budgets, and projection.
The paper should report stable defaults and avoid claiming robustness beyond
the tested settings.

Finally, external baseline comparisons require care. Baselines should be
matched by prompt, resolution, seed where possible, backbone assumptions, and
mask input. A method using a manual mask should not be compared as if it had
the same input conditions as an automatic-support method. Older or
backbone-mismatched baseline artifacts should be labeled as contextual evidence
unless rerun under the final protocol.

## 7. Conclusion

This paper presents DeCE-RF, a decoupled clean-estimate edit-preserve control
framework for localized Rectified Flow editing. The method keeps the
source-conditioned velocity as the base trajectory, decomposes the desired
clean displacement into edit and preservation components, estimates
operation-conditioned control geometry, and adapts component weights with
clean-estimate feedback. The formulation separates what to change, where to
change it, and how strongly to apply each correction during ODE sampling.

The intended claim is deliberately scoped: DeCE-RF aims to improve localized
edit-preserve control under reasonable support, not to solve fully automatic
general image editing. This scope is important. It allows the paper to present
support quality, local target formation, removal completion, and replacement
as explicit open challenges while still making a clear contribution to RF
editing control.

