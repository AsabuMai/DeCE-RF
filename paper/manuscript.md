# DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing

## Abstract

Rectified Flow models provide a natural velocity-field view of image generation,
but local image editing with these models remains difficult: a target-prompt
velocity entangles the intended local edit with global source drift. We propose
DeCE-RF, a decoupled clean-estimate edit-preserve control framework for
localized Rectified Flow editing. Instead of replacing the source trajectory
with the target-conditioned velocity, DeCE-RF keeps the source-conditioned
velocity as a base field and designs a decoupled edit-preserve displacement in
clean-estimate space. The controlled velocity takes the compact form
\(v_{\mathrm{DeCE}}=v_{\mathrm{src}}-t^{-1}\Delta_t^0\), where the clean
displacement \(\Delta_t^0\) is decomposed into an edit component that moves
localized regions toward the target clean estimate and a preservation component
that pulls non-edit regions back to the source latent. An operation-conditioned
support interface estimates the spatial geometry of these displacements, and
clean-estimate feedback adaptively balances their weights along the ODE
trajectory. Our experiments evaluate this control view across the strict Core-6
suite: attached accessory insertion, container-constrained insertion, surface
decal editing, local recoloring, surface material-strip editing, and
exposed-object removal. We further separate SD3-matched RF baselines from
native-backbone contextual comparisons, keeping the main claim tied to
matched-backbone evidence.

## 1. Introduction

Text-guided image editing asks a generative model to change a specified part of
an image while preserving the source identity, layout, and background. This
edit-preserve requirement is especially sharp for local edits such as adding an
accessory to an animal, inserting an object inside a container, placing a decal
on a surface, recoloring an object, editing a local material strip, or removing
a small exposed object from a scene. A target prompt alone often does not
specify which aspects of the
source should remain fixed. As a result, direct target guidance can improve
semantic alignment while also changing pose, texture, object identity, or
background content.

Rectified Flow (RF) models make this problem particularly interesting because
generation is expressed through a velocity field integrated along an ODE
trajectory. Instead of viewing editing only as a prompt substitution problem, we
can ask how the velocity field itself should be controlled. A direct target
velocity couples two competing goals: it pushes the sample toward the target
description, but it does not explicitly distinguish the local edit region from
the source content that should be reconstructed. This coupling suggests that
localized RF editing should be treated as a control problem rather than a
single global prompt-guidance problem.

We propose DeCE-RF, a decoupled clean-estimate edit-preserve control framework
for localized Rectified Flow image editing. Instead of replacing the source
trajectory with the target velocity, DeCE-RF keeps the source-conditioned
velocity as the base field and computes a decoupled clean displacement at each
ODE step. The displacement is designed in clean-estimate space: its edit
component moves edit regions toward the target clean estimate, while its
preservation component reconstructs the source latent in preserve regions.

The clean-estimate view is important. Under the linear RF path, a velocity
prediction at timestep \(t\) induces a clean estimate
\[
\hat{x}_0 = x_t - t v_\theta(x_t,t).
\]
This estimate provides a common space in which target-directed change and
source preservation can be measured. In this view, the edit and preservation
branches are not separate engineering modules. They are the RF
velocity forms of two spatial components of the same clean displacement. The
control geometry estimator sets where these components act, while the adaptive
controller updates their weights from clean-estimate feedback.

A second challenge is spatial support. Local editing requires more than knowing
that the target prompt differs from the source prompt; the controller also
needs a spatial interface specifying where edit displacement is allowed and
where preservation should dominate. We therefore frame support estimation as
operation-conditioned control geometry rather than an object-specific heuristic.
Given an operation such as `add_object`, `add_decal`, `remove_object`, or
`replace`, and an operation relation such as `above_host`, `on_face`, or
`on_surface`, the geometry estimator fuses token attention, clean-estimate
response, velocity response, and optional grounding evidence into edit, core,
contact, and preserve regions.

DeCE-RF is intentionally not framed as a fully general automatic image editor.
Its claim is narrower: when reasonable support is available, decoupled
clean-estimate displacement and feedback weighting provide a useful control
structure for localized RF editing. When support is inaccurate or the operation
requires difficult target formation, the method can fail. We treat these
failure cases as part of the analysis rather than hiding them behind a broad
success claim.

Our contributions are:

- We formulate localized Rectified Flow image editing as decoupled
  clean-estimate displacement control over a source-conditioned base trajectory.
- We show that edit and preservation branches arise by decomposing the
  desired clean displacement into an edit component and a preservation
  component, then mapping this displacement back to RF velocity.
- We introduce an operation-conditioned control geometry estimator that maps
  token attention, RF response, relation, and optional grounding evidence into
  the spatial geometry of the clean displacement.
- We update the edit and preservation displacement weights online using
  clean-estimate edit progress and preservation drift, yielding a closed-loop
  edit-preserve controller.
- We design experiments and ablations that isolate clean displacement,
  support geometry, and feedback weighting, and we analyze the failure modes
  induced by imperfect support and difficult local target formation.

## 2. Method

### 2.1 Overview

Given a source image \(I_s\), a source prompt \(c_s\), and a target prompt
\(c_t\), the goal is to produce an edited image that follows the target prompt
only in the intended local region while preserving the rest of the source. The
source image is encoded into a latent \(x_s\). We first invert the source along
the source-conditioned RF ODE to obtain an editable noisy latent \(z_T\). We
then integrate a controlled reverse ODE whose velocity is written as
\[
v_{\mathrm{DeCE}}
=
v_{\mathrm{src}}
-
\frac{1}{t}\Delta_t^0.
\]
Here \(v_{\mathrm{src}}\) is the source-conditioned model velocity and
\(\Delta_t^0\) is the desired clean-estimate displacement at timestep \(t\).
Under the RF linear path, adding a correction \(u_t\) to the source velocity
changes the controlled clean estimate as
\[
\hat{x}_0(u_t)
= x_t - t(v_{\mathrm{src}}+u_t)
= \hat{x}_{0,\mathrm{src}} - t u_t.
\]
Therefore, designing a clean displacement
\(\hat{x}_0(u_t)=\hat{x}_{0,\mathrm{src}}+\Delta_t^0\) implies
\[
u_t = -\frac{1}{t}\Delta_t^0,
\]
which yields the compact DeCE-RF velocity above.

The desired clean displacement is explicitly decoupled:
\[
\Delta_t^0
=
\Delta_t^{\mathrm{edit}}
+
\Delta_t^{\mathrm{pres}}.
\]
The implementation can be read as an efficient instantiation of this
displacement decomposition:

1. edit and preservation branches instantiate the two clean displacement
   components;
2. operation-conditioned support estimates where each component should act;
3. clean-estimate feedback updates the weights of these components along the
   ODE trajectory.

### 2.2 Rectified Flow Clean Estimates

We use the linear RF path convention
\[
x_t = (1-t)x_0 + t x_1,
\]
where \(x_0\) is a clean latent and \(x_1\) is a noise latent. Given a model
velocity \(v_\theta(x_t,t,c)\) under condition \(c\), the corresponding clean
estimate is
\[
\hat{x}_0(x_t,t,c) = x_t - t v_\theta(x_t,t,c).
\]
For the source and target prompts, we write
\[
v_{\mathrm{src}} = v_\theta(x_t,t,c_s), \qquad
v_{\mathrm{tar}} = v_\theta(x_t,t,c_t),
\]
and
\[
\hat{x}_{0,\mathrm{src}} = x_t - t v_{\mathrm{src}}, \qquad
\hat{x}_{0,\mathrm{tar}} = x_t - t v_{\mathrm{tar}}.
\]

Clean-estimate disagreement is used as a spatial response signal:
\[
D_{\mathrm{clean}}(i,j) =
\left\|\hat{x}_{0,\mathrm{tar}}(:,i,j)
- \hat{x}_{0,\mathrm{src}}(:,i,j)\right\|_2.
\]
Velocity disagreement provides a complementary signal:
\[
D_{\mathrm{vel}}(i,j) =
\left\|v_{\mathrm{tar}}(:,i,j) - v_{\mathrm{src}}(:,i,j)\right\|_2.
\]
These maps indicate where the target condition changes the RF dynamics relative
to the source condition.

### 2.3 Decoupled Edit-Preserve Displacement

A direct target-guided update uses the target condition as the dominant
velocity and therefore entangles local semantic change with global source
drift. DeCE-RF instead keeps the source-conditioned velocity as the base
trajectory:
\[
v_{\mathrm{base}} = v_{\mathrm{src}}.
\]
All editing behavior is expressed as a clean displacement that is mapped back
to velocity. The two branches used in the implementation are the edit and
preserve components of this displacement.

The edit component moves the edit region toward the target clean estimate:
\[
\Delta_t^{\mathrm{edit}}
=
\lambda_e(t) M_e \odot
\left(
\hat{x}_{0,\mathrm{tar}}
-
\hat{x}_{0,\mathrm{src}}
\right).
\]
Since
\[
\hat{x}_{0,\mathrm{tar}}-\hat{x}_{0,\mathrm{src}}
=
-t(v_{\mathrm{tar}}-v_{\mathrm{src}}),
\]
the target-source velocity difference used in the implementation is the RF
velocity form of this edit clean displacement.

The preservation component pulls preserve regions back toward the source latent:
\[
\Delta_t^{\mathrm{pres}}
=
\lambda_p(t) M_p \odot
\left(
x_s
-
\hat{x}_{0,\mathrm{src}}
\right).
\]
Mapping this clean displacement back to velocity gives the reconstruction
correction direction
\[
-\frac{1}{t}\Delta_t^{\mathrm{pres}}
=
\lambda_p(t) M_p \odot
\frac{\hat{x}_{0,\mathrm{src}}-x_s}{t}.
\]
Thus preservation is not a side loss or a separate heuristic; it is the clean
displacement component that cancels source drift in preserve regions.

The resulting controlled velocity can be written as
\[
v_{\mathrm{DeCE}}
=
v_{\mathrm{src}}
-
\frac{1}{t}
\left(
\Delta_t^{\mathrm{edit}}
+
\Delta_t^{\mathrm{pres}}
\right),
\]
with \(M_e\) typically approximated by \(M_{\mathrm{core}}\) plus a weak
contact region and \(M_p\) by the preserve region. Optional text, color,
reference, or local-target terms instantiate edit-side clean displacements
inside this same control framework and do not change the core formulation.

Equivalently, this displacement view can be written as the per-step
clean-estimate objective
\[
\arg\min_u
\left[
\lambda_p(t)
\left\|
M_p \odot \left(\hat{x}_0(u) - x_s\right)
\right\|^2
+
\lambda_e(t)
\left\|
M_e \odot \left(\hat{x}_0(u) - \hat{x}_{0,\mathrm{tar}}\right)
\right\|^2
\right],
\]
whose closed-form directions are exactly the edit and preserve displacements
above.

### 2.4 Operation-Conditioned Support Interface

The decoupled clean displacement requires a spatial interface specifying where
the edit and preserve components should act. DeCE-RF uses
an operation-conditioned control geometry estimator that maps operation
semantics and RF response evidence to displacement geometry.

Each edit is described by an operation type:
\[
o \in \{\texttt{add\_object}, \texttt{add\_decal},
\texttt{remove\_object}, \texttt{replace}, \texttt{recolor}\},
\]
and an optional relation:
\[
\rho \in \{\texttt{above\_host}, \texttt{on\_face},
\texttt{on\_surface}, \texttt{inside}, \texttt{remove\_source\_object}\}.
\]
The interface avoids object-specific rules such as a dedicated "sunglasses
rule" or "crown rule." Instead, relations define operation-level priors such
as face regions, host surfaces, or above-host insertion bands.

The evidence maps include:

- target or new-token attention \(A_{\mathrm{new}}\);
- host-token attention \(A_{\mathrm{host}}\);
- removed/source-token attention \(A_{\mathrm{removed}}\);
- clean-estimate disagreement \(D_{\mathrm{clean}}\);
- velocity disagreement \(D_{\mathrm{vel}}\);
- optional grounded or segmented masks;
- relation maps derived from the operation relation.

Candidate support maps are formed by normalized products or mixtures of these
evidence sources. Examples include relation-response maps for object insertion,
surface-local response maps for decals, and removed-source clean-disagreement
maps for removal. The operation type selects the default candidate family. The
selected support score is thresholded, area-constrained, component-filtered,
dilated, and smoothed to obtain an edit mask \(M_{\mathrm{edit}}\) and a core
mask \(M_{\mathrm{core}}\).

To avoid hard edit boundaries, DeCE-RF further decomposes the support into an
object/contact/preserve layering:
\[
M_{\mathrm{edit}}, \quad
M_{\mathrm{core}}, \quad
M_{\mathrm{contact}}, \quad
M_{\mathrm{preserve}}.
\]
The core region receives the strongest edit displacement. The contact region is
a soft boundary ring that permits weak blending. The preserve region receives
the preservation displacement. This spatial interface is the bridge between
operation semantics and the clean-estimate displacement.

### 2.5 Clean-Estimate Feedback Control

Fixed edit and preservation weights are brittle. A support region may be
slightly too large, an edit may be under-expressed, or preservation drift may
increase at different stages of the ODE. DeCE-RF therefore treats adaptive
control as feedback-updated displacement weights rather than as a separate
heuristic module.

At each timestep, the controller computes edit progress and preservation drift
in clean-estimate space. Let \(M_e\) be the edit mask and \(M_p\) the preserve
mask. We measure edit-region change and target gap using the current and target
clean estimates:
\[
\Delta_{\mathrm{cur}} = \hat{x}_{0,\mathrm{src}} - x_s,
\]
\[
\Delta_{\mathrm{tar}} = \hat{x}_{0,\mathrm{tar}} - x_s,
\]
\[
g_{\mathrm{edit}} =
\mathrm{RMS}_{M_e}(\Delta_{\mathrm{tar}}-\Delta_{\mathrm{cur}}).
\]
We measure preserve drift as
\[
d_{\mathrm{preserve}} =
\mathrm{RMS}_{M_p}(\hat{x}_{0,\mathrm{src}} - x_s).
\]

The edit weight increases when the edit target gap is large:
\[
w_{\mathrm{edit}} =
\mathrm{clip}(1 + \lambda_e \, \delta_e,
w_{\mathrm{edit}}^{\min}, w_{\mathrm{edit}}^{\max}),
\]
where \(\delta_e\) is an edit deficit derived from the target clean-space gap.
The preserve weight increases when preserve drift exceeds a budget:
\[
w_{\mathrm{preserve}} =
\mathrm{clip}(1 + \lambda_p \, \max(0,d_{\mathrm{preserve}}-\tau_p),
w_{\mathrm{preserve}}^{\min}, w_{\mathrm{preserve}}^{\max}).
\]

The controller then scales the two clean displacement components:
\[
\Delta_t^{\mathrm{pres}} \leftarrow w_{\mathrm{preserve}}\Delta_t^{\mathrm{pres}},
\qquad
\Delta_t^{\mathrm{edit}} \leftarrow w_{\mathrm{edit}}\Delta_t^{\mathrm{edit}}.
\]
When the edit displacement conflicts with preserve-region clean estimates, a
projection step removes the component of the edit displacement that is estimated to
increase preservation drift. This produces a closed-loop controller: the method
does not merely apply a fixed mask and fixed weights, but monitors the current
trajectory and adjusts the edit-preserve balance online.

### 2.6 ODE Integration

The controlled reverse ODE update is
\[
x_{t-\Delta t} =
x_t + (t_{i+1}-t_i)
\left(v_{\mathrm{src}} - t^{-1}\Delta_t^0\right).
\]
The same masks and clean-estimate feedback signals are logged at each step for
diagnostics. This allows the experiments to report not only final image
metrics, but also controller behavior such as edit gap, preserve drift,
adaptive weights, and projection magnitude.

### 2.7 Algorithm

```text
Input: source image I_s, source prompt c_s, target prompt c_t,
       operation o, relation rho

1. Encode I_s into source latent x_s.
2. Invert x_s along the source-conditioned RF ODE to obtain z_T.
3. At support-estimation timesteps:
      compute v_src and v_tar;
      compute clean and velocity disagreement maps;
      collect token attention and optional grounding evidence;
      build operation-conditioned support candidates;
      select and postprocess M_edit, M_core, M_contact, M_preserve.
4. For each reverse ODE timestep:
      compute v_src and v_tar;
      estimate x0_src and x0_tar;
      compute Delta_edit on M_edit/M_core;
      compute Delta_pres on M_preserve;
      measure edit gap and preserve drift in clean-estimate space;
      adapt edit and preserve weights;
      optionally project away preserve-conflicting edit displacement;
      update x_t using v_src - t^-1(Delta_edit + Delta_pres).
5. Decode the final latent to the edited image.
```

### 2.8 Discussion of Scope

DeCE-RF separates the control problem from the support-quality problem. If the
control geometry identifies a reasonable region, the controller can balance
target change and source preservation within that region. If support is wrong,
too small, too large, or semantically ambiguous, the controller may preserve
the source but fail to form the desired target object. This limitation is
central to the paper's scope: DeCE-RF is a localized RF control framework, not
a complete solution to automatic support proposal or general object formation.

## 3. Experiments

### 3.1 Experimental Setup

We evaluate DeCE-RF on localized text-guided image editing tasks that require
both target-directed change and source preservation. The strict Core-6 suite is
organized by operation type rather than by object category, because the method is
designed around operation-conditioned control. The active tasks are:

```text
cat_crown: add_object / above_host
bowl_apple_inside: add_object / inside
tshirt_star: add_decal / on_surface
red_chair_blue: recolor / inside
pillow_vertical_fabric_strip: add_decal / on_surface
backpack_remove_toy_charm: remove_object / remove_source_object
```

These tasks cover compact accessory insertion, container-constrained insertion,
surface decal placement, local object recoloring, surface material-strip editing,
and exposed-object removal. The add/decal/recolor tasks test whether the
controller can form a local target while preserving the surrounding source. The
removal task is included as a stress case because successful removal requires
suppressing source evidence rather than merely adding target evidence.

Unless otherwise stated, all SD3-matched methods use the same source image,
source prompt, target prompt, random seed, number of RF steps, image resolution,
and fixed evaluation masks. We report results over seeds \(10,11,12\). Native-
backbone FLUX comparisons are reported separately as contextual E2-B rows
because their backbone and input interface differ from SD3-DeCE. For local-edit
tasks, automatic metrics are not sufficient on their own; therefore, the
qualitative figures include the source image, edited image, edit support,
preserve support, and clean-estimate response maps.

### 3.2 Compared Methods

We compare the following methods:

- **Source only.** The source-conditioned trajectory is reconstructed without
  target editing. This measures reconstruction quality and gives a preservation
  reference.
- **Direct target.** The target-conditioned velocity is used directly as the
  editing signal. This is the coupled baseline: target change and source drift
  are not separated.
- **Generic support.** A weak automatic support baseline using prompt-response
  evidence without operation-conditioned relation structure.
- **DeCE-RF.** The full method with operation-conditioned support and
  feedback-updated displacement weights.
- **Fixed displacement weights.** An ablation-only internal control: the source
  velocity is kept as the base trajectory, with edit and preserve displacement
  components gated by operation geometry but with fixed weights.
- **Manual/external support.** When available, an externally specified support
  mask is used as an upper-bound diagnostic for support quality rather than as
  the main automatic method.
- **SD3-matched RF baselines.** FlowEdit, FlowAlign, and SplitFlow are reported
  as the primary E2-A external baselines because they run under the same SD3
  backbone and strict Core-6 protocol.
- **Native-backbone contextual baselines.** FLUX-based RF editors such as
  RF-Solver-Edit, ReFlex, FireFlow, and stable-flow are E2-B contextual rows, not
  pure algorithmic controls against SD3-DeCE. InstructPix2Pix and H-Edit/P2P are
  supplement-only non-RF positioning baselines.

This comparison separates the three claims of the paper. Direct target versus
fixed displacement weights tests the benefit of clean-estimate displacement
control over target-velocity replacement. Generic support versus
operation-conditioned geometry isolates the spatial geometry estimator. Fixed
displacement weights versus DeCE-RF isolates clean-estimate feedback, but the
fixed-weight row is reported as an ablation rather than a headline method.

### 3.3 Metrics

We evaluate preservation, edit strength, and controller behavior.

For preservation, we compute outside-mask pixel error and latent/feature
similarity between the edited output and the source image. We report outside
L1/RMSE, luma SSIM, and DINO source similarity where available. These metrics
measure whether the method changes source content outside the intended edit
region.

For edit strength, we compute inside-mask change and target alignment. We
report CLIP target score and CLIP target-source delta, and we pair these
metrics with qualitative inspection because CLIP can reward global prompt
changes that are undesirable for local editing.

For support quality, we report support area, leakage, and overlap with manual
or external masks when such masks are available. We also visualize
\(M_{\mathrm{edit}}\), \(M_{\mathrm{core}}\), \(M_{\mathrm{contact}}\), and
\(M_{\mathrm{preserve}}\), because support failures are often spatial and are
not well summarized by a single scalar.

For controller behavior, we log clean-estimate edit gap,
preservation drift, adaptive edit weight, adaptive preserve weight, and
projection magnitude over the RF trajectory. These diagnostics test whether the
feedback controller is acting as intended rather than only improving final
images by chance.

### 3.4 Main Results

The main result table should report each task and method across the primary
metrics:

```text
Task | Method | CLIP target delta | outside RMSE | DINO source sim |
     |        | luma SSIM         | support area | runtime / memory
```

The expected trend is that direct target guidance obtains stronger target
alignment but higher outside drift, while source-only reconstruction preserves
the input but does not perform the edit. Fixed displacement weights should reduce
outside drift relative to direct target guidance by keeping the source velocity
as the base trajectory and applying edit/preserve displacements only through local
control geometry. DeCE-RF should further improve the edit-preserve tradeoff by
increasing edit pressure when the local target gap remains large and increasing
reconstruction pressure when preserve drift exceeds the budget.

The main qualitative figure should use the completed strict Core-6 task set and
compare the paper-facing E1 methods:

```text
source | direct target | generic support | DeCE-RF | support overlay
```

The E2-A figure should separately show the SD3-matched RF baselines:

```text
source | FlowEdit-SD3 | FlowAlign-SD3 | SplitFlow-SD3 | DeCE-RF-SD3
```

Native-backbone FLUX rows, if runnable, should be labeled as contextual and not
merged into the SD3-matched algorithmic table. Each edited image should be paired
with the edit mask and an outside-drift heat map where space permits. This
figure is important because the central claim is not simply that the target
prompt score improves, but that target change is localized while source content
remains stable.

### 3.5 Ablation: Decoupled Clean-Estimate Displacement

To isolate the displacement formulation, we compare direct target guidance against variants
that progressively introduce the source-base trajectory and preserve term:

```text
direct target
edit term only on operation geometry
edit term with preserve term disabled
fixed displacement weights
```

This ablation tests whether the source-conditioned base velocity and
preserve displacement reduce drift outside the edit support. The expected evidence
is lower outside-mask error and higher source similarity for displacement-based
variants at comparable target alignment. If the displacement under-expresses the
edit, the result should be reported as a tradeoff rather than hidden; this is
precisely the motivation for feedback-updated weights.

### 3.6 Ablation: Operation-Conditioned Support

We compare support construction variants under the same controller:

```text
attention-only support
clean-disagreement-only support
generic support
operation-conditioned support
manual/external support
```

The key question is whether the operation-conditioned interface improves the
geometry of control. For `add_object / above_host`, support should focus on the
insertion band rather than the entire host object. For `add_decal /
on_surface`, support should remain on the host surface and avoid forming a
free-floating object. For `remove_object`, support should identify the source
object to suppress rather than the region where a new target token attends.

The paper should present support visualizations alongside final images. If
manual/external support substantially outperforms automatic support, that result
should be treated as evidence that support estimation remains the bottleneck,
while still validating the control formulation under reasonable support.

### 3.7 Ablation: Clean-Estimate Feedback

To isolate feedback, we compare fixed displacement weights against DeCE-RF under
both standard settings and stress settings:

```text
fixed weights
clean-estimate feedback
clean-estimate feedback with edit-strength sweep
clean-estimate feedback with support perturbations
```

The edit-strength sweep evaluates whether feedback stabilizes the tradeoff as
the target correction becomes stronger. The support perturbation test randomly
erodes, dilates, or partially drops the edit mask, evaluating whether the
controller can reduce drift when support is imperfect.

The main diagnostic figure should plot:

```text
timestep vs edit gap
timestep vs preserve drift
timestep vs edit weight
timestep vs preserve weight
```

A successful controller should show interpretable behavior: edit weight rises
when target progress is insufficient, preserve weight rises when drift exceeds
the budget, and the final image lands closer to the desired edit-preserve
frontier than fixed weights.

### 3.8 Failure Analysis

We explicitly analyze failures rather than presenting DeCE-RF as a universal
editor. The main failure types are:

- **Support failure.** The edit mask misses the true edit region or leaks into
  source content that should be preserved.
- **Target formation failure.** The support is plausible, but the local target
  object does not form because the local target signal is weak or ambiguous.
- **Removal failure.** The controller preserves source evidence too strongly,
  leaving traces of the object that should be removed.
- **Over-editing.** The edit branch succeeds locally but changes identity,
  texture, pose, or background outside the intended region.
- **Over-preservation.** The reconstruction branch prevents visible target
  formation.

These failures clarify the boundary of the method. DeCE-RF addresses the
edit-preserve control structure once a reasonable control geometry is
available; it does not solve open-ended semantic localization or all forms of
local object synthesis.

## 4. Discussion

DeCE-RF suggests that localized RF editing is better understood as a controlled
ODE problem than as a direct target-prompt substitution problem. The source
velocity provides a stable reconstruction trajectory, while editing and
preservation become explicit clean displacement components. This separation
makes it possible to ask targeted questions: where should edit displacement
act, where should preserve displacement dominate, and how should their
strengths change over time?

The operation-conditioned control geometry estimator is central to this view. Although
it is implemented through practical evidence maps, its conceptual role is not a
collection of object-specific rules. It is a translation layer from operation
semantics to control geometry. This distinction matters for the paper: the
method should not claim that a particular support heuristic is universally
optimal. Instead, it claims that localized RF editing benefits from exposing
support as first-class control geometry.

Clean-estimate feedback provides the third part of the story. Since the RF
velocity induces an estimate of the clean latent at every timestep, the
trajectory can be monitored in a space closer to the final image than raw noisy
latents. This makes edit progress and preservation drift measurable during
sampling. The controller then uses those measurements to adapt the edit and
preserve displacement weights online.

Together, these components form a coherent edit-preserve control framework:
the clean displacement defines what should be controlled, operation geometry
defines where each component acts, and feedback defines their time-varying
strength.

## 5. Limitations

The main limitation is support quality. When the operation-conditioned support
interface selects the wrong region, the controller can faithfully optimize the
wrong local problem. Better grounding, segmentation, or learned support
proposal models could improve this part of the pipeline.

A second limitation is local target formation. Some edits require generating a
new object with precise shape and placement from a small local signal. If the
target RF response is weak or diffuse, the controller may preserve the source
well but fail to synthesize the intended object.

Third, removal and replacement are harder than insertion. Removal requires
actively suppressing source evidence and hallucinating plausible background,
while replacement requires deciding which source attributes should remain and
which should change. These operations may need stronger inpainting priors or
operation-specific target priors.

Finally, the method introduces additional hyperparameters for support
postprocessing and feedback budgets. The experiments should report sensitivity
to these settings and identify stable defaults, especially for edit strength,
preserve drift budget, and support thresholding.

## 6. Conclusion

We introduced DeCE-RF, a decoupled clean-estimate edit-preserve control
framework for localized Rectified Flow editing. DeCE-RF keeps the
source-conditioned velocity as the base trajectory, decomposes the desired
clean displacement into edit and preserve components, and maps this displacement
back to RF velocity. Operation-conditioned support supplies the spatial
geometry of the displacement, and clean-estimate feedback updates its component
weights along the ODE trajectory. The planned experiments isolate the roles of
clean displacement, support geometry, and feedback weighting, while the failure
analysis clarifies that automatic support and difficult local target formation
remain open challenges.
