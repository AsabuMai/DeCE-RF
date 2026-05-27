# DeCE-RF: Decoupled Clean-Estimate Control for Localized Rectified Flow Image Editing

## Abstract

Rectified Flow models provide a natural velocity-field view of image generation,
but local image editing with these models remains difficult: a target-prompt
velocity can simultaneously drive the intended edit and disturb source content
that should remain fixed. We propose DeCE-RF, a decoupled clean-estimate control
framework for localized Rectified Flow image editing. DeCE-RF keeps the
source-conditioned velocity as the base trajectory and decomposes additional
control into a reconstruction correction over preserved regions and an edit
correction over localized support. To define where these corrections act, we
introduce an operation-conditioned support interface that fuses token attention,
Rectified Flow response signals, optional grounding, and operation-level
relations into edit, core, contact, and preserve regions. To decide how strongly
the corrections should act, DeCE-RF monitors clean-estimate-space edit progress
and preservation drift along the ODE trajectory and adaptively balances edit and
reconstruction weights. This formulation turns localized RF editing into an
explicit edit-preserve control problem. Our experiments evaluate the role of
decoupling, support localization, and clean-estimate feedback across localized
insertion, decal, and removal tasks, with failure analysis showing that support
quality remains the main bottleneck for fully automatic editing.

## 1. Introduction

Text-guided image editing asks a generative model to change a specified part of
an image while preserving the source identity, layout, and background. This
edit-preserve requirement is especially sharp for local edits such as adding an
accessory to an animal, placing a decal on a surface, or removing a small object
from a scene. A target prompt alone often does not specify which aspects of the
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

We propose DeCE-RF, a decoupled clean-estimate control framework for localized
Rectified Flow image editing. The central idea is to separate what should be
optimized, where each objective should act, and how strongly each objective
should be applied during the ODE trajectory. First, DeCE-RF decomposes the
editing dynamics into a source-conditioned base velocity, a reconstruction
correction, and an edit correction. Second, it localizes these corrections
through an operation-conditioned support interface that produces edit and
preserve control regions. Third, it uses clean-estimate feedback to adaptively
balance edit progress against preservation drift.

The clean-estimate view is important. Under the linear RF path, a velocity
prediction at timestep \(t\) induces a clean estimate
\[
\hat{x}_0 = x_t - t v_\theta(x_t,t).
\]
This estimate provides a common space in which target-directed change and
source preservation can be measured. DeCE-RF uses clean-estimate disagreement
to propose support regions, uses clean-estimate residuals to define
reconstruction corrections, and uses clean-estimate edit progress and drift as
feedback signals for adaptive control.

A second challenge is spatial support. Local editing requires more than knowing
that the target prompt differs from the source prompt; the controller also
needs a spatial interface specifying where edit corrections are allowed and
where reconstruction should dominate. We therefore frame support estimation as
an operation-conditioned interface rather than an object-specific heuristic.
Given an operation such as `add_object`, `add_decal`, `remove_object`, or
`replace`, and an operation relation such as `above_host`, `on_face`, or
`on_surface`, the support interface fuses token attention, clean-estimate
response, velocity response, and optional grounding evidence into control
geometry: edit, core, contact, and preserve regions.

DeCE-RF is intentionally not framed as a fully general automatic image editor.
Its claim is narrower: when reasonable support is available, explicit
edit-preserve decoupling and clean-estimate feedback provide a useful control
structure for localized RF editing. When support is inaccurate or the operation
requires difficult target formation, the method can fail. We treat these
failure cases as part of the analysis rather than hiding them behind a broad
success claim.

Our contributions are:

- We formulate localized RF image editing as an edit-preserve velocity-control
  problem, decomposing the ODE dynamics into source, reconstruction, and edit
  components.
- We introduce an operation-conditioned support interface that maps operation
  semantics and RF response evidence into edit and preserve control regions.
- We propose a clean-estimate feedback controller that dynamically balances
  edit progress and preservation drift during the reverse RF trajectory.
- We design experiments and ablations that isolate decoupling, support
  localization, and feedback control, and we analyze the failure modes induced
  by imperfect support and difficult local target formation.

## 2. Method

### 2.1 Overview

Given a source image \(I_s\), a source prompt \(c_s\), and a target prompt
\(c_t\), the goal is to produce an edited image that follows the target prompt
only in the intended local region while preserving the rest of the source. The
source image is encoded into a latent \(x_s\). We first invert the source along
the source-conditioned RF ODE to obtain an editable noisy latent \(z_T\). We
then integrate a controlled reverse ODE whose velocity is decomposed as
\[
v_{\mathrm{total}} =
v_{\mathrm{src}} + u_{\mathrm{rec}} + u_{\mathrm{edit}}.
\]
Here \(v_{\mathrm{src}}\) is the source-conditioned model velocity,
\(u_{\mathrm{rec}}\) is a reconstruction correction applied to preserved
regions, and \(u_{\mathrm{edit}}\) is a target-seeking correction applied to
localized edit support.

The method has three modules:

1. edit/reconstruction decoupling in RF velocity space;
2. operation-conditioned support construction for local control geometry;
3. clean-estimate feedback for adaptive edit-preserve balancing.

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

### 2.3 Edit-Reconstruction Velocity Decoupling

A direct target-guided update uses the target condition as the dominant
velocity and therefore entangles local semantic change with global source
drift. DeCE-RF instead keeps the source-conditioned velocity as the base
trajectory:
\[
v_{\mathrm{base}} = v_{\mathrm{src}}.
\]
All editing behavior is expressed as additional corrections. The reconstruction
branch preserves the source in non-edit regions, while the edit branch modifies
the localized support.

The reconstruction correction is defined in clean-estimate space. Let
\(M_{\mathrm{preserve}}\in[0,1]^{H\times W}\) denote the preserve mask. We
measure the residual between the current source-conditioned clean estimate and
the source latent:
\[
r_{\mathrm{rec}} = \hat{x}_{0,\mathrm{src}} - x_s.
\]
The desired clean-space displacement is \(-r_{\mathrm{rec}}\). Under the RF
linear path, a clean-space displacement \(\Delta x_0\) corresponds to a
velocity correction
\[
u = - \frac{\Delta x_0}{t}.
\]
Thus the reconstruction correction is a masked velocity field that pulls the
preserve region back toward the source clean latent.

The edit branch uses target-source RF differences and optional energy-inspired
terms over the edit support. The base edit direction is
\[
u_{\mathrm{edit,base}} = v_{\mathrm{tar}} - v_{\mathrm{src}},
\]
masked by the edit region. Additional terms can include clean-space anchor or
region surrogates, text reward gradients, reference-image guidance, and other
diagnostic controls. In the main formulation, these terms are treated as local
edit corrections gated by the support interface rather than as global target
velocities.

### 2.4 Operation-Conditioned Support Interface

The decoupled velocity fields require a spatial interface specifying where
each correction should act. DeCE-RF uses an operation-conditioned support
interface that maps operation semantics and RF response evidence to control
geometry.

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
The core region receives the strongest edit correction. The contact region is
a soft boundary ring that permits weak blending. The preserve region receives
reconstruction control. This spatial interface is the bridge between operation
semantics and RF velocity control.

### 2.5 Clean-Estimate Feedback Control

Fixed edit and reconstruction weights are brittle. A support region may be
slightly too large, an edit may be under-expressed, or preservation drift may
increase at different stages of the ODE. DeCE-RF therefore adapts the edit and
preserve weights using clean-estimate feedback.

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

The controller then scales the edit and reconstruction corrections:
\[
u_{\mathrm{rec}} \leftarrow w_{\mathrm{preserve}} u_{\mathrm{rec}},
\qquad
u_{\mathrm{edit}} \leftarrow w_{\mathrm{edit}} u_{\mathrm{edit}}.
\]
When the edit correction conflicts with preserve-region clean estimates, a
projection step removes the component of the edit update that is estimated to
increase preservation drift. This produces a closed-loop controller: the method
does not merely apply a fixed mask and fixed weights, but monitors the current
trajectory and adjusts the edit-preserve balance online.

### 2.6 ODE Integration

The controlled reverse ODE update is
\[
x_{t-\Delta t} =
x_t + (t_{i+1}-t_i)
\left(v_{\mathrm{src}} + u_{\mathrm{rec}} + u_{\mathrm{edit}}\right).
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
      compute reconstruction correction u_rec on M_preserve;
      compute edit correction u_edit on M_edit/M_core;
      measure edit gap and preserve drift in clean-estimate space;
      adapt edit and preserve weights;
      optionally project away preserve-conflicting edit components;
      update x_t using v_src + u_rec + u_edit.
5. Decode the final latent to the edited image.
```

### 2.8 Discussion of Scope

DeCE-RF separates the control problem from the support-quality problem. If the
support interface identifies a reasonable region, the controller can balance
target change and source preservation within that region. If support is wrong,
too small, too large, or semantically ambiguous, the controller may preserve
the source but fail to form the desired target object. This limitation is
central to the paper's scope: DeCE-RF is a localized RF control framework, not
a complete solution to automatic support proposal or general object formation.
