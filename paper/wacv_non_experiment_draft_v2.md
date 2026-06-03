# DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing

## Abstract

Localized image editing with Rectified Flow models requires a controller that
can express target change without allowing the target-conditioned trajectory to
regenerate source content that should remain fixed. We introduce DeCE-RF, a
decoupled clean-estimate edit-preserve control framework for localized
Rectified Flow editing. DeCE-RF keeps the source-conditioned velocity as the
base field and applies a clean-estimate displacement at each ODE step. The
controlled velocity is
`v_DeCE = v_src - t_eff^-1 Delta_t^0`, with
`Delta_t^0 = Delta_edit + Delta_pres`. The edit component is a gated
target-source clean displacement, while the preservation component is a
clean-estimate drift correction toward the source latent. An
operation-conditioned geometry estimator turns attention, RF response, optional
grounding, and operation relations into edit, core, contact, and preserve
regions. A feedback controller then adapts edit and preservation weights and
removes preserve-conflicting edit components during sampling. This formulation
turns localized RF editing into three reproducible subproblems: defining the
clean displacement, estimating its spatial geometry, and controlling its
time-varying edit-preserve balance. Experimental results will be inserted after
the final Core-6 fixed-mask runs are complete.

## 1. Introduction

Localized text-guided image editing asks a generative model to change one
region of an input image while preserving the identity, layout, and background
of the source. This requirement is difficult for Rectified Flow (RF) models
because a target-conditioned velocity is a global generative field. It can
increase target-prompt alignment, but it does not by itself distinguish the
intended edit from source content that should remain stable. The failure mode
is familiar: an accessory appears but the animal changes, a decal is generated
but the host object drifts, or a removal suppresses the target object while
damaging nearby structure.

The central observation of this paper is that RF velocities induce clean
estimates, and clean estimates provide the right space for separating edit and
preservation. Under the linear RF path, a velocity prediction at timestep `t`
defines `x0_hat = x_t - t v_theta(x_t,t)`. Therefore, an edit can be written
as a desired clean-estimate displacement rather than as a direct replacement of
the source trajectory by the target velocity. This displacement can be
decomposed into an edit term that moves a local region toward the target clean
estimate and a preservation term that pulls non-edit regions back toward the
source latent.

We propose DeCE-RF, a decoupled clean-estimate edit-preserve controller for
localized RF editing. DeCE-RF keeps the source-conditioned velocity as the
base trajectory and applies a clean displacement:

```text
v_DeCE = v_src - t_eff^-1 Delta_t^0,
Delta_t^0 = Delta_edit + Delta_pres.
```

The edit term is intentionally transparent: in velocity form it is equivalent
to a gated target-source velocity correction. The novelty is not hiding this
equivalence. Instead, DeCE-RF defines edit and preservation in the same
clean-estimate space, adds a preservation drift correction that is not a simple
latent blend, estimates layered control geometry rather than a single mask, and
uses clean-estimate feedback to adapt and project the displacement during
sampling.

Spatial geometry is the second part of the method. DeCE-RF uses operation and
relation labels, such as `add_object / above_host`, `add_decal / on_surface`,
`remove_object / remove_source_object`, and `recolor / on_surface`, to build
the control regions. In the current benchmark these labels are provided by the
experiment protocol, not inferred automatically. This input condition is
explicit: label-aware or mask-aware baselines must receive matched information,
or be reported in a separate input-setting group. The geometry estimator fuses
token attention, clean-estimate disagreement, velocity disagreement, optional
grounding, and relation maps into `M_core`, `M_contact`, `M_edit`, and
`M_preserve`.

The third part is feedback. During the RF ODE, DeCE-RF measures edit target
gap and preservation drift in clean-estimate space. It increases edit pressure
when the target gap remains active, increases preservation pressure when source
drift exceeds a budget, and projects away edit displacement components that
would increase preserve-region error. The result is a closed-loop local
controller rather than a fixed masked target-guidance rule.

The contributions are:

- **Clean-estimate displacement control for RF editing.** We derive the
  velocity correction `u_t = -t_eff^-1 Delta_t^0` and use it to express
  localized editing over a source-conditioned base trajectory.
- **Edit-preserve decomposition with closed-form local behavior.** We show
  that the edit branch is gated target-source velocity in RF form, while the
  preservation branch is clean-estimate source-drift correction in preserve
  regions.
- **Operation-conditioned geometry and feedback control.** We define a
  reproducible geometry estimator and a clean-estimate feedback/projection
  controller that can be isolated through support and feedback ablations.

We focus on localized edits under approximately correct support. The
limitations of automatic support, completion, and replacement are analyzed
separately rather than folded into the core claim.

## 2. Related Work

### Diffusion and Instruction-Based Image Editing

Text-guided image editing has been studied through image-to-image diffusion,
inversion, attention control, feature reuse, and instruction-following models.
These methods establish a broad toolkit for changing image content from text,
but many operate through target prompts or editing instructions whose locality
is indirect. Local preservation therefore often depends on inversion quality,
attention control, masks, or additional regularization. DeCE-RF is narrower:
it assumes an RF backbone and asks how local edit and preservation corrections
should be written inside the RF sampling dynamics.

### Localized Editing and Mask/Attention Control

Localized editing methods directly address the problem of changing a region
without disturbing the rest of the image. LIME is an important recent example:
it obtains localized masks from pretrained features and cross-attention, then
regularizes attention to confine edits without a user-specified ROI
[@lime2025]. GeoDiffuser shows a complementary operation-centric view by
casting image editing operations as geometric transformations and using SAM
segmentation plus optimization to preserve object style and plausibility
[@geodiffuser2025]. These works are close in problem setting, but they mainly
address where or how an operation should be localized in diffusion editing.
DeCE-RF instead addresses how localized edit and preservation should be
expressed as clean-estimate displacement control in RF sampling. When DeCE-RF
uses operation/relation labels or external grounding, those inputs must be
reported explicitly and matched for fair baselines.

### Rectified Flow and Flow-Matching Editing

RF editing methods study how velocity trajectories can be inverted, redirected,
or solved more accurately. FlowEdit constructs inversion-free source-to-target
editing paths with pretrained flow models [@flowedit2024]. RF-Solver-Edit
studies RF inversion and solver behavior for editing [@rfsolver2024], while
FireFlow emphasizes fast inversion-based editing [@fireflow2024]. h-Edit
decomposes editing with reconstruction and editing terms from a
Doob-transform perspective [@hedit2025]. Optimal-transport-guided RF editing
frames RF editing as a tradeoff between reconstruction fidelity and editing
flexibility, improving both inversion-based and direct RF editing paradigms
[@otrf2026]. DeCE-RF is complementary to these works. It does not primarily
optimize the transport or inversion path; it defines a localized
edit-preserve displacement on top of a source-conditioned RF trajectory.

### Preservation and Structure Control

Structure preservation has also been studied through explicit losses and
evaluation metrics. Edge-aware image manipulation introduces a structure
preservation loss to maintain pixel-level edge structure during diffusion
editing [@spl2026]. Metrics such as CLIP [@radford2021clip], DINOv2
[@oquab2023dinov2], SSIM [@wang2004ssim], and LPIPS [@zhang2018lpips] provide
different views of target alignment and source preservation. DeCE-RF differs
from image-space structure losses and post-processing: preservation is a
sampling-time clean-estimate correction gated by preserve geometry. The
evaluation protocol should therefore report edit-region alignment and
preserve-region fidelity separately under fixed evaluation masks.

## 3. Method

### 3.1 Problem Setting and Inputs

The input is a source image `I_s`, source prompt `c_s`, target prompt `c_t`,
and an operation descriptor `(o, rho)`, where `o` is an operation type and
`rho` is an optional relation. The current benchmark uses protocol-provided
operation and relation labels rather than an automatic parser:

```text
o in {add_object, add_decal, remove_object, replace, recolor}
rho in {above_host, on_face, on_surface, inside, remove_source_object}
```

This input condition is part of the method definition. If a baseline consumes
operation labels, grounding masks, or the same support masks, it should be
reported in a matched-input group. If a baseline does not receive these inputs,
it should be reported as an automatic or weaker-input comparison.

The source image is encoded into a latent `x_s`. The method obtains an editable
noisy latent `z_T` through source-conditioned RF inversion, then integrates a
controlled reverse ODE. At each timestep, the model is queried under source and
target conditions:

```text
v_src = v_theta(x_t, t, c_s),
v_tar = v_theta(x_t, t, c_t).
```

### 3.2 Clean Estimates and Velocity Conversion

Under the RF linear path,

```text
x_t = (1 - t) x_0 + t x_1.
```

A velocity prediction induces a clean estimate:

```text
x0_hat(x_t,t,c) = x_t - t v_theta(x_t,t,c).
```

For source and target conditions:

```text
x0_hat_src = x_t - t v_src,
x0_hat_tar = x_t - t v_tar.
```

If a velocity correction `u_t` is added to the source velocity, the clean
estimate changes as:

```text
x0_hat(v_src + u_t) = x0_hat_src - t u_t.
```

Therefore, to apply a desired clean displacement `Delta_t^0`,

```text
x0_hat(v_src + u_t) = x0_hat_src + Delta_t^0,
```

the velocity correction is:

```text
u_t = -Delta_t^0 / t_eff,
t_eff = max(t, epsilon_t).
```

In the current experiment runner, `epsilon_t = 0.05` for the linear-path
conversion. This prevents the `1/t` factor from becoming unstable near the end
of the reverse ODE. The implementation may also schedule or disable correction
terms at late timesteps through the experiment protocol.

### 3.3 Edit-Preserve Displacement

DeCE-RF keeps the source velocity as the base field:

```text
v_base = v_src.
```

The controlled velocity is:

```text
v_DeCE = v_src - t_eff^-1 Delta_t^0,
Delta_t^0 = Delta_edit + Delta_pres.
```

The edit displacement is:

```text
Delta_edit = lambda_e(t) M_e * (x0_hat_tar - x0_hat_src).
```

Using the RF clean-estimate relation:

```text
x0_hat_tar - x0_hat_src = -t (v_tar - v_src).
```

Thus:

```text
-t_eff^-1 Delta_edit
  approx lambda_e(t) M_e * (v_tar - v_src)
```

when `t_eff` equals `t`. This means the edit branch is a gated
target-source velocity correction in velocity form. The paper should state this
explicitly: the edit term alone is close to masked target-source guidance.

The preservation displacement is:

```text
Delta_pres = lambda_p(t) M_p * (x_s - x0_hat_src).
```

Its velocity form pulls the source-conditioned clean estimate back to the
source latent in preserve regions:

```text
-t_eff^-1 Delta_pres
  = lambda_p(t) M_p * (x0_hat_src - x_s) / t_eff.
```

This is the key complement to masked target guidance. Preservation is not
latent blending or post-processing; it is a sampling-time clean-estimate drift
correction in the same displacement space as editing.

### 3.4 Operation-Conditioned Geometry Estimator

The geometry estimator outputs `M_core`, `M_contact`, `M_edit`, and
`M_preserve`. It uses the operation descriptor and evidence maps:

```text
A_new       target/new-token attention
A_host      host-token attention
A_removed   removed/source-token attention
D_clean     ||x0_hat_tar - x0_hat_src||
D_vel       ||v_tar - v_src||
G           optional grounding or segmentation mask
R           relation map from rho
```

The default RF response map is:

```text
D_resp = norm(0.65 norm(D_clean) + 0.35 norm(D_vel)).
```

Candidate maps are selected by operation. Examples include:

```text
add_object + relation: relation_x_response = R * D_resp
add_decal: decal_surface_local_response
remove_object + grounding: seg_only
remove_object without grounding: A_removed * norm(D_clean)
recolor + relation: relation_only
```

For surface-local candidates, the implementation normalizes evidence within
the surface region and uses:

```text
S_surface = norm(R * (0.30 A_new_surface
                     + 0.50 D_clean_surface
                     + 0.20 D_vel_surface)).
```

The selected support score is postprocessed with the current matrix defaults:

```text
top percentile: 95
min area ratio: 0.02
max area ratio: 0.10
keep components: 1
dilate radius: 3
blur kernel: 3
```

Task protocols may override these defaults, but overrides must be recorded in
the command and metadata. Connected components are scored by support mass,
clean-response mass, area target, compactness, and relation overlap when a
relation map is available.

The final support is layered. `M_core` is the strongest edit region.
`M_contact` is a dilated soft ring around the core, optionally reduced at
strong latent structure edges. `M_edit` is the effective edit support combining
core and weak contact. `M_preserve` is the complement of object/contact support.
The controller uses `M_core` for strong edit pressure, `M_contact` for boundary
blending, and `M_preserve` for clean-estimate drift correction.

### 3.5 Feedback Weights and Preserve Projection

At each timestep, the controller computes clean-estimate diagnostics:

```text
current_delta = x0_hat_src - x_s
target_delta  = x0_hat_tar - x_s
target_gap    = x0_hat_tar - x0_hat_src
```

The edit target gap is measured in the edit mask, and preserve drift is:

```text
d_pres = RMS_M_preserve(x0_hat_src - x_s).
```

The main controller uses:

```text
w_edit = clip(1 + k_e deficit_e, w_e_min, w_e_max)
w_pres = clip(1 + k_p max(0, d_pres - tau_p), w_p_min, w_p_max).
```

In the current mainline, `k_e = 2.0`, `tau_p = 0.18`, `k_p = 2.5`,
`w_p_min = 1.0`, and `w_p_max = 1.65`; some decal and recolor protocols use
tighter preserve budgets. These values are experiment-protocol defaults and
must be reported with the run commands.

After weighting, the controller removes preserve-conflicting edit components.
Let the estimated preserve error be:

```text
r_p = M_p * (x0_hat_src - x_s).
```

Let `Delta_e` be the clean edit displacement after edit weighting and aligned
to the preserve mask. If:

```text
<M_p * Delta_e, r_p> > 0,
```

then the edit displacement is predicted to increase preserve error. The
projected edit displacement is:

```text
alpha = <M_p * Delta_e, r_p> / (||r_p||^2 + epsilon)
Delta_e <- Delta_e - gamma_proj * alpha * r_p.
```

`gamma_proj` is the projection strength. This projection matters mainly where
soft masks, contact regions, or imperfect support create overlap between edit
effects and preserve error. It should be ablated separately from fixed-weight
feedback.

### 3.6 ODE Integration and Logged Diagnostics

The reverse update is:

```text
x_{t-dt} = x_t + (t_next - t) v_DeCE.
```

Each run records the command, metadata, final image, support masks, and
per-step statistics. The logged statistics include edit target gap, edit
progress, preserve drift, adaptive edit weight, adaptive preserve weight,
projection norm, reconstruction-guidance norm, edit-guidance norm, total
velocity norm, runtime, and memory where available. These diagnostics are part
of the method evidence because DeCE-RF is a controller: final images should be
paired with trajectory behavior.

### 3.7 Algorithm

```text
Input: I_s, c_s, c_t, operation o, relation rho

1. Encode source image I_s into latent x_s.
2. Invert x_s under source condition c_s to obtain z_T.
3. Estimate support geometry:
   a. compute v_src, v_tar at support timesteps;
   b. compute D_clean and D_vel;
   c. collect A_new, A_host, A_removed, optional G, and relation R;
   d. select operation-conditioned candidate support;
   e. postprocess into M_core, M_contact, M_edit, M_preserve.
4. For each reverse ODE timestep:
   a. compute v_src, v_tar, x0_hat_src, x0_hat_tar;
   b. build Delta_edit and Delta_pres;
   c. compute edit gap and preserve drift;
   d. update w_edit and w_pres;
   e. project preserve-conflicting edit displacement;
   f. update x_t with v_src - t_eff^-1 (Delta_edit + Delta_pres).
5. Decode final latent.
```

## 4. Experiment Section Placeholder

The final experimental section should be inserted only after the active runs
finish. It should be written around quantified evidence, not qualitative
examples alone. The planned structure is:

```text
4.1 Tasks, inputs, and fixed-mask protocol
4.2 Metrics: edit-region alignment and preserve-region fidelity
4.3 Main comparison against RF and localized-editing baselines
4.4 Support geometry analysis
4.5 Feedback/projection ablation
4.6 Runtime, NFE, and memory
4.7 Failure analysis
```

The main paper-facing internal methods remain:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

`support_v3_fixed`, no-preserve, no-feedback, and no-projection variants belong
in component ablations. If external baselines receive different inputs
such as manual masks, operation labels, or grounding, the table must state that
input condition explicitly.

## 5. Discussion, Limitations, and Ethics

DeCE-RF separates localized RF editing into clean displacement, control
geometry, and feedback. This separation is useful because different failures
have different causes. A target response can be diffuse, support can select
the wrong region, or preserve drift can grow even when the edit forms. The
clean-estimate diagnostics make these cases visible during sampling rather
than only after final decoding.

The method has three main limitations. First, it depends on approximately
correct support. If the geometry estimator selects the wrong region, the
controller optimizes the wrong local problem. Second, local target formation
can remain weak for small objects, precise glyphs, and ambiguous replacement.
Third, removal and replacement require completion or attribute selection that a
base clean-displacement controller may not solve. Completion-prior or
replacement-route variants should therefore be reported as extensions, not
folded into the base DeCE-RF result.

The current implementation is SD3-specific and includes hyperparameters for
support postprocessing, contact blending, edit weights, preserve budgets, and
projection. These should be reported in the experiment protocol and tested by
ablation or sensitivity analysis where space allows.

Image editing methods can be misused for deception, impersonation, or
misleading visual evidence. DeCE-RF is a localized editing method and therefore
inherits these risks. Appropriate mitigations include watermarking or provenance
metadata for released examples, avoiding identity-sensitive demonstrations, and
releasing code with clear research-use documentation and limitations.

## 6. Conclusion

We presented DeCE-RF, a decoupled clean-estimate edit-preserve controller for
localized Rectified Flow editing. The method keeps the source velocity as the
base trajectory, writes local editing as a clean displacement, decomposes that
displacement into edit and preservation components, estimates layered
operation-conditioned control geometry, and adapts the edit-preserve balance
through clean-estimate feedback. The formulation makes explicit what is new
beyond masked target-source velocity: preservation is a clean-estimate drift
correction, edit and preservation share one displacement space, and the
controller monitors and projects the displacement during sampling. The
experimental section should now test this formulation with fixed-mask metrics,
input-matched baselines, component ablations, diagnostics, and failure
analysis.

