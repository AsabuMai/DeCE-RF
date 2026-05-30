# RF h-Edit Project: Updated Research Direction

## 1. Current Position

The project should no longer be positioned simply as:

> RF image editing with reconstruction/editing decomposition.

This framing is too close to several recent RF / flow-based editing works that already use source fidelity, target steering, velocity blending, flow decomposition, or mask preservation.

The updated direction should be:

> **Clean-estimate-space local control for Rectified Flow image editing.**

More specifically, the project should focus on:

> A region-adaptive closed-loop controller that diagnoses editing progress and preservation drift in the predicted clean-image space, then converts local clean-space constraints into Rectified Flow velocity corrections.

---

## 2. Why the Previous Framing Is Not Enough

The previous formulation was:

\[
\dot{x}_t = (v_{\mathrm{src}} + u_{\mathrm{rec}}) + u_{\mathrm{edit}} .
\]

This is still useful as the implementation-level decomposition, but it is not sufficient as the main novelty.

The general structure:

- source velocity / fidelity term,
- target edit velocity / steering term,
- mask preservation,

is already covered by many related RF editing methods.

Examples of nearby directions include:

- **FlowEdit**: source-to-target ODE mapping for flow-based editing.
- **SplitFlow**: flow decomposition and aggregation for complex edits.
- **SteerFlow**: source reconstruction / target editing velocity blending and adaptive masking.
- **FlowSlider**: fidelity term and steering term decomposition for flow-based editing.
- **ReFlex / RF-Solver / FireFlow**: inversion, feature / attention adaptation, and solver-oriented RF editing.

Therefore, if the project only claims:

> We split RF velocity into reconstruction and editing components.

then the novelty is weak.

---

## 3. The Real Differentiating Point

The most distinctive part of the current project is the clean-estimate-space control chain:

\[
\hat{x}_0(x_t,t) = x_t - t\,v_\theta(x_t,t)
\]

and

\[
\Delta x_0 \rightarrow u = -\frac{\Delta x_0}{t}.
\]

This means that local editing and preservation constraints are not only defined in velocity space.

Instead, the method first defines constraints in the predicted clean-image space:

- local reference constraints,
- edit-region constraints,
- preserve-region constraints,
- trajectory preservation,
- mask-guided local control,

and then maps these clean-space corrections back into RF velocity corrections.

This gives a clearer and more specific mechanism than simple source-target velocity blending.

---

## 4. Updated Core Idea

The updated method should be described as:

> **Region-Adaptive Clean-Estimate Control.**

At each ODE step, the method should:

1. Predict the current clean estimate:
   \[
   \hat{x}_{0,t} = x_t - t\,v_\theta(x_t,t)
   \]

2. Measure edit progress in the edit region.

3. Measure preservation drift in the preserve region.

4. Dynamically adjust edit and preserve velocities.

5. Remove or suppress editing components that damage the preserve region.

6. Convert clean-space corrections into RF velocity updates:
   \[
   u = -\frac{\Delta x_0}{t}.
   \]

---

## 5. Updated Main Formulation

The implementation-level ODE remains:

\[
\dot{x}_t =
\underbrace{(v_{\mathrm{src}} + u_{\mathrm{rec}})}_{\text{reconstruction-aware base field}}
+
\underbrace{u_{\mathrm{edit}}}_{\text{editing field}}.
\]

However, the main contribution should be reframed as:

\[
u_{\mathrm{rec}}, u_{\mathrm{edit}}
=
\mathcal{C}
\left(
\hat{x}_{0,t},
x_{\mathrm{src}},
M_{\mathrm{edit}},
M_{\mathrm{preserve}},
c_{\mathrm{src}},
c_{\mathrm{edit}}
\right),
\]

where \(\mathcal{C}\) is a clean-estimate-space local controller.

The controller operates in clean-image space and returns velocity corrections through the RF path relation.

---

## 6. Clean-Estimate-Space Diagnostics

### 6.1 Edit Progress

A simple edit-progress metric is:

\[
\text{edit\_change}(t)
=
\left\|
M_{\mathrm{edit}}
\odot
(\hat{x}_{0,t} - x_{\mathrm{src}})
\right\|.
\]

However, this only measures how much the edit region changes, not whether it changes in the correct direction.

A better target-direction progress metric is:

\[
\Delta_{\mathrm{cur}}
=
M_{\mathrm{edit}}\odot(\hat{x}_{0,t}-x_{\mathrm{src}}),
\]

\[
\Delta_{\mathrm{tar}}
=
M_{\mathrm{edit}}\odot(\hat{x}_{0}^{\mathrm{tar}}-x_{\mathrm{src}}),
\]

\[
p_{\mathrm{edit}}(t)
=
\frac{
\langle \Delta_{\mathrm{cur}}, \Delta_{\mathrm{tar}}\rangle
}{
\|\Delta_{\mathrm{tar}}\|^2+\epsilon
}.
\]

This measures whether the current clean estimate is moving along the target editing direction.

---

### 6.2 Preserve Drift

Preserve drift can be defined as:

\[
d_{\mathrm{pres}}(t)
=
\frac{
\left\|
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_{\mathrm{src}})
\right\|
}{
\left\|
M_{\mathrm{preserve}}\odot x_{\mathrm{src}}
\right\|+\epsilon
}.
\]

This measures how much the non-edit region has drifted away from the source image.

---

## 7. Adaptive Controller

The next method version should introduce an adaptive controller with three mechanisms.

---

### 7.1 Adaptive Edit Boost

If the edit progress is insufficient, increase the editing strength.

Condition:

\[
p_{\mathrm{edit}}(t) < \tau_{\mathrm{edit}}.
\]

Adaptive edit scale:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_{\mathrm{edit}}^0
\left[
1
+
k_{\mathrm{edit}}
\cdot
\sigma(\tau_{\mathrm{edit}} - p_{\mathrm{edit}}(t))
\right].
\]

Possible branches affected:

- target-source velocity branch,
- edit reference branch,
- anchor branch,
- target clean-estimate correction branch.

---

### 7.2 Adaptive Preserve Lock

If the preserve region drifts too much, increase reconstruction / trajectory preservation.

Condition:

\[
d_{\mathrm{pres}}(t) > \tau_{\mathrm{pres}}.
\]

Adaptive preserve scale:

\[
\lambda_{\mathrm{rec}}(t)
=
\lambda_{\mathrm{rec}}^0
\left[
1
+
k_{\mathrm{pres}}
\cdot
\sigma(d_{\mathrm{pres}}(t)-\tau_{\mathrm{pres}})
\right].
\]

Possible branches affected:

- reconstruction guidance,
- trajectory preservation,
- source anchor correction,
- preserve-region correction.

---

### 7.3 Orthogonal Conflict Projection

If the editing velocity damages the preserve region, remove its destructive component.

Let \(g_{\mathrm{pres}}\) be the preserve-drift direction:

\[
g_{\mathrm{pres}}
=
\nabla_{x_t}
\left\|
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_{\mathrm{src}})
\right\|^2.
\]

Then the projected editing velocity is:

\[
u_{\mathrm{edit}}^{\perp}
=
u_{\mathrm{edit}}
-
\frac{
\max(0,\langle u_{\mathrm{edit}}, g_{\mathrm{pres}}\rangle)
}{
\|g_{\mathrm{pres}}\|^2+\epsilon
}
g_{\mathrm{pres}}.
\]

This removes the component of the editing velocity that increases preserve-region drift.

Because ODE direction signs can be implementation-dependent, a practical version should include a one-step diagnostic:

1. Apply the current edit velocity for a small virtual step.
2. Check whether preserve drift increases.
3. If it increases, project or scale down the edit component.

---

## 8. Updated Method Name

Possible working names:

### Option 1

**Region-Adaptive Clean-Estimate Control for Rectified Flow Editing**

### Option 2

**Clean-Estimate Controlled Rectified Flow Editing**

### Option 3

**Closed-Loop Clean-Estimate Control for Local RF Editing**

The safest current name is:

> **Region-Adaptive Clean-Estimate Control**

because it emphasizes the real differentiating mechanism.

---

## 9. Difference from Related Work

### FlowEdit

FlowEdit constructs a source-to-target ODE for flow-based image editing.

Our difference:

> FlowEdit focuses on global source-to-target transport, while this project focuses on localized clean-estimate-space constraints and their conversion into RF velocity corrections.

---

### SplitFlow

SplitFlow decomposes complex editing prompts into multiple semantic flows and aggregates them.

Our difference:

> SplitFlow addresses semantic disentanglement and multi-prompt flow composition; this project addresses local edit / preserve control in clean-estimate space.

---

### SteerFlow

SteerFlow is highly related because it includes inversion, source fidelity, target editing velocity blending, trajectory interpolation, and adaptive masking.

Our difference must be:

> This project uses clean-estimate-space diagnostics and a closed-loop controller to adapt edit and preserve velocities, rather than relying only on fixed velocity interpolation or adaptive masking.

---

### FlowSlider

FlowSlider decomposes flow editing into fidelity and steering terms.

Our difference:

> FlowSlider controls continuous edit strength through fidelity / steering adjustment; this project controls local clean-space edit progress and preserve drift through a region-adaptive closed-loop controller.

---

### ReFlex / RF-Solver / FireFlow

These methods focus more on inversion quality, feature / attention adaptation, or solver quality.

Our difference:

> This project focuses on clean-estimate-space local control and region-adaptive velocity correction.

---

## 10. Implementation Plan

The next implementation should be called:

\[
\texttt{adaptive\_full}
\]

It should be compared directly against the current full method.

---

### 10.1 New Controller File

Create a new file:

```text
adaptive_controller.py
```

Suggested functions:

```python
compute_edit_progress(...)
compute_preserve_drift(...)
adaptive_edit_scale(...)
adaptive_preserve_scale(...)
project_edit_against_preserve(...)
clean_delta_to_velocity(...)
```

The existing `clean_delta_to_velocity` function in `energies.py` should become a central API.

---

### 10.2 Code Locations

Important current locations:

```text
rf_h_edit_project/sd3_hrec.py:2656
```

Existing \(x0_{\mathrm{tar}}\) and \(x0_{\mathrm{src}}\) are available here.  
This is a good location for computing edit progress and preserve drift.

```text
rf_h_edit_project/sd3_hrec.py:2690
```

This is where \(v_{\mathrm{rec}}\) is generated.  
Add adaptive reconstruction / preserve weights here.

```text
rf_h_edit_project/sd3_hrec.py:2772
```

This is where `v_edit_terms` are generated.  
Add adaptive edit / reference weights here.

```text
rf_h_edit_project/sd3_hrec.py:3150
```

This is where all edit guidance is summarized.  
Add projection, clipping, or conflict removal here.

```text
rf_h_edit_project/energies.py:21
```

The `clean_delta_to_velocity` function is mathematically central.  
Refactor around this API.

---

## 11. Minimal New Method: adaptive_full

Do not redesign the whole model.

Add only three mechanisms:

### 1. Adaptive Edit Boost

When edit progress is too low, increase the target / reference / anchor edit guidance.

### 2. Adaptive Preserve Lock

When preserve drift is too high, increase reconstruction / trajectory preservation.

### 3. Orthogonal Conflict Projection

When edit velocity conflicts with preserve stability, project out the destructive component.

This should convert the method from:

> a fixed combination of preservation and editing terms

to:

> a clean-estimate-space closed-loop controller.

---

## 12. Experiments for the Next Stage

Run:

\[
\texttt{full}
\quad \text{vs.} \quad
\texttt{adaptive\_full}
\]

on four tasks.

---

### Task 1: panda_sunglasses

Goal:

- local accessory insertion,
- verify mask and preserve control,
- check whether sunglasses become stronger without damaging the panda.

---

### Task 2: dog_sunglasses

Goal:

- test whether the local insertion behavior generalizes beyond panda.

---

### Task 3: panda_to_tiger

Goal:

- test object replacement,
- check whether adaptive control reduces overlay-like tiger texture failure.

---

### Task 4: backpack_remove

Goal:

- test removal / preservation-sensitive editing,
- ensure adaptive_full does not over-edit or damage background.

---

## 13. Evaluation Questions

For each task, compare:

- `full`
- `adaptive_full`

and answer:

1. Does adaptive_full improve edit success?
2. Does adaptive_full reduce preserve-region drift?
3. Does adaptive_full reduce overlay-like artifacts?
4. Does adaptive_full preserve source identity and background better?
5. Does adaptive_full avoid degrading tasks that already work?

---

## 14. Diagnostics to Save

For every run, save:

```text
edit_progress_curve
preserve_drift_curve
adaptive_edit_scale_curve
adaptive_preserve_scale_curve
edit_preserve_conflict_curve
mask_area_curve
velocity_norms
cosine_similarity_between_velocity_branches
```

These diagnostics are important because the proposed method is a controller, not just a final image generator.

The paper should show not only final images, but also that the controller reacts to edit progress and preserve drift.

---

## 15. Revised Contribution Statement

The revised contribution should not be:

> We split RF editing into reconstruction and editing terms.

That is not novel enough.

The revised contribution should be:

> We introduce a clean-estimate-space local control interface for Rectified Flow image editing. The method diagnoses edit progress and preserve drift in the predicted clean-image space, converts local clean-space constraints into RF velocity corrections, and adaptively balances editing and preservation through a closed-loop controller.

---

## 16. Current Conclusion

The project is still worth pursuing, but the innovation must be reframed.

Current similarity with related work:

> Medium to high.

Potentially defensible novelty:

> Clean-estimate-space local control + adaptive preservation/editing controller + conflict-aware velocity projection.

The next step is to implement `adaptive_full` and test whether it clearly improves over the current `full` setting.

If `adaptive_full` improves local insertion while not degrading object removal / replacement tasks, then the project will have a much clearer paper contribution.
