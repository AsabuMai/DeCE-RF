# RF h-Edit: Adaptive Full v1 To-Do Plan

## 0. Current Decision

The project should no longer rely on the broad claim:

> We decompose Rectified Flow editing into reconstruction and editing terms.

This is too close to existing RF / flow editing works.

The updated research point should be:

> **Clean-estimate-space local closed-loop control for Rectified Flow editing.**

The goal is to make `adaptive_full` evolve from a simple adaptive scaling prototype into a real controller that diagnoses the current editing trajectory in clean-estimate space and dynamically adjusts RF velocity corrections.

---

## 1. Current Status: adaptive_full v0

`adaptive_full v0` is already implemented.

It currently contains:

- RMS-style edit / preserve diagnostics
- adaptive edit / preserve weights
- masked opposing projection
- logging of controller responses

This is useful, but it is still a **v0 controller**.

### Limitation of v0

The current edit diagnostic mainly measures:

\[
\|M_{\mathrm{edit}} \odot (x_0^{tar} - x_0^{src})\|
\]

or a similar RMS-style difference.

This only tells us:

> whether there is enough change magnitude.

It does **not** tell us:

> whether the current trajectory is moving in the correct target direction.

Therefore, v0 can amplify edit signals, but it cannot reliably distinguish:

- meaningful semantic edit
- irrelevant visual change
- texture overlay
- wrong-direction movement

---

## 2. Main Goal of adaptive_full v1

`adaptive_full v1` should implement a **direction-aware clean-estimate-space controller**.

At each ODE step, it should:

1. Predict the current clean estimate:

   \[
   \hat{x}_{0,t} = x_t - t\,v_\theta(x_t,t)
   \]

2. Measure whether the current edit is moving toward the target direction.

3. Measure whether the preserve region is drifting away from the source image.

4. Adapt edit strength if edit progress is insufficient.

5. Adapt preserve strength if preserve drift is too large.

6. Project or suppress edit velocity if it damages the preserve region.

---

## 3. Core Formula: Clean Estimate

Use the linear Rectified Flow path:

\[
x_t = (1-t)x_0 + t x_1
\]

The clean estimate is:

\[
\hat{x}_0(x_t,t) = x_t - t\,v_\theta(x_t,t)
\]

This is the central interface of the method.

Local constraints should be defined in clean-estimate space, then converted back into RF velocity corrections.

---

## 4. Core Formula: Clean-Space Delta to Velocity Correction

If we want to move the clean estimate by:

\[
\Delta x_0
\]

then the corresponding RF velocity correction is:

\[
u = -\frac{\Delta x_0}{t}
\]

This relation should remain a core API.

Recommended function:

```python
clean_delta_to_velocity(delta_x0, t_scalar, eps=1e-6)
```

Expected behavior:

```python
u = - delta_x0 / max(t_scalar, eps)
```

This should be treated as the mathematical bridge between clean-space control and RF velocity correction.

---

## 5. v1 Component 1: Directional Edit Progress

### Problem with RMS edit progress

RMS edit progress only measures how much the edit region changes.

It does not measure whether the change is in the correct target direction.

### Directional edit progress

Define:

\[
\Delta_{\mathrm{cur}}
=
M_{\mathrm{edit}}\odot(\hat{x}_{0,t}-x_{0}^{ref})
\]

\[
\Delta_{\mathrm{tar}}
=
M_{\mathrm{edit}}\odot(\hat{x}_{0}^{tar}-x_{0}^{ref})
\]

Then define:

\[
p_{\mathrm{edit}}(t)
=
\frac{
\langle \Delta_{\mathrm{cur}}, \Delta_{\mathrm{tar}}\rangle
}{
\|\Delta_{\mathrm{tar}}\|^2+\epsilon
}
\]

### Interpretation

- \(p_{\mathrm{edit}} \approx 0\): almost no progress toward target direction
- \(p_{\mathrm{edit}} \approx 1\): target direction is mostly achieved
- \(p_{\mathrm{edit}} < 0\): current edit is moving against the target direction
- \(p_{\mathrm{edit}} > 1\): possible over-editing

### Implementation notes

Keep both diagnostics:

```python
edit_progress_rms
edit_progress_directional
```

Do not remove RMS yet. RMS is still useful for debugging edit magnitude.

---

## 6. v1 Component 2: Preserve Drift

Preserve drift can remain RMS-based because preservation means staying close to source.

Define:

\[
d_{\mathrm{pres}}(t)
=
\frac{
\left\|
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_{0}^{ref})
\right\|
}{
\left\|
M_{\mathrm{preserve}}\odot x_{0}^{ref}
\right\|+\epsilon
}
\]

### Interpretation

- small \(d_{\mathrm{pres}}\): non-edit region is preserved
- large \(d_{\mathrm{pres}}\): non-edit region is drifting

### Required logs

```python
preserve_drift_rms
preserve_drift_max
preserve_drift_curve
```

---

## 7. v1 Component 3: Adaptive Edit Boost

If directional edit progress is too low, increase edit strength.

Condition:

\[
p_{\mathrm{edit}}(t) < \tau_{\mathrm{edit}}
\]

Adaptive edit scale:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_{\mathrm{edit}}^0
\left[
1+
k_{\mathrm{edit}}
\cdot
\max(0,\tau_{\mathrm{edit}}-p_{\mathrm{edit}}(t))
\right]
\]

Then clamp:

\[
\lambda_{\mathrm{edit}}(t)
\in
[\lambda_{\min}, \lambda_{\max}]
\]

### Suggested affected branches

Start with only one or two branches:

- target-source velocity branch
- anchor / edit-reference branch

Do not adapt all branches at once in v1.

### Required logs

```python
adaptive_edit_scale
edit_progress_directional
edit_progress_rms
target_delta_norm
current_delta_norm
```

---

## 8. v1 Component 4: Adaptive Preserve Lock

If preserve drift is too high, increase reconstruction / trajectory preservation.

Condition:

\[
d_{\mathrm{pres}}(t) > \tau_{\mathrm{pres}}
\]

Adaptive preserve scale:

\[
\lambda_{\mathrm{pres}}(t)
=
\lambda_{\mathrm{pres}}^0
\left[
1+
k_{\mathrm{pres}}
\cdot
\max(0,d_{\mathrm{pres}}(t)-\tau_{\mathrm{pres}})
\right]
\]

Then clamp:

\[
\lambda_{\mathrm{pres}}(t)
\in
[\lambda_{\min}, \lambda_{\max}]
\]

### Suggested affected branches

- reconstruction guidance
- trajectory preservation
- source anchor correction
- preserve-region correction

### Required logs

```python
adaptive_preserve_scale
preserve_drift_rms
preserve_drift_max
```

---

## 9. v1 Component 5: Clean-Effect Projection

### Why velocity-space projection is risky

The actual ODE update has the form:

\[
x_{t-1}=x_t+(t_{i-1}-t_i)v
\]

The sign of \((t_{i-1}-t_i)\) can make direct velocity-space projection confusing.

Therefore, v1 should reason in clean-estimate space.

### Clean effect of edit velocity

If the edit velocity is:

\[
u_{\mathrm{edit}}
\]

then its effect on the clean estimate is:

\[
\Delta \hat{x}_0^{edit} = -t\,u_{\mathrm{edit}}
\]

### Preserve error

\[
e_{\mathrm{pres}}
=
M_{\mathrm{preserve}}\odot(\hat{x}_{0,t}-x_{0}^{ref})
\]

### Preserve-region edit effect

\[
\Delta_{\mathrm{pres}}^{edit}
=
M_{\mathrm{preserve}}\odot(-t\,u_{\mathrm{edit}})
\]

If:

\[
\langle e_{\mathrm{pres}}, \Delta_{\mathrm{pres}}^{edit}\rangle > 0
\]

then edit velocity is increasing preserve drift.

### Projection in clean-effect space

\[
\Delta_{\mathrm{pres}}^{edit,\perp}
=
\Delta_{\mathrm{pres}}^{edit}
-
\frac{
\langle \Delta_{\mathrm{pres}}^{edit}, e_{\mathrm{pres}}\rangle
}{
\|e_{\mathrm{pres}}\|^2+\epsilon
}
e_{\mathrm{pres}}
\]

Then map back to velocity:

\[
u_{\mathrm{edit}}^{\perp}
=
-\frac{\Delta x_0^{edit,\perp}}{t}
\]

### Simpler first implementation

For the first v1 implementation, do one of the following:

1. scale down preserve-region edit component when it increases drift
2. mask out destructive preserve-region component
3. implement full projection only after diagnostics are stable

### Required logs

```python
conflict_score
projection_ratio
preserve_drift_before_projection
preserve_drift_after_projection_estimate
```

---

## 10. Implementation Plan

### Current state

The adaptive controller is currently implemented directly inside:

```text
rf_h_edit_project/sd3_hrec.py
```

This is acceptable for validation.

### Later refactor

After v1 works, move the logic into:

```text
adaptive_controller.py
```

Suggested functions:

```python
compute_directional_edit_progress(...)
compute_preserve_drift(...)
compute_adaptive_edit_scale(...)
compute_adaptive_preserve_scale(...)
compute_clean_effect_projection(...)
apply_adaptive_controller(...)
```

---

## 11. Code Locations

Important locations in the current project:

```text
rf_h_edit_project/sd3_hrec.py
```

Main controller logic is currently inside this file.

Relevant locations:

```text
sd3_hrec.py: around x0_tar and x0_src_step computation
```

Use this area to compute:

- directional edit progress
- preserve drift

```text
sd3_hrec.py: around v_rec generation
```

Use this area to apply:

- adaptive preserve lock
- adaptive reconstruction scaling

```text
sd3_hrec.py: around v_edit_terms generation
```

Use this area to apply:

- adaptive edit boost
- branch-specific adaptive edit scaling

```text
sd3_hrec.py: around final edit guidance aggregation
```

Use this area to apply:

- clean-effect projection
- clipping
- conflict removal

```text
rf_h_edit_project/energies.py
```

Keep `clean_delta_to_velocity` as a central function.

---

## 12. Experiment Plan

Run:

```text
full
adaptive_full_v0
adaptive_full_v1
```

on the four current pretty tasks.

---

### Task 1: cat_crown

Purpose:

- local accessory / decoration insertion
- check whether adaptive edit boost improves local edit visibility
- check whether preserve lock prevents face / background drift

---

### Task 2: dog_sunglasses

Purpose:

- local accessory insertion
- test generalization beyond panda
- current v0 does not clearly improve this task, so it is important for v1

---

### Task 3: mug_heart

Purpose:

- local symbol / pattern addition
- good for testing whether edit progress is target-directional rather than just large visual change

---

### Task 4: backpack_remove_toy_charm

Purpose:

- removal / preservation-sensitive editing
- test whether adaptive controller avoids damaging the background or nearby object regions

---

## 13. Required Comparisons

For each task, compare:

```text
full
adaptive_full_v0
adaptive_full_v1
```

Optional:

```text
direct_target
base_only
anchor_only
```

### Main questions

1. Does v1 improve edit success compared with full?
2. Does v1 reduce preserve-region drift?
3. Does v1 improve over v0?
4. Does v1 reduce overlay-like artifacts?
5. Does v1 avoid degradation on tasks that already work?

---

## 14. Required Logs

Each run should save:

```text
edit_progress_rms_curve
edit_progress_directional_curve
preserve_drift_curve
adaptive_edit_scale_curve
adaptive_preserve_scale_curve
conflict_score_curve
projection_ratio_curve
edit_velocity_norm_curve
rec_velocity_norm_curve
mask_area_curve
cosine_velocity_branch_curves
```

These curves are important because the contribution is a controller, not only final image quality.

---

## 15. Expected Evidence for Innovation

To claim innovation, the results should show:

1. Directional edit progress increases more stably in v1.
2. Preserve drift is lower or better controlled in v1.
3. Adaptive edit scale responds when progress is insufficient.
4. Adaptive preserve scale responds when drift is high.
5. Projection activates when edit velocity damages preserve regions.
6. Final images show better edit-faithfulness trade-off than full and v0.

---

## 16. Success Criteria

### Strong success

`adaptive_full_v1` improves at least three of the four tasks:

- stronger local edit
- lower preserve drift
- fewer overlay-like artifacts
- better subjective visual quality

### Moderate success

`adaptive_full_v1` improves local insertion tasks but not object replacement / removal.

This is still useful and can define the method scope as:

> faithful local editing

### Weak success

Diagnostics react correctly, but final images do not improve.

This means the controller design is reasonable, but the editing velocity branch itself may still be weak.

---

## 17. Revised Research Claim

If v1 works, the project can claim:

> We propose a clean-estimate-space local closed-loop controller for Rectified Flow image editing. At each ODE step, the controller measures directional edit progress and preserve-region drift in the predicted clean-image space, then adaptively adjusts edit and preserve velocity corrections through edit boosting, preserve locking, and conflict-aware projection.

Do **not** claim:

> We are the first to decompose reconstruction and editing in RF editing.

That is not strong enough and is too close to existing work.

---

## 18. Immediate To-Do List

### Implementation

- [ ] Keep v0 implementation unchanged as baseline.
- [ ] Add directional edit progress computation.
- [ ] Add logs for RMS and directional progress.
- [ ] Add adaptive edit boost based on directional progress.
- [ ] Keep preserve drift RMS and add normalized preserve drift logs.
- [ ] Add clean-effect projection diagnostic.
- [ ] Add optional clean-effect projection / suppress mode.
- [ ] Save all controller curves.

### Experiments

- [ ] Run full on four pretty tasks.
- [ ] Run adaptive_full_v0 on four pretty tasks.
- [ ] Run adaptive_full_v1 on four pretty tasks.
- [ ] Generate comparison panels.
- [ ] Summarize success / failure per task.

### Documentation

- [ ] Update method document to mark v0 as implemented.
- [ ] Add v1 target: directional progress + clean-effect projection.
- [ ] Update task list to:
  - cat_crown
  - dog_sunglasses
  - mug_heart
  - backpack_remove_toy_charm
- [ ] Update contribution statement to focus on clean-estimate-space closed-loop control.

---

## 19. One-Sentence Summary

The next step is to move from `adaptive_full v0`, which only adapts based on RMS-style change magnitude, to `adaptive_full v1`, which uses directional edit progress, preserve drift, and clean-effect projection to form a true clean-estimate-space local closed-loop controller.
