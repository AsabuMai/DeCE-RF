# RF h-Edit: Mainline Research Summary and Next-Step Plan

## 0. Key Decision

The main research direction is **not** support proposal itself.

The support module is important, but it is not the core contribution.

The actual mainline should be:

\[
\boxed{
\text{clean-estimate-space diagnostics}
+
\text{closed-loop adaptive RF velocity control}
}
\]

In short:

> Support decides **where to edit / preserve**.  
> The controller decides **how to edit and how to preserve**.

Therefore, support-v3 should be treated as an auxiliary spatial interface, while the central method should focus on the clean-estimate-space controller.

---

## 1. Why Support Is Not the Mainline

Recently, we spent much effort improving support proposal because bad support makes the controller impossible to evaluate.

For example:

- `cat_crown`: if support does not cover the head-top region, the controller cannot place the crown.
- `mug_heart`: if support does not cover the mug surface, the controller cannot add the heart decal correctly.
- `backpack_remove_toy_charm`: support can be correct, but the object still remains if the editing dynamics cannot remove it.
- `dog_sunglasses`: support is already good, so it can be used as a positive local-editing case.

This shows that support is a **necessary condition** for fair evaluation, but it is not the main contribution.

The paper should not be framed as:

> We propose a new support / mask generation method.

Instead, it should be framed as:

> We propose a clean-estimate-space closed-loop controller for Rectified Flow image editing, with support masks used as spatial interfaces.

---

## 2. Core Mathematical Interface

The method is based on the linear Rectified Flow path:

\[
x_t = (1-t)x_0 + t x_1.
\]

The RF velocity field is:

\[
v_\theta(x_t,t).
\]

The path-induced clean estimate is:

\[
\boxed{
\hat{x}_0(x_t,t)
=
x_t - t\,v_\theta(x_t,t)
}
\]

This is the key interface.

Instead of controlling only in velocity space, we first evaluate the current image state in the clean-estimate space.

If we want to impose a clean-space correction:

\[
\Delta x_0,
\]

then the corresponding RF velocity correction is:

\[
\boxed{
u = -\frac{\Delta x_0}{t}
}
\]

This chain is central:

\[
\hat{x}_0
\rightarrow
\text{clean-space diagnostics}
\rightarrow
\Delta x_0
\rightarrow
u
\rightarrow
\text{RF ODE update}.
\]

---

## 3. Main Controller Concept

The current implementation-level dynamics can be written as:

\[
\dot{x}_t
=
\underbrace{(v_{\mathrm{src}} + u_{\mathrm{rec}})}_{\text{reconstruction-aware base field}}
+
\underbrace{u_{\mathrm{edit}}}_{\text{editing field}}.
\]

However, the main contribution should not be the decomposition itself.

The main contribution should be the **closed-loop controller** that adaptively changes these terms based on clean-estimate-space diagnostics.

The controller should answer three questions at each step:

1. Is the edit progressing toward the target?
2. Is the preserve region drifting away from the source?
3. Is the edit velocity damaging the preserve region?

---

## 4. Directional Edit Progress

RMS edit magnitude is not enough. It only measures whether something changed.

For adaptive control, we need to know whether the change is moving toward the target direction.

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

Then directional edit progress is:

\[
\boxed{
p_{\mathrm{edit}}(t)
=
\frac{
\langle \Delta_{\mathrm{cur}}, \Delta_{\mathrm{tar}}\rangle
}{
\|\Delta_{\mathrm{tar}}\|^2+\epsilon
}
}
\]

Interpretation:

- \(p_{\mathrm{edit}}\approx 0\): edit has not progressed toward the target.
- \(p_{\mathrm{edit}}\approx 1\): target-direction edit is mostly achieved.
- \(p_{\mathrm{edit}}<0\): current change moves against the target.
- \(p_{\mathrm{edit}}>1\): possible over-editing.

This should drive adaptive edit boost.

---

## 5. Preserve Drift

Preserve drift measures whether the non-edit region remains faithful to the source.

Define:

\[
\boxed{
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
}
\]

Interpretation:

- small \(d_{\mathrm{pres}}\): preserve region is stable.
- large \(d_{\mathrm{pres}}\): preserve region is drifting.

This should drive adaptive preserve lock.

---

## 6. Adaptive Edit Boost

If edit progress is insufficient:

\[
p_{\mathrm{edit}}(t)<\tau_{\mathrm{edit}},
\]

then increase editing strength:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_{\mathrm{edit}}^0
\left[
1+
k_{\mathrm{edit}}
\cdot
\max(0,\tau_{\mathrm{edit}}-p_{\mathrm{edit}}(t))
\right].
\]

Clamp the scale:

\[
\lambda_{\mathrm{edit}}(t)
\in
[\lambda_{\min},\lambda_{\max}].
\]

This should affect only selected editing branches first, such as:

- target-source velocity branch,
- anchor / edit reference branch.

Do not adapt all branches at once in the first version.

---

## 7. Adaptive Preserve Lock

If preserve drift is too large:

\[
d_{\mathrm{pres}}(t)>\tau_{\mathrm{pres}},
\]

then increase preservation strength:

\[
\lambda_{\mathrm{pres}}(t)
=
\lambda_{\mathrm{pres}}^0
\left[
1+
k_{\mathrm{pres}}
\cdot
\max(0,d_{\mathrm{pres}}(t)-\tau_{\mathrm{pres}})
\right].
\]

This can affect:

- reconstruction guidance,
- trajectory preserve,
- source anchor correction,
- preserve-region correction.

The goal is to stop target guidance from globally changing the image.

---

## 8. Clean-Effect Projection

Velocity-space projection can be confusing because the ODE step direction depends on the timestep sign.

So conflict should be diagnosed in clean-estimate space.

For edit velocity:

\[
u_{\mathrm{edit}},
\]

its clean-estimate effect is:

\[
\Delta \hat{x}_{0}^{edit}
=
-t\,u_{\mathrm{edit}}.
\]

Preserve error:

\[
e_{\mathrm{pres}}
=
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_{0}^{ref}).
\]

Preserve-region effect of edit velocity:

\[
\Delta_{\mathrm{pres}}^{edit}
=
M_{\mathrm{preserve}}
\odot
\Delta \hat{x}_{0}^{edit}.
\]

If:

\[
\langle e_{\mathrm{pres}}, \Delta_{\mathrm{pres}}^{edit}\rangle > 0,
\]

then the edit velocity increases preserve-region drift.

Suppress or project this component:

\[
\Delta_{\mathrm{pres}}^{edit,\perp}
=
\Delta_{\mathrm{pres}}^{edit}
-
\frac{
\langle \Delta_{\mathrm{pres}}^{edit},e_{\mathrm{pres}}\rangle
}{
\|e_{\mathrm{pres}}\|^2+\epsilon
}
e_{\mathrm{pres}}.
\]

Then map it back to velocity:

\[
u_{\mathrm{edit}}^{\perp}
=
-\frac{\Delta \hat{x}_{0}^{edit,\perp}}{t}.
\]

This is an important part of the controller because it directly uses clean-estimate-space reasoning.

---

## 9. Correct Role of Support

Support provides the spatial masks:

\[
M_{\mathrm{edit}},\quad M_{\mathrm{preserve}}.
\]

It tells the controller:

- where to measure edit progress,
- where to measure preserve drift,
- where to allow editing velocity,
- where to apply reconstruction / preservation.

But support is not the main method.

The support module can be:

- manual support,
- attention support,
- operation-aware support,
- segmentation / grounding support.

The controller should be evaluated under a fixed reliable support first.

---

## 10. How to Use support-v3 Results

The current support-v3 results should be used as follows.

### `cat_crown`

Use as a support-v3 positive example.

It shows that relation-aware support can improve spatial insertion.

### `dog_sunglasses`

Use as a non-regression example.

It shows that support-v3 does not destroy strong local attention cases.

### `mug_heart`

Use as controller bottleneck evidence.

Support can localize the decal region, but outside-mask drift remains too large.  
This is the best task for testing stronger preserve lock and clean-effect projection.

### `backpack_remove_toy_charm`

Use as limitation / removal-specific failure analysis.

Support is already reasonably good, but the ordinary editing velocity cannot remove and fill the object correctly.  
Removal needs a separate removal controller.

---

## 11. Main Experiment Should Return to Controller

To keep the research mainline clear, run controller experiments with fixed reliable support.

For example:

| Task | Fixed support choice |
|---|---|
| cat_crown | support-v3 relation support |
| dog_sunglasses | attention_x_clean support |
| mug_heart | surface support |
| backpack_remove_toy_charm | manual / segmentation support for removal analysis |

Then compare:

```text
fixed full
adaptive_full_v0
adaptive_full_v1
```

The key question is:

> Does adaptive_full_v1 improve edit/preserve trade-off under the same support?

---

## 12. Main Metrics for Controller Evaluation

### Editing quality

- CLIPΔ
- directional edit progress
- inside-mask L1
- visual target success

### Preservation quality

- outside-mask L1
- SSIM
- preserve drift curve
- background / identity consistency

### Controller behavior

- edit progress curve
- preserve drift curve
- adaptive edit scale curve
- adaptive preserve scale curve
- projection ratio curve
- edit/preserve conflict curve

The controller claim needs curves, not only final images.

---

## 13. Current Next Step

The next concrete task should be:

\[
\boxed{
\text{fix support, compare controller variants}
}
\]

Specifically:

1. Choose stable support for `cat_crown`, `dog_sunglasses`, and `mug_heart`.
2. Run:
   - `fixed_full`
   - `adaptive_full_v0`
   - `adaptive_full_v1`
3. Measure:
   - CLIPΔ
   - outside-mask L1
   - SSIM
   - directional edit progress
   - preserve drift
4. Check whether v1 reduces drift without killing the edit.

For `mug_heart`, the target is:

\[
\Delta_{\mathrm{CLIP}}>0
\]

and:

\[
\mathrm{outside\text{-}mask\ L1}<0.015.
\]

This is a good controller validation target.

---

## 14. Paper Narrative

The paper should be framed as:

> We introduce a clean-estimate-space closed-loop controller for Rectified Flow image editing. The controller diagnoses edit progress and preserve drift in the predicted clean-image space, then adapts RF velocity corrections to balance editing and preservation.

Support should be described as:

> a spatial interface used by the controller.

The paper should not be framed as:

> We propose a new support proposal method.

Support-v3 can be included as a practical support module or ablation, but the main contribution should remain the controller.

---

## 15. One-Sentence Summary

Support decides **where** to edit, but the main research contribution is the controller deciding **how strongly to edit and preserve** at each ODE step using clean-estimate-space diagnostics.
