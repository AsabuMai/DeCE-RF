# RF h-Edit: Next-Step Controller Validation Plan

## 0. Current Situation

The current state can be summarized as:

> support-v3 is no longer the main bottleneck for local insertion / decal tasks.  
> The main controller is running, but currently it only gives small and stable gains, not yet strong enough to support a major paper claim.

The current main methods are:

| Method | Meaning |
|---|---|
| `support_v3_fixed` | support-v3 + fixed control |
| `support_v3_controller_rmsgap` | support-v3 + RMS-gap adaptive controller, current main candidate |
| `support_v3_controller_progress` | support-v3 + directional-progress controller, currently too aggressive |

The current controller already includes:

- clean-estimate-space projection written back into `v_edit_total`
- adaptive preserve lock affecting `v_rec`
- trajectory preserve scaled by preserve weight
- direct preserve clean correction
- corrected evaluation mask, using `operation_v3_edit_mask.png`

---

## 1. Latest Seed10 Result Summary

Current seed10 results for three main tasks:

| Task | fixed CLIPΔ | rmsgap CLIPΔ | fixed outside | rmsgap outside | fixed SSIM | rmsgap SSIM |
|---|---:|---:|---:|---:|---:|---:|
| `cat_crown` | 0.0920 | 0.0978 | 0.0538 | 0.0540 | 0.8714 | 0.8738 |
| `dog_sunglasses` | 0.0983 | 0.1007 | 0.0458 | 0.0458 | 0.9038 | 0.9034 |
| `mug_heart` | 0.0422 | 0.0432 | 0.0078 | 0.0077 | 0.8646 | 0.8684 |

Interpretation:

- semantic alignment improves slightly;
- preservation is mostly unchanged;
- `mug_heart` is no longer a clear failure;
- background destruction is not obviously worse;
- however, the improvement is still small.

Therefore, the current claim should be conservative:

> Under a fixed support-v3 spatial interface, the clean-estimate-space adaptive controller provides small but stable edit-alignment gains without noticeably increasing preservation drift.

---

## 2. What Should Not Be Claimed Yet

Do **not** claim:

- support-v3 is the main contribution;
- the controller significantly lowers background drift;
- the directional progress controller is solved;
- the method comprehensively outperforms fixed baselines;
- the full local editing problem is solved.

The current evidence only supports:

> The controller shows weak but consistent improvement on seed10, and needs multi-seed validation.

---

## 3. Main Next Step

The next step should be:

\[
\boxed{
\text{freeze support-v3 and validate controller variants}
}
\]

Do not keep changing support while testing the controller.

Reason:

If support keeps changing, we cannot tell whether the improvement comes from:

- better support,
- better adaptive controller,
- different masks,
- or different editing dynamics.

So the next experiment should fix:

\[
M_{\mathrm{edit}},\quad M_{\mathrm{preserve}}
\]

and compare controller variants under the same spatial interface.

---

## 4. Main Experiment: Multi-Seed Controller Validation

### Tasks

Use the three tasks where support-v3 is currently usable:

```text
cat_crown
dog_sunglasses
mug_heart
```

Do not include `backpack_remove_toy_charm` in the main table yet.  
It is currently a removal-dynamics limitation case, not a controller validation case.

### Methods

Run:

```text
support_v3_fixed
support_v3_controller_rmsgap
support_v3_controller_progress
```

### Seeds

Run:

```text
seed 10
seed 11
seed 12
```

Total:

\[
3\text{ tasks} \times 3\text{ methods} \times 3\text{ seeds}
=
27\text{ runs}
\]

### Metrics

Save mean and standard deviation for:

```text
CLIPΔ
inside-mask L1
outside-mask L1
SSIM
```

Optional:

```text
LPIPS outside mask
DINO feature distance
mask area
runtime / NFE
```

---

## 5. Current Main Candidate

The current main candidate should be:

```text
support_v3_controller_rmsgap
```

Reason:

- it gives small CLIPΔ improvement;
- it does not increase outside-mask L1 significantly;
- it keeps SSIM roughly stable;
- it is less aggressive than directional-progress controller.

The `progress` version should not be the main result yet.

Current issue with `progress`:

- it boosts edit weight too early;
- it can reduce CLIPΔ for cat/dog;
- it can lower SSIM;
- it is too sensitive to progress target and gain.

---

## 6. Main Claim to Test

The main claim to test is:

> Under the same support-v3 spatial interface, the RMS-gap adaptive controller improves edit target alignment while preserving non-edit regions at a level comparable to fixed control.

In terms of metrics:

\[
\Delta \mathrm{CLIP}_{rmsgap}
>
\Delta \mathrm{CLIP}_{fixed}
\]

while:

\[
\mathrm{OutsideL1}_{rmsgap}
\approx
\mathrm{OutsideL1}_{fixed}
\]

and:

\[
\mathrm{SSIM}_{rmsgap}
\geq
\mathrm{SSIM}_{fixed}
\quad \text{or at least does not drop significantly.}
\]

---

## 7. Need Stronger Evidence Than Single-Point Metrics

The current CLIPΔ improvements are small.

Therefore, a single metric table is not enough.

The controller contribution should be validated through:

1. multi-seed stability,
2. trade-off curves,
3. controller behavior curves,
4. robustness tests.

---

# 8. Controller Behavior Curves

The contribution is a controller, so we need to show that the controller actually reacts to clean-estimate-space signals.

For at least seed10, save and plot:

```text
edit_rms_gap_curve
directional_progress_curve
preserve_drift_curve
adaptive_edit_weight_curve
adaptive_preserve_weight_curve
projection_ratio_curve
edit_velocity_norm_curve
rec_velocity_norm_curve
```

These curves should show:

- edit boost increases when edit gap is high;
- preserve lock increases when preserve drift is high;
- projection activates when edit velocity threatens preserve regions;
- the controller is not just a constant rescaling.

Without these curves, the closed-loop control claim is weak.

---

## 9. Robustness Experiment 1: Edit-Strength Sweep

The current improvement may be too small at one default parameter point.

A better test is whether the controller improves the trade-off curve.

### Setup

Fix support-v3 and sweep edit strength:

```text
edit scale = 0.5
edit scale = 0.75
edit scale = 1.0
edit scale = 1.25
edit scale = 1.5
```

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
```

### Plot

Plot:

\[
\mathrm{CLIP}\Delta
\quad \text{vs.} \quad
\mathrm{outside\text{-}mask\ L1}
\]

### Desired result

The controller should provide a better trade-off:

- higher CLIPΔ at the same outside-mask L1;
- or lower outside-mask L1 at the same CLIPΔ;
- or a more stable curve across edit scales.

This is stronger evidence than a small gain at one default setting.

---

## 10. Robustness Experiment 2: Support Perturbation

Real support masks are imperfect.

Test whether the controller can compensate for imperfect support.

### Perturbations

Apply small perturbations to support-v3 masks:

```text
erode mask
dilate mask
shift mask slightly
add small holes
add boundary noise
```

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
```

### Desired result

The controller should be more robust than fixed control when support is slightly imperfect.

This would support the claim:

> The controller can stabilize editing under imperfect spatial support.

---

# 11. Hybrid Controller Direction

The current directional-progress controller is too aggressive.

Instead of using it directly, develop a safer hybrid controller:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_0
\left[
1
+
k_r \cdot \mathrm{RMSGap}(t)
+
k_p \cdot G(t)\cdot
\max(0,\tau_p-p_{\mathrm{edit}}(t))
\right]
\]

where the gate is:

\[
G(t)
=
\mathbf{1}[d_{\mathrm{pres}}(t)<\tau_{\mathrm{pres}}]
\cdot
\mathbf{1}[\|\Delta_{\mathrm{tar}}\|>\epsilon].
\]

Meaning:

- use RMS-gap as the stable main controller;
- use directional progress only when target direction is reliable;
- disable progress boost if preserve drift is already high;
- do not boost edit if target delta is too weak or noisy.

---

## 12. EMA Smoothing for Directional Progress

Directional progress can be noisy.

Use EMA:

\[
\bar{p}_{\mathrm{edit}}(t)
=
\rho \bar{p}_{\mathrm{edit}}(t-1)
+
(1-\rho)p_{\mathrm{edit}}(t).
\]

Suggested:

```text
rho = 0.7 to 0.9
```

Use:

\[
\bar{p}_{\mathrm{edit}}(t)
\]

instead of raw \(p_{\mathrm{edit}}(t)\).

This may make the progress controller less aggressive.

---

## 13. Preserve Mechanism Refinement

Direct preserve clean correction works, but it is sensitive to scale.

Observed:

```text
scale 1.0 = too strong
scale 0.5 = more stable
```

So use a capped adaptive preserve schedule:

\[
\lambda_{\mathrm{pres}}(t)
=
\lambda_{\mathrm{pres}}^0
+
k_{\mathrm{pres}}\max(0,d_{\mathrm{pres}}(t)-\tau_{\mathrm{pres}})
\]

with:

\[
\lambda_{\mathrm{pres}}(t)
\leq
\lambda_{\mathrm{pres}}^{max}.
\]

Task-type defaults:

| Task type | Preserve strength |
|---|---|
| accessory insertion | medium |
| decal | strong |
| removal | separate controller |

For `mug_heart`, preserve should be stronger because only a small decal should change.

---

# 14. Why `mug_heart` Is Important

`mug_heart` is currently the best controller validation case.

Reason:

- support is already usable;
- edit succeeds somewhat;
- preserve drift remains the main issue.

Current target:

\[
\Delta_{\mathrm{CLIP}}>0
\]

and:

\[
\mathrm{outside\text{-}mask\ L1}<0.015.
\]

If the controller can keep CLIPΔ positive while reducing outside-mask L1, this would strongly support the controller claim.

---

# 15. Treatment of `backpack_remove_toy_charm`

Do not include `backpack_remove_toy_charm` in the main controller result yet.

Current issue:

> Support is reasonably good, but ordinary target-edit velocity cannot remove and fill the object.

This is a separate removal-dynamics problem.

Use it as:

```text
limitation
failure analysis
future removal controller motivation
```

Possible future removal dynamics:

\[
\hat{x}_{0}^{noobj}
=
x_t - t v_{\theta}^{noobj}(x_t,t)
\]

\[
\Delta x_0^{fill}
=
M_{\mathrm{remove}}
\odot
(\hat{x}_{0}^{noobj}-\hat{x}_{0}^{cur})
\]

\[
u_{\mathrm{fill}}
=
-\frac{\Delta x_0^{fill}}{t}.
\]

But this is not the current mainline.

---

# 16. Immediate Execution Order

## Step 1: Multi-Seed Validation

Run:

```text
cat_crown
dog_sunglasses
mug_heart
```

with:

```text
support_v3_fixed
support_v3_controller_rmsgap
support_v3_controller_progress
```

for:

```text
seed 10
seed 11
seed 12
```

Save mean and standard deviation.

---

## Step 2: Controller Curves

For at least seed10, plot:

```text
edit_rms_gap_curve
directional_progress_curve
preserve_drift_curve
adaptive_edit_weight_curve
adaptive_preserve_weight_curve
projection_ratio_curve
```

---

## Step 3: Edit-Strength Sweep

For at least two tasks:

```text
cat_crown
mug_heart
```

run:

```text
edit scale = 0.5, 0.75, 1.0, 1.25, 1.5
```

Compare:

```text
fixed
rmsgap
```

Plot:

```text
CLIPΔ vs outside-mask L1
```

---

## Step 4: Support Perturbation Sweep

For at least two tasks:

```text
dog_sunglasses
mug_heart
```

test:

```text
erode
dilate
shift
holes
boundary noise
```

Compare:

```text
fixed
rmsgap
```

---

## Step 5: Hybrid Controller

Only after Steps 1–4, implement:

```text
support_v3_controller_hybrid
```

with:

- RMS-gap base control;
- gated directional-progress boost;
- EMA smoothing;
- preserve-drift gate.

---

# 17. Current Conservative Claim

If rmsgap remains stable across seeds, the current conservative claim is:

> Under a fixed support-v3 spatial interface, clean-estimate-space adaptive control gives small but consistent semantic gains without noticeably increasing preserve-region drift.

This is a weak but clean claim.

To make it stronger, we need:

- trade-off curves;
- robustness tests;
- controller behavior curves.

---

# 18. Final One-Sentence Summary

The next step is to freeze support-v3 and validate the controller itself: first show that `rmsgap` gives stable multi-seed gains, then prove its value through edit-strength and support-perturbation robustness tests, and only afterward develop a safer hybrid directional-progress controller.
