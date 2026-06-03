# RF h-Edit: Detailed Next-Step Plan for Controller Validation and Improvement

## 0. Current Status

The current project status can be summarized as:

> **support-v3 is usable as a spatial interface, and the main controller has been verified to run. However, the controller currently provides only small but stable gains, so the next step is to make the controller evidence stronger and the control signal more principled.**

The current main variants are:

| Method | Meaning | Current status |
|---|---|---|
| `support_v3_fixed` | support-v3 + fixed edit / preserve control | strong baseline |
| `support_v3_controller_rmsgap` | support-v3 + RMS-gap adaptive controller | current main candidate |
| `support_v3_controller_progress` | support-v3 + directional-progress controller | theoretically attractive but too aggressive |
| `support_v3_controller_hybrid` | RMS-gap + gated progress controller | works, but not stable enough to replace rmsgap |

The current seed10–12 multi-seed results suggest:

- `rmsgap` gives small CLIP improvements.
- Outside-mask preservation is mostly unchanged.
- SSIM is mostly stable.
- `progress` is unstable and too aggressive.
- `hybrid` has potential but still needs better gating.
- The main current claim should remain conservative.

Current conservative claim:

> Under a fixed support-v3 spatial interface, the clean-estimate-space RMS-gap adaptive controller provides small but consistent edit-alignment gains without noticeably increasing preservation drift.

---

## 1. What We Should Not Do Next

Do **not** immediately:

1. change support-v3 again;
2. add more guidance branches;
3. use `progress` as the main result;
4. include `backpack_remove_toy_charm` as a main success case;
5. claim that the full local editing problem is solved;
6. claim that the controller significantly improves preservation unless the numbers prove it.

Reason:

The current next step is not to expand the method.  
The next step is to **validate and strengthen the controller claim**.

---

## 2. Main Research Question for the Next Stage

The next-stage research question should be:

\[
\boxed{
\text{Does clean-estimate-space adaptive control improve the edit-preserve trade-off under fixed support?}
}
\]

In other words, under the same support mask:

\[
M_{\mathrm{edit}},\quad M_{\mathrm{preserve}},
\]

does the adaptive controller improve:

- edit alignment,
- while keeping preserve drift stable?

This should be tested independently of support changes.

---

## 3. Core Mathematical Setup

The project still uses the linear Rectified Flow path:

\[
x_t = (1-t)x_0 + t x_1.
\]

The RF velocity is:

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

The clean-space correction to velocity correction is:

\[
\boxed{
u
=
-\frac{\Delta x_0}{t}
}
\]

The implementation-level dynamics are:

\[
\dot{x}_t
=
\underbrace{
(v_{\mathrm{src}} + u_{\mathrm{rec}})
}_{\text{reconstruction-aware base field}}
+
\underbrace{
u_{\mathrm{edit}}
}_{\text{editing field}}.
\]

The main contribution is not the decomposition alone.  
The main contribution should be the **closed-loop adaptive control** over these velocity terms.

---

## 4. Current Controller Variants

### 4.1 Fixed Control

The fixed baseline uses fixed edit / preserve weights:

\[
\lambda_{\mathrm{edit}}(t)=\lambda_{\mathrm{edit}}^0,
\]

\[
\lambda_{\mathrm{pres}}(t)=\lambda_{\mathrm{pres}}^0.
\]

This is the baseline:

```text
support_v3_fixed
```

---

### 4.2 RMS-Gap Controller

The current main controller is:

```text
support_v3_controller_rmsgap
```

It uses an RMS-style edit gap to decide whether more edit force is needed.

A general form is:

\[
g_{\mathrm{rms}}(t)
=
\left\|
M_{\mathrm{edit}}
\odot
(\Delta_{\mathrm{tar}}-\Delta_{\mathrm{cur}})
\right\|.
\]

where:

\[
\Delta_{\mathrm{cur}}
=
\hat{x}_{0,t}-x_0^{ref},
\]

\[
\Delta_{\mathrm{tar}}
=
\hat{x}_{0,t}^{tar}-x_0^{ref}.
\]

The adaptive edit strength can be written as:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_{\mathrm{edit}}^0
\left[
1+
k_g\,g_{\mathrm{rms}}(t)
\right].
\]

Current observation:

- stable;
- conservative;
- small semantic gain;
- does not significantly worsen outside-mask preservation.

---

### 4.3 Directional-Progress Controller

The directional-progress controller uses:

\[
p_{\mathrm{edit}}(t)
=
\frac{
\langle \Delta_{\mathrm{cur}},\Delta_{\mathrm{tar}}\rangle
}{
\|\Delta_{\mathrm{tar}}\|^2+\epsilon
}.
\]

This is theoretically more meaningful because it measures whether the current change moves in the target direction.

However, current results show it is too aggressive.

Issues:

- target direction can be noisy for small local objects;
- decal tasks have weak target deltas;
- early boost can over-edit;
- SSIM can drop;
- inside-mask damage can increase.

Therefore, it should not be the main method yet.

---

### 4.4 Hybrid Controller

The current hybrid combines RMS-gap and directional progress.

Current observation:

- it works;
- it improves cat/dog CLIP slightly;
- but it can hurt mug preservation;
- gate design is not mature.

Therefore, hybrid should remain a future branch, not the current main result.

---

## 5. Immediate Goal

The immediate goal is:

\[
\boxed{
\text{Make rmsgap controller evidence stronger}
}
\]

This means:

1. organize the multi-seed results;
2. produce per-run deltas;
3. produce trade-off plots;
4. produce controller behavior curves;
5. normalize and improve the RMS-gap signal;
6. later design a safer hybrid controller.

---

# Part I: Analysis and Reporting

## 6. Step 1 — Multi-Seed Summary Table

### 6.1 Tasks

Use:

```text
cat_crown
dog_sunglasses
mug_heart
```

Do not include `backpack_remove_toy_charm` in the main controller table yet.

Reason:

`backpack_remove_toy_charm` is currently a removal-dynamics failure case, not a clean controller validation case.

---

### 6.2 Methods

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
support_v3_controller_progress
```

Optionally include:

```text
support_v3_controller_hybrid
```

but only as an additional branch.

---

### 6.3 Seeds

Use:

```text
seed 10
seed 11
seed 12
```

---

### 6.4 Metrics

For each task / method / seed, save:

```text
CLIP_delta
inside_mask_L1
outside_mask_L1
source_SSIM
```

Optional:

```text
outside_LPIPS
DINO_distance
runtime
NFE
mask_area
```

---

### 6.5 Output Files

Create:

```text
experiments/controller_validation_YYYY-MM-DD/
  controller_mean_std.csv
  controller_per_run.csv
  controller_per_run_delta.csv
  controller_success_rate.csv
```

---

## 7. Step 2 — Per-Run Delta Analysis

Averaged bar charts are not enough.

Compute per-run deltas relative to fixed control.

For each task and seed:

\[
\Delta_{\mathrm{CLIP}}^{rmsgap}
=
\mathrm{CLIP}_{rmsgap}
-
\mathrm{CLIP}_{fixed}
\]

\[
\Delta_{\mathrm{outside}}^{rmsgap}
=
\mathrm{OutsideL1}_{rmsgap}
-
\mathrm{OutsideL1}_{fixed}
\]

\[
\Delta_{\mathrm{SSIM}}^{rmsgap}
=
\mathrm{SSIM}_{rmsgap}
-
\mathrm{SSIM}_{fixed}
\]

---

### 7.1 Success Definition

Define a case as successful if:

\[
\Delta_{\mathrm{CLIP}} > 0
\]

and:

\[
\Delta_{\mathrm{outside}} < \epsilon_{\mathrm{outside}}
\]

where a suggested tolerance is:

```text
epsilon_outside = 0.001 or 0.002
```

A stricter success criterion:

\[
\Delta_{\mathrm{CLIP}} > 0
\]

\[
\Delta_{\mathrm{outside}} \le 0
\]

\[
\Delta_{\mathrm{SSIM}} \ge -0.002.
\]

---

### 7.2 Success Rate

Report:

```text
CLIP improved in X / 9 cases
Preservation not worse in Y / 9 cases
Both satisfied in Z / 9 cases
```

This may be more convincing than only reporting tiny average improvements.

---

## 8. Step 3 — Trade-Off Plot

The main claim is edit-preserve trade-off.

Plot:

\[
\mathrm{CLIP}\Delta
\quad \text{vs.} \quad
\mathrm{outside\text{-}mask\ L1}.
\]

Each point represents one task/seed.

Compare:

```text
fixed
rmsgap
progress
hybrid
```

Expected useful pattern:

- rmsgap points shift slightly upward in CLIP without shifting right in outside L1;
- progress may shift up or down but often worsens preservation;
- hybrid should be diagnosed separately.

---

### 8.1 Plot Files

Save:

```text
tradeoff_all_tasks.png
tradeoff_cat_crown.png
tradeoff_dog_sunglasses.png
tradeoff_mug_heart.png
```

---

## 9. Step 4 — Controller Behavior Curves

Since the method is a controller, final images alone are not enough.

For representative cases, save curves over ODE steps:

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

Representative cases:

```text
cat_crown seed10
dog_sunglasses seed10
mug_heart seed10
```

---

### 9.1 What Curves Should Show

The curves should demonstrate that:

1. edit weight increases when edit gap is high;
2. preserve weight increases when preserve drift is high;
3. projection activates when edit velocity damages preserve region;
4. the controller is not a constant scale trick.

---

### 9.2 Output Files

Save:

```text
curves_cat_crown_seed10.png
curves_dog_sunglasses_seed10.png
curves_mug_heart_seed10.png
```

and raw CSV files:

```text
curves_cat_crown_seed10.csv
curves_dog_sunglasses_seed10.csv
curves_mug_heart_seed10.csv
```

---

# Part II: Strengthening the RMS-Gap Controller

## 10. Step 5 — Normalize RMS Gap

Current RMS gap may be scale-dependent across tasks.

Use a normalized RMS gap:

\[
\boxed{
g_{\mathrm{rms}}(t)
=
\frac{
\left\|
M_{\mathrm{edit}}
\odot
(\Delta_{\mathrm{tar}}-\Delta_{\mathrm{cur}})
\right\|
}{
\left\|
M_{\mathrm{edit}}
\odot
\Delta_{\mathrm{tar}}
\right\|+\epsilon
}
}
\]

where:

\[
\Delta_{\mathrm{cur}}
=
\hat{x}_{0,t}-x_0^{ref}
\]

\[
\Delta_{\mathrm{tar}}
=
\hat{x}_{0,t}^{tar}-x_0^{ref}.
\]

This makes the gap more comparable across tasks.

---

## 11. Step 6 — Add Dead Zone

Do not boost edit when the gap is already small.

Define:

\[
g_{\mathrm{active}}(t)
=
\max(0,g_{\mathrm{rms}}(t)-\tau_g).
\]

Then:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_0
\left[
1+
k_g g_{\mathrm{active}}(t)
\right].
\]

Suggested initial values:

```text
tau_g = 0.10 to 0.20
k_g = 0.5 to 1.5
```

The exact range should be tuned lightly.

---

## 12. Step 7 — Add Preserve Gate

If preserve drift is already high, do not keep increasing edit weight.

Define preserve drift:

\[
d_{\mathrm{pres}}(t)
=
\frac{
\left\|
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_0^{ref})
\right\|
}{
\left\|
M_{\mathrm{preserve}}
\odot x_0^{ref}
\right\|+\epsilon
}.
\]

Define gate:

\[
G_{\mathrm{pres}}(t)
=
\mathbf{1}[d_{\mathrm{pres}}(t)<\tau_{\mathrm{pres}}].
\]

Then:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_0
\left[
1+
G_{\mathrm{pres}}(t)\,
k_g\max(0,g_{\mathrm{rms}}(t)-\tau_g)
\right].
\]

This is especially important for `mug_heart`.

---

## 13. Step 8 — Cap Adaptive Weights

Use caps:

\[
\lambda_{\mathrm{edit}}(t)
\leq
\lambda_{\mathrm{edit}}^{max}
\]

\[
\lambda_{\mathrm{pres}}(t)
\leq
\lambda_{\mathrm{pres}}^{max}.
\]

Suggested first values:

```text
lambda_edit_max = 1.25 to 1.5 times base
lambda_pres_max = 1.5 to 2.0 times base
```

Avoid overly large gains.

---

## 14. Step 9 — Task-Type Preserve Defaults

Different edit operations need different preserve strength.

| Task type | Preserve strategy |
|---|---|
| accessory insertion | medium preserve |
| decal / small pattern | strong preserve |
| removal | separate removal controller |

For `mug_heart`, preserve should be stronger than for `cat_crown` or `dog_sunglasses`.

---

## 15. Step 10 — Mug Heart Special Target

`mug_heart` is currently the best task for validating preserve control.

Current target:

\[
\Delta_{\mathrm{CLIP}} > 0
\]

and:

\[
\mathrm{outside\text{-}mask\ L1}<0.015.
\]

The goal is:

- keep heart visible;
- reduce mug/background drift;
- improve SSIM;
- avoid gray global shift.

---

### 15.1 Strict Edit Masking for Decal

For decal:

\[
u_{\mathrm{edit}}^{masked}
=
M_{\mathrm{core}}\odot u_{\mathrm{edit}}
+
\lambda_{\mathrm{ring}}M_{\mathrm{ring}}\odot u_{\mathrm{edit}}.
\]

Suggested:

```text
lambda_ring = 0.2 to 0.4
```

In preserve region:

\[
M_{\mathrm{preserve}}\odot u_{\mathrm{edit}}
=
0.
\]

---

### 15.2 Preserve Correction Only in Preserve Region

For decal, preserve correction should not suppress the heart region.

Use:

\[
u_{\mathrm{pres}}
=
M_{\mathrm{preserve}}
\odot
u_{\mathrm{rec}}
+
\lambda_{\mathrm{ring}}^{rec}
M_{\mathrm{ring}}
\odot
u_{\mathrm{rec}}.
\]

Do not apply strong preserve inside:

\[
M_{\mathrm{core}}.
\]

---

### 15.3 Clean-Effect Projection

For edit velocity:

\[
u_{\mathrm{edit}},
\]

its clean-effect is:

\[
\Delta \hat{x}_{0}^{edit}
=
-t\,u_{\mathrm{edit}}.
\]

If:

\[
\left\langle
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_0^{ref}),
\;
M_{\mathrm{preserve}}
\odot
\Delta \hat{x}_{0}^{edit}
\right\rangle
>0,
\]

then suppress or project the preserve-region effect.

---

# Part III: Robustness Experiments

## 16. Step 11 — Edit Strength Sweep

To show controller value more clearly, test robustness across edit strength.

Edit scales:

```text
0.5
0.75
1.0
1.25
1.5
```

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
```

Tasks:

```text
cat_crown
mug_heart
```

Optional:

```text
dog_sunglasses
```

Plot:

\[
\mathrm{CLIP}\Delta
\quad \text{vs.} \quad
\mathrm{outside\text{-}mask\ L1}
\]

Desired result:

- rmsgap gives a better trade-off curve;
- fixed becomes unstable at high edit scale;
- rmsgap keeps outside drift controlled.

---

## 17. Step 12 — Support Perturbation Sweep

Test whether controller is robust to imperfect support.

Perturbations:

```text
erode
dilate
shift
holes
boundary_noise
```

Compare:

```text
support_v3_fixed
support_v3_controller_rmsgap
```

Tasks:

```text
dog_sunglasses
mug_heart
```

Desired result:

- rmsgap is at least as stable as fixed;
- ideally, rmsgap compensates mild mask errors;
- if not, report that support quality remains critical.

---

# Part IV: Future Hybrid Controller

## 18. Step 13 — Safe Hybrid Controller

Only after the rmsgap controller is validated, develop a safer hybrid.

Use:

\[
\lambda_{\mathrm{edit}}(t)
=
\lambda_0
\left[
1
+
k_r g_{\mathrm{rms}}(t)
+
k_p G(t)\max(0,\tau_p-\bar{p}_{\mathrm{edit}}(t))
\right]
\]

where:

\[
G(t)
=
\mathbf{1}[d_{\mathrm{pres}}(t)<\tau_{\mathrm{pres}}]
\cdot
\mathbf{1}[\|\Delta_{\mathrm{tar}}\|>\epsilon]
\cdot
\mathbf{1}[c_{\mathrm{support}}>\tau_{\mathrm{sup}}].
\]

This means progress boost is only enabled when:

1. preserve drift is under control;
2. target direction is reliable;
3. support is confident.

---

## 19. Step 14 — EMA Directional Progress

Use EMA smoothing:

\[
\bar{p}_{\mathrm{edit}}(t)
=
\rho \bar{p}_{\mathrm{edit}}(t-1)
+
(1-\rho)p_{\mathrm{edit}}(t).
\]

Suggested:

```text
rho = 0.8
```

This reduces noise in small-object and decal tasks.

---

## 20. Hybrid Is Not Main Result Yet

Do not replace rmsgap with hybrid until:

- hybrid improves CLIP without hurting outside L1;
- hybrid improves mug preservation;
- hybrid works across seeds;
- hybrid does not degrade dog_sunglasses.

For now:

```text
rmsgap = main candidate
hybrid = experimental branch
```

---

# Part V: Removal Task

## 21. Treatment of Backpack Remove

Do not include:

```text
backpack_remove_toy_charm
```

in the main controller claim yet.

Reason:

It is not mainly a controller problem.  
It needs removal-specific dynamics.

Current conclusion:

> support can be correct, but ordinary target-edit velocity cannot reliably remove and fill the object.

Use as:

```text
limitation
failure analysis
future work
```

---

## 22. Future Removal Controller

Possible future formulation:

No-object clean estimate:

\[
\hat{x}_{0}^{noobj}
=
x_t - t v_{\theta}^{noobj}(x_t,t).
\]

Fill displacement:

\[
\Delta x_0^{fill}
=
M_{\mathrm{remove}}
\odot
(\hat{x}_{0}^{noobj}-\hat{x}_{0}^{cur}).
\]

Velocity correction:

\[
u_{\mathrm{fill}}
=
-\frac{\Delta x_0^{fill}}{t}.
\]

Removal dynamics:

\[
u_{\mathrm{remove}}
=
u_{\mathrm{fill}}
+
u_{\mathrm{sup}}
+
u_{\mathrm{ring-rec}}.
\]

This should be a separate module.

---

# Part VI: Documentation and Reporting

## 23. Current Stage Claim

The current stage claim should be:

> Under a fixed support-v3 spatial interface, RMS-gap based clean-estimate-space adaptive control gives small but consistent semantic gains while preserving non-edit regions at a comparable level to fixed control.

Do not overclaim.

---

## 24. Stronger Claim Needed Later

To make a stronger claim, we need:

1. multi-seed consistency;
2. edit-preserve trade-off curves;
3. robustness to edit strength;
4. robustness to support perturbation;
5. controller behavior curves;
6. visual evidence;
7. comparison with external baselines later.

---

## 25. Immediate To-Do Checklist

### Analysis

- [ ] Generate `per_run_delta.csv`.
- [ ] Generate `success_rate.csv`.
- [ ] Generate mean/std tables.
- [ ] Generate trade-off plots.
- [ ] Generate controller behavior curves.

### Controller improvement

- [ ] Implement normalized RMS gap.
- [ ] Add RMS dead zone.
- [ ] Add preserve gate.
- [ ] Add adaptive weight caps.
- [ ] Add task-type preserve defaults.
- [ ] Strengthen mug_heart preserve control.

### Robustness

- [ ] Run edit-strength sweep.
- [ ] Run support perturbation sweep.
- [ ] Plot robustness curves.

### Hybrid

- [ ] Add gated directional-progress term.
- [ ] Add EMA progress smoothing.
- [ ] Keep hybrid as branch, not main method.

### Reporting

- [ ] Keep `rmsgap` as current main result.
- [ ] Mark `progress` as unstable.
- [ ] Mark `hybrid` as promising but immature.
- [ ] Treat `backpack_remove` as limitation.

---

## 26. One-Sentence Summary

The next step is not to change support or add new guidance, but to validate and strengthen the controller: first prove that the RMS-gap controller gives stable multi-seed edit-preserve gains, then normalize and gate the RMS signal, and only afterward introduce a safer hybrid controller with gated directional progress.
