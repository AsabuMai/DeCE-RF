# Codex Task List for `rf_h_edit_project`

## Goal

Clean up the current RF / SD3 h-Edit prototype so that it becomes a reliable method-validation codebase for the following research claim:

> A reconstruction/editing-decoupled velocity formulation can improve the trade-off between editing effectiveness and faithfulness in ODE / Rectified Flow image editing.

The current code already contains the main ingredients:

- SD3 / Rectified Flow sampling loop
- source inversion
- linear-path clean estimate
- reconstruction-side surrogate terms
- editing-side surrogate terms
- direct target-vs-source velocity difference
- CLIP / text / DDS / attention-based guidance branches

The next step is **not** to add more guidance terms. The priority is to make the existing dynamics mathematically consistent, debuggable, and experimentally comparable.

---

## Core Mathematical Convention

Use the linear Rectified Flow path convention:

\[
x_t = (1-t)x_0 + t x_1,
\qquad
\hat{x}_0(x_t,t)=x_t-t v_\theta(x_t,t).
\]

If we add a velocity correction \(u\), then

\[
\hat{x}_0' = x_t - t(v_\theta+u)
= \hat{x}_0 - t u.
\]

Therefore, if we want the clean estimate to move by a desired clean-space displacement

\[
\Delta \hat{x}_0,
\]

the corresponding velocity correction should be

\[
u = -\frac{\Delta \hat{x}_0}{t}.
\]

This conversion is the most important thing to verify in the current implementation.

---

# P0: Fix / Verify Clean-Delta-to-Velocity Conversion

## Problem

Several current surrogate terms appear to convert clean-space deltas into velocity corrections using `delta / t`. Under the current clean estimate

```python
x0_hat = x_t - t * v
```

a desired clean-space displacement `delta_x0` should map to

```python
u = -delta_x0 / t
```

because

```python
x0_hat_new = x0_hat - t * u
```

and therefore

```python
x0_hat_new - x0_hat = delta_x0
```

requires

```python
u = -delta_x0 / t
```

## Required Code Change

Add a helper in `energies.py`:

```python
def expand_t_like(t_scalar: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=target.device, dtype=target.dtype)
    t_scalar = t_scalar.to(device=target.device, dtype=target.dtype).clamp_min(eps)
    while t_scalar.ndim < target.ndim:
        t_scalar = t_scalar.view(*t_scalar.shape, 1)
    return t_scalar


def clean_delta_to_velocity(delta_x0: torch.Tensor, t_scalar: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Convert a desired clean-estimate displacement into a velocity correction.

    RF linear path convention:
        x0_hat = x_t - t * v

    If v' = v + u, then:
        x0_hat' - x0_hat = -t * u

    Therefore:
        u = -delta_x0 / t
    """
    t = expand_t_like(t_scalar, delta_x0, eps=eps)
    return -delta_x0 / t
```

## Apply This Helper

### 1. Editing anchor correction

Current intention:

```python
delta_x0 = x0_tar - x0_src
```

Desired behavior:

```python
u_anchor = clean_delta_to_velocity(delta_x0, t_scalar)
```

So in `editing_velocity_surrogate_target_anchor()`, replace the raw `delta / t_scalar` conversion with the helper above.

### 2. Region delta correction

Current intention:

```python
delta_x0 = x0_tar - x0_src
```

Desired behavior:

```python
u_region = clean_delta_to_velocity(delta_x0, t_scalar)
```

Apply the same conversion in `editing_velocity_surrogate_region_delta()`.

### 3. Reconstruction correction

If reconstruction wants the clean estimate to move toward the source image,

```python
delta_x0 = x_src - x0_pred
```

then the corresponding velocity correction should be:

```python
u_rec = clean_delta_to_velocity(delta_x0, t_scalar)
```

This means the velocity correction is proportional to:

```python
(x0_pred - x_src) / t
```

not simply:

```python
-(x0_pred - x_src)
```

Please add a new reconstruction surrogate mode rather than deleting the old one immediately.

Suggested option name:

```python
velocity_conversion_mode: Literal["legacy", "linear_path"] = "linear_path"
```

Use `legacy` only for comparison.

---

# P0: Add Unit Tests for Velocity Conversion

Create a small test file, for example:

```text
tests/test_linear_path_velocity_conversion.py
```

The test does not need SD3. Use toy tensors.

## Test 1: clean delta conversion

```python
x_t = torch.randn(2, 4, 8, 8)
v = torch.randn_like(x_t)
t = torch.tensor(0.5)
x0_hat = x_t - t * v

delta_x0 = torch.randn_like(x_t) * 0.1
u = clean_delta_to_velocity(delta_x0, t)

x0_hat_new = x_t - t * (v + u)
assert torch.allclose(x0_hat_new - x0_hat, delta_x0, atol=1e-5)
```

## Test 2: reconstruction target

```python
x_src = torch.randn_like(x_t)
delta_x0 = x_src - x0_hat
u = clean_delta_to_velocity(delta_x0, t)
x0_hat_new = x_t - t * (v + u)
assert torch.allclose(x0_hat_new, x_src, atol=1e-5)
```

## Test 3: target anchor

```python
x0_src = torch.randn_like(x_t)
x0_tar = torch.randn_like(x_t)
delta_x0 = x0_tar - x0_src
u = clean_delta_to_velocity(delta_x0, t)
x0_new = x0_src - t * u
assert torch.allclose(x0_new, x0_tar, atol=1e-5)
```

Note: These tests validate the linear-path conversion only. They do not validate image quality.

---

# P0: Fix Beta / Branch-Scale Double Multiplication

## Problem

In `sd3_hrec.py`, if `beta_max is None`, it is currently set to the maximum of multiple branch scales:

```python
beta_max = max(
    edit_hedit_guidance_scale,
    edit_guidance_scale,
    edit_region_guidance_scale,
    edit_target_guidance_scale,
    edit_source_guidance_scale,
    edit_clip_guidance_scale,
    edit_text_guidance_scale,
    edit_dds_guidance_scale,
    edit_app_guidance_scale,
)
```

But branch scales are also applied inside the individual edit terms. This can unintentionally produce scale multiplication such as:

```text
branch_scale * beta_t
```

where `beta_t` itself already came from the branch scale.

## Required Code Change

Use a clean convention:

### Recommended convention

- `beta_t` is the **global time schedule**.
- branch-specific `edit_*_guidance_scale` values are **relative branch weights**.

Therefore, if `beta_max is None`, use:

```python
beta_max = 1.0
```

Do not infer `beta_max` from the maximum branch scale.

## Logging

Add these fields to `step_stat`:

```python
"beta_max": float(beta_max),
"beta_t": float(beta_t),
"edit_hedit_guidance_scale": float(edit_hedit_guidance_scale),
"edit_guidance_scale": float(edit_guidance_scale),
"edit_region_guidance_scale": float(edit_region_guidance_scale),
"edit_clip_guidance_scale": float(edit_clip_guidance_scale),
"edit_text_guidance_scale": float(edit_text_guidance_scale),
```

The goal is to make effective guidance strength inspectable.

---

# P1: Clarify Surrogate Velocity vs True Energy Gradient

## Problem

The current code has both:

```python
editing_energy_total(...)
editing_velocity_surrogate_total(...)
```

but the actual ODE uses mostly hand-designed surrogate velocities, not exact gradients of the logged energies.

This is acceptable, but the code and documentation should not imply that every term is exactly:

\[
v_{\mathrm{edit}} = \nabla E_{\mathrm{edit}}.
\]

## Required Documentation Change

Update comments / README / method notes to use this terminology:

```text
The current implementation uses energy-inspired surrogate velocity fields.
Some branches are exact autograd gradients, while others are manually constructed velocity surrogates derived from clean-space displacement or feature-space differences.
```

Use notation like:

\[
\dot{x}_t
=
\underbrace{v_{src} + u_{rec}}_{\text{reconstruction-aware base field}}
+
\underbrace{u_{edit}}_{\text{editing field}},
\]

where

\[
u_{edit}
=
\lambda_{flow}(v_{tar}-v_{src})
+
\lambda_{anchor}u_{anchor}
+
\lambda_{region}u_{region}
+
\lambda_{clip}u_{clip}
+
\cdots.
\]

## Optional Code Change

Add a config flag:

```python
--edit-field-mode surrogate | autograd
```

For now, it is fine if only `surrogate` is fully implemented. The flag makes the design explicit.

---

# P1: Add Diagnostic Cosine Similarities

Add diagnostics to detect whether different editing branches agree or cancel each other.

## Suggested helper

```python
def cosine_safe(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    denom = a_flat.norm() * b_flat.norm() + eps
    return float(torch.dot(a_flat, b_flat) / denom)
```

## Add to stats

At each step, log:

```python
"cos_base_anchor": cosine_safe(v_edit_terms["base"], v_edit_terms["anchor"]),
"cos_base_region": cosine_safe(v_edit_terms["base"], v_edit_terms["region"]),
"cos_anchor_region": cosine_safe(v_edit_terms["anchor"], v_edit_terms["region"]),
"cos_rec_base": cosine_safe(v_rec, v_base),
"cos_rec_edit_total": cosine_safe(v_rec, v_edit_total),
```

Expected use:

- If `cos_base_anchor < 0`, the anchor branch may be fighting the base target-source velocity.
- If `cos_rec_edit_total` is strongly negative, the edit and reconstruction fields are in strong conflict.
- These values should be saved in the JSON stats for each run.

---

# P1: Standardize Baseline Configs / Scripts

Create a `scripts/` folder with reproducible commands.

Suggested files:

```text
scripts/run_base_only.sh
scripts/run_direct_target.sh
scripts/run_decoupled_rec.sh
scripts/run_anchor_only.sh
scripts/run_ablation_all.sh
```

Each script should save:

- output image
- stats JSON
- exact command / config

Use output folder structure like:

```text
outputs/
  base_only/
  direct_target/
  decoupled_rec/
  anchor_only/
  ablations/
```

## Baseline 1: Base / source only

Purpose: test source inversion and source-conditioned base dynamics.

```bash
python run_edit_sd3.py \
  --image h_edit_compare/panda.jpg \
  --source-prompt "a panda" \
  --prompt "a tiger" \
  --output outputs/base_only/result.png \
  --stats-output outputs/base_only/stats.json \
  --edit-hedit-guidance-scale 0 \
  --edit-guidance-scale 0 \
  --edit-region-guidance-scale 0 \
  --edit-target-guidance-scale 0 \
  --edit-source-guidance-scale 0 \
  --edit-clip-guidance-scale 0 \
  --edit-text-guidance-scale 0 \
  --edit-dds-guidance-scale 0 \
  --edit-app-guidance-scale 0 \
  --rec-guidance-scale 0 \
  --beta-max 1.0 \
  --log-every 1
```

## Baseline 2: Direct target flow

Purpose: test target-conditioned flow without reconstruction correction.

```bash
python run_edit_sd3.py \
  --image h_edit_compare/panda.jpg \
  --source-prompt "a panda" \
  --prompt "a tiger" \
  --output outputs/direct_target/result.png \
  --stats-output outputs/direct_target/stats.json \
  --edit-hedit-guidance-scale 1.0 \
  --rec-guidance-scale 0 \
  --edit-guidance-scale 0 \
  --edit-region-guidance-scale 0 \
  --edit-clip-guidance-scale 0 \
  --edit-text-guidance-scale 0 \
  --edit-dds-guidance-scale 0 \
  --edit-app-guidance-scale 0 \
  --beta-max 1.0 \
  --log-every 1
```

This approximately tests:

\[
v_{src} + (v_{tar}-v_{src}) = v_{tar}.
\]

## Baseline 3: Decoupled reconstruction + direct edit

Purpose: test whether reconstruction correction improves faithfulness under the same target flow edit.

```bash
python run_edit_sd3.py \
  --image h_edit_compare/panda.jpg \
  --source-prompt "a panda" \
  --prompt "a tiger" \
  --output outputs/decoupled_rec/result.png \
  --stats-output outputs/decoupled_rec/stats.json \
  --edit-hedit-guidance-scale 1.0 \
  --rec-guidance-scale 0.25 \
  --edit-guidance-scale 0 \
  --edit-region-guidance-scale 0 \
  --edit-clip-guidance-scale 0 \
  --edit-text-guidance-scale 0 \
  --edit-dds-guidance-scale 0 \
  --edit-app-guidance-scale 0 \
  --beta-max 1.0 \
  --log-every 1
```

## Baseline 4: Anchor-only edit branch

Purpose: isolate the clean-delta-to-velocity conversion.

```bash
python run_edit_sd3.py \
  --image h_edit_compare/panda.jpg \
  --source-prompt "a panda" \
  --prompt "a tiger" \
  --output outputs/anchor_only/result.png \
  --stats-output outputs/anchor_only/stats.json \
  --edit-hedit-guidance-scale 0 \
  --edit-guidance-scale 1.0 \
  --rec-guidance-scale 0 \
  --edit-region-guidance-scale 0 \
  --edit-clip-guidance-scale 0 \
  --edit-text-guidance-scale 0 \
  --edit-dds-guidance-scale 0 \
  --edit-app-guidance-scale 0 \
  --beta-max 1.0 \
  --log-every 1
```

Compare this with direct target flow. If the visual edit direction is opposite or the cosine diagnostic is negative, the clean-to-velocity sign is likely still wrong.

---

# P1: Save Full Run Metadata

For each run, save a metadata JSON next to the output image.

Suggested fields:

```json
{
  "image": "...",
  "source_prompt": "...",
  "target_prompt": "...",
  "seed": 10,
  "num_inference_steps": 28,
  "src_guidance_scale": 3.5,
  "tar_guidance_scale": 10.5,
  "rec_guidance_scale": 0.25,
  "edit_hedit_guidance_scale": 1.0,
  "edit_guidance_scale": 0.0,
  "beta_max": 1.0,
  "alpha_max": 0.25,
  "velocity_conversion_mode": "linear_path",
  "git_commit": "... optional ..."
}
```

This will make experiments reproducible.

---

# P2: Add a Compact Experiment Summary Script

Add a small script:

```text
scripts/summarize_stats.py
```

Input:

```bash
python scripts/summarize_stats.py outputs/*/stats.json
```

Output a CSV with columns:

```text
run_name, final_rec_energy, final_edit_anchor_energy, avg_rec_norm, avg_edit_norm, avg_cos_base_anchor, avg_cos_rec_edit_total, output_path
```

This is useful before implementing expensive image-level metrics.

---

# P2: Update README

Add a short section:

```markdown
## Current Method Convention

We use the linear RF path:

x_t = (1-t)x_0 + t x_1

and the clean estimate:

x0_hat = x_t - t * v_theta(x_t,t)

The controlled editing dynamics are implemented as:

xdot_t = v_src + u_rec + u_edit

where u_rec is a reconstruction-aware preserve correction and u_edit is a sum of target-seeking editing velocity surrogates.
```

Also document that some edit branches are exact gradients while others are surrogate velocity fields.

---

# Acceptance Criteria

Codex should finish the task only when all of the following are true:

1. `python -m py_compile *.py` passes.
2. The new toy tests for clean-delta-to-velocity conversion pass.
3. `beta_max` no longer defaults to the max branch scale.
4. There is a clear `linear_path` velocity conversion mode.
5. Anchor and region edit velocity branches use the same clean-delta-to-velocity convention.
6. Reconstruction correction has a `linear_path` mode or an explicitly documented legacy mode.
7. Stats JSON includes effective scales and cosine diagnostics.
8. At least three baseline scripts exist:
   - base only
   - direct target
   - decoupled reconstruction + direct target
9. README or method notes explain that the implementation currently uses energy-inspired surrogate velocity fields.

---

# Do Not Do Yet

Do **not** add more editing heuristics before completing the P0/P1 cleanup.

Do **not** tune many prompts before the sign / scale issues are fixed.

Do **not** claim the current surrogate velocities are exact gradients unless implemented with autograd.

Do **not** optimize for visual quality before the direct-vs-decoupled baselines are reproducible.

---

# Main Expected Outcome

After this cleanup, the project should be able to fairly compare:

\[
\text{base flow only}
\]

\[
\text{direct target guidance}
\]

\[
\text{decoupled reconstruction/editing guidance}
\]

and determine whether the decoupled formulation truly improves the editing-effectiveness / faithfulness trade-off.
