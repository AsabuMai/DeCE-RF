# RMSGAP Mainline Mechanism Audit

Date: 2026-05-12

## Decision

The current mainline is:

```text
support_v3_controller_rmsgap
```

This should be treated as the stable controller branch. `support_v3_fixed` is
the baseline. `support_v3_core_target_transport` / M22 is not the mainline.

The claim should be framed as clean-estimate-space closed-loop control, not as a
new mask method:

```text
Support decides where the edit is allowed.
RMSGAP decides how much edit/preserve correction is allowed during the RF ODE.
```

## Project Flow

The active run path is:

```text
scripts/run_pretty_matrix.sh
  -> run_edit_sd3.py
    -> sd3_hrec.HRecSD3Edit()
      -> sd3_model_ops.py for SD3 velocity/inversion/x0 prediction
      -> operation_support_v3.py + spatial_masks.py for M_edit/M_core/M_preserve
      -> energies.py for clean-estimate reconstruction/edit surrogates
      -> guidance_fields.py for optional decode-space color/reference guidance
      -> stats.json/result.png/metadata.json
```

Main active files:

- `scripts/run_pretty_matrix.sh`: task table, method table, support-v3 routing,
  and experiment command assembly.
- `run_edit_sd3.py`: CLI surface and argument forwarding.
- `sd3_hrec.py`: main RF ODE loop, diagnostics, rmsgap controller, masking,
  guidance fusion, and output writing.
- `sd3_model_ops.py`: SD3 CFG velocity calls, source inversion, source Q/K/V
  injection, feature extraction, and linear RF clean estimate.
- `operation_support_v3.py`: operation-conditioned support candidate builder.
- `spatial_masks.py`: mask morphology, component filtering, object/contact
  layering, mask stats, and external mask loading.
- `energies.py`: clean-estimate-space reconstruction/edit energy and velocity
  surrogate terms.
- `scripts/evaluate_paper_metrics.py` and
  `scripts/analyze_controller_stress.py`: metrics and stress summaries.

## Task And Method Routing

Tasks are fixed in `scripts/run_pretty_matrix.sh::task_config`.

Each task defines:

- source image
- source prompt
- target prompt
- target/new tokens
- host tokens
- removed/source tokens
- edit operation
- support relation
- optional semantic phrase for GroundingDINO/SAM cache

Current 12-case dev support policy is frozen in:

```text
docs/support_policy_v1.md
```

Method routing of interest:

```text
M17 = support_v3_fixed
M18 = support_v3_controller_rmsgap
M21 = support_v3_controller_rmsgap_normgate
M22 = support_v3_core_target_transport
```

## Fixed vs RMSGAP vs M22

`support_v3_fixed`:

```text
EDIT_HEDIT_GUIDANCE_SCALE = 0.65
EDIT_TEXT_GUIDANCE_SCALE  = 0.08
REC_GUIDANCE_SCALE        = 0.22
TRAJECTORY_PRESERVE_SCALE = 0.12
ADAPTIVE_CLEAN_CONTROL    = 0
```

`support_v3_controller_rmsgap`:

```text
same base edit/preserve terms as fixed
ADAPTIVE_CLEAN_CONTROL = 1
ADAPTIVE_EDIT_TARGET_RMS = 0.42
ADAPTIVE_EDIT_GAIN = 2.0
ADAPTIVE_EDIT_WEIGHT_MAX = 1.55
ADAPTIVE_PRESERVE_DRIFT_BUDGET = 0.18
ADAPTIVE_PRESERVE_GAIN = 2.5
ADAPTIVE_PRESERVE_WEIGHT_MAX = 1.65
ADAPTIVE_PROJECTION_SCALE = 0.65
```

`support_v3_core_target_transport` / M22:

```text
EDIT_HEDIT_GUIDANCE_SCALE = 0.0
REGION_TARGET_TRANSPORT_SCALE = 1.0
REGION_TARGET_OUTSIDE_LOCK_SCALE = 1.0
ADAPTIVE_CLEAN_CONTROL = 0
```

Visual seed10 result: M22 is not better than fixed/rmsgap and is often weaker.
Do not expand it as the main branch.

## Support Mechanism

`operation_support_v3.py` builds candidate support maps from:

- target/new token attention
- host token attention
- removed/source token attention
- clean-estimate disagreement map
- RF velocity disagreement map
- optional grounded/SAM mask
- optional relation region

The operation default chooses candidates by edit type:

```text
add_object + relation -> relation_x_response
add_decal             -> decal_surface_local_response
remove_object + seg   -> seg_only
replace + seg         -> seg_only
recolor + relation    -> relation_only
```

After postprocessing, `sd3_hrec.py` turns this into:

```text
M_edit
M_core
M_contact
M_preserve
```

With `mask_layering_mode=object_contact`, `spatial_masks.build_object_contact_masks`
uses:

```text
object/core: strongest edit region
contact ring: weak blending ring
preserve: outside object/contact support
structure edge: optional contact-ring protection
```

This means support-v3 is a spatial interface, not the controller itself.

## RF ODE Mechanism

At every reverse ODE step, `sd3_hrec.py` computes:

```text
v_tar = SD3 CFG velocity under target prompt
v_src = SD3 CFG velocity under source prompt
v_base = v_src
```

Then it predicts clean estimates using the linear RF path:

```text
x0_tar      = z_t - t * v_tar
x0_src_step = z_t - t * v_src
```

The final step velocity is:

```text
v_total = v_base + v_rec + v_edit_total
```

where:

- `v_base`: source-prompt RF velocity.
- `v_rec`: preserve-region clean-estimate reconstruction correction.
- `v_edit_total`: masked edit guidance, text guidance, optional auxiliary
  guidance, and optional removal/transport branches.

## HEdit Split

`energies.editing_velocity_surrogate_total()` supports:

```text
total =
  lambda_base   * base_edit_velocity
+ lambda_anchor * v_anchor
+ lambda_region * v_region
+ lambda_target * v_target
+ lambda_source * v_source
```

But current fixed/rmsgap uses almost only:

```text
base_edit_velocity = v_tar - v_src_edit
lambda_base = EDIT_HEDIT_GUIDANCE_SCALE
```

The other clean-estimate edit terms are available but not active in the current
main run:

```text
--edit-guidance-scale          # anchor
--edit-region-guidance-scale   # region
--edit-target-guidance-scale   # target feature attract
--edit-source-guidance-scale   # source feature suppress
```

Spatially, edit guidance is weighted by:

```text
edit_core_scale * M_core + edit_subject_scale * (M_edit - M_core)
```

Current defaults:

```text
edit_core_scale = 1.35
edit_subject_scale = 0.35
```

## RMSGAP Controller

When `ADAPTIVE_CLEAN_CONTROL=1`, `sd3_hrec.py` computes diagnostics in
clean-estimate space:

```text
current_delta = x0_src_step - x_source
target_delta  = x0_tar - x_source
target_gap    = x0_tar - x0_src_step
```

Main logged quantities:

```text
adaptive_edit_change_rms       = RMS(current_delta in M_edit)
adaptive_edit_target_rms_value = RMS(target_delta in M_edit)
adaptive_edit_target_gap_rms   = RMS(target_gap in M_edit)
adaptive_edit_progress         = <current_delta, target_delta> / ||target_delta||^2
adaptive_preserve_drift        = RMS(x0_src_step - x_source in M_preserve)
```

Current rmsgap mode is `legacy`:

```text
adaptive_edit_deficit = max(0, adaptive_edit_target_rms - adaptive_edit_target_gap_rms)
adaptive_edit_weight  = clamp(1 + edit_gain * deficit)
```

Because `adaptive_edit_target_rms = 0.42`, this boosts edit only when the
target gap is already small. Empirically, it behaves as:

```text
late-stage low-residual finishing boost
```

not as an early large-gap pursuit controller.

The preserve side is:

```text
adaptive_preserve_excess = max(0, adaptive_preserve_drift - preserve_drift_budget)
adaptive_preserve_weight = clamp(1 + preserve_gain * preserve_excess)
```

Then:

```text
v_rec        *= adaptive_preserve_weight
v_edit_total *= adaptive_edit_weight
```

Finally, if the edit field is predicted to worsen preserve clean-space error,
the projection block removes the destructive component from `v_edit_total`.

This is the real rmsgap contribution:

```text
closed-loop late edit finishing + preserve-aware edit projection
```

## Current Evidence

Older 3-task, 3-seed controller audit:

```text
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_mechanism_audit.md
```

Against `support_v3_fixed`, current rmsgap showed:

```text
CLIP improved:                9 / 9
outside L1 within eps=0.001:  9 / 9
SSIM not worse by 0.002:      9 / 9
strict success:               6 / 9
mean delta CLIP:              +0.003601
mean delta SSIM:              +0.001742
```

New support policy v1 visual gate:

```text
experiments/support_v3_2026-05-11/support_policy_v1_audit/support_policy_v1_edit_gate_audit.md
```

This shows:

- support policy v1 is coherent enough for controlled comparisons;
- pure recolor is now retired from the main evaluation set because it is better
  treated as a deterministic color-transfer baseline rather than a generative
  local-editing claim;
- fixed/rmsgap are reliable on simple decal, remove, and some medium
  replacements;
- small-object target formation remains weak;
- rmsgap is visually comparable to fixed but not consistently better on seed10;
- M22 should not replace rmsgap.

## Mainline Implication

Do not try to prove that rmsgap makes the target object appear from nothing.
That is not what the current controller does.

The defensible rmsgap claim should be:

```text
Given a fixed automatic support and a working local edit field,
rmsgap improves the edit-preserve trade-off by applying clean-estimate-space
closed-loop finishing and preserve-aware projection.
```

The next code work should therefore keep rmsgap as the controller shell and
improve the edit field inside it.

## What To Change Next

Do not tune per image. Change operation-conditioned edit-field composition:

```text
recolor:
  diagnostic only; keep the code path available, but exclude from main
  rmsgap evidence and default gates.

remove_object:
  source suppress / removal branch can be tested, but default currently stays
  off unless it beats fixed visually.

add_decal:
  activate small, generic target formation inside M_core:
  lower local rec in core, increase target/text or target feature attract,
  keep rmsgap preserve projection outside.

replace:
  split into source-object suppression + new-object formation.
  Current support covers the old object correctly, but target insertion is weak.

add_object/on_face:
  keep small relation support; avoid high edit scale that destroys placement.
```

This keeps generality because the rules depend on operation and mask geometry,
not on task id or image filename.

## Immediate Experimental Protocol

For the next gate:

```text
methods: fixed, rmsgap, rmsgap_plus_operation_editfield
tasks: tote_leaf, cat_replace_bell_heart_tag, backpack_replace_patch_blue,
       rabbit_sunglasses, plus 2-3 already-working controls
seed: 10
```

Success criterion:

```text
The new edit field must visibly improve target formation without making rmsgap
lose its preservation behavior. If it only works by destroying outside regions,
it is not a mainline improvement.
```
