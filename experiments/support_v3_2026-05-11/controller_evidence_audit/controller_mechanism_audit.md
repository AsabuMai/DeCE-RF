# Controller Mechanism Audit

Date: 2026-05-11

## Objective

This audit is for the current technical task, not for paper writing. The goal is
to understand whether the current controller is genuinely useful, what mechanism
it is using, and what should be tested next without disturbing the working
baseline.

## Artifacts

```text
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_per_run.csv
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_per_run_delta.csv
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_success_rate.csv
experiments/support_v3_2026-05-11/controller_evidence_audit/tradeoff_all_tasks.png
experiments/support_v3_2026-05-11/controller_evidence_audit/tradeoff_cat_crown.png
experiments/support_v3_2026-05-11/controller_evidence_audit/tradeoff_dog_sunglasses.png
experiments/support_v3_2026-05-11/controller_evidence_audit/tradeoff_mug_heart.png
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_behavior_combined_seed10.png
```

## Per-run Result

Relative to `support_v3_fixed`, current `support_v3_controller_rmsgap` gives:

```text
CLIP improved:                  9 / 9
outside L1 within eps=0.001:    9 / 9
outside L1 not increased:       6 / 9
SSIM not worse by more 0.002:   9 / 9
loose success:                  9 / 9
strict success:                 6 / 9
mean delta CLIP:                +0.003601
mean delta outside L1:          +0.000049
mean delta inside L1:           -0.010559
mean delta SSIM:                +0.001742
```

Interpretation:

- The rmsgap controller has a real effect, but the effect is small.
- Its main improvement is edit alignment and inside-region behavior, not a large
  outside-preservation improvement.
- The conservative claim is therefore correct: small consistent edit gains with
  comparable non-edit preservation.

Progress-only controller is not a good main branch:

```text
strict success:                 0 / 9
mean delta CLIP:                -0.002876
mean delta outside L1:          +0.000431
mean delta inside L1:           +0.013298
mean delta SSIM:                -0.008171
```

Hybrid seed10 is promising but not enough to replace rmsgap yet:

```text
CLIP improved:                  3 / 3
strict success:                 2 / 3
mean delta CLIP:                +0.004212
mean delta outside L1:          +0.000054
mean delta inside L1:           -0.010301
mean delta SSIM:                +0.001212
```

## Important Mechanism Finding

The current method name `support_v3_controller_rmsgap` is slightly misleading if
read against the newer mathematical plan.

The current implementation uses:

```text
adaptive_edit_deficit = max(0, adaptive_edit_target_rms - adaptive_edit_target_gap_rms)
adaptive_edit_weight  = 1 + adaptive_edit_gain * adaptive_edit_deficit
```

With `adaptive_edit_target_rms = 0.42`, the controller boosts edit only when the
measured RMS gap falls below the threshold. It does not boost when the gap is
large.

Seed10 diagnostics confirm this:

```text
cat_crown:
  first step gap 3.1644, weight 1.0000
  first boost step 22, t 0.4093, gap 0.3751, weight 1.0899
  last step gap 0.0021, weight 1.5500

dog_sunglasses:
  first step gap 4.1244, weight 1.0000
  first boost step 22, t 0.4093, gap 0.3580, weight 1.1241
  last step gap 0.0016, weight 1.5500

mug_heart:
  first step gap 1.4746, weight 1.0000
  first boost step 18, t 0.6022, gap 0.3370, weight 1.1659
  last step gap 0.0022, weight 1.5500
```

Therefore, the current rmsgap controller should be understood as:

```text
late-stage low-residual finishing boost
```

not:

```text
large-gap pursuit controller
```

This likely explains why it is stable. It avoids injecting extra edit force in
early high-noise/high-gap steps and only boosts after the trajectory is already
near the local target.

## Current Technical Conclusion

Keep `support_v3_controller_rmsgap` unchanged as the current main baseline. It
is empirically useful and stable.

However, the newer normalized RMS-gap idea should be tested as a separate branch
instead of replacing the current controller.

## Next Branch

Add a new method:

```text
support_v3_controller_rmsgap_normgate
```

It should implement:

```text
normalized_gap = gap_rms / (target_delta_rms + eps)
active_gap = max(0, normalized_gap - dead_zone)
preserve_gate = preserve_drift < preserve_gate_budget
edit_weight = 1 + gain * active_gap * preserve_gate
```

This branch tests the mathematically direct controller from the detailed plan.
It must be evaluated against the current rmsgap before any replacement decision.

Decision rule:

```text
If normgate improves CLIP without increasing outside L1 and does not degrade
mug preservation, expand to seeds 10-12.

If it fails seed10, keep current rmsgap and record normgate as a negative result.
```

## Normgate Seed10 Result

Implemented:

```text
support_v3_controller_rmsgap_normgate
```

Configuration:

```text
adaptive_rmsgap_mode = normgate
dead_zone = 0.15
preserve_gate_budget = 0.18
adaptive_edit_gain = 0.55
adaptive_edit_weight_max = 1.35
```

Run:

```text
TASKS="P1 P2 P3" METHODS="M21" SEEDS="10" DEVICE=0 SKIP_EXISTING=1 REGENERATE_MASKS=0 ALLOW_MASK_DOWNLOAD=0 scripts/run_pretty_matrix.sh
```

Evaluation:

```text
experiments/support_v3_2026-05-11/controller_normgate_seed10_metrics_clip.csv
experiments/support_v3_2026-05-11/controller_normgate_seed10_metrics_clip.json
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_normgate_seed10_delta.csv
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_normgate_seed10_visual.png
```

Seed10 comparison against current rmsgap:

```text
cat_crown:
  delta CLIP +0.000607
  delta outside L1 +0.000012
  delta inside L1 +0.000410
  delta SSIM -0.000201

dog_sunglasses:
  delta CLIP -0.000591
  delta outside L1 -0.000006
  delta inside L1 -0.001431
  delta SSIM +0.000459

mug_heart:
  delta CLIP +0.000039
  delta outside L1 +0.000021
  delta inside L1 +0.000663
  delta SSIM -0.002195
```

Controller diagnostics:

```text
cat_crown:       active_steps 4, gate_off_steps 18, max_weight 1.1603
dog_sunglasses:  active_steps 3, gate_off_steps 19, max_weight 1.1275
mug_heart:       active_steps 7, gate_off_steps 13, max_weight 1.2506
```

Decision:

```text
Do not expand normgate to seeds 10-12 in this configuration.
```

Reason:

- It does not consistently improve CLIP over current rmsgap.
- It slightly worsens outside L1 for cat and mug.
- It degrades mug SSIM by about 0.0022 relative to current rmsgap, crossing the
  strict preservation tolerance.

Technical interpretation:

The direct normalized-gap controller is more principled mathematically, but this
first setting does not beat the existing late-stage finishing controller. The
current rmsgap behavior should remain the main working baseline. Normgate can be
kept as a negative result or revisited later with a stricter preserve gate and
lower mug/decal boost, but it should not block current progress.
