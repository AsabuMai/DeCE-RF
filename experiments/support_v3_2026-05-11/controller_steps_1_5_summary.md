# Support-v3 Controller Steps 1-5 Summary

Date: 2026-05-11

## Scope

This run follows `docs/rf_h_edit_controller_next_steps.md` Steps 1-5 for the
climate/spatial-support mainline:

1. multi-seed controller validation;
2. controller diagnostic curves;
3. edit-strength sweep;
4. support perturbation robustness;
5. hybrid controller implementation and seed10 validation.

The active methods are:

```text
support_v3_fixed
support_v3_controller_rmsgap
support_v3_controller_progress
support_v3_controller_hybrid
```

The old aliases `fixed_full_support_v3`, `adaptive_full_support_v3_v0`, and
`adaptive_full_support_v3_v1` are retained only as script compatibility names.

## Step 1: Multi-seed Validation

Command:

```text
TASKS="P1 P2 P3" METHODS="M17 M18 M19" SEEDS="10 11 12" DEVICE=0 SKIP_EXISTING=1 REGENERATE_MASKS=0 ALLOW_MASK_DOWNLOAD=0 scripts/run_pretty_matrix.sh
```

Evaluation output:

```text
experiments/support_v3_2026-05-11/controller_multiseed_metrics_clip.csv
experiments/support_v3_2026-05-11/controller_multiseed_metrics_clip.json
experiments/support_v3_2026-05-11/controller_multiseed_summary.csv
```

Result: 27/27 complete.

Key mean results:

```text
cat_crown:
  fixed    CLIP 0.092001 outside 0.053744 inside 0.165701 SSIM 0.871580
  rmsgap   CLIP 0.099463 outside 0.054045 inside 0.135995 SSIM 0.874010
  progress CLIP 0.087521 outside 0.054769 inside 0.197253 SSIM 0.862605

dog_sunglasses:
  fixed    CLIP 0.098302 outside 0.045829 inside 0.183537 SSIM 0.903563
  rmsgap   CLIP 0.100457 outside 0.045776 inside 0.183267 SSIM 0.903219
  progress CLIP 0.092818 outside 0.045888 inside 0.186031 SSIM 0.902243

mug_heart:
  fixed    CLIP 0.042326 outside 0.007800 inside 0.077925 SSIM 0.864663
  rmsgap   CLIP 0.043510 outside 0.007699 inside 0.076223 SSIM 0.867803
  progress CLIP 0.043662 outside 0.008010 inside 0.083772 SSIM 0.850444
```

Conclusion: `support_v3_controller_rmsgap` is the stable main candidate.
The pure progress controller is more destructive and does not give reliable
semantic gain.

## Step 2: Controller Curves

Output directory:

```text
experiments/support_v3_2026-05-11/controller_curves_seed10/
```

Generated:

```text
controller_curves_seed10.csv
18 PNG curves: task x metric for rmsgap/progress
```

Tracked metrics:

```text
edit_rms_gap
directional_progress
preserve_drift
adaptive_edit_weight
adaptive_preserve_weight
projection_ratio
```

## Step 3: Edit-strength Sweep

Generated 20 seed10 runs for cat_crown and mug_heart:

```text
support_v3_fixed_edit050/075/100/125/150
support_v3_controller_rmsgap_edit050/075/100/125/150
```

Evaluation output:

```text
experiments/support_v3_2026-05-11/controller_edit_strength_sweep_metrics_clip.csv
experiments/support_v3_2026-05-11/controller_edit_strength_sweep_metrics_clip.json
experiments/support_v3_2026-05-11/controller_edit_strength_sweep/edit_strength_tradeoff.csv
experiments/support_v3_2026-05-11/controller_edit_strength_sweep/cat_crown_clip_vs_outside_l1.png
experiments/support_v3_2026-05-11/controller_edit_strength_sweep/mug_heart_clip_vs_outside_l1.png
```

Result: 20/20 complete.

Conclusion: simply increasing edit strength is not the bottleneck. Cat is best
around rmsgap 1.0 and degrades at 1.5. Mug often prefers lower strength, so the
limitation is support/controller quality rather than raw edit force.

## Step 4: Support Perturbation Sweep

Implemented mask perturbation controls:

```text
--edit-mask-erode-kernel
--edit-mask-hole-fraction
--edit-mask-boundary-noise-scale
```

Existing controls used:

```text
--edit-mask-dilate-kernel
--edit-mask-shift-x
--edit-mask-shift-y
```

Generated 20 seed10 runs on dog_sunglasses and mug_heart:

```text
erode
dilate
shift
holes
boundary_noise
```

Evaluation output:

```text
experiments/support_v3_2026-05-11/controller_support_perturb_metrics_clip.csv
experiments/support_v3_2026-05-11/controller_support_perturb_metrics_clip.json
experiments/support_v3_2026-05-11/controller_support_perturb/support_perturb_summary.csv
experiments/support_v3_2026-05-11/controller_support_perturb/support_perturb_rmsgap_minus_fixed.csv
```

Plots:

```text
experiments/support_v3_2026-05-11/controller_support_perturb/dog_sunglasses_clip_target_minus_source.png
experiments/support_v3_2026-05-11/controller_support_perturb/dog_sunglasses_outside_mask_l1.png
experiments/support_v3_2026-05-11/controller_support_perturb/dog_sunglasses_inside_mask_l1.png
experiments/support_v3_2026-05-11/controller_support_perturb/mug_heart_clip_target_minus_source.png
experiments/support_v3_2026-05-11/controller_support_perturb/mug_heart_outside_mask_l1.png
experiments/support_v3_2026-05-11/controller_support_perturb/mug_heart_inside_mask_l1.png
```

Result: 20/20 complete.

Conclusion: rmsgap is at least comparable under small support errors and often
slightly better than fixed. The gain is modest, but it is consistent enough to
support keeping rmsgap as the main controller baseline.

## Step 5: Hybrid Controller

Implemented:

```text
support_v3_controller_hybrid
```

Hybrid formula behavior:

```text
RMS-gap base deficit
+ gated directional-progress deficit
+ EMA-smoothed progress
+ preserve-drift gate
+ target-delta reliability gate
```

New CLI/controller parameters:

```text
--adaptive-hybrid-progress-target
--adaptive-hybrid-progress-gain
--adaptive-hybrid-progress-ema-decay
--adaptive-hybrid-preserve-gate-budget
```

Run:

```text
TASKS="P1 P2 P3" METHODS="M20" SEEDS="10" DEVICE=0 SKIP_EXISTING=1 REGENERATE_MASKS=0 ALLOW_MASK_DOWNLOAD=0 scripts/run_pretty_matrix.sh
```

Evaluation output:

```text
experiments/support_v3_2026-05-11/controller_hybrid_seed10_metrics_clip.csv
experiments/support_v3_2026-05-11/controller_hybrid_seed10_metrics_clip.json
```

Result: 12/12 complete in the four-method seed10 comparison.

Seed10 metrics:

```text
cat_crown:
  fixed    CLIP 0.091988 outside 0.053763 inside 0.167657 SSIM 0.871412
  rmsgap   CLIP 0.097850 outside 0.054044 inside 0.137336 SSIM 0.873774
  progress CLIP 0.089067 outside 0.054767 inside 0.198760 SSIM 0.862285
  hybrid   CLIP 0.100162 outside 0.054039 inside 0.137649 SSIM 0.873920

dog_sunglasses:
  fixed    CLIP 0.098324 outside 0.045825 inside 0.182653 SSIM 0.903803
  rmsgap   CLIP 0.100651 outside 0.045777 inside 0.182736 SSIM 0.903407
  progress CLIP 0.092668 outside 0.045856 inside 0.187826 SSIM 0.901688
  hybrid   CLIP 0.101297 outside 0.045776 inside 0.182836 SSIM 0.903414

mug_heart:
  fixed    CLIP 0.042220 outside 0.007824 inside 0.078059 SSIM 0.864599
  rmsgap   CLIP 0.043229 outside 0.007690 inside 0.076019 SSIM 0.868434
  progress CLIP 0.043742 outside 0.007959 inside 0.083226 SSIM 0.850829
  hybrid   CLIP 0.043710 outside 0.007759 inside 0.076981 SSIM 0.866115
```

Hybrid gate diagnostics:

```text
cat_crown:       gate_steps 6,  max_boost 0.056257, max_weight 1.55
dog_sunglasses:  gate_steps 5,  max_boost 0.006752, max_weight 1.55
mug_heart:       gate_steps 11, max_boost 0.132422, max_weight 1.55
```

Conclusion: hybrid is implemented and works as a controlled experimental
branch. It slightly improves cat/dog CLIP over rmsgap at seed10, but mug gains
CLIP only marginally while losing some preservation. Current mainline should
still report rmsgap as the stable controller, with hybrid as an exploratory
next refinement rather than the primary result.

## Current Bottleneck

The bottleneck is not raw edit strength. The limiting factors are:

```text
1. support quality and relation selection for small localized edits;
2. controller ability to increase edit without expanding inside-mask damage;
3. progress signal noisiness on weak or small target deltas;
4. preservation tradeoff on decal-like edits such as mug_heart.
```

Recommended next step: freeze `support_v3_controller_rmsgap` for main tables,
then tune hybrid only with a stricter progress gate or task-aware lower boost
for decal edits.
