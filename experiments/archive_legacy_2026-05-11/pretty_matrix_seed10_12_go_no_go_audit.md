# Pretty Matrix Seeds 10-12 Go/No-Go Audit

Date: 2026-05-10

## Coverage

Completed matrix:

```text
4 tasks x 6 methods x seeds 10,11,12 = 72 runs
```

Tasks:

- `cat_crown`
- `dog_sunglasses`
- `mug_heart`
- `backpack_remove_toy_charm`

Methods:

- `base_only`
- `direct_target`
- `full_no_ref`
- `full_no_rec`
- `full_no_traj`
- `full`

Each run has:

```text
result.png
stats.json
metadata.json
command.txt
```

Review grids were generated for seeds 10, 11, and 12 under:

```text
outputs/pretty_matrix/*_seed*_go_no_go_review.png
```

Full-method seed stability grids were generated under:

```text
outputs/pretty_matrix/*_full_seed_stability_review.png
```

## Visual Decisions

| Task | Decision | Notes |
| --- | --- | --- |
| `cat_crown` | Pass | `full` is stable across seeds 10, 11, and 12. The crown is visible and the cat identity/background are preserved well enough for the main qualitative set. Ablations remain visually strong, so this task is useful for success demonstration but weak for isolating module contributions. |
| `dog_sunglasses` | Weak pass | `full` consistently adds eyeglasses/sunglasses across seeds, but the lenses are semi-transparent and subtle. This task is valuable for the preservation/edit tradeoff because `direct_target` changes the dog more strongly, but it should not be described as a strong accessory insertion case. A seed-10 tuning check found that `full_no_ref` gives darker lenses than the current reference-guided `full`; lowering reference guidance and changing the prompt to `opaque black sunglasses` did not fix the transparency. |
| `mug_heart` | Pass | `full` is stable across all three seeds. The red heart is clearly inserted and the mug is preserved. `full_no_ref` remains the important ablation because the reference prior materially supports the printed-heart appearance. |
| `backpack_remove_toy_charm` | Pass | `full` is stable across seeds 10, 11, and 12. The lower yellow toy charm is removed while the pink ring, zipper, backpack fabric, and source cartoon patch remain. The upper cartoon patch is present in the source image and should not be counted as an artifact. |

## Go/No-Go

Go for the next experimental stage.

The four-task qualitative matrix is sufficient to proceed to baseline parity
setup and a small external-method comparison. The current claim should remain
narrow:

```text
clean-estimate-space rectified-flow control for localized image editing,
with automatic support/reference construction
```

Do not claim broad state-of-the-art image editing. The strongest evidence is
localized preservation with a usable edit, not universal edit strength.

## Baseline Update

FlowEdit has been run for the same four tasks and seeds 10, 11, and 12. The
runner and manifest are:

```text
scripts/run_flowedit_baseline.py
experiments/baseline_parity_manifest.csv
```

The generated comparison grids are:

```text
outputs/baselines/flowedit/*/flowedit_vs_full_seeds10_12.png
```

Initial visual readout: FlowEdit often produces a stronger target object, but it
frequently redraws the source image instead of preserving local identity and
layout. This is especially clear for `cat_crown`, `dog_sunglasses`, and
`backpack_remove_toy_charm`.

SplitFlow has also been run for the same four tasks and seeds. It improves the
target edit strength on `dog_sunglasses`, where the sunglasses are darker and
more explicit, but still changes dog identity. It remains weak on
`backpack_remove_toy_charm`, where it redraws the scene and keeps or regenerates
a toy charm.

FireFlow has also been run for the same four tasks and seeds using its official
FLUX-dev fast-editing configuration. It adds visible sunglasses and crowns, but
again changes identity/pose. It also fails the localized backpack removal case.

RF-Solver-Edit has also been run for the same four tasks and seeds using its
official FLUX-dev image-editing configuration. The public script does not expose
seed control, so these are official-configuration qualitative runs rather than
strict seed-matched baselines.

ReFlex and SteerFlow are recorded as failed baseline rows rather than omitted:
ReFlex cannot run within the current 24GB GPU and public-code constraints, and
SteerFlow has no public runnable code found as of 2026-05-10. Overall,
FlowEdit, SplitFlow, FireFlow, and RF-Solver-Edit support positioning our method
around localized preservation rather than unconstrained target-prompt strength.

## Immediate Next Step

1. Convert the seed 10-12 visual audit into a compact CSV with task/method/seed
   ratings.
2. Convert the completed/failed baseline manifest into a compact paper table
   that separates runnable qualitative baselines from unavailable/resource-
   blocked baselines.
3. Treat `dog_sunglasses` as the first method-improvement task if time remains.
   The likely fix is not prompt tuning, but a glasses-specific reference gate or
   a darker/less conflicting glasses reference mechanism.
