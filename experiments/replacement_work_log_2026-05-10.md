# Replacement Work Log

Date: 2026-05-10

## Goal

Add an object-replacement experiment that is separate from the existing
semantic-removal backpack task. The replacement task should use a new image,
not one already assigned to the main qualitative set.

## Decisions

The backpack image should remain dedicated to deletion:

- `backpack_remove_toy_charm` is the current semantic-removal success case.
- Reusing the same backpack image for replacement makes the paper narrative
  confusing, so backpack replacement probes are not recommended as final
  evidence.

The current replacement candidate to continue is:

- `dog_replace_tennis_ball_star`
- Source image: `data/replacement_candidates/commons_dog_with_tennis_ball.jpg`
- Edit: green tennis ball in dog's mouth -> red star-shaped dog toy
- Source: Wikimedia Commons `Dog with tennis ball.jpg`
- Current review: `outputs/pretty_matrix/dog_replace_tennis_ball_star_seed10_review.png`

## Probes Run

| Task | Status | Notes |
| --- | --- | --- |
| `backpack_replace_patch_badge` | Do not use | Circular badge replacement left visible boundary/indent artifacts. |
| `backpack_replace_patch_blue` | Supplemental only | Stable, but it is mostly attribute/color replacement rather than object replacement. |
| `backpack_replace_toy_heart_charm` | Do not use as final | Better replacement semantics, but it reuses the backpack deletion image. |
| `cat_replace_bell_heart_tag` | Weak | New image, but target bell is too small; output is hard to see and not a strong figure. |
| `dog_replace_tennis_ball_star` | Best current replacement candidate | New image, visible object change, automatic tennis-ball mask is good. |

## Current Outputs

Main candidate:

- `outputs/pretty_matrix/dog_replace_tennis_ball_star/full/seed_10/result.png`
- `outputs/pretty_matrix/dog_replace_tennis_ball_star/base_only/seed_10/result.png`
- `outputs/pretty_matrix/dog_replace_tennis_ball_star/direct_target/seed_10/result.png`
- `outputs/pretty_matrix/dog_replace_tennis_ball_star_seed10_review.png`

Other probe outputs:

- `outputs/pretty_matrix/backpack_replace_patch_blue_seed10_review.png`
- `outputs/pretty_matrix/backpack_replace_patch_blue_full_seeds_review.png`
- `outputs/pretty_matrix/backpack_replace_toy_heart_charm_seed10_review.png`
- `outputs/pretty_matrix/cat_replace_bell_heart_tag/full/seed_10/result.png`

## Code Changes

- `scripts/run_pretty_matrix.sh`
  - Added replacement task configs:
    - `P8/backpack_replace_patch_blue`
    - `P9/backpack_replace_toy_heart_charm`
    - `P10/cat_replace_bell_heart_tag`
    - `P11/dog_replace_tennis_ball_star`
  - `P11` is the only replacement candidate currently worth continuing.

- `scripts/make_mask_badge_reference.py`
  - Expanded from badge-only reference generation to generic replacement-shape
    reference construction.
  - Supports `ellipse`, `semantic`, `heart`, and `star` shapes.
  - Supports inpaint/patch/median background filling before adding replacement
    shape.

## Next Steps

1. Continue with `P11` only for replacement.
2. Try a slightly larger or sharper star reference if needed; current M4 output
   is visible but a bit soft.
3. Run `P11` over seeds `10 11 12` for M4 after visual tuning.
4. If stable, add `P11` as a supplemental object-replacement task in
   `experiments/pretty_matrix.md` and the academic-pipeline plan.
5. Keep the four main tasks unchanged:
   `cat_crown`, `dog_sunglasses`, `mug_heart`, `backpack_remove_toy_charm`.
