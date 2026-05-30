# Seed-10 New Task Visual Gate

Date: 2026-05-28

Tasks:

```text
tshirt_star
dog_remove_tennis_ball
```

Methods:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

Run policy: seed 10 only. Do not expand to seeds 10/11/12 unless the seed-10
visual gate passes.

## Artifacts

```text
experiments/support_v3_2026-05-11/visual_gates/tshirt_star_seed10_gate.png
experiments/support_v3_2026-05-11/visual_gates/dog_remove_tennis_ball_seed10_gate.png
```

Run directories:

```text
outputs/pretty_matrix/tshirt_star/{base_only,direct_target,adaptive_full_generic_support,support_v3_controller_rmsgap}/seed_10
outputs/pretty_matrix/dog_remove_tennis_ball/{base_only,direct_target,adaptive_full_generic_support,support_v3_controller_rmsgap}/seed_10
```

## Gate Readout

| Task | DeCE-RF read | Gate decision | Reason |
| --- | --- | --- | --- |
| `tshirt_star` | partial | hold / tune | A small red star appears, but it is too small and faint for a main-matrix success. Direct target forms a stronger star but changes the shirt geometry more. |
| `dog_remove_tennis_ball` | fail | do not expand | The green ball is not cleanly removed; DeCE-RF leaves a green residual/ball region and introduces a tongue-like artifact. |

## Notes

- `tshirt_star` support generation finds the full T-shirt surface. This is
  plausible for a decal task, but the current DeCE-RF edit strength is too
  conservative.
- `dog_remove_tennis_ball` support generation localizes the green tennis ball
  cleanly, so the failure is not mainly a support-localization failure. It is a
  removal/target-formation issue.
- Do not run the full 3-seed expansion for these two tasks yet.

## Suggested Next Step

Tune only seed 10:

```text
tshirt_star: increase decal/edit strength or use a larger/brighter star prior.
dog_remove_tennis_ball: test a removal-specific stronger source-object suppression path before expanding.
```

## 2026-05-29 Surface Removal Follow-up

Goal: find a second exposed surface-removal task that could expand the main
matrix beyond `backpack_remove_toy_charm`.

Tested candidates:

```text
laptop_remove_sticker
fridge_remove_yellow_magnet
fridge_remove_peach_magnet
whiteboard_remove_yellow_letter
```

All were run as seed-10 visual gates only. Do not expand to seeds 11/12 unless
a seed-10 result passes visually.

| Task | Support read | DeCE-RF read | Gate decision |
| --- | --- | --- | --- |
| `laptop_remove_sticker` | accurate support over the sticker | colorful sticker is mostly suppressed, but the model leaves barcode/label-like marks or surface damage under prompt variants | fail; limitation |
| `fridge_remove_yellow_magnet` | accurate support over the yellow round magnet | default leaves a small yellow residual; stronger clean-blue variant removes more but distorts the nearby red deer magnet | fail; limitation |
| `fridge_remove_peach_magnet` | accurate support over the upper-right peach round magnet | default is too conservative; stronger variant removes the target but damages neighboring stormtrooper magnets | fail; limitation |
| `whiteboard_remove_yellow_letter` | accurate support over the yellow magnetic letter | the target letter changes into another letter rather than being removed | fail; limitation |

Readout:

```text
These failures are not primarily support-localization failures. They show a
local target-formation / object-erasure limitation: the method can identify the
right source object, but removal requires plausible surface completion without
hallucinating a residual symbol, tag, or replacement object.
```

Decision:

```text
Stop trying to force surface sticker/magnet/letter removal into the main
matrix. Keep these runs as stress/limitation evidence. If the paper needs more
than the completed core-5, expand through replacement and appearance/recolor
seed-10 gates instead of more hard removal probes.
```
