# Failure Analysis: Current Non-Usable Paper Tasks

Date: 2026-05-09

Scope: visual and record-level review of the current full-method outputs for
`backpack_blue`, `yellow_car_blue`, and `rabbit_sunglasses` under
`outputs/main_matrix`.

## Summary

Only `cat_crown` is currently a clean positive result. The other three planned
paper tasks should not be used as success examples.

| Task | Current failure | Primary cause |
| --- | --- | --- |
| `backpack_blue` | Hybrid object: a large blue object appears beside a still-red backpack. | The semantic support/ref setup treats the edit like local object insertion around the backpack instead of constrained surface recoloring of the backpack itself. |
| `yellow_car_blue` | Color miss: the car remains mostly yellow. | The vehicle paint/reference path produces an overly broad/fragmented paint support and weak chroma reference signal; the edit field preserves structure but does not deliver a strong color change. |
| `rabbit_sunglasses` | Localization error: small black artifact near the rabbit head/ear, not stable sunglasses on the eye. | The target is a tiny side-profile accessory; the auto eye/support mask is too large initially and then area-guarded to a coarse box, so the edit has poor geometric placement. |

## T2: `backpack_blue`

Current command path:

```text
outputs/main_matrix/backpack_blue/full/seed_10/command.txt
```

Observed setup:

- Target prompt asks for a surface attribute change: backpack fabric changed to
  clean blue.
- `make_semantic_mask.py` inferred phrase `burgundy back`, with relation
  `around` from the backpack support rule.
- The support box expands around the backpack and covers a large rectangular
  context region.
- The full method uses `object_contact` layering and `edit_ref_guidance_scale`
  0.20 with the support mask as the reference mask.

Why it fails:

- The method is being asked to do a surface recolor, but the mask/layout
  support behaves like object insertion/replacement.
- The target prompt contains strong generative language for a blue backpack.
  With a broad support region, the model can satisfy the target by creating a
  new blue backpack-like object rather than recoloring the existing red one.
- The reference overlay confirms a large blue rectangular influence region
  around and beyond the backpack, so surrounding content is allowed to move.

Next experiments:

1. Force a tight object mask for the actual backpack body; avoid `around`
   support for recolor tasks.
2. Use `mask_layering_mode=object` or equivalent tight replacement instead of
   `object_contact`.
3. Disable text/object insertion terms for this task and test a pure
   surface-reference/color branch.
4. Try an easier object-color task with a large, unoccluded surface before
   keeping backpack as a paper task.

## T3: `yellow_car_blue`

Current command path:

```text
outputs/main_matrix/yellow_car_blue/full/seed_10/command.txt
```

Observed setup:

- Semantic mask phrase is inferred as `same same car body changed clean`.
- The semantic car mask covers roughly half the image, including windows and
  non-paint structure.
- `make_vehicle_paint_mask.py` detects the vehicle and excludes wheels/windows,
  but the resulting paint mask is fragmented and heavily tied to yellow-color
  confidence.
- The surface recolor reference overlay is not a clean blue car body; it leaves
  many yellow regions and produces weak teal/blue patches.
- Current metric rows show negative CLIP target-source delta for full method on
  all three seeds.

Why it fails:

- The task needs a strong material/color transform while preserving car
  geometry. The current RF edit field is preservation-biased.
- The active edit terms do not include `edit_color_guidance_scale`; color change
  relies mainly on text plus a weak image reference branch.
- The vehicle-paint mask is not a clean contiguous body mask. It excludes many
  details correctly, but the remaining support is not strong enough to push a
  full yellow-to-blue recolor.
- The reference itself is not a decisive blue target because luma/gradient
  preservation and the current recolor mode keep too much original appearance.

Next experiments:

1. Replace the vehicle paint mask with a manually verified contiguous body mask.
2. Test `edit_color_guidance_scale > 0` with explicit `yellow -> blue` color
   guidance and the car body mask.
3. Increase or simplify the surface reference strength after confirming the
   reference image is visibly blue in the intended body region.
4. Consider dropping this task from the paper unless a simple color-only
   baseline can be made to work first.

## T4: `rabbit_sunglasses`

Current command path:

```text
outputs/main_matrix/rabbit_sunglasses/full/seed_10/command.txt
```

Observed setup:

- `make_semantic_mask.py` infers phrase `rabbit eyes`.
- The anchor box is plausible, but the generated SAM mask covers about 57% of
  the image.
- Runtime `mask_area_guard_applied=True`; the mask is clipped to a coarse box
  around the eye/head region.
- No surface reference is used.
- The target object is very small relative to the image and partly ambiguous in
  side profile.

Why it fails:

- The edit needs precise accessory placement, but the auto mask is too coarse.
- Side-profile sunglasses have poor target geometry: one visible eye, small
  face area, and ear/grass structures competing with the edit.
- Text guidance and h-edit guidance can add a dark local shape, but they do not
  provide enough geometric constraint to place a recognizable pair of glasses.

Next experiments:

1. Use a manually drawn or prompted tight mask around the visible eye/bridge
   area.
2. Switch the task to a front-facing animal/person where glasses occupy a
   larger, symmetric face region.
3. Add a reference accessory image or source-reference injection only after
   the tight mask is verified.
4. Treat this side-profile rabbit task as a robustness failure, not as a main
   paper result.

## Paper Implication

The current evidence supports a narrow preservation claim on successful local
insertion, mostly `cat_crown`. It does not support a reliable local editing
claim across the planned matrix. Before paper use, add at least several
visually successful tasks with verified tight masks and task-specific edit
branches.
