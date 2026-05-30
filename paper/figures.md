# Figure Plan

Current source of truth: core-5 results in `paper/results.md`, with the final
experiment design updated to Core-6 after one recolor visual gate.

Use only complete runs with `result.png`, `stats.json`, `metadata.json`, and
`command.txt`.

## Main Figures

Primary qualitative grid:

```text
Source | Direct target | Generic support | DeCE-RF | Support overlay
```

Rows:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
recolor task: red_chair_blue or red_office_chair_to_blue_office_chair
```

Current generated core-5 grids:

```text
experiments/support_v3_2026-05-11/paper_grids/core5_main_seed10_grid.png
experiments/support_v3_2026-05-11/paper_grids/core5_main_seed11_grid.png
experiments/support_v3_2026-05-11/paper_grids/core5_main_seed12_grid.png
```

Paper-use guidance:

- `cat_crown`: use as compact insertion success.
- `dog_sunglasses`: use as positive control / edit-preserve tradeoff.
- `mug_heart`: use as clean rigid-surface decal success.
- `tshirt_star`: use as clothing-surface decal generalization.
- `backpack_remove_toy_charm`: use as exposed-object removal success with a
  caveat that global CLIP underestimates removal quality.
- `red_chair_blue` / `red_office_chair_to_blue_office_chair`: use only after
  the seed-10 recolor gate passes; its role is localized attribute editing.

## Extension Probe Figure

Rows:

```text
laptop_remove_sticker
whiteboard_probe_red_star_sticker
```

Columns:

```text
Source | Base DeCE-RF | Extension route | Support / gate annotation
```

Paper-use guidance:

- `laptop_remove_sticker`: show high-confidence completion clean-delta as a
  planar removal extension. Label it as `DeCE-RF + completion prior`, not as
  the base DeCE-RF method.
- `whiteboard_probe_red_star_sticker`: show non-glyph replacement in a
  semantic letter field. Label it as `DeCE-RF + replacement route`.

## Support Figure

Recommended row:

```text
mug_heart or tshirt_star
```

Panels:

```text
attention evidence | clean disagreement | velocity disagreement |
operation-conditioned support | M_edit/M_core | M_preserve
```

## Feedback Figure

Use `cat_crown`, `dog_sunglasses`, or `mug_heart`, comparing:

```text
support_v3_fixed | DeCE-RF
```

Curves:

```text
edit target gap
preserve drift
adaptive edit weight
adaptive preserve weight
projection norm / ratio
```

## Limitation Figure

Keep failures tied to the current claim boundary:

```text
dog_remove_tennis_ball: occluded-object removal / host completion failure
whiteboard_remove_yellow_letter: glyph-field hallucination under blank removal
dog_replace_tennis_ball_star: partial replacement with mixed target/source residual
fridge magnet removals: accurate support but cluttered-surface completion damage
```

Do not reuse the old backpack-blue/yellow-car/rabbit-sunglasses panels as
current main-result evidence unless they are explicitly labeled as legacy
diagnostics.
