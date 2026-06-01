# Figure Plan

Current source of truth: `paper/wacv_experiment_design.md`.

The main paper should use five figures by default and six at most. Target
about 50-70 result image cells total. More than that starts to read as a
gallery and weakens the algorithmic story.

Completed server evidence is documented in `paper/results.md`. The updated
Core-6 design is category-based: attached accessory, container-constrained
spatial insertion, surface decal, local recolor, surface pattern editing, and
simple exposed-object removal. The server grids cover T1/T3/T4/T6 evidence plus
supplementary T1/T3 examples, but not the updated strict T2/T5 rows.

Use only complete runs with `result.png`, `stats.json`, `metadata.json`, and
`command.txt`.

## Main-Paper Figure Budget

| Figure | Content | Approx. result cells | Role |
| --- | --- | ---: | --- |
| Figure 1 | teaser: two examples, Source/Target/Direct/Generic/DeCE-RF | 10 | motivation |
| Figure 2 | method overview | 0 | explain the algorithm |
| Figure 3 | E1 Core-6 qualitative grid | 24-30 | main effect |
| Figure 4 | E2 RF-native baseline qualitative comparison | 15 | current alternatives |
| Figure 5 | E4 Pareto + timestep diagnostics | 0 | controller evidence |
| Figure 6 | E5 extension/failure cases, optional | 12-18 | scope boundary |

Main-paper target:

```text
tight: 45-55 result image cells
complete: 60-75 result image cells
```

Supplement target:

```text
150-300 image cells: all seeds, full grids, support masks, RF baselines,
Pareto sweeps, and failure taxonomy.
```

## E1 Main Qualitative Grid

Primary qualitative grid:

```text
Source | Target/Instruction | Direct target | Generic support | DeCE-RF
```

Updated strict Core-6 target rows:

```text
T1 attached accessory: dog_sunglasses; cat_crown as secondary/teaser
T2 container-constrained spatial insertion: bowl_apple_inside
T3 surface decal/logo: mug_heart
T4 local recolor: red_chair_blue
T5 surface pattern edit: pillow_blue_stripes
T6 simple exposed removal: backpack_remove_toy_charm
```

Current generated server evidence grids:

```text
experiments/support_v3_2026-05-11/paper_grids/core6_main_seed10_grid.png
experiments/support_v3_2026-05-11/paper_grids/core6_main_seed11_grid.png
experiments/support_v3_2026-05-11/paper_grids/core6_main_seed12_grid.png
```

Paper-use guidance:

- `dog_sunglasses`: use as the attached-accessory success case.
- `cat_crown`: use as T1 secondary/teaser; do not use as the final T2 spatial
  insertion row.
- `mug_heart`: use as clean rigid-surface decal success.
- `tshirt_star`: server evidence supports this as a clothing/surface decal
  expansion. Do not relabel it as the final T5 row unless the paper narrows T5
  to clothing-decal evidence.
- `bowl_apple_inside`: updated T2 target, but not yet a successful generated
  row. Server bowl/apple probes remain diagnostic because placement drifts
  toward the rim/shadow.
- `pillow_blue_stripes`: updated T5 target, but not yet implemented or
  generated.
- `backpack_remove_toy_charm`: use as exposed-object removal success with a
  caveat that global CLIP underestimates removal quality.
- `red_chair_blue`: server evidence supports it as the localized
  attribute/recolor row; do not claim general recoloring.
- `red_office_chair_to_blue_office_chair`: fallback only if `red_chair_blue`
  fails final visual audit.

Do not include `support_v3_fixed` in this main qualitative grid unless the
layout still fits. Fixed DeCE belongs mainly in the controller ablation figure
and table.

## E2 RF Baseline Figure

Use three representative examples:

```text
Source | FlowEdit / RF baseline | OT-RF or RF-Solver | Direct target | DeCE-RF
```

This figure should be compact because the main E2 evidence is the RF-native
baseline table.

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
