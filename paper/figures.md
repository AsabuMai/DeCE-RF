# Figure Plan

Current source of truth: `paper/wacv_experiment_design.md`.

The main paper should use five figures by default and six at most. Target
about 50-70 result image cells total. More than that starts to read as a
gallery and weakens the algorithmic story.

Completed legacy server evidence is documented in `paper/archive_old_core6_20260602/old_core6_server_results.md`.
The active strict Core-6 evidence is now the revised category-based Phase 1
matrix under `experiments/support_v3_2026-06-02/`: attached accessory,
container-constrained spatial insertion, surface decal, local recolor, surface
material strip editing, and simple exposed-object removal. Use the archived
server grids only as supplementary diagnostics; they are not the source of
truth for the updated strict T2/T5 rows.

Use only complete runs with `result.png`, `stats.json`, `metadata.json`, and
`command.txt`.

## Main-Paper Figure Budget

| Figure | Content | Approx. result cells | Role |
| --- | --- | ---: | --- |
| Figure 1 | teaser: two examples, Source/Target/Direct/Generic/DeCE-RF | 10 | motivation |
| Figure 2 | method overview | 0 | explain the algorithm |
| Figure 3 | E1 Core-6 qualitative grid | 24-30 | main effect |
| Figure 4 | E2-A SD3-matched RF baseline qualitative comparison | 15 | matched RF alternatives |
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
T1 attached accessory: cat_crown
T2 container-constrained spatial insertion: bowl_apple_inside
T3 surface decal/logo: tshirt_star
T4 local recolor: red_chair_blue
T5 surface material strip edit: pillow_vertical_fabric_strip
T6 simple exposed removal: backpack_remove_toy_charm
```

Current generated strict evidence and audit grids:

```text
experiments/support_v3_2026-06-02/visual_audit/
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
```

Paper-use guidance:

- `cat_crown`: use as the attached-accessory success case.
- `dog_sunglasses`: diagnostic only; DeCE-RF eyewear placement is too high for a strong figure.
- `tshirt_star`: use as the strict surface-decal success case.
- `mug_heart`: diagnostic only; visually clean but too small/weak for the main grid.
- `bowl_apple_inside`: use as the strict T2 insertion row.
- `pillow_vertical_fabric_strip`: use as the strict T5 surface-material success case; DeCE-RF adds a perspective-aligned blue silk strip with clean top and bottom boundaries.
- `backpack_remove_toy_charm`: use as exposed-object removal success with a
  caveat that global CLIP underestimates removal quality.
- `red_chair_blue`: server evidence supports it as the localized
  attribute/recolor row; do not claim general recoloring.
- `red_office_chair_to_blue_office_chair`: fallback only if `red_chair_blue`
  fails final visual audit.

Do not include `support_v3_fixed` in this main qualitative grid unless the
layout still fits. Fixed DeCE belongs mainly in the controller ablation figure
and table.

## E2-A SD3-Matched RF Baseline Figure

Status: SD3-matched target-mode RF comparison is complete for FlowEdit,
FlowAlign, and SplitFlow. Do not present it as a complete comparison against
every RF or FLUX baseline; FireFlow, RF-Solver-Edit, ReFlex, and stable-flow are
native-backbone contextual rows blocked by FLUX.1-dev checkpoint access or
missing strict adapter output.

Current E2 audit artifacts:

```text
experiments/support_v3_2026-06-02/e2_baseline_download_registry.csv
experiments/support_v3_2026-06-02/e2_baseline_runnable_validation.csv
experiments/support_v3_2026-06-02/e2_baseline_audit.md
```

As of the 2026-06-03 audit, 14 external repositories are downloaded and 2
additional E2-B candidates are registered as planned entries. FlowEdit,
FlowAlign, and SplitFlow have revised strict target-mode outputs for 6 tasks x
seeds 10/11/12 and are available as the SD3-matched E2-A comparison. RF-Solver-
Edit (`rf_solver_edit`), ReFlex (`reflex`), FireFlow (`fireflow`), and
stable-flow (`stable_flow`) are native-backbone FLUX contextual rows blocked by
gated FLUX.1-dev access or adapter gaps; OT-RF / OTIP (`ot_rf_otip`) and DVRF
(`dvrf`) are planned contextual candidates that still need repo/env/adapter
validation.

Current SD3-matched figure grids:

```text
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed10_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed11_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed12_grid.png
```

Figure 4 can now be a compact SD3-matched E2-A comparison:

```text
Source | FlowEdit | FlowAlign | SplitFlow | DeCE-RF
```

Use three to six strict rows depending on space. Label it explicitly as a
SD3-matched target-mode RF comparison, not as a broad RF/FLUX victory claim.

Use two or three representative strict examples:

```text
Source | external target-mode RF baseline | Direct target | DeCE-RF
```

This figure should be compact because the main E2 evidence is the SD3-matched
Table 2a plus the native-backbone contextual audit/status table.

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
tshirt_star
```

Panels:

```text
attention evidence | clean disagreement | velocity disagreement |
operation-conditioned support | M_edit/M_core | M_preserve
```

## Feedback Figure

Use `cat_crown`, `tshirt_star`, or `pillow_vertical_fabric_strip`, comparing:

```text
support_v3_fixed | DeCE-RF across stress levels
```

Curves:

```text
edit-preserve Pareto frontier
edit target gap
preserve drift
adaptive edit weight
adaptive preserve weight
projection norm / ratio
```

Do not make the feedback claim from a single fixed-vs-full comparison. The
figure should show either a Pareto frontier or timestep trajectories.

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
