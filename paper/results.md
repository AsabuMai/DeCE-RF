# Current Results

Date: 2026-05-28

Scope: core-5 DeCE-RF matrix with fixed per-task evaluation masks and an
internal visual audit. This replaces the older core-4 result summary. As of
2026-05-30, the planned final main benchmark is Core-6 by adding one recolor
task after a seed-10 visual gate.

## Protocol

Main paper-facing methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

`support_v3_fixed` is an internal ablation/control only. It should not appear as
a headline main-table method.

All headline preservation metrics use a fixed per-task evaluation mask shared
across methods and seeds. A method's own support mask is used only for support
diagnostics, not for headline preservation metrics.

`tshirt_star` used seed 10 as a visual-gate/calibration seed. The selected
`clothing_decal` preset was frozen before evaluating seeds 11 and 12.

## Artifacts

Quantitative metrics:

```text
experiments/support_v3_2026-05-11/core5_seed10_12_fixed_mask_metrics.csv
experiments/support_v3_2026-05-11/core5_seed10_12_fixed_mask_metrics.json
experiments/support_v3_2026-05-11/core5_fixed_mask_audit_summary.csv
experiments/support_v3_2026-05-11/core5_fixed_mask_audit_summary.md
```

Visual audit:

```text
experiments/support_v3_2026-05-11/core5_visual_audit_filled.csv
experiments/support_v3_2026-05-11/core5_visual_audit_summary.md
```

Qualitative grids:

```text
experiments/support_v3_2026-05-11/paper_grids/core5_main_seed10_grid.png
experiments/support_v3_2026-05-11/paper_grids/core5_main_seed11_grid.png
experiments/support_v3_2026-05-11/paper_grids/core5_main_seed12_grid.png
```

## Coverage

Main matrix:

```text
5 tasks x 4 paper-facing methods x 3 seeds = 60/60 complete
```

Planned final main matrix:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72 runs
```

The sixth task should be one localized recolor / attribute-editing case:

```text
preferred: red_chair_blue
fallback: red_office_chair_to_blue_office_chair
```

Current fixed-control ablation cache:

```text
4 core tasks x support_v3_fixed x 3 seeds = 12/12 complete
```

The fixed-control ablation currently covers the original core-4 task set. It is
reported as component evidence, not as the core-5 main comparison.

## Fixed-Mask Main Matrix

Values are means over seeds 10, 11, and 12. `Edit score` is CLIP
target-source delta. It is useful for add/decal tasks but can be misleading for
removal, so removal conclusions rely on visual audit. DINO/source can also
underestimate preservation when a large local decal changes global image
embedding; interpret it with outside-mask drift and visual audit.

| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cat_crown | RF reconstruction / base reconstruction | 3 | 0.0812 | 0.1207 | 0.5122 | 0.5048 | -0.0042 |
| cat_crown | Direct target guidance | 3 | 0.1150 | 0.4203 | 0.4358 | 0.4225 | 0.0007 |
| cat_crown | Generic support control | 3 | 0.0554 | 0.1422 | 0.6577 | 0.8281 | -0.0020 |
| cat_crown | DeCE-RF | 3 | 0.0559 | 0.0884 | 0.6504 | 0.9677 | 0.1018 |
| dog_sunglasses | RF reconstruction / base reconstruction | 3 | 0.0662 | 0.0551 | 0.6137 | 0.9067 | -0.0251 |
| dog_sunglasses | Direct target guidance | 3 | 0.1031 | 0.4330 | 0.5465 | 0.6598 | 0.0581 |
| dog_sunglasses | Generic support control | 3 | 0.0536 | 0.0581 | 0.6338 | 0.9279 | 0.0966 |
| dog_sunglasses | DeCE-RF | 3 | 0.0534 | 0.1326 | 0.6236 | 0.9592 | 0.0842 |
| mug_heart | RF reconstruction / base reconstruction | 3 | 0.0129 | 0.0359 | 0.9590 | 0.8820 | -0.0287 |
| mug_heart | Direct target guidance | 3 | 0.0246 | 0.2839 | 0.9138 | 0.8306 | -0.0276 |
| mug_heart | Generic support control | 3 | 0.0081 | 0.0080 | 0.9654 | 0.9599 | -0.0274 |
| mug_heart | DeCE-RF | 3 | 0.0105 | 0.0079 | 0.9574 | 0.8114 | 0.0425 |
| tshirt_star | RF reconstruction / base reconstruction | 3 | 0.0220 | 0.0128 | 0.8889 | 0.9379 | -0.0070 |
| tshirt_star | Direct target guidance | 3 | 0.0417 | 0.0152 | 0.8446 | 0.8672 | -0.0006 |
| tshirt_star | Generic support control | 3 | 0.0292 | 0.0101 | 0.8648 | 0.9089 | -0.0094 |
| tshirt_star | DeCE-RF | 3 | 0.0175 | 0.0657 | 0.8780 | 0.6217 | 0.0799 |
| backpack_remove_toy_charm | RF reconstruction / base reconstruction | 3 | 0.0657 | 0.0892 | 0.6146 | 0.8077 | 0.0056 |
| backpack_remove_toy_charm | Direct target guidance | 3 | 0.0755 | 0.2125 | 0.5619 | 0.7595 | -0.0078 |
| backpack_remove_toy_charm | Generic support control | 3 | 0.0354 | 0.0913 | 0.7960 | 0.9477 | -0.0077 |
| backpack_remove_toy_charm | DeCE-RF | 3 | 0.0397 | 0.2201 | 0.7369 | 0.8845 | -0.0164 |

## Internal Visual Audit

This is an internal visual audit, not a user study. Scores use a 1-5 scale. For
`edit_success`, `source_preservation`, `locality`, and `overall`, higher is
better. For `artifact_severity`, higher is worse.

Method means:

| Method | n | Edit success | Source preservation | Locality | Artifact severity | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF reconstruction / base reconstruction | 15 | 1.00 | 2.40 | 2.80 | 2.60 | 1.00 |
| Direct target guidance | 15 | 2.60 | 1.40 | 1.80 | 3.40 | 2.00 |
| Generic support control | 15 | 1.60 | 4.40 | 4.20 | 1.20 | 2.40 |
| DeCE-RF | 15 | 4.40 | 4.00 | 4.20 | 1.60 | 4.40 |

Task x method readout:

| Task | Method | Edit success | Preservation | Locality | Artifact | Overall | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cat_crown | Generic support control | 1.00 | 4.00 | 4.00 | 1.00 | 2.00 | over-preserves; crown absent |
| cat_crown | DeCE-RF | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | clear localized crown |
| dog_sunglasses | Generic support control | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | good result with moderate preservation |
| dog_sunglasses | DeCE-RF | 4.00 | 4.00 | 4.00 | 1.00 | 4.00 | localized glasses with strong preservation |
| mug_heart | Generic support control | 1.00 | 5.00 | 5.00 | 1.00 | 2.00 | excellent preservation but heart absent |
| mug_heart | DeCE-RF | 5.00 | 4.00 | 5.00 | 1.00 | 5.00 | clean localized decal |
| tshirt_star | Generic support control | 1.00 | 4.00 | 4.00 | 1.00 | 2.00 | preserves source but misses red star |
| tshirt_star | DeCE-RF | 5.00 | 4.00 | 4.00 | 2.00 | 5.00 | clear red clothing decal with strong preservation |
| backpack_remove_toy_charm | Generic support control | 1.00 | 5.00 | 4.00 | 1.00 | 2.00 | preserves source but fails removal |
| backpack_remove_toy_charm | DeCE-RF | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | yellow charm removed; backpack, patch, pink strap, and hardware preserved |

## Fixed-Control Ablation

`support_v3_fixed` isolates decoupled clean-estimate displacement with
operation-conditioned support but without feedback-updated control. This
ablation currently covers the original core-4 tasks.

| Task | Fixed edit score | DeCE-RF edit score | Fixed outside L1 | DeCE-RF outside L1 | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| cat_crown | 0.0921 | 0.1018 | 0.0568 | 0.0559 | RMSGAP improves edit score with comparable preservation. |
| dog_sunglasses | 0.0896 | 0.0842 | 0.0538 | 0.0534 | RMSGAP does not improve CLIP score, but visual audit rates artifact lower. |
| mug_heart | 0.0420 | 0.0425 | 0.0106 | 0.0105 | Small stable improvement. |
| backpack_remove_toy_charm | -0.0180 | -0.0164 | 0.0403 | 0.0397 | Slight metric improvement; visual audit marks DeCE-RF as successful removal despite weak global CLIP. |

## Interpretation

- DeCE-RF is strongest on localized add/decal tasks: `cat_crown`,
  `mug_heart`, and `tshirt_star` all show the requested local edit where generic
  support often over-preserves.
- `tshirt_star` adds a clothing-surface decal case with natural folds and
  shadows. The frozen clothing-decal preset is stable across seeds 10/11/12.
- `dog_sunglasses` is a competitive accessory case: DeCE-RF and generic support
  both score well, with DeCE-RF showing lower artifact severity.
- `backpack_remove_toy_charm` is a successful localized exposed-object removal
  under visual audit; global CLIP/edit score underestimates removal completion.
- `dog_remove_tennis_ball` remains outside the main table as a limitation:
  occluded-object removal requiring host-mouth completion is not solved.
- Additional seed-10 surface-removal gates (`laptop_remove_sticker`,
  `fridge_remove_yellow_magnet`, `fridge_remove_peach_magnet`, and
  `whiteboard_remove_yellow_letter`) should also remain outside the main table.
  Their support masks are accurate, but the local generator leaves residual
  labels/marks, changes the target into another object, or damages nearby
  objects. These runs support a limitation statement: localization can succeed
  while object erasure / surface completion remains unsolved.
- `laptop_remove_sticker` is now reserved for a separate high-confidence
  completion-prior extension probe, not the base DeCE-RF main table.
- `whiteboard_probe_red_star_sticker` is reserved for a separate non-glyph
  replacement probe. The blank/T/A whiteboard probes remain diagnostic evidence
  for glyph-field hallucination and weak precise glyph control.
- Direct target guidance is the most aggressive editor but has the worst source
  preservation and artifact scores.

## Claim Boundary

The current evidence supports:

```text
Operation-conditioned support plus decoupled clean-estimate edit-preserve
control improves the edit-preserve balance on selected localized add-object,
surface/clothing decal, exposed-object removal, and, pending tomorrow's gate,
localized recolor/attribute editing tasks.
```

The current evidence does not support:

```text
broad arbitrary removal/replacement
occluded-object removal requiring substantial host completion
precise glyph replacement
large standalone gains from feedback control over fixed DeCE displacement
state-of-the-art general-purpose image editing
```

## Next Evidence Needed

Decision after the 2026-05-29 and 2026-05-30 visual gates:

1. Do not keep searching for a surface sticker/magnet/letter removal task to
   force into the main matrix. The recent gates show a repeatable target
   completion limitation rather than a support-localization failure.
2. Treat the current 60-run core-5 matrix as the completed base evidence.
   Because five tasks is thin for a paper submission, expand the main matrix
   only through one controlled recolor / attribute-editing task.
3. Do not promote weak replacement cases such as `dog_replace_tennis_ball_star`,
   `backpack_replace_patch_blue`, or `cat_replace_bell_heart_tag` into the
   core table.
4. Report `laptop_remove_sticker` as a high-confidence completion-prior
   extension probe and `whiteboard_probe_red_star_sticker` as a non-glyph
   replacement probe.
5. Keep `whiteboard_remove_yellow_letter`, `dog_remove_tennis_ball`, fridge
   removals, and precise glyph replacement as limitations/diagnostics.
6. The next required controlled experiment is the recolor seed-10 gate. If it
   passes, expand to seeds 10/11/12 and complete `support_v3_fixed` for the
   final Core-6 task set as component evidence.
