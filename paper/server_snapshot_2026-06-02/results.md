# Current Results

Date: 2026-06-01

Scope: Core-6 DeCE-RF matrix with fixed per-task evaluation masks. This
supersedes the earlier Core-5 quantitative summary by adding one localized
recolor/attribute-editing task, `red_chair_blue`, after seed-10 visual gating
and mask correction.

## Protocol

Paper-facing methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

`support_v3_fixed` is retained as a component ablation/control only. It should
not appear as a headline main-table method.

All headline preservation metrics use fixed per-task evaluation masks shared
across methods and seeds. For `red_chair_blue`, the fixed mask is a seed-10
semantic surface mask with an 8-pixel left-boundary expansion at 368x512 to
include the chair's mixed upholstery boundary while preserving the right
boundary. This mask is independent of per-method support masks.

## Artifacts

Quantitative metrics:

```text
experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.csv
experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.json
experiments/support_v3_2026-05-11/core6_fixed_mask_audit_summary.csv
experiments/support_v3_2026-05-11/core6_fixed_mask_audit_summary.md
```

Fixed-control ablation:

```text
experiments/support_v3_2026-05-11/core6_fixed_control_metrics.csv
experiments/support_v3_2026-05-11/core6_fixed_control_metrics.json
experiments/support_v3_2026-05-11/core6_fixed_control_summary.csv
experiments/support_v3_2026-05-11/core6_fixed_control_summary.md
```

Qualitative grids:

```text
experiments/support_v3_2026-05-11/paper_grids/core6_main_seed10_grid.png
experiments/support_v3_2026-05-11/paper_grids/core6_main_seed11_grid.png
experiments/support_v3_2026-05-11/paper_grids/core6_main_seed12_grid.png
```

Visual audit:

```text
experiments/support_v3_2026-05-11/core6_visual_audit_template.csv
experiments/support_v3_2026-05-11/core6_visual_audit_filled.csv
experiments/support_v3_2026-05-11/core6_visual_audit_summary.md
```

## Coverage

Main matrix:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72/72 complete
```

Fixed-control ablation:

```text
6 tasks x 2 methods x 3 seeds = 36/36 complete
```

Core tasks:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
red_chair_blue
```

## Fixed-Mask Main Matrix

Fixed per-task evaluation masks are reused across all methods and seeds. Values are means over seeds 10, 11, and 12.

| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score | Inside blue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cat_crown | RF reconstruction / base reconstruction | 3 | 0.0831 | 0.0596 | 0.5122 | 0.5048 | -0.0042  0.0002 |
| cat_crown | Direct target guidance | 3 | 0.1251 | 0.1016 | 0.4358 | 0.4225 | 0.0007  0.0000 |
| cat_crown | Generic support control | 3 | 0.0595 | 0.0115 | 0.6577 | 0.8281 | -0.0020  0.1119 |
| cat_crown | DeCE-RF | 3 | 0.0542 | 0.1477 | 0.6504 | 0.9677 | 0.1018  0.0156 |
| dog_sunglasses | RF reconstruction / base reconstruction | 3 | 0.0615 | 0.0960 | 0.6137 | 0.9067 | -0.0251  0.0197 |
| dog_sunglasses | Direct target guidance | 3 | 0.1055 | 0.2187 | 0.5465 | 0.6598 | 0.0581  0.2859 |
| dog_sunglasses | Generic support control | 3 | 0.0422 | 0.1385 | 0.6338 | 0.9279 | 0.0966  0.1174 |
| dog_sunglasses | DeCE-RF | 3 | 0.0488 | 0.1192 | 0.6236 | 0.9592 | 0.0842  0.2181 |
| mug_heart | RF reconstruction / base reconstruction | 3 | 0.0142 | 0.0066 | 0.9590 | 0.8820 | -0.0287  0.0059 |
| mug_heart | Direct target guidance | 3 | 0.0349 | 0.0424 | 0.9138 | 0.8306 | -0.0276  0.0000 |
| mug_heart | Generic support control | 3 | 0.0082 | 0.0056 | 0.9654 | 0.9599 | -0.0274  0.0115 |
| mug_heart | DeCE-RF | 3 | 0.0076 | 0.0728 | 0.9574 | 0.8114 | 0.0425  0.0000 |
| tshirt_star | RF reconstruction / base reconstruction | 3 | 0.0220 | 0.0128 | 0.8889 | 0.9379 | -0.0070  0.5595 |
| tshirt_star | Direct target guidance | 3 | 0.0417 | 0.0152 | 0.8446 | 0.8672 | -0.0006  0.5808 |
| tshirt_star | Generic support control | 3 | 0.0292 | 0.0101 | 0.8648 | 0.9089 | -0.0094  0.3947 |
| tshirt_star | DeCE-RF | 3 | 0.0175 | 0.0657 | 0.8780 | 0.6217 | 0.0799  0.4941 |
| backpack_remove_toy_charm | RF reconstruction / base reconstruction | 3 | 0.0657 | 0.0892 | 0.6146 | 0.8077 | 0.0056  0.0000 |
| backpack_remove_toy_charm | Direct target guidance | 3 | 0.0755 | 0.2125 | 0.5619 | 0.7595 | -0.0078  0.0000 |
| backpack_remove_toy_charm | Generic support control | 3 | 0.0354 | 0.0913 | 0.7960 | 0.9477 | -0.0077  0.0000 |
| backpack_remove_toy_charm | DeCE-RF | 3 | 0.0397 | 0.2201 | 0.7369 | 0.8845 | -0.0164  0.0000 |
| red_chair_blue | RF reconstruction / base reconstruction | 3 | 0.1244 | 0.0874 | 0.3246 | 0.5454 | -0.0244  0.0000 |
| red_chair_blue | Direct target guidance | 3 | 0.1194 | 0.0959 | 0.3275 | 0.5603 | -0.0226  0.0000 |
| red_chair_blue | Generic support control | 3 | 0.0590 | 0.2238 | 0.5213 | 0.8872 | 0.0019  0.9036 |
| red_chair_blue | DeCE-RF | 3 | 0.0591 | 0.2251 | 0.5220 | 0.8857 | 0.0065  0.8981 |

## Fixed-Control Ablation

Fixed per-task evaluation masks are reused across all methods and seeds. Values are means over seeds 10, 11, and 12.

| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cat_crown | Fixed DeCE displacement | 3 | 0.0540 | 0.1838 | 0.6503 | 0.9667 | 0.0921 |
| cat_crown | DeCE-RF | 3 | 0.0542 | 0.1477 | 0.6504 | 0.9677 | 0.1018 |
| dog_sunglasses | Fixed DeCE displacement | 3 | 0.0487 | 0.1179 | 0.6227 | 0.9606 | 0.0896 |
| dog_sunglasses | DeCE-RF | 3 | 0.0488 | 0.1192 | 0.6236 | 0.9592 | 0.0842 |
| mug_heart | Fixed DeCE displacement | 3 | 0.0077 | 0.0749 | 0.9573 | 0.8074 | 0.0420 |
| mug_heart | DeCE-RF | 3 | 0.0076 | 0.0728 | 0.9574 | 0.8114 | 0.0425 |
| tshirt_star | Fixed DeCE displacement | 3 | 0.0175 | 0.0703 | 0.8773 | 0.6514 | 0.0772 |
| tshirt_star | DeCE-RF | 3 | 0.0175 | 0.0657 | 0.8780 | 0.6217 | 0.0799 |
| backpack_remove_toy_charm | Fixed DeCE displacement | 3 | 0.0403 | 0.2234 | 0.7392 | 0.8689 | -0.0180 |
| backpack_remove_toy_charm | DeCE-RF | 3 | 0.0397 | 0.2201 | 0.7369 | 0.8845 | -0.0164 |
| red_chair_blue | Fixed DeCE displacement | 3 | 0.0591 | 0.2241 | 0.5221 | 0.8896 | 0.0006 |
| red_chair_blue | DeCE-RF | 3 | 0.0591 | 0.2251 | 0.5220 | 0.8857 | 0.0065 |

## Internal Visual Audit

This is an internal visual audit, not a user study. Scores use a 1-5 scale. For `edit_success`, `source_preservation`, `locality`, and `overall`, higher is better. For `artifact_severity`, higher is worse.

Review grids:

- `paper_grids/core6_main_seed10_grid.png`
- `paper_grids/core6_main_seed11_grid.png`
- `paper_grids/core6_main_seed12_grid.png`

## Method Means

| Method | n | Edit success | Source preservation | Locality | Artifact severity | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF reconstruction / base reconstruction | 18 | 1.00 | 2.33 | 2.67 | 2.67 | 1.00 |
| Direct target guidance | 18 | 2.33 | 1.33 | 1.67 | 3.50 | 1.83 |
| Generic support control | 18 | 2.00 | 4.33 | 4.17 | 1.33 | 2.67 |
| DeCE-RF | 18 | 4.33 | 3.83 | 4.17 | 1.83 | 4.17 |

## Task x Method Means

| Task | Method | Edit success | Source preservation | Locality | Artifact severity | Overall | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cat_crown` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 3.00 | 3.00 | 1.00 | No crown and visible RF reconstruction drift |
| `cat_crown` | Direct target guidance | 1.00 | 1.00 | 1.00 | 4.00 | 1.00 | Strong identity and scene drift without useful crown formation |
| `cat_crown` | Generic support control | 1.00 | 4.00 | 4.00 | 1.00 | 2.00 | Source is preserved but crown does not form |
| `cat_crown` | DeCE-RF | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | Clear crown with source mostly preserved |
| `dog_sunglasses` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 3.00 | 3.00 | 1.00 | No glasses and noticeable reconstruction blur |
| `dog_sunglasses` | Direct target guidance | 5.00 | 1.00 | 2.00 | 4.00 | 3.00 | Strong glasses but dog identity and background drift heavily |
| `dog_sunglasses` | Generic support control | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | Glasses form with moderate preservation |
| `dog_sunglasses` | DeCE-RF | 4.00 | 4.00 | 4.00 | 1.00 | 4.00 | Localized glasses with better source preservation |
| `mug_heart` | RF reconstruction / base reconstruction | 1.00 | 3.00 | 3.00 | 2.00 | 1.00 | No heart and softened reconstruction |
| `mug_heart` | Direct target guidance | 1.00 | 1.00 | 2.00 | 3.00 | 1.00 | No heart and mug geometry changes |
| `mug_heart` | Generic support control | 1.00 | 5.00 | 5.00 | 1.00 | 2.00 | Excellent preservation but heart is absent |
| `mug_heart` | DeCE-RF | 5.00 | 4.00 | 5.00 | 1.00 | 5.00 | Clean localized heart decal with strong preservation |
| `backpack_remove_toy_charm` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 2.00 | 3.00 | 1.00 | Removal target not solved and backpack details drift |
| `backpack_remove_toy_charm` | Direct target guidance | 4.00 | 2.00 | 2.00 | 3.00 | 3.00 | Charm is mostly removed but backpack structure changes |
| `backpack_remove_toy_charm` | Generic support control | 1.00 | 5.00 | 4.00 | 1.00 | 2.00 | Preserves source but fails to remove charm |
| `backpack_remove_toy_charm` | DeCE-RF | 4.00 | 3.00 | 4.00 | 3.00 | 3.00 | Yellow dangling toy charm is removed and the upper cartoon patch is correctly retained; pink strap and metal hardware are mostly preserved, but the zipper/fabric behind the removed charm is locally smoothed and partially hallucinated. |
| `tshirt_star` | RF reconstruction / base reconstruction | 1.00 | 3.00 | 3.00 | 2.00 | 1.00 | No red star; RF reconstruction preserves broad layout but smooths shirt folds and body detail |
| `tshirt_star` | Direct target guidance | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | Tiny red mark appears, but the person/composition changes and the requested medium star is not formed |
| `tshirt_star` | Generic support control | 1.00 | 4.00 | 4.00 | 1.00 | 2.00 | Source is preserved, but the red star decal is absent |
| `tshirt_star` | DeCE-RF | 5.00 | 4.00 | 4.00 | 2.00 | 5.00 | Clearly visible red star on the chest with pose, jeans, background, and shirt structure mostly preserved |
| `red_chair_blue` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 2.00 | 3.00 | 1.00 | No blue recolor; RF reconstruction changes composition and misses the requested attribute edit. |
| `red_chair_blue` | Direct target guidance | 1.00 | 1.00 | 1.00 | 4.00 | 1.00 | Does not produce the requested blue chair and changes scene/composition substantially. |
| `red_chair_blue` | Generic support control | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | Chair becomes blue with good locality and source preservation; mild boundary/texture artifacts remain. |
| `red_chair_blue` | DeCE-RF | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | Localized blue recolor with source layout preserved; mild boundary/texture artifacts remain. |

## Interpretation

The Core-6 visual audit keeps the fixed rerun protocol and treats `red_chair_blue` as the promoted recolor/attribute-edit case. DeCE-RF remains the highest-overall method mean, driven by localized add/decal success and the chair recolor case. Generic support remains a strong preservation baseline, but it over-preserves on several add/decal tasks. Direct target guidance remains aggressive and often changes source identity or composition.

For `backpack_remove_toy_charm`, DeCE-RF removes the intended dangling yellow charm and preserves the upper cartoon patch, but the zipper/fabric region that was occluded by the removed charm is locally smoothed. This row should be discussed as a partial preservation artifact rather than a perfect removal example.

## Fixed-Control Ablation

`support_v3_fixed` isolates decoupled clean-estimate displacement with
operation-conditioned support but without feedback-updated control. This
ablation now covers the promoted Core-6 task set. The fixed-vs-feedback gap is modest, so this supports component evidence rather than a headline claim.

| Task | Fixed edit score | DeCE-RF edit score | Fixed outside L1 | DeCE-RF outside L1 | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| cat_crown | 0.0921 | 0.1018 | 0.0568 | 0.0559 | RMSGAP improves edit score with comparable preservation. |
| dog_sunglasses | 0.0896 | 0.0842 | 0.0538 | 0.0534 | RMSGAP does not improve CLIP score, but visual audit rates artifact lower. |
| mug_heart | 0.0420 | 0.0425 | 0.0106 | 0.0105 | Small stable improvement. |
| backpack_remove_toy_charm | -0.0180 | -0.0164 | 0.0403 | 0.0397 | Slight metric improvement; visual audit marks target charm removal as successful but notes local zipper/fabric smoothing. |

## Interpretation

- DeCE-RF is strongest on localized add/decal tasks: `cat_crown`,
  `mug_heart`, and `tshirt_star` all show the requested local edit where generic
  support often over-preserves.
- `tshirt_star` adds a clothing-surface decal case with natural folds and
  shadows. The frozen clothing-decal preset is stable across seeds 10/11/12.
- `dog_sunglasses` is a competitive accessory case: DeCE-RF and generic support
  both score well, with DeCE-RF showing lower artifact severity.
- `backpack_remove_toy_charm` removes the intended dangling charm, but the
  zipper/fabric region occluded by the charm is locally smoothed. This is a
  useful removal case with a preservation caveat, not a perfect completion
  example.
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
surface/clothing decal, exposed-object removal with a local completion caveat,
and localized recolor/attribute editing tasks.
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

After the 2026-06-01 Core-6 rerun:

1. Treat the internal Core-6 matrix as complete for the main DeCE-RF evidence
   package: 72 paper-facing runs plus the 18 fixed-control runs.
2. Keep weak replacement and difficult completion cases out of the main table.
   Use them only in limitation or extension-probe sections.
3. Report `laptop_remove_sticker` as a high-confidence completion-prior
   extension probe and `whiteboard_probe_red_star_sticker` as a non-glyph
   replacement probe, not as base DeCE-RF rows.
4. The next experimental priority is baseline coverage under the Core-6
   protocol: first same-support masked inpainting / masked img2img, then one
   RF-native baseline block such as FlowEdit or RF inversion + target
   resampling.
5. Existing controller stress results are enough for a modest feedback-control
   claim. Add more stress runs only if feedback becomes a reviewer-critical
   point.
