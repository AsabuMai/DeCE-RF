# Core-6 Phase 1 Images And Prompts

Current source of truth for experiment scale and phase gates:

```text
paper/wacv_experiment_design.md
```

This file fixes the first-pass image/prompt plan for Phase 1. The goal is not
to build the final benchmark yet. The goal is to run about 170 outputs and
decide whether DeCE-RF has enough signal to justify Phase 2.

## License Notes

Prefer already-downloaded local images with known provenance. For new web
assets, prefer Pexels or Unsplash pages with clear license pages. Pexels states
that its photos/videos are free to use, attribution is not required, and
modification is allowed. Unsplash similarly grants free use under its license,
but attribution is still recommended in the supplement.

Web sources checked:

```text
Pexels license:
https://www.pexels.com/license/

T2 needs a new empty-bowl source candidate:
empty white bowl on wooden table, no fruit inside

Pexels T5 pillow candidates:
https://www.pexels.com/photo/cushions-on-sofa-16451330/
https://www.pexels.com/photo/pillow-on-sofa-13177992/
https://www.pexels.com/photo/photo-of-pillows-on-sofa-4635231/
```

## Phase 1 Canonical Set

| Core ID | Category | Phase 1 task id | Source image | Status |
| --- | --- | --- | --- | --- |
| T1 | Attached accessory addition | `cat_crown` | `data/paper_images/cat_sitting_in_grass.jpg` | visual pass; replaces dog sunglasses for main strict set |
| T2 | Container-constrained spatial insertion | `bowl_apple_inside` | `data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg` | implemented; seed-10 visual gate passed |
| T3 | Surface decal / logo addition | `tshirt_star` | `data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg` | visual pass; replaces mug heart for main strict set |
| T4 | Local recoloring | `red_chair_blue` | `data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg` | existing candidate, visual audit required |
| T5 | Surface material strip editing | `pillow_vertical_fabric_strip` | `data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg` | implemented; seeds 10/11/12 human visual gate passed |
| T6 | Simple exposed-object removal | `backpack_remove_toy_charm` | `data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg` | existing runner task |

Temporary fallback:

```text
If T2 cannot be implemented quickly, do not move `cat_crown` into T2. Keep
`cat_crown` as a T1 attached-accessory/contact example and label Phase 1 as
missing the container-constrained spatial-insertion case until
`bowl_apple_inside` or a similar `add_object + inside` task is ready.

If T5 cannot be implemented quickly, use `tshirt_star` only as a second T3
decal diagnostic and label Phase 1 as Core-5 + T3 expansion, not full Core-6.
```

## T1 Attached Accessory Addition

Primary runner task:

```text
cat_crown
```

Source image:

```text
data/paper_images/cat_sitting_in_grass.jpg
```

Source prompt:

```text
A photo of a cat sitting in grass.
```

Target prompt:

```text
A photo of the same cat sitting in the same grass, wearing a small golden crown
on its head.
```

Attention / changed words:

```text
attention: crown,head
changed: crown
```

Use:

```text
Treat cat_crown as the strict Phase 1 T1 attached-accessory/contact example.
Dog_sunglasses remains diagnostic only because the quick audit found the DeCE-RF
eyewear placement too high for a strong main-paper row.
```

## T2 Container-Constrained Spatial Insertion

Primary new task id:

```text
bowl_apple_inside
```

Required source image:

```text
An empty ceramic bowl with a clearly visible interior and no fruit already inside the bowl. The current implementation-gate asset is a blue ceramic bowl in a top-down table setting; use a white-bowl replacement only if final paper wording requires it.
```

Proposed local path after download:

```text
data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg
```

Source prompt:

```text
A top-down photo of an empty blue ceramic bowl on a wooden board in a tidy table setting, with no fruit inside the bowl.
```

Target prompt:

```text
A top-down photo of the same blue ceramic bowl on the same wooden board, with one small red apple centered inside the bowl, while the bowl, board, tableware, leaves, and background remain unchanged.
```

Attention / changed words:

```text
attention: apple,bowl,inside
changed: apple
```

Relation label:

```text
add_object + inside
```

Support configuration:

```text
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=apple
SUPPORT_HOST_TOKENS=bowl
SUPPORT_V3_RELATION=inside_container
SEMANTIC_PHRASE=blue ceramic bowl
```

Why this task:

```text
Tests spatial insertion into an existing container using a relation that the
current support constructor already implements. The edit region should be the
bowl interior/free-space region, while the bowl rim, bowl exterior, table, and
background are preserved.
```

Implementation caveat:

```text
The implementation uses `inside_container`, a narrower container-interior relation derived from the grounded bowl mask. The seed-10 gate passed human visual review: the red apple is centered inside the bowl and surrounding objects remain stable. Keep this relation frozen for Phase 1 unless a better final source image replaces the current blue-bowl gate asset.
```

Do not use these as main T2 until relation constructors are implemented:

```text
next_to
beside
near
on_desk
```

Do not use `cat_crown` as the T2 backup. If `bowl_apple_inside` is blocked,
mark T2 as missing in Phase 1 and keep `cat_crown` under T1.

## T3 Surface Decal / Logo Addition

Runner task:

```text
tshirt_star
```

Source image:

```text
data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg
```

Source prompt:

```text
A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting.
```

Target prompt:

```text
The same person wearing the same white t-shirt and blue jeans, with a clearly visible medium-sized bright red star printed on the center chest, while preserving the fabric folds, shadows, jeans, pose, and background.
```

Attention / changed words:

```text
attention: star,t-shirt,chest
changed: star
```

Why this task:

```text
Tests surface-local clothing decal insertion while preserving shirt silhouette, fabric folds, pose, jeans, and background. Mug_heart remains diagnostic only because the main edit was visually too small for the strict grid.
```

Phase 2 expansion:

```text
mug_heart
tote_leaf
```

## T4 Local Recoloring

Runner task:

```text
red_chair_blue
```

Source image:

```text
data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg
```

Source prompt:

```text
A photo of a red armless rounded upholstered chair in a stylish room.
```

Target prompt:

```text
A photo of the same armless rounded upholstered chair in the same stylish room,
with only the fabric color changed to deep blue, no armrests added.
```

Attention / changed words:

```text
attention: chair,blue
changed: chair
```

Why this task:

```text
Tests low-geometry appearance editing: color should change while chair shape,
fabric structure, room layout, plants, floor, and wall remain stable.
```

Gate:

```text
Keep only if visual audit confirms local chair recolor rather than scene-wide
style drift.
```

## T5 Surface Material Strip Editing

T5 should be treated as a local surface material-strip edit, not full material
transfer. It uses the same broad operation family as T3, but with a larger
perspective-aligned surface support and a visible fabric/material change.

Runner task:

```text
pillow_vertical_fabric_strip
```

Preferred web source candidates:

```text
https://www.pexels.com/photo/cushions-on-sofa-16451330/
https://www.pexels.com/photo/pillow-on-sofa-13177992/
```

Proposed local path after download:

```text
data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg
```

Source prompt:

```text
A cozy living room photo with a plain neutral pillow on a sofa, soft natural
light, and a simple home interior.
```

Target prompt:

```text
A photo of the same plain grey pillow with one vertical glossy blue silk strip
sewn down the center of the pillow surface, tucked naturally into the top
pillow seam with soft satin highlights and smooth silk texture, while the
pillow shape, sofa, table, wall, and background remain unchanged.
```

Attention / changed words:

```text
attention: blue,silk,vertical,strip,pillow
changed: blue,silk,strip
```

Relation label:

```text
add_decal + on_surface
```

Support configuration:

```text
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_PRESET=surface_strip
SUPPORT_NEW_TOKENS=blue,silk,strip
SUPPORT_HOST_TOKENS=pillow
SUPPORT_V3_RELATION=on_surface
DECAL_SHAPE=slanted_rectangle
DECAL_COLOR=58,132,215
DECAL_BOX=0.565,0.445,0.665,0.688
DECAL_SLANT_X=-0.08
DECAL_PERSPECTIVE_Y=0.055
DECAL_EDGE_FEATHER_RADIUS=7.0
DECAL_TOP_FEATHER_FRAC=0.16
DECAL_TOP_FEATHER_MIN_ALPHA=0.0
DECAL_OPACITY=0.80
```

Why this task:

```text
Tests medium-size surface material editing. This is larger than the T3 decal,
but it should still preserve the pillow outline, sofa, room layout,
lighting, and background.
```

Avoid naming this task:

```text
full material edit
material transfer
ceramic-to-metallic
wood-to-plastic
```

Current implementation interpretation:

```text
add_decal-like surface pattern overlay
```

## T6 Simple Exposed-Object Removal

Runner task:

```text
backpack_remove_toy_charm
```

Source image:

```text
data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg
```

Source prompt:

```text
A close-up photo of a grey backpack with a yellow dangling toy charm attached
to a pink keychain strap.
```

Target prompt:

```text
A close-up photo of the same grey backpack with the yellow dangling toy charm
removed, pink strap, zipper, and fabric preserved.
```

Attention / changed words:

```text
attention: toy,charm,backpack,zipper,fabric
changed: toy,charm
```

Why this task:

```text
Tests exposed local removal without requiring large hidden-background or host
completion. Hard removal and occluded removal remain E5 boundary cases.
```

## First Implementation Checklist

1. Download and register one empty-bowl T2 source image.
2. Download and register one plain-pillow T5 source image.
3. Runner task definitions for `bowl_apple_inside` and
   `pillow_vertical_fabric_strip` is implemented.
4. Seed-10 DeCE-RF gates for T2 and T5 passed human visual review.
5. Keep the task definitions frozen for Phase 1 E1/E4 unless replacing the T2 source image intentionally.
6. Freeze `cat_crown` as T1 after replacing the weaker dog sunglasses row.
7. Freeze `tshirt_star` as T3 after replacing the weaker mug heart row; keep the perspective-aligned pillow silk strip as T5.
