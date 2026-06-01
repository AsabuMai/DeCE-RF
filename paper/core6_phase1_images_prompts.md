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
| T1 | Attached accessory addition | `dog_sunglasses` primary; `cat_crown` secondary/teaser | dog: `data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg`; cat: `data/paper_images/cat_sitting_in_grass.jpg` | existing runner tasks |
| T2 | Container-constrained spatial insertion | `bowl_apple_inside` | needs an empty-bowl source image | new task needed |
| T3 | Surface decal / logo addition | `mug_heart` | `data/pretty_free_candidates/pexels_white_mug_6312107.jpg` | existing runner task |
| T4 | Local recoloring | `red_chair_blue` | `data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg` | existing candidate, visual audit required |
| T5 | Surface pattern editing | `pillow_blue_stripes` | web pillow candidate | new task needed |
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
dog_sunglasses
```

Source image:

```text
data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg
```

Source prompt:

```text
A front-facing portrait of a dog in snow.
```

Target prompt:

```text
A front-facing portrait of the same dog wearing black sunglasses in snow.
```

Attention / changed words:

```text
attention: sunglasses,eyes
changed: sunglasses
```

Why this task:

```text
Tests contact-aware accessory insertion on a host face while preserving the dog
identity, pose, snow, and background.
```

Secondary / teaser runner task:

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
Treat cat_crown as a T1 attached-accessory/contact example. It is useful for
the teaser and supplement, but it should not be used as the final T2 spatial
insertion definition.
```

## T2 Container-Constrained Spatial Insertion

Primary new task id:

```text
bowl_apple_inside
```

Required source image:

```text
An empty white bowl on a wooden table, with a simple background and no fruit
already inside the bowl.
```

Proposed local path after download:

```text
data/pretty_free_candidates/pexels_empty_white_bowl_table_phase1.jpg
```

Source prompt:

```text
A photo of an empty white bowl on a wooden table.
```

Target prompt:

```text
A photo of the same white bowl with a small red apple inside it on the wooden
table, with the bowl shape, table, and background unchanged.
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
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=small red apple inside the bowl
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
The current `inside` relation in operation_support_v3 is closer to host-mask
support than a true container-interior mask. Before final E1/E4 runs, inspect
the generated support masks. If the mask covers the bowl rim/exterior too much,
add a container-interior refinement or keep this as a T2 diagnostic rather than
headline evidence.
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
mug_heart
```

Source image:

```text
data/pretty_free_candidates/pexels_white_mug_6312107.jpg
```

Source prompt:

```text
A minimalist photo of a plain white ceramic mug on a grey background.
```

Target prompt:

```text
A minimalist photo of the same white ceramic mug with a small red heart printed
on the front, on the same grey background.
```

Attention / changed words:

```text
attention: heart,mug,front
changed: heart
```

Why this task:

```text
Tests surface-local marking while preserving object boundary, mug geometry,
background, and lighting.
```

Phase 2 expansion:

```text
tshirt_star
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

## T5 Surface Pattern Editing

T5 should be treated as a local surface-pattern edit, not full material
transfer. It uses the same broad operation family as T3, but with a larger
surface pattern support.

Runner task:

```text
pillow_blue_stripes
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
A photo of the same plain pillow with subtle blue stripes on its surface,
with the pillow shape, sofa, and background unchanged.
```

Attention / changed words:

```text
attention: blue,stripes,pillow
changed: blue,stripes
```

Relation label:

```text
add_decal + on_surface
```

Support configuration:

```text
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=blue,stripes
SUPPORT_HOST_TOKENS=pillow
SUPPORT_V3_RELATION=on_surface
DECAL_SHAPE=stripes
DECAL_COLOR=blue
DECAL_BOX=fixed medium/broad surface box
```

Why this task:

```text
Tests medium-size surface pattern editing. This is larger than the T3 heart
decal, but it should still preserve the pillow outline, sofa, room layout,
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
3. Add runner task definitions for `bowl_apple_inside` and
   `pillow_blue_stripes`.
4. Run seed-10 only for T2 and T5 with DeCE-RF first.
5. Promote only visually stable T2/T5 tasks into Phase 1 E1/E4 matrices.
6. Keep `cat_crown` as a T1 attached-accessory/teaser example, not as T2.
7. Keep `tshirt_star` as a T3 fallback/supplementary example, not as the final
   T5 definition.
