# Phase 2 Core-6 Images And Prompts

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-02
- Verification Status: PLANNED, image-fit screened only; no Phase 2 edit outputs inspected
- Version Label: phase2_core6_task_manifest_v0

## Purpose

This manifest prepares the Phase 2 Core-6 expansion:

```text
6 categories x 3 source examples x 4 internal methods x 3 seeds = 216 E1 outputs
```

The first row in each category keeps the Phase 1 canonical task. The second and
third rows expand the same operation family without changing the base DeCE-RF
controller. These rows should be frozen before Phase 2 output inspection.

## Selection Rules

- Prefer clean, localized edit regions with visible host geometry.
- Avoid tasks whose success requires unsupported relations such as `next_to`,
  `beside`, `near`, or `on_desk`.
- Use fixed operation/relation policies rather than per-image tuning.
- Use fixed evaluation masks for metrics only; do not pass evaluation masks to
  headline methods as editing inputs.
- Treat T5 as localized same-color material replacement: change local material identity while preserving color, geometry, folds, and shading.
- Treat T6 as simple exposed-object removal, not occluded host completion.

## Relation-Aware Support Policy

For DeCE-RF support construction, each edit is described by an operation, an
anchor phrase, and a reusable spatial relation:

```text
A = Ground(x, anchor)
S = Phi_r(A; theta_r)
```

Here `A` is the grounded anchor mask, `r` is a relation primitive, and `S` is
the edit support used by the controller. The relation registry maps task-level
relations to shared support geometry rather than per-image manual masks.

Supported Phase 2 relation primitives:

| Primitive | Intended use |
| --- | --- |
| `inside_object` / `inside` | object-level recolor or replacement |
| `inside_container` | object insertion into a bowl, cup, or container |
| `above_host` | crown, hat, or above-head accessory |
| `below_host` | bow tie, necklace, or below-head attachment |
| `on_surface` | decal, logo, surface pattern, or localized material panel |
| `on_face` | front-facing glasses, mask, or face-attached accessory |
| `on_profile_face` | side-profile visible-eye accessory |
| `remove_source_object` | exposed source-object removal |

## Phase 2 Task Table

| Core ID | Category | Phase 2 task id | Source image | Operation | Relation | Fit |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Attached accessory addition | `cat_crown` | `data/paper_images/cat_sitting_in_grass.jpg` | `add_object` | `above_host` | Phase 1 canonical; clean attached crown example |
| T1 | Attached accessory addition | `dog_bow_tie_phase2` | `data/paper_images/dog_sitting_cc0.jpg` | `add_object` | `below_host` | dog neck accessory; tests non-head attached accessory geometry |
| T1 | Attached accessory addition | `dog_front_sunglasses_phase2` | `data/pretty_free_candidates/commons_dog_front_medusa.jpg` | `add_object` | `on_face` | close-up front-facing glasses; tests face-attached accessory geometry |
| T2 | Container-constrained insertion | `bowl_apple_inside` | `data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg` | `add_object` | `inside_container` | Phase 1 canonical; visible bowl interior |
| T2 | Spatial object insertion | `white_bowl_orange_tabletop_phase2` | `data/phase2_candidates/pexels_sideview_bowl_wood_table_26161017.jpg` | `add_object` | `on_surface` | side-view tabletop placement with a larger visible fruit; replaces the too-small strawberry repair |
| T2 | Container-constrained insertion | `brown_bowl_lemon_phase2` | `data/phase2_candidates/pexels_wooden_bowls_topdown_6962757.jpg` | `add_object` | `inside_container` | top-down wooden bowl interior; preserve larger bowl and cloth |
| T3 | Surface decal / logo addition | `tshirt_star` | `data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg` | `add_decal` | `on_surface` | Phase 1 canonical; clear shirt surface |
| T3 | Surface decal / logo addition | `mug_heart` | `data/pretty_free_candidates/pexels_white_mug_6312107.jpg` | `add_decal` | `on_surface` | clean mug surface; useful compact decal expansion |
| T3 | Surface decal / logo addition | `tote_leaf` | `data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg` | `add_decal` | `on_surface` | flat tote panel; enlarged dark-green logo for visibility |
| T4 | Local recoloring | `red_office_chair_to_blue_office_chair` | `data/pretty_free_candidates/unsplash_red_office_chair_concrete_lvVWRzm_NwY.jpg` | `recolor` | `inside` | red-plastic-shell-to-blue office chair recolor; preserve concrete floor, wall, metal base, and wheels |
| T4 | Local recoloring | `green_mug_orange_phase2` | `data/pretty_free_candidates/pexels_green_mug_marble_7828522.jpg` | `recolor` | `inside` | green-to-orange mug recolor; preserve marble block and pink background |
| T4 | Local recoloring | `yellow_vase_blue_phase2` | `data/pretty_free_candidates/pexels_yellow_ceramic_vase_8356822.jpg` | `recolor` | `inside` | yellow-to-blue vase recolor; preserve white fabric, lighting, and shadows |
| T5 | Localized same-color material replacement | `pillow_same_color_corduroy_panel` | `data/phase2_candidates/pexels_white_pillow_brown_sofa_6312089.jpg` | `add_decal` | `on_surface` | canonical T5 replacement; same-color corduroy panel, not symbol/decal/recolor |
| T5 | Localized same-color material replacement | `pillow_same_color_linen_panel` | `data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg` | `add_decal` | `on_surface` | dark grey pillow on rattan-backed sofa; same-color woven linen panel |
| T5 | Localized same-color material replacement | `pillow_same_color_terry_panel` | `data/phase2_candidates/pexels_green_chair_white_pillow_6312055.jpg` | `add_decal` | `on_surface` | high-contrast white pillow on green chair; same-color terry cloth panel |
| T6 | Simple exposed-object removal | `backpack_remove_toy_charm` | `data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg` | `remove_object` | `remove_source_object` | Phase 1 canonical; exposed charm |
| T6 | Simple exposed-object removal | `backpack_remove_silver_keychain_phase2` | `data/phase2_candidates/pexels_backpack_keychain_35171863.jpg` | `remove_object` | `remove_source_object` | dark but localized keychain; needs support-mask gate |
| T6 | Simple exposed-object removal | `bag_remove_decorative_tag_phase2` | `data/phase2_candidates/pexels_bag_decorative_keychain_30853733.jpg` | `remove_object` | `remove_source_object` | visible decorative tag; simple exposed removal |

## Prompt And Support Definitions

### T1-1 `cat_crown`

Source prompt:

```text
A photo of a cat sitting in grass.
```

Target prompt:

```text
A photo of the same cat sitting in the same grass, wearing a small golden crown centered on top of its head between the ears.
```

Support:

```text
attention: crown,head
changed: crown
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=crown
SUPPORT_HOST_TOKENS=cat,head
SUPPORT_V3_RELATION=above_host
SEMANTIC_PHRASE=cat
```

### T1-2 `dog_bow_tie_phase2`

Source prompt:

```text
A photo of a shaggy grey and white dog sitting in grass.
```

Target prompt:

```text
A photo of the same shaggy grey and white dog sitting in the same grass, wearing a small red bow tie attached at the front of its neck, while the dog, grass, pose, and background remain unchanged.
```

Support:

```text
attention: bow,tie,neck,chest
changed: bow,tie
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=bow,tie
SUPPORT_HOST_TOKENS=dog head,neck,chest
SUPPORT_V3_RELATION=below_host
SEMANTIC_PHRASE=dog head
# relation registry: below_host -> below-head attachment support
```

### T1-3 `dog_front_sunglasses_phase2`

Source prompt:

```text
A close-up front-facing portrait of a dog indoors.
```

Target prompt:

```text
A close-up front-facing portrait of the same dog wearing black sunglasses aligned across both eyes indoors, while the dog face, ears, fur, nose, floor, and lighting remain unchanged.
```

Support:

```text
attention: sunglasses,eyes
changed: sunglasses
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=sunglasses
SUPPORT_HOST_TOKENS=dog head,eyes
SUPPORT_V3_RELATION=on_face
SEMANTIC_PHRASE=dog head
```

### T2-1 `bowl_apple_inside`

Source prompt:

```text
A top-down photo of an empty blue ceramic bowl on a wooden board in a tidy table setting, with no fruit inside the bowl.
```

Target prompt:

```text
A top-down photo of the same blue ceramic bowl on the same wooden board, with one small red apple centered inside the bowl, while the bowl, board, tableware, leaves, and background remain unchanged.
```

Support:

```text
attention: apple,bowl,inside
changed: apple
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=apple
SUPPORT_HOST_TOKENS=bowl,ceramic bowl
SUPPORT_V3_RELATION=inside_containe
SEMANTIC_PHRASE=blue ceramic bowl
```

### T2-2 `white_bowl_orange_tabletop_phase2`

Source prompt:

```text
A side-view photo of a white bowl sitting on a wooden tabletop, with empty wooden table space to the left of the bowl.
```

Target prompt:

```text
A side-view photo of the same white bowl sitting on the same wooden tabletop, with one medium bright orange fruit resting on the empty wooden table space to the left of the bowl, while the bowl, tabletop, wall, lighting, and background remain unchanged.
```

Support:

```text
attention: orange,fruit,wooden,table,left
changed: orange,fruit
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=orange,fruit
SUPPORT_HOST_TOKENS=wooden tabletop,empty wooden table space,table
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=wooden tabletop
```

### T2-3 `brown_bowl_lemon_phase2`

Source prompt:

```text
A top-down photo of a small empty wooden bowl on a pale wooden table beside a larger wooden bowl and folded cloth.
```

Target prompt:

```text
A top-down photo of the same small wooden bowl on the same pale wooden table, with one small yellow lemon wedge centered inside the small bowl, while the bowls, table, folded cloth, lighting, and background remain unchanged.
```

Support:

```text
attention: lemon,bowl,inside
changed: lemon
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=lemon,wedge
SUPPORT_HOST_TOKENS=bowl,ceramic bowl
SUPPORT_V3_RELATION=inside_containe
SEMANTIC_PHRASE=small wooden bowl
```

### T3-1 `tshirt_star`

Source prompt:

```text
A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting.
```

Target prompt:

```text
The same person wearing the same white t-shirt and blue jeans, with a clearly visible medium-sized bright red star printed on the center chest, while preserving the fabric folds, shadows, jeans, pose, and background.
```

Support:

```text
attention: star,t-shirt,chest
changed: sta
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=sta
SUPPORT_HOST_TOKENS=t-shirt,shirt
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=t-shirt
DECAL_SHAPE=sta
DECAL_COLOR=red
TASK_EDIT_STRENGTH_MULTIPLIER=1.5
# formal seed-10 path: no explicit decal_reference/final decal mask for this row
```

### T3-2 `mug_heart`

Source prompt:

```text
A minimalist photo of a plain white ceramic mug on a grey background.
```

Target prompt:

```text
A minimalist photo of the same white ceramic mug with a small flat hard-edged solid red heart decal printed cleanly on the front, with sharp printed edges and no glow or blur, while the mug shape, handle, highlights, bottom shadow, grey background, and lighting remain unchanged.
```

Support:

```text
attention: heart,mug,front
changed: heart
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=heart
SUPPORT_HOST_TOKENS=mug
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=mug
DECAL_SHAPE=heart
DECAL_COLOR=red
SUPPORT_PRESET=localized_decal
DECAL_BOX=0.390,0.415,0.535,0.570
DECAL_OPACITY=1.0
TASK_DECAL_REF_GUIDANCE_SCALE=0.55
TASK_DECAL_REF_LUMA_PRESERVE=0.00
TASK_DECAL_REF_GRADIENT_PRESERVE=0.00
TASK_DECAL_REF_SMOOTH_KERNEL=1
TASK_COMPLETION_CLEAN_DELTA_SCALE=1.80
```

### T3-3 `tote_leaf`

Source prompt:

```text
A photo of a plain beige canvas tote bag held in front of a dark green wall.
```

Target prompt:

```text
A photo of the same beige canvas tote bag with a large centered dark green leaf logo printed clearly on the front panel, while the hand, straps, bag shape, wall, and lighting remain unchanged.
```

Support:

```text
attention: leaf,logo,tote,bag
changed: leaf,logo
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=leaf,logo
SUPPORT_HOST_TOKENS=tote,bag
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=tote bag
DECAL_SHAPE=leaf
DECAL_COLOR=green
```

### T4-1 `red_office_chair_to_blue_office_chair`

Source prompt:

```text
A photo of a red office chair on a concrete floor.
```

Target prompt:

```text
A photo of the same office chair on the same concrete floor, with only the red plastic seat and back shell changed to deep blue while the silver metal base, wheels, wall, and floor remain unchanged.
```

Support:

```text
attention: chair,blue
changed: chai
SUPPORT_EDIT_OPERATION=recolo
SUPPORT_NEW_TOKENS=blue
SUPPORT_HOST_TOKENS=chair,red chair,office chair,plastic shell
SUPPORT_REMOVED_TOKENS=chai
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=chai
RECOLOR_SOURCE_COLOR=red
RECOLOR_SOURCE_MASK_THRESHOLD=0.20
RECOLOR_SOURCE_KEEP_COMPONENTS=1
RECOLOR_SOURCE_MIN_AREA=200
RECOLOR_SOURCE_DILATE_KERNEL=7
RECOLOR_SOURCE_DILATE_ITERATIONS=1
RECOLOR_SOURCE_DILATE_MIN_SATURATION=0.12
RECOLOR_FILL_HOLES=0
RECOLOR_TARGET_COLOR=0.03,0.16,0.78
RECOLOR_SURFACE_NAME=chai
RECOLOR_SURFACE_REFINE_ERODE_ITERATIONS=0
RECOLOR_SURFACE_REFINE_DILATE_KERNEL=0
RECOLOR_SURFACE_REFINE_DILATE_ITERATIONS=0
RECOLOR_REF_BLEND=1.0
RECOLOR_REF_MASK_BLUR=1
RECOLOR_REF_GUIDANCE_SCALE=0.58
RECOLOR_COLOR_GUIDANCE_SCALE=0.10
RECOLOR_TRAJECTORY_SUBJECT_PRESERVE_SCALE=0.10
MASK_LAYERING_MODE=recolor_trimap
RECOLOR_TRIMAP_BOUNDARY_EDIT_SCALE=0.8
RECOLOR_TRIMAP_BOUNDARY_PRESERVE_SCALE=0.0
```

### T4-2 `green_mug_orange_phase2`

Source prompt:

```text
A studio photo of a plain green ceramic mug on a white marble block against a pink background.
```

Target prompt:

```text
A studio photo of the same plain ceramic mug on the same white marble block against the same pink background, with only the mug color changed from green to orange.
```

Support:

```text
attention: mug,orange
changed: mug
SUPPORT_EDIT_OPERATION=recolo
SUPPORT_NEW_TOKENS=orange
SUPPORT_HOST_TOKENS=mug,green mug
SUPPORT_REMOVED_TOKENS=mug
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=mug
RECOLOR_TARGET_COLOR=orange
RECOLOR_SURFACE_NAME=mug
RECOLOR_SURFACE_REFINE_ERODE_ITERATIONS=0
RECOLOR_SURFACE_REFINE_DILATE_KERNEL=3
RECOLOR_SURFACE_REFINE_DILATE_ITERATIONS=1
RECOLOR_REF_BLEND=1.0
RECOLOR_REF_MASK_BLUR=1
RECOLOR_REF_GUIDANCE_SCALE=0.58
RECOLOR_COLOR_GUIDANCE_SCALE=0.10
RECOLOR_TRAJECTORY_SUBJECT_PRESERVE_SCALE=0.10
MASK_LAYERING_MODE=recolor_trimap
RECOLOR_TRIMAP_BOUNDARY_EDIT_SCALE=0.8
RECOLOR_TRIMAP_BOUNDARY_PRESERVE_SCALE=0.0
```

### T4-3 `yellow_vase_blue_phase2`

Source prompt:

```text
A still-life photo of a small yellow ceramic vase resting on white fabric.
```

Target prompt:

```text
A still-life photo of the same small ceramic vase resting on the same white fabric, with only the vase color changed from yellow to deep blue while the fabric, lighting, and shadows remain unchanged.
```

Support:

```text
attention: vase,blue
changed: vase
SUPPORT_EDIT_OPERATION=recolo
SUPPORT_NEW_TOKENS=blue
SUPPORT_HOST_TOKENS=vase,yellow vase
SUPPORT_REMOVED_TOKENS=vase
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=vase
RECOLOR_TARGET_COLOR=blue
RECOLOR_SURFACE_NAME=vase
RECOLOR_SURFACE_REFINE_ERODE_ITERATIONS=0
RECOLOR_SURFACE_REFINE_DILATE_KERNEL=2
RECOLOR_SURFACE_REFINE_DILATE_ITERATIONS=1
RECOLOR_REF_BLEND=1.0
RECOLOR_REF_MASK_BLUR=1
RECOLOR_REF_GUIDANCE_SCALE=0.54
RECOLOR_COLOR_GUIDANCE_SCALE=0.12
RECOLOR_TRAJECTORY_SUBJECT_PRESERVE_SCALE=0.30
MASK_LAYERING_MODE=recolor_trimap
RECOLOR_TRIMAP_BOUNDARY_EDIT_SCALE=0.8
RECOLOR_TRIMAP_BOUNDARY_PRESERVE_SCALE=0.10
```

### T5-1 `pillow_same_color_corduroy_panel`

Source prompt:

```text
A cozy living room photo with a plain white pillow on a brown sofa.
```

Target prompt:

```text
A photo of the same plain white pillow on the same brown sofa, with only the center vertical surface panel changed into same-color white corduroy fabric with fine vertical ribs, while preserving the pillow shape, folds, lighting, contour, sofa, background, and all surrounding pillow fabric.
```

Support:

```text
attention: corduroy,ribs,pillow,panel
changed: corduroy,ribs,panel
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_PRESET=material_panel
SUPPORT_NEW_TOKENS=corduroy,ribs,material
SUPPORT_HOST_TOKENS=white pillow,center panel
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=white pillow
DECAL_SHAPE=corduroy_panel
DECAL_BOX=0.390,0.310,0.560,0.700
DECAL_COLOR=128,128,128
```

### T5-2 `pillow_same_color_linen_panel`

Source prompt:

```text
A cozy living room photo with a plain dark grey pillow on a rattan-backed sofa beside a wooden table, soft natural light, and a simple home interior.
```

Target prompt:

```text
A photo of the same plain dark grey pillow on the same rattan-backed sofa, with only the center surface panel changed into same-color dark grey coarse woven linen, while preserving the pillow shape, seams, folds, lighting, sofa, table, wall, and all surrounding pillow fabric.
```

Support:

```text
attention: linen,woven,grey,pillow,panel
changed: linen,woven,panel
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_PRESET=material_panel
SUPPORT_NEW_TOKENS=linen,woven,material
SUPPORT_HOST_TOKENS=dark grey pillow,center panel
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=grey pillow
DECAL_SHAPE=linen_panel
DECAL_BOX=0.535,0.500,0.685,0.665
DECAL_COLOR=128,128,128
```

### T5-3 `pillow_same_color_terry_panel`

Source prompt:

```text
A photo of a plain white throw pillow centered on a green velvet armchair against a tiled wall.
```

Target prompt:

```text
A photo of the same plain white throw pillow on the same green velvet armchair, with only the center rectangular surface panel changed into same-color white terry cloth texture, while preserving the pillow shape, seams, shadows, chair, tiled wall, floor, and all surrounding pillow fabric.
```

Support:

```text
attention: terry,cloth,texture,pillow,panel
changed: terry,cloth,panel
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_PRESET=material_panel
SUPPORT_NEW_TOKENS=terry,cloth,material
SUPPORT_HOST_TOKENS=white pillow,center panel
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=white pillow
DECAL_SHAPE=terry_panel
DECAL_BOX=0.475,0.295,0.625,0.505
DECAL_COLOR=128,128,128
```

Retired T5 pattern probes: `pillow_vertical_fabric_strip`, `white_pillow_blue_dots_phase2`, and `white_pillow_blue_cross_phase2` are no longer formal Core-6 T5 tasks because they are too close to T3 symbolic decal insertion or T4 color/appearance change.

### T6-1 `backpack_remove_toy_charm`

Source prompt:

```text
A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap.
```

Target prompt:

```text
A close-up photo of the same grey backpack with the yellow dangling toy charm removed, pink strap, zipper, and fabric preserved.
```

Support:

```text
attention: toy,charm,backpack,zipper,fabric
changed: toy,charm
SUPPORT_EDIT_OPERATION=remove_object
SUPPORT_REMOVED_TOKENS=toy,charm
SUPPORT_HOST_TOKENS=backpack
SUPPORT_V3_RELATION=remove_source_object
SEMANTIC_PHRASE=yellow toy charm
```

### T6-2 `backpack_remove_silver_keychain_phase2`

Source prompt:

```text
A close-up photo of a dark backpack with a small silver keychain hanging from the front.
```

Target prompt:

```text
A close-up photo of the same dark backpack with the small silver keychain removed, while the backpack fabric, zipper, and dark background remain unchanged.
```

Support:

```text
attention: silver,keychain,backpack
changed: keychain
SUPPORT_EDIT_OPERATION=remove_object
SUPPORT_REMOVED_TOKENS=silver,keychain
SUPPORT_HOST_TOKENS=backpack
SUPPORT_V3_RELATION=remove_source_object
SEMANTIC_PHRASE=silver keychain
```

### T6-3 `bag_remove_decorative_tag_phase2`

Source prompt:

```text
A close-up photo of a black bag strap with a small decorative hanging tag attached.
```

Target prompt:

```text
A close-up photo of the same black bag strap with the small decorative hanging tag removed, while the strap, bag, and background remain unchanged.
```

Support:

```text
attention: decorative,tag,strap,bag
changed: decorative,tag
SUPPORT_EDIT_OPERATION=remove_object
SUPPORT_REMOVED_TOKENS=decorative,tag
SUPPORT_HOST_TOKENS=bag,strap
SUPPORT_V3_RELATION=remove_source_object
SEMANTIC_PHRASE=decorative tag
```

## Fit Risks To Check Before Full Phase 2

- `dog_bow_tie_phase2`: repaired seed-10 support uses a `dog head` anchor plus
  a below-head neck band; inspect seed stability before expansion.
- `dog_front_sunglasses_phase2`: close-up front-facing face support should be checked
  for large face redraw or sunglasses drifting off the eye band.
- `white_bowl_orange_tabletop_phase2`: the original top-down bowl source was
  rejected because the strawberry edit damaged the bowl geometry. A late
  side-view strawberry repair preserved geometry but the object was too small
  for a paper-facing figure. The active source now uses a larger orange fruit
  on the side-view wooden tabletop and should be checked for selecting the
  wooden tabletop region rather than the bowl body.
- `white_pillow_blue_cross_phase2`: multiple pillows may confuse grounding.
  Run a support-only gate before full E1.
- `backpack_remove_silver_keychain_phase2`: the image is dark. Keep only if the
  support mask localizes the silver keychain.

## Preview Artifact

The current candidate preview grid was generated as:

```text
data/phase2_candidates/phase2_core6_selected_sheet.jpg
```
