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
- Treat T5 as surface-pattern editing, not full material transfer.
- Treat T6 as simple exposed-object removal, not occluded host completion.

## Phase 2 Task Table

| Core ID | Category | Phase 2 task id | Source image | Operation | Relation | Fit |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Attached accessory addition | `cat_crown` | `data/paper_images/cat_sitting_in_grass.jpg` | `add_object` | `above_host` | Phase 1 canonical; clean attached crown example |
| T1 | Attached accessory addition | `dog_crown_phase2` | `data/paper_images/dog_sitting_cc0.jpg` | `add_object` | `above_host` | visible head and open space above; good same-relation expansion |
| T1 | Attached accessory addition | `cat_side_crown_phase2` | `data/paper_images/cat_on_grass.jpg` | `add_object` | `above_host` | side-view cat; keeps T1 on the same above-head relation |
| T2 | Container-constrained insertion | `bowl_apple_inside` | `data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg` | `add_object` | `inside_container` | Phase 1 canonical; visible bowl interior |
| T2 | Container-constrained insertion | `white_bowl_strawberry_phase2` | `data/phase2_candidates/pexels_white_bowl_plate_6863362.jpg` | `add_object` | `inside_container` | white bowl interior; keeps T2 on container-style placement |
| T2 | Container-constrained insertion | `brown_bowl_lemon_phase2` | `data/phase2_candidates/pexels_brown_ceramic_bowl_black_surface_7236397.jpg` | `add_object` | `inside_container` | close-up bowl interior; preserve chopsticks and table |
| T3 | Surface decal / logo addition | `tshirt_star` | `data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg` | `add_decal` | `on_surface` | Phase 1 canonical; clear shirt surface |
| T3 | Surface decal / logo addition | `mug_heart` | `data/pretty_free_candidates/pexels_white_mug_6312107.jpg` | `add_decal` | `on_surface` | clean mug surface; useful compact decal expansion |
| T3 | Surface decal / logo addition | `tote_leaf` | `data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg` | `add_decal` | `on_surface` | flat tote panel; good larger host surface |
| T4 | Local recoloring | `red_chair_blue` | `data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg` | `recolor` | `inside` | Phase 1 canonical; chair-level recolor |
| T4 | Local recoloring | `red_chair_product_blue_phase2` | `data/pretty_free_candidates/pexels_red_chair_white_bg_4172380.jpg` | `recolor` | `inside` | simple product image; clean silhouette |
| T4 | Local recoloring | `red_chair_restaurant_blue_phase2` | `data/pretty_free_candidates/pexels_red_chair_restaurant_32696868.jpg` | `recolor` | `inside` | scene-context recolor; tests background preservation |
| T5 | Surface pattern editing | `pillow_vertical_fabric_strip` | `data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg` | `add_decal` | `on_surface` | Phase 1 canonical; perspective-aligned blue silk strip |
| T5 | Surface pattern editing | `white_pillow_blue_dots_phase2` | `data/phase2_candidates/pexels_white_pillow_brown_sofa_6312089.jpg` | `add_decal` | `on_surface` | clean single pillow; strong T5 expansion |
| T5 | Surface pattern editing | `white_pillow_blue_cross_phase2` | `data/phase2_candidates/pexels_white_pillow_sofa_13941365.jpg` | `add_decal` | `on_surface` | multiple pillows; target front/center white pillow only |
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

### T1-2 `dog_crown_phase2`

Source prompt:

```text
A photo of a shaggy grey and white dog sitting in grass.
```

Target prompt:

```text
A photo of the same shaggy grey and white dog sitting in the same grass, wearing a small golden crown centered on top of its head.
```

Support:

```text
attention: crown,head
changed: crown
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=crown
SUPPORT_HOST_TOKENS=dog,head
SUPPORT_V3_RELATION=above_host
SEMANTIC_PHRASE=dog
```

### T1-3 `cat_side_crown_phase2`

Source prompt:

```text
A side-view photo of a cat standing in green grass.
```

Target prompt:

```text
A side-view photo of the same cat standing in the same green grass, wearing a small golden crown centered on top of its head.
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
SUPPORT_V3_RELATION=inside_container
SEMANTIC_PHRASE=blue ceramic bowl
```

### T2-2 `white_bowl_strawberry_phase2`

Source prompt:

```text
A top-down photo of an empty white ceramic bowl on a round white plate on a marble table.
```

Target prompt:

```text
A top-down photo of the same white ceramic bowl on the same round white plate and marble table, with one small red strawberry centered inside the bowl.
```

Support:

```text
attention: strawberry,bowl,inside
changed: strawberry
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=strawberry
SUPPORT_HOST_TOKENS=bowl,white bowl,ceramic bowl
SUPPORT_V3_RELATION=inside_container
SEMANTIC_PHRASE=white ceramic bowl
```

### T2-3 `brown_bowl_lemon_phase2`

Source prompt:

```text
A close-up top-down photo of an empty brown ceramic bowl on a dark table with chopsticks nearby.
```

Target prompt:

```text
A close-up top-down photo of the same brown ceramic bowl on the same dark table, with one small yellow lemon wedge centered inside the bowl, while the bowl, chopsticks, and table remain unchanged.
```

Support:

```text
attention: lemon,bowl,inside
changed: lemon
SUPPORT_EDIT_OPERATION=add_object
SUPPORT_NEW_TOKENS=lemon,wedge
SUPPORT_HOST_TOKENS=bowl,ceramic bowl
SUPPORT_V3_RELATION=inside_container
SEMANTIC_PHRASE=brown ceramic bowl
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
changed: star
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=star
SUPPORT_HOST_TOKENS=t-shirt,shirt
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=t-shirt
DECAL_SHAPE=star
DECAL_COLOR=red
```

### T3-2 `mug_heart`

Source prompt:

```text
A minimalist photo of a plain white ceramic mug on a grey background.
```

Target prompt:

```text
A minimalist photo of the same white ceramic mug with a clear red heart printed on the front, on the same grey background.
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
```

### T3-3 `tote_leaf`

Source prompt:

```text
A photo of a plain beige canvas tote bag held in front of a dark green wall.
```

Target prompt:

```text
A photo of the same beige canvas tote bag with a medium-sized green leaf logo printed on the front panel, while the hand, wall, bag shape, and lighting remain unchanged.
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

### T4-1 `red_chair_blue`

Source prompt:

```text
A photo of a red armless rounded upholstered chair in a stylish room.
```

Target prompt:

```text
A photo of the same armless rounded upholstered chair in the same stylish room, with only the fabric color changed to deep blue, no armrests added.
```

Support:

```text
attention: chair,blue
changed: chair
SUPPORT_EDIT_OPERATION=recolor
SUPPORT_NEW_TOKENS=blue
SUPPORT_HOST_TOKENS=chair
SUPPORT_REMOVED_TOKENS=chair
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=chair
RECOLOR_TARGET_COLOR=blue
RECOLOR_SURFACE_NAME=chair
```

### T4-2 `red_chair_product_blue_phase2`

Source prompt:

```text
A product photo of a simple red chair on a plain white background.
```

Target prompt:

```text
A product photo of the same simple chair on the same plain white background, with only the chair color changed from red to deep blue.
```

Support:

```text
attention: chair,blue
changed: chair
SUPPORT_EDIT_OPERATION=recolor
SUPPORT_NEW_TOKENS=blue
SUPPORT_HOST_TOKENS=chair
SUPPORT_REMOVED_TOKENS=chair
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=chair
RECOLOR_TARGET_COLOR=blue
RECOLOR_SURFACE_NAME=chair
```

### T4-3 `red_chair_restaurant_blue_phase2`

Source prompt:

```text
A photo of a red wooden chair beside a restaurant table in a warm indoor room.
```

Target prompt:

```text
A photo of the same wooden chair beside the same restaurant table, with only the chair color changed to deep blue while the table, wall, floor, and lighting remain unchanged.
```

Support:

```text
attention: chair,blue
changed: chair
SUPPORT_EDIT_OPERATION=recolor
SUPPORT_NEW_TOKENS=blue
SUPPORT_HOST_TOKENS=chair
SUPPORT_REMOVED_TOKENS=chair
SUPPORT_V3_RELATION=inside
SEMANTIC_PHRASE=chair
RECOLOR_TARGET_COLOR=blue
RECOLOR_SURFACE_NAME=chair
```

### T5-1 `pillow_vertical_fabric_strip`

Source prompt:

```text
A cozy living room photo with a plain grey pillow on a sofa, soft natural light, and a simple home interior.
```

Target prompt:

```text
A photo of the same plain grey pillow with a single vertical blue silk fabric strip aligned to the pillow perspective, with the pillow shape, sofa, table, wall, and background unchanged.
```

Support:

```text
attention: blue,silk,strip,pillow
changed: blue,silk,strip
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=blue,silk,strip
SUPPORT_HOST_TOKENS=pillow
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=grey pillow
SUPPORT_PRESET=surface_strip
DECAL_SHAPE=strip
DECAL_COLOR=42,108,196
```

### T5-2 `white_pillow_blue_dots_phase2`

Source prompt:

```text
A living room photo with a plain white square pillow on a brown sofa, with soft indoor light.
```

Target prompt:

```text
A living room photo of the same plain white square pillow on the same brown sofa, with small blue polka dots printed across the pillow surface while the sofa and background remain unchanged.
```

Support:

```text
attention: blue,dots,pillow
changed: blue,dots
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=blue,dots
SUPPORT_HOST_TOKENS=pillow
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=white pillow
DECAL_SHAPE=dots
DECAL_COLOR=42,108,196
```

### T5-3 `white_pillow_blue_cross_phase2`

Source prompt:

```text
A cozy interior photo with white pillows on a sofa against a textured wall.
```

Target prompt:

```text
A cozy interior photo of the same sofa and wall, with a blue diagonal cross pattern printed on the front white pillow while the other pillows and background remain unchanged.
```

Support:

```text
attention: blue,cross,pillow
changed: blue,cross
SUPPORT_EDIT_OPERATION=add_decal
SUPPORT_NEW_TOKENS=blue,cross
SUPPORT_HOST_TOKENS=front pillow,white pillow,pillow
SUPPORT_V3_RELATION=on_surface
SEMANTIC_PHRASE=front white pillow
DECAL_SHAPE=cross
DECAL_COLOR=42,108,196
```

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

- `cat_side_crown_phase2`: side-view head geometry is less frontal than
  `cat_crown` and `dog_crown_phase2`; run a support-only gate before full E1.
- `white_bowl_strawberry_phase2`: the bowl is not centered in the source image;
  support should be checked to ensure it selects the bowl interior rather than
  the surrounding plate.
- `white_pillow_blue_cross_phase2`: multiple pillows may confuse grounding.
  Run a support-only gate before full E1.
- `backpack_remove_silver_keychain_phase2`: the image is dark. Keep only if the
  support mask localizes the silver keychain.

## Preview Artifact

The current candidate preview grid was generated as:

```text
data/phase2_candidates/phase2_core6_selected_sheet.jpg
```
