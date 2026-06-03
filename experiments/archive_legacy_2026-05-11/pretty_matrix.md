# Pretty Matrix Task Manifest

Date: 2026-05-09

Purpose: replace one-off `easy_demo` probes with a small, reproducible task set
for method debugging before any paper-scale run.

## Tasks

| ID | Name | Edit type | Source image | License note | Source prompt | Target prompt |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | `cat_crown` | accessory insertion | `data/paper_images/cat_sitting_in_grass.jpg` | Wikimedia Commons, CC BY-SA 3.0; attribution/share-alike required | A photo of a cat sitting in grass. | A photo of the same cat sitting in the same grass, wearing a small golden crown on its head. |
| P2 | `dog_sunglasses` | accessory insertion | `data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg` | Unsplash free-to-use license; not CC0 | A front-facing portrait of a dog in snow. | A front-facing portrait of the same dog wearing black sunglasses in snow. |
| P3 | `mug_heart` | local decal / surface logo | `data/pretty_free_candidates/pexels_white_mug_6312107.jpg` | Pexels free-to-use license; not CC0 | A minimalist photo of a plain white ceramic mug on a grey background. | A minimalist photo of the same white ceramic mug with a small red heart printed on the front, on the same grey background. |
| P4 | `red_chair_blue` | surface recolor / limitation | `data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg` | Pexels free-to-use license; not CC0; limitation case | A photo of a red armless rounded upholstered chair in a stylish room. | A photo of the same armless rounded upholstered chair in the same stylish room, with only the fabric color changed to deep blue, no armrests added. |
| P5 | `tshirt_star` | local decal / surface logo | `data/pretty_free_candidates/pexels_white_tshirt_mockup_12025472.jpg` | Pexels free-to-use license; not CC0; main-candidate if visual audit passes | A product photo of a plain white t-shirt on a light grey background. | A product photo of the same white t-shirt with a red star printed on the chest, on the same light grey background. |
| P6 | `tote_leaf` | local decal / surface logo | `data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg` | Pexels free-to-use license; not CC0; supplement/backup only after leaf reference fix | A photo of a plain white canvas tote bag held in front of a green wall. | A photo of the same white canvas tote bag with a green leaf logo printed on the front, in front of the same green wall. |
| P7 | `backpack_remove_toy_charm` | semantic object removal | `data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg` | Unsplash free-to-use license; not CC0; main-candidate after visual audit | A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap. | A close-up photo of the same grey backpack with the yellow dangling toy charm removed, pink strap, zipper, and fabric preserved. |
| P8 | `backpack_replace_patch_blue` | semantic object replacement / attribute-local edit | `data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg` | Unsplash free-to-use license; not CC0; supplemental replacement stress test | A close-up photo of a grey backpack with a colorful cartoon patch on the front pocket. | A close-up photo of the same grey backpack with the colorful cartoon patch replaced by a plain blue fabric patch, zipper and fabric preserved. |

## Method Smoke Matrix

Before full experiments, run a seed-10 smoke matrix over the recommended main
tasks. As of the 2026-05-09 visual audit, P4 should not be a main success case
because pure surface recoloring is better handled by deterministic color
transforms and the RF/ODE edit can introduce structure/texture drift.

```bash
TASKS="P1 P2 P3 P7" METHODS="M0 M1 M4" SEEDS="10" DEVICE=4 \
  bash scripts/run_pretty_matrix.sh
```

Use P4 as limitation/appendix, P5 only if the printed star becomes visually
cleaner, P6 only as a supplementary logo/decal backup after fixing the leaf
reference shape/placement, and P8 as a replacement stress test rather than a
main success case unless further visual audit across seeds is clean. Only expand
to seeds `10 11 12` and more methods after visual review confirms the `full`
route is acceptable on the selected main tasks.

## Go/No-Go Validation Matrix

Before moving this project into paper-writing mode, run the compact validation
matrix below. It is designed to test the narrowed claim: clean-estimate-space RF
control for localized image editing.

```bash
TASKS="P1 P2 P3 P7" METHODS="M0 M1 M5 M6 M7 M4" SEEDS="10 11 12" DEVICE=4 \
  bash scripts/run_pretty_matrix.sh
```

Method aliases:

| ID | Method | Purpose |
| --- | --- | --- |
| M0 | `base_only` | Source-conditioned baseline; should preserve but not edit. |
| M1 | `direct_target` | Strong target edit without the localized full route. |
| M5 | `full_no_ref` | Full support/mask route without local edit-reference guidance. |
| M6 | `full_no_rec` | Full route without reconstruction correction. |
| M7 | `full_no_traj` | Full route without trajectory preservation. |
| M4 | `full` | Complete localized method. |

Go/no-go gate:

1. `full` should be visually successful on at least 9 of 12 task/seed pairs.
2. `full` should reduce outside-mask drift relative to `direct_target` on most
   rows.
3. `full_no_ref`, `full_no_rec`, and `full_no_traj` should expose meaningful
   degradations that support the module claims.
4. At least one matched external baseline should be run on the same images and
   prompts before writing a paper.
5. Visual audit and functional-correctness scoring should accompany CLIP/DINO
   metrics.

Initialize the audit sheet with:

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/init_pretty_visual_audit.py \
  --output experiments/pretty_matrix_visual_audit_go_no_go.csv
```

## Task-Specific Routes

| Edit type | Current full-method route |
| --- | --- |
| accessory insertion | Semantic/SAM support for headwear; structure-derived eye mask for sunglasses. |
| local decal / surface logo | Fixed decal mask and composited decal reference from `scripts/make_decal_reference.py`; passed through `--final-edit-mask` and edit-reference guidance. |
| semantic object removal | Semantic/SAM support over the removable object; passed through `--final-edit-mask` with reconstruction and trajectory preservation to keep nearby context. |
| semantic object replacement / attribute-local edit | Semantic/SAM support over the source object plus an automatically generated replacement reference from `scripts/make_mask_badge_reference.py`; the current P8 route recolors the full semantic patch support into a plain blue fabric patch, while the script also keeps an ellipse/badge mode for harder replacement probes. |
| surface recolor | Task-specific color mask/box or Semantic/SAM object support plus masked surface recolor reference from `scripts/make_surface_recolor_reference.py`. |

## Paper Guard

These tasks are not yet paper-ready evidence. They are a cleaner validation
set for debugging mask/support and edit branches. Paper use requires:

1. Complete run records under `outputs/pretty_matrix/<task>/<method>/seed_<seed>/`.
2. A visual audit recording success/failure labels.
3. Metric regeneration with task/method/seed filters.
4. Explicit license notes in the manuscript or supplement.
