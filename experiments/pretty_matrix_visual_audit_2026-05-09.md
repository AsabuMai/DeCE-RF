# Pretty Matrix Visual Audit

Date: 2026-05-09

## Run

Background session:

```bash
TASKS="P1 P2 P3 P7" METHODS="M0 M1 M4" SEEDS="10 11 12" DEVICE=7 \
  SKIP_EXISTING=1 REGENERATE_MASKS=0 bash scripts/run_pretty_matrix.sh
```

Log:

```text
outputs/pretty_matrix/logs/rfedit_pretty_20260509_170946.log
```

Coverage after completion:

- `cat_crown`: M0/M1/M4, seeds 10/11/12.
- `dog_sunglasses`: M0/M1/M4, seeds 10/11/12.
- `mug_heart`: M0/M1/M4, seeds 10/11/12.
- `backpack_remove_toy_charm`: M0/M1/M4, seeds 10/11/12.

## Review Images

- `outputs/pretty_matrix/cat_crown_seed_method_review.png`
- `outputs/pretty_matrix/dog_sunglasses_seed_method_review.png`
- `outputs/pretty_matrix/mug_heart_seed_method_review.png`
- `outputs/pretty_matrix/backpack_remove_toy_charm_seed_method_review.png`
- `outputs/pretty_matrix/pretty_main4_seed10_review.png`
- `outputs/pretty_matrix/backpack_replace_patch_blue_seed10_review.png`
- `outputs/pretty_matrix/backpack_replace_patch_blue_full_seeds_review.png`

## Visual Decisions

| Task | Decision | Notes |
| --- | --- | --- |
| `cat_crown` | Main success | M4 preserves the original cat/background and inserts a stable crown across three seeds. M0/M1 visibly drift. |
| `dog_sunglasses` | Main / weak success | Automatic front-eye support is stable. Glasses are aligned but somewhat translucent. A stronger edit-reference/darkness setting was tested and reverted because it introduced blue glare and worse artifacts. |
| `mug_heart` | Main success | Heart decal is visible and stable across seeds; M4 preserves mug geometry better than M0/M1. |
| `backpack_remove_toy_charm` | Main success | M4 consistently removes the lower yellow toy charm while preserving the pink strap, zipper, fabric, and upper patch. This is stronger than adding another flat logo because it tests semantic removal. |
| `backpack_replace_patch_blue` | Supplemental replacement probe | M4 replaces the upper cartoon patch with a plain blue fabric patch across seeds 10/11/12 while preserving the lower keychain/toy and most backpack structure. Keep it as attribute-local replacement evidence rather than a full object-swap benchmark. |
| `red_chair_blue` | Limitation only | Pure recolor remains better handled by deterministic color transforms; do not use as a main success case. |
| `tshirt_star` | Exclude from main | The printed star remains blurry. |
| `tote_leaf` | Exclude from main | The generated leaf reference reads like a diamond/tag rather than a clean printed leaf. |

## Paper Framing

Use the main qualitative set as four complementary localized edits:

1. Accessory insertion: `cat_crown`.
2. Accessory insertion with automatic eye support/reference: `dog_sunglasses`.
3. Local printed decal: `mug_heart`.
4. Semantic object removal: `backpack_remove_toy_charm`.

Avoid claiming that the method solves general recoloring. Present `red_chair_blue`
as a limitation where a deterministic YUV-chroma transform is a strong
non-generative baseline.

## Follow-Up Adjustment Log

- `dog_sunglasses`: tested a no-highlight glasses reference to reduce blue glare
  and increase opacity. It did not improve the three-seed visual result and made
  some lenses look slightly bluish, so the original stable reference route was
  restored and rerun for seeds 10/11/12. Keep the task as a weak success rather
  than over-tuning it.
- `backpack_replace_patch_blue`: revised P8 from a circular badge swap to a
  cleaner attribute-local replacement. The semantic mask is automatic
  (`colorful cartoon patch`, SAM support area about 4.6%). The replacement
  reference is generated deterministically from the semantic mask by converting
  the full source patch support into a plain blue fabric patch. M4 is stable
  across seeds 10/11/12 and much more localized than M0/M1.
- 2026-05-10 replacement update: do not use the backpack image for final
  replacement because it is already the deletion example. A new Wikimedia
  Commons dog image was added under
  `data/replacement_candidates/commons_dog_with_tennis_ball.jpg`; current best
  replacement candidate is `dog_replace_tennis_ball_star`, reviewed at
  `outputs/pretty_matrix/dog_replace_tennis_ball_star_seed10_review.png`.
