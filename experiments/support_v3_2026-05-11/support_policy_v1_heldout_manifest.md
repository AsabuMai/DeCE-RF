# Support Policy V1 Held-Out Manifest

This manifest freezes a candidate held-out set before any held-out support or
editing results are inspected. The policy in `docs/support_policy_v1.md` must not
be changed for these cases unless the change is recorded as a general operation
rule and the held-out split is reset.

2026-05-12 update: recolor is retired from the main rmsgap evaluation scope.
Rows H6-H11 are therefore excluded from active held-out evaluation. Before any
held-out claim, reset this manifest with non-recolor replacement cases.

| ID | Image | Source prompt | Target prompt | Operation | Relation |
| --- | --- | --- | --- | --- | --- |
| H1 | `data/paper_images/cat_on_grass.jpg` | A photo of a cat standing in grass. | A photo of the same cat standing in grass, wearing a small golden crown on its head. | `add_object` | `above_host` |
| H2 | `data/pretty_free_candidates/pexels_black_cat_studio_14453017.jpg` | A studio photo of a black cat. | A studio photo of the same black cat wearing small black sunglasses. | `add_object` | `on_face` |
| H3 | `data/paper_images/woman_portrait_50062201233.jpg` | A portrait photo of a woman. | A portrait photo of the same woman wearing small black sunglasses. | `add_object` | `on_face` |
| H4 | `data/paper_images/herschel_backpack_by_rocks_unsplash.jpg` | A photo of a backpack near rocks. | A photo of the same backpack with a small blue fabric patch on the front pocket. | `add_decal` | `on_surface` |
| H5 | `data/paper_images/herschel_backpack_by_rocks_unsplash.jpg` | A photo of a backpack near rocks. | A photo of the same backpack with a red star logo printed on the front pocket. | `add_decal` | `on_surface` |
| H6 | `data/pretty_free_candidates/unsplash_white_sneakers_laces_32Yh622c0g.jpg` | A photo of white sneakers with white laces. | A photo of the same white sneakers with the laces changed to blue. | `recolor` | `inside` |
| H7 | `data/pretty_free_candidates/unsplash_white_grey_sneakers_laces_HyfBIObAA4Y.jpg` | A photo of white and grey sneakers with white laces. | A photo of the same sneakers with the laces changed to blue. | `recolor` | `inside` |
| H8 | `data/pretty_free_candidates/unsplash_grey_white_shoelace_2fjRiPbSiA.jpg` | A close-up photo of a grey shoe with white shoelaces. | A close-up photo of the same grey shoe with the shoelaces changed to blue. | `recolor` | `inside` |
| H9 | `data/pretty_free_candidates/pexels_red_chair_restaurant_32696868.jpg` | A photo of a red chair in a restaurant. | A photo of the same chair with the fabric changed to deep blue. | `recolor` | `inside` |
| H10 | `data/pretty_free_candidates/pexels_red_chair_white_bg_4172380.jpg` | A product photo of a red chair on a white background. | A product photo of the same chair with the fabric changed to deep blue. | `recolor` | `inside` |
| H11 | `data/paper_images/yellow_car_side_unsplash.jpg` | A side-view photo of a yellow car. | A side-view photo of the same car with the paint changed to blue. | `recolor` | `inside` |
| H12 | `data/paper_images/dog_sitting_cc0.jpg` | A photo of a dog sitting. | A photo of the same dog sitting, wearing a small golden crown on its head. | `add_object` | `above_host` |

Held-out run rule: do not edit this table after viewing held-out masks/results.
