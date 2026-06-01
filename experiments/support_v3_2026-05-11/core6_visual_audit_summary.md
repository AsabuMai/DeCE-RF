# Core-6 Internal Visual Audit

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
