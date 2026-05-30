# Core-5 Internal Visual Audit

This is an internal visual audit, not a user study. Scores use a 1-5 scale.
For `edit_success`, `source_preservation`, `locality`, and `overall`, higher is better. For `artifact_severity`, higher is worse.

Review grids:

- `paper_grids/core5_main_seed10_grid.png`
- `paper_grids/core5_main_seed11_grid.png`
- `paper_grids/core5_main_seed12_grid.png`

## Method Means

| Method | n | Edit success | Source preservation | Locality | Artifact severity | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF reconstruction / base reconstruction | 15 | 1.00 | 2.40 | 2.80 | 2.60 | 1.00 |
| Direct target guidance | 15 | 2.60 | 1.40 | 1.80 | 3.40 | 2.00 |
| Generic support control | 15 | 1.60 | 4.40 | 4.20 | 1.20 | 2.40 |
| DeCE-RF | 15 | 4.40 | 4.00 | 4.20 | 1.60 | 4.40 |

## Task x Method Means

| Task | Method | Edit success | Source preservation | Locality | Artifact severity | Overall | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cat_crown` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 3.00 | 3.00 | 1.00 | no crown; reconstruction drift |
| `cat_crown` | Direct target guidance | 1.00 | 1.00 | 1.00 | 4.00 | 1.00 | target guidance changes identity without useful crown |
| `cat_crown` | Generic support control | 1.00 | 4.00 | 4.00 | 1.00 | 2.00 | over-preserves; crown absent |
| `cat_crown` | DeCE-RF | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | clear localized crown |
| `dog_sunglasses` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 3.00 | 3.00 | 1.00 | no glasses; reconstruction blur |
| `dog_sunglasses` | Direct target guidance | 5.00 | 1.00 | 2.00 | 4.00 | 3.00 | strong glasses but large identity drift |
| `dog_sunglasses` | Generic support control | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | good result with moderate preservation |
| `dog_sunglasses` | DeCE-RF | 4.00 | 4.00 | 4.00 | 1.00 | 4.00 | localized glasses with strong preservation |
| `mug_heart` | RF reconstruction / base reconstruction | 1.00 | 3.00 | 3.00 | 2.00 | 1.00 | no heart; softened reconstruction |
| `mug_heart` | Direct target guidance | 1.00 | 1.00 | 2.00 | 3.00 | 1.00 | heart absent; mug geometry changes |
| `mug_heart` | Generic support control | 1.00 | 5.00 | 5.00 | 1.00 | 2.00 | excellent preservation but heart absent |
| `mug_heart` | DeCE-RF | 5.00 | 4.00 | 5.00 | 1.00 | 5.00 | clean localized decal |
| `tshirt_star` | RF reconstruction / base reconstruction | 1.00 | 3.00 | 3.00 | 2.00 | 1.00 | no red star; reconstruction smooths shirt |
| `tshirt_star` | Direct target guidance | 2.00 | 2.00 | 2.00 | 3.00 | 2.00 | tiny red mark but person/composition drift; medium star not formed |
| `tshirt_star` | Generic support control | 1.00 | 4.00 | 4.00 | 1.00 | 2.00 | preserves source but misses red star |
| `tshirt_star` | DeCE-RF | 5.00 | 4.00 | 4.00 | 2.00 | 5.00 | clear red clothing decal with strong preservation |
| `backpack_remove_toy_charm` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 2.00 | 3.00 | 1.00 | removal not solved; source drifts |
| `backpack_remove_toy_charm` | Direct target guidance | 4.00 | 2.00 | 2.00 | 3.00 | 3.00 | charm mostly removed but structure changes |
| `backpack_remove_toy_charm` | Generic support control | 1.00 | 5.00 | 4.00 | 1.00 | 2.00 | preserves source but fails removal |
| `backpack_remove_toy_charm` | DeCE-RF | 4.00 | 4.00 | 4.00 | 2.00 | 4.00 | yellow charm removed; surrounding backpack, patch, strap, and hardware preserved |

## Interpretation

The core-5 visual audit strengthens the main edit-preserve story. DeCE-RF has the highest overall mean after adding `tshirt_star`: it forms the requested red clothing decal while preserving pose, jeans, background, and most shirt structure. Direct target guidance remains the aggressive baseline but changes source composition and does not reliably form the medium star on `tshirt_star`. Generic support remains a strong preservation baseline, but it over-preserves and misses decal formation on `mug_heart` and `tshirt_star`.

`dog_remove_tennis_ball` remains outside this table as limitation evidence rather than a tuned success case.
