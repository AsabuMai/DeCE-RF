# Core-4 Internal Visual Audit

This is an internal visual audit, not a user study. Scores use a 1-5 scale.
For `edit_success`, `source_preservation`, `locality`, and `overall`, higher is
better. For `artifact_severity`, higher is worse.

Review grids:

- `paper_grids/core4_main_seed10_grid.png`
- `paper_grids/core4_main_seed11_grid.png`
- `paper_grids/core4_main_seed12_grid.png`

## Method Means

| Method | n | Edit success | Source preservation | Locality | Artifact severity | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF reconstruction / base reconstruction | 12 | 1.00 | 2.25 | 2.75 | 2.75 | 1.00 |
| Direct target guidance | 12 | 2.75 | 1.25 | 1.75 | 3.50 | 2.00 |
| Generic support control | 12 | 1.75 | 4.50 | 4.25 | 1.25 | 2.50 |
| DeCE-RF | 12 | 3.50 | 4.00 | 4.25 | 1.50 | 3.75 |

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
| `backpack_remove_toy_charm` | RF reconstruction / base reconstruction | 1.00 | 2.00 | 2.00 | 3.00 | 1.00 | removal not solved; source drifts |
| `backpack_remove_toy_charm` | Direct target guidance | 4.00 | 2.00 | 2.00 | 3.00 | 3.00 | charm mostly removed but structure changes |
| `backpack_remove_toy_charm` | Generic support control | 1.00 | 5.00 | 4.00 | 1.00 | 2.00 | preserves source but fails removal |
| `backpack_remove_toy_charm` | DeCE-RF | 1.00 | 4.00 | 4.00 | 2.00 | 2.00 | removal failure; charm remains |

## Interpretation

The visual audit matches the fixed-mask metrics. Direct target guidance is the
most aggressive editor but has the worst source preservation and artifact
scores. Generic support is a strong preservation baseline, but it often
over-preserves and misses the requested edit. DeCE-RF gives the best overall
balance on the add/decal tasks, especially `cat_crown` and `mug_heart`, while
`backpack_remove_toy_charm` should be reported as a removal boundary/failure
case rather than a success case.
