# E5 Boundary, Extension, And Failure Cases

Date: 2026-06-04

Selected outputs: 36/36 complete.
Figure 6 candidate: `experiments/support_v3_2026-06-02/e5_boundary_extension/e5_figure6_boundary_extension_seed10.png`

## Scope

- Positive extensions are reported separately from the Core-6 main table.
- Failure rows use controlled labels and support limitation wording.
- These rows are not aggregated into the base DeCE-RF mean.

## Selected Rows

| Task | Route | Seeds | Category | Failure/extension label |
| --- | --- | ---: | --- | --- |
| `laptop_remove_sticker` | base_dece_rf | 3/3 | positive_extension | high-confidence completion prior |
| `fridge_remove_yellow_magnet` | base_dece_rf | 3/3 | failure_limit | cluttered-surface damage |
| `fridge_remove_peach_magnet` | base_dece_rf | 3/3 | failure_limit | cluttered-surface damage |
| `whiteboard_remove_yellow_letter` | base_dece_rf | 3/3 | failure_limit | semantic glyph hallucination |
| `dog_remove_tennis_ball` | base_dece_rf | 3/3 | failure_limit | removal completion failure |
| `dog_replace_tennis_ball_star` | base_dece_rf | 3/3 | failure_limit | replacement ambiguity |
| `laptop_remove_sticker` | completion_prior_route | 3/3 | positive_extension | high-confidence completion prior |
| `fridge_remove_yellow_magnet` | completion_prior_route | 3/3 | failure_limit | cluttered-surface damage |
| `fridge_remove_peach_magnet` | completion_prior_route | 3/3 | failure_limit | cluttered-surface damage |
| `whiteboard_remove_yellow_letter` | completion_prior_route | 3/3 | failure_limit | semantic glyph hallucination |
| `dog_remove_tennis_ball` | completion_prior_route | 3/3 | failure_limit | removal completion failure |
| `whiteboard_probe_red_star_sticker` | replacement_target_route | 3/3 | positive_extension | replacement target route |

Paper-safe wording: E5 documents where the method extends or stops. It should be used for Figure 6 and limitations, not as a main-table quantitative claim.
