# E4 Controller And Robustness Ablation

Scope: cat_crown, tshirt_star, and pillow_vertical_fabric_strip.
Base fixed-vs-feedback rows use seeds 10/11/12. Edit-strength stress uses seed10 across multipliers 0.50, 0.75, 1.00, 1.25, 1.50, and 2.00.

Figure 5 Pareto: `experiments/support_v3_2026-06-02/e4_controller_ablation/e4_figure5_edit_strength_pareto.png`
Controller trajectory figure: `experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_tshirt_star_seed10.png`

## Base Component Table

| Variant | n | Outside L1 down | Inside L1 | Source SSIM up | Local edit L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed DeCE displacement | 9 | 0.0333 | 0.0720 | 0.9181 | 0.0720 |
| DeCE-RF | 9 | 0.0329 | 0.0672 | 0.9205 | 0.0672 |

## Trajectory Summary

| Variant | n | Preserve drift | Preserve weight | Edit weight | Projection norm | Preserve correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed DeCE displacement | 9 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| DeCE-RF | 9 | 0.2357 | 1.3154 | 1.1481 | 0.0543 | 3.9190 |

## Edit-Strength Stress

| Variant | Multiplier | n | Outside L1 down | Local edit L1 | Source SSIM up |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed DeCE displacement | 0.50 | 3 | 0.0317 | 0.0577 | 0.9297 |
| Fixed DeCE displacement | 0.75 | 3 | 0.0322 | 0.0664 | 0.9266 |
| Fixed DeCE displacement | 1.00 | 3 | 0.0331 | 0.0730 | 0.9191 |
| Fixed DeCE displacement | 1.25 | 3 | 0.0329 | 0.0773 | 0.9188 |
| Fixed DeCE displacement | 1.50 | 3 | 0.0332 | 0.0737 | 0.9167 |
| Fixed DeCE displacement | 2.00 | 3 | 0.0337 | 0.0759 | 0.9142 |
| DeCE-RF | 0.50 | 3 | 0.0315 | 0.0555 | 0.9337 |
| DeCE-RF | 0.75 | 3 | 0.0321 | 0.0635 | 0.9263 |
| DeCE-RF | 1.00 | 3 | 0.0328 | 0.0680 | 0.9184 |
| DeCE-RF | 1.25 | 3 | 0.0329 | 0.0727 | 0.9200 |
| DeCE-RF | 1.50 | 3 | 0.0332 | 0.0733 | 0.9182 |
| DeCE-RF | 2.00 | 3 | 0.0334 | 0.0762 | 0.9152 |

Interpretation: E4 treats feedback as a stabilizer/robustness component rather than the sole source of the headline gain. Fixed DeCE displacement keeps operation-conditioned support and fixed clean-estimate edit-preserve displacement; DeCE-RF adds feedback-updated weights, projection, and preserve clean correction. The stress curve uses fixed-mask local edit L1 as an edit-pressure proxy, so it should be discussed as an edit-preserve tradeoff rather than a standalone semantic success score.
