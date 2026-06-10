# E4 Controller And Robustness Ablation

Scope: Phase2 T1-T4, 12 tasks.
Base fixed-vs-feedback rows use seeds 10/11/12. Edit-strength stress uses seed10 across multipliers 0.50, 0.75, 1.00, 1.25, 1.50, and 2.00.

Figure 5 Pareto: `experiments/support_v3_2026-06-02/e4_controller_ablation_t1_t4/e4_figure5_edit_strength_pareto.png`
Controller trajectory figure: `experiments/support_v3_2026-06-02/e4_controller_ablation_t1_t4/e4_controller_trajectory_tshirt_star_seed10.png`

## Base Component Table

| Variant | n | Outside L1 down | Inside L1 | Source SSIM up | Local edit L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed DeCE displacement | 36 | 0.0350 | 0.1729 | 0.9147 | 0.1729 |
| DeCE-RF | 36 | 0.0343 | 0.1772 | 0.9144 | 0.1772 |

## Trajectory Summary

| Variant | n | Preserve drift | Preserve weight | Edit weight | Projection norm | Preserve correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed DeCE displacement | 36 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| DeCE-RF | 36 | 0.2134 | 1.2338 | 1.1600 | 0.0454 | 2.8844 |

## Edit-Strength Stress

| Variant | Multiplier | n | Outside L1 down | Local edit L1 | Source SSIM up |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed DeCE displacement | 0.50 | 12 | 0.0342 | 0.1726 | 0.9121 |
| Fixed DeCE displacement | 0.75 | 12 | 0.0345 | 0.1750 | 0.9102 |
| Fixed DeCE displacement | 1.00 | 12 | 0.0350 | 0.1736 | 0.9149 |
| Fixed DeCE displacement | 1.25 | 12 | 0.0349 | 0.1849 | 0.9068 |
| Fixed DeCE displacement | 1.50 | 12 | 0.0350 | 0.1885 | 0.9028 |
| Fixed DeCE displacement | 2.00 | 12 | 0.0353 | 0.1995 | 0.8937 |
| DeCE-RF | 0.50 | 12 | 0.0341 | 0.1733 | 0.9096 |
| DeCE-RF | 0.75 | 12 | 0.0345 | 0.1799 | 0.9070 |
| DeCE-RF | 1.00 | 12 | 0.0342 | 0.1922 | 0.9064 |
| DeCE-RF | 1.25 | 12 | 0.0347 | 0.1829 | 0.9030 |
| DeCE-RF | 1.50 | 12 | 0.0350 | 0.1822 | 0.9126 |
| DeCE-RF | 2.00 | 12 | 0.0353 | 0.1987 | 0.8942 |

Interpretation: E4 treats feedback as a stabilizer/robustness component rather than the sole source of the headline gain. Fixed DeCE displacement keeps operation-conditioned support and fixed clean-estimate edit-preserve displacement; DeCE-RF adds feedback-updated weights, projection, and preserve clean correction. The stress curve uses fixed-mask local edit L1 as an edit-pressure proxy, so it should be discussed as an edit-preserve tradeoff rather than a standalone semantic success score.
