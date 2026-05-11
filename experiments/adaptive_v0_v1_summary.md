# Adaptive v0 vs v1 Ablation Summary

Rows are averaged over seeds 10/11/12 and ODE logged steps. v0 uses RMS target; v1 uses clean-estimate directional progress target.

| task | method | edit_progress | edit_deficit | edit_weight | preserve_drift | preserve_weight | clean_conflict | clean_proj_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cat_crown | adaptive_full_v0 | 0.4083 | 0.0612 | 1.1038 | 0.3005 | 1.3107 | 2.0188 | 0.0259 |
| cat_crown | adaptive_full | 0.4586 | 0.2851 | 1.3138 | 0.3011 | 1.3110 |  |  |
| dog_sunglasses | adaptive_full_v0 | 0.2617 | 0.0478 | 1.0768 | 0.2602 | 1.2202 | 0.6932 | 0.0046 |
| dog_sunglasses | adaptive_full | 0.3894 | 0.3215 | 1.3528 | 0.2651 | 1.2310 |  |  |
| mug_heart | adaptive_full_v0 | 0.3725 | 0.0835 | 1.1389 | 0.1694 | 1.0302 | 0.0306 | 0.0038 |
| mug_heart | adaptive_full | 0.4525 | 0.3016 | 1.3037 | 0.1661 | 1.0275 |  |  |
| backpack_remove_toy_charm | adaptive_full_v0 | 0.7615 | 0.1401 | 1.2339 | 0.2682 | 1.2604 | 0.6651 | 0.0076 |
| backpack_remove_toy_charm | adaptive_full | 0.7932 | 0.0510 | 1.0832 | 0.2692 | 1.2630 |  |  |
| OVERALL | adaptive_full_v0 | 0.4510 | 0.0832 | 1.1383 | 0.2496 | 1.2054 | 0.8519 | 0.0104 |
| OVERALL | adaptive_full | 0.5234 | 0.2398 | 1.2634 | 0.2504 | 1.2082 |  |  |

Artifacts:
- `outputs/pretty_matrix/adaptive_v0_v1_seed10_overview.png`
- `outputs/pretty_matrix/cat_crown/full_v0_v1_seeds10_12.png`
- `outputs/pretty_matrix/dog_sunglasses/full_v0_v1_seeds10_12.png`
- `outputs/pretty_matrix/mug_heart/full_v0_v1_seeds10_12.png`
- `outputs/pretty_matrix/backpack_remove_toy_charm/full_v0_v1_seeds10_12.png`
