# E3 Support Geometry Ablation

Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11/12.
This diagnostic uses saved support/debug masks and fixed external evaluation masks. Attention/clean/velocity rows are support-map diagnostics, not runnable method rows.

Figure 4 panel: `experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_tshirt_star_seed10_figure4_panel.png`
Seed10 task sheet: `experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_seed10_task_sheet.png`

| Variant | n | IoU up | Precision up | Recall up | Area | Outside L1 down | Edit score | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Attention only | 9 | 0.0405 | 0.1178 | 0.0611 | 0.0460 | - | - | semantic localization only |
| Clean disagreement | 9 | 0.0438 | 0.1479 | 0.0617 | 0.0471 | - | - | source-target clean-estimate gap |
| Velocity disagreement | 9 | 0.0436 | 0.1478 | 0.0613 | 0.0469 | - | - | RF response without relation prior |
| Grounding/SAM | 9 | 0.3574 | 0.4971 | 0.7366 | 0.1731 | - | - | external segmentation only |
| Generic support | 9 | 0.0935 | 0.2962 | 0.1205 | 0.0433 | 0.0383 | -0.0049 | weak automatic support |
| Operation support | 9 | 0.2966 | 0.4038 | 0.4520 | 0.0737 | 0.0348 | 0.0539 | operation-conditioned geometry estimator |

## Support-Quality Correlation

Correlation is computed only over runnable rows with downstream outputs: generic support and operation-conditioned support.

| Support metric | Downstream metric | n | Pearson r | Spearman r |
| --- | --- | ---: | ---: | ---: |
| support_iou | outside_mask_l1 | 18 | -0.5809 | -0.3722 |
| support_iou | source_ssim_luma | 18 | 0.2511 | 0.3149 |
| support_iou | edit_score | 18 | -0.1380 | -0.3722 |
| support_precision | outside_mask_l1 | 18 | -0.5337 | -0.3149 |
| support_precision | source_ssim_luma | 18 | 0.1785 | 0.2004 |
| support_precision | edit_score | 18 | -0.3989 | -0.4867 |
| support_recall | outside_mask_l1 | 18 | -0.6193 | -0.5439 |
| support_recall | source_ssim_luma | 18 | 0.3818 | 0.4867 |
| support_recall | edit_score | 18 | 0.0989 | -0.1431 |
| support_area_ratio | outside_mask_l1 | 18 | -0.4314 | -0.4232 |
| support_area_ratio | source_ssim_luma | 18 | 0.0898 | 0.4232 |
| support_area_ratio | edit_score | 18 | -0.3431 | -0.4797 |

Interpretation: E3 treats the support/mask as an explicit experimental object. Operation-conditioned support should be discussed as geometry estimation rather than as a hidden implementation detail. Downstream metrics are attached only to runnable rows with saved outputs; diagnostic heatmaps are evaluated for support quality only.
