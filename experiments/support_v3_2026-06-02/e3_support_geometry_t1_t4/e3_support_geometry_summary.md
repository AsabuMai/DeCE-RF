# E3 Support Geometry Ablation

Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11/12.
This diagnostic uses saved support/debug masks and fixed external evaluation masks. Attention/clean/velocity rows are support-map diagnostics, not runnable method rows.

Figure 4 panel: `experiments/support_v3_2026-06-02/e3_support_geometry_t1_t4/e3_support_geometry_tshirt_star_seed10_figure4_panel.png`
Seed10 task sheet: `experiments/support_v3_2026-06-02/e3_support_geometry_t1_t4/e3_support_geometry_seed10_task_sheet.png`

| Variant | n | IoU up | Precision up | Recall up | Area | Outside L1 down | Edit score | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Attention only | 36 | 0.0588 | 0.0839 | 0.1098 | 0.0454 | - | - | semantic localization only |
| Clean disagreement | 36 | 0.0644 | 0.1172 | 0.1277 | 0.0474 | - | - | source-target clean-estimate gap |
| Velocity disagreement | 36 | 0.0646 | 0.1174 | 0.1286 | 0.0478 | - | - | RF response without relation prior |
| Grounding/SAM | 36 | 0.3220 | 0.3292 | 0.8205 | 0.1963 | - | - | external segmentation only |
| Generic support | 36 | 0.2044 | 0.2457 | 0.2817 | 0.0513 | 0.0496 | 0.0018 | weak automatic support |
| Operation support | 36 | 0.5242 | 0.5661 | 0.7470 | 0.0693 | 0.0427 | 0.0663 | operation-conditioned geometry estimator |

## Support-Quality Correlation

Correlation is computed only over runnable rows with downstream outputs: generic support and operation-conditioned support.

| Support metric | Downstream metric | n | Pearson r | Spearman r |
| --- | --- | ---: | ---: | ---: |
| support_iou | outside_mask_l1 | 18 | -0.2757 | -0.2247 |
| support_iou | source_ssim_luma | 18 | -0.1204 | -0.2880 |
| support_iou | edit_score | 18 | 0.1998 | 0.5666 |
| support_precision | outside_mask_l1 | 18 | -0.2700 | -0.2754 |
| support_precision | source_ssim_luma | 18 | -0.1258 | -0.1994 |
| support_precision | edit_score | 18 | 0.2139 | 0.5856 |
| support_recall | outside_mask_l1 | 18 | -0.1687 | -0.2944 |
| support_recall | source_ssim_luma | 18 | -0.2335 | -0.1424 |
| support_recall | edit_score | 18 | 0.1506 | 0.6046 |
| support_area_ratio | outside_mask_l1 | 18 | 0.0049 | 0.1872 |
| support_area_ratio | source_ssim_luma | 18 | -0.2623 | -0.3370 |
| support_area_ratio | edit_score | 18 | -0.2178 | -0.0437 |

Interpretation: E3 treats the support/mask as an explicit experimental object. Operation-conditioned support should be discussed as geometry estimation rather than as a hidden implementation detail. Downstream metrics are attached only to runnable rows with saved outputs; diagnostic heatmaps are evaluated for support quality only.
