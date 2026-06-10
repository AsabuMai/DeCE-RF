# E3 Support Geometry Ablation Complete

Date: 2026-06-04

Status: complete for the current paper-drafting package.

## Scope

```text
3 tasks x 6 support geometry variants x 3 seeds = 54 support-map rows
tasks: cat_crown, tshirt_star, backpack_remove_toy_charm
seeds: 10, 11, 12
```

Variants:

```text
attention_only
clean_disagreement
velocity_disagreement
grounding_sam
generic_support
operation_conditioned_support
```

## Artifact List

```text
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_mask_metrics.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_summary.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_by_task_summary.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_correlation.csv
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_summary.md
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_tshirt_star_seed10_figure4_panel.png
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_seed10_task_sheet.png
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry_files.txt
experiments/support_v3_2026-06-02/e3_support_geometry/e3_support_geometry.sha256
```

## Row Interpretation

Support-map diagnostic rows:

```text
attention_only
clean_disagreement
velocity_disagreement
grounding_sam
```

These rows evaluate support geometry only. They are not runnable editing-method rows.

Runnable downstream rows:

```text
generic_support = adaptive_full_generic_support
operation_conditioned_support = support_v3_controller_rmsgap support path
```

These rows have both support-quality metrics and downstream edit/preserve metrics.

## Main Results

| Variant | n | IoU | Precision | Recall | Area | Outside L1 | Edit score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention only | 9 | 0.0405 | 0.1178 | 0.0611 | 0.0460 | - | - |
| Clean disagreement | 9 | 0.0438 | 0.1479 | 0.0617 | 0.0471 | - | - |
| Velocity disagreement | 9 | 0.0436 | 0.1478 | 0.0613 | 0.0469 | - | - |
| Grounding/SAM | 9 | 0.3574 | 0.4971 | 0.7366 | 0.1731 | - | - |
| Generic support | 9 | 0.0935 | 0.2962 | 0.1205 | 0.0433 | 0.0383 | -0.0049 |
| Operation support | 9 | 0.2966 | 0.4038 | 0.4520 | 0.0737 | 0.0348 | 0.0539 |

## Correlation Readout

Correlation is computed only over runnable downstream rows:

```text
generic_support
operation_conditioned_support
```

Most useful support-quality correlations:

| Relationship | n | Pearson r | Spearman r | Interpretation |
| --- | ---: | ---: | ---: | --- |
| support_iou vs outside_mask_l1 | 18 | -0.5809 | -0.3722 | better support overlap is associated with lower outside drift |
| support_recall vs outside_mask_l1 | 18 | -0.6193 | -0.5439 | better edit-region coverage is associated with lower outside drift |
| support_recall vs source_ssim_luma | 18 | 0.3818 | 0.4867 | better coverage is associated with higher source preservation |

Do not overclaim edit-score correlation. The current local/CLIP edit score is task-sensitive, and the 18-row correlation mixes insertion, decal, and removal cases.

## Paper-Safe Claim

```text
E3 treats the support mask as an explicit experimental object. Operation-
conditioned support improves fixed-mask overlap and downstream edit behavior
relative to weak generic support, while Grounding/SAM alone tends to over-cover
the object/host region. This supports the claim that DeCE-RF's support geometry
is not merely generic segmentation or raw attention evidence.
```

## Claim Boundary

```text
Attention, clean-disagreement, velocity-disagreement, and Grounding/SAM rows are
support-map diagnostics. They should not be reported as full editing baselines.
Downstream edit/preserve metrics are reported only for runnable saved-output
rows.
```

## Completion Decision

E3 is complete for the current manuscript package. Optional future extensions:

```text
manual upper-bound support
support shrink/dilate perturbation
broader 6-task or 12-case expansion
```

These are not required before moving to E4 or paper table/figure packaging.
