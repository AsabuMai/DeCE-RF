# E1 Migration Validation: Batch Seed 10

Date: 2026-06-03

Purpose: validate that the strict Core-6 DeCE-RF row can run on the new server
without rerunning the full E1 matrix.

Run package:

```text
outputs/migration_e1_dece_seed10_batch_20260603_161806/
```

Protocol:

```text
6 strict Core-6 tasks x support_v3_controller_rmsgap x seed 10 = 6 outputs
Batch runner: scripts/run_wacv_phase1_batch.sh
SD3 pipeline loading: once per batch
MODEL_OFFLOAD=0 on A100
ALLOW_MASK_DOWNLOAD=1 for missing GroundingDINO/SAM cache
```

Artifacts:

```text
fixed_mask_metrics.csv
fixed_mask_metrics.json
old_vs_migration_metric_diff.json
```

Completion:

```text
results: 6/6
stats: 6/6
metadata: 6/6
batch status: complete=6 skipped=0 failed=0
```

Old-vs-migration fixed-mask metric differences:

```text
max |delta outside_mask_l1| = 0.001060
max |delta source_ssim_luma| = 0.007297
max |delta inside_mask_l1| = 0.013028
```

Interpretation: outside-mask/source-preservation differences are small. The
migration check supports keeping the original strict E1 matrix as the main
paper evidence while using the new server for follow-up batches.
