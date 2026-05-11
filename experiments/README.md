# Experiment Reproducibility Plan

This directory tracks the experiments needed to turn the RF h-Edit prototype
into a submission-ready paper package.

## Current Readiness

The codebase already records many single-run outputs under `outputs/`. A run is
paper-reproducible only when the folder contains all four files:

```text
result.png
stats.json
metadata.json
command.txt
```

Run the audit with:

```bash
cd /home/Wu_25R8111/rf_h_edit_project
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/audit_experiment_records.py \
  --outputs-dir outputs \
  --json-output experiments/output_record_audit.json
```

The script exits with a non-zero status if any experiment folder is missing a
required record file.

## Minimum Paper Matrix

Use a small fixed benchmark before adding more qualitative probes.

| Question | Tasks | Required comparisons |
| --- | --- | --- |
| Does the RF edit direction work? | accessory insertion, color edit, semantic replacement | base only, direct target, anchor only |
| Does reconstruction decoupling help? | same fixed tasks | direct target, direct target + reconstruction, full method |
| Does spatial support matter? | local accessory insertion | broad attention mask, changed-token mask, structure mask, external diagnostic mask |

## Required Metrics

The current `stats.json` files are useful process logs, but they are not enough
for a paper table. Add a metric script that writes one row per run with:

```text
edit_success_score
source_preservation_lpips
source_preservation_dino
mask_outside_l1
mask_outside_lpips
runtime_seconds
peak_memory_gb
```

If a metric cannot be computed for a run, record `null` and explain why in the
table notes instead of silently dropping the run.

## Submission Rule

Only use a result in the paper if:

1. `command.txt` reruns the result from a clean checkout.
2. `metadata.json` includes `git_commit`, prompts, seed, and all guidance/mask settings.
3. The run appears in the fixed experiment matrix, not only in a one-off probe.
4. The corresponding failure cases are documented when the method is unstable.
