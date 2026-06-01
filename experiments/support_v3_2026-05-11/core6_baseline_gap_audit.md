# Core-6 Baseline Gap Audit

Date: 2026-06-01

## Status

The Core-6 internal DeCE-RF matrix is complete. External baseline source code has now been downloaded on the current server, but matched Core-6 external/RF baseline outputs are still pending because the environments, model weights, and per-method runners have not yet been validated.

## Downloaded Baseline Repositories

Repository roots are under `/workspace/baselines/src/`. The pinned clone status is recorded in:

```text
/workspace/baselines/download_status.tsv
/workspace/baselines/baseline_repos.tsv
```

Downloaded code repositories:

```text
FlowEdit
SplitFlow
FireFlow
RF-Solver-Edit
ReFlex
FlowAlign
stable-flow
ZONE
instruct-pix2pix
pix2pix-zero
MasaCtrl
prompt-to-prompt
h-edit
ledits_pp
```

Notes:

- h-Edit and LEDITS++ source code have been added to the downloaded baseline pool.
- SteerFlow: no public runnable repository has been confirmed yet.
- DiffEdit: should be treated as a diffusers-pipeline baseline rather than a separate downloaded project repository, unless a specific implementation is selected later.

## Current Server Check

- `/home/Wu_25R8111/FlowEdit`: missing on the current server; superseded by `/workspace/baselines/src/FlowEdit` for the new baseline setup.
- `/home/Wu_25R8111/ENTER/envs/flowedit/bin/python`: missing on the current server; baseline environments still need to be recreated or mapped.
- `outputs/baselines/flowedit` and `outputs/baselines/splitflow`: no current Core-6 result images found.
- No `tshirt_star` or `red_chair_blue` baseline outputs were found under `outputs/baselines`.

## Available Baseline Evidence

Archived core-4 baseline summaries:

```text
experiments/archive_legacy_2026-05-11/baseline_parity_manifest.csv
experiments/archive_legacy_2026-05-11/baseline_summary.csv
experiments/archive_legacy_2026-05-11/baseline_summary.md
experiments/archive_legacy_2026-05-11/baseline_visual_scores_seed10_12.csv
```

These cover the earlier four-task set and should be reported only as contextual or availability evidence unless rerun under the Core-6 protocol.

## New Diagnostic Baseline

A same-support classical inpainting diagnostic has been generated for `backpack_remove_toy_charm` only:

```text
experiments/support_v3_2026-05-11/backpack_same_support_inpaint_metrics.csv
experiments/support_v3_2026-05-11/backpack_same_support_inpaint_metrics.json
experiments/support_v3_2026-05-11/backpack_same_support_inpaint_summary.md
```

This is a removal-only diagnostic, not a main Core-6 baseline.


## External Baseline Plan

Focused protocol note:

```text
experiments/support_v3_2026-05-11/core6_external_baseline_plan.md
experiments/support_v3_2026-05-11/core6_external_baseline_protocol.md
experiments/support_v3_2026-05-11/core6_external_baseline_manifest.csv
```

`LEDITS++` is marked include/priority-1. `h-Edit` is marked include-after-smoke/priority-2 because its text-guided runner is PieBench/P2P-oriented and needs a Core-6 schema adapter.

## Completed LEDITS++ Baseline

LEDITS++ has been run for all Core-6 tasks and seeds 10/11/12. It is a completed external diffusion-editing baseline, not an RF/flow-family baseline.

```text
outputs/external_baselines/<task>/ledits_pp/seed_<seed>/result.png
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.csv
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.json
experiments/support_v3_2026-05-11/leditspp_core6_baseline_summary.md
experiments/support_v3_2026-05-11/visual_gates/leditspp_core6_seed10_12_grid.png
```

Visual caveat: LEDITS++ often performs the target edit by nonlocal rewriting or identity/background changes, so it should be reported with model-family and locality caveats.

## h-Edit Smoke Status

h-Edit is technically runnable through `/workspace/baselines/envs/h-edit-py312`, but the first Core-6 `mug_heart` seed-10 smoke did not form the red-heart target. Keep h-Edit in the external baseline plan, but do not run or report full Core-6 h-Edit rows until a target-formation smoke gate passes.

```text
experiments/support_v3_2026-05-11/hedit_core6_smoke_summary.md
experiments/support_v3_2026-05-11/visual_gates/hedit_mug_heart_seed10_smoke.png
```

## Recommended Next Baseline Step

Validate or create runnable environments for the downloaded baseline repositories, then create a new Core-6 baseline manifest for:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
red_chair_blue
```

Minimum RF block for a fair WACV comparison:

```text
FlowEdit or RF inversion + target resampling
Direct target / vanilla RF guidance
DeCE-RF
```

Do not use the archived core-4 RF baseline rows as headline Core-6 results.
