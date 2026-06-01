# Core-6 External Baseline Plan

Date: 2026-06-01

## Decision

Add `LEDITS++` and `h-Edit` to the external baseline plan.

These are valid external diffusion-editing baselines for Core-6, but they should be reported separately from RF/flow-family baselines because their backbone and control mechanisms differ from DeCE-RF.

## Baseline Groups

RF / flow-family candidates:

```text
FlowEdit
SplitFlow
FireFlow
RF-Solver-Edit
ReFlex
FlowAlign
stable-flow
ZONE
```

Diffusion text-editing candidates:

```text
LEDITS++ / ledits_pp
h-Edit
InstructPix2Pix
Pix2Pix-Zero
MasaCtrl
Prompt-to-Prompt
```

## LEDITS++ Plan

Status: completed for Core-6 seeds 10/11/12 using the diffusers-integrated pipeline; retain as external diffusion-editing baseline with nonlocal-rewrite caveat.

Rationale: repository provides a package-style pipeline and also has diffusers integration. It is suitable for a Core-6 baseline runner with source image inversion followed by text-guided editing.

Protocol:

```text
source images: Core-6 source images
prompts: task source prompt + target/edit prompt
seeds: 10, 11, 12 where supported
resolution: match the baseline's documented/default safe resolution; record deviations
masks: do not pass the fixed evaluation mask as an editing mask unless explicitly testing same-support diagnostics
metrics: reuse fixed Core-6 evaluation masks
```

Required before full run:

```text
install or isolate ledits_pp environment
run one seed-10 smoke task
verify output naming and manifest fields
verify metrics script can ingest generated images
```

## h-Edit Plan

Status: technical smoke runnable, target-formation smoke not yet passed. A compatibility env exists at `/workspace/baselines/envs/h-edit-py312`; the first `mug_heart` seed-10 smoke preserved the source but did not form the red heart, so full Core-6 h-Edit expansion is gated on another adapter/settings smoke.

Rationale: h-Edit is a recent training-free diffusion editing method with a text-guided P2P variant. It is scientifically relevant, but it is not an RF method and its released text-guided runner is PieBench/P2P-oriented, so Core-6 requires an adapter.

Closest protocol:

```text
method: implicit h-Edit-R + P2P text-guided variant
input schema: source image, source prompt, target prompt, editing instruction, blended_word
smoke gate: seed 10 on one add/decal task and one recolor/removal task if feasible
full run: Core-6 x seeds 10/11/12 only after smoke gate passes
```

Fairness notes:

```text
report as diffusion bridge / attention-control baseline, not RF baseline
record backbone, inversion mode, P2P settings, cfg values, optimization steps
use fixed evaluation masks only for metrics, not as hidden editing input
```

## Reporting Boundary

External baselines are used to contextualize DeCE-RF, not to replace the internal ablation logic. The main claim remains localized Rectified Flow edit-preserve control. Any baseline that fails installation, model loading, or Core-6 schema adaptation should remain in the manifest with the concrete failure reason rather than being silently omitted.

## Completed LEDITS++ Artifacts

```text
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.csv
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.json
experiments/support_v3_2026-05-11/leditspp_core6_baseline_summary.md
experiments/support_v3_2026-05-11/visual_gates/leditspp_core6_seed10_12_grid.png
```

## h-Edit Environment Check

Current check result: not directly runnable in the project `.venv`. Required next step is to create an isolated h-Edit environment or container matching `text-guided/environment_p2p.yaml`, then run a seed-10 smoke gate through a Core-6 YAML adapter.

```text
missing in current .venv: nltk, clip, pytorch_lightning
version mismatch: diffusers 0.31 vs h-Edit environment_p2p diffusers 0.18
server package manager: no conda/mamba found in PATH during initial check
```

## h-Edit Smoke Artifacts

```text
outputs/external_baselines/mug_heart/h_edit/seed_10/result.png
experiments/support_v3_2026-05-11/hedit_mug_heart_seed10_smoke_metrics.csv
experiments/support_v3_2026-05-11/hedit_mug_heart_seed10_smoke_metrics.json
experiments/support_v3_2026-05-11/hedit_core6_smoke_summary.md
experiments/support_v3_2026-05-11/visual_gates/hedit_mug_heart_seed10_smoke.png
```

## Baseline Protocol

Use this protocol before expanding any additional external baseline rows:

```text
experiments/support_v3_2026-05-11/core6_external_baseline_protocol.md
```

## Complete Baseline Manifest

```text
experiments/support_v3_2026-05-11/core6_external_baseline_manifest.csv
```
