# Core-6 Complete External Baseline Protocol

Date: 2026-06-01

## Purpose

External baselines should contextualize DeCE-RF against established image-editing methods. They should not replace the internal ablation logic for the paper's main claim, which is localized Rectified Flow edit-preserve control.

This protocol defines a complete baseline suite, not just one-off smoke tests. It separates method families, freezes input contracts, records failures, and keeps mask usage explicit.

## Main Claim Boundary

The paper's main evidence remains:

```text
base_only / RF reconstruction
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap / DeCE-RF
support_v3_fixed as internal ablation only
```

External baselines are used for context and reviewer expectations. They should be reported by family/backbone and control interface, not merged into the internal DeCE-RF ablation logic.

## Non-Negotiable Fairness Rules

1. Use the same Core-6 source images and task prompts for every method.
2. Use seeds 10, 11, and 12 when the method exposes seed/generator control.
3. Use fixed Core-6 evaluation masks only for metrics in headline rows. Do not pass those evaluation masks as editing inputs.
4. Record the baseline's native control interface: text-only, inversion-based text editing, P2P word control, method-native mask, manual mask, or same-support diagnostic.
5. Do not tune hyperparameters per image after visual inspection. Smoke tests may validate that a runner is operational, but full-run settings must be frozen before expanding to all seeds.
6. Failed installation, failed schema adaptation, target-formation failure, or missing model weights stays in the manifest with a concrete reason.
7. Do not select the better-looking variant per task as the reported baseline. If two variants are scientifically needed, report them as separate rows.

## Complete Baseline Suite

Use four blocks.

### Block 0: Internal Controls

These are not external baselines, but they define the claim and must stay in the main table:

```text
RF reconstruction / base_only
Direct target guidance / direct_target
Generic support control / adaptive_full_generic_support
DeCE-RF / support_v3_controller_rmsgap
```

Ablation-only internal control:

```text
support_v3_fixed
```

### Block A: Matched RF / Flow-Family Baselines

Closest-family external baselines. These are the highest priority for a WACV-style comparison if runnable:

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

Report status per method:

```text
source code downloaded
environment available
model/checkpoint available
Core-6 adapter available
complete rows
failed rows and concrete reason
```

### Block B: External Diffusion Text-Editing Baselines

Broader image-editing context. These are valid baselines but should be separated from RF/flow-family rows:

```text
LEDITS++
h-Edit-R + P2P
InstructPix2Pix
Pix2Pix-Zero
MasaCtrl
Prompt-to-Prompt
```

These rows answer: how do established diffusion editors behave on the same localized Core-6 tasks?

### Block C: Same-Support Diagnostics

These are diagnostic controls, not headline external baselines:

```text
same-support inpainting for removal-only cases
same-support LEDITS++ or h-Edit only if explicitly labeled diagnostic
same-support mask-aware editors if a mask-conditioned runner exists
```

Purpose: isolate whether the bottleneck is support geometry or generator/completion prior.

## Mask Policy

Use two clearly separated mask protocols.

### Headline External Baseline Rows

```text
editing mask input: method-native only, or none
fixed eval mask: metrics only
DeCE-RF support mask passed to external baseline: no
```

This is the default external baseline table.

### Same-Support Diagnostic Rows

```text
editing mask input: frozen shared task support/eval-style mask
fixed eval mask: metrics
eligible for main external table: no
eligible for diagnostic table: yes
```

Use this only to answer mechanistic questions such as: if another generator receives the same support region, does it preserve better or complete the removal better?

## Task Set

Core-6 tasks:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
red_chair_blue
```

Seeds:

```text
10 11 12
```

Full external baseline target:

```text
method x 6 tasks x 3 seeds
```

Rows are allowed to be incomplete only if the manifest records a reproducible failure reason.

## LEDITS++ Design

Primary row name:

```text
LEDITS++ (SD1.5, native text editing)
```

Status: complete for Core-6 seeds 10/11/12.

Protocol used:

```text
pipeline: diffusers LEditsPPPipelineStableDiffusion
backbone: sd-legacy/stable-diffusion-v1-5
input: source image + source prompt + task-specific editing prompt
mask input: none for headline row
metrics: fixed Core-6 eval masks
```

Completed artifacts:

```text
outputs/external_baselines/<task>/ledits_pp/seed_<seed>/result.png
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.csv
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.json
experiments/support_v3_2026-05-11/leditspp_core6_baseline_summary.md
experiments/support_v3_2026-05-11/visual_gates/leditspp_core6_seed10_12_grid.png
```

Reporting note: include LEDITS++ in the external diffusion table with a nonlocal-rewrite caveat. Do not use it as an RF/flow-family baseline.

## h-Edit Complete Baseline Design

h-Edit should not be represented by a single opportunistic smoke output. For the complete baseline, report one pre-registered row.

### h-Edit Row

```text
h-Edit-R + P2P (method-native word control)
```

Rationale:

- `h-Edit-R + P2P` is the strong/default text-guided route in the released text-guided experiments.
- Using one author-recommended route keeps the external baseline table simple and avoids a proliferation of h-Edit variants.
- `h-Edit-R` without P2P can still be used as a diagnostic if target formation fails, but it should not appear as a main external baseline row.

### h-Edit Full-Run Gate

Before full Core-6 expansion, run a pre-registered seed-10 gate:

```text
dog_sunglasses: add accessory / P2P-compatible
mug_heart or tshirt_star: add decal / target formation stress
red_chair_blue: recolor / attribute edit
```

Gate pass criteria:

```text
runner completes and writes result.png, metadata.json, stats.json, command.txt
h-Edit-R + P2P shows visible target formation on at least two of the three gate tasks
no hidden fixed eval/support mask is passed as editing input
one configuration is frozen before seeds 11/12 and remaining tasks
```

Current state:

```text
technical smoke: passed for mug_heart seed10
mug_heart target formation: failed under h-Edit-R + P2P with mug/mug blended word
full Core-6 expansion: gated, not yet approved
```

### h-Edit Adapter Table

Freeze one adapter table before full run.

| task | h-Edit-R prompt | P2P blended word | note |
| --- | --- | --- | --- |
| `cat_crown` | source prompt -> target prompt | `head crown` | add-object target; P2P may be unstable |
| `dog_sunglasses` | source prompt -> target prompt | `eyes sunglasses` | best P2P-compatible accessory gate |
| `mug_heart` | source prompt -> target prompt | `mug heart` | previous `mug mug` was invalid for target formation |
| `tshirt_star` | source prompt -> target prompt | `shirt star` | decal target-formation stress |
| `backpack_remove_toy_charm` | source prompt -> target prompt | no local blend or `charm charm` | removal wording may fail; record as limitation |
| `red_chair_blue` | source prompt -> target prompt | `red blue` or `chair chair` | recolor is P2P-compatible but must avoid shape drift |

If the adapter table changes after looking at outputs, create a new run ID and keep the older smoke outputs in the audit trail.

### h-Edit Hyperparameter Policy

Use seed-10 gate for small configuration selection only. Do not tune per task.

Smoke grid:

```text
variant: h_edit_R_p2p
num_diffusion_steps: 50 final; 20 only technical smoke
weight_reconstruction: 0.0, 0.1 on seed-10 gate only
cfg_src: 1.0
cfg_src_edit: 5.0
cfg_tar: 7.5, 10.5 on seed-10 gate only
eta: 1.0
P2P xa/sa: paper default xa=0.4, sa=0.35
```

Final full-run config:

```text
selected once from gate results
applied unchanged to all Core-6 tasks and seeds
recorded in command.txt and metadata.json
```

## InstructPix2Pix Design

Row name:

```text
InstructPix2Pix (instruction-only)
```

Protocol:

```text
input: source image + editing instruction
prompt: use a short imperative instruction, not the full target caption
mask input: none
seeds: 10/11/12
```

Task instruction examples:

```text
cat_crown: add a small golden crown on the cat's head
dog_sunglasses: add black sunglasses on the dog's eyes
mug_heart: add a small red heart printed on the front of the mug
tshirt_star: add a bright red star printed on the center of the t-shirt
backpack_remove_toy_charm: remove the yellow dangling toy charm while preserving the backpack
red_chair_blue: change only the chair fabric from red to deep blue
```

## Prompt-to-Prompt / MasaCtrl / Pix2Pix-Zero Design

These methods require word-level or inversion-specific adapters. They should be complete only if the adapter is predeclared and the run writes the standard manifest fields.

Recommended handling:

```text
Prompt-to-Prompt: report only tasks with meaningful source-target word alignment; mark unsupported add/remove cases explicitly.
MasaCtrl: report as attention-control baseline if the runner supports the task without manual masks.
Pix2Pix-Zero: report as inversion/direction baseline; record source/target text embedding construction.
```

Do not silently omit unsupported tasks. Use `status=unsupported_by_interface` or `status=adapter_failed` in the manifest.

## External Baseline Manifest

Every external baseline row must map to a manifest record.

Required fields:

```text
baseline
family
task
seed
status
source_image
source_prompt
target_prompt
editing_instruction
result_image
metadata
command
control_interface
mask_input_to_editor
eval_mask
backbone
runner
config_id
failure_reason
notes
```

Allowed statuses:

```text
pending
complete
smoke_failed
adapter_failed
env_failed
model_failed
unsupported_by_interface
excluded_diagnostic_only
```

## Metrics and Reporting

Use the existing fixed-mask metric pipeline for every complete row:

```text
scripts/evaluate_paper_metrics.py
--eval-mask-dir experiments/support_v3_2026-05-11/eval_masks
```

External baseline table columns:

```text
method
family/backbone
control interface
mask input to editor
complete rows
failed rows/reason
CLIP target-source delta
DINO/source similarity
outside-mask L1
inside-mask L1
source SSIM
visual audit overall
notes
```

Qualitative figure:

```text
source | DeCE-RF | LEDITS++ | h-Edit-R | h-Edit-R+P2P | one RF external baseline if runnable
```

Only include h-Edit columns after the h-Edit gate passes. Until then, show h-Edit in a diagnostic smoke/failure figure or appendix.

## Current Decision

- LEDITS++: completed and eligible for the external diffusion baseline table.
- h-Edit: technically runnable, but full Core-6 baseline is gated. Next action is a pre-registered h-Edit-R + P2P seed-10 gate on `dog_sunglasses`, `tshirt_star` or `mug_heart`, and `red_chair_blue`.
- RF/flow-family baselines: source repos are downloaded; environments and adapters still need validation before rows can be claimed.
- Same-support diagnostics: allowed only in diagnostic tables, not the headline external baseline table.

## Manifest Artifact

```text
experiments/support_v3_2026-05-11/core6_external_baseline_manifest.csv
```
