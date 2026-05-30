# Baseline Parity Status

This file records which external-baseline artifacts can be used for paper
tables. The rule is strict: a baseline is table-eligible only if it uses the
same source image, target prompt, resolution, seed, and mask/support condition
as the RF h-Edit main matrix.

## Current Status

FlowEdit is complete for the go/no-go scope:

```text
4 tasks x seeds 10,11,12 = 12 matched FlowEdit runs
```

Artifacts:

```text
outputs/baselines/flowedit/<task_name>/seed_<seed>/
  result.png
  metadata.json
  command.txt
  run.log
```

Review grids:

```text
outputs/baselines/flowedit/cat_crown/flowedit_vs_full_seeds10_12.png
outputs/baselines/flowedit/dog_sunglasses/flowedit_vs_full_seeds10_12.png
outputs/baselines/flowedit/mug_heart/flowedit_vs_full_seeds10_12.png
outputs/baselines/flowedit/backpack_remove_toy_charm/flowedit_vs_full_seeds10_12.png
```

The matched condition is not pixel-identical to the pretty matrix because the
official FlowEdit SD3 runner required a preprocessed image. The runner resizes
the source image to `max_image_size=512` and crops to multiples of 16 before
calling FlowEdit. Source prompt, target prompt, seed, SD3 model family,
`T_steps=28`, and `n_max=24` are matched.

Visual readout:

- `cat_crown`: FlowEdit adds a crown but redraws the cat and background.
- `dog_sunglasses`: FlowEdit adds stronger sunglasses but changes dog identity
  and composition substantially.
- `mug_heart`: FlowEdit can add the heart, but some seeds shift the mug
  viewpoint or scene layout.
- `backpack_remove_toy_charm`: FlowEdit does not perform localized removal; it
  regenerates a different backpack/keychain scene and often keeps a charm.

Use FlowEdit as a qualitative external baseline for the localized-preservation
claim. Do not report it as a strict fully matched quantitative baseline unless
the resize/crop preprocessing is also applied to our method or clearly disclosed
as a baseline-specific compatibility step.

SplitFlow is also complete for the go/no-go scope:

```text
4 tasks x seeds 10,11,12 = 12 official SplitFlow runs
```

Artifacts:

```text
outputs/baselines/splitflow/<task_name>/seed_<seed>/
  result.png
  metadata.json
  command.txt
  run.log
```

Review grids:

```text
outputs/baselines/splitflow/cat_crown/splitflow_vs_flowedit_vs_full_seeds10_12.png
outputs/baselines/splitflow/dog_sunglasses/splitflow_vs_flowedit_vs_full_seeds10_12.png
outputs/baselines/splitflow/mug_heart/splitflow_vs_flowedit_vs_full_seeds10_12.png
outputs/baselines/splitflow/backpack_remove_toy_charm/splitflow_vs_flowedit_vs_full_seeds10_12.png
```

SplitFlow uses its official default schedule (`T_steps=50`, `n_max=33`,
`tar_guidance_scale=13.5`) because the public implementation hard-codes the
aggregation midpoint around step 28. Treat these runs as official-configuration
qualitative baselines, not as strict timestep-matched runs against our 28-step
pretty matrix.

Visual readout:

- `cat_crown`: SplitFlow adds a crown but changes the cat identity and pose.
- `dog_sunglasses`: SplitFlow gives stronger/darker sunglasses than our current
  `full` result, but it changes dog identity and face geometry.
- `mug_heart`: SplitFlow is competitive; it adds a clean heart with moderate
  viewpoint/lighting changes.
- `backpack_remove_toy_charm`: SplitFlow does not solve localized removal. It
  redraws the backpack/keychain scene and keeps or regenerates a dangling toy.

FireFlow is also complete for the go/no-go scope:

```text
4 tasks x seeds 10,11,12 = 12 official FireFlow runs
```

Artifacts:

```text
outputs/baselines/fireflow/<task_name>/seed_<seed>/
  result.png
  metadata.json
  command.txt
  run.log
```

Review grids:

```text
outputs/baselines/fireflow/cat_crown/fireflow_splitflow_flowedit_vs_full_seeds10_12.png
outputs/baselines/fireflow/dog_sunglasses/fireflow_splitflow_flowedit_vs_full_seeds10_12.png
outputs/baselines/fireflow/mug_heart/fireflow_splitflow_flowedit_vs_full_seeds10_12.png
outputs/baselines/fireflow/backpack_remove_toy_charm/fireflow_splitflow_flowedit_vs_full_seeds10_12.png
```

FireFlow uses FLUX-dev rather than SD3, so treat these as
official-configuration qualitative baselines. The completed runs used the
official fast-editing recipe: `num_steps=8`, `inject=1`, `guidance=2`,
`start_layer_index=0`, `end_layer_index=37`, `sampling_strategy=fireflow`, and
`offload=True`.

Visual readout:

- `cat_crown`: FireFlow adds a crown but changes cat identity and pose.
- `dog_sunglasses`: FireFlow adds clear sunglasses, but changes dog identity,
  face shape, and clothing/background details.
- `mug_heart`: FireFlow preserves a simple mug layout but the heart is small.
- `backpack_remove_toy_charm`: FireFlow does not remove the dangling charm
  reliably; it mostly preserves or redraws the toy.

RF-Solver-Edit is complete for the go/no-go scope:

```text
4 tasks x seeds 10,11,12 = 12 official RF-Solver-Edit runs
```

Artifacts:

```text
outputs/baselines/rf_solver_edit/<task_name>/seed_<seed>/
  result.png
  metadata.json
  command.txt
  run.log
```

RF-Solver-Edit uses FLUX-dev rather than SD3. Treat these as
official-configuration qualitative baselines. The public image-editing script
does not expose seed control, so the manifest seed is recorded as requested
metadata but is not actually controllable by the upstream runner.

ReFlex is not complete under current hardware and code constraints:

```text
4 tasks x seeds 10,11,12 = 12 failed ReFlex rows
```

Concrete evidence:

```text
outputs/baselines/reflex/mug_heart/seed_10/run.log
```

The public ReFlex code assumes a 4096-token latent layout, which corresponds to
1024-square FLUX operation. That configuration OOMs on the available 24GB GPU.
A local dynamic-latent compatibility patch allowed the 512 path to advance
further, but it then failed with CUDA attention-index assertions. The manifest
keeps all ReFlex rows as `failed` with this resource/compatibility reason rather
than silently omitting the baseline.

SteerFlow is not runnable locally as of 2026-05-10:

```text
4 tasks x seeds 10,11,12 = 12 failed SteerFlow rows
```

Search found the paper `SteerFlow: Steering Rectified Flows for Faithful
Inversion-Based Image Editing` (`arXiv:2604.01715`, dated 2026-04-02), but no
local repository or public GitHub/code release was found. The manifest keeps all
SteerFlow rows as `failed` with this no-code reason.

SplitFlow note, checked on 2026-05-10:

- Official repository is available at
  `https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow`.
- Local clone: `/home/Wu_25R8111/SplitFlow`.
- Commit checked: `c9ae45d2a6e386d6d00b91f19ec30dac9a20e786`.
- The runner structure is close to FlowEdit: `run_script.py`,
  `SD3_SplitFlow.yaml`, and `edits.yaml`.
- It uses SD3 plus `mistralai/Mistral-7B-Instruct-v0.3` for target-prompt
  decomposition. SD3 is already cached locally, but the Mistral model is not.
- Practical cost: memory and runtime are higher than FlowEdit because the
  official script loads both SD3 and Mistral. The completed runs used GPU 5 and
  took roughly 1.5 minutes per row after model initialization.
- YAML compatibility risk: the official script indexes
  `source_prompt[0]` and `target_prompts[0]`; a matched runner should write the
  temporary dataset using list-valued prompts or patch the script behavior
  locally.

Existing exploratory artifacts include:

- `outputs/cat_crown_other_rf_20260504/fireflow_512/`
- `outputs/cat_crown_other_rf_20260504/reflex/`
- `outputs/bus_blue_other_models_20260505/fireflow_512/`
- `h_edit_compare/results_polar*/`
- `results/baselines/`

These are useful for qualitative debugging and method comparison notes, but
they are not matched T1-T4 x seed 10/11/12 paper baselines.

## Required Before Paper Use

For each external method, create records in the same format as
`outputs/main_matrix`:

```text
outputs/baselines/<method>/<task_name>/seed_<seed>/
  result.png
  stats.json
  metadata.json
  command.txt
  masks/
```

The metadata must state:

- source image path
- source prompt
- target prompt
- seed
- resolution
- mask/support source
- method repository or commit when available
- exact command
- failure reason if the method cannot run

If a method cannot be run under matched conditions, report it as a limitation
instead of using the exploratory result as a main baseline.

## Matched Baseline Manifest

The current go/no-go baseline manifest is:

```text
experiments/baseline_parity_manifest.csv
```

Regenerate it with:

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/init_baseline_parity_manifest.py \
  --output experiments/baseline_parity_manifest.csv \
  --overwrite
```

Default scope:

```text
Baselines: flowedit fireflow rf_solver_edit reflex steerflow
Tasks:     cat_crown dog_sunglasses mug_heart backpack_remove_toy_charm
Seeds:     10 11 12
```

Each manifest row starts as `status=pending`. A row can become
`status=complete` only when `result.png`, `metadata.json`, and `command.txt`
exist under the listed matched path and the row's `matched_conditions` field
states that source image, source prompt, target prompt, resolution, and seed
match the pretty-matrix task.

If a baseline cannot run, keep the row and set `status=failed` with a concrete
`failure_reason` such as OOM, missing code, unsupported SD3/FLUX checkpoint, or
license/resource mismatch.
