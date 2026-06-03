# SD3 Batch Runner Usage

Use this path for any internal SD3 / DeCE-RF run with more than one image. It
keeps one SD3 pipeline loaded and executes queued `command.txt` files in order.
This is much faster than launching `run_edit_sd3.py` once per image.

## Slurm Rule

Run generation on an A100 interactive/noninteractive Slurm allocation, not on
the master/login node:

```bash
srun -p a100 -w a100-01 --gres shard:1 /bin/bash -lc 'cd /cluster/users/grad/2025/25t8103/project && <command>'
```

## Recommended Migration Check

Use this when changing server/environment and validating that E1 still runs:

```bash
srun -p a100 -w a100-01 --gres shard:1 /bin/bash -lc '
cd /cluster/users/grad/2025/25t8103/project && \
RUN_ID=migration_e1_dece_seed10_$(date +%Y%m%d_%H%M%S) \
SCOPE=strict \
METHODS="support_v3_controller_rmsgap" \
SEEDS="10" \
MODEL_OFFLOAD=0 \
ALLOW_MASK_DOWNLOAD=1 \
RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1 \
bash scripts/run_wacv_phase1_batch.sh
'
```

Expected matrix:

```text
6 strict Core-6 tasks x support_v3_controller_rmsgap x seed 10 = 6 outputs
```

After completion, evaluate the run without mixing it into the main E1 metrics:

```bash
RUN_ID=<run_id_from_output_dir>
mkdir -p experiments/support_v3_2026-06-02/${RUN_ID}
.venv/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir outputs/${RUN_ID} \
  --csv-output experiments/support_v3_2026-06-02/${RUN_ID}/fixed_mask_metrics.csv \
  --json-output experiments/support_v3_2026-06-02/${RUN_ID}/fixed_mask_metrics.json \
  --task-names all \
  --method-names support_v3_controller_rmsgap \
  --seeds 10 \
  --eval-mask-dir experiments/support_v3_2026-06-02/eval_masks
```

## Full Strict E1 Seed-10 Batch

Use this when a stronger server smoke is needed:

```bash
srun -p a100 -w a100-01 --gres shard:1 /bin/bash -lc '
cd /cluster/users/grad/2025/25t8103/project && \
RUN_ID=e1_strict_seed10_batch_$(date +%Y%m%d_%H%M%S) \
SCOPE=strict \
METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
SEEDS="10" \
MODEL_OFFLOAD=0 \
ALLOW_MASK_DOWNLOAD=1 \
RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1 \
bash scripts/run_wacv_phase1_batch.sh
'
```

Expected matrix:

```text
6 strict Core-6 tasks x 4 E1 methods x seed 10 = 24 outputs
```

## Phase 2 / Larger Batches

For larger batches, set `TASKS`, `METHODS`, and `SEEDS` explicitly:

```bash
srun -p a100 -w a100-01 --gres shard:1 /bin/bash -lc '
cd /cluster/users/grad/2025/25t8103/project && \
RUN_ID=phase2_dece_subset_$(date +%Y%m%d_%H%M%S) \
TASKS="cat_crown bowl_apple_inside tshirt_star" \
METHODS="support_v3_controller_rmsgap" \
SEEDS="10 11 12" \
MODEL_OFFLOAD=0 \
ALLOW_MASK_DOWNLOAD=1 \
RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1 \
bash scripts/run_wacv_phase1_batch.sh
'
```

## Important Environment Variables

- `RUN_ID`: output directory name under `outputs/` unless `OUTPUT_ROOT` is set.
- `OUTPUT_ROOT`: explicit output directory. Defaults to `outputs/${RUN_ID}` in
  the batch wrapper.
- `SCOPE`: `strict`, `implemented`, or `old_server_evidence`; use `strict` for
  paper-facing Core-6 work.
- `TASKS`: optional explicit task list; overrides the scope task list.
- `METHODS`: method list, e.g. `support_v3_controller_rmsgap` or all four E1
  methods.
- `SEEDS`: seed list, e.g. `10` or `10 11 12`.
- `MODEL_OFFLOAD=0`: recommended on A100; keeps SD3 resident on GPU.
- `ALLOW_MASK_DOWNLOAD=1`: allow missing GroundingDINO/SAM cache to be fetched.
- `RF_H_EDIT_ALLOW_CLIP_DOWNLOAD=1`: allow CLIP reward/model cache fetch if a
  run requires it.
- `PREPARE_ONLY=1`: only generate masks/references and batch manifest; do not
  run SD3 generation.
- `BATCH_SKIP_EXISTING=1`: skip queued commands that already have complete
  result/stats/metadata.

## What The Wrapper Does

1. `scripts/run_wacv_phase1_batch.sh` calls `scripts/run_wacv_phase1.sh` with a
   `BATCH_MANIFEST` path.
2. `scripts/run_pretty_matrix.sh` prepares masks/reference images and writes one
   `command.txt` per run, but queues it instead of launching Python.
3. `scripts/run_sd3_batch.py` loads SD3 once and executes all queued commands.
4. A `batch_summary.json` file records status and runtime per command.

## Current Validated Example

The first completed migration validation is:

```text
outputs/migration_e1_dece_seed10_batch_20260603_161806/
experiments/support_v3_2026-06-02/migration_e1_seed10_batch_20260603_161806/
```

It completed 6/6 DeCE-RF seed-10 strict Core-6 outputs. Old-vs-new fixed-mask
metric deltas were small for outside-mask/source-preservation metrics, so the
original full E1 matrix remains the main paper evidence.
