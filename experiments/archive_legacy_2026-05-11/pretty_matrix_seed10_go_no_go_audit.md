# Pretty Matrix Seed-10 Go/No-Go Audit

Date: 2026-05-10

Run command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TASKS="P1 P2 P3 P7" METHODS="M0 M1 M5 M6 M7 M4" SEEDS="10" DEVICE=5 \
SKIP_EXISTING=1 bash scripts/run_pretty_matrix.sh
```

The first attempt on `DEVICE=4` failed with SD3/T5 CUDA OOM during
`cat_crown/full_no_ref`. Retrying on `DEVICE=5` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed.

## Coverage

All 24 seed-10 runs are complete:

```text
4 tasks x 6 methods x seed 10
```

Each run has:

```text
result.png
stats.json
metadata.json
command.txt
```

Review grids:

- `outputs/pretty_matrix/cat_crown_seed10_go_no_go_review.png`
- `outputs/pretty_matrix/dog_sunglasses_seed10_go_no_go_review.png`
- `outputs/pretty_matrix/mug_heart_seed10_go_no_go_review.png`
- `outputs/pretty_matrix/backpack_remove_toy_charm_seed10_go_no_go_review.png`

Audit CSV initialized at:

```text
experiments/pretty_matrix_visual_audit_seed10.csv
```

## Initial Visual Decisions

| Task | Initial decision | Notes |
| --- | --- | --- |
| `cat_crown` | Pass | `full` preserves the scene and inserts a visible crown. `direct_target` changes cat identity strongly. `full_no_ref`, `full_no_rec`, and `full_no_traj` are also visually strong on seed 10, so this task may not isolate the ablation effects well. |
| `dog_sunglasses` | Weak pass | `direct_target` creates strong sunglasses but heavily changes the dog. `full` is more preserved but the glasses are translucent/weak. `full_no_ref` gives darker glasses but changes face appearance more. This task supports the preservation/edit tradeoff but needs careful wording. |
| `mug_heart` | Pass | `full` produces a clear red heart while preserving the mug better than `direct_target`. `full_no_ref` nearly fails, which supports the local-reference prior. `full_no_rec` and `full_no_traj` remain usable. |
| `backpack_remove_toy_charm` | Pass | `full` removes the lower yellow toy charm while preserving the pink ring, zipper, backpack fabric, and the upper cartoon patch. The upper cartoon patch is part of the source image, not a generated artifact. `base_only` and `direct_target` fail to preserve the source semantics: they distort the patch and leave or smear the charm. |

## Immediate Implication

Seed 10 justifies the next small expansion to seeds 11 and 12 for the four
go/no-go tasks. All four tasks have a usable `full` result, with
`dog_sunglasses` remaining the weakest success case because the sunglasses are
visually subtle.

Recommended next step:

1. Run seeds 11 and 12 for `cat_crown`, `dog_sunglasses`, and `mug_heart`.
2. Keep the original P7 removal prompt for `backpack_remove_toy_charm`; adding
   explicit upper-patch preservation to the prompt was tested and made the
   removal route regress into object replacement.
3. Fill `experiments/pretty_matrix_visual_audit_seed10.csv` with the visual
   ratings before starting baseline parity runs.
