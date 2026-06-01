#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rf_h_edit
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TASKS="P1 P2 P3 P5 P7 P4"
export METHODS="M0 M1 M10 M18 M17"
export SEEDS="10 11 12"
export REGENERATE_MASKS=1
export SKIP_EXISTING=0
export ALLOW_MASK_DOWNLOAD=0
export SEMANTIC_MASK_CACHE_METHOD=support_v3_controller_rmsgap
export DEVICE=0
export LOW_VRAM=0
bash scripts/run_pretty_matrix.sh
.venv/bin/python scripts/evaluate_paper_metrics.py   --outputs-dir outputs/pretty_matrix   --task-names cat_crown,dog_sunglasses,mug_heart,tshirt_star,backpack_remove_toy_charm,red_chair_blue   --method-names base_only,direct_target,adaptive_full_generic_support,support_v3_controller_rmsgap   --seeds 10,11,12   --eval-mask-dir experiments/support_v3_2026-05-11/eval_masks   --clip-model openai/clip-vit-base-patch32   --dino-model facebook/dinov2-small   --csv-output experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.csv   --json-output experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.json
.venv/bin/python scripts/evaluate_paper_metrics.py   --outputs-dir outputs/pretty_matrix   --task-names cat_crown,dog_sunglasses,mug_heart,tshirt_star,backpack_remove_toy_charm,red_chair_blue   --method-names support_v3_fixed,support_v3_controller_rmsgap   --seeds 10,11,12   --eval-mask-dir experiments/support_v3_2026-05-11/eval_masks   --clip-model openai/clip-vit-base-patch32   --dino-model facebook/dinov2-small   --csv-output experiments/support_v3_2026-05-11/core6_fixed_control_metrics.csv   --json-output experiments/support_v3_2026-05-11/core6_fixed_control_metrics.json
cp experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.csv experiments/support_v3_2026-05-11/core6_full_rerun_20260601_064458_main_metrics.csv
cp experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.json experiments/support_v3_2026-05-11/core6_full_rerun_20260601_064458_main_metrics.json
cp experiments/support_v3_2026-05-11/core6_fixed_control_metrics.csv experiments/support_v3_2026-05-11/core6_full_rerun_20260601_064458_fixed_control_metrics.csv
cp experiments/support_v3_2026-05-11/core6_fixed_control_metrics.json experiments/support_v3_2026-05-11/core6_full_rerun_20260601_064458_fixed_control_metrics.json
echo "completed_utc=2026-06-01T06:44:58Z" >> experiments/support_v3_2026-05-11/logs/core6_full_rerun_20260601_064458/protocol.txt
