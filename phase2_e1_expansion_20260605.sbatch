#!/usr/bin/env bash
#SBATCH -J phase2_e1
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_e1_expansion_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

echo "[phase2-e1] start=$(date -Is)"
echo "[phase2-e1] host=$(hostname)"
echo "[phase2-e1] cwd=$(pwd)"
echo "[phase2-e1] job_id=${SLURM_JOB_ID:-none}"
echo "[phase2-e1] cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export SKIP_EXISTING=1
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512

TASKS_PHASE2_E1="
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
bowl_apple_inside
white_bowl_strawberry_phase2
brown_bowl_lemon_phase2
tshirt_star
mug_heart
tote_leaf
red_chair_blue
red_chair_product_blue_phase2
red_chair_restaurant_blue_phase2
pillow_vertical_fabric_strip
white_pillow_blue_dots_phase2
white_pillow_blue_cross_phase2
backpack_remove_toy_charm
backpack_remove_silver_keychain_phase2
bag_remove_decorative_tag_phase2
"

METHODS_PHASE2_E1="
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
"

SEEDS_PHASE2_E1="10 11 12"

echo "[phase2-e1] tasks=$(echo ${TASKS_PHASE2_E1})"
echo "[phase2-e1] methods=$(echo ${METHODS_PHASE2_E1})"
echo "[phase2-e1] seeds=${SEEDS_PHASE2_E1}"

TASKS="${TASKS_PHASE2_E1}" \
METHODS="${METHODS_PHASE2_E1}" \
SEEDS="${SEEDS_PHASE2_E1}" \
bash scripts/run_pretty_matrix.sh

echo "[phase2-e1] finished_generation=$(date -Is)"

"${PYTHON}" - <<'PY'
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
tasks = """
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
bowl_apple_inside
white_bowl_strawberry_phase2
brown_bowl_lemon_phase2
tshirt_star
mug_heart
tote_leaf
red_chair_blue
red_chair_product_blue_phase2
red_chair_restaurant_blue_phase2
pillow_vertical_fabric_strip
white_pillow_blue_dots_phase2
white_pillow_blue_cross_phase2
backpack_remove_toy_charm
backpack_remove_silver_keychain_phase2
bag_remove_decorative_tag_phase2
""".split()
methods = """
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
""".split()
seeds = "10 11 12".split()

missing = []
complete = 0
for task in tasks:
    for method in methods:
        for seed in seeds:
            d = root / "outputs" / "pretty_matrix" / task / method / f"seed_{seed}"
            needed = ["result.png", "metadata.json", "stats.json"]
            absent = [name for name in needed if not (d / name).exists()]
            if absent:
                missing.append((task, method, seed, absent))
            else:
                complete += 1

out = root / "experiments" / "support_v3_2026-06-02" / "phase2_e1_expansion_completion_20260605.txt"
with out.open("w", encoding="utf-8") as f:
    f.write(f"complete={complete}\n")
    f.write(f"expected={len(tasks) * len(methods) * len(seeds)}\n")
    for row in missing:
        f.write(repr(row) + "\n")

print(f"[phase2-e1] complete={complete}")
print(f"[phase2-e1] expected={len(tasks) * len(methods) * len(seeds)}")
print(f"[phase2-e1] missing={len(missing)}")
print(f"[phase2-e1] completion_file={out}")
PY

echo "[phase2-e1] end=$(date -Is)"
