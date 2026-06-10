#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=24:00:00
#SBATCH -J e2_t1t4_one
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_t1_t4_one_%x_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_t1_t4_one_%x_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
MANIFEST="$PROJECT/experiments/support_v3_2026-06-02/e2_t1_t4_formal_baseline_manifest.csv"
export HF_HOME="$PROJECT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$PROJECT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT/.cache/huggingface/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export SPLITFLOW_DISABLE_LLM=1
export DISABLE_BASELINE_NSFW=1
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
SEEDS="10 11 12"
BASELINE="${BASELINE:?set BASELINE}"
INIT_MANIFEST="${INIT_MANIFEST:-0}"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
echo "baseline=$BASELINE"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

mkdir -p "$BASE/logs"
if [[ "$INIT_MANIFEST" == "1" ]]; then
  python3 -m py_compile scripts/init_e2_t1_t4_formal_manifest.py
  python3 scripts/init_e2_t1_t4_formal_manifest.py \
    --out "$MANIFEST" \
    --baselines "flowedit flowalign splitflow fireflow rf_solver_edit reflex" \
    --tasks "$TASKS" \
    --seeds "$SEEDS"
else
  test -f "$MANIFEST"
fi

case "$BASELINE" in
  flowedit)
    python3 -m py_compile scripts/archive_legacy_2026-05-11/run_flowedit_baseline.py
    "$BASE/envs/flowedit-py310/bin/python" scripts/archive_legacy_2026-05-11/run_flowedit_baseline.py \
      --manifest "$MANIFEST" \
      --flowedit-root "$BASE/src/FlowEdit" \
      --python "$BASE/envs/flowedit-py310/bin/python" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --max-image-size 512 \
      --skip-complete
    ;;
  flowalign)
    python3 -m py_compile scripts/run_flowalign_baseline.py
    "$BASE/envs/flowalign-py310/bin/python" scripts/run_flowalign_baseline.py \
      --manifest "$MANIFEST" \
      --flowalign-root "$BASE/src/FlowAlign" \
      --python "$BASE/envs/flowalign-py310/bin/python" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --max-image-size 512 \
      --skip-complete
    ;;
  splitflow)
    python3 -m py_compile scripts/archive_legacy_2026-05-11/run_splitflow_baseline.py
    "$BASE/envs/splitflow-py310/bin/python" scripts/archive_legacy_2026-05-11/run_splitflow_baseline.py \
      --manifest "$MANIFEST" \
      --splitflow-root "$BASE/src/SplitFlow" \
      --python "$BASE/envs/splitflow-py310/bin/python" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --max-image-size 512 \
      --skip-complete
    ;;
  fireflow)
    python3 -m py_compile scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py
    "$BASE/envs/fireflow-py310/bin/python" scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py \
      --manifest "$MANIFEST" \
      --fireflow-root "$BASE/src/FireFlow" \
      --python "$BASE/envs/fireflow-py310/bin/python" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --max-image-size 512 \
      --skip-complete
    ;;
  rf_solver_edit)
    python3 -m py_compile scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py
    "$BASE/envs/rf-solver-edit-py310/bin/python" scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py \
      --manifest "$MANIFEST" \
      --rf-solver-root "$BASE/src/RF-Solver-Edit/FLUX_Image_Edit" \
      --python "$BASE/envs/rf-solver-edit-py310/bin/python" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --max-image-size 512 \
      --skip-complete
    ;;
  reflex)
    python3 -m py_compile scripts/archive_legacy_2026-05-11/run_reflex_baseline.py
    export REFLEX_LOCAL_FILES_ONLY=1
    "$BASE/envs/reflex-py310/bin/python" scripts/archive_legacy_2026-05-11/run_reflex_baseline.py \
      --manifest "$MANIFEST" \
      --reflex-root "$BASE/src/ReFlex" \
      --python "$BASE/envs/reflex-py310/bin/python" \
      --tasks "$TASKS" \
      --seeds "$SEEDS" \
      --max-image-size 512 \
      --skip-complete
    ;;
  *)
    echo "Unknown BASELINE=$BASELINE" >&2
    exit 2
    ;;
esac

date
