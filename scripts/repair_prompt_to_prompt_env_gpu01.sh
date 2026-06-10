#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:30:00
#SBATCH -J repair-ptp
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/repair_prompt_to_prompt_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/repair_prompt_to_prompt_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
LOG_DIR="$BASE/logs/repair_prompt_to_prompt_${SLURM_JOB_ID:-manual}"
SUMMARY="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to repair prompt-to-prompt outside a100-01" >&2
  exit 2
fi

run_step() {
  local name="$1"
  local log="$LOG_DIR/$name.log"
  shift
  echo "==== $name" > "$log"
  echo "cmd=$*" >> "$log"
  if "$@" >> "$log" 2>&1; then
    echo "$name,ok,$log" >> "$SUMMARY"
    echo "[$name] ok"
  else
    local status=$?
    echo "$name,failed:$status,$log" >> "$SUMMARY"
    echo "[$name] failed:$status"
  fi
}

touch "$SUMMARY"

run_step prompt_to_prompt_pin \
  "$BASE/envs/prompt-to-prompt-py310/bin/python" -m pip install \
  "diffusers==0.13.1" "transformers==4.26.1" \
  "tokenizers==0.13.3" "huggingface-hub==0.13.4" "accelerate==0.18.0"

run_step prompt_to_prompt_import \
  "$BASE/envs/prompt-to-prompt-py310/bin/python" - <<'PY'
import diffusers
from diffusers import StableDiffusionPipeline, DDIMScheduler
print("prompt-to-prompt import ok", diffusers.__version__)
PY

echo "summary=$SUMMARY"
cat "$SUMMARY"
