#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J rem2-gen
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/rem2_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/rem2_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
RUN_ID="rem2_${SLURM_JOB_ID:-manual}"
RUN_DIR="$BASE/generation_matrix/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
SUMMARY="$RUN_DIR/summary.csv"
HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HF_HUB_CACHE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -s "$HOME/.cache/huggingface/token" ]]; then
  clean_hf_token="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN="$clean_hf_token"
  export HUGGINGFACE_HUB_TOKEN="$clean_hf_token"
  export HUGGING_FACE_HUB_TOKEN="$clean_hf_token"
fi

mkdir -p "$LOG_DIR"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run generation smoke outside a100-01" >&2
  exit 2
fi

printf 'baseline,status,output,log,note\n' > "$SUMMARY"

csv_escape() {
  python3 - "$1" <<'PY'
import csv
import io
import sys
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow([sys.argv[1]])
print(buf.getvalue().strip())
PY
}

record() {
  local baseline="$1"
  local status="$2"
  local output="$3"
  local log="$4"
  local note="$5"
  printf '%s,%s,%s,%s,%s\n' \
    "$(csv_escape "$baseline")" \
    "$(csv_escape "$status")" \
    "$(csv_escape "$output")" \
    "$(csv_escape "$log")" \
    "$(csv_escape "$note")" >> "$SUMMARY"
  echo "[$baseline] $status :: $note"
}

run_logged() {
  local baseline="$1"
  local timeout_s="$2"
  local log="$3"
  shift 3
  echo "==== $baseline" > "$log"
  echo "date=$(date)" >> "$log"
  echo "cwd=$(pwd)" >> "$log"
  timeout "$timeout_s" "$@" >> "$log" 2>&1
}

run_prompt_to_prompt() {
  local baseline=prompt-to-prompt
  local out="$RUN_DIR/prompt_to_prompt/result.png"
  local log="$LOG_DIR/prompt_to_prompt.log"
  mkdir -p "$(dirname "$out")"
  (
    cd "$SRC/prompt-to-prompt" || exit 1
    run_logged "$baseline" 30m "$log" \
      "$BASE/envs/prompt-to-prompt-py310/bin/python" - "$out" <<'PY'
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import ptp_utils

out = sys.argv[1]
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
_set_timesteps = pipe.scheduler.set_timesteps
def _set_timesteps_compat(num_inference_steps, **kwargs):
    return _set_timesteps(num_inference_steps)
pipe.scheduler.set_timesteps = _set_timesteps_compat
generator = torch.Generator(device="cpu").manual_seed(10)
images, _ = ptp_utils.text2image_ldm_stable(
    pipe,
    ["A painting of a cat", "A painting of a lion"],
    controller=None,
    num_inference_steps=2,
    guidance_scale=7.5,
    generator=generator,
)
Image.fromarray(images[-1]).save(out)
PY
  )
  local status=$?
  if [[ "$status" -eq 0 && -s "$out" ]]; then
    record "$baseline" "complete" "$out" "$log" "Prompt-to-Prompt generated an image."
  else
    record "$baseline" "failed_runtime" "$out" "$log" "Exit status $status."
  fi
}

run_ledits() {
  local baseline=ledits_pp
  local log="$LOG_DIR/ledits_pp.log"
  run_logged "$baseline" 45m "$log" \
    "$BASE/envs/ledits-pp-py310/bin/python" "$PROJECT/scripts/run_leditspp_baseline.py" \
    --root "$PROJECT" \
    --outputs-dir "$RUN_DIR/ledits_pp_outputs" \
    --tasks mug_heart \
    --seeds 10 \
    --image-size 512 \
    --num-inversion-steps 10 \
    --allow-download
  local status=$?
  local result
  result="$(find "$RUN_DIR/ledits_pp_outputs" -type f -name 'result.png' 2>/dev/null | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "LEDITS++ runner generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_prompt_to_prompt
run_ledits

echo "summary=$SUMMARY"
cat "$SUMMARY"
