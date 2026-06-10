#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=03:00:00
#SBATCH -J repaired-gen
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/repaired_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/repaired_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
OUT_ROOT="$BASE/generation_matrix"
RUN_ID="repaired_${SLURM_JOB_ID:-manual}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
SUMMARY="$RUN_DIR/summary.csv"
SOURCE_IMAGE="$PROJECT/data/paper_images/cat_sitting_in_grass_1024.jpg"
HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HF_HUB_CACHE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -s "$HOME/.cache/huggingface/token" ]]; then
  clean_hf_token="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN="$clean_hf_token"
  export HUGGINGFACE_HUB_TOKEN="$clean_hf_token"
  export HUGGING_FACE_HUB_TOKEN="$clean_hf_token"
fi

mkdir -p "$LOG_DIR" "$HF_HUB_CACHE"
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

run_zone() {
  local baseline=ZONE
  local sam="$SRC/ZONE/ckpts/sam_vit_h_4b8939.pth"
  local log="$LOG_DIR/zone.log"
  if [[ ! -s "$sam" ]]; then
    record "$baseline" "missing_checkpoint" "" "$log" "Missing SAM checkpoint $sam."
    return
  fi
  (
    cd "$SRC/ZONE" || exit 1
    run_logged "$baseline" 45m "$log" \
      "$BASE/envs/zone-py310/bin/python" inference.py \
      --instruction "make the cat wear a golden crown" \
      --image_path "$SOURCE_IMAGE" \
      --sam_ckpt_path "$sam" \
      --output_path "$RUN_DIR/zone" \
      --inference_steps 2 \
      --seed 10
  )
  local status=$?
  local result
  result="$(find "$RUN_DIR/zone" -type f -name '*.png' 2>/dev/null | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "ZONE generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_instruct_pix2pix() {
  local baseline=instruct-pix2pix
  local out="$RUN_DIR/instruct_pix2pix/result.png"
  local log="$LOG_DIR/instruct_pix2pix.log"
  mkdir -p "$(dirname "$out")"
  run_logged "$baseline" 30m "$log" \
    "$BASE/envs/instruct-pix2pix-py39/bin/python" - "$SOURCE_IMAGE" "$out" <<'PY'
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

src, out = sys.argv[1], sys.argv[2]
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
image = Image.open(src).convert("RGB").resize((512, 512))
result = pipe(
    "add a small golden crown on the cat",
    image=image,
    num_inference_steps=2,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
).images[0]
result.save(out)
PY
  local status=$?
  if [[ "$status" -eq 0 && -s "$out" ]]; then
    record "$baseline" "complete" "$out" "$log" "Diffusers InstructPix2Pix pipeline generated an image."
  else
    record "$baseline" "failed_runtime" "$out" "$log" "Exit status $status."
  fi
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

run_hedit() {
  local baseline=h-edit
  local out_dir="$RUN_DIR/h_edit"
  local log="$LOG_DIR/h_edit.log"
  mkdir -p "$out_dir"
  (
    cd "$SRC/h-edit/text-guided" || exit 1
    run_logged "$baseline" 45m "$log" \
      "$BASE/envs/h-edit-py310/bin/python" main_demo.py \
      --device_num 0 \
      --data_path ./assets/demo \
      --output_path "$out_dir" \
      --num_diffusion_steps 2 \
      --optimization_steps 1
  )
  local status=$?
  local result
  result="$(find "$out_dir" -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "h-edit demo generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
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

run_pix2pix_zero() {
  local baseline=pix2pix-zero
  local out_dir="$RUN_DIR/pix2pix_zero"
  local log="$LOG_DIR/pix2pix_zero.log"
  mkdir -p "$out_dir"
  (
    cd "$SRC/pix2pix-zero" || exit 1
    run_logged "$baseline" 30m "$log" \
      "$BASE/envs/pix2pix-zero-py310/bin/python" src/edit_synthetic.py \
      --results_folder "$out_dir" \
      --prompt_str "a high resolution painting of a cat in the style of van gogh" \
      --task_name cat2dog \
      --num_ddim_steps 2 \
      --use_float_16
  )
  local status=$?
  if [[ "$status" -eq 0 && -s "$out_dir/edit.png" ]]; then
    record "$baseline" "complete" "$out_dir/edit.png" "$log" "Synthetic edit generated an image."
  else
    record "$baseline" "failed_runtime" "$out_dir/edit.png" "$log" "Exit status $status."
  fi
}

run_masactrl() {
  local baseline=MasaCtrl
  local log="$LOG_DIR/masactrl.log"
  (
    cd "$SRC/MasaCtrl" || exit 1
    run_logged "$baseline" 45m "$log" "$BASE/envs/masactrl-py310/bin/python" run_synthesis_sdxl.py
  )
  local status=$?
  local result
  result="$(find "$SRC/MasaCtrl/workdir/masactrl_exp" -type f -name '*.png' 2>/dev/null | tail -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "Official SDXL synthesis script generated images."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_zone
run_instruct_pix2pix
run_prompt_to_prompt
run_hedit
run_ledits
run_pix2pix_zero
run_masactrl

echo "summary=$SUMMARY"
cat "$SUMMARY"
