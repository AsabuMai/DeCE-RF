#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=04:00:00
#SBATCH -J gen-matrix16
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/matrix16_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/matrix16_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
OUT_ROOT="$BASE/generation_matrix"
RUN_ID="matrix16_${SLURM_JOB_ID:-manual}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
SUMMARY="$RUN_DIR/summary.csv"
SOURCE_IMAGE="$PROJECT/data/paper_images/cat_sitting_in_grass.jpg"
SOURCE_PROMPT="A photo of a cat sitting in grass."
TARGET_PROMPT="A photo of the same cat sitting in the same grass, wearing a small golden crown on its head."
HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HF_HUB_CACHE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$HF_HUB_CACHE"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run generation matrix outside a100-01" >&2
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
  echo "cwd=$(pwd)" >> "$log"
  timeout "$timeout_s" "$@" >> "$log" 2>&1
}

existing_result() {
  local baseline="$1"
  local output="$2"
  if [[ -s "$output" ]]; then
    record "$baseline" "existing_complete" "$output" "" "Existing real generation output found."
  else
    record "$baseline" "missing_existing_output" "$output" "" "Expected existing output was not found."
  fi
}

existing_result "FlowEdit" "$PROJECT/outputs/baselines/flowedit/cat_crown/seed_10/result.png"
existing_result "FlowAlign" "$PROJECT/outputs/baselines/flowalign/cat_crown/seed_10/result.png"
existing_result "SplitFlow" "$PROJECT/outputs/baselines/splitflow/cat_crown/seed_10/result.png"
existing_result "OT-RF" "$SRC/OT-RF/outputs/smoke_otrf_sd3/SD3/src_face_input/tar_0/output_T_steps_4_n_avg_1_cfg_enc_3.5_cfg_dec7.5_n_min_0_n_max_2_seed42.png"
existing_result "DeltaRectifiedFlowSampling" "$SRC/DeltaRectifiedFlowSampling/outputs/smoke_dvrf_sd3/SD3/src_a_cat_sitting_on_a_table/tgt_0/0.01_eta_1.0_descendingSGDT_steps_4_num_steps_1_cfg_enc_3.5_cfg_dec7.5_seed41.png"

run_fireflow() {
  local baseline=FireFlow
  local out_dir="$RUN_DIR/fireflow"
  local log="$LOG_DIR/fireflow.log"
  mkdir -p "$out_dir"
  (
    cd "$SRC/FireFlow/src" || exit 1
    run_logged "$baseline" 45m "$log" \
      "$BASE/envs/fireflow-py310/bin/python" edit.py \
      --source_prompt "$SOURCE_PROMPT" \
      --target_prompt "$TARGET_PROMPT" \
      --guidance 2 \
      --source_img_dir "$SOURCE_IMAGE" \
      --num_steps 2 \
      --inject 1 \
      --start_layer_index 0 \
      --end_layer_index 37 \
      --name flux-dev \
      --sampling_strategy fireflow \
      --output_prefix smoke_fireflow \
      --output_dir "$out_dir" \
      --feature_path "$out_dir/features" \
      --offload \
      --seed 10
  )
  local status=$?
  local result
  result="$(find "$out_dir" -type f \( -name '*.jpg' -o -name '*.png' \) | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "Official FLUX-dev entrypoint generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_rf_solver() {
  local baseline=RF-Solver-Edit
  local out_dir="$RUN_DIR/rf_solver_edit"
  local log="$LOG_DIR/rf_solver_edit.log"
  mkdir -p "$out_dir"
  (
    cd "$SRC/RF-Solver-Edit/FLUX_Image_Edit/src" || exit 1
    run_logged "$baseline" 45m "$log" \
      "$BASE/envs/rf-solver-edit-py310/bin/python" edit.py \
      --source_prompt "$SOURCE_PROMPT" \
      --target_prompt "$TARGET_PROMPT" \
      --guidance 2 \
      --source_img_dir "$SOURCE_IMAGE" \
      --num_steps 2 \
      --inject 3 \
      --name flux-dev \
      --output_dir "$out_dir" \
      --feature_path "$out_dir/features" \
      --offload
  )
  local status=$?
  local result
  result="$(find "$out_dir" -type f \( -name '*.jpg' -o -name '*.png' \) | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "Official FLUX-dev entrypoint generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_reflex() {
  local baseline=ReFlex
  local out_dir="$RUN_DIR/reflex"
  local log="$LOG_DIR/reflex.log"
  mkdir -p "$out_dir"
  run_logged "$baseline" 45m "$log" \
    "$BASE/envs/reflex-py310/bin/python" "$SRC/ReFlex/img_edit.py" \
    --gpu 0 \
    --seed 10 \
    --img_path "$SOURCE_IMAGE" \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --results_dir "$out_dir" \
    --feature_steps 1 \
    --attn_topk 5
  local status=$?
  local result
  result="$(find "$out_dir" -type f -name 'target_0.png' | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "Official FLUX-dev entrypoint generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_stable_flow() {
  local baseline=stable-flow
  local out="$RUN_DIR/stable_flow/result.jpg"
  local log="$LOG_DIR/stable_flow.log"
  mkdir -p "$(dirname "$out")"
  local token_file="$HOME/.cache/huggingface/token"
  if [[ ! -s "$token_file" ]]; then
    record "$baseline" "missing_token" "" "$log" "No Hugging Face token at default location."
    return
  fi
  local token
  token="$(cat "$token_file")"
  (
    cd "$SRC/stable-flow" || exit 1
    run_logged "$baseline" 45m "$log" \
      "$BASE/envs/stable-flow-py311/bin/python" run_stable_flow.py \
      --hf_token "$token" \
      --prompts "$SOURCE_PROMPT" "$TARGET_PROMPT" \
      --output_path "$out" \
      --seed 10 \
      --cpu_offload
  )
  local status=$?
  if [[ "$status" -eq 0 && -s "$out" ]]; then
    record "$baseline" "complete" "$out" "$log" "Stable Flow generated an image grid."
  else
    record "$baseline" "failed_runtime" "$out" "$log" "Exit status $status."
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
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator(device="cuda").manual_seed(10)
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
    record "$baseline" "complete" "$out" "$log" "Prompt-to-Prompt utility generated an image."
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
  result="$(find "$out_dir" -type f -name '*.png' | head -1)"
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
    --num-inversion-steps 2 \
    --allow-download
  local status=$?
  local result
  result="$(find "$RUN_DIR/ledits_pp_outputs" -type f -name 'result.png' | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "LEDITS++ runner generated an image."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$log" "Exit status $status."
  fi
}

run_zone() {
  local baseline=ZONE
  local sam="$SRC/ZONE/ckpts/sam_vit_h_4b8939.pth"
  local log="$LOG_DIR/zone.log"
  if [[ ! -s "$sam" ]]; then
    record "$baseline" "missing_checkpoint" "" "$log" "Missing SAM checkpoint $sam."
    return
  fi
  run_logged "$baseline" 45m "$log" \
    "$BASE/envs/zone-py310/bin/python" "$SRC/ZONE/inference.py" \
    --instruction "make the cat wear a golden crown" \
    --image_path "$SOURCE_IMAGE" \
    --sam_ckpt_path "$sam" \
    --output_path "$RUN_DIR/zone" \
    --inference_steps 2 \
    --seed 10
  local status=$?
  local result
  result="$(find "$RUN_DIR/zone" -type f -name '*.png' | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$log" "ZONE generated an image."
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
run_reflex
run_stable_flow
run_fireflow
run_rf_solver

echo "summary=$SUMMARY"
cat "$SUMMARY"
