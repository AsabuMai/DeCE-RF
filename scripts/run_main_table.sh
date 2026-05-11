#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"

TASK="${TASK:-T1}"
METHOD="${METHOD:-M4}"
SEED="${SEED:-10}"
TASKS="${TASKS:-${TASK}}"
METHODS="${METHODS:-${METHOD}}"
SEEDS="${SEEDS:-${SEED}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PHOTO_PROMPT_MODE="${PHOTO_PROMPT_MODE:-off}"
BASELINE_OBJECT_MASK_PROVIDER="${BASELINE_OBJECT_MASK_PROVIDER:-attention_velocity}"
MAIN_OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER:-semantic_velocity}"
GROUNDING_MODEL="${GROUNDING_MODEL:-IDEA-Research/grounding-dino-base}"
SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}"
ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD:-0}"
REGENERATE_SUPPORT_MASK="${REGENERATE_SUPPORT_MASK:-0}"
REGENERATE_SURFACE_REF="${REGENERATE_SURFACE_REF:-0}"
SURFACE_REF_MODE="${SURFACE_REF_MODE:-auto}"
SURFACE_REF_GUIDANCE_SCALE="${SURFACE_REF_GUIDANCE_SCALE:-0.20}"
SURFACE_REF_RECOLOR_MODE="${SURFACE_REF_RECOLOR_MODE:-yuv-chroma}"
SURFACE_REF_BLEND="${SURFACE_REF_BLEND:-0.92}"
SURFACE_REF_MASK_BLUR="${SURFACE_REF_MASK_BLUR:-5}"
METHOD_NAME_SUFFIX="${METHOD_NAME_SUFFIX:-}"
FORCE_OBJECT_MASK_PROVIDER="${FORCE_OBJECT_MASK_PROVIDER:-}"
FORCE_MASK_LAYERING_MODE="${FORCE_MASK_LAYERING_MODE:-}"
FORCE_EDIT_HEDIT_GUIDANCE_SCALE="${FORCE_EDIT_HEDIT_GUIDANCE_SCALE:-}"
FORCE_EDIT_TEXT_GUIDANCE_SCALE="${FORCE_EDIT_TEXT_GUIDANCE_SCALE:-}"
FORCE_REC_GUIDANCE_SCALE="${FORCE_REC_GUIDANCE_SCALE:-}"
FORCE_TRAJECTORY_PRESERVE_SCALE="${FORCE_TRAJECTORY_PRESERVE_SCALE:-}"
FORCE_STRUCT_GUIDANCE_SCALE="${FORCE_STRUCT_GUIDANCE_SCALE:-}"
SOURCE_INJECT_Q_SCALE="${SOURCE_INJECT_Q_SCALE:-0.0}"
SOURCE_INJECT_K_SCALE="${SOURCE_INJECT_K_SCALE:-0.0}"
SOURCE_INJECT_V_SCALE="${SOURCE_INJECT_V_SCALE:-0.0}"
SOURCE_INJECT_LAYER_FROM="${SOURCE_INJECT_LAYER_FROM:--1}"
SOURCE_INJECT_LAYER_TO="${SOURCE_INJECT_LAYER_TO:--1}"
SOURCE_INJECT_STEPS="${SOURCE_INJECT_STEPS:-8}"
SOURCE_INJECT_MASK_MODE="${SOURCE_INJECT_MASK_MODE:-none}"
SOURCE_INJECT_MASK_BOX="${SOURCE_INJECT_MASK_BOX:-}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
N_MAX="${N_MAX:-24}"

task_config() {
  local task_id="$1"
  TASK_SURFACE_REF_ENABLED="0"
  SURFACE_REF_SOURCE_COLOR=""
  SURFACE_REF_TARGET_COLOR=""
  SURFACE_REF_NAME=""
  SURFACE_REF_MASK_KIND="support"
  case "${task_id}" in
    T1|cat_crown)
      TASK_NAME="cat_crown"
      IMAGE="${ROOT}/data/paper_images/cat_sitting_in_grass.jpg"
      SOURCE_PROMPT="A photo of a cat sitting in grass."
      TARGET_PROMPT="A photo of the same cat sitting in the same grass, wearing a small golden crown on its head."
      OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER}"
      MASK_LAYERING_MODE="object_contact"
      ATTENTION_TARGET_WORDS="crown,head"
      SEMANTIC_PHRASE=""
      SUPPORT_RELATION="auto"
      ;;
    T2|backpack_blue)
      TASK_NAME="backpack_blue"
      IMAGE="${ROOT}/data/paper_images/herschel_backpack_by_rocks_unsplash.jpg"
      SOURCE_PROMPT="A photo of a burgundy backpack sitting on rocks outdoors."
      TARGET_PROMPT="A photo of the same backpack sitting on the same rocks outdoors, with the backpack fabric changed to clean blue."
      OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER}"
      MASK_LAYERING_MODE="object_contact"
      ATTENTION_TARGET_WORDS="backpack,blue"
      SEMANTIC_PHRASE=""
      SUPPORT_RELATION="auto"
      TASK_SURFACE_REF_ENABLED="1"
      SURFACE_REF_TARGET_COLOR="blue"
      SURFACE_REF_NAME="backpack"
      SURFACE_REF_MASK_KIND="support"
      ;;
    T3|yellow_car_blue)
      TASK_NAME="yellow_car_blue"
      IMAGE="${ROOT}/data/paper_images/yellow_car_side_unsplash.jpg"
      SOURCE_PROMPT="A photo of a yellow car parked on a street."
      TARGET_PROMPT="A photo of the same car parked on the same street, with the car body changed to clean blue."
      OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER}"
      MASK_LAYERING_MODE="object_contact"
      ATTENTION_TARGET_WORDS="car,blue"
      SEMANTIC_PHRASE=""
      SUPPORT_RELATION="auto"
      TASK_SURFACE_REF_ENABLED="1"
      SURFACE_REF_SOURCE_COLOR="yellow"
      SURFACE_REF_TARGET_COLOR="blue"
      SURFACE_REF_NAME="car body"
      SURFACE_REF_MASK_KIND="vehicle_paint"
      ;;
    T4|rabbit_sunglasses)
      TASK_NAME="rabbit_sunglasses"
      IMAGE="${ROOT}/data/paper_images/rabbit_side_view.jpg"
      SOURCE_PROMPT="A photo of a rabbit sitting outdoors in side profile."
      TARGET_PROMPT="A photo of the same rabbit sitting outdoors in side profile, wearing small black sunglasses."
      OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER}"
      MASK_LAYERING_MODE="object_contact"
      ATTENTION_TARGET_WORDS="sunglasses,eyes"
      SEMANTIC_PHRASE=""
      SUPPORT_RELATION="auto"
      ;;
    T5|red_chair_blue)
      TASK_NAME="red_chair_blue"
      IMAGE="${ROOT}/data/paper_images/red_chair_cc0.jpg"
      SOURCE_PROMPT="A photo of a red chair indoors."
      TARGET_PROMPT="A photo of the same chair indoors, with the chair changed to clean blue."
      OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER}"
      MASK_LAYERING_MODE="object_contact"
      ATTENTION_TARGET_WORDS="chair,blue"
      SEMANTIC_PHRASE=""
      SUPPORT_RELATION="auto"
      TASK_SURFACE_REF_ENABLED="1"
      SURFACE_REF_TARGET_COLOR="blue"
      SURFACE_REF_NAME="chair"
      SURFACE_REF_MASK_KIND="support"
      ;;
    T6|dog_crown)
      TASK_NAME="dog_crown"
      IMAGE="${ROOT}/data/paper_images/dog_sitting_cc0.jpg"
      SOURCE_PROMPT="A photo of a dog sitting."
      TARGET_PROMPT="A photo of the same dog sitting, wearing a small golden crown on its head."
      OBJECT_MASK_PROVIDER="${MAIN_OBJECT_MASK_PROVIDER}"
      MASK_LAYERING_MODE="object_contact"
      ATTENTION_TARGET_WORDS="crown,head"
      SEMANTIC_PHRASE=""
      SUPPORT_RELATION="auto"
      ;;
    *)
      echo "Unknown TASK '${task_id}'. Valid: T1 T2 T3 T4 T5 T6." >&2
      exit 2
      ;;
  esac
}

configure_task_method() {
  if [[ "${METHOD_NAME}" != "full" ]]; then
    OBJECT_MASK_PROVIDER="${BASELINE_OBJECT_MASK_PROVIDER}"
  fi
  if [[ "${SURFACE_REF_MODE}" != "off" && "${TASK_SURFACE_REF_ENABLED}" == "1" && "${METHOD_NAME}" == "full" ]]; then
    EDIT_REF_GUIDANCE_SCALE="${SURFACE_REF_GUIDANCE_SCALE}"
  fi
  if [[ -n "${FORCE_OBJECT_MASK_PROVIDER}" ]]; then
    OBJECT_MASK_PROVIDER="${FORCE_OBJECT_MASK_PROVIDER}"
  fi
  if [[ -n "${FORCE_MASK_LAYERING_MODE}" ]]; then
    MASK_LAYERING_MODE="${FORCE_MASK_LAYERING_MODE}"
  fi
  if [[ -n "${FORCE_EDIT_HEDIT_GUIDANCE_SCALE}" ]]; then
    EDIT_HEDIT_GUIDANCE_SCALE="${FORCE_EDIT_HEDIT_GUIDANCE_SCALE}"
  fi
  if [[ -n "${FORCE_EDIT_TEXT_GUIDANCE_SCALE}" ]]; then
    EDIT_TEXT_GUIDANCE_SCALE="${FORCE_EDIT_TEXT_GUIDANCE_SCALE}"
  fi
  if [[ -n "${FORCE_REC_GUIDANCE_SCALE}" ]]; then
    REC_GUIDANCE_SCALE="${FORCE_REC_GUIDANCE_SCALE}"
  fi
  if [[ -n "${FORCE_TRAJECTORY_PRESERVE_SCALE}" ]]; then
    TRAJECTORY_PRESERVE_SCALE="${FORCE_TRAJECTORY_PRESERVE_SCALE}"
  fi
  if [[ -n "${FORCE_STRUCT_GUIDANCE_SCALE}" ]]; then
    STRUCT_GUIDANCE_SCALE="${FORCE_STRUCT_GUIDANCE_SCALE}"
  fi
  if [[ -n "${METHOD_NAME_SUFFIX}" ]]; then
    METHOD_NAME="${METHOD_NAME}_${METHOD_NAME_SUFFIX}"
  fi
}

method_config() {
  local method_id="$1"

  SRC_GUIDANCE_SCALE="1.0"
  BASE_GUIDANCE_SCALE="1.0"
  TAR_GUIDANCE_SCALE="10.5"
  EDIT_HEDIT_GUIDANCE_SCALE="0.0"
  EDIT_GUIDANCE_SCALE="0.0"
  EDIT_REGION_GUIDANCE_SCALE="0.0"
  EDIT_TARGET_GUIDANCE_SCALE="0.0"
  EDIT_SOURCE_GUIDANCE_SCALE="0.0"
  EDIT_CLIP_GUIDANCE_SCALE="0.0"
  EDIT_TEXT_GUIDANCE_SCALE="0.0"
  EDIT_DDS_GUIDANCE_SCALE="0.0"
  EDIT_APP_GUIDANCE_SCALE="0.0"
  EDIT_COLOR_GUIDANCE_SCALE="0.0"
  EDIT_REF_GUIDANCE_SCALE="0.0"
  REC_GUIDANCE_SCALE="0.0"
  STRUCT_GUIDANCE_SCALE="0.5"
  TRAJECTORY_PRESERVE_SCALE="0.0"

  case "${method_id}" in
    M0|base_only)
      METHOD_NAME="base_only"
      ;;
    M1|direct_target)
      METHOD_NAME="direct_target"
      EDIT_HEDIT_GUIDANCE_SCALE="1.0"
      ;;
    M2|anchor_only)
      METHOD_NAME="anchor_only"
      EDIT_GUIDANCE_SCALE="1.0"
      ;;
    M3|decoupled_rec)
      METHOD_NAME="decoupled_rec"
      EDIT_HEDIT_GUIDANCE_SCALE="1.0"
      REC_GUIDANCE_SCALE="0.25"
      ;;
    M4|full)
      METHOD_NAME="full"
      EDIT_HEDIT_GUIDANCE_SCALE="0.55"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.25"
      TRAJECTORY_PRESERVE_SCALE="0.15"
      ;;
    *)
      echo "Unknown METHOD '${method_id}'. Valid: M0 M1 M2 M3 M4." >&2
      exit 2
      ;;
  esac
}

ensure_support_mask() {
  local out_dir="$1"
  SUPPORT_MASK=""
  if [[ "${OBJECT_MASK_PROVIDER}" != "semantic" && "${OBJECT_MASK_PROVIDER}" != "semantic_velocity" ]]; then
    return 0
  fi

  SUPPORT_MASK="${out_dir}/masks/semantic_base_generated.png"
  local support_meta="${out_dir}/masks/semantic_base_generated.json"
  local anchor_mask="${out_dir}/masks/semantic_anchor_generated.png"
  if [[ "${REGENERATE_SUPPORT_MASK}" != "1" && -s "${SUPPORT_MASK}" ]]; then
    return 0
  fi

  local mask_cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_semantic_mask.py"
    --image "${IMAGE}"
    --source-prompt "${SOURCE_PROMPT}"
    --prompt "${TARGET_PROMPT}"
    --output "${SUPPORT_MASK}"
    --metadata-output "${support_meta}"
    --anchor-output "${anchor_mask}"
    --device "cuda:${DEVICE}"
    --grounding-model "${GROUNDING_MODEL}"
    --sam-model "${SAM_MODEL}"
    --support-relation "${SUPPORT_RELATION}"
  )
  if [[ -n "${SEMANTIC_PHRASE}" ]]; then
    mask_cmd+=(--phrase "${SEMANTIC_PHRASE}")
  fi
  if [[ "${ALLOW_MASK_DOWNLOAD}" == "1" ]]; then
    mask_cmd+=(--allow-download)
  fi

  printf '%q ' "${mask_cmd[@]}" > "${out_dir}/mask_command.txt"
  printf '\n' >> "${out_dir}/mask_command.txt"
  echo "[main-matrix] generating SAM support: ${SUPPORT_MASK}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/mask_command.txt"
    return 0
  fi
  "${mask_cmd[@]}"
}

ensure_surface_reference() {
  local out_dir="$1"
  EDIT_REF_IMAGE=""
  EDIT_REF_MASK=""
  EDIT_REF_STRUCTURE_IMAGE=""
  EDIT_REF_CHROMA_MODE="yuv"
  EDIT_REF_CHROMA_MAGNITUDE_SCALE="1.0"
  EDIT_REF_LUMA_PRESERVE_SCALE="0.35"
  EDIT_REF_GRADIENT_PRESERVE_SCALE="0.15"
  EDIT_REF_DARKNESS_GUARD_SCALE="0.0"
  EDIT_REF_DARKNESS_GUARD_MARGIN="0.03"
  EDIT_REF_SMOOTH_KERNEL="1"
  EDIT_REF_LOWFREQ_SUPPRESS_KERNEL="0"
  EDIT_REF_LOWFREQ_SUPPRESS_SCALE="0.0"
  EDIT_REF_SCHEDULE_START="0.0"
  EDIT_REF_SCHEDULE_STOP="0.0"
  EDIT_REF_SCHEDULE_POWER="1.0"
  EDIT_REF_MAX_STRUCT_RMS_RATIO="0.0"
  EDIT_REF_PROJECT_STRUCT_CONFLICT="0.0"

  if [[ "${EDIT_REF_GUIDANCE_SCALE}" == "0.0" || "${EDIT_REF_GUIDANCE_SCALE}" == "0" ]]; then
    return 0
  fi
  if [[ "${TASK_SURFACE_REF_ENABLED}" != "1" || "${SURFACE_REF_MODE}" == "off" ]]; then
    return 0
  fi

  local ref_mask=""
  if [[ "${SURFACE_REF_MASK_KIND}" == "vehicle_paint" ]]; then
    local paint_dir="${out_dir}/masks/vehicle_paint"
    ref_mask="${paint_dir}/paint_mask.png"
    local paint_cmd=(
      "${PYTHON}" "${ROOT}/scripts/make_vehicle_paint_mask.py"
      --image "${IMAGE}"
      --output-dir "${paint_dir}"
      --vehicle-phrase "car"
      --source-color "${SURFACE_REF_SOURCE_COLOR}"
      --color-mode "soft"
      --device "cuda:${DEVICE}"
      --grounding-model "${GROUNDING_MODEL}"
      --sam-model "${SAM_MODEL}"
    )
    if [[ "${ALLOW_MASK_DOWNLOAD}" == "1" ]]; then
      paint_cmd+=(--allow-download)
    fi
    printf '%q ' "${paint_cmd[@]}" > "${out_dir}/surface_ref_command.txt"
    printf '\n' >> "${out_dir}/surface_ref_command.txt"
    if [[ "${REGENERATE_SURFACE_REF}" == "1" || ! -s "${ref_mask}" ]]; then
      echo "[main-matrix] generating vehicle paint mask: ${ref_mask}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        cat "${out_dir}/surface_ref_command.txt"
      else
        "${paint_cmd[@]}"
      fi
    fi
  else
    ref_mask="${SUPPORT_MASK}"
    if [[ -z "${ref_mask}" ]]; then
      echo "ERROR: surface reference for ${TASK_NAME} requires a support mask." >&2
      exit 2
    fi
    : > "${out_dir}/surface_ref_command.txt"
  fi

  EDIT_REF_IMAGE="${out_dir}/masks/surface_recolor_reference.png"
  EDIT_REF_MASK="${ref_mask}"
  EDIT_REF_STRUCTURE_IMAGE="${IMAGE}"
  local ref_meta="${out_dir}/masks/surface_recolor_reference.json"
  local ref_overlay="${out_dir}/masks/surface_recolor_reference_overlay.png"
  local ref_cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_surface_recolor_reference.py"
    --image "${IMAGE}"
    --surface-mask "${ref_mask}"
    --output "${EDIT_REF_IMAGE}"
    --target-color "${SURFACE_REF_TARGET_COLOR}"
    --luma-image "${IMAGE}"
    --mode "${SURFACE_REF_RECOLOR_MODE}"
    --blend "${SURFACE_REF_BLEND}"
    --mask-blur "${SURFACE_REF_MASK_BLUR}"
    --surface-name "${SURFACE_REF_NAME}"
    --overlay-output "${ref_overlay}"
    --metadata-output "${ref_meta}"
  )
  printf '%q ' "${ref_cmd[@]}" >> "${out_dir}/surface_ref_command.txt"
  printf '\n' >> "${out_dir}/surface_ref_command.txt"
  if [[ "${REGENERATE_SURFACE_REF}" == "1" || ! -s "${EDIT_REF_IMAGE}" ]]; then
    echo "[main-matrix] generating surface recolor reference: ${EDIT_REF_IMAGE}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      tail -n 1 "${out_dir}/surface_ref_command.txt"
    else
      "${ref_cmd[@]}"
    fi
  fi
}

run_one() {
  local task_id="$1"
  local method_id="$2"
  local seed="$3"

  task_config "${task_id}"
  method_config "${method_id}"
  configure_task_method

  local out_dir="${ROOT}/outputs/main_matrix/${TASK_NAME}/${METHOD_NAME}/seed_${seed}"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${out_dir}/result.png" && -s "${out_dir}/stats.json" && -s "${out_dir}/metadata.json" && -s "${out_dir}/command.txt" ]]; then
    echo "[main-matrix] skip existing complete run: ${out_dir}"
    return 0
  fi
  mkdir -p "${out_dir}/masks"
  ensure_support_mask "${out_dir}"
  ensure_surface_reference "${out_dir}"

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${IMAGE}"
    --source-prompt "${SOURCE_PROMPT}"
    --prompt "${TARGET_PROMPT}"
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --mask-output-dir "${out_dir}/masks"
    --seed "${seed}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --n-max "${N_MAX}"
    --src-guidance-scale "${SRC_GUIDANCE_SCALE}"
    --base-guidance-scale "${BASE_GUIDANCE_SCALE}"
    --tar-guidance-scale "${TAR_GUIDANCE_SCALE}"
    --edit-hedit-guidance-scale "${EDIT_HEDIT_GUIDANCE_SCALE}"
    --edit-guidance-scale "${EDIT_GUIDANCE_SCALE}"
    --edit-region-guidance-scale "${EDIT_REGION_GUIDANCE_SCALE}"
    --edit-target-guidance-scale "${EDIT_TARGET_GUIDANCE_SCALE}"
    --edit-source-guidance-scale "${EDIT_SOURCE_GUIDANCE_SCALE}"
    --edit-clip-guidance-scale "${EDIT_CLIP_GUIDANCE_SCALE}"
    --edit-text-guidance-scale "${EDIT_TEXT_GUIDANCE_SCALE}"
    --edit-dds-guidance-scale "${EDIT_DDS_GUIDANCE_SCALE}"
    --edit-app-guidance-scale "${EDIT_APP_GUIDANCE_SCALE}"
    --edit-color-guidance-scale "${EDIT_COLOR_GUIDANCE_SCALE}"
    --edit-ref-guidance-scale "${EDIT_REF_GUIDANCE_SCALE}"
    --rec-guidance-scale "${REC_GUIDANCE_SCALE}"
    --struct-guidance-scale "${STRUCT_GUIDANCE_SCALE}"
    --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
    --trajectory-subject-preserve-scale 0
    --source-inject-q-scale "${SOURCE_INJECT_Q_SCALE}"
    --source-inject-k-scale "${SOURCE_INJECT_K_SCALE}"
    --source-inject-v-scale "${SOURCE_INJECT_V_SCALE}"
    --source-inject-layer-from "${SOURCE_INJECT_LAYER_FROM}"
    --source-inject-layer-to "${SOURCE_INJECT_LAYER_TO}"
    --source-inject-steps "${SOURCE_INJECT_STEPS}"
    --source-inject-mask-mode "${SOURCE_INJECT_MASK_MODE}"
    --object-mask-provider "${OBJECT_MASK_PROVIDER}"
    --mask-layering-mode "${MASK_LAYERING_MODE}"
    --attention-mask-target-words "${ATTENTION_TARGET_WORDS}"
    --rec-stop-timestep 0.08
    --beta-max 1.0
    --velocity-conversion-mode linear_path
    --linear-path-t-min 0.05
    --photo-prompt-mode "${PHOTO_PROMPT_MODE}"
    --log-every 7
  )
  if [[ -n "${SUPPORT_MASK}" ]]; then
    cmd+=(--support-mask "${SUPPORT_MASK}")
  fi
  if [[ -n "${SOURCE_INJECT_MASK_BOX}" ]]; then
    cmd+=(--source-inject-mask-box "${SOURCE_INJECT_MASK_BOX}")
  fi
  if [[ -n "${EDIT_REF_IMAGE}" ]]; then
    cmd+=(
      --edit-ref-image "${EDIT_REF_IMAGE}"
      --edit-ref-mask "${EDIT_REF_MASK}"
      --edit-ref-structure-image "${EDIT_REF_STRUCTURE_IMAGE}"
      --edit-ref-chroma-mode "${EDIT_REF_CHROMA_MODE}"
      --edit-ref-chroma-magnitude-scale "${EDIT_REF_CHROMA_MAGNITUDE_SCALE}"
      --edit-ref-luma-preserve-scale "${EDIT_REF_LUMA_PRESERVE_SCALE}"
      --edit-ref-gradient-preserve-scale "${EDIT_REF_GRADIENT_PRESERVE_SCALE}"
      --edit-ref-darkness-guard-scale "${EDIT_REF_DARKNESS_GUARD_SCALE}"
      --edit-ref-darkness-guard-margin "${EDIT_REF_DARKNESS_GUARD_MARGIN}"
      --edit-ref-smooth-kernel "${EDIT_REF_SMOOTH_KERNEL}"
      --edit-ref-lowfreq-suppress-kernel "${EDIT_REF_LOWFREQ_SUPPRESS_KERNEL}"
      --edit-ref-lowfreq-suppress-scale "${EDIT_REF_LOWFREQ_SUPPRESS_SCALE}"
      --edit-ref-schedule-start "${EDIT_REF_SCHEDULE_START}"
      --edit-ref-schedule-stop "${EDIT_REF_SCHEDULE_STOP}"
      --edit-ref-schedule-power "${EDIT_REF_SCHEDULE_POWER}"
      --edit-ref-max-struct-rms-ratio "${EDIT_REF_MAX_STRUCT_RMS_RATIO}"
      --edit-ref-project-struct-conflict "${EDIT_REF_PROJECT_STRUCT_CONFLICT}"
    )
  fi

  {
    printf 'TASK=%q METHOD=%q SEED=%q DEVICE=%q ' "${task_id}" "${method_id}" "${seed}" "${DEVICE}"
    if [[ -f "${out_dir}/mask_command.txt" ]]; then
      printf 'MASK_COMMAND_FILE=%q ' "${out_dir}/mask_command.txt"
    fi
    if [[ -f "${out_dir}/surface_ref_command.txt" ]]; then
      printf 'SURFACE_REF_COMMAND_FILE=%q ' "${out_dir}/surface_ref_command.txt"
    fi
    printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${cmd[@]}"
    printf '\n'
  } > "${out_dir}/command.txt"

  echo "[main-matrix] task=${task_id}/${TASK_NAME} method=${method_id}/${METHOD_NAME} seed=${seed}"
  echo "[main-matrix] out=${out_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/command.txt"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
}

for task_id in ${TASKS}; do
  for method_id in ${METHODS}; do
    for seed in ${SEEDS}; do
      run_one "${task_id}" "${method_id}" "${seed}"
    done
  done
done
