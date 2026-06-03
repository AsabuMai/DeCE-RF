#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${ROOT:-${DEFAULT_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/pretty_matrix}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ -z "${HF_HOME:-}" && -d "${ROOT}/../.cache/huggingface" ]]; then
  export HF_HOME="${ROOT}/../.cache/huggingface"
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
if [[ "${HF_HUB_ENABLE_HF_TRANSFER}" == "1" ]] && ! "${PYTHON}" -c "import hf_transfer" >/dev/null 2>&1; then
  echo "[pretty-matrix] disabling HF_HUB_ENABLE_HF_TRANSFER because hf_transfer is not installed"
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
DEVICE="${DEVICE:-0}"

TASK="${TASK:-P1}"
METHOD="${METHOD:-full}"
SEED="${SEED:-10}"
TASKS="${TASKS:-${TASK}}"
METHODS="${METHODS:-${METHOD}}"
SEEDS="${SEEDS:-${SEED}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SUPPORT_DEBUG_ONLY="${SUPPORT_DEBUG_ONLY:-0}"

GROUNDING_MODEL="${GROUNDING_MODEL:-IDEA-Research/grounding-dino-base}"
SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}"
ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD:-0}"
REGENERATE_MASKS="${REGENERATE_MASKS:-0}"
SUPPORT_V3_TEMPORAL_AGGREGATION="${SUPPORT_V3_TEMPORAL_AGGREGATION:-mean}"
SUPPORT_TOP_PERCENTILE_DEFAULT="${SUPPORT_TOP_PERCENTILE:-95}"
SUPPORT_MIN_AREA_RATIO_DEFAULT="${SUPPORT_MIN_AREA_RATIO:-0.02}"
SUPPORT_MAX_AREA_RATIO_DEFAULT="${SUPPORT_MAX_AREA_RATIO:-0.10}"
SUPPORT_KEEP_COMPONENTS_DEFAULT="${SUPPORT_KEEP_COMPONENTS:-1}"
SUPPORT_DILATE_RADIUS_DEFAULT="${SUPPORT_DILATE_RADIUS:-3}"
SUPPORT_BLUR_KERNEL_DEFAULT="${SUPPORT_BLUR_KERNEL:-3}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
N_MAX="${N_MAX:-24}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
LOW_VRAM="${LOW_VRAM:-0}"
PHOTO_PROMPT_MODE="${PHOTO_PROMPT_MODE:-both}"


source "${SCRIPT_DIR}/core6_tasks.sh"
source "${SCRIPT_DIR}/core6_methods.sh"
ensure_semantic_mask() {
  local out_dir="$1"
  SUPPORT_MASK="${out_dir}/masks/semantic_support.png"
  local support_meta="${out_dir}/masks/semantic_support.json"
  local anchor_mask="${out_dir}/masks/semantic_anchor.png"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${SUPPORT_MASK}" ]]; then
    return 0
  fi
  if [[ -n "${SEMANTIC_MASK_CACHE_METHOD:-}" && "${REGENERATE_MASKS}" != "1" ]]; then
    local explicit_cache_dir="${ROOT}/outputs/pretty_matrix/${TASK_NAME}/${SEMANTIC_MASK_CACHE_METHOD}/seed_${CURRENT_SEED}/masks"
    if [[ -s "${explicit_cache_dir}/semantic_support.png" ]]; then
      echo "[pretty-matrix] reusing semantic mask: ${explicit_cache_dir}/semantic_support.png -> ${SUPPORT_MASK}"
      cp "${explicit_cache_dir}/semantic_support.png" "${SUPPORT_MASK}"
      if [[ -s "${explicit_cache_dir}/semantic_support.json" ]]; then
        cp "${explicit_cache_dir}/semantic_support.json" "${support_meta}"
      fi
      if [[ -s "${explicit_cache_dir}/semantic_anchor.png" ]]; then
        cp "${explicit_cache_dir}/semantic_anchor.png" "${anchor_mask}"
      fi
      return 0
    fi
  fi
  if [[ "${REUSE_SEMANTIC_MASKS:-1}" == "1" && "${REGENERATE_MASKS}" != "1" ]]; then
    local cache_dir
    cache_dir="$(find "${ROOT}/outputs/pretty_matrix/${TASK_NAME}" -path "*/seed_${CURRENT_SEED}/masks/semantic_support.png" -type f 2>/dev/null \
      | grep "/${CANONICAL_METHOD_NAME}" \
      | grep -v "^${SUPPORT_MASK}$" \
      | head -1 \
      | xargs -r dirname || true)"
    if [[ -n "${cache_dir}" && -s "${cache_dir}/semantic_support.png" ]]; then
      echo "[pretty-matrix] reusing semantic mask: ${cache_dir}/semantic_support.png -> ${SUPPORT_MASK}"
      cp "${cache_dir}/semantic_support.png" "${SUPPORT_MASK}"
      if [[ -s "${cache_dir}/semantic_support.json" ]]; then
        cp "${cache_dir}/semantic_support.json" "${support_meta}"
      fi
      if [[ -s "${cache_dir}/semantic_anchor.png" ]]; then
        cp "${cache_dir}/semantic_anchor.png" "${anchor_mask}"
      fi
      return 0
    fi
  fi
  local cmd=(
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
    --dilate "${SEMANTIC_DILATE}"
    --blur "${SEMANTIC_BLUR}"
  )
  if [[ -n "${SEMANTIC_PHRASE}" ]]; then
    cmd+=(--phrase "${SEMANTIC_PHRASE}")
  fi
  if [[ "${ALLOW_MASK_DOWNLOAD}" == "1" ]]; then
    cmd+=(--allow-download)
  fi
  printf '%q ' "${cmd[@]}" > "${out_dir}/mask_command.txt"
  printf '\n' >> "${out_dir}/mask_command.txt"
  echo "[pretty-matrix] generating semantic mask: ${SUPPORT_MASK}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/mask_command.txt"
  else
    "${cmd[@]}"
  fi
}

ensure_decal_reference() {
  local out_dir="$1"
  DECAL_REF_IMAGE="${out_dir}/masks/decal_reference.png"
  DECAL_MASK="${out_dir}/masks/decal_mask.png"
  local decal_overlay="${out_dir}/masks/decal_mask_overlay.png"
  local decal_meta="${out_dir}/masks/decal_reference.json"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${DECAL_REF_IMAGE}" && -s "${DECAL_MASK}" ]]; then
    return 0
  fi
  local cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_decal_reference.py"
    --image "${IMAGE}"
    --output "${DECAL_REF_IMAGE}"
    --mask-output "${DECAL_MASK}"
    --overlay-output "${decal_overlay}"
    --metadata-output "${decal_meta}"
    --shape "${DECAL_SHAPE}"
    --color "${DECAL_COLOR}"
    --box "${DECAL_BOX}"
    --slant-x "${DECAL_SLANT_X}"
    --perspective-y "${DECAL_PERSPECTIVE_Y}"
    --edge-feather-radius "${DECAL_EDGE_FEATHER_RADIUS}"
    --top-feather-frac "${DECAL_TOP_FEATHER_FRAC}"
    --top-feather-min-alpha "${DECAL_TOP_FEATHER_MIN_ALPHA}"
    --opacity "${DECAL_OPACITY:-0.72}"
  )
  printf '%q ' "${cmd[@]}" > "${out_dir}/decal_command.txt"
  printf '\n' >> "${out_dir}/decal_command.txt"
  echo "[pretty-matrix] generating decal reference: ${DECAL_REF_IMAGE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/decal_command.txt"
  else
    "${cmd[@]}"
  fi
}


ensure_refined_surface_mask() {
  local out_dir="$1"
  local source_mask="${SUPPORT_MASK}"
  local refined_mask="${out_dir}/masks/surface_refined_mask.png"
  local refined_meta="${out_dir}/masks/surface_refined_mask.json"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${refined_mask}" ]]; then
    SUPPORT_MASK="${refined_mask}"
    return 0
  fi
  local cmd=(
    "${PYTHON}" "${ROOT}/scripts/refine_surface_mask.py"
    --mask "${source_mask}"
    --output "${refined_mask}"
    --metadata-output "${refined_meta}"
    --threshold "0.50"
    --erode-kernel "3"
    --erode-iterations "2"
    --blur-kernel "3"
  )
  printf '%q ' "${cmd[@]}" > "${out_dir}/refine_mask_command.txt"
  printf '\n' >> "${out_dir}/refine_mask_command.txt"
  echo "[pretty-matrix] refining surface mask: ${refined_mask}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/refine_mask_command.txt"
  else
    "${cmd[@]}"
  fi
  SUPPORT_MASK="${refined_mask}"
}


ensure_recolor_reference() {
  local out_dir="$1"
  if [[ -n "${RECOLOR_BOX}" ]]; then
    SUPPORT_MASK="${out_dir}/masks/recolor_color_mask.png"
    local support_overlay="${out_dir}/masks/recolor_color_mask_overlay.png"
    local support_meta="${out_dir}/masks/recolor_color_mask.json"
    local mask_cmd=(
      "${PYTHON}" "${ROOT}/scripts/make_source_color_mask.py"
      --image "${IMAGE}"
      --output "${SUPPORT_MASK}"
      --overlay-output "${support_overlay}"
      --metadata-output "${support_meta}"
      --source-color "${RECOLOR_SOURCE_COLOR}"
      --box "${RECOLOR_BOX}"
      --mask-threshold "0.20"
      --keep-components "2"
      --min-area "80"
      --fill-holes
      --open-kernel "1"
      --close-kernel "5"
    )
    printf '%q ' "${mask_cmd[@]}" > "${out_dir}/mask_command.txt"
    printf '\n' >> "${out_dir}/mask_command.txt"
    echo "[pretty-matrix] generating recolor color mask: ${SUPPORT_MASK}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      cat "${out_dir}/mask_command.txt"
    else
      "${mask_cmd[@]}"
    fi
  fi
  RECOLOR_REF_IMAGE="${out_dir}/masks/surface_recolor_reference.png"
  local recolor_overlay="${out_dir}/masks/surface_recolor_overlay.png"
  local recolor_meta="${out_dir}/masks/surface_recolor_reference.json"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${RECOLOR_REF_IMAGE}" ]]; then
    return 0
  fi
  local cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_surface_recolor_reference.py"
    --image "${IMAGE}"
    --surface-mask "${SUPPORT_MASK}"
    --output "${RECOLOR_REF_IMAGE}"
    --target-color "${RECOLOR_TARGET_COLOR}"
    --luma-image "${IMAGE}"
    --mode "yuv-chroma"
    --blend "0.78"
    --mask-blur "5"
    --surface-name "${RECOLOR_SURFACE_NAME}"
    --overlay-output "${recolor_overlay}"
    --metadata-output "${recolor_meta}"
  )
  printf '%q ' "${cmd[@]}" > "${out_dir}/recolor_command.txt"
  printf '\n' >> "${out_dir}/recolor_command.txt"
  echo "[pretty-matrix] generating recolor reference: ${RECOLOR_REF_IMAGE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/recolor_command.txt"
  else
    "${cmd[@]}"
  fi
}

apply_recolor_edit_config() {
  local out_dir="$1"
  if [[ -z "${RECOLOR_BOX}" ]]; then
    if [[ -z "${SUPPORT_MASK}" ]]; then
      OBJECT_MASK_PROVIDER="semantic_velocity"
      ensure_semantic_mask "${out_dir}"
    fi
    if [[ "${SUPPORT_MASK}" == */semantic_support.png ]]; then
      ensure_refined_surface_mask "${out_dir}"
    fi
  else
    OBJECT_MASK_PROVIDER="${OBJECT_MASK_PROVIDER:-attention_velocity}"
  fi
  ensure_recolor_reference "${out_dir}"
  EDIT_HEDIT_GUIDANCE_SCALE="${RECOLOR_HEDIT_GUIDANCE_SCALE:-0.18}"
  EDIT_TEXT_GUIDANCE_SCALE="${RECOLOR_TEXT_GUIDANCE_SCALE:-0.02}"
  REC_GUIDANCE_SCALE="${RECOLOR_REC_GUIDANCE_SCALE:-0.58}"
  TRAJECTORY_PRESERVE_SCALE="${RECOLOR_TRAJECTORY_PRESERVE_SCALE:-0.48}"
  EDIT_REF_GUIDANCE_SCALE="${RECOLOR_REF_GUIDANCE_SCALE:-0.46}"
  EDIT_COLOR_GUIDANCE_SCALE="${RECOLOR_COLOR_GUIDANCE_SCALE:-0.06}"
  EDIT_COLOR_ARGS=(
    --edit-color-guidance-scale "${EDIT_COLOR_GUIDANCE_SCALE}"
    --edit-color-target "${RECOLOR_TARGET_COLOR}"
    --edit-color-mask-image "${SUPPORT_MASK}"
    --edit-color-mask-threshold 0.20
    --edit-color-target-chroma-scale 0.82
    --edit-color-luma-preserve-scale 0.72
    --edit-color-luma-gradient-preserve-scale 0.36
  )
  if [[ -n "${RECOLOR_SOURCE_COLOR}" ]]; then
    EDIT_COLOR_ARGS+=(--edit-color-source "${RECOLOR_SOURCE_COLOR}")
  fi
  FINAL_MASK_ARGS=(--final-edit-mask "${SUPPORT_MASK}" --final-edit-mask-mode replace)
  REF_ARGS=(
    --edit-ref-image "${RECOLOR_REF_IMAGE}"
    --edit-ref-mask "${SUPPORT_MASK}"
    --edit-ref-structure-image "${IMAGE}"
    --edit-ref-chroma-mode yuv
    --edit-ref-luma-preserve-scale 0.82
    --edit-ref-gradient-preserve-scale 0.42
    --edit-ref-smooth-kernel 1
  )
}

run_one() {
  local task_id="$1"
  local method_id="$2"
  local seed="$3"
  task_config "${task_id}"
  SUPPORT_V3_CANDIDATE="${SUPPORT_V3_CANDIDATE_OVERRIDE:-${SUPPORT_V3_CANDIDATE}}"
  method_config "${method_id}"
  local method_support_score="${SUPPORT_SCORE:-}"
  SOURCE_PROMPT="${SOURCE_PROMPT_OVERRIDE:-${SOURCE_PROMPT}}"
  TARGET_PROMPT="${TARGET_PROMPT_OVERRIDE:-${TARGET_PROMPT}}"
  DECAL_BOX="${DECAL_BOX_OVERRIDE:-${DECAL_BOX}}"
  CANONICAL_METHOD_NAME="${METHOD_NAME}"
  CURRENT_SEED="${seed}"
  if [[ -n "${METHOD_NAME_SUFFIX:-}" ]]; then
    METHOD_NAME="${METHOD_NAME}${METHOD_NAME_SUFFIX}"
  fi

  local out_dir="${OUTPUT_ROOT}/${TASK_NAME}/${METHOD_NAME}/seed_${seed}"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${out_dir}/result.png" && -s "${out_dir}/stats.json" && -s "${out_dir}/metadata.json" && -s "${out_dir}/command.txt" ]]; then
    echo "[pretty-matrix] skip existing complete run: ${out_dir}"
    return 0
  fi
  mkdir -p "${out_dir}/masks"

  SUPPORT_MASK=""
  DECAL_REF_IMAGE=""
  DECAL_MASK=""
  RECOLOR_REF_IMAGE=""
  AUTO_STRUCTURE_FLAGS=()
  FINAL_MASK_ARGS=()
  REF_ARGS=()
  EDIT_COLOR_ARGS=()

  if [[ "${METHOD_ROUTE}" == "full" && "${GENERIC_SUPPORT}" == "1" ]]; then
    OBJECT_MASK_PROVIDER="generic_support"
    ATTENTION_TARGET_WORDS="${CHANGED_TARGET_WORDS:-${ATTENTION_TARGET_WORDS}}"
    if [[ "${GENERIC_SUPPORT_V2}" == "1" ]]; then
      SUPPORT_SCORE="${SUPPORT_V2_CANDIDATE:-${SUPPORT_SCORE}}"
      ATTENTION_TARGET_WORDS="${SUPPORT_NEW_TOKENS:-${ATTENTION_TARGET_WORDS}}"
    fi
    if [[ "${GENERIC_SUPPORT_V3}" == "1" ]]; then
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${method_support_score:-${SUPPORT_V3_CANDIDATE:-operation_default}}"
      ATTENTION_TARGET_WORDS="${SUPPORT_NEW_TOKENS:-${ATTENTION_TARGET_WORDS}}"
      case "${SUPPORT_EDIT_OPERATION}:${SUPPORT_V3_RELATION}" in
        add_object:above_host)
          SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ABOVE_MIN_AREA_RATIO:-0.008}"
          SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ABOVE_MAX_AREA_RATIO:-0.035}"
          SUPPORT_DILATE_RADIUS="${SUPPORT_V3_ABOVE_DILATE_RADIUS:-2}"
          SUPPORT_BLUR_KERNEL="${SUPPORT_V3_ABOVE_BLUR_KERNEL:-3}"
          ;;
        add_decal:*)
          if [[ "${SUPPORT_PRESET}" == "surface_strip" ]]; then
            if [[ "${SUPPORT_SCORE}" == "operation_default" || "${SUPPORT_SCORE}" == "auto" || -z "${SUPPORT_SCORE}" ]]; then
              SUPPORT_SCORE="${SUPPORT_V3_SURFACE_STRIP_CANDIDATE:-host_spawn_center_x_response}"
            fi
            SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_SURFACE_STRIP_MIN_AREA_RATIO:-0.025}"
            SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_SURFACE_STRIP_MAX_AREA_RATIO:-0.070}"
            SUPPORT_DILATE_RADIUS="${SUPPORT_V3_SURFACE_STRIP_DILATE_RADIUS:-2}"
            SUPPORT_BLUR_KERNEL="${SUPPORT_V3_SURFACE_STRIP_BLUR_KERNEL:-3}"
            REC_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_STRIP_REC_GUIDANCE_SCALE:-0.42}"
            TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_SURFACE_STRIP_TRAJECTORY_PRESERVE_SCALE:-0.24}"
            ADAPTIVE_PRESERVE_DRIFT_BUDGET="${SUPPORT_V3_SURFACE_STRIP_PRESERVE_DRIFT_BUDGET:-0.14}"
            ADAPTIVE_PRESERVE_GAIN="${SUPPORT_V3_SURFACE_STRIP_PRESERVE_GAIN:-4.0}"
            EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_STRIP_TEXT_GUIDANCE_SCALE:-0.12}"
            EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_STRIP_HEDIT_GUIDANCE_SCALE:-0.68}"
          else
            SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_DECAL_MIN_AREA_RATIO:-0.005}"
          fi
          if [[ "${SUPPORT_PRESET}" == "clothing_decal" ]]; then
            SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_CLOTHING_DECAL_MAX_AREA_RATIO:-${SUPPORT_V3_DECAL_MAX_AREA_RATIO:-0.060}}"
          elif [[ "${SUPPORT_PRESET}" != "surface_strip" ]]; then
            SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_DECAL_MAX_AREA_RATIO:-0.035}"
          fi
          if [[ "${SUPPORT_PRESET}" != "surface_strip" ]]; then
            SUPPORT_DILATE_RADIUS="${SUPPORT_V3_DECAL_DILATE_RADIUS:-1}"
            SUPPORT_BLUR_KERNEL="${SUPPORT_V3_DECAL_BLUR_KERNEL:-1}"
            REC_GUIDANCE_SCALE="${SUPPORT_V3_DECAL_REC_GUIDANCE_SCALE:-0.45}"
            TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_DECAL_TRAJECTORY_PRESERVE_SCALE:-0.30}"
            ADAPTIVE_PRESERVE_DRIFT_BUDGET="${SUPPORT_V3_DECAL_PRESERVE_DRIFT_BUDGET:-0.12}"
            ADAPTIVE_PRESERVE_GAIN="${SUPPORT_V3_DECAL_PRESERVE_GAIN:-4.5}"
          fi
          ;;
        remove_object:*)
          SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_REMOVAL_MIN_AREA_RATIO:-0.005}"
          SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_REMOVAL_MAX_AREA_RATIO:-0.060}"
          SUPPORT_DILATE_RADIUS="${SUPPORT_V3_REMOVAL_DILATE_RADIUS:-1}"
          SUPPORT_BLUR_KERNEL="${SUPPORT_V3_REMOVAL_BLUR_KERNEL:-1}"
          REMOVAL_CONTROLLER_MODE="${SUPPORT_V3_REMOVAL_CONTROLLER_MODE:-none}"
          REMOVAL_FILL_SCALE="${SUPPORT_V3_REMOVAL_FILL_SCALE:-0.70}"
          REMOVAL_SUPPRESSION_SCALE="${SUPPORT_V3_REMOVAL_SUPPRESSION_SCALE:-0.35}"
          REMOVAL_RING_REC_SCALE="${SUPPORT_V3_REMOVAL_RING_REC_SCALE:-0.40}"
          ;;
        recolor:*)
          SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_RECOLOR_MIN_AREA_RATIO:-0.035}"
          SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_RECOLOR_MAX_AREA_RATIO:-0.180}"
          SUPPORT_DILATE_RADIUS="${SUPPORT_V3_RECOLOR_DILATE_RADIUS:-1}"
          SUPPORT_BLUR_KERNEL="${SUPPORT_V3_RECOLOR_BLUR_KERNEL:-1}"
          REC_GUIDANCE_SCALE="${SUPPORT_V3_RECOLOR_REC_GUIDANCE_SCALE:-0.58}"
          TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_RECOLOR_TRAJECTORY_PRESERVE_SCALE:-0.48}"
          ADAPTIVE_PRESERVE_DRIFT_BUDGET="${SUPPORT_V3_RECOLOR_PRESERVE_DRIFT_BUDGET:-0.10}"
          ADAPTIVE_PRESERVE_GAIN="${SUPPORT_V3_RECOLOR_PRESERVE_GAIN:-3.0}"
          ;;
      esac
      local v3_relation="${SUPPORT_V3_RELATION:-auto}"
      # The semantic mask is used as grounding evidence for v3. Relation
      # proposal itself is built inside operation_support_v3.
      SUPPORT_RELATION="inside"
      ensure_semantic_mask "${out_dir}"
      SUPPORT_RELATION="${v3_relation}"
      if [[ "${SUPPORT_PRESET}" == "surface_strip" ]]; then
        ensure_decal_reference "${out_dir}"
        EDIT_REF_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_STRIP_REF_GUIDANCE_SCALE:-0.30}"
        FINAL_MASK_ARGS=(--final-edit-mask "${DECAL_MASK}" --final-edit-mask-mode replace)
        REF_ARGS=(
          --edit-ref-image "${DECAL_REF_IMAGE}"
          --edit-ref-mask "${DECAL_MASK}"
          --edit-ref-structure-image "${IMAGE}"
          --edit-ref-chroma-mode yuv
          --edit-ref-luma-preserve-scale "${SUPPORT_V3_SURFACE_STRIP_REF_LUMA_PRESERVE:-0.32}"
          --edit-ref-gradient-preserve-scale "${SUPPORT_V3_SURFACE_STRIP_REF_GRADIENT_PRESERVE:-0.08}"
          --edit-ref-smooth-kernel 1
        )
      fi
    fi
  elif [[ "${METHOD_ROUTE}" == "full" && "${MANUAL_SUPPORT}" == "1" ]]; then
    OBJECT_MASK_PROVIDER="semantic"
    ensure_semantic_mask "${out_dir}"
  elif [[ "${METHOD_ROUTE}" == "full" ]]; then
    case "${TASK_KIND}" in
      accessory_semantic)
        OBJECT_MASK_PROVIDER="semantic_velocity"
        ensure_semantic_mask "${out_dir}"
        ;;
      decal)
        ensure_decal_reference "${out_dir}"
        EDIT_REF_GUIDANCE_SCALE="0.32"
        FINAL_MASK_ARGS=(--final-edit-mask "${DECAL_MASK}" --final-edit-mask-mode replace)
        REF_ARGS=(
          --edit-ref-image "${DECAL_REF_IMAGE}"
          --edit-ref-mask "${DECAL_MASK}"
          --edit-ref-structure-image "${IMAGE}"
          --edit-ref-chroma-mode yuv
          --edit-ref-luma-preserve-scale 0.20
          --edit-ref-gradient-preserve-scale 0.05
          --edit-ref-smooth-kernel 1
        )
        ;;
      recolor_semantic)
        if [[ -n "${RECOLOR_BOX}" ]]; then
          OBJECT_MASK_PROVIDER="attention_velocity"
        else
          OBJECT_MASK_PROVIDER="semantic_velocity"
        fi
        ;;
      remove_semantic)
        OBJECT_MASK_PROVIDER="semantic_velocity"
        ensure_semantic_mask "${out_dir}"
        EDIT_HEDIT_GUIDANCE_SCALE="0.78"
        EDIT_TEXT_GUIDANCE_SCALE="0.10"
        REC_GUIDANCE_SCALE="0.20"
        TRAJECTORY_PRESERVE_SCALE="0.16"
        FINAL_MASK_ARGS=(--final-edit-mask "${SUPPORT_MASK}" --final-edit-mask-mode replace)
        ;;
    esac
  fi

  if [[ "${METHOD_ROUTE}" == "full" && "${TASK_KIND}" == "recolor_semantic" ]]; then
    apply_recolor_edit_config "${out_dir}"
  fi

  case "${METHOD_ABLATION}" in
    none)
      ;;
    no_ref)
      EDIT_REF_GUIDANCE_SCALE="0.0"
      REF_ARGS=()
      ;;
    no_rec)
      REC_GUIDANCE_SCALE="0.0"
      ;;
    no_traj)
      TRAJECTORY_PRESERVE_SCALE="0.0"
      ;;
    *)
      echo "Unknown METHOD_ABLATION '${METHOD_ABLATION}'." >&2
      exit 2
      ;;
  esac

  if [[ "${EDIT_STRENGTH_MULTIPLIER:-1.0}" != "1.0" ]]; then
    EDIT_HEDIT_GUIDANCE_SCALE="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]) * float(sys.argv[2]):.8g}")' "${EDIT_HEDIT_GUIDANCE_SCALE}" "${EDIT_STRENGTH_MULTIPLIER}")"
    EDIT_GUIDANCE_SCALE="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]) * float(sys.argv[2]):.8g}")' "${EDIT_GUIDANCE_SCALE}" "${EDIT_STRENGTH_MULTIPLIER}")"
    EDIT_REGION_GUIDANCE_SCALE="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]) * float(sys.argv[2]):.8g}")' "${EDIT_REGION_GUIDANCE_SCALE}" "${EDIT_STRENGTH_MULTIPLIER}")"
    EDIT_TARGET_GUIDANCE_SCALE="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]) * float(sys.argv[2]):.8g}")' "${EDIT_TARGET_GUIDANCE_SCALE}" "${EDIT_STRENGTH_MULTIPLIER}")"
    EDIT_SOURCE_GUIDANCE_SCALE="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]) * float(sys.argv[2]):.8g}")' "${EDIT_SOURCE_GUIDANCE_SCALE}" "${EDIT_STRENGTH_MULTIPLIER}")"
    EDIT_TEXT_GUIDANCE_SCALE="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]) * float(sys.argv[2]):.8g}")' "${EDIT_TEXT_GUIDANCE_SCALE}" "${EDIT_STRENGTH_MULTIPLIER}")"
  fi

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${IMAGE}"
    --source-prompt "${SOURCE_PROMPT}"
    --prompt "${TARGET_PROMPT}"
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --mask-output-dir "${out_dir}/masks"
    --max-image-size "${MAX_IMAGE_SIZE}"
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
    --edit-text-guidance-scale "${EDIT_TEXT_GUIDANCE_SCALE}"
    --edit-text-source-scale "${EDIT_TEXT_SOURCE_SCALE}"
    --edit-text-core-weight "${EDIT_TEXT_CORE_WEIGHT}"
    --edit-text-subject-weight "${EDIT_TEXT_SUBJECT_WEIGHT}"
    --edit-ref-guidance-scale "${EDIT_REF_GUIDANCE_SCALE}"
    --rec-guidance-scale "${REC_GUIDANCE_SCALE}"
    --struct-guidance-scale "${STRUCT_GUIDANCE_SCALE}"
    --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
    --edit-initial-noise-scale "${EDIT_INITIAL_NOISE_SCALE:-0.0}"
    --edit-initial-noise-region "${EDIT_INITIAL_NOISE_REGION:-core}"
    --region-target-transport-scale "${REGION_TARGET_TRANSPORT_SCALE}"
    --region-target-outside-lock-scale "${REGION_TARGET_OUTSIDE_LOCK_SCALE}"
    --object-mask-provider "${OBJECT_MASK_PROVIDER}"
    --mask-layering-mode "${MASK_LAYERING_MODE}"
    --attention-mask-target-words "${ATTENTION_TARGET_WORDS}"
    --support-score "${SUPPORT_SCORE}"
    --support-top-percentile "${SUPPORT_TOP_PERCENTILE}"
    --support-min-area-ratio "${SUPPORT_MIN_AREA_RATIO}"
    --support-max-area-ratio "${SUPPORT_MAX_AREA_RATIO}"
    --support-keep-components "${SUPPORT_KEEP_COMPONENTS}"
    --support-dilate-radius "${SUPPORT_DILATE_RADIUS}"
    --support-blur-kernel "${SUPPORT_BLUR_KERNEL}"
    --attention-mask-max-area-ratio 0.22
    --attention-mask-fallback-threshold 0.74
    --rec-stop-timestep 0.08
    --beta-max 1.0
    --velocity-conversion-mode linear_path
    --linear-path-t-min 0.05
    --photo-prompt-mode "${PHOTO_PROMPT_MODE}"
    --log-every 7
    --edit-mask-dilate-kernel "${EDIT_MASK_DILATE_KERNEL:-0}"
    --edit-mask-erode-kernel "${EDIT_MASK_ERODE_KERNEL:-0}"
    --edit-mask-smooth-kernel "${EDIT_MASK_SMOOTH_KERNEL:-0}"
    --edit-mask-hole-fraction "${EDIT_MASK_HOLE_FRACTION:-0.0}"
    --edit-mask-boundary-noise-scale "${EDIT_MASK_BOUNDARY_NOISE_SCALE:-0.0}"
    --edit-mask-shift-y "${EDIT_MASK_SHIFT_Y:-0.0}"
    --edit-mask-shift-x "${EDIT_MASK_SHIFT_X:-0.0}"
  )
  if [[ "${LOW_VRAM}" == "1" ]]; then
    cmd+=(--low-vram)
  fi
  if [[ -n "${EDIT_TEXT_SOURCE_PROMPT}" ]]; then
    cmd+=(--edit-text-source-prompt "${EDIT_TEXT_SOURCE_PROMPT}")
  fi
  if [[ -n "${EDIT_TEXT_TARGET_PROMPT}" ]]; then
    cmd+=(--edit-text-target-prompt "${EDIT_TEXT_TARGET_PROMPT}")
  fi
  if [[ -n "${EDIT_LOCAL_TARGET_PROMPT}" ]]; then
    cmd+=(
      --edit-local-target-prompt "${EDIT_LOCAL_TARGET_PROMPT}"
      --edit-local-target-guidance-scale "${EDIT_LOCAL_TARGET_GUIDANCE_SCALE}"
    )
    if [[ -n "${EDIT_LOCAL_TARGET_CFG_SCALE}" ]]; then
      cmd+=(--edit-local-target-cfg-scale "${EDIT_LOCAL_TARGET_CFG_SCALE}")
    fi
  fi
  if [[ "${GENERIC_SUPPORT_V2}" == "1" || "${GENERIC_SUPPORT_V3}" == "1" ]]; then
    cmd+=(
      --support-candidate "${SUPPORT_SCORE}"
      --edit-operation "${SUPPORT_EDIT_OPERATION}"
    )
    if [[ "${GENERIC_SUPPORT_V3}" == "1" ]]; then
      cmd+=(
        --support-mode operation_v3
        --relation "${SUPPORT_V3_RELATION}"
        --grounding-method grounded_sam
        --support-temporal-aggregation "${SUPPORT_V3_TEMPORAL_AGGREGATION}"
        --save-support-debug
      )
      if [[ "${SUPPORT_DEBUG_ONLY}" == "1" ]]; then
        cmd+=(--support-debug-only)
      fi
      if [[ "${REMOVAL_CONTROLLER_MODE}" != "none" ]]; then
        cmd+=(
          --removal-controller-mode "${REMOVAL_CONTROLLER_MODE}"
          --removal-fill-scale "${REMOVAL_FILL_SCALE}"
          --removal-suppression-scale "${REMOVAL_SUPPRESSION_SCALE}"
          --removal-ring-rec-scale "${REMOVAL_RING_REC_SCALE}"
        )
      fi
    fi
    if [[ -n "${SUPPORT_NEW_TOKENS}" ]]; then
      cmd+=(--new-tokens "${SUPPORT_NEW_TOKENS}")
    fi
    if [[ -n "${SUPPORT_HOST_TOKENS}" ]]; then
      cmd+=(--host-tokens "${SUPPORT_HOST_TOKENS}")
    fi
    if [[ -n "${SUPPORT_REMOVED_TOKENS}" ]]; then
      cmd+=(--removed-tokens "${SUPPORT_REMOVED_TOKENS}")
    fi
  fi
  if [[ -n "${SUPPORT_MASK}" ]]; then
    cmd+=(--support-mask "${SUPPORT_MASK}")
  fi
  if [[ "${#AUTO_STRUCTURE_FLAGS[@]}" -gt 0 ]]; then
    cmd+=("${AUTO_STRUCTURE_FLAGS[@]}")
  fi
  if [[ "${#FINAL_MASK_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${FINAL_MASK_ARGS[@]}")
  fi
  if [[ "${#EDIT_COLOR_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EDIT_COLOR_ARGS[@]}")
  fi
  if [[ "${#REF_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${REF_ARGS[@]}")
  fi
  if [[ "${ADAPTIVE_CLEAN_CONTROL}" == "1" ]]; then
    cmd+=(
      --adaptive-clean-control
      --adaptive-edit-target-progress "${ADAPTIVE_EDIT_TARGET_PROGRESS}"
      --adaptive-edit-target-rms "${ADAPTIVE_EDIT_TARGET_RMS}"
      --adaptive-rmsgap-mode "${ADAPTIVE_RMSGAP_MODE}"
      --adaptive-rmsgap-dead-zone "${ADAPTIVE_RMSGAP_DEAD_ZONE}"
      --adaptive-rmsgap-preserve-gate-budget "${ADAPTIVE_RMSGAP_PRESERVE_GATE_BUDGET}"
      --adaptive-hybrid-progress-target "${ADAPTIVE_HYBRID_PROGRESS_TARGET}"
      --adaptive-hybrid-progress-gain "${ADAPTIVE_HYBRID_PROGRESS_GAIN}"
      --adaptive-hybrid-progress-ema-decay "${ADAPTIVE_HYBRID_PROGRESS_EMA_DECAY}"
      --adaptive-hybrid-preserve-gate-budget "${ADAPTIVE_HYBRID_PRESERVE_GATE_BUDGET}"
      --adaptive-preserve-drift-budget "${ADAPTIVE_PRESERVE_DRIFT_BUDGET}"
      --adaptive-edit-gain "${ADAPTIVE_EDIT_GAIN}"
      --adaptive-preserve-gain "${ADAPTIVE_PRESERVE_GAIN}"
      --adaptive-edit-weight-min "${ADAPTIVE_EDIT_WEIGHT_MIN}"
      --adaptive-edit-weight-max "${ADAPTIVE_EDIT_WEIGHT_MAX}"
      --adaptive-preserve-weight-min "${ADAPTIVE_PRESERVE_WEIGHT_MIN}"
      --adaptive-preserve-weight-max "${ADAPTIVE_PRESERVE_WEIGHT_MAX}"
      --adaptive-projection-scale "${ADAPTIVE_PROJECTION_SCALE}"
      --adaptive-preserve-clean-correction-scale "${ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE}"
    )
  fi

  {
    printf 'TASK=%q METHOD=%q SEED=%q DEVICE=%q ' "${task_id}" "${method_id}" "${seed}" "${DEVICE}"
    printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${cmd[@]}"
    printf '\n'
  } > "${out_dir}/command.txt"

  echo "[pretty-matrix] task=${TASK_NAME} method=${METHOD_NAME} seed=${seed}"
  echo "[pretty-matrix] out=${out_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/command.txt"
  else
    CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
  fi
}

for task_id in ${TASKS}; do
  for method_id in ${METHODS}; do
    for seed in ${SEEDS}; do
      run_one "${task_id}" "${method_id}" "${seed}"
    done
  done
done
