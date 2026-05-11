#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"

TASK="${TASK:-P1}"
METHOD="${METHOD:-full}"
SEED="${SEED:-10}"
TASKS="${TASKS:-${TASK}}"
METHODS="${METHODS:-${METHOD}}"
SEEDS="${SEEDS:-${SEED}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

GROUNDING_MODEL="${GROUNDING_MODEL:-IDEA-Research/grounding-dino-base}"
SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}"
ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD:-0}"
REGENERATE_MASKS="${REGENERATE_MASKS:-0}"
SUPPORT_V3_TEMPORAL_AGGREGATION="${SUPPORT_V3_TEMPORAL_AGGREGATION:-mean}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
N_MAX="${N_MAX:-24}"
PHOTO_PROMPT_MODE="${PHOTO_PROMPT_MODE:-both}"

task_config() {
  local task_id="$1"
  TASK_NAME=""
  TASK_KIND=""
  IMAGE=""
  SOURCE_PROMPT=""
  TARGET_PROMPT=""
  ATTENTION_TARGET_WORDS=""
  CHANGED_TARGET_WORDS=""
  SUPPORT_EDIT_OPERATION="auto"
  SUPPORT_NEW_TOKENS=""
  SUPPORT_HOST_TOKENS=""
  SUPPORT_REMOVED_TOKENS=""
  SUPPORT_V2_CANDIDATE=""
  SUPPORT_V3_CANDIDATE="operation_default"
  SUPPORT_V3_RELATION="auto"
  SEMANTIC_PHRASE=""
  SEMANTIC_DILATE="0"
  SEMANTIC_BLUR="0"
  SUPPORT_RELATION="auto"
  DECAL_SHAPE=""
  DECAL_COLOR=""
  DECAL_BOX=""
  RECOLOR_TARGET_COLOR=""
  RECOLOR_SURFACE_NAME=""
  RECOLOR_BOX=""
  RECOLOR_SOURCE_COLOR=""
  REPLACEMENT_COLOR="blue"
  REPLACEMENT_SHAPE="semantic"
  REPLACEMENT_OPACITY="0.98"
  REPLACEMENT_SCALE="0.78"
  REPLACEMENT_BLUR="0.7"
  REPLACEMENT_BACKGROUND_MODE="patch"
  REPLACEMENT_OUTLINE="0"
  REPLACEMENT_HEDIT_GUIDANCE_SCALE="0.62"
  REPLACEMENT_TEXT_GUIDANCE_SCALE="0.06"
  REPLACEMENT_REF_GUIDANCE_SCALE="0.42"
  REPLACEMENT_REC_GUIDANCE_SCALE="0.20"
  REPLACEMENT_TRAJECTORY_PRESERVE_SCALE="0.14"

  case "${task_id}" in
    P1|cat_crown)
      TASK_NAME="cat_crown"
      TASK_KIND="accessory_semantic"
      IMAGE="${ROOT}/data/paper_images/cat_sitting_in_grass.jpg"
      SOURCE_PROMPT="A photo of a cat sitting in grass."
      TARGET_PROMPT="A photo of the same cat sitting in the same grass, wearing a small golden crown on its head."
      ATTENTION_TARGET_WORDS="crown,head"
      CHANGED_TARGET_WORDS="crown"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="crown"
      SUPPORT_HOST_TOKENS="cat,head"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="above_host"
      SEMANTIC_PHRASE="cat"
      ;;
    P2|dog_sunglasses)
      TASK_NAME="dog_sunglasses"
      TASK_KIND="accessory_semantic_glasses"
      IMAGE="${ROOT}/data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg"
      SOURCE_PROMPT="A front-facing portrait of a dog in snow."
      TARGET_PROMPT="A front-facing portrait of the same dog wearing black sunglasses in snow."
      ATTENTION_TARGET_WORDS="sunglasses,eyes"
      CHANGED_TARGET_WORDS="sunglasses"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="sunglasses"
      SUPPORT_HOST_TOKENS="dog,eyes"
      SUPPORT_V2_CANDIDATE="attention_x_clean"
      SUPPORT_V3_RELATION="on_face"
      SEMANTIC_PHRASE="dog head"
      SUPPORT_RELATION="front_glasses_auto"
      ;;
    P3|mug_heart)
      TASK_NAME="mug_heart"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_white_mug_6312107.jpg"
      SOURCE_PROMPT="A minimalist photo of a plain white ceramic mug on a grey background."
      TARGET_PROMPT="A minimalist photo of the same white ceramic mug with a small red heart printed on the front, on the same grey background."
      ATTENTION_TARGET_WORDS="heart,mug,front"
      CHANGED_TARGET_WORDS="heart"
      SUPPORT_EDIT_OPERATION="add_decal"
      SUPPORT_NEW_TOKENS="heart"
      SUPPORT_HOST_TOKENS="mug"
      SUPPORT_V2_CANDIDATE="new_x_host_x_clean"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="mug"
      DECAL_SHAPE="heart"
      DECAL_COLOR="red"
      DECAL_BOX="0.375,0.405,0.515,0.585"
      ;;
    P4|red_chair_blue)
      TASK_NAME="red_chair_blue"
      TASK_KIND="recolor_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg"
      SOURCE_PROMPT="A photo of a red armless rounded upholstered chair in a stylish room."
      TARGET_PROMPT="A photo of the same armless rounded upholstered chair in the same stylish room, with only the fabric color changed to deep blue, no armrests added."
      ATTENTION_TARGET_WORDS="chair,blue"
      SEMANTIC_PHRASE="chair"
      SUPPORT_RELATION="inside"
      RECOLOR_TARGET_COLOR="blue"
      RECOLOR_SURFACE_NAME="chair"
      ;;
    P5|tshirt_star)
      TASK_NAME="tshirt_star"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_white_tshirt_mockup_12025472.jpg"
      SOURCE_PROMPT="A product photo of a plain white t-shirt on a light grey background."
      TARGET_PROMPT="A product photo of the same white t-shirt with a red star printed on the chest, on the same light grey background."
      ATTENTION_TARGET_WORDS="star,t-shirt,chest"
      DECAL_SHAPE="star"
      DECAL_COLOR="red"
      DECAL_BOX="0.405,0.340,0.565,0.560"
      ;;
    P6|tote_leaf)
      TASK_NAME="tote_leaf"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg"
      SOURCE_PROMPT="A photo of a plain white canvas tote bag held in front of a green wall."
      TARGET_PROMPT="A photo of the same white canvas tote bag with a green leaf logo printed on the front, in front of the same green wall."
      ATTENTION_TARGET_WORDS="leaf,logo,tote,bag"
      DECAL_SHAPE="leaf"
      DECAL_COLOR="green"
      DECAL_BOX="0.365,0.505,0.610,0.705"
      ;;
    P7|backpack_remove_toy_charm)
      TASK_NAME="backpack_remove_toy_charm"
      TASK_KIND="remove_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg"
      SOURCE_PROMPT="A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap."
      TARGET_PROMPT="A close-up photo of the same grey backpack with the yellow dangling toy charm removed, pink strap, zipper, and fabric preserved."
      ATTENTION_TARGET_WORDS="toy,charm,backpack,zipper,fabric"
      CHANGED_TARGET_WORDS="toy,charm"
      SUPPORT_EDIT_OPERATION="remove_object"
      SUPPORT_NEW_TOKENS=""
      SUPPORT_HOST_TOKENS="backpack"
      SUPPORT_REMOVED_TOKENS="toy,charm"
      SUPPORT_V2_CANDIDATE="removed_src_x_clean"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="yellow dangling toy charm"
      SEMANTIC_DILATE="6"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P8|backpack_replace_patch_blue|backpack_replace_patch_badge)
      TASK_NAME="backpack_replace_patch_blue"
      TASK_KIND="replace_semantic_badge"
      IMAGE="${ROOT}/data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg"
      SOURCE_PROMPT="A close-up photo of a grey backpack with a colorful cartoon patch on the front pocket."
      TARGET_PROMPT="A close-up photo of the same grey backpack with the colorful cartoon patch replaced by a plain blue fabric patch, zipper and fabric preserved."
      ATTENTION_TARGET_WORDS="blue,patch,backpack,fabric"
      SEMANTIC_PHRASE="colorful cartoon patch"
      SEMANTIC_DILATE="3"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P9|backpack_replace_toy_heart_charm)
      TASK_NAME="backpack_replace_toy_heart_charm"
      TASK_KIND="replace_semantic_badge"
      IMAGE="${ROOT}/data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg"
      SOURCE_PROMPT="A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap."
      TARGET_PROMPT="A close-up photo of the same grey backpack with the yellow dangling toy charm replaced by a small red heart-shaped keychain charm, pink strap, zipper, and fabric preserved."
      ATTENTION_TARGET_WORDS="heart,charm,backpack,zipper,fabric"
      SEMANTIC_PHRASE="yellow dangling toy charm"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      REPLACEMENT_COLOR="red"
      REPLACEMENT_SHAPE="heart"
      REPLACEMENT_OPACITY="0.96"
      REPLACEMENT_SCALE="0.72"
      REPLACEMENT_BLUR="0.8"
      REPLACEMENT_BACKGROUND_MODE="inpaint"
      REPLACEMENT_OUTLINE="1"
      REPLACEMENT_REF_GUIDANCE_SCALE="0.58"
      REPLACEMENT_REC_GUIDANCE_SCALE="0.24"
      REPLACEMENT_TRAJECTORY_PRESERVE_SCALE="0.18"
      ;;
    P10|cat_replace_bell_heart_tag)
      TASK_NAME="cat_replace_bell_heart_tag"
      TASK_KIND="replace_semantic_badge"
      IMAGE="${ROOT}/data/replacement_candidates/commons_black_white_cat_belled_collar.jpg"
      SOURCE_PROMPT="A close-up photo of a black and white cat wearing a collar with a small gold bell."
      TARGET_PROMPT="A close-up photo of the same black and white cat wearing the same collar with the small gold bell replaced by a red heart-shaped collar tag."
      ATTENTION_TARGET_WORDS="heart,tag,collar,bell,cat"
      SEMANTIC_PHRASE="small gold bell"
      SEMANTIC_DILATE="4"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      REPLACEMENT_COLOR="red"
      REPLACEMENT_SHAPE="heart"
      REPLACEMENT_OPACITY="0.95"
      REPLACEMENT_SCALE="0.90"
      REPLACEMENT_BLUR="0.6"
      REPLACEMENT_BACKGROUND_MODE="inpaint"
      REPLACEMENT_OUTLINE="1"
      REPLACEMENT_REF_GUIDANCE_SCALE="0.54"
      REPLACEMENT_REC_GUIDANCE_SCALE="0.20"
      REPLACEMENT_TRAJECTORY_PRESERVE_SCALE="0.16"
      ;;
    P11|dog_replace_tennis_ball_star|dog_replace_tennis_ball_frisbee)
      TASK_NAME="dog_replace_tennis_ball_star"
      TASK_KIND="replace_semantic_badge"
      IMAGE="${ROOT}/data/replacement_candidates/commons_dog_with_tennis_ball.jpg"
      SOURCE_PROMPT="A photo of a wet dog in a pool holding a green tennis ball in its mouth."
      TARGET_PROMPT="A photo of the same wet dog in the same pool holding a red star-shaped dog toy in its mouth instead of the green tennis ball."
      ATTENTION_TARGET_WORDS="star,toy,mouth,dog"
      SEMANTIC_PHRASE="green tennis ball"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      REPLACEMENT_COLOR="red"
      REPLACEMENT_SHAPE="star"
      REPLACEMENT_OPACITY="0.94"
      REPLACEMENT_SCALE="0.95"
      REPLACEMENT_BLUR="0.8"
      REPLACEMENT_BACKGROUND_MODE="inpaint"
      REPLACEMENT_OUTLINE="1"
      REPLACEMENT_REF_GUIDANCE_SCALE="0.56"
      REPLACEMENT_REC_GUIDANCE_SCALE="0.22"
      REPLACEMENT_TRAJECTORY_PRESERVE_SCALE="0.16"
      ;;
    *)
      echo "Unknown TASK '${task_id}'. Valid: P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11." >&2
      exit 2
      ;;
  esac
}

method_config() {
  local method_id="$1"
  METHOD_NAME=""
  METHOD_ROUTE="base"
  METHOD_ABLATION="none"
  SRC_GUIDANCE_SCALE="1.0"
  BASE_GUIDANCE_SCALE="1.0"
  TAR_GUIDANCE_SCALE="10.5"
  EDIT_HEDIT_GUIDANCE_SCALE="0.0"
  EDIT_TEXT_GUIDANCE_SCALE="0.0"
  EDIT_REF_GUIDANCE_SCALE="0.0"
  REC_GUIDANCE_SCALE="0.0"
  STRUCT_GUIDANCE_SCALE="0.45"
  TRAJECTORY_PRESERVE_SCALE="0.0"
  ADAPTIVE_CLEAN_CONTROL="0"
  ADAPTIVE_EDIT_TARGET_PROGRESS="0.65"
  ADAPTIVE_EDIT_TARGET_RMS="0.0"
  ADAPTIVE_PRESERVE_DRIFT_BUDGET="0.18"
  ADAPTIVE_EDIT_GAIN="2.0"
  ADAPTIVE_PRESERVE_GAIN="2.5"
  ADAPTIVE_EDIT_WEIGHT_MIN="0.85"
  ADAPTIVE_EDIT_WEIGHT_MAX="1.55"
  ADAPTIVE_PRESERVE_WEIGHT_MIN="1.0"
  ADAPTIVE_PRESERVE_WEIGHT_MAX="1.65"
  ADAPTIVE_PROJECTION_SCALE="0.65"
  GENERIC_SUPPORT="0"
  GENERIC_SUPPORT_V2="0"
  GENERIC_SUPPORT_V3="0"
  MANUAL_SUPPORT="0"
  SUPPORT_SCORE="attention_x_clean"
  SUPPORT_TOP_PERCENTILE="${SUPPORT_TOP_PERCENTILE:-95}"
  SUPPORT_MIN_AREA_RATIO="${SUPPORT_MIN_AREA_RATIO:-0.02}"
  SUPPORT_MAX_AREA_RATIO="${SUPPORT_MAX_AREA_RATIO:-0.10}"
  SUPPORT_KEEP_COMPONENTS="${SUPPORT_KEEP_COMPONENTS:-1}"
  SUPPORT_DILATE_RADIUS="${SUPPORT_DILATE_RADIUS:-3}"
  SUPPORT_BLUR_KERNEL="${SUPPORT_BLUR_KERNEL:-3}"
  OBJECT_MASK_PROVIDER="attention_velocity"
  MASK_LAYERING_MODE="object_contact"

  case "${method_id}" in
    M0|base_only)
      METHOD_NAME="base_only"
      ;;
    M1|direct_target)
      METHOD_NAME="direct_target"
      EDIT_HEDIT_GUIDANCE_SCALE="1.0"
      ;;
    M4|full)
      METHOD_NAME="full"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ;;
    M8|adaptive_full)
      METHOD_NAME="adaptive_full"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ;;
    M9|adaptive_full_v0)
      METHOD_NAME="adaptive_full_v0"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      ;;
    M10|adaptive_full_generic_support|generic_support|generic_support_v1)
      METHOD_NAME="adaptive_full_generic_support"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      OBJECT_MASK_PROVIDER="generic_support"
      ;;
    M15|adaptive_full_support_v2|support_v2_minimal|operation_aware_support_v2)
      METHOD_NAME="adaptive_full_support_v2"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V2="1"
      OBJECT_MASK_PROVIDER="generic_support"
      ;;
    M16|adaptive_full_support_v3|support_v3_grounded)
      METHOD_NAME="adaptive_full_support_v3"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ;;
    manual_support)
      METHOD_NAME="manual_support"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="0"
      MANUAL_SUPPORT="1"
      OBJECT_MASK_PROVIDER="semantic"
      ;;
    M11|adaptive_full_attention_only|generic_attention_only)
      METHOD_NAME="adaptive_full_attention_only"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      OBJECT_MASK_PROVIDER="generic_support"
      SUPPORT_SCORE="attention_only"
      ;;
    M12|adaptive_full_clean_only|generic_clean_only)
      METHOD_NAME="adaptive_full_clean_only"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      OBJECT_MASK_PROVIDER="generic_support"
      SUPPORT_SCORE="clean_disagreement_only"
      ;;
    M13|adaptive_full_velocity_only|generic_velocity_only)
      METHOD_NAME="adaptive_full_velocity_only"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      OBJECT_MASK_PROVIDER="generic_support"
      SUPPORT_SCORE="velocity_disagreement_only"
      ;;
    M14|adaptive_full_attention_x_velocity|generic_attention_x_velocity)
      METHOD_NAME="adaptive_full_attention_x_velocity"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      GENERIC_SUPPORT="1"
      OBJECT_MASK_PROVIDER="generic_support"
      SUPPORT_SCORE="attention_x_velocity"
      ;;
    M5|full_no_ref)
      METHOD_NAME="full_no_ref"
      METHOD_ROUTE="full"
      METHOD_ABLATION="no_ref"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ;;
    M6|full_no_rec)
      METHOD_NAME="full_no_rec"
      METHOD_ROUTE="full"
      METHOD_ABLATION="no_rec"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ;;
    M7|full_no_traj)
      METHOD_NAME="full_no_traj"
      METHOD_ROUTE="full"
      METHOD_ABLATION="no_traj"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ;;
    *)
      echo "Unknown METHOD '${method_id}'. Valid: M0 M1 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 M16 / base_only direct_target full full_no_ref full_no_rec full_no_traj adaptive_full adaptive_full_v0 adaptive_full_generic_support generic_support support_v2_minimal support_v3_grounded generic_attention_only generic_clean_only generic_velocity_only generic_attention_x_velocity." >&2
      exit 2
      ;;
  esac
}

ensure_semantic_mask() {
  local out_dir="$1"
  SUPPORT_MASK="${out_dir}/masks/semantic_support.png"
  local support_meta="${out_dir}/masks/semantic_support.json"
  local anchor_mask="${out_dir}/masks/semantic_anchor.png"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${SUPPORT_MASK}" ]]; then
    return 0
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
    --opacity "0.94"
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

ensure_badge_reference() {
  local out_dir="$1"
  BADGE_REF_IMAGE="${out_dir}/masks/badge_reference.png"
  BADGE_MASK="${out_dir}/masks/badge_reference_mask.png"
  local badge_overlay="${out_dir}/masks/badge_reference_overlay.png"
  local badge_meta="${out_dir}/masks/badge_reference.json"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${BADGE_REF_IMAGE}" && -s "${BADGE_MASK}" ]]; then
    return 0
  fi
  local cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_mask_badge_reference.py"
    --image "${IMAGE}"
    --semantic-mask "${SUPPORT_MASK}"
    --output "${BADGE_REF_IMAGE}"
    --mask-output "${BADGE_MASK}"
    --overlay-output "${badge_overlay}"
    --metadata-output "${badge_meta}"
    --color "${REPLACEMENT_COLOR}"
    --badge-shape "${REPLACEMENT_SHAPE}"
    --opacity "${REPLACEMENT_OPACITY}"
    --scale "${REPLACEMENT_SCALE}"
    --blur "${REPLACEMENT_BLUR}"
    --background-mode "${REPLACEMENT_BACKGROUND_MODE}"
  )
  if [[ "${REPLACEMENT_OUTLINE}" == "1" ]]; then
    cmd+=(--outline)
  fi
  printf '%q ' "${cmd[@]}" > "${out_dir}/badge_reference_command.txt"
  printf '\n' >> "${out_dir}/badge_reference_command.txt"
  echo "[pretty-matrix] generating badge reference: ${BADGE_REF_IMAGE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/badge_reference_command.txt"
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

ensure_glasses_reference() {
  local out_dir="$1"
  GLASSES_REF_IMAGE="${out_dir}/masks/glasses_reference.png"
  GLASSES_MASK="${out_dir}/masks/glasses_reference_mask.png"
  local glasses_overlay="${out_dir}/masks/glasses_reference_overlay.png"
  local glasses_meta="${out_dir}/masks/glasses_reference.json"
  local support_meta="${out_dir}/masks/semantic_support.json"
  if [[ "${REGENERATE_MASKS}" != "1" && -s "${GLASSES_REF_IMAGE}" && -s "${GLASSES_MASK}" ]]; then
    return 0
  fi
  local cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_glasses_reference.py"
    --image "${IMAGE}"
    --semantic-metadata "${support_meta}"
    --output "${GLASSES_REF_IMAGE}"
    --mask-output "${GLASSES_MASK}"
    --overlay-output "${glasses_overlay}"
    --metadata-output "${glasses_meta}"
  )
  printf '%q ' "${cmd[@]}" > "${out_dir}/glasses_reference_command.txt"
  printf '\n' >> "${out_dir}/glasses_reference_command.txt"
  echo "[pretty-matrix] generating glasses reference: ${GLASSES_REF_IMAGE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${out_dir}/glasses_reference_command.txt"
  else
    "${cmd[@]}"
  fi
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

run_one() {
  local task_id="$1"
  local method_id="$2"
  local seed="$3"
  task_config "${task_id}"
  method_config "${method_id}"

  local out_dir="${ROOT}/outputs/pretty_matrix/${TASK_NAME}/${METHOD_NAME}/seed_${seed}"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${out_dir}/result.png" && -s "${out_dir}/stats.json" && -s "${out_dir}/metadata.json" && -s "${out_dir}/command.txt" ]]; then
    echo "[pretty-matrix] skip existing complete run: ${out_dir}"
    return 0
  fi
  mkdir -p "${out_dir}/masks"

  SUPPORT_MASK=""
  DECAL_REF_IMAGE=""
  DECAL_MASK=""
  GLASSES_REF_IMAGE=""
  GLASSES_MASK=""
  BADGE_REF_IMAGE=""
  BADGE_MASK=""
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
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE:-operation_default}"
      ATTENTION_TARGET_WORDS="${SUPPORT_NEW_TOKENS:-${ATTENTION_TARGET_WORDS}}"
      local v3_relation="${SUPPORT_V3_RELATION:-auto}"
      # The semantic mask is used as grounding evidence for v3. Relation
      # proposal itself is built inside operation_support_v3.
      SUPPORT_RELATION="inside"
      ensure_semantic_mask "${out_dir}"
      SUPPORT_RELATION="${v3_relation}"
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
      accessory_structure)
        OBJECT_MASK_PROVIDER="structure"
        AUTO_STRUCTURE_FLAGS=(--auto-structure-boxes --auto-structure-external-mask --structure-glasses-angle-mode auto)
        EDIT_HEDIT_GUIDANCE_SCALE="0.62"
        EDIT_TEXT_GUIDANCE_SCALE="0.04"
        TRAJECTORY_PRESERVE_SCALE="0.18"
        REC_GUIDANCE_SCALE="0.0"
        ;;
      accessory_semantic_glasses)
        OBJECT_MASK_PROVIDER="semantic"
        ensure_semantic_mask "${out_dir}"
        ensure_glasses_reference "${out_dir}"
        EDIT_HEDIT_GUIDANCE_SCALE="0.62"
        EDIT_TEXT_GUIDANCE_SCALE="0.04"
        EDIT_REF_GUIDANCE_SCALE="0.46"
        TRAJECTORY_PRESERVE_SCALE="0.18"
        REC_GUIDANCE_SCALE="0.0"
        FINAL_MASK_ARGS=(--final-edit-mask "${GLASSES_MASK}" --final-edit-mask-mode replace)
        REF_ARGS=(
          --edit-ref-image "${GLASSES_REF_IMAGE}"
          --edit-ref-mask "${GLASSES_MASK}"
          --edit-ref-structure-image "${IMAGE}"
          --edit-ref-chroma-mode yuv
          --edit-ref-luma-preserve-scale 0.12
          --edit-ref-gradient-preserve-scale 0.04
          --edit-ref-darkness-guard-scale 0.18
          --edit-ref-smooth-kernel 1
        )
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
        if [[ -z "${RECOLOR_BOX}" ]]; then
          OBJECT_MASK_PROVIDER="semantic_velocity"
          ensure_semantic_mask "${out_dir}"
          ensure_refined_surface_mask "${out_dir}"
        else
          OBJECT_MASK_PROVIDER="attention_velocity"
        fi
        ensure_recolor_reference "${out_dir}"
        EDIT_HEDIT_GUIDANCE_SCALE="0.18"
        EDIT_TEXT_GUIDANCE_SCALE="0.02"
        REC_GUIDANCE_SCALE="0.58"
        TRAJECTORY_PRESERVE_SCALE="0.48"
        EDIT_REF_GUIDANCE_SCALE="0.46"
        EDIT_COLOR_GUIDANCE_SCALE="0.06"
        EDIT_COLOR_ARGS=(
          --edit-color-guidance-scale "${EDIT_COLOR_GUIDANCE_SCALE}"
          --edit-color-source "${RECOLOR_SOURCE_COLOR}"
          --edit-color-target "${RECOLOR_TARGET_COLOR}"
          --edit-color-mask-image "${SUPPORT_MASK}"
          --edit-color-mask-threshold 0.20
          --edit-color-target-chroma-scale 0.82
          --edit-color-luma-preserve-scale 0.72
          --edit-color-luma-gradient-preserve-scale 0.36
        )
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
      replace_semantic_badge)
        OBJECT_MASK_PROVIDER="semantic_velocity"
        ensure_semantic_mask "${out_dir}"
        ensure_badge_reference "${out_dir}"
        EDIT_HEDIT_GUIDANCE_SCALE="${REPLACEMENT_HEDIT_GUIDANCE_SCALE}"
        EDIT_TEXT_GUIDANCE_SCALE="${REPLACEMENT_TEXT_GUIDANCE_SCALE}"
        EDIT_REF_GUIDANCE_SCALE="${REPLACEMENT_REF_GUIDANCE_SCALE}"
        REC_GUIDANCE_SCALE="${REPLACEMENT_REC_GUIDANCE_SCALE}"
        TRAJECTORY_PRESERVE_SCALE="${REPLACEMENT_TRAJECTORY_PRESERVE_SCALE}"
        FINAL_MASK_ARGS=(--final-edit-mask "${SUPPORT_MASK}" --final-edit-mask-mode replace)
        REF_ARGS=(
          --edit-ref-image "${BADGE_REF_IMAGE}"
          --edit-ref-mask "${SUPPORT_MASK}"
          --edit-ref-structure-image "${IMAGE}"
          --edit-ref-chroma-mode yuv
          --edit-ref-luma-preserve-scale 0.18
          --edit-ref-gradient-preserve-scale 0.05
          --edit-ref-smooth-kernel 1
        )
        ;;
    esac
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
    --edit-text-guidance-scale "${EDIT_TEXT_GUIDANCE_SCALE}"
    --edit-ref-guidance-scale "${EDIT_REF_GUIDANCE_SCALE}"
    --rec-guidance-scale "${REC_GUIDANCE_SCALE}"
    --struct-guidance-scale "${STRUCT_GUIDANCE_SCALE}"
    --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
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
  )
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
      --adaptive-preserve-drift-budget "${ADAPTIVE_PRESERVE_DRIFT_BUDGET}"
      --adaptive-edit-gain "${ADAPTIVE_EDIT_GAIN}"
      --adaptive-preserve-gain "${ADAPTIVE_PRESERVE_GAIN}"
      --adaptive-edit-weight-min "${ADAPTIVE_EDIT_WEIGHT_MIN}"
      --adaptive-edit-weight-max "${ADAPTIVE_EDIT_WEIGHT_MAX}"
      --adaptive-preserve-weight-min "${ADAPTIVE_PRESERVE_WEIGHT_MIN}"
      --adaptive-preserve-weight-max "${ADAPTIVE_PRESERVE_WEIGHT_MAX}"
      --adaptive-projection-scale "${ADAPTIVE_PROJECTION_SCALE}"
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
