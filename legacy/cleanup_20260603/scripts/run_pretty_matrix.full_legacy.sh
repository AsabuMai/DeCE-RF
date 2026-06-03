#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${ROOT:-${DEFAULT_ROOT}}"
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
  SUPPORT_LOCAL_TARGET_PROMPT=""
  SUPPORT_V2_CANDIDATE=""
  SUPPORT_V3_CANDIDATE="operation_default"
  SUPPORT_V3_RELATION="auto"
  SEMANTIC_PHRASE=""
  SEMANTIC_DILATE="0"
  SEMANTIC_BLUR="0"
  SUPPORT_RELATION="auto"
  SUPPORT_PRESET=""
  DECAL_SHAPE=""
  DECAL_COLOR=""
  DECAL_BOX=""
  DECAL_SLANT_X="0.0"
  DECAL_PERSPECTIVE_Y="0.0"
  DECAL_EDGE_FEATHER_RADIUS="0.0"
  DECAL_TOP_FEATHER_FRAC="0.0"
  DECAL_TOP_FEATHER_MIN_ALPHA="0.0"
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
  REPLACEMENT_OLD_PHRASE=""
  REPLACEMENT_TARGET_PHRASE=""

  case "${task_id}" in
    P1|cat_crown)
      TASK_NAME="cat_crown"
      TASK_KIND="accessory_semantic"
      IMAGE="${ROOT}/data/paper_images/cat_sitting_in_grass.jpg"
      SOURCE_PROMPT="A photo of a cat sitting in grass."
      TARGET_PROMPT="A photo of the same cat sitting in the same grass, wearing a small golden crown centered on top of its head between the ears."
      ATTENTION_TARGET_WORDS="crown,head"
      CHANGED_TARGET_WORDS="crown"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="crown"
      SUPPORT_HOST_TOKENS="cat,head"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="above_host"
      SEMANTIC_PHRASE="cat"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a small golden crown centered on top of a cat's head between the ears."
      ;;
    P2|dog_sunglasses)
      TASK_NAME="dog_sunglasses"
      TASK_KIND="accessory_semantic_glasses"
      IMAGE="${ROOT}/data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg"
      SOURCE_PROMPT="A front-facing portrait of a dog in snow."
      TARGET_PROMPT="A front-facing portrait of the same dog wearing black sunglasses aligned across both eyes in snow."
      ATTENTION_TARGET_WORDS="sunglasses,eyes"
      CHANGED_TARGET_WORDS="sunglasses"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="sunglasses"
      SUPPORT_HOST_TOKENS="dog,eyes"
      SUPPORT_V2_CANDIDATE="attention_x_clean"
      SUPPORT_V3_RELATION="on_face"
      SEMANTIC_PHRASE="dog head"
      SUPPORT_RELATION="front_glasses_auto"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up front-facing photo of black sunglasses aligned horizontally across both eyes of a dog."
      ;;
    bowl_apple_inside)
      TASK_NAME="bowl_apple_inside"
      TASK_KIND="object_in_container"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg"
      SOURCE_PROMPT="A top-down photo of an empty blue ceramic bowl on a wooden board in a tidy table setting, with no fruit inside the bowl."
      TARGET_PROMPT="A top-down photo of the same blue ceramic bowl on the same wooden board, with one small red apple centered inside the bowl, while the bowl, board, tableware, leaves, and background remain unchanged."
      ATTENTION_TARGET_WORDS="apple,bowl,inside"
      CHANGED_TARGET_WORDS="apple"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="apple"
      SUPPORT_HOST_TOKENS="bowl,ceramic bowl"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="inside_container"
      SEMANTIC_PHRASE="blue ceramic bowl"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up top-down photo of one small red apple centered inside the empty blue ceramic bowl, fully within the inner basin and away from the rim."
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
    P4O|red_office_chair_to_blue_office_chair|red_office_chair_blue)
      TASK_NAME="red_office_chair_to_blue_office_chair"
      TASK_KIND="recolor_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/unsplash_red_office_chair_concrete_lvVWRzm_NwY.jpg"
      SOURCE_PROMPT="A photo of a red office chair on a concrete floor."
      TARGET_PROMPT="A photo of the same red office chair on the same concrete floor, with only the chair color changed to deep blue."
      ATTENTION_TARGET_WORDS="chair,blue"
      CHANGED_TARGET_WORDS="chair"
      SUPPORT_EDIT_OPERATION="recolor"
      SUPPORT_NEW_TOKENS="blue"
      SUPPORT_HOST_TOKENS="chair"
      SUPPORT_REMOVED_TOKENS="chair"
      SUPPORT_V3_RELATION="inside"
      SEMANTIC_PHRASE="chair"
      SUPPORT_RELATION="inside"
      RECOLOR_TARGET_COLOR="blue"
      RECOLOR_SURFACE_NAME="chair"
      ;;
    P4|red_chair_blue|red_chair_to_blue_chair)
      TASK_NAME="red_chair_blue"
      TASK_KIND="recolor_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg"
      SOURCE_PROMPT="A photo of a red armless rounded upholstered chair in a stylish room."
      TARGET_PROMPT="A photo of the same armless rounded upholstered chair in the same stylish room, with only the fabric color changed to deep blue, no armrests added."
      ATTENTION_TARGET_WORDS="chair,blue"
      CHANGED_TARGET_WORDS="chair"
      SUPPORT_EDIT_OPERATION="recolor"
      SUPPORT_NEW_TOKENS="blue"
      SUPPORT_HOST_TOKENS="chair"
      SUPPORT_REMOVED_TOKENS="chair"
      SUPPORT_V3_RELATION="inside"
      SEMANTIC_PHRASE="chair"
      SUPPORT_RELATION="inside"
      RECOLOR_TARGET_COLOR="blue"
      RECOLOR_SURFACE_NAME="chair"
      ;;
    P5|tshirt_star)
      TASK_NAME="tshirt_star"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg"
      SOURCE_PROMPT="A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting."
      TARGET_PROMPT="The same person wearing the same white t-shirt and blue jeans, with a clearly visible medium-sized bright red star printed on the center chest, while preserving the fabric folds, shadows, jeans, pose, and background."
      ATTENTION_TARGET_WORDS="star,t-shirt,chest"
      CHANGED_TARGET_WORDS="star"
      SUPPORT_EDIT_OPERATION="add_decal"
      SUPPORT_PRESET="clothing_decal"
      SUPPORT_NEW_TOKENS="star"
      SUPPORT_HOST_TOKENS="t-shirt,shirt"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="t-shirt"
      SUPPORT_RELATION="inside"
      DECAL_SHAPE="star"
      DECAL_COLOR="red"
      DECAL_BOX="0.40,0.30,0.62,0.55"
      ;;
    pillow_blue_stripes)
      TASK_NAME="pillow_blue_stripes"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg"
      SOURCE_PROMPT="A cozy living room photo with a plain grey pillow on a sofa, soft natural light, and a simple home interior."
      TARGET_PROMPT="A photo of the same plain grey pillow with subtle blue horizontal stripes printed on its surface, with the pillow shape, sofa, table, wall, and background unchanged."
      ATTENTION_TARGET_WORDS="blue,stripes,pillow"
      CHANGED_TARGET_WORDS="blue,stripes"
      SUPPORT_EDIT_OPERATION="add_decal"
      SUPPORT_PRESET="surface_pattern"
      SUPPORT_NEW_TOKENS="blue,stripes"
      SUPPORT_HOST_TOKENS="pillow"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="grey pillow"
      SUPPORT_RELATION="inside"
      DECAL_SHAPE="stripes"
      DECAL_COLOR="42,108,196"
      DECAL_BOX="0.40,0.44,0.72,0.70"
      ;;
    pillow_vertical_fabric_strip)
      TASK_NAME="pillow_vertical_fabric_strip"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg"
      SOURCE_PROMPT="A cozy living room photo with a plain grey pillow on a sofa, soft natural light, and a simple home interior."
      TARGET_PROMPT="A photo of the same plain grey pillow with one vertical glossy blue silk strip sewn down the center of the pillow surface, tucked naturally into the top pillow seam with soft satin highlights and smooth silk texture, while the pillow shape, sofa, table, wall, and background remain unchanged."
      ATTENTION_TARGET_WORDS="blue,silk,vertical,strip,pillow"
      CHANGED_TARGET_WORDS="blue,silk,strip"
      SUPPORT_EDIT_OPERATION="add_decal"
      SUPPORT_PRESET="surface_strip"
      SUPPORT_NEW_TOKENS="blue,silk,strip"
      SUPPORT_HOST_TOKENS="pillow"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="grey pillow"
      SUPPORT_RELATION="inside"
      DECAL_SHAPE="slanted_rectangle"
      DECAL_COLOR="58,132,215"
      DECAL_BOX="0.565,0.445,0.665,0.688"
      DECAL_SLANT_X="-0.08"
      DECAL_PERSPECTIVE_Y="0.055"
      DECAL_EDGE_FEATHER_RADIUS="7.0"
      DECAL_TOP_FEATHER_FRAC="0.16"
      DECAL_TOP_FEATHER_MIN_ALPHA="0.0"
      DECAL_OPACITY="0.80"
      ;;
    P6|tote_leaf)
      TASK_NAME="tote_leaf"
      TASK_KIND="decal"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg"
      SOURCE_PROMPT="A photo of a plain white canvas tote bag held in front of a green wall."
      TARGET_PROMPT="A photo of the same white canvas tote bag with a green leaf logo printed on the front, in front of the same green wall."
      ATTENTION_TARGET_WORDS="leaf,logo,tote,bag"
      CHANGED_TARGET_WORDS="leaf,logo"
      SUPPORT_EDIT_OPERATION="add_decal"
      SUPPORT_NEW_TOKENS="leaf,logo"
      SUPPORT_HOST_TOKENS="tote,bag"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="canvas tote bag"
      SUPPORT_RELATION="inside"
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
      CHANGED_TARGET_WORDS="blue,patch"
      SUPPORT_EDIT_OPERATION="replace"
      SUPPORT_NEW_TOKENS="blue,patch"
      SUPPORT_HOST_TOKENS="backpack"
      SUPPORT_REMOVED_TOKENS="patch"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="colorful cartoon patch"
      REPLACEMENT_OLD_PHRASE="colorful cartoon patch"
      REPLACEMENT_TARGET_PHRASE="plain blue fabric patch"
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
      CHANGED_TARGET_WORDS="heart,charm"
      SUPPORT_EDIT_OPERATION="replace"
      SUPPORT_NEW_TOKENS="heart,charm"
      SUPPORT_HOST_TOKENS="backpack"
      SUPPORT_REMOVED_TOKENS="toy,charm"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="yellow dangling toy charm"
      REPLACEMENT_OLD_PHRASE="yellow dangling toy charm"
      REPLACEMENT_TARGET_PHRASE="small red heart-shaped keychain charm"
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
      CHANGED_TARGET_WORDS="heart,tag"
      SUPPORT_EDIT_OPERATION="replace"
      SUPPORT_NEW_TOKENS="heart,tag"
      SUPPORT_HOST_TOKENS="cat,collar"
      SUPPORT_REMOVED_TOKENS="bell"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="small gold bell"
      REPLACEMENT_OLD_PHRASE="small gold bell"
      REPLACEMENT_TARGET_PHRASE="red heart-shaped collar tag"
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
      CHANGED_TARGET_WORDS="star,toy"
      SUPPORT_EDIT_OPERATION="replace"
      SUPPORT_NEW_TOKENS="star,toy"
      SUPPORT_HOST_TOKENS="dog,mouth"
      SUPPORT_REMOVED_TOKENS="tennis,ball"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="green tennis ball"
      REPLACEMENT_OLD_PHRASE="green tennis ball"
      REPLACEMENT_TARGET_PHRASE="red star-shaped dog toy"
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
    P14|dog_remove_tennis_ball)
      TASK_NAME="dog_remove_tennis_ball"
      TASK_KIND="remove_semantic"
      IMAGE="${ROOT}/data/replacement_candidates/commons_dog_with_tennis_ball.jpg"
      SOURCE_PROMPT="A photo of a wet dog in a pool holding a green tennis ball in its mouth."
      TARGET_PROMPT="A photo of the same wet dog in the same pool with the green tennis ball removed, mouth, face, fur, and pool preserved."
      ATTENTION_TARGET_WORDS="tennis,ball,mouth,dog"
      CHANGED_TARGET_WORDS="tennis,ball"
      SUPPORT_EDIT_OPERATION="remove_object"
      SUPPORT_NEW_TOKENS=""
      SUPPORT_HOST_TOKENS="dog,mouth"
      SUPPORT_REMOVED_TOKENS="tennis,ball"
      SUPPORT_V2_CANDIDATE="removed_src_x_clean"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="green tennis ball"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P15|laptop_remove_sticker)
      TASK_NAME="laptop_remove_sticker"
      TASK_KIND="remove_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_laptop_sticker_24381087.jpg"
      SOURCE_PROMPT="A top-view photo of a silver laptop on a desk with a colorful paper sticker on the palm rest, keyboard, trackpad, tea cup, and plate visible."
      TARGET_PROMPT="A top-view photo of the same silver laptop on the same desk with the colorful paper sticker removed from the palm rest, preserving the smooth laptop surface, keyboard, trackpad, tea cup, plate, lighting, and composition."
      ATTENTION_TARGET_WORDS="sticker,laptop,palm,rest"
      CHANGED_TARGET_WORDS="sticker"
      SUPPORT_EDIT_OPERATION="remove_object"
      SUPPORT_NEW_TOKENS=""
      SUPPORT_HOST_TOKENS="laptop,palm rest"
      SUPPORT_REMOVED_TOKENS="sticker,paper sticker"
      SUPPORT_V2_CANDIDATE="removed_src_x_clean"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="paper sticker"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P16|fridge_remove_yellow_magnet)
      TASK_NAME="fridge_remove_yellow_magnet"
      TASK_KIND="remove_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_blue_fridge_magnets_15555954.jpg"
      SOURCE_PROMPT="A photo of a bright blue refrigerator door covered with colorful fridge magnets, including a yellow round magnet with a black insect near the upper middle."
      TARGET_PROMPT="A photo of the same bright blue refrigerator door with the yellow round insect magnet removed, preserving the smooth blue fridge surface and all other magnets."
      ATTENTION_TARGET_WORDS="yellow,round,insect,magnet,fridge"
      CHANGED_TARGET_WORDS="yellow,round,insect,magnet"
      SUPPORT_EDIT_OPERATION="remove_object"
      SUPPORT_NEW_TOKENS=""
      SUPPORT_HOST_TOKENS="blue refrigerator door,fridge surface"
      SUPPORT_REMOVED_TOKENS="yellow round magnet,yellow insect magnet,magnet"
      SUPPORT_V2_CANDIDATE="removed_src_x_clean"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="yellow round magnet with black insect"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P17|fridge_remove_peach_magnet)
      TASK_NAME="fridge_remove_peach_magnet"
      TASK_KIND="remove_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_blue_fridge_magnets_15555954.jpg"
      SOURCE_PROMPT="A photo of a bright blue refrigerator door covered with colorful fridge magnets, including a peach-colored round magnet near the upper right."
      TARGET_PROMPT="A photo of the same bright blue refrigerator door with the upper-right peach-colored round magnet removed, preserving the smooth blue fridge surface and all other magnets."
      ATTENTION_TARGET_WORDS="upper,right,peach,round,magnet,fridge"
      CHANGED_TARGET_WORDS="peach,round,magnet"
      SUPPORT_EDIT_OPERATION="remove_object"
      SUPPORT_NEW_TOKENS=""
      SUPPORT_HOST_TOKENS="blue refrigerator door,fridge surface"
      SUPPORT_REMOVED_TOKENS="upper-right peach round magnet,peach round magnet,magnet"
      SUPPORT_V2_CANDIDATE="removed_src_x_clean"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="upper right peach round magnet"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P18|whiteboard_remove_yellow_letter)
      TASK_NAME="whiteboard_remove_yellow_letter"
      TASK_KIND="remove_semantic"
      IMAGE="${ROOT}/data/pretty_free_candidates/pexels_magnetic_letters_whiteboard_5099795.jpg"
      SOURCE_PROMPT="A close-up photo of colorful magnetic plastic letters on a whiteboard, including a yellow letter I near the center-left."
      TARGET_PROMPT="A close-up photo of the same whiteboard with the yellow letter I removed, preserving the whiteboard surface, faint marker scribbles, and all other magnetic letters."
      ATTENTION_TARGET_WORDS="yellow,letter,I,magnetic,whiteboard"
      CHANGED_TARGET_WORDS="yellow,letter,I"
      SUPPORT_EDIT_OPERATION="remove_object"
      SUPPORT_NEW_TOKENS=""
      SUPPORT_HOST_TOKENS="whiteboard,whiteboard surface"
      SUPPORT_REMOVED_TOKENS="yellow letter I,magnetic letter I,letter I"
      SUPPORT_V2_CANDIDATE="removed_src_x_clean"
      SUPPORT_V3_RELATION="remove_source_object"
      SEMANTIC_PHRASE="yellow letter I"
      SEMANTIC_DILATE="5"
      SEMANTIC_BLUR="1"
      SUPPORT_RELATION="inside"
      ;;
    P12|rabbit_sunglasses)
      TASK_NAME="rabbit_sunglasses"
      TASK_KIND="accessory_semantic_glasses"
      IMAGE="${ROOT}/data/paper_images/rabbit_side_view.jpg"
      SOURCE_PROMPT="A photo of a rabbit sitting outdoors in side profile."
      TARGET_PROMPT="A photo of the same rabbit sitting outdoors in side profile, wearing small black sunglasses aligned over the visible eye area."
      ATTENTION_TARGET_WORDS="sunglasses,eyes"
      CHANGED_TARGET_WORDS="sunglasses"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="sunglasses"
      SUPPORT_HOST_TOKENS="rabbit,eyes"
      SUPPORT_V2_CANDIDATE="attention_x_clean"
      SUPPORT_V3_RELATION="on_face"
      SEMANTIC_PHRASE="rabbit head"
      SUPPORT_RELATION="front_glasses_auto"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up side-profile photo of small black sunglasses aligned over the rabbit's visible eye area."
      ;;
    P13|dog_crown)
      TASK_NAME="dog_crown"
      TASK_KIND="accessory_semantic"
      IMAGE="${ROOT}/data/paper_images/dog_sitting_cc0.jpg"
      SOURCE_PROMPT="A photo of a dog sitting."
      TARGET_PROMPT="A photo of the same dog sitting, wearing a small golden crown centered on top of its head between the ears."
      ATTENTION_TARGET_WORDS="crown,head"
      CHANGED_TARGET_WORDS="crown"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="crown"
      SUPPORT_HOST_TOKENS="dog,head"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="above_host"
      SEMANTIC_PHRASE="dog"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a small golden crown centered on top of a dog's head between the ears."
      ;;
    P19|web_plate_apple)
      TASK_NAME="web_plate_apple"
      TASK_KIND="object_on_surface"
      IMAGE="${ROOT}/data/web_add_object_candidates/pexels_empty_white_plate_2611817.jpg"
      SOURCE_PROMPT="A top-down minimalist photo of an empty white ceramic bowl on a pale marble surface."
      TARGET_PROMPT="A top-down minimalist photo of the same white ceramic bowl on the same pale marble surface, with one small red apple placed exactly in the center of the visible bowl interior, away from the rim."
      ATTENTION_TARGET_WORDS="apple,bowl,center"
      CHANGED_TARGET_WORDS="apple"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="apple"
      SUPPORT_HOST_TOKENS="bowl,plate"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="inside"
      SEMANTIC_PHRASE="white ceramic bowl"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up top-down photo of one small red apple centered inside the white ceramic bowl, fully within the inner circular basin and away from the rim."
      ;;
    P20|web_vase_flowers)
      TASK_NAME="web_vase_flowers"
      TASK_KIND="object_in_container"
      IMAGE="${ROOT}/data/web_add_object_candidates/pexels_white_ceramic_vase_36382219.jpg"
      SOURCE_PROMPT="A minimalist photo of an empty white ceramic vase displayed in a softly lit alcove."
      TARGET_PROMPT="A minimalist photo of the same white ceramic vase in the same softly lit alcove, with a small bouquet of yellow flowers emerging from the vase opening at the top center."
      ATTENTION_TARGET_WORDS="yellow,flowers,vase"
      CHANGED_TARGET_WORDS="flowers"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="flowers,bouquet"
      SUPPORT_HOST_TOKENS="vase"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="above_host"
      SEMANTIC_PHRASE="white ceramic vase"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a small bouquet of yellow flowers emerging directly from the top opening of a white ceramic vase, centered on the vase mouth."
      ;;
    P21|web_chair_cushion)
      TASK_NAME="web_chair_cushion"
      TASK_KIND="object_on_furniture"
      IMAGE="${ROOT}/data/web_add_object_candidates/pexels_empty_chair_room_20027127.jpg"
      SOURCE_PROMPT="A minimalist photo of a single empty chair in a bright empty room."
      TARGET_PROMPT="A minimalist photo of the same single empty chair in the same bright empty room, with a red cushion placed flat on the center of the chair seat."
      ATTENTION_TARGET_WORDS="red,cushion,chair,seat"
      CHANGED_TARGET_WORDS="cushion"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="cushion"
      SUPPORT_HOST_TOKENS="chair,seat"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="chair seat"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a red cushion placed flat on the center of a chair seat, aligned with the seat and not covering the chair back."
      ;;
    P22|web_frame_landscape)
      TASK_NAME="web_frame_landscape"
      TASK_KIND="object_inside_frame"
      IMAGE="${ROOT}/data/web_add_object_candidates/pexels_blank_frames_7318843.jpg"
      SOURCE_PROMPT="A photo of blank wooden picture frames standing on a wooden table in sunlight."
      TARGET_PROMPT="A photo of the same wooden picture frames on the same wooden table, with a colorful mountain landscape picture centered inside the blank area of the large front frame."
      ATTENTION_TARGET_WORDS="landscape,picture,frame,mountain"
      CHANGED_TARGET_WORDS="landscape,picture"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="landscape,picture,mountain"
      SUPPORT_HOST_TOKENS="front frame,large frame"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="inside"
      SEMANTIC_PHRASE="large front picture frame"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a colorful mountain landscape picture centered within the inner blank rectangle of a large wooden picture frame."
      ;;
    P23|web_desk_mug)
      TASK_NAME="web_desk_mug"
      TASK_KIND="object_on_desk"
      IMAGE="${ROOT}/data/web_add_object_candidates2/pexels_empty_desks_office_7534208.jpg"
      SOURCE_PROMPT="A photo of an empty modern office desk with chairs and a framed picture on the wall."
      TARGET_PROMPT="A photo of the same empty modern office desk with a small red coffee mug placed on the visible wooden desktop near the back center, preserving the chairs, wall picture, and room layout."
      ATTENTION_TARGET_WORDS="red,coffee,mug,desk"
      CHANGED_TARGET_WORDS="mug"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="mug,coffee mug"
      SUPPORT_HOST_TOKENS="wooden desk,desk"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="wooden desk"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a small red coffee mug sitting upright on the visible wooden desktop near the back center edge."
      ;;
    P24|web_wall_clock)
      TASK_NAME="web_wall_clock"
      TASK_KIND="object_on_wall"
      IMAGE="${ROOT}/data/web_add_object_candidates2/pexels_blank_room_7045322.jpg"
      SOURCE_PROMPT="A photo of an empty modern room with a long brick wall and wooden floor."
      TARGET_PROMPT="A photo of the same empty modern room with a round black wall clock hanging on the right brick wall at about eye level, preserving the floor, lighting, and room layout."
      ATTENTION_TARGET_WORDS="black,wall,clock,brick"
      CHANGED_TARGET_WORDS="clock"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="clock,wall clock"
      SUPPORT_HOST_TOKENS="brick wall,wall"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="brick wall"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a round black wall clock mounted on the right brick wall at about eye level, centered between brick rows."
      ;;
    P25|web_shelf_books)
      TASK_NAME="web_shelf_books"
      TASK_KIND="object_on_shelf"
      IMAGE="${ROOT}/data/web_add_object_candidates2/pexels_empty_shelves_7195887.jpg"
      SOURCE_PROMPT="A photo of an empty modern walk-in closet with dark wooden shelves."
      TARGET_PROMPT="A photo of the same empty modern walk-in closet with a small upright stack of colorful books placed on the middle right dark wooden shelf, preserving the closet layout and lighting."
      ATTENTION_TARGET_WORDS="colorful,books,shelf"
      CHANGED_TARGET_WORDS="books"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="books,colorful books"
      SUPPORT_HOST_TOKENS="shelf,shelves"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="dark wooden shelves"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up photo of a small upright stack of colorful books placed on the middle right dark wooden shelf."
      ;;
    P26|web_bowl_spoon)
      TASK_NAME="web_bowl_spoon"
      TASK_KIND="object_in_bowl"
      IMAGE="${ROOT}/data/web_add_object_candidates/pexels_empty_white_plate_2611817.jpg"
      SOURCE_PROMPT="A top-down minimalist photo of an empty white ceramic bowl on a pale marble surface."
      TARGET_PROMPT="A top-down minimalist photo of the same white ceramic bowl on the same pale marble surface, with a small stainless steel spoon placed diagonally inside the lower-right part of the bowl, bowl handle end pointing toward the lower right."
      ATTENTION_TARGET_WORDS="spoon,bowl"
      CHANGED_TARGET_WORDS="spoon"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="spoon"
      SUPPORT_HOST_TOKENS="bowl"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="inside"
      SEMANTIC_PHRASE="white ceramic bowl"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up top-down photo of a stainless steel spoon placed diagonally inside the lower-right part of a white ceramic bowl, with the handle pointing toward the lower right."
      ;;
    P27|web_notebook_pen)
      TASK_NAME="web_notebook_pen"
      TASK_KIND="object_on_notebook"
      IMAGE="${ROOT}/data/web_add_candidates/pexels_blank_notebook_5705957.jpg"
      SOURCE_PROMPT="A top-down photo of an open blank spiral notebook on a pale peach desk."
      TARGET_PROMPT="A top-down photo of the same open blank spiral notebook on the same pale peach desk, with a blue pen placed diagonally across the center of the notebook page, away from the spiral binding."
      ATTENTION_TARGET_WORDS="blue,pen,notebook,page"
      CHANGED_TARGET_WORDS="pen"
      SUPPORT_EDIT_OPERATION="add_object"
      SUPPORT_NEW_TOKENS="pen,blue pen"
      SUPPORT_HOST_TOKENS="notebook,page"
      SUPPORT_V2_CANDIDATE="new_plus_host_x_clean"
      SUPPORT_V3_RELATION="on_surface"
      SEMANTIC_PHRASE="notebook page"
      SUPPORT_RELATION="inside"
      SUPPORT_LOCAL_TARGET_PROMPT="A close-up top-down photo of a blue pen placed diagonally across the center of a blank notebook page, away from the spiral binding and not along the page edge."
      ;;
    *)
      echo "Unknown TASK '${task_id}'. Valid: P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19 P20 P21 P22 P23 P24 P25 P26 P27 plus bowl_apple_inside pillow_blue_stripes pillow_vertical_fabric_strip." >&2
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
  EDIT_GUIDANCE_SCALE="0.0"
  EDIT_REGION_GUIDANCE_SCALE="0.0"
  EDIT_TARGET_GUIDANCE_SCALE="0.0"
  EDIT_SOURCE_GUIDANCE_SCALE="0.0"
  EDIT_TEXT_GUIDANCE_SCALE="0.0"
  EDIT_TEXT_SOURCE_SCALE="0.8"
  EDIT_TEXT_CORE_WEIGHT="1.0"
  EDIT_TEXT_SUBJECT_WEIGHT="0.3"
  EDIT_TEXT_SOURCE_PROMPT=""
  EDIT_TEXT_TARGET_PROMPT=""
  EDIT_LOCAL_TARGET_PROMPT=""
  EDIT_LOCAL_TARGET_GUIDANCE_SCALE="0.0"
  EDIT_LOCAL_TARGET_CFG_SCALE=""
  EDIT_REF_GUIDANCE_SCALE="0.0"
  REC_GUIDANCE_SCALE="0.0"
  STRUCT_GUIDANCE_SCALE="0.45"
  TRAJECTORY_PRESERVE_SCALE="0.0"
  REGION_TARGET_TRANSPORT_SCALE="0.0"
  REGION_TARGET_OUTSIDE_LOCK_SCALE="0.0"
  ADAPTIVE_CLEAN_CONTROL="0"
  ADAPTIVE_EDIT_TARGET_PROGRESS="0.65"
  ADAPTIVE_EDIT_TARGET_RMS="0.0"
  ADAPTIVE_RMSGAP_MODE="legacy"
  ADAPTIVE_RMSGAP_DEAD_ZONE="0.0"
  ADAPTIVE_RMSGAP_PRESERVE_GATE_BUDGET="0.0"
  ADAPTIVE_HYBRID_PROGRESS_TARGET="0.0"
  ADAPTIVE_HYBRID_PROGRESS_GAIN="0.0"
  ADAPTIVE_HYBRID_PROGRESS_EMA_DECAY="0.0"
  ADAPTIVE_HYBRID_PRESERVE_GATE_BUDGET="0.0"
  ADAPTIVE_PRESERVE_DRIFT_BUDGET="0.18"
  ADAPTIVE_EDIT_GAIN="2.0"
  ADAPTIVE_PRESERVE_GAIN="2.5"
  ADAPTIVE_EDIT_WEIGHT_MIN="0.85"
  ADAPTIVE_EDIT_WEIGHT_MAX="1.55"
  ADAPTIVE_PRESERVE_WEIGHT_MIN="1.0"
  ADAPTIVE_PRESERVE_WEIGHT_MAX="1.65"
  ADAPTIVE_PROJECTION_SCALE="0.65"
  ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="0.0"
  REMOVAL_CONTROLLER_MODE="none"
  REMOVAL_FILL_SCALE="0.0"
  REMOVAL_SUPPRESSION_SCALE="0.0"
  REMOVAL_RING_REC_SCALE="0.0"
  OPERATION_EDIT_FIELD="0"
  GENERIC_SUPPORT="0"
  GENERIC_SUPPORT_V2="0"
  GENERIC_SUPPORT_V3="0"
  MANUAL_SUPPORT="0"
  SUPPORT_SCORE="attention_x_clean"
  SUPPORT_TOP_PERCENTILE="${SUPPORT_TOP_PERCENTILE_DEFAULT}"
  SUPPORT_MIN_AREA_RATIO="${SUPPORT_MIN_AREA_RATIO_DEFAULT}"
  SUPPORT_MAX_AREA_RATIO="${SUPPORT_MAX_AREA_RATIO_DEFAULT}"
  SUPPORT_KEEP_COMPONENTS="${SUPPORT_KEEP_COMPONENTS_DEFAULT}"
  SUPPORT_DILATE_RADIUS="${SUPPORT_DILATE_RADIUS_DEFAULT}"
  SUPPORT_BLUR_KERNEL="${SUPPORT_BLUR_KERNEL_DEFAULT}"
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
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      ;;
    M17|support_v3_fixed|fixed_full_support_v3)
      METHOD_NAME="support_v3_fixed"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="0"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ;;
    M18|support_v3_controller_rmsgap|adaptive_full_support_v3_v0)
      METHOD_NAME="support_v3_controller_rmsgap"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      ;;
    M19|support_v3_controller_progress|adaptive_full_support_v3_v1)
      METHOD_NAME="support_v3_controller_progress"
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
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      ;;
    M20|support_v3_controller_hybrid)
      METHOD_NAME="support_v3_controller_hybrid"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="${SUPPORT_V3_HYBRID_RMS_TARGET:-0.42}"
      ADAPTIVE_HYBRID_PROGRESS_TARGET="${SUPPORT_V3_HYBRID_PROGRESS_TARGET:-0.65}"
      ADAPTIVE_HYBRID_PROGRESS_GAIN="${SUPPORT_V3_HYBRID_PROGRESS_GAIN:-0.50}"
      ADAPTIVE_HYBRID_PROGRESS_EMA_DECAY="${SUPPORT_V3_HYBRID_PROGRESS_EMA_DECAY:-0.75}"
      ADAPTIVE_HYBRID_PRESERVE_GATE_BUDGET="${SUPPORT_V3_HYBRID_PRESERVE_GATE_BUDGET:-0.18}"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      ;;
    M21|support_v3_controller_rmsgap_normgate)
      METHOD_NAME="support_v3_controller_rmsgap_normgate"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.0"
      ADAPTIVE_RMSGAP_MODE="normgate"
      ADAPTIVE_RMSGAP_DEAD_ZONE="${SUPPORT_V3_NORMGATE_DEAD_ZONE:-0.15}"
      ADAPTIVE_RMSGAP_PRESERVE_GATE_BUDGET="${SUPPORT_V3_NORMGATE_PRESERVE_GATE_BUDGET:-0.18}"
      ADAPTIVE_EDIT_GAIN="${SUPPORT_V3_NORMGATE_EDIT_GAIN:-0.55}"
      ADAPTIVE_EDIT_WEIGHT_MAX="${SUPPORT_V3_NORMGATE_EDIT_WEIGHT_MAX:-1.35}"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      ;;
    M22|support_v3_core_target_transport|core_target_transport)
      METHOD_NAME="support_v3_core_target_transport"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_CTT_HEDIT_GUIDANCE_SCALE:-0.0}"
      EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_CTT_TEXT_GUIDANCE_SCALE:-0.08}"
      REC_GUIDANCE_SCALE="${SUPPORT_V3_CTT_REC_GUIDANCE_SCALE:-0.12}"
      TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_CTT_TRAJECTORY_PRESERVE_SCALE:-0.0}"
      REGION_TARGET_TRANSPORT_SCALE="${SUPPORT_V3_CTT_TRANSPORT_SCALE:-1.0}"
      REGION_TARGET_OUTSIDE_LOCK_SCALE="${SUPPORT_V3_CTT_OUTSIDE_LOCK_SCALE:-1.0}"
      ADAPTIVE_CLEAN_CONTROL="0"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ;;
    M23|support_v3_controller_rmsgap_opfield|rmsgap_opfield)
      METHOD_NAME="support_v3_controller_rmsgap_opfield"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      OPERATION_EDIT_FIELD="1"
      ;;
    M24|support_v3_controller_rmsgap_replace_v2|rmsgap_replace_v2|replace_v2)
      METHOD_NAME="support_v3_controller_rmsgap_replace_v2"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      OPERATION_EDIT_FIELD="replace_v2"
      ;;
    M25|support_v3_controller_rmsgap_replace_editor_v0|rmsgap_replace_editor_v0|replace_editor_v0)
      METHOD_NAME="support_v3_controller_rmsgap_replace_editor_v0"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      EDIT_TEXT_SOURCE_SCALE="0.8"
      EDIT_TEXT_CORE_WEIGHT="1.0"
      EDIT_TEXT_SUBJECT_WEIGHT="0.3"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      OPERATION_EDIT_FIELD="replace_editor_v0"
      ;;
    M26|support_v3_controller_rmsgap_replace_editor_v1|rmsgap_replace_editor_v1|replace_editor_v1)
      METHOD_NAME="support_v3_controller_rmsgap_replace_editor_v1"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      EDIT_TEXT_SOURCE_SCALE="0.8"
      EDIT_TEXT_CORE_WEIGHT="1.0"
      EDIT_TEXT_SUBJECT_WEIGHT="0.3"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      SUPPORT_SCORE="${SUPPORT_V3_CANDIDATE}"
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      OPERATION_EDIT_FIELD="replace_editor_v1"
      ;;
    M27|support_v3_controller_rmsgap_add_editor_v1|rmsgap_add_editor_v1|add_editor_v1)
      METHOD_NAME="support_v3_controller_rmsgap_add_editor_v1"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      case "${SUPPORT_V3_RELATION}" in
        inside|inside_host|on_surface|surface)
          SUPPORT_SCORE="${SUPPORT_V3_ADD_EDITOR_CANDIDATE:-seg_x_response}"
          ;;
        *)
          SUPPORT_SCORE="${SUPPORT_V3_ADD_EDITOR_CANDIDATE:-${SUPPORT_V3_CANDIDATE}}"
          ;;
      esac
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      OPERATION_EDIT_FIELD="add_editor_v1"
      ;;
    M28|support_v3_controller_rmsgap_add_editor_v2|rmsgap_add_editor_v2|add_editor_v2)
      METHOD_NAME="support_v3_controller_rmsgap_add_editor_v2"
      METHOD_ROUTE="full"
      METHOD_ABLATION="none"
      EDIT_HEDIT_GUIDANCE_SCALE="0.65"
      EDIT_TEXT_GUIDANCE_SCALE="0.08"
      REC_GUIDANCE_SCALE="0.22"
      TRAJECTORY_PRESERVE_SCALE="0.12"
      ADAPTIVE_CLEAN_CONTROL="1"
      ADAPTIVE_EDIT_TARGET_PROGRESS="0.0"
      ADAPTIVE_EDIT_TARGET_RMS="0.42"
      GENERIC_SUPPORT="1"
      GENERIC_SUPPORT_V3="1"
      OBJECT_MASK_PROVIDER="operation_support_v3"
      case "${SUPPORT_V3_RELATION}" in
        inside|inside_host)
          SUPPORT_SCORE="${SUPPORT_V3_ADD_EDITOR_CANDIDATE:-seg_x_response}"
          ;;
        on_surface|surface)
          SUPPORT_SCORE="${SUPPORT_V3_ADD_EDITOR_CANDIDATE:-host_spawn_wide_x_response}"
          ;;
        above_host|above|on_top|top)
          SUPPORT_SCORE="${SUPPORT_V3_ADD_EDITOR_CANDIDATE:-host_top_contact_x_response}"
          ;;
        *)
          SUPPORT_SCORE="${SUPPORT_V3_ADD_EDITOR_CANDIDATE:-${SUPPORT_V3_CANDIDATE}}"
          ;;
      esac
      ADAPTIVE_PRESERVE_CLEAN_CORRECTION_SCALE="${SUPPORT_V3_DIRECT_PRESERVE_SCALE:-0.5}"
      OPERATION_EDIT_FIELD="add_editor_v2"
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
      echo "Unknown METHOD '${method_id}'. Valid: M0 M1 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 / base_only direct_target full full_no_ref full_no_rec full_no_traj adaptive_full adaptive_full_v0 adaptive_full_generic_support generic_support support_v2_minimal support_v3_grounded support_v3_fixed support_v3_controller_rmsgap support_v3_controller_progress support_v3_controller_hybrid support_v3_controller_rmsgap_normgate support_v3_core_target_transport support_v3_controller_rmsgap_opfield support_v3_controller_rmsgap_replace_v2 support_v3_controller_rmsgap_replace_editor_v0 support_v3_controller_rmsgap_replace_editor_v1 fixed_full_support_v3 adaptive_full_support_v3_v0 adaptive_full_support_v3_v1 generic_attention_only generic_clean_only generic_velocity_only generic_attention_x_velocity." >&2
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
          elif [[ "${SUPPORT_PRESET}" == "surface_pattern" ]]; then
            if [[ "${SUPPORT_SCORE}" == "operation_default" || "${SUPPORT_SCORE}" == "auto" || -z "${SUPPORT_SCORE}" ]]; then
              SUPPORT_SCORE="${SUPPORT_V3_SURFACE_PATTERN_CANDIDATE:-host_spawn_wide_x_response}"
            fi
            SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_SURFACE_PATTERN_MIN_AREA_RATIO:-0.045}"
            SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_SURFACE_PATTERN_MAX_AREA_RATIO:-0.110}"
            SUPPORT_DILATE_RADIUS="${SUPPORT_V3_SURFACE_PATTERN_DILATE_RADIUS:-4}"
            SUPPORT_BLUR_KERNEL="${SUPPORT_V3_SURFACE_PATTERN_BLUR_KERNEL:-5}"
            REC_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_PATTERN_REC_GUIDANCE_SCALE:-0.40}"
            TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_SURFACE_PATTERN_TRAJECTORY_PRESERVE_SCALE:-0.22}"
            ADAPTIVE_PRESERVE_DRIFT_BUDGET="${SUPPORT_V3_SURFACE_PATTERN_PRESERVE_DRIFT_BUDGET:-0.15}"
            ADAPTIVE_PRESERVE_GAIN="${SUPPORT_V3_SURFACE_PATTERN_PRESERVE_GAIN:-3.8}"
            EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_PATTERN_TEXT_GUIDANCE_SCALE:-0.12}"
            EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_PATTERN_HEDIT_GUIDANCE_SCALE:-0.68}"
          else
            SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_DECAL_MIN_AREA_RATIO:-0.005}"
          fi
          if [[ "${SUPPORT_PRESET}" == "clothing_decal" ]]; then
            SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_CLOTHING_DECAL_MAX_AREA_RATIO:-${SUPPORT_V3_DECAL_MAX_AREA_RATIO:-0.060}}"
          elif [[ "${SUPPORT_PRESET}" != "surface_pattern" && "${SUPPORT_PRESET}" != "surface_strip" ]]; then
            SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_DECAL_MAX_AREA_RATIO:-0.035}"
          fi
          if [[ "${SUPPORT_PRESET}" != "surface_pattern" && "${SUPPORT_PRESET}" != "surface_strip" ]]; then
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
      if [[ "${SUPPORT_PRESET}" == "surface_pattern" ]]; then
        ensure_decal_reference "${out_dir}"
        EDIT_REF_GUIDANCE_SCALE="${SUPPORT_V3_SURFACE_PATTERN_REF_GUIDANCE_SCALE:-0.18}"
        REF_ARGS=(
          --edit-ref-image "${DECAL_REF_IMAGE}"
          --edit-ref-mask "${DECAL_MASK}"
          --edit-ref-structure-image "${IMAGE}"
          --edit-ref-chroma-mode yuv
          --edit-ref-luma-preserve-scale "${SUPPORT_V3_SURFACE_PATTERN_REF_LUMA_PRESERVE:-0.35}"
          --edit-ref-gradient-preserve-scale "${SUPPORT_V3_SURFACE_PATTERN_REF_GRADIENT_PRESERVE:-0.08}"
          --edit-ref-smooth-kernel 1
        )
      fi
      if [[ "${OPERATION_EDIT_FIELD}" != "0" ]]; then
        case "${SUPPORT_EDIT_OPERATION}:${SUPPORT_V3_RELATION}" in
          add_decal:*)
            EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_DECAL_HEDIT:-0.65}"
            EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_DECAL_TEXT:-0.12}"
            EDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_DECAL_ANCHOR:-0.04}"
            EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_DECAL_REGION:-0.06}"
            REC_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_DECAL_REC:-0.30}"
            TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_OPFIELD_DECAL_TRAJ:-0.18}"
            ;;
          replace:*)
            if [[ "${OPERATION_EDIT_FIELD}" == "replace_v2" ]]; then
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_HEDIT:-0.82}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_TEXT:-0.14}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_ANCHOR:-0.04}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_REGION:-0.08}"
              EDIT_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_TARGET:-0.04}"
              EDIT_SOURCE_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_SOURCE:-0.08}"
              REC_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_V2_REC:-0.16}"
              TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_REPLACE_V2_TRAJ:-0.08}"
            elif [[ "${OPERATION_EDIT_FIELD}" == "replace_editor_v0" || "${OPERATION_EDIT_FIELD}" == "replace_editor_v1" ]]; then
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_HEDIT:-0.88}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_TEXT:-0.22}"
              EDIT_TEXT_SOURCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_TEXT_SOURCE:-1.15}"
              EDIT_TEXT_CORE_WEIGHT="${SUPPORT_V3_REPLACE_EDITOR_TEXT_CORE:-1.45}"
              EDIT_TEXT_SUBJECT_WEIGHT="${SUPPORT_V3_REPLACE_EDITOR_TEXT_SUBJECT:-0.20}"
              EDIT_TEXT_SOURCE_PROMPT="${REPLACEMENT_OLD_PHRASE}"
              EDIT_TEXT_TARGET_PROMPT="${REPLACEMENT_TARGET_PHRASE}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_ANCHOR:-0.04}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_REGION:-0.08}"
              EDIT_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_TARGET:-0.04}"
              EDIT_SOURCE_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_SOURCE:-0.10}"
              REC_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_REC:-0.12}"
              TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_TRAJ:-0.04}"
              REGION_TARGET_TRANSPORT_SCALE="${SUPPORT_V3_REPLACE_EDITOR_TRANSPORT:-0.85}"
              REGION_TARGET_OUTSIDE_LOCK_SCALE="${SUPPORT_V3_REPLACE_EDITOR_OUTSIDE_LOCK:-0.80}"
              EDIT_INITIAL_NOISE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_NOISE:-0.18}"
              EDIT_INITIAL_NOISE_REGION="${SUPPORT_V3_REPLACE_EDITOR_NOISE_REGION:-core}"
              if [[ "${OPERATION_EDIT_FIELD}" == "replace_editor_v1" ]]; then
                EDIT_LOCAL_TARGET_PROMPT="${SUPPORT_V3_REPLACE_EDITOR_LOCAL_TARGET_PROMPT:-A close-up photo of ${REPLACEMENT_TARGET_PHRASE}.}"
                EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_REPLACE_EDITOR_LOCAL_TARGET:-0.30}"
                EDIT_LOCAL_TARGET_CFG_SCALE="${SUPPORT_V3_REPLACE_EDITOR_LOCAL_CFG:-8.0}"
              fi
            else
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_REPLACE_HEDIT:-0.75}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_REPLACE_TEXT:-0.12}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_REPLACE_ANCHOR:-0.04}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_REPLACE_REGION:-0.06}"
              EDIT_SOURCE_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_REPLACE_SOURCE:-0.04}"
            fi
            ;;
          add_object:on_face|add_object:face|add_object:eye_band|add_object:eyes)
            if [[ "${OPERATION_EDIT_FIELD}" == add_editor_v* ]]; then
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_HEDIT:-0.70}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_TEXT:-0.14}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_ANCHOR:-0.03}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_REGION:-0.05}"
              REC_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_REC:-0.14}"
              TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_TRAJ:-0.06}"
              SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_FACE_MIN_AREA_RATIO:-0.006}"
              SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_FACE_MAX_AREA_RATIO:-0.035}"
              REGION_TARGET_TRANSPORT_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_TRANSPORT:-0.70}"
              REGION_TARGET_OUTSIDE_LOCK_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_OUTSIDE_LOCK:-0.90}"
              EDIT_INITIAL_NOISE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_NOISE:-0.05}"
              EDIT_INITIAL_NOISE_REGION="${SUPPORT_V3_ADD_EDITOR_FACE_NOISE_REGION:-core}"
              EDIT_LOCAL_TARGET_PROMPT="${SUPPORT_V3_ADD_EDITOR_FACE_LOCAL_TARGET_PROMPT:-${SUPPORT_LOCAL_TARGET_PROMPT:-A close-up photo of ${SUPPORT_NEW_TOKENS} on ${SUPPORT_HOST_TOKENS}.}}"
              EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_LOCAL_TARGET:-0.32}"
              EDIT_LOCAL_TARGET_CFG_SCALE="${SUPPORT_V3_ADD_EDITOR_FACE_LOCAL_CFG:-8.0}"
            else
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_FACE_HEDIT:-0.65}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_FACE_TEXT:-0.10}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_FACE_ANCHOR:-0.03}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_FACE_REGION:-0.03}"
            fi
            ;;
          add_object:*)
            if [[ "${OPERATION_EDIT_FIELD}" == add_editor_v* ]]; then
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_HEDIT:-0.72}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_TEXT:-0.14}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_ANCHOR:-0.03}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_REGION:-0.05}"
              REC_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_REC:-0.12}"
              TRAJECTORY_PRESERVE_SCALE="${SUPPORT_V3_ADD_EDITOR_TRAJ:-0.05}"
              REGION_TARGET_TRANSPORT_SCALE="${SUPPORT_V3_ADD_EDITOR_TRANSPORT:-0.95}"
              REGION_TARGET_OUTSIDE_LOCK_SCALE="${SUPPORT_V3_ADD_EDITOR_OUTSIDE_LOCK:-0.85}"
              EDIT_INITIAL_NOISE_SCALE="${SUPPORT_V3_ADD_EDITOR_NOISE:-0.10}"
              if [[ "${OPERATION_EDIT_FIELD}" == "add_editor_v2" ]]; then
                case "${SUPPORT_V3_RELATION}" in
                  inside|inside_host)
                    SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MIN_AREA_RATIO:-0.035}"
                    SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MAX_AREA_RATIO:-0.12}"
                    EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_LOCAL_TARGET:-0.38}"
                    ;;
                  on_surface|surface)
                    SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MIN_AREA_RATIO:-0.030}"
                    SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MAX_AREA_RATIO:-0.100}"
                    REGION_TARGET_TRANSPORT_SCALE="${SUPPORT_V3_ADD_EDITOR_TRANSPORT:-0.72}"
                    REGION_TARGET_OUTSIDE_LOCK_SCALE="${SUPPORT_V3_ADD_EDITOR_OUTSIDE_LOCK:-0.90}"
                    EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_LOCAL_TARGET:-0.44}"
                    ;;
                  above_host|above|on_top|top)
                    SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MIN_AREA_RATIO:-0.010}"
                    SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MAX_AREA_RATIO:-0.045}"
                    REGION_TARGET_TRANSPORT_SCALE="${SUPPORT_V3_ADD_EDITOR_TRANSPORT:-0.70}"
                    REGION_TARGET_OUTSIDE_LOCK_SCALE="${SUPPORT_V3_ADD_EDITOR_OUTSIDE_LOCK:-0.92}"
                    EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_LOCAL_TARGET:-0.50}"
                    ;;
                  *)
                    SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MIN_AREA_RATIO:-0.035}"
                    SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MAX_AREA_RATIO:-0.12}"
                    EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_LOCAL_TARGET:-0.38}"
                    ;;
                esac
              else
                SUPPORT_MIN_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MIN_AREA_RATIO:-0.035}"
                SUPPORT_MAX_AREA_RATIO="${SUPPORT_V3_ADD_EDITOR_MAX_AREA_RATIO:-0.12}"
                EDIT_LOCAL_TARGET_GUIDANCE_SCALE="${SUPPORT_V3_ADD_EDITOR_LOCAL_TARGET:-0.38}"
              fi
              EDIT_INITIAL_NOISE_REGION="${SUPPORT_V3_ADD_EDITOR_NOISE_REGION:-core}"
              EDIT_LOCAL_TARGET_PROMPT="${SUPPORT_V3_ADD_EDITOR_LOCAL_TARGET_PROMPT:-${SUPPORT_LOCAL_TARGET_PROMPT:-A close-up photo of ${SUPPORT_NEW_TOKENS} on ${SUPPORT_HOST_TOKENS}.}}"
              EDIT_LOCAL_TARGET_CFG_SCALE="${SUPPORT_V3_ADD_EDITOR_LOCAL_CFG:-8.0}"
            else
              EDIT_HEDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_OBJECT_HEDIT:-0.65}"
              EDIT_TEXT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_OBJECT_TEXT:-0.10}"
              EDIT_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_OBJECT_ANCHOR:-0.03}"
              EDIT_REGION_GUIDANCE_SCALE="${SUPPORT_V3_OPFIELD_OBJECT_REGION:-0.03}"
            fi
            ;;
        esac
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
