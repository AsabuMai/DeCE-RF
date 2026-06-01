from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import PIL.Image
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import set_seed

from edit_preprocess import (
    build_proposal_diff_mask,
    clamp_normalized_box,
    estimate_dark_eye_structure_boxes,
    estimate_foreground_head_structure_boxes,
    expand_normalized_box,
    parse_normalized_box,
    parse_word_list,
    save_structure_glasses_mask,
)
from sd3_hrec import HRecSD3Edit


def _pil_to_unit_tensor(image: PIL.Image.Image) -> torch.Tensor:
    data = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return data.unsqueeze(0)


def _mask_to_unit_tensor(mask: PIL.Image.Image) -> torch.Tensor:
    data = torch.frombuffer(bytearray(mask.convert("L").tobytes()), dtype=torch.uint8)
    data = data.view(mask.height, mask.width).float() / 255.0
    return data.view(1, 1, mask.height, mask.width)


def _unit_tensor_to_pil(image: torch.Tensor) -> PIL.Image.Image:
    array = (
        image[0]
        .detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return PIL.Image.fromarray(array)


def _rgb_to_yuv_tensor(image_rgb: torch.Tensor) -> torch.Tensor:
    r = image_rgb[:, 0:1]
    g = image_rgb[:, 1:2]
    b = image_rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492111 * (b - y)
    v = 0.877283 * (r - y)
    return torch.cat([y, u, v], dim=1)


def _yuv_to_rgb_tensor(image_yuv: torch.Tensor) -> torch.Tensor:
    y = image_yuv[:, 0:1]
    u = image_yuv[:, 1:2]
    v = image_yuv[:, 2:3]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat([r, g, b], dim=1)


def _low_pass_tensor(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return tensor
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)


def apply_y_highpass_texture_restore(
    result_image: PIL.Image.Image,
    source_image: PIL.Image.Image,
    mask_image: PIL.Image.Image,
    strength: float,
    kernel_size: int,
) -> PIL.Image.Image:
    result_rgb = _pil_to_unit_tensor(result_image)
    source_rgb = _pil_to_unit_tensor(source_image.resize(result_image.size, PIL.Image.Resampling.BILINEAR))
    mask = _mask_to_unit_tensor(mask_image.resize(result_image.size, PIL.Image.Resampling.BILINEAR)).clamp(0.0, 1.0)
    result_yuv = _rgb_to_yuv_tensor(result_rgb)
    source_yuv = _rgb_to_yuv_tensor(source_rgb)
    source_y_high = source_yuv[:, :1] - _low_pass_tensor(source_yuv[:, :1], kernel_size)
    restored_y = (result_yuv[:, :1] + float(strength) * source_y_high).clamp(0.0, 1.0)
    restored_rgb = _yuv_to_rgb_tensor(torch.cat([restored_y, result_yuv[:, 1:]], dim=1)).clamp(0.0, 1.0)
    return _unit_tensor_to_pil((result_rgb * (1.0 - mask) + restored_rgb * mask).clamp(0.0, 1.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone SD3 h-Edit prototype.")
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--source-prompt", required=True, help="Source prompt.")
    parser.add_argument("--prompt", required=True, help="Target prompt.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=512,
        help="Resize the source image so its longest side is at most this size before VAE encoding.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Prefer sequential CPU offload over model CPU offload for small-memory GPUs.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument(
        "--src-guidance-scale",
        type=float,
        default=1.0,
        help="Legacy source guidance default used by inversion/base unless the split source CFG args are set.",
    )
    parser.add_argument("--tar-guidance-scale", type=float, default=10.5)
    parser.add_argument(
        "--inversion-guidance-scale",
        type=float,
        default=None,
        help="Optional source CFG used only for source inversion. Defaults to --src-guidance-scale.",
    )
    parser.add_argument(
        "--base-guidance-scale",
        type=float,
        default=None,
        help="Optional source CFG used only for the reverse ODE base prior. Defaults to --src-guidance-scale.",
    )
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--rec-guidance-scale", type=float, default=0.0)
    parser.add_argument("--struct-guidance-scale", type=float, default=0.5)
    parser.add_argument("--edit-hedit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-src-cfg-scale", type=float, default=None)
    parser.add_argument("--edit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-region-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-source-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-clip-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-clip-match-base-scale", type=float, default=0.0)
    parser.add_argument("--edit-image-tv-scale", type=float, default=0.0)
    parser.add_argument("--edit-text-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-text-source-scale", type=float, default=0.8)
    parser.add_argument("--edit-text-core-weight", type=float, default=1.0)
    parser.add_argument("--edit-text-subject-weight", type=float, default=0.3)
    parser.add_argument(
        "--edit-text-source-prompt",
        type=str,
        default=None,
        help="Optional local source/old-object phrase for CLIP text edit guidance.",
    )
    parser.add_argument(
        "--edit-text-target-prompt",
        type=str,
        default=None,
        help="Optional local target/new-object phrase for CLIP text edit guidance.",
    )
    parser.add_argument(
        "--edit-local-target-prompt",
        type=str,
        default=None,
        help="Optional local target prompt used to build a target-formation RF clean prior.",
    )
    parser.add_argument("--edit-local-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-local-target-cfg-scale", type=float, default=None)
    parser.add_argument("--identity-break-stop", type=float, default=0.55)
    parser.add_argument("--target-attract-start", type=float, default=0.25)
    parser.add_argument("--edit-dds-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-dds-source-scale", type=float, default=0.8)
    parser.add_argument("--edit-app-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-source", type=str, default=None)
    parser.add_argument("--edit-color-target", type=str, default=None)
    parser.add_argument("--edit-color-mask-image", type=str, default=None)
    parser.add_argument("--edit-color-mask-threshold", type=float, default=0.38)
    parser.add_argument("--edit-color-mask-softness", type=float, default=0.10)
    parser.add_argument("--edit-color-luma-gate-min", type=float, default=0.0)
    parser.add_argument("--edit-color-luma-gate-softness", type=float, default=0.08)
    parser.add_argument("--edit-color-detail-protect-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-detail-protect-threshold", type=float, default=0.35)
    parser.add_argument("--edit-color-detail-protect-softness", type=float, default=0.08)
    parser.add_argument("--edit-color-alpha-matte", action="store_true")
    parser.add_argument("--edit-color-alpha-matte-mode", choices=("color", "closed_form"), default="color")
    parser.add_argument("--edit-color-alpha-matte-kernel-size", type=int, default=9)
    parser.add_argument("--edit-color-alpha-matte-threshold", type=float, default=None)
    parser.add_argument("--edit-color-alpha-matte-softness", type=float, default=None)
    parser.add_argument("--edit-color-alpha-matte-max-size", type=int, default=256)
    parser.add_argument("--edit-color-alpha-matte-epsilon", type=float, default=1e-7)
    parser.add_argument("--edit-color-alpha-matte-constraint-scale", type=float, default=100.0)
    parser.add_argument("--edit-color-target-chroma-scale", type=float, default=1.0)
    parser.add_argument("--edit-color-smooth-kernel", type=int, default=5)
    parser.add_argument("--edit-color-luma-preserve-scale", type=float, default=0.35)
    parser.add_argument("--edit-color-luma-gradient-preserve-scale", type=float, default=0.15)
    parser.add_argument("--edit-color-texture-preserve-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-texture-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-boundary-chroma-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-boundary-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-clean-projection-scale", type=float, default=0.0)
    parser.add_argument(
        "--edit-color-clean-projection-mode",
        choices=("soft", "strict", "yuv_texture"),
        default="soft",
    )
    parser.add_argument("--edit-color-clean-projection-texture-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-clean-projection-luma-texture-scale", type=float, default=1.0)
    parser.add_argument("--edit-color-clean-projection-chroma-texture-scale", type=float, default=0.25)
    parser.add_argument("--edit-color-clean-projection-delta-lowpass-kernel", type=int, default=0)
    parser.add_argument("--edit-color-clean-projection-alpha-power", type=float, default=1.0)
    parser.add_argument("--edit-color-clean-projection-boundary-boost", type=float, default=0.0)
    parser.add_argument("--edit-color-clean-projection-boundary-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-clean-projection-composite-mode", choices=("blend", "matte"), default="blend")
    parser.add_argument("--edit-color-clean-projection-background-kernel-size", type=int, default=31)
    parser.add_argument("--edit-color-clean-projection-target-mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--edit-color-clean-projection-refresh-interval", type=int, default=0)
    parser.add_argument("--edit-color-texture-restore", action="store_true")
    parser.add_argument("--edit-color-texture-restore-mask", type=str, default=None)
    parser.add_argument("--edit-color-texture-restore-strength", type=float, default=0.8)
    parser.add_argument("--edit-color-texture-restore-kernel-size", type=int, default=9)
    parser.add_argument("--edit-ref-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-ref-image", type=str, default=None)
    parser.add_argument("--edit-ref-mask", type=str, default=None)
    parser.add_argument("--edit-ref-structure-image", type=str, default=None)
    parser.add_argument("--edit-ref-chroma-mode", choices=("yuv", "yuv_direction"), default="yuv")
    parser.add_argument("--edit-ref-chroma-magnitude-scale", type=float, default=1.0)
    parser.add_argument("--edit-ref-luma-preserve-scale", type=float, default=0.35)
    parser.add_argument("--edit-ref-gradient-preserve-scale", type=float, default=0.15)
    parser.add_argument("--edit-ref-darkness-guard-scale", type=float, default=0.0)
    parser.add_argument("--edit-ref-darkness-guard-margin", type=float, default=0.03)
    parser.add_argument("--edit-ref-smooth-kernel", type=int, default=1)
    parser.add_argument("--edit-ref-lowfreq-suppress-kernel", type=int, default=0)
    parser.add_argument("--edit-ref-lowfreq-suppress-scale", type=float, default=0.0)
    parser.add_argument("--edit-ref-schedule-start", type=float, default=0.0)
    parser.add_argument("--edit-ref-schedule-stop", type=float, default=0.0)
    parser.add_argument("--edit-ref-schedule-power", type=float, default=1.0)
    parser.add_argument("--edit-ref-max-struct-rms-ratio", type=float, default=0.0)
    parser.add_argument("--edit-ref-project-struct-conflict", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-scale", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-image", type=str, default=None)
    parser.add_argument("--completion-clean-delta-mask", type=str, default=None)
    parser.add_argument("--completion-clean-delta-schedule-start", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-stop", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-power", type=float, default=1.0)
    parser.add_argument("--edit-core-scale", type=float, default=1.35)
    parser.add_argument("--edit-subject-scale", type=float, default=0.35)
    parser.add_argument(
        "--source-inject-q-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental FireFlow-style source visual-Q add strength used only "
            "when evaluating the target velocity."
        ),
    )
    parser.add_argument(
        "--source-inject-k-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental FireFlow-style source visual-K add strength used only "
            "when evaluating the target velocity."
        ),
    )
    parser.add_argument(
        "--source-inject-v-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental FireFlow/RF-Solver-style source visual-V injection "
            "strength used only when evaluating the target velocity."
        ),
    )
    parser.add_argument(
        "--source-inject-layer-from",
        type=int,
        default=-1,
        help="First SD3 transformer block for source V injection. -1 uses the last third.",
    )
    parser.add_argument(
        "--source-inject-layer-to",
        type=int,
        default=-1,
        help="Exclusive SD3 transformer block end for source V injection. -1 uses all later blocks.",
    )
    parser.add_argument(
        "--source-inject-steps",
        type=int,
        default=8,
        help="Number of early reverse-ODE steps that use source V injection. <=0 applies to all active steps.",
    )
    parser.add_argument(
        "--source-inject-mask-mode",
        type=str,
        choices=("none", "edit", "core", "preserve", "box"),
        default="none",
        help="Optional spatial gate for source Q/K/V injection.",
    )
    parser.add_argument(
        "--source-inject-mask-box",
        type=str,
        default=None,
        help="Optional normalized source-injection gate box as x0,y0,x1,y1.",
    )
    parser.add_argument("--edit-bound-scale", type=float, default=0.0)
    parser.add_argument("--clip-start-timestep", type=float, default=0.0)
    parser.add_argument("--clip-stop-timestep", type=float, default=0.6)
    parser.add_argument("--preserve-blend-scale", type=float, default=0.0)
    parser.add_argument("--preserve-blend-start-timestep", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--alpha-schedule", type=str, default="constant")
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--beta-schedule", type=str, default="constant")
    parser.add_argument(
        "--adaptive-clean-control",
        action="store_true",
        default=False,
        help="Enable per-step clean-estimate diagnostics and closed-loop local guidance scaling.",
    )
    parser.add_argument(
        "--adaptive-edit-target-progress",
        type=float,
        default=0.0,
        help=(
            "Target directional edit progress in clean-estimate space. "
            "When >0, this supersedes --adaptive-edit-target-rms for edit boost."
        ),
    )
    parser.add_argument("--adaptive-edit-target-rms", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-rmsgap-mode",
        type=str,
        default="legacy",
        choices=["legacy", "normgate"],
    )
    parser.add_argument("--adaptive-rmsgap-dead-zone", type=float, default=0.0)
    parser.add_argument("--adaptive-rmsgap-preserve-gate-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-progress-target", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-progress-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-progress-ema-decay", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-preserve-gate-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-drift-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-edit-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-edit-weight-min", type=float, default=0.7)
    parser.add_argument("--adaptive-edit-weight-max", type=float, default=1.8)
    parser.add_argument("--adaptive-preserve-weight-min", type=float, default=1.0)
    parser.add_argument("--adaptive-preserve-weight-max", type=float, default=2.0)
    parser.add_argument("--adaptive-projection-scale", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-clean-correction-scale", type=float, default=0.0)
    parser.add_argument(
        "--removal-controller-mode",
        choices=("none", "clean_fill"),
        default="none",
        help="Removal-specific local clean-fill controller. Disabled unless edit operation is remove_object.",
    )
    parser.add_argument("--removal-fill-scale", type=float, default=0.0)
    parser.add_argument("--removal-suppression-scale", type=float, default=0.0)
    parser.add_argument("--removal-ring-rec-scale", type=float, default=0.0)
    parser.add_argument(
        "--velocity-conversion-mode",
        type=str,
        choices=("legacy", "linear_path"),
        default="linear_path",
        help="How clean-space displacement surrogates are converted into velocity corrections.",
    )
    parser.add_argument(
        "--linear-path-t-min",
        type=float,
        default=0.05,
        help="Minimum denominator for linear-path clean-delta-to-velocity conversion in image runs.",
    )
    parser.add_argument(
        "--rec-stop-timestep",
        type=float,
        default=0.08,
        help="Disable reconstruction guidance below this normalized timestep to avoid low-t blow-up.",
    )
    parser.add_argument(
        "--trajectory-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "Blend preserve regions toward the saved source inversion trajectory "
            "after each reverse ODE step. 0 disables trajectory anchoring."
        ),
    )
    parser.add_argument(
        "--trajectory-subject-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "Additionally blend the attention subject ring (subject minus core) "
            "toward the source trajectory. This helps local edits keep identity "
            "while leaving the high-confidence core editable."
        ),
    )
    parser.add_argument(
        "--edit-initial-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Blend the initial inverted latent toward random noise inside the "
            "attention edit region. Useful for local insertions that need more "
            "generation freedom than source inversion provides."
        ),
    )
    parser.add_argument(
        "--edit-initial-noise-region",
        type=str,
        choices=("core", "subject"),
        default="core",
        help="Which attention mask region receives --edit-initial-noise-scale.",
    )
    parser.add_argument(
        "--region-target-transport-scale",
        type=float,
        default=0.0,
        help=(
            "Enable region-conditioned target transport. Values >0 add a scheduled "
            "target-flow takeover in the edit core and a weaker target blend in the ring."
        ),
    )
    parser.add_argument(
        "--region-target-outside-lock-scale",
        type=float,
        default=0.0,
        help=(
            "After each RF step, blend outside/ring latents toward the saved source "
            "trajectory so stronger core target transport does not drift the background."
        ),
    )
    parser.add_argument(
        "--attention-mask-mode",
        type=str,
        choices=("changed_union", "target_changed", "subject_union", "source_subject", "target_subject"),
        default="changed_union",
        help="Which attention map is converted into the editable mask.",
    )
    parser.add_argument(
        "--attention-mask-target-words",
        type=str,
        default=None,
        help=(
            "Optional comma-separated target prompt words used for token attention. "
            "Defaults to the changed words inferred from source/target prompts."
        ),
    )
    parser.add_argument(
        "--attention-mask-source-words",
        type=str,
        default=None,
        help=(
            "Optional comma-separated source prompt words used for token attention. "
            "Defaults to the changed words inferred from source/target prompts."
        ),
    )
    parser.add_argument(
        "--attention-mask-subject-threshold",
        type=float,
        default=0.48,
        help="Threshold on the normalized attention map for the editable subject mask.",
    )
    parser.add_argument(
        "--attention-mask-core-threshold",
        type=float,
        default=0.72,
        help="Threshold on the normalized attention map for the high-confidence edit core.",
    )
    parser.add_argument(
        "--attention-mask-max-area-ratio",
        type=float,
        default=0.25,
        help=(
            "If the final attention edit mask covers more than this binary area ratio, "
            "shrink it to a conservative high-response target-token box. <=0 disables."
        ),
    )
    parser.add_argument(
        "--attention-mask-fallback-threshold",
        type=float,
        default=0.72,
        help="Normalized high-response threshold used by the large-mask fallback box.",
    )
    parser.add_argument(
        "--object-mask-provider",
        choices=(
            "attention",
            "velocity_diff",
            "attention_velocity",
            "semantic",
            "semantic_velocity",
            "generic_support",
            "operation_support_v3",
            "structure",
            "proposal_diff",
            "auto",
        ),
        default="velocity_diff",
        help=(
            "Source used for the object edit support. 'attention' uses target-token attention "
            "and is image-agnostic; 'velocity_diff' uses this model's target-vs-source RF velocity "
            "difference; 'attention_velocity' fuses changed-token attention with RF velocity difference "
            "for the generic no-oracle path; 'semantic' uses --semantic-base-mask; "
            "'semantic_velocity' refines --semantic-base-mask with RF velocity difference; "
            "'generic_support' uses changed-token attention times clean/velocity disagreement; "
            "'operation_support_v3' uses operation-aware relation/surface/segmentation candidates; "
            "'structure' keeps the optional external structure mask path; "
            "'proposal_diff' extracts changed support from a proposal image; "
            "'auto' uses proposal_diff when --proposal-edit-image is set, otherwise attention_velocity."
        ),
    )
    parser.add_argument(
        "--support-mode",
        choices=("generic", "operation_v3"),
        default=None,
        help="Convenience selector. operation_v3 maps --object-mask-provider to operation_support_v3.",
    )
    parser.add_argument(
        "--semantic-base-mask",
        "--support-mask",
        dest="semantic_base_mask",
        type=str,
        default=None,
        help=(
            "Optional support mask consumed only by object providers semantic and semantic_velocity. "
            "--support-mask is the preferred spelling; --semantic-base-mask is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--support-score",
        choices=(
            "attention_only",
            "clean_disagreement_only",
            "velocity_disagreement_only",
            "attention_x_clean",
            "attention_x_velocity",
            "host_x_clean",
            "new_x_host_x_clean",
            "new_plus_host_x_clean",
            "removed_src_x_clean",
            "removed_src_x_velocity",
            "src_tar_attn_x_clean",
            "seg_x_clean",
            "seg_x_velocity",
            "seg_x_response",
            "relation_x_clean",
            "relation_x_velocity",
            "relation_x_response",
            "host_surface_x_clean",
            "host_surface_x_response",
            "new_x_surface_x_clean",
            "surface_local_clean",
            "surface_local_response",
            "decal_surface_local_response",
            "new_x_surface_local_response",
            "spawn_center",
            "spawn_center_x_response",
            "new_x_spawn_center",
            "spawn_lower_center",
            "spawn_lower_center_x_response",
            "host_spawn_center",
            "host_spawn_center_x_response",
            "host_spawn_wide",
            "host_spawn_wide_x_response",
            "host_top_contact",
            "host_top_contact_x_response",
            "auto",
            "operation_default",
            "score_auto",
        ),
        default="attention_x_clean",
        help="Score/candidate used by generic_support or operation_support_v3.",
    )
    parser.add_argument(
        "--support-candidate",
        choices=(
            "attention_only",
            "clean_disagreement_only",
            "velocity_disagreement_only",
            "attention_x_clean",
            "attention_x_velocity",
            "host_x_clean",
            "new_x_host_x_clean",
            "new_plus_host_x_clean",
            "removed_src_x_clean",
            "removed_src_x_velocity",
            "src_tar_attn_x_clean",
            "seg_x_clean",
            "seg_x_velocity",
            "seg_x_response",
            "relation_x_clean",
            "relation_x_velocity",
            "relation_x_response",
            "host_surface_x_clean",
            "host_surface_x_response",
            "new_x_surface_x_clean",
            "surface_local_clean",
            "surface_local_response",
            "decal_surface_local_response",
            "new_x_surface_local_response",
            "spawn_center",
            "spawn_center_x_response",
            "new_x_spawn_center",
            "spawn_lower_center",
            "spawn_lower_center_x_response",
            "host_spawn_center",
            "host_spawn_center_x_response",
            "host_spawn_wide",
            "host_spawn_wide_x_response",
            "host_top_contact",
            "host_top_contact_x_response",
            "auto",
            "operation_default",
            "score_auto",
        ),
        default=None,
        help="Operation-aware support candidate. Overrides --support-score when set.",
    )
    parser.add_argument(
        "--edit-operation",
        choices=("auto", "add_object", "add_decal", "remove_object", "replace", "recolor"),
        default="auto",
        help="Operation label recorded for operation-aware generic support.",
    )
    parser.add_argument("--new-tokens", type=str, default=None)
    parser.add_argument("--host-tokens", type=str, default=None)
    parser.add_argument("--removed-tokens", type=str, default=None)
    parser.add_argument(
        "--relation",
        "--support-relation",
        dest="support_relation",
        choices=("auto", "none", "above_host", "on_face", "on_surface", "remove_source_object", "inside_host", "inside"),
        default="auto",
        help="Operation-level relation/layout proposal used by operation_support_v3.",
    )
    parser.add_argument(
        "--grounding-method",
        choices=("none", "external_mask", "grounded_sam", "sam", "sam2", "clipseg"),
        default="external_mask",
        help=(
            "Grounding source label for operation_support_v3. Current runtime consumes "
            "--support-mask as the external grounding mask and records this label."
        ),
    )
    parser.add_argument(
        "--save-support-debug",
        action="store_true",
        default=False,
        help="Save operation_support_v3 candidate maps when --mask-output-dir is set.",
    )
    parser.add_argument(
        "--support-debug-only",
        action="store_true",
        default=False,
        help="Stop after automatic support estimation/debug output instead of running the edit ODE.",
    )
    parser.add_argument(
        "--support-temporal-aggregation",
        choices=("single", "mean", "max"),
        default="single",
        help="Aggregate operation_support_v3 clean/velocity evidence over one or multiple ODE timesteps.",
    )
    parser.add_argument("--support-attention-power", type=float, default=1.0)
    parser.add_argument("--support-disagreement-power", type=float, default=1.0)
    parser.add_argument("--support-top-percentile", type=float, default=90.0)
    parser.add_argument("--support-min-area-ratio", type=float, default=0.02)
    parser.add_argument("--support-max-area-ratio", type=float, default=0.30)
    parser.add_argument("--support-keep-components", type=int, default=2)
    parser.add_argument("--support-dilate-radius", type=int, default=5)
    parser.add_argument("--support-blur-kernel", type=int, default=5)
    parser.add_argument("--attention-velocity-support-pad-x", type=float, default=0.015)
    parser.add_argument("--attention-velocity-support-pad-y", type=float, default=0.010)
    parser.add_argument("--attention-velocity-support-min-width", type=float, default=0.18)
    parser.add_argument("--attention-velocity-support-min-height", type=float, default=0.065)
    parser.add_argument(
        "--mask-layering-mode",
        choices=("none", "object_contact"),
        default="object_contact",
        help="Split edit support into object/contact/preserve layers before editing.",
    )
    parser.add_argument(
        "--mask-object-threshold",
        type=float,
        default=0.45,
        help="Threshold used to extract the strong object-edit layer from the edit mask.",
    )
    parser.add_argument(
        "--mask-contact-dilate-kernel",
        type=int,
        default=7,
        help="Dilation kernel used to build the weak contact ring around the object mask.",
    )
    parser.add_argument(
        "--mask-contact-scale",
        type=float,
        default=0.25,
        help="Edit strength multiplier inside the contact ring.",
    )
    parser.add_argument(
        "--mask-contact-edge-threshold",
        type=float,
        default=0.55,
        help="Latent edge threshold for protecting strong source structure inside the contact ring.",
    )
    parser.add_argument(
        "--mask-contact-edge-protect-scale",
        type=float,
        default=0.75,
        help="How strongly source latent edges suppress the contact edit ring.",
    )
    parser.add_argument(
        "--mask-output-dir",
        type=str,
        default=None,
        help="Optional directory for saving attention-mask debug PNGs.",
    )
    parser.add_argument(
        "--edit-mask-dilate-kernel",
        type=int,
        default=0,
        help="Optionally dilate the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-erode-kernel",
        type=int,
        default=0,
        help="Optionally erode the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-smooth-kernel",
        type=int,
        default=0,
        help="Optionally smooth the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-hole-fraction",
        type=float,
        default=0.0,
        help="Drop a random fraction of edit/core mask pixels for support robustness tests.",
    )
    parser.add_argument(
        "--edit-mask-boundary-noise-scale",
        type=float,
        default=0.0,
        help="Add random perturbation on edit/core mask boundaries for support robustness tests.",
    )
    parser.add_argument(
        "--edit-mask-component-threshold",
        type=float,
        default=0.0,
        help="Threshold used by optional attention-mask connected-component filtering. 0 uses 0.5.",
    )
    parser.add_argument(
        "--edit-mask-keep-components",
        type=int,
        default=0,
        help="Keep only the top-N attention-mask connected components by mask mass. 0 keeps all.",
    )
    parser.add_argument(
        "--edit-mask-component-y-min",
        type=float,
        default=None,
        help="Optional minimum normalized component center y for attention-mask component filtering.",
    )
    parser.add_argument(
        "--edit-mask-component-y-max",
        type=float,
        default=None,
        help="Optional maximum normalized component center y for attention-mask component filtering.",
    )
    parser.add_argument(
        "--edit-mask-shift-y",
        type=float,
        default=0.0,
        help="Translate the attention-derived edit/core masks vertically as a fraction of latent height.",
    )
    parser.add_argument(
        "--edit-mask-shift-x",
        type=float,
        default=0.0,
        help="Translate the attention-derived edit/core masks horizontally as a fraction of latent width.",
    )
    parser.add_argument(
        "--auto-local-boxes",
        action="store_true",
        default=False,
        help=(
            "Derive local edit/source-injection/preserve boxes from the attention mask. "
            "Manual box arguments still override their corresponding auto boxes."
        ),
    )
    parser.add_argument(
        "--auto-box-threshold",
        type=float,
        default=0.35,
        help="Attention-mask threshold used to derive --auto-local-boxes anchors.",
    )
    parser.add_argument("--auto-edit-pad-x", type=float, default=0.08)
    parser.add_argument("--auto-edit-pad-y", type=float, default=0.04)
    parser.add_argument("--auto-edit-min-width", type=float, default=0.28)
    parser.add_argument("--auto-edit-min-height", type=float, default=0.10)
    parser.add_argument("--auto-source-pad-x", type=float, default=0.14)
    parser.add_argument("--auto-source-pad-y", type=float, default=0.08)
    parser.add_argument("--auto-preserve-pad-x", type=float, default=0.04)
    parser.add_argument("--auto-preserve-start-offset", type=float, default=0.06)
    parser.add_argument("--auto-preserve-height", type=float, default=0.24)
    parser.add_argument(
        "--auto-structure-boxes",
        action="store_true",
        default=False,
        help=(
            "Estimate local accessory boxes from source-image structure before SD3 runs. "
            "Currently uses a no-model dark-component heuristic for eye-like regions."
        ),
    )
    parser.add_argument(
        "--auto-structure-mode",
        choices=("dark_eyes",),
        default="dark_eyes",
        help="Image-structure heuristic used by --auto-structure-boxes.",
    )
    parser.add_argument(
        "--auto-structure-external-mask",
        action="store_true",
        default=False,
        help="Use an automatically generated glasses-shaped structure mask as M_edit.",
    )
    parser.add_argument(
        "--structure-glasses-angle-mode",
        choices=("auto", "zero"),
        default="auto",
        help="Use detected eye-line angle for generated glasses masks, or force horizontal.",
    )
    parser.add_argument(
        "--structure-glasses-max-angle",
        type=float,
        default=8.0,
        help="Maximum absolute rotation angle in degrees for generated glasses masks.",
    )
    parser.add_argument(
        "--edit-mask-box",
        type=str,
        default=None,
        help="Optional normalized edit box as x0,y0,x1,y1. Useful for local accessory placement.",
    )
    parser.add_argument(
        "--edit-mask-box-mode",
        type=str,
        choices=("replace", "intersect", "union"),
        default="replace",
        help="How --edit-mask-box combines with the attention-derived edit mask.",
    )
    parser.add_argument(
        "--edit-mask-exclude-box",
        type=str,
        default=None,
        help="Optional normalized box subtracted from the final edit/core masks.",
    )
    parser.add_argument(
        "--edit-mask-use-core-as-subject",
        action="store_true",
        default=False,
        help=(
            "Use the high-confidence core mask as the final M_edit support. "
            "Useful for diagnosing local edits where the subject mask is too broad."
        ),
    )
    parser.add_argument(
        "--external-edit-mask",
        "--final-edit-mask",
        dest="external_edit_mask",
        type=str,
        default=None,
        help=(
            "Optional grayscale image applied late to the final M_edit support. "
            "--final-edit-mask is the preferred spelling; --external-edit-mask is kept for compatibility. "
            "This changes only the mask support; the RF/ODE edit dynamics are unchanged."
        ),
    )
    parser.add_argument(
        "--external-edit-mask-mode",
        "--final-edit-mask-mode",
        dest="external_edit_mask_mode",
        type=str,
        choices=("replace", "intersect", "union"),
        default="replace",
        help="How --final-edit-mask combines with the current edit mask.",
    )
    parser.add_argument(
        "--proposal-edit-image",
        type=str,
        default=None,
        help="Optional candidate edited image used by --object-mask-provider proposal_diff.",
    )
    parser.add_argument("--proposal-mask-threshold", type=float, default=0.22)
    parser.add_argument("--proposal-mask-keep-components", type=int, default=2)
    parser.add_argument("--proposal-mask-min-area", type=int, default=24)
    parser.add_argument("--proposal-mask-dilate", type=int, default=9)
    parser.add_argument("--proposal-mask-erode", type=int, default=0)
    parser.add_argument("--proposal-mask-blur", type=int, default=17)
    parser.add_argument("--proposal-mask-dark-bias", type=float, default=1.0)
    parser.add_argument(
        "--edit-field-mode",
        type=str,
        choices=("surrogate", "rf_diff", "text_diff", "rf_text_diff"),
        default="surrogate",
        help=(
            "Edit correction field used in the controlled ODE. "
            "'rf_diff' uses the target-vs-source RF velocity difference; "
            "'text_diff' uses the source-target CLIP text differential reward; "
            "'rf_text_diff' combines both. 'surrogate' keeps all explicitly "
            "enabled legacy surrogate branches for ablations."
        ),
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--clean-estimate-debug-dir", type=str, default=None)
    parser.add_argument("--stats-output", type=str, default=None)
    parser.add_argument("--metadata-output", type=str, default=None)
    parser.add_argument("--mask-blend", action="store_true", default=False)
    parser.add_argument(
        "--mask-blend-mode",
        choices=("subject", "core"),
        default="subject",
        help="Which final edit support is used when --mask-blend is enabled.",
    )
    parser.add_argument(
        "--final-preserve-box",
        type=str,
        default=None,
        help="Optional normalized box pasted back from the source latent after editing.",
    )
    parser.add_argument(
        "--photo-prompt-mode",
        type=str,
        choices=("off", "source", "both"),
        default="source",
        help=(
            "Optionally rewrite prompts into an explicit photorealistic style "
            "without changing the editing math. 'source' only stabilizes the "
            "base prior; 'both' applies the same style prior to source and target."
        ),
    )
    return parser


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def make_photorealistic_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    style_cues = (
        "photo",
        "photograph",
        "photorealistic",
        "realistic",
        "natural lighting",
        "high-resolution",
    )
    if any(cue in lowered for cue in style_cues):
        return prompt
    return (
        f"A realistic photograph of {prompt.rstrip('.').lower()}, "
        "natural lighting, high-resolution photo."
    )


def load_image_latent(
    pipe,
    image_path: str,
    device: torch.device,
    max_image_size: int,
) -> torch.Tensor:
    image = PIL.Image.open(image_path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        resized_w = max(16, int(round(image.width * scale)))
        resized_h = max(16, int(round(image.height * scale)))
        image = image.resize((resized_w, resized_h), PIL.Image.Resampling.LANCZOS)
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_src = pipe.image_processor.preprocess(image).to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    return (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def main() -> None:
    run_started_at = time.perf_counter()
    args = build_parser().parse_args()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    edit_mask_box = parse_normalized_box(args.edit_mask_box, "--edit-mask-box")
    edit_mask_exclude_box = parse_normalized_box(args.edit_mask_exclude_box, "--edit-mask-exclude-box")
    source_inject_mask_box = parse_normalized_box(args.source_inject_mask_box, "--source-inject-mask-box")
    final_preserve_box = parse_normalized_box(args.final_preserve_box, "--final-preserve-box")
    external_edit_mask_path = args.external_edit_mask
    external_edit_mask_mode = args.external_edit_mask_mode
    structure_box_metadata: dict[str, object] = {"structure_found": False}
    proposal_mask_metadata: dict[str, object] = {"proposal_diff_found": False}
    if args.auto_structure_boxes:
        structure_boxes, structure_box_metadata = estimate_dark_eye_structure_boxes(
            args.image,
            max_image_size=args.max_image_size,
        )
        if structure_boxes["edit"] is not None and edit_mask_box is None:
            edit_mask_box = structure_boxes["edit"]
            args.auto_local_boxes = False
        if structure_boxes["source_inject"] is not None and source_inject_mask_box is None:
            source_inject_mask_box = structure_boxes["source_inject"]
        if structure_boxes["preserve"] is not None:
            if edit_mask_exclude_box is None:
                edit_mask_exclude_box = structure_boxes["preserve"]
            if final_preserve_box is None:
                final_preserve_box = structure_boxes["preserve"]
        if (
            args.auto_structure_external_mask
            and structure_boxes["edit"] is not None
            and external_edit_mask_path is None
        ):
            if args.mask_output_dir is None:
                raise ValueError("--auto-structure-external-mask requires --mask-output-dir")
            external_edit_mask_path = save_structure_glasses_mask(
                args.image,
                args.max_image_size,
                structure_boxes["edit"],
                os.path.join(args.mask_output_dir, "structure_glasses_edit_mask.png"),
                angle_deg=(
                    0.0
                    if args.structure_glasses_angle_mode == "zero"
                    else max(
                        -abs(args.structure_glasses_max_angle),
                        min(
                            abs(args.structure_glasses_max_angle),
                            float(structure_box_metadata.get("structure_eye_line_angle_deg", 0.0)),
                        ),
                    )
                ),
            )
            external_edit_mask_mode = "replace"
            structure_box_metadata["structure_external_edit_mask"] = external_edit_mask_path
        print(f"[structure] {json.dumps(structure_box_metadata, sort_keys=True)}")
    resolved_object_mask_provider = args.object_mask_provider
    if args.support_mode == "operation_v3":
        resolved_object_mask_provider = "operation_support_v3"
    elif args.support_mode == "generic" and resolved_object_mask_provider == "operation_support_v3":
        resolved_object_mask_provider = "generic_support"
    if resolved_object_mask_provider == "auto":
        resolved_object_mask_provider = "proposal_diff" if args.proposal_edit_image else "attention_velocity"
    if resolved_object_mask_provider == "proposal_diff":
        if not args.proposal_edit_image:
            raise ValueError("--object-mask-provider proposal_diff requires --proposal-edit-image")
        if external_edit_mask_path is None:
            if args.mask_output_dir is None:
                raise ValueError("--object-mask-provider proposal_diff requires --mask-output-dir")
            external_edit_mask_path, proposal_mask_metadata = build_proposal_diff_mask(
                source_image_path=args.image,
                proposal_image_path=args.proposal_edit_image,
                max_image_size=args.max_image_size,
                output_path=os.path.join(args.mask_output_dir, "proposal_diff_edit_mask.png"),
                threshold=args.proposal_mask_threshold,
                keep_components=args.proposal_mask_keep_components,
                min_area=args.proposal_mask_min_area,
                dilate=args.proposal_mask_dilate,
                erode=args.proposal_mask_erode,
                blur=args.proposal_mask_blur,
                dark_bias=args.proposal_mask_dark_bias,
            )
            external_edit_mask_mode = "replace"
            proposal_mask_metadata["proposal_diff_found"] = True
        print(f"[proposal_diff] {json.dumps(proposal_mask_metadata, sort_keys=True)}")
    attention_mask_target_words = parse_word_list(args.attention_mask_target_words)
    attention_mask_source_words = parse_word_list(args.attention_mask_source_words)
    set_seed(args.seed)

    source_prompt = args.source_prompt
    target_prompt = args.prompt
    if args.photo_prompt_mode in {"source", "both"}:
        source_prompt = make_photorealistic_prompt(source_prompt)
    if args.photo_prompt_mode == "both":
        target_prompt = make_photorealistic_prompt(target_prompt)
    if source_prompt != args.source_prompt or target_prompt != args.prompt:
        print(f"[prompt] source: {source_prompt}")
        print(f"[prompt] target: {target_prompt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    scheduler = pipe.scheduler
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if args.low_vram and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    x_src = load_image_latent(
        pipe=pipe,
        image_path=args.image,
        device=device,
        max_image_size=args.max_image_size,
    ).to(device)
    x_tar = HRecSD3Edit(
        pipe=pipe,
        scheduler=scheduler,
        x_src=x_src,
        src_prompt=source_prompt,
        tar_prompt=target_prompt,
        negative_prompt="",
        T_steps=args.num_inference_steps,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        inversion_guidance_scale=args.inversion_guidance_scale,
        base_guidance_scale=args.base_guidance_scale,
        n_max=args.n_max,
        eta=args.eta,
        rec_guidance_scale=args.rec_guidance_scale,
        struct_guidance_scale=args.struct_guidance_scale,
        edit_hedit_guidance_scale=args.edit_hedit_guidance_scale,
        edit_field_mode=args.edit_field_mode,
        edit_src_cfg_scale=args.edit_src_cfg_scale,
        edit_guidance_scale=args.edit_guidance_scale,
        edit_region_guidance_scale=args.edit_region_guidance_scale,
        edit_target_guidance_scale=args.edit_target_guidance_scale,
        edit_source_guidance_scale=args.edit_source_guidance_scale,
        edit_clip_guidance_scale=args.edit_clip_guidance_scale,
        edit_clip_match_base_scale=args.edit_clip_match_base_scale,
        edit_image_tv_scale=args.edit_image_tv_scale,
        edit_text_guidance_scale=args.edit_text_guidance_scale,
        edit_text_source_scale=args.edit_text_source_scale,
        edit_text_core_weight=args.edit_text_core_weight,
        edit_text_subject_weight=args.edit_text_subject_weight,
        edit_text_source_prompt=args.edit_text_source_prompt or None,
        edit_text_target_prompt=args.edit_text_target_prompt or None,
        edit_local_target_prompt=args.edit_local_target_prompt or None,
        edit_local_target_guidance_scale=args.edit_local_target_guidance_scale,
        edit_local_target_cfg_scale=args.edit_local_target_cfg_scale,
        identity_break_stop=args.identity_break_stop,
        target_attract_start=args.target_attract_start,
        edit_dds_guidance_scale=args.edit_dds_guidance_scale,
        edit_dds_source_scale=args.edit_dds_source_scale,
        edit_app_guidance_scale=args.edit_app_guidance_scale,
        edit_color_guidance_scale=args.edit_color_guidance_scale,
        edit_color_source=args.edit_color_source,
        edit_color_target=args.edit_color_target,
        edit_color_mask_path=args.edit_color_mask_image or None,
        edit_color_mask_threshold=args.edit_color_mask_threshold,
        edit_color_mask_softness=args.edit_color_mask_softness,
        edit_color_luma_gate_min=args.edit_color_luma_gate_min,
        edit_color_luma_gate_softness=args.edit_color_luma_gate_softness,
        edit_color_detail_protect_scale=args.edit_color_detail_protect_scale,
        edit_color_detail_protect_threshold=args.edit_color_detail_protect_threshold,
        edit_color_detail_protect_softness=args.edit_color_detail_protect_softness,
        edit_color_alpha_matte=args.edit_color_alpha_matte,
        edit_color_alpha_matte_mode=args.edit_color_alpha_matte_mode,
        edit_color_alpha_matte_kernel_size=args.edit_color_alpha_matte_kernel_size,
        edit_color_alpha_matte_threshold=args.edit_color_alpha_matte_threshold,
        edit_color_alpha_matte_softness=args.edit_color_alpha_matte_softness,
        edit_color_alpha_matte_max_size=args.edit_color_alpha_matte_max_size,
        edit_color_alpha_matte_epsilon=args.edit_color_alpha_matte_epsilon,
        edit_color_alpha_matte_constraint_scale=args.edit_color_alpha_matte_constraint_scale,
        edit_color_target_chroma_scale=args.edit_color_target_chroma_scale,
        edit_color_smooth_kernel=args.edit_color_smooth_kernel,
        edit_color_luma_preserve_scale=args.edit_color_luma_preserve_scale,
        edit_color_luma_gradient_preserve_scale=args.edit_color_luma_gradient_preserve_scale,
        edit_color_texture_preserve_scale=args.edit_color_texture_preserve_scale,
        edit_color_texture_kernel_size=args.edit_color_texture_kernel_size,
        edit_color_boundary_chroma_scale=args.edit_color_boundary_chroma_scale,
        edit_color_boundary_kernel_size=args.edit_color_boundary_kernel_size,
        edit_color_clean_projection_scale=args.edit_color_clean_projection_scale,
        edit_color_clean_projection_mode=args.edit_color_clean_projection_mode,
        edit_color_clean_projection_texture_kernel_size=args.edit_color_clean_projection_texture_kernel_size,
        edit_color_clean_projection_luma_texture_scale=args.edit_color_clean_projection_luma_texture_scale,
        edit_color_clean_projection_chroma_texture_scale=args.edit_color_clean_projection_chroma_texture_scale,
        edit_color_clean_projection_delta_lowpass_kernel=args.edit_color_clean_projection_delta_lowpass_kernel,
        edit_color_clean_projection_alpha_power=args.edit_color_clean_projection_alpha_power,
        edit_color_clean_projection_boundary_boost=args.edit_color_clean_projection_boundary_boost,
        edit_color_clean_projection_boundary_kernel_size=args.edit_color_clean_projection_boundary_kernel_size,
        edit_color_clean_projection_composite_mode=args.edit_color_clean_projection_composite_mode,
        edit_color_clean_projection_background_kernel_size=args.edit_color_clean_projection_background_kernel_size,
        edit_color_clean_projection_target_mode=args.edit_color_clean_projection_target_mode,
        edit_color_clean_projection_refresh_interval=args.edit_color_clean_projection_refresh_interval,
        edit_ref_guidance_scale=args.edit_ref_guidance_scale,
        edit_ref_image_path=args.edit_ref_image or None,
        edit_ref_mask_path=args.edit_ref_mask or None,
        edit_ref_structure_image_path=args.edit_ref_structure_image or None,
        edit_ref_chroma_mode=args.edit_ref_chroma_mode,
        edit_ref_chroma_magnitude_scale=args.edit_ref_chroma_magnitude_scale,
        edit_ref_luma_preserve_scale=args.edit_ref_luma_preserve_scale,
        edit_ref_gradient_preserve_scale=args.edit_ref_gradient_preserve_scale,
        edit_ref_darkness_guard_scale=args.edit_ref_darkness_guard_scale,
        edit_ref_darkness_guard_margin=args.edit_ref_darkness_guard_margin,
        edit_ref_smooth_kernel=args.edit_ref_smooth_kernel,
        edit_ref_lowfreq_suppress_kernel=args.edit_ref_lowfreq_suppress_kernel,
        edit_ref_lowfreq_suppress_scale=args.edit_ref_lowfreq_suppress_scale,
        edit_ref_schedule_start=args.edit_ref_schedule_start,
        edit_ref_schedule_stop=args.edit_ref_schedule_stop,
        edit_ref_schedule_power=args.edit_ref_schedule_power,
        edit_ref_max_struct_rms_ratio=args.edit_ref_max_struct_rms_ratio,
        edit_ref_project_struct_conflict=args.edit_ref_project_struct_conflict,
        completion_clean_delta_scale=args.completion_clean_delta_scale,
        completion_clean_delta_image_path=args.completion_clean_delta_image or None,
        completion_clean_delta_mask_path=args.completion_clean_delta_mask or None,
        completion_clean_delta_schedule_start=args.completion_clean_delta_schedule_start,
        completion_clean_delta_schedule_stop=args.completion_clean_delta_schedule_stop,
        completion_clean_delta_schedule_power=args.completion_clean_delta_schedule_power,
        edit_core_scale=args.edit_core_scale,
        edit_subject_scale=args.edit_subject_scale,
        source_inject_q_scale=args.source_inject_q_scale,
        source_inject_k_scale=args.source_inject_k_scale,
        source_inject_v_scale=args.source_inject_v_scale,
        source_inject_layer_from=args.source_inject_layer_from,
        source_inject_layer_to=args.source_inject_layer_to,
        source_inject_steps=args.source_inject_steps,
        source_inject_mask_mode=args.source_inject_mask_mode,
        source_inject_mask_box=source_inject_mask_box,
        edit_bound_scale=args.edit_bound_scale,
        clip_start_timestep=args.clip_start_timestep,
        clip_stop_timestep=args.clip_stop_timestep,
        preserve_blend_scale=args.preserve_blend_scale,
        preserve_blend_start_timestep=args.preserve_blend_start_timestep,
        alpha_max=args.alpha_max,
        alpha_schedule=args.alpha_schedule,
        beta_max=args.beta_max,
        beta_schedule=args.beta_schedule,
        adaptive_clean_control=args.adaptive_clean_control,
        adaptive_edit_target_progress=args.adaptive_edit_target_progress,
        adaptive_edit_target_rms=args.adaptive_edit_target_rms,
        adaptive_rmsgap_mode=args.adaptive_rmsgap_mode,
        adaptive_rmsgap_dead_zone=args.adaptive_rmsgap_dead_zone,
        adaptive_rmsgap_preserve_gate_budget=args.adaptive_rmsgap_preserve_gate_budget,
        adaptive_hybrid_progress_target=args.adaptive_hybrid_progress_target,
        adaptive_hybrid_progress_gain=args.adaptive_hybrid_progress_gain,
        adaptive_hybrid_progress_ema_decay=args.adaptive_hybrid_progress_ema_decay,
        adaptive_hybrid_preserve_gate_budget=args.adaptive_hybrid_preserve_gate_budget,
        adaptive_preserve_drift_budget=args.adaptive_preserve_drift_budget,
        adaptive_edit_gain=args.adaptive_edit_gain,
        adaptive_preserve_gain=args.adaptive_preserve_gain,
        adaptive_edit_weight_min=args.adaptive_edit_weight_min,
        adaptive_edit_weight_max=args.adaptive_edit_weight_max,
        adaptive_preserve_weight_min=args.adaptive_preserve_weight_min,
        adaptive_preserve_weight_max=args.adaptive_preserve_weight_max,
        adaptive_projection_scale=args.adaptive_projection_scale,
        adaptive_preserve_clean_correction_scale=args.adaptive_preserve_clean_correction_scale,
        removal_controller_mode=args.removal_controller_mode,
        removal_fill_scale=args.removal_fill_scale,
        removal_suppression_scale=args.removal_suppression_scale,
        removal_ring_rec_scale=args.removal_ring_rec_scale,
        velocity_conversion_mode=args.velocity_conversion_mode,
        linear_path_t_min=args.linear_path_t_min,
        rec_stop_timestep=args.rec_stop_timestep,
        trajectory_preserve_scale=args.trajectory_preserve_scale,
        trajectory_subject_preserve_scale=args.trajectory_subject_preserve_scale,
        edit_initial_noise_scale=args.edit_initial_noise_scale,
        edit_initial_noise_region=args.edit_initial_noise_region,
        region_target_transport_scale=args.region_target_transport_scale,
        region_target_outside_lock_scale=args.region_target_outside_lock_scale,
        attention_mask_mode=args.attention_mask_mode,
        attention_mask_target_words=attention_mask_target_words,
        attention_mask_source_words=attention_mask_source_words,
        attention_mask_subject_threshold=args.attention_mask_subject_threshold,
        attention_mask_core_threshold=args.attention_mask_core_threshold,
        attention_mask_max_area_ratio=args.attention_mask_max_area_ratio,
        attention_mask_fallback_threshold=args.attention_mask_fallback_threshold,
        object_mask_provider=resolved_object_mask_provider,
        semantic_base_mask_path=args.semantic_base_mask,
        support_score=args.support_candidate or args.support_score,
        support_edit_operation=args.edit_operation,
        support_relation=args.support_relation,
        support_grounding_method=args.grounding_method,
        save_support_debug_maps=args.save_support_debug,
        support_debug_only=args.support_debug_only,
        support_temporal_aggregation=args.support_temporal_aggregation,
        support_new_tokens=parse_word_list(args.new_tokens),
        support_host_tokens=parse_word_list(args.host_tokens),
        support_removed_tokens=parse_word_list(args.removed_tokens),
        support_attention_power=args.support_attention_power,
        support_disagreement_power=args.support_disagreement_power,
        support_top_percentile=args.support_top_percentile,
        support_min_area_ratio=args.support_min_area_ratio,
        support_max_area_ratio=args.support_max_area_ratio,
        support_keep_components=args.support_keep_components,
        support_dilate_radius=args.support_dilate_radius,
        support_blur_kernel=args.support_blur_kernel,
        attention_velocity_support_pad_x=args.attention_velocity_support_pad_x,
        attention_velocity_support_pad_y=args.attention_velocity_support_pad_y,
        attention_velocity_support_min_width=args.attention_velocity_support_min_width,
        attention_velocity_support_min_height=args.attention_velocity_support_min_height,
        mask_layering_mode=args.mask_layering_mode,
        mask_object_threshold=args.mask_object_threshold,
        mask_contact_dilate_kernel=args.mask_contact_dilate_kernel,
        mask_contact_scale=args.mask_contact_scale,
        mask_contact_edge_threshold=args.mask_contact_edge_threshold,
        mask_contact_edge_protect_scale=args.mask_contact_edge_protect_scale,
        mask_output_dir=args.mask_output_dir,
        edit_mask_dilate_kernel=args.edit_mask_dilate_kernel,
        edit_mask_erode_kernel=args.edit_mask_erode_kernel,
        edit_mask_smooth_kernel=args.edit_mask_smooth_kernel,
        edit_mask_hole_fraction=args.edit_mask_hole_fraction,
        edit_mask_boundary_noise_scale=args.edit_mask_boundary_noise_scale,
        edit_mask_component_threshold=args.edit_mask_component_threshold,
        edit_mask_keep_components=args.edit_mask_keep_components,
        edit_mask_component_y_min=args.edit_mask_component_y_min,
        edit_mask_component_y_max=args.edit_mask_component_y_max,
        edit_mask_shift_y=args.edit_mask_shift_y,
        edit_mask_shift_x=args.edit_mask_shift_x,
        auto_local_boxes=args.auto_local_boxes,
        auto_box_threshold=args.auto_box_threshold,
        auto_edit_pad_x=args.auto_edit_pad_x,
        auto_edit_pad_y=args.auto_edit_pad_y,
        auto_edit_min_width=args.auto_edit_min_width,
        auto_edit_min_height=args.auto_edit_min_height,
        auto_source_pad_x=args.auto_source_pad_x,
        auto_source_pad_y=args.auto_source_pad_y,
        auto_preserve_pad_x=args.auto_preserve_pad_x,
        auto_preserve_start_offset=args.auto_preserve_start_offset,
        auto_preserve_height=args.auto_preserve_height,
        edit_mask_box=edit_mask_box,
        edit_mask_box_mode=args.edit_mask_box_mode,
        edit_mask_exclude_box=edit_mask_exclude_box,
        edit_mask_use_core_as_subject=args.edit_mask_use_core_as_subject,
        external_edit_mask_path=external_edit_mask_path,
        external_edit_mask_mode=external_edit_mask_mode,
        log_every=args.log_every,
        stats_output_path=args.stats_output,
        clean_estimate_debug_dir=args.clean_estimate_debug_dir,
        mask_blend=args.mask_blend,
        mask_blend_mode=args.mask_blend_mode,
        final_preserve_box=final_preserve_box,
    )

    x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    texture_restore_mask_path = None
    if args.edit_color_texture_restore:
        candidate_mask_paths = []
        if args.edit_color_texture_restore_mask:
            candidate_mask_paths.append(args.edit_color_texture_restore_mask)
        if args.mask_output_dir:
            candidate_mask_paths.extend(
                [
                    os.path.join(args.mask_output_dir, "color_edit_alpha_matte.png"),
                    os.path.join(args.mask_output_dir, "color_edit_mask.png"),
                    os.path.join(args.mask_output_dir, "surface_refined_mask.png"),
                ]
            )
        if args.edit_color_mask_image:
            candidate_mask_paths.append(args.edit_color_mask_image)
        if external_edit_mask_path:
            candidate_mask_paths.append(external_edit_mask_path)
        for candidate in candidate_mask_paths:
            if candidate and os.path.exists(candidate):
                texture_restore_mask_path = candidate
                break
        if texture_restore_mask_path is None:
            raise ValueError("--edit-color-texture-restore requires a color/edit mask")
        image_tar[0] = apply_y_highpass_texture_restore(
            result_image=image_tar[0],
            source_image=PIL.Image.open(args.image).convert("RGB"),
            mask_image=PIL.Image.open(texture_restore_mask_path).convert("L"),
            strength=args.edit_color_texture_restore_strength,
            kernel_size=args.edit_color_texture_restore_kernel_size,
        )
    image_tar[0].save(args.output)
    metadata_output = args.metadata_output
    if metadata_output is None:
        root, _ = os.path.splitext(args.output)
        metadata_output = f"{root}_metadata.json"
    metadata = {
        "image": args.image,
        "source_prompt": args.source_prompt,
        "effective_source_prompt": source_prompt,
        "target_prompt": args.prompt,
        "effective_target_prompt": target_prompt,
        "output": args.output,
        "stats_output": args.stats_output,
        "clean_estimate_debug_dir": args.clean_estimate_debug_dir,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "n_max": args.n_max,
        "src_guidance_scale": args.src_guidance_scale,
        "tar_guidance_scale": args.tar_guidance_scale,
        "inversion_guidance_scale": (
            args.inversion_guidance_scale
            if args.inversion_guidance_scale is not None
            else args.src_guidance_scale
        ),
        "base_guidance_scale": (
            args.base_guidance_scale
            if args.base_guidance_scale is not None
            else args.src_guidance_scale
        ),
        "edit_src_cfg_scale": (
            args.edit_src_cfg_scale
            if args.edit_src_cfg_scale is not None
            else (
                args.base_guidance_scale
                if args.base_guidance_scale is not None
                else args.src_guidance_scale
            )
        ),
        "rec_guidance_scale": args.rec_guidance_scale,
        "struct_guidance_scale": args.struct_guidance_scale,
        "edit_hedit_guidance_scale": args.edit_hedit_guidance_scale,
        "edit_field_mode": args.edit_field_mode,
        "edit_guidance_scale": args.edit_guidance_scale,
        "edit_region_guidance_scale": args.edit_region_guidance_scale,
        "edit_target_guidance_scale": args.edit_target_guidance_scale,
        "edit_source_guidance_scale": args.edit_source_guidance_scale,
        "edit_clip_guidance_scale": args.edit_clip_guidance_scale,
        "edit_text_guidance_scale": args.edit_text_guidance_scale,
        "edit_text_source_scale": args.edit_text_source_scale,
        "edit_text_core_weight": args.edit_text_core_weight,
        "edit_text_subject_weight": args.edit_text_subject_weight,
        "edit_text_source_prompt": args.edit_text_source_prompt or None,
        "edit_text_target_prompt": args.edit_text_target_prompt or None,
        "edit_local_target_prompt": args.edit_local_target_prompt or None,
        "edit_local_target_guidance_scale": args.edit_local_target_guidance_scale,
        "edit_local_target_cfg_scale": args.edit_local_target_cfg_scale,
        "edit_dds_guidance_scale": args.edit_dds_guidance_scale,
        "edit_app_guidance_scale": args.edit_app_guidance_scale,
        "edit_color_guidance_scale": args.edit_color_guidance_scale,
        "edit_color_source": args.edit_color_source,
        "edit_color_target": args.edit_color_target,
        "edit_color_mask_image": args.edit_color_mask_image or None,
        "edit_color_mask_threshold": args.edit_color_mask_threshold,
        "edit_color_mask_softness": args.edit_color_mask_softness,
        "edit_color_luma_gate_min": args.edit_color_luma_gate_min,
        "edit_color_luma_gate_softness": args.edit_color_luma_gate_softness,
        "edit_color_detail_protect_scale": args.edit_color_detail_protect_scale,
        "edit_color_detail_protect_threshold": args.edit_color_detail_protect_threshold,
        "edit_color_detail_protect_softness": args.edit_color_detail_protect_softness,
        "edit_color_alpha_matte": args.edit_color_alpha_matte,
        "edit_color_alpha_matte_mode": args.edit_color_alpha_matte_mode,
        "edit_color_alpha_matte_kernel_size": args.edit_color_alpha_matte_kernel_size,
        "edit_color_alpha_matte_threshold": args.edit_color_alpha_matte_threshold,
        "edit_color_alpha_matte_softness": args.edit_color_alpha_matte_softness,
        "edit_color_alpha_matte_max_size": args.edit_color_alpha_matte_max_size,
        "edit_color_alpha_matte_epsilon": args.edit_color_alpha_matte_epsilon,
        "edit_color_alpha_matte_constraint_scale": args.edit_color_alpha_matte_constraint_scale,
        "edit_color_target_chroma_scale": args.edit_color_target_chroma_scale,
        "edit_color_smooth_kernel": args.edit_color_smooth_kernel,
        "edit_color_luma_preserve_scale": args.edit_color_luma_preserve_scale,
        "edit_color_luma_gradient_preserve_scale": args.edit_color_luma_gradient_preserve_scale,
        "edit_color_texture_preserve_scale": args.edit_color_texture_preserve_scale,
        "edit_color_texture_kernel_size": args.edit_color_texture_kernel_size,
        "edit_color_boundary_chroma_scale": args.edit_color_boundary_chroma_scale,
        "edit_color_boundary_kernel_size": args.edit_color_boundary_kernel_size,
        "edit_color_clean_projection_scale": args.edit_color_clean_projection_scale,
        "edit_color_clean_projection_mode": args.edit_color_clean_projection_mode,
        "edit_color_clean_projection_texture_kernel_size": args.edit_color_clean_projection_texture_kernel_size,
        "edit_color_clean_projection_luma_texture_scale": args.edit_color_clean_projection_luma_texture_scale,
        "edit_color_clean_projection_chroma_texture_scale": args.edit_color_clean_projection_chroma_texture_scale,
        "edit_color_clean_projection_delta_lowpass_kernel": args.edit_color_clean_projection_delta_lowpass_kernel,
        "edit_color_clean_projection_alpha_power": args.edit_color_clean_projection_alpha_power,
        "edit_color_clean_projection_boundary_boost": args.edit_color_clean_projection_boundary_boost,
        "edit_color_clean_projection_boundary_kernel_size": args.edit_color_clean_projection_boundary_kernel_size,
        "edit_color_clean_projection_composite_mode": args.edit_color_clean_projection_composite_mode,
        "edit_color_clean_projection_background_kernel_size": args.edit_color_clean_projection_background_kernel_size,
        "edit_color_clean_projection_target_mode": args.edit_color_clean_projection_target_mode,
        "edit_color_clean_projection_refresh_interval": args.edit_color_clean_projection_refresh_interval,
        "edit_color_texture_restore": args.edit_color_texture_restore,
        "edit_color_texture_restore_mask": texture_restore_mask_path,
        "edit_color_texture_restore_strength": args.edit_color_texture_restore_strength,
        "edit_color_texture_restore_kernel_size": args.edit_color_texture_restore_kernel_size,
        "edit_ref_guidance_scale": args.edit_ref_guidance_scale,
        "edit_ref_image": args.edit_ref_image or None,
        "edit_ref_mask": args.edit_ref_mask or None,
        "edit_ref_structure_image": args.edit_ref_structure_image or None,
        "edit_ref_chroma_mode": args.edit_ref_chroma_mode,
        "edit_ref_chroma_magnitude_scale": args.edit_ref_chroma_magnitude_scale,
        "edit_ref_luma_preserve_scale": args.edit_ref_luma_preserve_scale,
        "edit_ref_gradient_preserve_scale": args.edit_ref_gradient_preserve_scale,
        "edit_ref_darkness_guard_scale": args.edit_ref_darkness_guard_scale,
        "edit_ref_darkness_guard_margin": args.edit_ref_darkness_guard_margin,
        "edit_ref_smooth_kernel": args.edit_ref_smooth_kernel,
        "edit_ref_lowfreq_suppress_kernel": args.edit_ref_lowfreq_suppress_kernel,
        "edit_ref_lowfreq_suppress_scale": args.edit_ref_lowfreq_suppress_scale,
        "edit_ref_schedule_start": args.edit_ref_schedule_start,
        "edit_ref_schedule_stop": args.edit_ref_schedule_stop,
        "edit_ref_schedule_power": args.edit_ref_schedule_power,
        "edit_ref_max_struct_rms_ratio": args.edit_ref_max_struct_rms_ratio,
        "edit_ref_project_struct_conflict": args.edit_ref_project_struct_conflict,
        "completion_clean_delta_scale": args.completion_clean_delta_scale,
        "completion_clean_delta_image": args.completion_clean_delta_image or None,
        "completion_clean_delta_mask": args.completion_clean_delta_mask or None,
        "completion_clean_delta_schedule_start": args.completion_clean_delta_schedule_start,
        "completion_clean_delta_schedule_stop": args.completion_clean_delta_schedule_stop,
        "completion_clean_delta_schedule_power": args.completion_clean_delta_schedule_power,
        "source_inject_q_scale": args.source_inject_q_scale,
        "source_inject_k_scale": args.source_inject_k_scale,
        "source_inject_v_scale": args.source_inject_v_scale,
        "source_inject_layer_from": args.source_inject_layer_from,
        "source_inject_layer_to": args.source_inject_layer_to,
        "source_inject_steps": args.source_inject_steps,
        "source_inject_mask_mode": args.source_inject_mask_mode,
        "source_inject_mask_box": source_inject_mask_box,
        "alpha_max": args.alpha_max if args.alpha_max is not None else args.rec_guidance_scale,
        "alpha_schedule": args.alpha_schedule,
        "beta_max": args.beta_max if args.beta_max is not None else 1.0,
        "beta_schedule": args.beta_schedule,
        "adaptive_clean_control": args.adaptive_clean_control,
        "adaptive_edit_target_progress": args.adaptive_edit_target_progress,
        "adaptive_edit_target_rms": args.adaptive_edit_target_rms,
        "adaptive_rmsgap_mode": args.adaptive_rmsgap_mode,
        "adaptive_rmsgap_dead_zone": args.adaptive_rmsgap_dead_zone,
        "adaptive_rmsgap_preserve_gate_budget": args.adaptive_rmsgap_preserve_gate_budget,
        "adaptive_hybrid_progress_target": args.adaptive_hybrid_progress_target,
        "adaptive_hybrid_progress_gain": args.adaptive_hybrid_progress_gain,
        "adaptive_hybrid_progress_ema_decay": args.adaptive_hybrid_progress_ema_decay,
        "adaptive_hybrid_preserve_gate_budget": args.adaptive_hybrid_preserve_gate_budget,
        "adaptive_preserve_drift_budget": args.adaptive_preserve_drift_budget,
        "adaptive_edit_gain": args.adaptive_edit_gain,
        "adaptive_preserve_gain": args.adaptive_preserve_gain,
        "adaptive_edit_weight_min": args.adaptive_edit_weight_min,
        "adaptive_edit_weight_max": args.adaptive_edit_weight_max,
        "adaptive_preserve_weight_min": args.adaptive_preserve_weight_min,
        "adaptive_preserve_weight_max": args.adaptive_preserve_weight_max,
        "adaptive_projection_scale": args.adaptive_projection_scale,
        "adaptive_preserve_clean_correction_scale": args.adaptive_preserve_clean_correction_scale,
        "removal_controller_mode": args.removal_controller_mode,
        "removal_fill_scale": args.removal_fill_scale,
        "removal_suppression_scale": args.removal_suppression_scale,
        "removal_ring_rec_scale": args.removal_ring_rec_scale,
        "velocity_conversion_mode": args.velocity_conversion_mode,
        "linear_path_t_min": args.linear_path_t_min,
        "rec_stop_timestep": args.rec_stop_timestep,
        "trajectory_preserve_scale": args.trajectory_preserve_scale,
        "trajectory_subject_preserve_scale": args.trajectory_subject_preserve_scale,
        "edit_initial_noise_scale": args.edit_initial_noise_scale,
        "edit_initial_noise_region": args.edit_initial_noise_region,
        "region_target_transport_scale": args.region_target_transport_scale,
        "region_target_outside_lock_scale": args.region_target_outside_lock_scale,
        "attention_mask_mode": args.attention_mask_mode,
        "attention_mask_target_words": attention_mask_target_words,
        "attention_mask_source_words": attention_mask_source_words,
        "attention_mask_subject_threshold": args.attention_mask_subject_threshold,
        "attention_mask_core_threshold": args.attention_mask_core_threshold,
        "attention_mask_max_area_ratio": args.attention_mask_max_area_ratio,
        "attention_mask_fallback_threshold": args.attention_mask_fallback_threshold,
        "object_mask_provider": resolved_object_mask_provider,
        "requested_object_mask_provider": args.object_mask_provider,
        "semantic_base_mask": args.semantic_base_mask,
        "support_mode": args.support_mode,
        "support_score": args.support_score,
        "support_candidate": args.support_candidate or args.support_score,
        "edit_operation": args.edit_operation,
        "support_relation": args.support_relation,
        "grounding_method": args.grounding_method,
        "save_support_debug": args.save_support_debug,
        "support_debug_only": args.support_debug_only,
        "support_temporal_aggregation": args.support_temporal_aggregation,
        "new_tokens": parse_word_list(args.new_tokens),
        "host_tokens": parse_word_list(args.host_tokens),
        "removed_tokens": parse_word_list(args.removed_tokens),
        "support_attention_power": args.support_attention_power,
        "support_disagreement_power": args.support_disagreement_power,
        "support_top_percentile": args.support_top_percentile,
        "support_min_area_ratio": args.support_min_area_ratio,
        "support_max_area_ratio": args.support_max_area_ratio,
        "support_keep_components": args.support_keep_components,
        "support_dilate_radius": args.support_dilate_radius,
        "support_blur_kernel": args.support_blur_kernel,
        "attention_velocity_support_pad_x": args.attention_velocity_support_pad_x,
        "attention_velocity_support_pad_y": args.attention_velocity_support_pad_y,
        "attention_velocity_support_min_width": args.attention_velocity_support_min_width,
        "attention_velocity_support_min_height": args.attention_velocity_support_min_height,
        "mask_layering_mode": args.mask_layering_mode,
        "mask_object_threshold": args.mask_object_threshold,
        "mask_contact_dilate_kernel": args.mask_contact_dilate_kernel,
        "mask_contact_scale": args.mask_contact_scale,
        "mask_contact_edge_threshold": args.mask_contact_edge_threshold,
        "mask_contact_edge_protect_scale": args.mask_contact_edge_protect_scale,
        "mask_output_dir": args.mask_output_dir,
        "edit_mask_dilate_kernel": args.edit_mask_dilate_kernel,
        "edit_mask_erode_kernel": args.edit_mask_erode_kernel,
        "edit_mask_smooth_kernel": args.edit_mask_smooth_kernel,
        "edit_mask_hole_fraction": args.edit_mask_hole_fraction,
        "edit_mask_boundary_noise_scale": args.edit_mask_boundary_noise_scale,
        "edit_mask_component_threshold": args.edit_mask_component_threshold,
        "edit_mask_keep_components": args.edit_mask_keep_components,
        "edit_mask_component_y_min": args.edit_mask_component_y_min,
        "edit_mask_component_y_max": args.edit_mask_component_y_max,
        "edit_mask_shift_y": args.edit_mask_shift_y,
        "edit_mask_shift_x": args.edit_mask_shift_x,
        "auto_local_boxes": args.auto_local_boxes,
        "auto_box_threshold": args.auto_box_threshold,
        "auto_edit_pad_x": args.auto_edit_pad_x,
        "auto_edit_pad_y": args.auto_edit_pad_y,
        "auto_edit_min_width": args.auto_edit_min_width,
        "auto_edit_min_height": args.auto_edit_min_height,
        "auto_source_pad_x": args.auto_source_pad_x,
        "auto_source_pad_y": args.auto_source_pad_y,
        "auto_preserve_pad_x": args.auto_preserve_pad_x,
        "auto_preserve_start_offset": args.auto_preserve_start_offset,
        "auto_preserve_height": args.auto_preserve_height,
        "auto_structure_boxes": args.auto_structure_boxes,
        "auto_structure_mode": args.auto_structure_mode,
        "auto_structure_external_mask": args.auto_structure_external_mask,
        **structure_box_metadata,
        **proposal_mask_metadata,
        "proposal_edit_image": args.proposal_edit_image,
        "proposal_mask_threshold": args.proposal_mask_threshold,
        "proposal_mask_keep_components": args.proposal_mask_keep_components,
        "proposal_mask_min_area": args.proposal_mask_min_area,
        "proposal_mask_dilate": args.proposal_mask_dilate,
        "proposal_mask_erode": args.proposal_mask_erode,
        "proposal_mask_blur": args.proposal_mask_blur,
        "proposal_mask_dark_bias": args.proposal_mask_dark_bias,
        "edit_mask_box": edit_mask_box,
        "edit_mask_box_mode": args.edit_mask_box_mode,
        "edit_mask_exclude_box": edit_mask_exclude_box,
        "edit_mask_use_core_as_subject": args.edit_mask_use_core_as_subject,
        "external_edit_mask": external_edit_mask_path,
        "external_edit_mask_mode": external_edit_mask_mode,
        "edit_field_mode": args.edit_field_mode,
        "mask_blend": args.mask_blend,
        "mask_blend_mode": args.mask_blend_mode,
        "final_preserve_box": final_preserve_box,
        "photo_prompt_mode": args.photo_prompt_mode,
        "git_commit": current_git_commit(),
        "runtime_seconds": time.perf_counter() - run_started_at,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else None
        ),
    }
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved result to {args.output}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
