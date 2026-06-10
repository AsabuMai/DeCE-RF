from __future__ import annotations

import copy
import json
import os
from typing import Optional

import PIL.Image

import torch
from tqdm import tqdm

from clip_text_reward import CLIPReferenceState, LocalCLIPTextReward
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from attention_mask import extract_attention_masks, extract_prompt_attention_map
from energies import (
    clean_delta_to_velocity,
    cosine_safe,
    editing_energy_total,
    editing_velocity_surrogate_total,
    reconstruction_energy_total,
    reconstruction_velocity_surrogate_total,
)
from generic_support import build_generic_support
from guidance_fields import (
    decode_latent_to_unit_image,
    infer_source_color_rgb,
    infer_target_color_rgb,
    limit_guidance_rms_relative_to_anchor,
    masked_chroma_luma_loss,
    masked_reference_image_loss,
    masked_recolor_texture_boundary_loss,
    masked_rms,
    masked_total_variation,
    ramp_schedule_from_progress,
    remove_opposing_guidance_component,
    rgb_to_yuv,
    smooth_guidance_field,
    source_color_similarity_mask,
    suppress_low_frequency_guidance,
)
from operation_support_v3 import (
    build_operation_support_v3,
    compute_clean_disagreement,
    compute_velocity_disagreement,
    save_support_debug,
)
from schedules import get_schedule_value
from spatial_masks import (
    _attention_object_mask_from_map,
    _attention_velocity_object_mask,
    _box_from_mask,
    _box_to_list,
    _clamp_box,
    _conservative_attention_box,
    _expand_box,
    _largest_component_box_from_mask,
    _largest_component_mask_from_mask,
    _mask_binary_area_ratio,
    _semantic_velocity_object_mask,
    _top_components_mask_from_mask,
    _velocity_diff_object_mask,
    build_object_contact_masks,
    build_recolor_trimap_masks,
    dilate_spatial_mask,
    filter_spatial_mask_components,
    latent_structure_edge_mask,
    load_external_image_like,
    load_external_mask_like,
    normalized_box_mask_like,
    save_mask_image,
    smooth_spatial_mask,
    spatial_mask_stats,
    translate_spatial_mask,
)




def _tensor_unit_image_to_pil(image: torch.Tensor) -> PIL.Image.Image:
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


def _encode_unit_image_to_latent(pipe, image: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    image_input = (image.to(dtype=vae_dtype) * 2.0 - 1.0).clamp(-1.0, 1.0)
    autocast_enabled = image_input.device.type == "cuda"
    with torch.autocast("cuda", enabled=autocast_enabled), torch.inference_mode():
        latent_denorm = vae.encode(image_input).latent_dist.mode()
    return (latent_denorm - vae.config.shift_factor) * vae.config.scaling_factor


def _yuv_to_rgb(image_yuv: torch.Tensor) -> torch.Tensor:
    y = image_yuv[:, 0:1]
    u = image_yuv[:, 1:2]
    v = image_yuv[:, 2:3]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat([r, g, b], dim=1)


def _spatial_low_pass(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return tensor
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)


def _luma_low_pass(luma: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return _spatial_low_pass(luma, kernel_size)


def _estimate_local_background(
    image: torch.Tensor,
    alpha: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    outside = (1.0 - alpha).clamp(0.0, 1.0)
    if kernel_size <= 1:
        return image
    pad = kernel_size // 2
    weighted = torch.nn.functional.avg_pool2d(image * outside, kernel_size=kernel_size, stride=1, padding=pad)
    denom = torch.nn.functional.avg_pool2d(outside, kernel_size=kernel_size, stride=1, padding=pad).clamp_min(1e-4)
    estimate = weighted / denom
    return torch.where(denom > 1e-3, estimate, image).clamp(0.0, 1.0)


def _soft_mask_boundary_band(alpha: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return torch.zeros_like(alpha)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    dilated = torch.nn.functional.max_pool2d(alpha, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -torch.nn.functional.max_pool2d(-alpha, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def _estimate_recolor_boundary_alpha(
    source_image: torch.Tensor,
    mask: torch.Tensor,
    source_rgb: torch.Tensor,
    threshold: float,
    softness: float,
    boundary_kernel_size: int,
) -> torch.Tensor:
    """Use source color evidence only in a narrow band around the support edge."""
    alpha = mask.to(dtype=torch.float32, device=source_image.device)
    if alpha.shape[-2:] != source_image.shape[-2:]:
        alpha = torch.nn.functional.interpolate(
            alpha,
            size=source_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    alpha = alpha.clamp(0.0, 1.0)
    boundary = _soft_mask_boundary_band(alpha, boundary_kernel_size)
    if float(boundary.max().item()) <= 0.0:
        return alpha
    color_alpha = source_color_similarity_mask(
        source_image.to(dtype=torch.float32),
        source_rgb.to(dtype=torch.float32, device=source_image.device),
        None,
        threshold=threshold,
        softness=softness,
    ).clamp(0.0, 1.0)
    return (alpha * (1.0 - boundary) + color_alpha * boundary).clamp(0.0, 1.0)


def _closed_form_alpha_crop(
    image: "np.ndarray",
    known_alpha: "np.ndarray",
    known_mask: "np.ndarray",
    epsilon: float,
    constraint_scale: float,
) -> "np.ndarray":
    import numpy as np
    from scipy import sparse
    from scipy.sparse import linalg

    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("closed-form matting expects RGB input")
    radius = 1
    win_size = (2 * radius + 1) ** 2
    flat_image = image.reshape(-1, 3).astype(np.float64)
    inds = np.arange(height * width).reshape(height, width)
    row_inds = []
    col_inds = []
    values = []
    eye = np.eye(3)
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            win_inds = inds[y - radius : y + radius + 1, x - radius : x + radius + 1].reshape(-1)
            win_i = flat_image[win_inds]
            mean = win_i.mean(axis=0)
            centered = win_i - mean
            cov = centered.T @ centered / win_size
            inv = np.linalg.inv(cov + (float(epsilon) / win_size) * eye)
            affinity = (1.0 + centered @ inv @ centered.T) / win_size
            local_l = np.eye(win_size) - affinity
            row_inds.extend(np.repeat(win_inds, win_size))
            col_inds.extend(np.tile(win_inds, win_size))
            values.extend(local_l.reshape(-1))
    n = height * width
    laplacian = sparse.coo_matrix((values, (row_inds, col_inds)), shape=(n, n)).tocsr()
    known = known_mask.reshape(-1).astype(bool)
    alpha0 = known_alpha.reshape(-1).astype(np.float64)
    constraints = sparse.diags(known.astype(np.float64) * float(constraint_scale), format="csr")
    rhs = known.astype(np.float64) * float(constraint_scale) * alpha0
    alpha = linalg.spsolve(laplacian + constraints, rhs)
    return np.clip(alpha.reshape(height, width), 0.0, 1.0).astype(np.float32)


def _estimate_recolor_closed_form_alpha(
    source_image: torch.Tensor,
    mask: torch.Tensor,
    boundary_kernel_size: int,
    max_size: int,
    epsilon: float,
    constraint_scale: float,
) -> torch.Tensor:
    import numpy as np

    alpha = mask.to(dtype=torch.float32, device=source_image.device)
    if alpha.shape[-2:] != source_image.shape[-2:]:
        alpha = torch.nn.functional.interpolate(
            alpha,
            size=source_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    alpha = alpha.clamp(0.0, 1.0)
    hard = (alpha > 0.5).to(dtype=torch.float32)
    kernel_size = max(3, int(boundary_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    eroded = -torch.nn.functional.max_pool2d(-hard, kernel_size=kernel_size, stride=1, padding=pad)
    dilated = torch.nn.functional.max_pool2d(hard, kernel_size=kernel_size, stride=1, padding=pad)
    unknown = (dilated - eroded).clamp(0.0, 1.0) > 0.0
    if not bool(unknown.any().item()):
        return alpha

    active = (dilated[0, 0] > 0.0).detach().cpu().numpy()
    ys, xs = np.where(active)
    if len(xs) == 0:
        return alpha
    margin = max(2, pad + 2)
    height, width = alpha.shape[-2:]
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(width, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(height, int(ys.max()) + margin + 1)

    image_crop = source_image[:, :, y0:y1, x0:x1].to(dtype=torch.float32)
    alpha_crop = alpha[:, :, y0:y1, x0:x1]
    eroded_crop = eroded[:, :, y0:y1, x0:x1]
    dilated_crop = dilated[:, :, y0:y1, x0:x1]
    crop_h, crop_w = alpha_crop.shape[-2:]
    scale = min(1.0, float(max(16, int(max_size))) / float(max(crop_h, crop_w)))
    solve_h = max(8, int(round(crop_h * scale)))
    solve_w = max(8, int(round(crop_w * scale)))
    if (solve_h, solve_w) != (crop_h, crop_w):
        image_solve = torch.nn.functional.interpolate(
            image_crop,
            size=(solve_h, solve_w),
            mode="bilinear",
            align_corners=False,
        )
        alpha_solve = torch.nn.functional.interpolate(
            alpha_crop,
            size=(solve_h, solve_w),
            mode="bilinear",
            align_corners=False,
        )
        eroded_solve = torch.nn.functional.interpolate(eroded_crop, size=(solve_h, solve_w), mode="nearest")
        dilated_solve = torch.nn.functional.interpolate(dilated_crop, size=(solve_h, solve_w), mode="nearest")
    else:
        image_solve = image_crop
        alpha_solve = alpha_crop
        eroded_solve = eroded_crop
        dilated_solve = dilated_crop

    known_fg = eroded_solve[0, 0] > 0.5
    known_bg = dilated_solve[0, 0] < 0.5
    known_mask = (known_fg | known_bg).detach().cpu().numpy()
    known_alpha = alpha_solve[0, 0].detach().cpu().numpy()
    known_alpha[known_fg.detach().cpu().numpy()] = 1.0
    known_alpha[known_bg.detach().cpu().numpy()] = 0.0
    if not known_mask.any() or known_mask.all():
        return alpha
    image_np = image_solve[0].detach().cpu().permute(1, 2, 0).numpy()
    alpha_np = _closed_form_alpha_crop(
        image_np,
        known_alpha,
        known_mask,
        epsilon=epsilon,
        constraint_scale=constraint_scale,
    )
    alpha_tensor = torch.from_numpy(alpha_np).to(dtype=torch.float32, device=source_image.device).view(1, 1, solve_h, solve_w)
    if (solve_h, solve_w) != (crop_h, crop_w):
        alpha_tensor = torch.nn.functional.interpolate(
            alpha_tensor,
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    unknown_crop = ((dilated_crop - eroded_crop).clamp(0.0, 1.0) > 0.0).to(dtype=torch.float32)
    refined_crop = alpha_crop * (1.0 - unknown_crop) + alpha_tensor * unknown_crop
    refined = alpha.clone()
    refined[:, :, y0:y1, x0:x1] = refined_crop.clamp(0.0, 1.0)
    return refined.clamp(0.0, 1.0)


def _prepare_recolor_projection_alpha(
    alpha: torch.Tensor,
    size: tuple[int, int],
    alpha_power: float,
    boundary_boost: float,
    boundary_kernel_size: int,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.nn.functional.interpolate(
        alpha.to(dtype=torch.float32, device=device),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    alpha_power = max(0.01, float(alpha_power))
    shaped = alpha.pow(alpha_power)
    if boundary_boost <= 0.0:
        return shaped
    boundary_kernel_size = max(1, int(boundary_kernel_size))
    if boundary_kernel_size % 2 == 0:
        boundary_kernel_size += 1
    if boundary_kernel_size <= 1:
        inner_boundary = alpha
    else:
        pad = boundary_kernel_size // 2
        eroded = -torch.nn.functional.max_pool2d(
            -alpha,
            kernel_size=boundary_kernel_size,
            stride=1,
            padding=pad,
        )
        inner_boundary = (alpha - eroded).clamp(0.0, 1.0) * alpha
    return (shaped + float(boundary_boost) * inner_boundary).clamp(0.0, 1.0)


def _build_recolor_clean_projection_image(
    current_image: torch.Tensor,
    source_image: torch.Tensor,
    reference_image: torch.Tensor | None,
    target_rgb: torch.Tensor,
    alpha: torch.Tensor,
    mode: str,
    texture_kernel_size: int,
    luma_texture_scale: float,
    chroma_texture_scale: float,
    composite_mode: str,
    background_kernel_size: int,
) -> torch.Tensor:
    current_yuv = rgb_to_yuv(current_image.to(torch.float32))
    source_yuv = rgb_to_yuv(source_image.to(torch.float32))
    if reference_image is not None:
        chroma_yuv = rgb_to_yuv(reference_image.to(torch.float32))
    else:
        target_image = target_rgb.to(dtype=torch.float32, device=current_image.device).view(1, 3, 1, 1).expand_as(
            current_image
        )
        chroma_yuv = rgb_to_yuv(target_image)

    mode = mode.strip().lower()
    if mode == "strict":
        target_y = source_yuv[:, :1]
    elif mode == "soft":
        source_low = _luma_low_pass(source_yuv[:, :1], texture_kernel_size)
        current_low = _luma_low_pass(current_yuv[:, :1], texture_kernel_size)
        target_y = (current_low + (source_yuv[:, :1] - source_low)).clamp(0.0, 1.0)
    elif mode == "yuv_texture":
        source_low = _luma_low_pass(source_yuv, texture_kernel_size)
        current_low_y = _luma_low_pass(current_yuv[:, :1], texture_kernel_size)
        luma_high = source_yuv[:, :1] - source_low[:, :1]
        chroma_high = source_yuv[:, 1:] - source_low[:, 1:]
        target_y = (current_low_y + float(luma_texture_scale) * luma_high).clamp(0.0, 1.0)
        chroma_yuv = torch.cat(
            [
                chroma_yuv[:, :1],
                chroma_yuv[:, 1:] + float(chroma_texture_scale) * chroma_high,
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Unsupported recolor clean projection mode: {mode}")

    target_yuv = torch.cat([target_y, chroma_yuv[:, 1:]], dim=1)
    target_rgb_image = _yuv_to_rgb(target_yuv).clamp(0.0, 1.0)
    alpha = alpha.to(dtype=torch.float32, device=current_image.device).clamp(0.0, 1.0)
    composite_mode = composite_mode.strip().lower()
    if composite_mode == "blend":
        background = current_image.to(torch.float32)
    elif composite_mode == "matte":
        background = _estimate_local_background(
            source_image.to(dtype=torch.float32, device=current_image.device),
            alpha,
            kernel_size=background_kernel_size,
        )
    else:
        raise ValueError(f"Unsupported recolor clean projection composite mode: {composite_mode}")
    return (background * (1.0 - alpha) + target_rgb_image * alpha).clamp(0.0, 1.0)


def _build_recolor_projection_latent_target(
    pipe,
    current_image: torch.Tensor,
    source_image: torch.Tensor,
    reference_image: torch.Tensor | None,
    target_rgb: torch.Tensor,
    composite_alpha_image: torch.Tensor,
    gate_alpha_image: torch.Tensor,
    mode: str,
    texture_kernel_size: int,
    luma_texture_scale: float,
    chroma_texture_scale: float,
    composite_mode: str,
    background_kernel_size: int,
    latent_size: tuple[int, int],
    latent_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference_image is not None and reference_image.shape[-2:] != current_image.shape[-2:]:
        reference_image = torch.nn.functional.interpolate(
            reference_image.to(dtype=torch.float32, device=current_image.device),
            size=current_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    target_image = _build_recolor_clean_projection_image(
        current_image=current_image,
        source_image=source_image.to(dtype=torch.float32, device=current_image.device),
        reference_image=reference_image,
        target_rgb=target_rgb.to(device=current_image.device),
        alpha=composite_alpha_image.to(dtype=torch.float32, device=current_image.device),
        mode=mode,
        texture_kernel_size=texture_kernel_size,
        luma_texture_scale=luma_texture_scale,
        chroma_texture_scale=chroma_texture_scale,
        composite_mode=composite_mode,
        background_kernel_size=background_kernel_size,
    )
    target_latent = _encode_unit_image_to_latent(pipe, target_image).to(dtype=torch.float32, device=latent_device)
    gate_latent = torch.nn.functional.interpolate(
        gate_alpha_image.to(dtype=torch.float32, device=latent_device),
        size=latent_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    return target_latent, gate_latent


def _mask_bbox_for_image(
    mask: torch.Tensor | None,
    image_size: tuple[int, int],
    pad_fraction: float = 0.75,
) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    width, height = image_size
    mask_img = torch.nn.functional.interpolate(
        mask.detach().float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    active = mask_img > 0.2
    if not bool(active.any().item()):
        return None
    ys, xs = active.nonzero(as_tuple=True)
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    side = max(x1 - x0, y1 - y0)
    pad = int(round(side * pad_fraction))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = max(side // 2 + pad, 48)
    return (
        max(0, cx - half),
        max(0, cy - half),
        min(width, cx + half),
        min(height, cy + half),
    )


def _save_clean_estimate_debug_images(
    pipe,
    x0_src_step: torch.Tensor,
    x0_tar: torch.Tensor,
    edit_mask: torch.Tensor | None,
    output_dir: str,
    step_index: int,
    t_value: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        src_image = decode_latent_to_unit_image(pipe, x0_src_step)
        tar_image = decode_latent_to_unit_image(pipe, x0_tar)
    src_pil = _tensor_unit_image_to_pil(src_image)
    tar_pil = _tensor_unit_image_to_pil(tar_image)
    prefix = f"step_{step_index:03d}_t_{t_value:.4f}"
    src_pil.save(os.path.join(output_dir, f"{prefix}_x0_src.png"))
    tar_pil.save(os.path.join(output_dir, f"{prefix}_x0_tar.png"))

    gap = (tar_image - src_image).detach().float().abs().mean(dim=1, keepdim=True)
    gap = gap / gap.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
    gap_pil = _tensor_unit_image_to_pil(gap.repeat(1, 3, 1, 1))
    gap_pil.save(os.path.join(output_dir, f"{prefix}_target_gap.png"))

    bbox = _mask_bbox_for_image(edit_mask, tar_pil.size)
    if bbox is not None:
        src_pil.crop(bbox).save(os.path.join(output_dir, f"{prefix}_x0_src_crop.png"))
        tar_pil.crop(bbox).save(os.path.join(output_dir, f"{prefix}_x0_tar_crop.png"))
        gap_pil.crop(bbox).save(os.path.join(output_dir, f"{prefix}_target_gap_crop.png"))
    with open(os.path.join(output_dir, f"{prefix}_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({"step": step_index, "t": t_value, "crop_box": bbox}, handle, indent=2)


from sd3_model_ops import (
    _cfg_v_sd3_with_grad,
    calc_cfg_v_sd3,
    calc_cfg_v_sd3_with_source_qkv_injection,
    extract_sd3_feature_structure_map,
    invert_source_sd3,
    predict_x0_from_linear_rf_path,
    scale_noise,
)

def HRecSD3Edit(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    T_steps: int = 28,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 10.5,
    inversion_guidance_scale: Optional[float] = None,
    base_guidance_scale: Optional[float] = None,
    n_max: int = 24,
    eta: float = 0.8,
    rec_guidance_scale: float = 0.0,
    struct_guidance_scale: float = 0.5,
    edit_hedit_guidance_scale: float = 0.0,
    edit_field_mode: str = "surrogate",
    edit_src_cfg_scale: Optional[float] = None,
    edit_guidance_scale: float = 0.0,
    edit_region_guidance_scale: float = 0.0,
    edit_target_guidance_scale: float = 0.0,
    edit_source_guidance_scale: float = 0.0,
    edit_clip_guidance_scale: float = 0.0,
    edit_clip_match_base_scale: float = 0.0,
    edit_image_tv_scale: float = 0.0,
    edit_text_guidance_scale: float = 0.0,
    edit_text_source_scale: float = 0.8,
    edit_text_core_weight: float = 1.0,
    edit_text_subject_weight: float = 0.3,
    edit_text_source_prompt: str | None = None,
    edit_text_target_prompt: str | None = None,
    edit_local_target_prompt: str | None = None,
    edit_local_target_guidance_scale: float = 0.0,
    edit_local_target_cfg_scale: Optional[float] = None,
    identity_break_stop: float = 0.55,
    target_attract_start: float = 0.25,
    edit_dds_guidance_scale: float = 0.0,
    edit_dds_source_scale: float = 0.8,
    edit_app_guidance_scale: float = 0.0,
    edit_color_guidance_scale: float = 0.0,
    edit_color_source: str | None = None,
    edit_color_target: str | None = None,
    edit_color_mask_path: str | None = None,
    edit_color_mask_threshold: float = 0.38,
    edit_color_mask_softness: float = 0.10,
    edit_color_luma_gate_min: float = 0.0,
    edit_color_luma_gate_softness: float = 0.08,
    edit_color_detail_protect_scale: float = 0.0,
    edit_color_detail_protect_threshold: float = 0.35,
    edit_color_detail_protect_softness: float = 0.08,
    edit_color_alpha_matte: bool = False,
    edit_color_alpha_matte_mode: str = "color",
    edit_color_alpha_matte_kernel_size: int = 9,
    edit_color_alpha_matte_threshold: float | None = None,
    edit_color_alpha_matte_softness: float | None = None,
    edit_color_alpha_matte_max_size: int = 256,
    edit_color_alpha_matte_epsilon: float = 1e-7,
    edit_color_alpha_matte_constraint_scale: float = 100.0,
    edit_color_target_chroma_scale: float = 1.0,
    edit_color_smooth_kernel: int = 5,
    edit_color_luma_preserve_scale: float = 0.35,
    edit_color_luma_gradient_preserve_scale: float = 0.15,
    edit_color_texture_preserve_scale: float = 0.0,
    edit_color_texture_kernel_size: int = 7,
    edit_color_boundary_chroma_scale: float = 0.0,
    edit_color_boundary_kernel_size: int = 7,
    edit_color_clean_projection_scale: float = 0.0,
    edit_color_clean_projection_mode: str = "soft",
    edit_color_clean_projection_texture_kernel_size: int = 7,
    edit_color_clean_projection_luma_texture_scale: float = 1.0,
    edit_color_clean_projection_chroma_texture_scale: float = 0.25,
    edit_color_clean_projection_delta_lowpass_kernel: int = 0,
    edit_color_clean_projection_alpha_power: float = 1.0,
    edit_color_clean_projection_boundary_boost: float = 0.0,
    edit_color_clean_projection_boundary_kernel_size: int = 7,
    edit_color_clean_projection_composite_mode: str = "blend",
    edit_color_clean_projection_background_kernel_size: int = 31,
    edit_color_clean_projection_target_mode: str = "static",
    edit_color_clean_projection_refresh_interval: int = 0,
    edit_ref_guidance_scale: float = 0.0,
    edit_ref_image_path: str | None = None,
    edit_ref_mask_path: str | None = None,
    edit_ref_structure_image_path: str | None = None,
    edit_ref_chroma_mode: str = "yuv",
    edit_ref_chroma_magnitude_scale: float = 1.0,
    edit_ref_luma_preserve_scale: float = 0.35,
    edit_ref_gradient_preserve_scale: float = 0.15,
    edit_ref_darkness_guard_scale: float = 0.0,
    edit_ref_darkness_guard_margin: float = 0.03,
    edit_ref_smooth_kernel: int = 1,
    edit_ref_lowfreq_suppress_kernel: int = 0,
    edit_ref_lowfreq_suppress_scale: float = 0.0,
    edit_ref_schedule_start: float = 0.0,
    edit_ref_schedule_stop: float = 0.0,
    edit_ref_schedule_power: float = 1.0,
    edit_ref_max_struct_rms_ratio: float = 0.0,
    edit_ref_project_struct_conflict: float = 0.0,
    completion_clean_delta_scale: float = 0.0,
    completion_clean_delta_image_path: str | None = None,
    completion_clean_delta_mask_path: str | None = None,
    completion_clean_delta_schedule_start: float = 0.0,
    completion_clean_delta_schedule_stop: float = 0.0,
    completion_clean_delta_schedule_power: float = 1.0,
    edit_core_scale: float = 1.35,
    edit_subject_scale: float = 0.35,
    source_inject_q_scale: float = 0.0,
    source_inject_k_scale: float = 0.0,
    source_inject_v_scale: float = 0.0,
    source_inject_layer_from: int = -1,
    source_inject_layer_to: int = -1,
    source_inject_steps: int = 8,
    source_inject_mask_mode: str = "none",
    source_inject_mask_box: tuple[float, float, float, float] | None = None,
    edit_bound_scale: float = 0.0,
    clip_start_timestep: float = 0.0,
    clip_stop_timestep: float = 0.6,
    preserve_blend_scale: float = 0.0,
    preserve_blend_start_timestep: float = 0.5,
    alpha_max: Optional[float] = None,
    alpha_schedule: str = "constant",
    beta_max: Optional[float] = None,
    beta_schedule: str = "constant",
    adaptive_clean_control: bool = False,
    adaptive_edit_target_progress: float = 0.0,
    adaptive_edit_target_rms: float = 0.0,
    adaptive_rmsgap_mode: str = "legacy",
    adaptive_rmsgap_dead_zone: float = 0.0,
    adaptive_rmsgap_preserve_gate_budget: float = 0.0,
    adaptive_hybrid_progress_target: float = 0.0,
    adaptive_hybrid_progress_gain: float = 0.0,
    adaptive_hybrid_progress_ema_decay: float = 0.0,
    adaptive_hybrid_preserve_gate_budget: float = 0.0,
    adaptive_preserve_drift_budget: float = 0.0,
    adaptive_edit_gain: float = 0.0,
    adaptive_preserve_gain: float = 0.0,
    adaptive_edit_weight_min: float = 0.7,
    adaptive_edit_weight_max: float = 1.8,
    adaptive_preserve_weight_min: float = 1.0,
    adaptive_preserve_weight_max: float = 2.0,
    adaptive_projection_scale: float = 0.0,
    adaptive_preserve_clean_correction_scale: float = 0.0,
    removal_controller_mode: str = "none",
    removal_fill_scale: float = 0.0,
    removal_suppression_scale: float = 0.0,
    removal_ring_rec_scale: float = 0.0,
    velocity_conversion_mode: str = "linear_path",
    linear_path_t_min: float = 0.05,
    rec_stop_timestep: float = 0.08,
    trajectory_preserve_scale: float = 0.0,
    trajectory_subject_preserve_scale: float = 0.0,
    edit_initial_noise_scale: float = 0.0,
    edit_initial_noise_region: str = "core",
    region_target_transport_scale: float = 0.0,
    region_target_outside_lock_scale: float = 0.0,
    attention_mask_mode: str = "changed_union",
    attention_mask_target_words: list[str] | None = None,
    attention_mask_source_words: list[str] | None = None,
    attention_mask_subject_threshold: float = 0.48,
    attention_mask_core_threshold: float = 0.72,
    attention_mask_max_area_ratio: float = 0.25,
    attention_mask_fallback_threshold: float = 0.72,
    object_mask_provider: str = "attention",
    semantic_base_mask_path: str | None = None,
    support_score: str = "attention_x_clean",
    support_edit_operation: str = "auto",
    support_relation: str = "auto",
    support_grounding_method: str = "external_mask",
    save_support_debug_maps: bool = False,
    support_debug_only: bool = False,
    support_temporal_aggregation: str = "single",
    support_new_tokens: list[str] | None = None,
    support_host_tokens: list[str] | None = None,
    support_removed_tokens: list[str] | None = None,
    support_attention_power: float = 1.0,
    support_disagreement_power: float = 1.0,
    support_top_percentile: float = 90.0,
    support_min_area_ratio: float = 0.02,
    support_max_area_ratio: float = 0.30,
    support_keep_components: int = 2,
    support_dilate_radius: int = 5,
    support_blur_kernel: int = 5,
    attention_velocity_support_pad_x: float = 0.015,
    attention_velocity_support_pad_y: float = 0.010,
    attention_velocity_support_min_width: float = 0.18,
    attention_velocity_support_min_height: float = 0.065,
    mask_layering_mode: str = "object_contact",
    mask_object_threshold: float = 0.45,
    mask_contact_dilate_kernel: int = 7,
    mask_contact_scale: float = 0.25,
    mask_contact_edge_threshold: float = 0.55,
    mask_contact_edge_protect_scale: float = 0.75,
    mask_trimap_inner_erode_kernel: int = 3,
    mask_trimap_outer_dilate_kernel: int = 5,
    mask_trimap_boundary_edit_scale: float = 0.8,
    mask_trimap_boundary_preserve_scale: float = 0.0,
    mask_output_dir: str | None = None,
    edit_mask_dilate_kernel: int = 0,
    edit_mask_erode_kernel: int = 0,
    edit_mask_smooth_kernel: int = 0,
    edit_mask_hole_fraction: float = 0.0,
    edit_mask_boundary_noise_scale: float = 0.0,
    edit_mask_component_threshold: float = 0.0,
    edit_mask_keep_components: int = 0,
    edit_mask_component_y_min: float | None = None,
    edit_mask_component_y_max: float | None = None,
    edit_mask_shift_y: float = 0.0,
    edit_mask_shift_x: float = 0.0,
    auto_local_boxes: bool = False,
    auto_box_threshold: float = 0.35,
    auto_edit_pad_x: float = 0.08,
    auto_edit_pad_y: float = 0.04,
    auto_edit_min_width: float = 0.28,
    auto_edit_min_height: float = 0.10,
    auto_source_pad_x: float = 0.14,
    auto_source_pad_y: float = 0.08,
    auto_preserve_pad_x: float = 0.04,
    auto_preserve_start_offset: float = 0.06,
    auto_preserve_height: float = 0.24,
    edit_mask_box: tuple[float, float, float, float] | None = None,
    edit_mask_box_mode: str = "replace",
    edit_mask_exclude_box: tuple[float, float, float, float] | None = None,
    edit_mask_use_core_as_subject: bool = False,
    external_edit_mask_path: str | None = None,
    external_edit_mask_mode: str = "replace",
    mask_blend: bool = False,
    mask_blend_mode: str = "subject",
    final_preserve_box: tuple[float, float, float, float] | None = None,
    log_every: int = 0,
    stats_output_path: Optional[str] = None,
    clean_estimate_debug_dir: str | None = None,
):
    """
    RF image editing via source inversion + controlled reverse ODE.

    The active implementation follows the project logic:
    - a source-conditioned RF base prior
    - a separate reconstruction guidance branch
    - a separate editing guidance branch
    both defined from the same clean estimate x_hat_0.

    The current implementation uses energy-inspired surrogate velocity fields.
    Some branches are exact autograd gradients, while others are manually
    constructed velocity surrogates derived from clean-space displacement or
    feature-space differences.
    """
    device = x_src.device
    valid_source_inject_mask_modes = {"none", "edit", "core", "preserve", "box"}
    if source_inject_mask_mode not in valid_source_inject_mask_modes:
        raise ValueError(f"Unsupported source_inject_mask_mode: {source_inject_mask_mode}")
    valid_object_mask_providers = {
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
    }
    if object_mask_provider not in valid_object_mask_providers:
        raise ValueError(f"Unsupported object_mask_provider: {object_mask_provider}")
    valid_edit_field_modes = {"surrogate", "rf_diff", "text_diff", "rf_text_diff"}
    if edit_field_mode not in valid_edit_field_modes:
        raise ValueError(f"Unsupported edit_field_mode: {edit_field_mode}")
    use_legacy_edit_surrogates = edit_field_mode == "surrogate"
    use_rf_diff_edit_field = edit_field_mode in {"surrogate", "rf_diff", "rf_text_diff"}
    use_text_diff_edit_field = edit_field_mode in {"surrogate", "text_diff", "rf_text_diff"}
    use_auxiliary_edit_fields = edit_field_mode == "surrogate"
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    if inversion_guidance_scale is None:
        inversion_guidance_scale = src_guidance_scale
    if base_guidance_scale is None:
        base_guidance_scale = src_guidance_scale
    if edit_src_cfg_scale is None:
        edit_src_cfg_scale = base_guidance_scale
    if edit_local_target_cfg_scale is None:
        edit_local_target_cfg_scale = tar_guidance_scale
    source_encode_guidance_scale = max(
        float(inversion_guidance_scale),
        float(base_guidance_scale),
        float(edit_src_cfg_scale),
        float(edit_local_target_cfg_scale),
        1.0,
    )

    with torch.no_grad():
        pipe._guidance_scale = source_encode_guidance_scale
        (
            src_prompt_embeds,
            src_negative_prompt_embeds,
            src_pooled_prompt_embeds,
            src_negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=src_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            device=device,
        )

        pipe._guidance_scale = tar_guidance_scale
        (
            tar_prompt_embeds,
            tar_negative_prompt_embeds,
            tar_pooled_prompt_embeds,
            tar_negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=tar_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            device=device,
        )
        local_target_prompt_embeds = None
        local_target_negative_prompt_embeds = None
        local_target_pooled_prompt_embeds = None
        local_target_negative_pooled_prompt_embeds = None
        if edit_local_target_prompt:
            pipe._guidance_scale = edit_local_target_cfg_scale
            (
                local_target_prompt_embeds,
                local_target_negative_prompt_embeds,
                local_target_pooled_prompt_embeds,
                local_target_negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=edit_local_target_prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                device=device,
            )

    # --- Source inversion: forward ODE x_src → z_T ---
    print("[inversion] running source forward ODE ...")
    source_inject_enabled = max(source_inject_q_scale, source_inject_k_scale, source_inject_v_scale) > 0.0
    inversion_result = invert_source_sd3(
        pipe=pipe,
        x_src=x_src,
        negative_prompt_embeds=src_negative_prompt_embeds,
        prompt_embeds=src_prompt_embeds,
        negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
        pooled_prompt_embeds=src_pooled_prompt_embeds,
        guidance_scale=inversion_guidance_scale,
        timesteps=timesteps,
        T_steps=T_steps,
        n_max=n_max,
        return_trajectory=(
            trajectory_preserve_scale > 0.0
            or trajectory_subject_preserve_scale > 0.0
            or source_inject_enabled
            or region_target_outside_lock_scale > 0.0
        ),
    )
    if (
        trajectory_preserve_scale > 0.0
        or trajectory_subject_preserve_scale > 0.0
        or source_inject_enabled
        or region_target_outside_lock_scale > 0.0
    ):
        z_T, source_trajectory_by_timestep = inversion_result
    else:
        z_T = inversion_result
        source_trajectory_by_timestep = {}
    print("[inversion] done.")

    if alpha_max is None:
        alpha_max = rec_guidance_scale
    if beta_max is None:
        beta_max = 1.0
    color_source = infer_source_color_rgb(src_prompt, edit_color_source)
    color_target = infer_target_color_rgb(src_prompt, tar_prompt, edit_color_target)
    n_transformer_blocks = len(pipe.transformer.transformer_blocks)
    if source_inject_layer_from < 0:
        source_inject_layer_from = max(0, (2 * n_transformer_blocks) // 3)
    if source_inject_layer_to < 0:
        source_inject_layer_to = n_transformer_blocks
    source_inject_layer_from = max(0, min(int(source_inject_layer_from), n_transformer_blocks))
    source_inject_layer_to = max(source_inject_layer_from, min(int(source_inject_layer_to), n_transformer_blocks))

    # Extract attention mask for soft E_rec (optional)
    M_edit = None
    M_core = None
    M_preserve = None
    M_contact = None
    M_structure_edge = None
    M_attention_object = None
    M_velocity_object = None
    M_attention_velocity_object = None
    M_semantic_base = None
    M_semantic_velocity_object = None
    M_generic_support_attention = None
    M_generic_support_host = None
    M_generic_support_removed = None
    M_generic_support_clean = None
    M_generic_support_velocity = None
    M_generic_support_score = None
    M_operation_support_grounding = None
    M_operation_support_relation = None
    generic_support_stats: dict[str, float | int | str] = {}
    auto_anchor_mask = None
    auto_anchor_box = None
    mask_area_guard_applied = False
    mask_area_before_guard = None
    mask_area_guard_box = None
    resolved_edit_mask_box = edit_mask_box
    resolved_edit_mask_exclude_box = edit_mask_exclude_box
    resolved_source_inject_mask_box = source_inject_mask_box
    resolved_final_preserve_box = final_preserve_box
    source_feature_maps: dict[int, torch.Tensor] = {}
    source_attention_maps: dict[int, torch.Tensor] = {}
    clip_reward: LocalCLIPTextReward | None = None
    clip_reference: CLIPReferenceState | None = None
    grad_vae = None
    if (
        rec_guidance_scale > 0.0
        or edit_hedit_guidance_scale > 0.0
        or edit_guidance_scale > 0.0
        or edit_region_guidance_scale > 0.0
        or edit_target_guidance_scale > 0.0
        or edit_source_guidance_scale > 0.0
        or edit_clip_guidance_scale > 0.0
        or edit_text_guidance_scale > 0.0
        or edit_dds_guidance_scale > 0.0
        or edit_app_guidance_scale > 0.0
        or edit_color_guidance_scale > 0.0
        or adaptive_clean_control
        or trajectory_preserve_scale > 0.0
        or trajectory_subject_preserve_scale > 0.0
        or edit_initial_noise_scale > 0.0
        or external_edit_mask_path is not None
        or auto_local_boxes
        or object_mask_provider
        in {
            "attention",
            "velocity_diff",
            "attention_velocity",
            "semantic",
            "semantic_velocity",
            "generic_support",
            "operation_support_v3",
            "auto",
        }
        or source_inject_mask_mode in {"edit", "core", "preserve"}
    ):
        with torch.no_grad():
            t_mid = timesteps[len(timesteps) // 2]
            masks = extract_attention_masks(
                pipe=pipe,
                x_src=x_src,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                src_prompt_embeds=src_prompt_embeds,
                src_pooled_embeds=src_pooled_prompt_embeds,
                tar_prompt_embeds=tar_prompt_embeds,
                tar_pooled_embeds=tar_pooled_prompt_embeds,
                t=t_mid,
                mode=attention_mask_mode,
                target_token_words=attention_mask_target_words,
                source_token_words=attention_mask_source_words,
                subject_threshold=attention_mask_subject_threshold,
                core_threshold=attention_mask_core_threshold,
            )
            M_edit = masks["subject"].to(dtype=x_src.dtype)
            M_core = masks["core"].to(dtype=x_src.dtype)
            M_preserve = masks["preserve"].to(dtype=x_src.dtype)
            if object_mask_provider in {
                "velocity_diff",
                "attention_velocity",
                "semantic_velocity",
                "generic_support",
                "operation_support_v3",
            }:
                v_src_mid = calc_cfg_v_sd3(
                    pipe=pipe,
                    latents=x_src,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=base_guidance_scale,
                    t=t_mid,
                )
                v_tar_mid = calc_cfg_v_sd3(
                    pipe=pipe,
                    latents=x_src,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t_mid,
                )
                M_velocity_object = _velocity_diff_object_mask(
                    source_velocity=v_src_mid,
                    target_velocity=v_tar_mid,
                    threshold=attention_mask_fallback_threshold,
                    fallback=M_core,
                    keep_components=3 if object_mask_provider == "attention_velocity" else 1,
                )
                if M_velocity_object is not None:
                    M_edit = M_velocity_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if object_mask_provider in {"generic_support", "operation_support_v3"}:
                object_source = masks.get("target_changed")
                if support_new_tokens is not None:
                    new_masks = extract_attention_masks(
                        pipe=pipe,
                        x_src=x_src,
                        src_prompt=src_prompt,
                        tar_prompt=tar_prompt,
                        src_prompt_embeds=src_prompt_embeds,
                        src_pooled_embeds=src_pooled_prompt_embeds,
                        tar_prompt_embeds=tar_prompt_embeds,
                        tar_pooled_embeds=tar_pooled_prompt_embeds,
                        t=t_mid,
                        mode=attention_mask_mode,
                        target_token_words=support_new_tokens,
                        source_token_words=None,
                        subject_threshold=attention_mask_subject_threshold,
                        core_threshold=attention_mask_core_threshold,
                    )
                    object_source = new_masks.get("target_changed")
                if object_source is None or float(object_source.detach().float().max().item()) <= 1e-6:
                    object_source = masks.get("combined")
                host_source = None
                if support_host_tokens is not None:
                    host_masks = extract_attention_masks(
                        pipe=pipe,
                        x_src=x_src,
                        src_prompt=src_prompt,
                        tar_prompt=tar_prompt,
                        src_prompt_embeds=src_prompt_embeds,
                        src_pooled_embeds=src_pooled_prompt_embeds,
                        tar_prompt_embeds=tar_prompt_embeds,
                        tar_pooled_embeds=tar_pooled_prompt_embeds,
                        t=t_mid,
                        mode=attention_mask_mode,
                        target_token_words=support_host_tokens,
                        source_token_words=support_host_tokens,
                        subject_threshold=attention_mask_subject_threshold,
                        core_threshold=attention_mask_core_threshold,
                    )
                    host_source = torch.maximum(host_masks["source_changed"], host_masks["target_changed"])
                removed_source = None
                if support_removed_tokens is not None:
                    removed_masks = extract_attention_masks(
                        pipe=pipe,
                        x_src=x_src,
                        src_prompt=src_prompt,
                        tar_prompt=tar_prompt,
                        src_prompt_embeds=src_prompt_embeds,
                        src_pooled_embeds=src_pooled_prompt_embeds,
                        tar_prompt_embeds=tar_prompt_embeds,
                        tar_pooled_embeds=tar_pooled_prompt_embeds,
                        t=t_mid,
                        mode=attention_mask_mode,
                        target_token_words=None,
                        source_token_words=support_removed_tokens,
                        subject_threshold=attention_mask_subject_threshold,
                        core_threshold=attention_mask_core_threshold,
                    )
                    removed_source = removed_masks["source_changed"]
                support_grounding = None
                if (
                    object_mask_provider == "operation_support_v3"
                    and semantic_base_mask_path is not None
                    and (support_grounding_method or "external_mask").strip().lower() != "none"
                ):
                    support_grounding = load_external_mask_like(M_edit, semantic_base_mask_path)
                    M_operation_support_grounding = support_grounding
                if object_mask_provider == "operation_support_v3":
                    temporal_mode = (support_temporal_aggregation or "single").strip().lower()
                    clean_map_override = None
                    velocity_map_override = None
                    temporal_step_count = 1
                    if temporal_mode in {"mean", "max"}:
                        support_indices = sorted(
                            {
                                max(0, min(len(timesteps) - 1, len(timesteps) // 4)),
                                max(0, min(len(timesteps) - 1, len(timesteps) // 2)),
                                max(0, min(len(timesteps) - 1, (3 * len(timesteps)) // 4)),
                            }
                        )
                        clean_maps = []
                        velocity_maps = []
                        mid_index = len(timesteps) // 2
                        for support_index in support_indices:
                            support_t = timesteps[support_index]
                            if support_index == mid_index:
                                v_src_support = v_src_mid
                                v_tar_support = v_tar_mid
                            else:
                                v_src_support = calc_cfg_v_sd3(
                                    pipe=pipe,
                                    latents=x_src,
                                    negative_prompt_embeds=src_negative_prompt_embeds,
                                    prompt_embeds=src_prompt_embeds,
                                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                                    guidance_scale=base_guidance_scale,
                                    t=support_t,
                                )
                                v_tar_support = calc_cfg_v_sd3(
                                    pipe=pipe,
                                    latents=x_src,
                                    negative_prompt_embeds=tar_negative_prompt_embeds,
                                    prompt_embeds=tar_prompt_embeds,
                                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                    guidance_scale=tar_guidance_scale,
                                    t=support_t,
                                )
                            clean_maps.append(compute_clean_disagreement(x_src, support_t, v_src_support, v_tar_support))
                            velocity_maps.append(compute_velocity_disagreement(v_src_support, v_tar_support))
                        temporal_step_count = len(support_indices)
                        clean_stack = torch.stack(clean_maps, dim=0)
                        velocity_stack = torch.stack(velocity_maps, dim=0)
                        if temporal_mode == "max":
                            clean_map_override = clean_stack.max(dim=0).values
                            velocity_map_override = velocity_stack.max(dim=0).values
                        else:
                            clean_map_override = clean_stack.mean(dim=0)
                            velocity_map_override = velocity_stack.mean(dim=0)
                    elif temporal_mode != "single":
                        raise ValueError(f"Unsupported support_temporal_aggregation: {support_temporal_aggregation}")
                    generic = build_operation_support_v3(
                        attention_map=object_source,
                        x_t=x_src,
                        t=t_mid,
                        source_velocity=v_src_mid,
                        target_velocity=v_tar_mid,
                        host_attention_map=host_source,
                        removed_attention_map=removed_source,
                        grounding_mask=support_grounding,
                        edit_operation=support_edit_operation,
                        relation=support_relation,
                        candidate=support_score,
                        attention_power=support_attention_power,
                        disagreement_power=support_disagreement_power,
                        top_percentile=support_top_percentile,
                        min_area_ratio=support_min_area_ratio,
                        max_area_ratio=support_max_area_ratio,
                        keep_components=support_keep_components,
                        dilate_radius=support_dilate_radius,
                        blur_kernel=support_blur_kernel,
                        clean_map_override=clean_map_override,
                        velocity_map_override=velocity_map_override,
                        temporal_aggregation=temporal_mode,
                        temporal_steps=temporal_step_count,
                    )
                    M_operation_support_relation = generic.relation_map
                    if save_support_debug_maps and mask_output_dir is not None:
                        save_support_debug(generic, mask_output_dir)
                else:
                    generic = build_generic_support(
                        attention_map=object_source,
                        x_t=x_src,
                        t=t_mid,
                        source_velocity=v_src_mid,
                        target_velocity=v_tar_mid,
                        host_attention_map=host_source,
                        removed_attention_map=removed_source,
                        edit_operation=support_edit_operation,
                        score_mode=support_score,
                        attention_power=support_attention_power,
                        disagreement_power=support_disagreement_power,
                        top_percentile=support_top_percentile,
                        min_area_ratio=support_min_area_ratio,
                        max_area_ratio=support_max_area_ratio,
                        keep_components=support_keep_components,
                        dilate_radius=support_dilate_radius,
                        blur_kernel=support_blur_kernel,
                    )
                M_edit = generic.edit_mask.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                M_core = torch.minimum(generic.core_mask.to(dtype=x_src.dtype).clamp(0.0, 1.0), M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                M_generic_support_attention = generic.attention_map
                M_generic_support_host = generic.host_attention_map
                M_generic_support_removed = generic.removed_attention_map
                M_generic_support_clean = generic.clean_disagreement_map
                M_generic_support_velocity = generic.velocity_disagreement_map
                M_generic_support_score = generic.support_score
                generic_support_stats = dict(generic.stats)
            if object_mask_provider in {"attention", "attention_velocity", "auto"}:
                object_source = masks.get("target_changed")
                if object_source is None or float(object_source.detach().float().max().item()) <= 1e-6:
                    object_source = masks.get("combined")
                M_attention_object = _attention_object_mask_from_map(
                    object_source,
                    threshold=attention_mask_fallback_threshold,
                    fallback=M_core,
                    keep_components=3 if object_mask_provider == "attention_velocity" else 1,
                )
                if M_attention_object is not None:
                    M_edit = M_attention_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if object_mask_provider == "attention_velocity":
                M_attention_velocity_object = _attention_velocity_object_mask(
                    M_attention_object,
                    M_velocity_object,
                    fallback=M_core,
                    support_pad_x=attention_velocity_support_pad_x,
                    support_pad_y=attention_velocity_support_pad_y,
                    support_min_width=attention_velocity_support_min_width,
                    support_min_height=attention_velocity_support_min_height,
                )
                if M_attention_velocity_object is not None:
                    M_edit = M_attention_velocity_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if object_mask_provider in {"semantic", "semantic_velocity"}:
                if semantic_base_mask_path is None:
                    raise ValueError("--object-mask-provider semantic(_velocity) requires --semantic-base-mask")
                M_semantic_base = load_external_mask_like(M_edit, semantic_base_mask_path)
                if object_mask_provider == "semantic_velocity":
                    M_semantic_velocity_object = _semantic_velocity_object_mask(
                        M_semantic_base,
                        M_velocity_object,
                        fallback=M_semantic_base,
                    )
                    M_edit = M_semantic_velocity_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                else:
                    M_edit = M_semantic_base.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                M_core = M_edit
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            auto_anchor_mask = M_core
            if (
                edit_mask_keep_components > 0
                or edit_mask_component_y_min is not None
                or edit_mask_component_y_max is not None
            ):
                component_threshold = edit_mask_component_threshold if edit_mask_component_threshold > 0.0 else 0.5
                M_edit = filter_spatial_mask_components(
                    M_edit,
                    threshold=component_threshold,
                    keep_components=edit_mask_keep_components,
                    center_y_min=edit_mask_component_y_min,
                    center_y_max=edit_mask_component_y_max,
                ).clamp(0.0, 1.0)
                M_core = filter_spatial_mask_components(
                    M_core,
                    threshold=component_threshold,
                    keep_components=edit_mask_keep_components,
                    center_y_min=edit_mask_component_y_min,
                    center_y_max=edit_mask_component_y_max,
                ).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                auto_anchor_mask = M_core
                if (
                    object_mask_provider in {"attention", "attention_velocity", "auto"}
                    and M_attention_object is not None
                    and float(M_edit.detach().float().max().item()) <= 1e-6
                    and float(M_attention_object.detach().float().max().item()) > 1e-6
                ):
                    M_edit = M_attention_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                    auto_anchor_mask = M_core
                    print("[mask] component filters removed the attention object; restored generic attention object mask")
            if edit_mask_shift_y != 0.0 or edit_mask_shift_x != 0.0:
                M_edit = translate_spatial_mask(M_edit, shift_y=edit_mask_shift_y, shift_x=edit_mask_shift_x)
                M_core = translate_spatial_mask(M_core, shift_y=edit_mask_shift_y, shift_x=edit_mask_shift_x)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                auto_anchor_mask = M_core
            if edit_mask_dilate_kernel > 1:
                M_edit = dilate_spatial_mask(M_edit, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                M_core = dilate_spatial_mask(M_core, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_erode_kernel > 1:
                M_edit = (1.0 - dilate_spatial_mask((1.0 - M_edit).clamp(0.0, 1.0), kernel_size=edit_mask_erode_kernel)).clamp(0.0, 1.0)
                M_core = (1.0 - dilate_spatial_mask((1.0 - M_core).clamp(0.0, 1.0), kernel_size=edit_mask_erode_kernel)).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_hole_fraction > 0.0:
                hole_keep = (torch.rand_like(M_edit) >= float(edit_mask_hole_fraction)).to(dtype=M_edit.dtype)
                M_edit = (M_edit * hole_keep).clamp(0.0, 1.0)
                M_core = torch.minimum((M_core * hole_keep).clamp(0.0, 1.0), M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_boundary_noise_scale > 0.0:
                band_kernel = 3
                edit_eroded = (1.0 - dilate_spatial_mask((1.0 - M_edit).clamp(0.0, 1.0), kernel_size=band_kernel)).clamp(0.0, 1.0)
                edit_band = (dilate_spatial_mask(M_edit, kernel_size=band_kernel) - edit_eroded).clamp(0.0, 1.0)
                core_eroded = (1.0 - dilate_spatial_mask((1.0 - M_core).clamp(0.0, 1.0), kernel_size=band_kernel)).clamp(0.0, 1.0)
                core_band = (dilate_spatial_mask(M_core, kernel_size=band_kernel) - core_eroded).clamp(0.0, 1.0)
                M_edit = (M_edit + (torch.rand_like(M_edit) - 0.5) * float(edit_mask_boundary_noise_scale) * edit_band).clamp(0.0, 1.0)
                M_core = (M_core + (torch.rand_like(M_core) - 0.5) * float(edit_mask_boundary_noise_scale) * core_band).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_smooth_kernel > 1:
                M_edit = smooth_spatial_mask(M_edit, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                M_core = smooth_spatial_mask(M_core, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            mask_area_before_guard = _mask_binary_area_ratio(M_edit, threshold=0.5)
            skip_attention_velocity_soft_guard = (
                object_mask_provider == "attention_velocity"
                and M_attention_velocity_object is not None
                and mask_area_before_guard <= 1.30 * attention_mask_max_area_ratio
            )
            if (
                attention_mask_max_area_ratio > 0.0
                and mask_area_before_guard > attention_mask_max_area_ratio
                and not skip_attention_velocity_soft_guard
            ):
                if object_mask_provider == "velocity_diff" and M_velocity_object is not None:
                    guard_source = M_velocity_object
                else:
                    guard_source = masks.get("target_changed")
                    if guard_source is None or float(guard_source.detach().float().max().item()) <= 1e-6:
                        guard_source = masks.get("combined")
                guard_anchor_box = _conservative_attention_box(
                    guard_source,
                    threshold=attention_mask_fallback_threshold,
                    fallback=_box_from_mask(M_core, threshold=auto_box_threshold),
                )
                if guard_anchor_box is not None:
                    mask_area_guard_box = _expand_box(
                        guard_anchor_box,
                        pad_x=auto_edit_pad_x,
                        pad_y_top=auto_edit_pad_y,
                        pad_y_bottom=auto_edit_pad_y,
                        min_width=0.28,
                        min_height=0.12,
                    )
                    guard_mask = normalized_box_mask_like(M_edit, mask_area_guard_box)
                    M_edit = (M_edit * guard_mask).clamp(0.0, 1.0)
                    M_core = (M_core * guard_mask).clamp(0.0, 1.0)
                    M_core = torch.minimum(M_core, M_edit)
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                    auto_anchor_mask = guard_mask
                    auto_anchor_box = mask_area_guard_box
                    mask_area_guard_applied = True
                    if resolved_edit_mask_box is None:
                        resolved_edit_mask_box = mask_area_guard_box
                    print(
                        "[mask_guard] "
                        f"area={mask_area_before_guard:.2%} "
                        f"limit={attention_mask_max_area_ratio:.2%} "
                        f"box={_box_to_list(mask_area_guard_box)}"
                    )
            if auto_local_boxes:
                if auto_anchor_box is None:
                    auto_anchor_box = _box_from_mask(
                        auto_anchor_mask,
                        threshold=auto_box_threshold,
                        fallback=_box_from_mask(M_edit, threshold=auto_box_threshold),
                    )
                if auto_anchor_box is not None:
                    if resolved_edit_mask_box is None:
                        resolved_edit_mask_box = _expand_box(
                            auto_anchor_box,
                            pad_x=auto_edit_pad_x,
                            pad_y_top=auto_edit_pad_y,
                            pad_y_bottom=auto_edit_pad_y,
                            min_width=auto_edit_min_width,
                            min_height=auto_edit_min_height,
                        )
                    if resolved_source_inject_mask_box is None and source_inject_mask_mode == "box":
                        resolved_source_inject_mask_box = _expand_box(
                            auto_anchor_box,
                            pad_x=auto_source_pad_x,
                            pad_y_top=auto_source_pad_y,
                            pad_y_bottom=auto_source_pad_y,
                            min_width=0.56,
                            min_height=0.24,
                        )
                    if resolved_edit_mask_box is None:
                        preserve_x0, preserve_y0, preserve_x1, preserve_y1 = auto_anchor_box
                    else:
                        preserve_x0, preserve_y0, preserve_x1, preserve_y1 = resolved_edit_mask_box
                    preserve_width = max(preserve_x1 - preserve_x0, 1e-6)
                    px0 = max(0.0, preserve_x0 + 0.15 * preserve_width - auto_preserve_pad_x)
                    px1 = min(1.0, preserve_x1 - 0.05 * preserve_width + auto_preserve_pad_x)
                    y1 = preserve_y1
                    py0 = min(1.0, y1 + auto_preserve_start_offset)
                    py1 = min(1.0, py0 + auto_preserve_height)
                    if py1 > py0:
                        preserve_box = _clamp_box((px0, py0, px1, py1))
                        if resolved_edit_mask_exclude_box is None:
                            resolved_edit_mask_exclude_box = preserve_box
                        if resolved_final_preserve_box is None:
                            resolved_final_preserve_box = preserve_box
            if resolved_edit_mask_box is not None:
                box_mask = normalized_box_mask_like(M_edit, resolved_edit_mask_box)
                if edit_mask_box_mode == "replace":
                    M_edit = box_mask
                    M_core = box_mask
                elif edit_mask_box_mode == "intersect":
                    M_edit = M_edit * box_mask
                    M_core = M_core * box_mask
                elif edit_mask_box_mode == "union":
                    M_edit = torch.maximum(M_edit, box_mask)
                    M_core = torch.maximum(M_core, box_mask)
                else:
                    raise ValueError(f"Unsupported edit_mask_box_mode: {edit_mask_box_mode}")
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if resolved_edit_mask_exclude_box is not None:
                exclude_mask = normalized_box_mask_like(M_edit, resolved_edit_mask_exclude_box)
                keep_mask = (1.0 - exclude_mask).clamp(0.0, 1.0)
                M_edit = (M_edit * keep_mask).clamp(0.0, 1.0)
                M_core = (M_core * keep_mask).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            external_mask = None
            if external_edit_mask_path is not None:
                external_mask = load_external_mask_like(M_edit, external_edit_mask_path)
                if external_edit_mask_mode == "replace":
                    M_edit = external_mask
                    M_core = external_mask
                elif external_edit_mask_mode == "intersect":
                    M_edit = M_edit * external_mask
                    M_core = M_core * external_mask
                elif external_edit_mask_mode == "union":
                    M_edit = torch.maximum(M_edit, external_mask)
                    M_core = torch.maximum(M_core, external_mask)
                else:
                    raise ValueError(f"Unsupported external_edit_mask_mode: {external_edit_mask_mode}")
                if edit_mask_dilate_kernel > 1:
                    M_edit = dilate_spatial_mask(M_edit, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                    M_core = dilate_spatial_mask(M_core, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                if edit_mask_erode_kernel > 1:
                    M_edit = (1.0 - dilate_spatial_mask((1.0 - M_edit).clamp(0.0, 1.0), kernel_size=edit_mask_erode_kernel)).clamp(0.0, 1.0)
                    M_core = (1.0 - dilate_spatial_mask((1.0 - M_core).clamp(0.0, 1.0), kernel_size=edit_mask_erode_kernel)).clamp(0.0, 1.0)
                if edit_mask_hole_fraction > 0.0:
                    hole_keep = (torch.rand_like(M_edit) >= float(edit_mask_hole_fraction)).to(dtype=M_edit.dtype)
                    M_edit = (M_edit * hole_keep).clamp(0.0, 1.0)
                    M_core = (M_core * hole_keep).clamp(0.0, 1.0)
                if edit_mask_boundary_noise_scale > 0.0:
                    band_kernel = 3
                    edit_eroded = (1.0 - dilate_spatial_mask((1.0 - M_edit).clamp(0.0, 1.0), kernel_size=band_kernel)).clamp(0.0, 1.0)
                    edit_band = (dilate_spatial_mask(M_edit, kernel_size=band_kernel) - edit_eroded).clamp(0.0, 1.0)
                    core_eroded = (1.0 - dilate_spatial_mask((1.0 - M_core).clamp(0.0, 1.0), kernel_size=band_kernel)).clamp(0.0, 1.0)
                    core_band = (dilate_spatial_mask(M_core, kernel_size=band_kernel) - core_eroded).clamp(0.0, 1.0)
                    M_edit = (M_edit + (torch.rand_like(M_edit) - 0.5) * float(edit_mask_boundary_noise_scale) * edit_band).clamp(0.0, 1.0)
                    M_core = (M_core + (torch.rand_like(M_core) - 0.5) * float(edit_mask_boundary_noise_scale) * core_band).clamp(0.0, 1.0)
                if edit_mask_smooth_kernel > 1:
                    M_edit = smooth_spatial_mask(M_edit, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                    M_core = smooth_spatial_mask(M_core, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if mask_layering_mode == "object_contact":
                M_edit, M_core, M_contact, M_preserve, M_structure_edge = build_object_contact_masks(
                    edit_mask=M_edit,
                    core_mask=M_core,
                    structure_reference=x_src,
                    object_threshold=mask_object_threshold,
                    contact_dilate_kernel=mask_contact_dilate_kernel,
                    contact_scale=mask_contact_scale,
                    contact_edge_threshold=mask_contact_edge_threshold,
                    contact_edge_protect_scale=mask_contact_edge_protect_scale,
                )
            elif mask_layering_mode == "recolor_trimap":
                M_edit, M_core, M_contact, M_preserve = build_recolor_trimap_masks(
                    edit_mask=M_edit,
                    core_mask=M_core,
                    object_threshold=mask_object_threshold,
                    inner_erode_kernel=mask_trimap_inner_erode_kernel,
                    outer_dilate_kernel=mask_trimap_outer_dilate_kernel,
                    boundary_edit_scale=mask_trimap_boundary_edit_scale,
                    boundary_preserve_scale=mask_trimap_boundary_preserve_scale,
                )
            elif mask_layering_mode != "none":
                raise ValueError(f"Unsupported mask_layering_mode: {mask_layering_mode}")
            if edit_mask_use_core_as_subject:
                M_edit = M_core.clamp(0.0, 1.0)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                for name, mask in masks.items():
                    save_mask_image(mask.to(dtype=torch.float32), os.path.join(mask_output_dir, f"{name}.png"))
                if M_attention_object is not None:
                    save_mask_image(
                        M_attention_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "attention_object.png"),
                    )
                if M_velocity_object is not None:
                    save_mask_image(
                        M_velocity_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "velocity_diff_object.png"),
                    )
                if M_attention_velocity_object is not None:
                    save_mask_image(
                        M_attention_velocity_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "attention_velocity_object.png"),
                    )
                if M_semantic_base is not None:
                    save_mask_image(
                        M_semantic_base.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "semantic_base.png"),
                    )
                if M_semantic_velocity_object is not None:
                    save_mask_image(
                        M_semantic_velocity_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "semantic_velocity_object.png"),
                    )
                if M_generic_support_attention is not None:
                    save_mask_image(
                        M_generic_support_attention.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_attention_map.png"),
                    )
                if M_generic_support_host is not None:
                    save_mask_image(
                        M_generic_support_host.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_host_attention_map.png"),
                    )
                if M_generic_support_removed is not None:
                    save_mask_image(
                        M_generic_support_removed.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_removed_attention_map.png"),
                    )
                if M_generic_support_clean is not None:
                    save_mask_image(
                        M_generic_support_clean.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_clean_disagreement_map.png"),
                    )
                if M_generic_support_velocity is not None:
                    save_mask_image(
                        M_generic_support_velocity.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_velocity_disagreement_map.png"),
                    )
                if M_generic_support_score is not None:
                    save_mask_image(
                        M_generic_support_score.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_support_score.png"),
                    )
                if M_operation_support_grounding is not None:
                    save_mask_image(
                        M_operation_support_grounding.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "operation_v3_grounding_mask.png"),
                    )
                if M_operation_support_relation is not None:
                    save_mask_image(
                        M_operation_support_relation.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "operation_v3_relation_map.png"),
                    )
                save_mask_image(M_edit.to(dtype=torch.float32), os.path.join(mask_output_dir, "subject_final.png"))
                save_mask_image(M_core.to(dtype=torch.float32), os.path.join(mask_output_dir, "core_final.png"))
                save_mask_image(M_preserve.to(dtype=torch.float32), os.path.join(mask_output_dir, "preserve_final.png"))
                if M_contact is not None:
                    save_mask_image(
                        M_contact.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "contact_final.png"),
                    )
                if M_structure_edge is not None:
                    save_mask_image(
                        M_structure_edge.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "structure_edge.png"),
                    )
                if external_mask is not None:
                    save_mask_image(
                        external_mask.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "external_edit_mask.png"),
                    )
            if edit_initial_noise_scale > 0.0:
                if edit_initial_noise_region == "core":
                    noise_gate = M_core
                elif edit_initial_noise_region == "subject":
                    noise_gate = M_edit
                else:
                    raise ValueError(f"Unsupported edit_initial_noise_region: {edit_initial_noise_region}")
                noise_gate = noise_gate.to(device=z_T.device, dtype=torch.float32)
                if noise_gate.shape[-2:] != z_T.shape[-2:]:
                    noise_gate = torch.nn.functional.interpolate(
                        noise_gate,
                        size=z_T.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                noise_gate = (float(edit_initial_noise_scale) * noise_gate).clamp(0.0, 1.0)
                edit_noise = torch.randn_like(z_T, dtype=torch.float32)
                z_T = ((1.0 - noise_gate) * z_T.to(torch.float32) + noise_gate * edit_noise).to(z_T.dtype)
        m_subject = M_edit.squeeze()
        m_core = M_core.squeeze() if M_core is not None else None
        print(
            f"[mask] min={m_subject.min():.3f} "
            f"max={m_subject.max():.3f} "
            f"mean={m_subject.mean():.3f} "
            f"edit(>0.5)={(m_subject > 0.5).float().mean():.2%} "
            f"preserve(<0.2)={(M_preserve.squeeze() < 0.2).float().mean():.2%}"
            + (
                ""
                if m_core is None
                else f" core(>0.5)={(m_core > 0.5).float().mean():.2%}"
            )
        )
    source_inject_mask = None
    if source_inject_mask_mode == "box":
        if resolved_source_inject_mask_box is None:
            raise ValueError("--source-inject-mask-mode box requires --source-inject-mask-box")
        ref_mask = M_edit if M_edit is not None else x_src[:, :1]
        source_inject_mask = normalized_box_mask_like(ref_mask, resolved_source_inject_mask_box)
    elif source_inject_mask_mode == "edit":
        if M_edit is None:
            raise ValueError("--source-inject-mask-mode edit requires an extracted edit mask")
        source_inject_mask = M_edit
    elif source_inject_mask_mode == "core":
        if M_core is None:
            raise ValueError("--source-inject-mask-mode core requires an extracted core mask")
        source_inject_mask = M_core
    elif source_inject_mask_mode == "preserve":
        if M_preserve is None:
            raise ValueError("--source-inject-mask-mode preserve requires an extracted preserve mask")
        source_inject_mask = M_preserve

    if source_inject_mask is not None:
        source_inject_mask = source_inject_mask.to(device=x_src.device, dtype=x_src.dtype).clamp(0.0, 1.0)
        if mask_output_dir is not None:
            os.makedirs(mask_output_dir, exist_ok=True)
            save_mask_image(
                source_inject_mask.to(dtype=torch.float32),
                os.path.join(mask_output_dir, "source_inject_mask.png"),
            )
    mask_stats = spatial_mask_stats(M_edit, prefix="mask")
    core_mask_stats = spatial_mask_stats(M_core, prefix="core_mask")
    contact_mask_stats = spatial_mask_stats(M_contact, prefix="contact_mask")
    structure_edge_stats = spatial_mask_stats(M_structure_edge, prefix="structure_edge")
    preserve_mask_stats = spatial_mask_stats(M_preserve, prefix="preserve_mask")
    source_inject_mask_stats = spatial_mask_stats(source_inject_mask, prefix="source_inject_mask")
    if support_debug_only:
        debug_stats = {
            "mode": "support_debug_only",
            **generic_support_stats,
            **mask_stats,
            **core_mask_stats,
            **contact_mask_stats,
            **structure_edge_stats,
            **preserve_mask_stats,
            **source_inject_mask_stats,
        }
        if stats_output_path is not None:
            with open(stats_output_path, "w", encoding="utf-8") as f:
                json.dump([debug_stats], f, indent=2)
            print(f"[stats] saved support debug stats to {stats_output_path}")
        return x_src

    color_mask_image = None
    color_mask_latent = None
    color_projection_alpha_image = None
    source_color_reference = None
    ref_mask_image = None
    ref_mask_latent = None
    ref_image_reference = None
    ref_structure_reference = None
    completion_clean_target = None
    completion_clean_mask_latent = None
    needs_clip_reward = (
        (use_auxiliary_edit_fields and edit_clip_guidance_scale > 0.0)
        or (use_text_diff_edit_field and edit_text_guidance_scale > 0.0)
    )
    needs_recolor_projection = edit_color_clean_projection_scale > 0.0
    needs_color_reference = edit_color_guidance_scale > 0.0 or needs_recolor_projection
    if needs_clip_reward or needs_color_reference or edit_ref_guidance_scale > 0.0:
        # Offloaded diffusers modules wrap the pipeline VAE with accelerate
        # hooks that produce inference tensors. A dedicated copy keeps the VAE
        # decode path differentiable for image-space edit rewards.
        grad_vae = copy.deepcopy(pipe.vae).to(device)
        grad_vae.eval()
        for param in grad_vae.parameters():
            param.requires_grad_(False)
    if needs_color_reference and color_source is not None and M_edit is not None:
        with torch.no_grad():
            source_color_image = decode_latent_to_unit_image(pipe, x_src, vae_override=grad_vae)
            source_color_reference = source_color_image.detach()
            if edit_color_mask_path is not None:
                external_color_mask = load_external_mask_like(source_color_image[:, :1], edit_color_mask_path)
                edit_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_color_image.dtype, device=source_color_image.device),
                    size=source_color_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                color_mask_image = (external_color_mask * edit_mask_image).clamp(0.0, 1.0)
            else:
                color_mask_image = source_color_similarity_mask(
                    source_color_image,
                    color_source[1],
                    M_edit,
                    threshold=edit_color_mask_threshold,
                    softness=edit_color_mask_softness,
                    luma_gate_min=edit_color_luma_gate_min,
                    luma_gate_softness=edit_color_luma_gate_softness,
                    detail_protect_scale=edit_color_detail_protect_scale,
                    detail_protect_threshold=edit_color_detail_protect_threshold,
                    detail_protect_softness=edit_color_detail_protect_softness,
                )
            if edit_color_alpha_matte:
                alpha_matte_mode = edit_color_alpha_matte_mode.strip().lower()
                if alpha_matte_mode == "color":
                    color_projection_alpha_image = _estimate_recolor_boundary_alpha(
                        source_color_image,
                        color_mask_image,
                        color_source[1],
                        threshold=edit_color_mask_threshold
                        if edit_color_alpha_matte_threshold is None
                        else edit_color_alpha_matte_threshold,
                        softness=edit_color_mask_softness
                        if edit_color_alpha_matte_softness is None
                        else edit_color_alpha_matte_softness,
                        boundary_kernel_size=edit_color_alpha_matte_kernel_size,
                    )
                elif alpha_matte_mode == "closed_form":
                    color_projection_alpha_image = _estimate_recolor_closed_form_alpha(
                        source_color_image,
                        color_mask_image,
                        boundary_kernel_size=edit_color_alpha_matte_kernel_size,
                        max_size=edit_color_alpha_matte_max_size,
                        epsilon=edit_color_alpha_matte_epsilon,
                        constraint_scale=edit_color_alpha_matte_constraint_scale,
                    )
                else:
                    raise ValueError(f"Unsupported recolor alpha matte mode: {edit_color_alpha_matte_mode}")
            else:
                color_projection_alpha_image = color_mask_image
            color_mask_latent = torch.nn.functional.interpolate(
                color_mask_image,
                size=x_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                save_mask_image(color_mask_image, os.path.join(mask_output_dir, "color_edit_mask.png"))
                if edit_color_alpha_matte:
                    save_mask_image(
                        color_projection_alpha_image,
                        os.path.join(mask_output_dir, "color_edit_alpha_matte.png"),
                    )
    if edit_ref_guidance_scale > 0.0:
        if edit_ref_image_path is None:
            raise ValueError("--edit-ref-guidance-scale requires --edit-ref-image")
        with torch.no_grad():
            source_ref_image = decode_latent_to_unit_image(pipe, x_src, vae_override=grad_vae)
            ref_image_reference = load_external_image_like(source_ref_image, edit_ref_image_path).detach()
            if edit_ref_structure_image_path is not None:
                ref_structure_reference = load_external_image_like(source_ref_image, edit_ref_structure_image_path).detach()
            if edit_ref_mask_path is not None:
                ref_mask_image = load_external_mask_like(source_ref_image[:, :1], edit_ref_mask_path)
            elif color_mask_image is not None:
                ref_mask_image = color_mask_image.detach()
            elif M_edit is not None:
                ref_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_ref_image.dtype, device=source_ref_image.device),
                    size=source_ref_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            else:
                ref_mask_image = torch.ones_like(source_ref_image[:, :1])
            if M_edit is not None:
                edit_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_ref_image.dtype, device=source_ref_image.device),
                    size=source_ref_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                ref_mask_image = (ref_mask_image * edit_mask_image).clamp(0.0, 1.0)
            ref_mask_latent = torch.nn.functional.interpolate(
                ref_mask_image,
                size=x_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                save_mask_image(ref_mask_image, os.path.join(mask_output_dir, "ref_edit_mask.png"))
    if completion_clean_delta_scale > 0.0:
        if completion_clean_delta_image_path is None:
            raise ValueError("--completion-clean-delta-scale requires --completion-clean-delta-image")
        with torch.no_grad():
            source_clean_image = decode_latent_to_unit_image(pipe, x_src)
            completion_image = load_external_image_like(
                source_clean_image,
                completion_clean_delta_image_path,
            ).detach()
            completion_clean_target = _encode_unit_image_to_latent(pipe, completion_image).detach().to(
                device=x_src.device,
                dtype=torch.float32,
            )
            if completion_clean_target.shape[-2:] != x_src.shape[-2:]:
                completion_clean_target = torch.nn.functional.interpolate(
                    completion_clean_target,
                    size=x_src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if completion_clean_delta_mask_path is not None:
                completion_mask_image = load_external_mask_like(
                    source_clean_image[:, :1],
                    completion_clean_delta_mask_path,
                )
            elif M_edit is not None:
                completion_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_clean_image.dtype, device=source_clean_image.device),
                    size=source_clean_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            else:
                completion_mask_image = torch.ones_like(source_clean_image[:, :1])
            if M_edit is not None:
                edit_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_clean_image.dtype, device=source_clean_image.device),
                    size=source_clean_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                completion_mask_image = (completion_mask_image * edit_mask_image).clamp(0.0, 1.0)
            completion_clean_mask_latent = torch.nn.functional.interpolate(
                completion_mask_image,
                size=x_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                save_mask_image(
                    completion_mask_image,
                    os.path.join(mask_output_dir, "completion_clean_delta_mask.png"),
                )
    if needs_clip_reward:
        clip_reward = LocalCLIPTextReward(device=device)
        with torch.no_grad():
            source_image = decode_latent_to_unit_image(pipe, x_src, vae_override=grad_vae)
            clip_reference = clip_reward.prepare_reference(
                source_image=source_image,
                source_prompt=edit_text_source_prompt or src_prompt,
                target_prompt=edit_text_target_prompt or tar_prompt,
                mask=M_edit,
            )

    recolor_projection_target_mode = edit_color_clean_projection_target_mode.strip().lower()
    if recolor_projection_target_mode not in {"static", "dynamic"}:
        raise ValueError(
            "Unsupported recolor clean projection target mode: "
            f"{edit_color_clean_projection_target_mode}"
        )
    recolor_projection_refresh_interval = max(0, int(edit_color_clean_projection_refresh_interval))
    recolor_projection_target_latent: torch.Tensor | None = None
    recolor_projection_gate_latent: torch.Tensor | None = None
    recolor_projection_eval_count = 0
    recolor_projection_refresh_count = 0
    recolor_projection_enabled = (
        edit_color_clean_projection_scale > 0.0
        and source_color_reference is not None
        and color_projection_alpha_image is not None
        and color_target is not None
    )
    if recolor_projection_enabled:
        with torch.no_grad():
            recolor_projection_composite_alpha_image = _prepare_recolor_projection_alpha(
                alpha=color_projection_alpha_image,
                size=source_color_reference.shape[-2:],
                alpha_power=edit_color_clean_projection_alpha_power,
                boundary_boost=edit_color_clean_projection_boundary_boost,
                boundary_kernel_size=edit_color_clean_projection_boundary_kernel_size,
                device=source_color_reference.device,
            )
            recolor_projection_gate_alpha_image = _prepare_recolor_projection_alpha(
                alpha=color_mask_image,
                size=source_color_reference.shape[-2:],
                alpha_power=edit_color_clean_projection_alpha_power,
                boundary_boost=edit_color_clean_projection_boundary_boost,
                boundary_kernel_size=edit_color_clean_projection_boundary_kernel_size,
                device=source_color_reference.device,
            )
            if recolor_projection_target_mode == "static":
                recolor_projection_target_latent, recolor_projection_gate_latent = (
                    _build_recolor_projection_latent_target(
                        pipe=pipe,
                        current_image=source_color_reference.to(dtype=torch.float32),
                        source_image=source_color_reference.to(dtype=torch.float32),
                        reference_image=ref_image_reference,
                        target_rgb=color_target[1],
                        composite_alpha_image=recolor_projection_composite_alpha_image,
                        gate_alpha_image=recolor_projection_gate_alpha_image,
                        mode=edit_color_clean_projection_mode,
                        texture_kernel_size=edit_color_clean_projection_texture_kernel_size,
                        luma_texture_scale=edit_color_clean_projection_luma_texture_scale,
                        chroma_texture_scale=edit_color_clean_projection_chroma_texture_scale,
                        composite_mode=edit_color_clean_projection_composite_mode,
                        background_kernel_size=edit_color_clean_projection_background_kernel_size,
                        latent_size=x_src.shape[-2:],
                        latent_device=x_src.device,
                    )
                )
                recolor_projection_eval_count += 1
                recolor_projection_refresh_count += 1

    # --- Controlled reverse ODE ---
    z_t = z_T.clone()
    step_stats: list[dict[str, float | str | None]] = []
    active_edit_step = 0
    adaptive_edit_progress_ema: float | None = None
    clean_debug_active_steps: set[int] = set()
    if clean_estimate_debug_dir is not None:
        active_step_count = min(int(n_max), len(timesteps)) if n_max > 0 else len(timesteps)
        clean_debug_active_steps = {0, max(0, active_step_count // 2), max(0, active_step_count - 1)}

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        sigma = t_i.clamp(min=1e-4)
        alpha_t = get_schedule_value(alpha_schedule, i, len(timesteps), alpha_max)
        if rec_stop_timestep > 0.0 and float(t_i.item()) < rec_stop_timestep:
            alpha_t = 0.0
        beta_t = get_schedule_value(beta_schedule, i, len(timesteps), beta_max)
        if len(timesteps) <= 1:
            edit_stage_progress = 1.0
        else:
            edit_stage_progress = i / float(len(timesteps) - 1)
        identity_break_weight = max(0.0, 1.0 - (edit_stage_progress / max(identity_break_stop, 1e-6)))
        if edit_stage_progress <= target_attract_start:
            target_attract_weight = 0.0
        else:
            target_attract_weight = min(
                1.0,
                (edit_stage_progress - target_attract_start) / max(1.0 - target_attract_start, 1e-6),
            )
        latents_dtype = z_t.dtype

        # Base prior: source-conditioned reference field, closer to h-edit's rec_term.
        source_inject_active = (
            source_inject_enabled
            and source_inject_layer_to > source_inject_layer_from
            and (source_inject_steps <= 0 or active_edit_step < source_inject_steps)
        )
        source_inject_counts = {"all": 0, "q": 0, "k": 0, "v": 0}
        with torch.no_grad():
            if source_inject_active:
                source_latent_for_inject = source_trajectory_by_timestep.get(int(t.item()), x_src)
                v_tar, source_inject_counts = calc_cfg_v_sd3_with_source_qkv_injection(
                    pipe=pipe,
                    latents=z_t,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                    source_latents=source_latent_for_inject.to(latents_dtype),
                    source_prompt_embeds=src_prompt_embeds,
                    source_pooled_prompt_embeds=src_pooled_prompt_embeds,
                    q_strength=source_inject_q_scale,
                    k_strength=source_inject_k_scale,
                    v_strength=source_inject_v_scale,
                    layer_from=source_inject_layer_from,
                    layer_to=source_inject_layer_to,
                    spatial_gate=source_inject_mask,
                )
            else:
                v_tar = calc_cfg_v_sd3(
                    pipe=pipe,
                    latents=z_t,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
            v_src = calc_cfg_v_sd3(
                pipe=pipe,
                latents=z_t,
                negative_prompt_embeds=src_negative_prompt_embeds,
                prompt_embeds=src_prompt_embeds,
                negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                pooled_prompt_embeds=src_pooled_prompt_embeds,
                guidance_scale=base_guidance_scale,
                t=t,
            )

        v_tar = v_tar.to(torch.float32)
        v_src = v_src.to(torch.float32)
        v_base = v_src
        x0_tar = predict_x0_from_linear_rf_path(z_t, v_tar, t_i).to(torch.float32)
        x0_src_step = predict_x0_from_linear_rf_path(z_t, v_src, t_i).to(torch.float32)
        if clean_estimate_debug_dir is not None and active_edit_step in clean_debug_active_steps:
            _save_clean_estimate_debug_images(
                pipe=pipe,
                x0_src_step=x0_src_step,
                x0_tar=x0_tar,
                edit_mask=M_edit,
                output_dir=clean_estimate_debug_dir,
                step_index=active_edit_step,
                t_value=float(t_i.detach().item()),
            )
        base_edit_norm = v_base.norm().item()
        adaptive_edit_progress = 0.0
        adaptive_edit_change_rms = 0.0
        adaptive_edit_target_rms_value = 0.0
        adaptive_edit_target_gap_rms = 0.0
        adaptive_edit_deficit = 0.0
        adaptive_rmsgap_normalized = 0.0
        adaptive_rmsgap_active_gap = 0.0
        adaptive_rmsgap_preserve_gate = 1.0
        adaptive_hybrid_progress_ema_value = 0.0
        adaptive_hybrid_progress_deficit = 0.0
        adaptive_hybrid_progress_gate = 0.0
        adaptive_hybrid_progress_boost = 0.0
        adaptive_preserve_drift = 0.0
        adaptive_preserve_excess = 0.0
        adaptive_edit_weight = 1.0
        adaptive_preserve_weight = 1.0
        adaptive_projection_dot = 0.0
        adaptive_projection_norm = 0.0
        adaptive_clean_conflict_score = 0.0
        adaptive_clean_projection_ratio = 0.0
        adaptive_preserve_drift_after_projection_estimate = 0.0
        adaptive_preserve_clean_correction_norm = 0.0
        removal_controller_norm = 0.0
        removal_fill_norm = 0.0
        removal_suppression_norm = 0.0
        removal_ring_rec_norm = 0.0
        completion_clean_delta_norm = 0.0
        completion_clean_delta_schedule_weight = 1.0
        completion_clean_delta_rms = 0.0
        region_target_transport_norm = 0.0
        region_target_transport_core_beta = 0.0
        region_target_transport_ring_beta = 0.0
        region_target_transport_core_gamma = 0.0
        region_target_outside_lock_norm = 0.0
        region_target_outside_lock_weight = 0.0
        region_target_ring_lock_weight = 0.0
        if adaptive_clean_control:
            edit_diag_gate = None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device)
            preserve_diag_gate = None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device)
            source_clean = x_src.to(torch.float32)
            current_delta = x0_src_step - source_clean
            target_delta = x0_tar - source_clean
            target_gap = x0_tar - x0_src_step
            adaptive_edit_change_rms = float(masked_rms(current_delta, edit_diag_gate).item())
            adaptive_edit_target_rms_value = float(masked_rms(target_delta, edit_diag_gate).item())
            adaptive_edit_target_gap_rms = float(masked_rms(target_gap, edit_diag_gate).item())
            if edit_diag_gate is None:
                edit_progress_num = (current_delta * target_delta).sum()
                edit_progress_den = target_delta.square().sum().clamp_min(1e-8)
            else:
                edit_gate = edit_diag_gate.to(dtype=torch.float32, device=z_t.device)
                if edit_gate.shape[-2:] != current_delta.shape[-2:]:
                    edit_gate = torch.nn.functional.interpolate(
                        edit_gate,
                        size=current_delta.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                edit_gate = edit_gate.clamp(0.0, 1.0)
                edit_progress_num = (current_delta * target_delta * edit_gate).sum()
                edit_progress_den = (target_delta.square() * edit_gate).sum().clamp_min(1e-8)
            adaptive_edit_progress = float((edit_progress_num / edit_progress_den).detach().item())
            adaptive_preserve_drift = float(masked_rms(x0_src_step - source_clean, preserve_diag_gate).item())
            target_delta_reliable = adaptive_edit_target_rms_value > 1e-6
            if adaptive_hybrid_progress_ema_decay > 0.0:
                ema_decay = max(0.0, min(0.99, float(adaptive_hybrid_progress_ema_decay)))
                if adaptive_edit_progress_ema is None:
                    adaptive_edit_progress_ema = adaptive_edit_progress
                else:
                    adaptive_edit_progress_ema = (
                        ema_decay * adaptive_edit_progress_ema
                        + (1.0 - ema_decay) * adaptive_edit_progress
                    )
                adaptive_hybrid_progress_ema_value = float(adaptive_edit_progress_ema)
            else:
                adaptive_edit_progress_ema = adaptive_edit_progress
                adaptive_hybrid_progress_ema_value = adaptive_edit_progress
            if adaptive_rmsgap_mode == "normgate":
                if target_delta_reliable:
                    adaptive_rmsgap_normalized = adaptive_edit_target_gap_rms / max(
                        adaptive_edit_target_rms_value,
                        1e-6,
                    )
                    adaptive_rmsgap_active_gap = max(
                        0.0,
                        adaptive_rmsgap_normalized - float(adaptive_rmsgap_dead_zone),
                    )
                    if (
                        adaptive_rmsgap_preserve_gate_budget > 0.0
                        and adaptive_preserve_drift >= float(adaptive_rmsgap_preserve_gate_budget)
                    ):
                        adaptive_rmsgap_preserve_gate = 0.0
                        adaptive_rmsgap_active_gap = 0.0
                    adaptive_edit_deficit = adaptive_rmsgap_active_gap
            elif (
                adaptive_hybrid_progress_target > 0.0
                and adaptive_hybrid_progress_gain > 0.0
                and adaptive_edit_target_rms > 0.0
            ):
                adaptive_edit_deficit = max(0.0, float(adaptive_edit_target_rms) - adaptive_edit_target_gap_rms)
                preserve_gate_ok = (
                    adaptive_hybrid_preserve_gate_budget <= 0.0
                    or adaptive_preserve_drift < float(adaptive_hybrid_preserve_gate_budget)
                )
                if preserve_gate_ok and target_delta_reliable:
                    adaptive_hybrid_progress_gate = 1.0
                    adaptive_hybrid_progress_deficit = max(
                        0.0,
                        float(adaptive_hybrid_progress_target) - adaptive_hybrid_progress_ema_value,
                    )
                    adaptive_hybrid_progress_boost = (
                        float(adaptive_hybrid_progress_gain) * adaptive_hybrid_progress_deficit
                    )
                    adaptive_edit_deficit += adaptive_hybrid_progress_boost
            elif adaptive_edit_target_progress > 0.0:
                adaptive_edit_deficit = max(0.0, float(adaptive_edit_target_progress) - adaptive_edit_progress)
            elif adaptive_edit_target_rms > 0.0:
                adaptive_edit_deficit = max(0.0, float(adaptive_edit_target_rms) - adaptive_edit_target_gap_rms)
            if adaptive_preserve_drift_budget > 0.0:
                adaptive_preserve_excess = max(0.0, adaptive_preserve_drift - float(adaptive_preserve_drift_budget))
            adaptive_edit_weight = 1.0 + float(adaptive_edit_gain) * adaptive_edit_deficit
            adaptive_preserve_weight = 1.0 + float(adaptive_preserve_gain) * adaptive_preserve_excess
            adaptive_edit_weight = max(
                float(adaptive_edit_weight_min),
                min(float(adaptive_edit_weight_max), adaptive_edit_weight),
            )
            adaptive_preserve_weight = max(
                float(adaptive_preserve_weight_min),
                min(float(adaptive_preserve_weight_max), adaptive_preserve_weight),
            )

        # Reconstruction correction: x_hat_0 -> E_rec -> u_rec surrogate.
        rec_energy_terms = None
        rec_terms = None
        rec_guidance_norm = 0.0
        current_rec_feature_map = None
        source_rec_feature_map = None
        if alpha_t != 0.0 and rec_guidance_scale > 0.0:
            step_key = int(t.item())
            if struct_guidance_scale != 0.0:
                if step_key not in source_feature_maps:
                    with torch.no_grad():
                        source_feature_maps[step_key] = extract_sd3_feature_structure_map(
                            pipe=pipe,
                            x_latent=x_src,
                            prompt_embeds=src_prompt_embeds,
                            pooled_embeds=src_pooled_prompt_embeds,
                            t=t,
                            detach=True,
                        ).to(dtype=torch.float32, device=z_t.device)
                source_rec_feature_map = source_feature_maps[step_key]
                with torch.no_grad():
                    current_rec_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_src_step,
                        prompt_embeds=src_prompt_embeds,
                        pooled_embeds=src_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)

            rec_terms = reconstruction_velocity_surrogate_total(
                x0_pred=x0_src_step,
                x_src=x_src.to(torch.float32),
                t_scalar=t_i,
                M_preserve=None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device),
                current_feature_map=current_rec_feature_map,
                source_feature_map=source_rec_feature_map,
                lambda_latent=1.0,
                lambda_struct=0.0,
                lambda_feature=struct_guidance_scale,
                velocity_conversion_mode=velocity_conversion_mode,
                velocity_t_min=linear_path_t_min,
            )
            rec_energy_terms = reconstruction_energy_total(
                x0_pred=x0_src_step,
                x_src=x_src.to(torch.float32),
                M_preserve=None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device),
                current_feature_map=current_rec_feature_map,
                source_feature_map=source_rec_feature_map,
                lambda_latent=1.0,
                lambda_struct=0.0,
                lambda_feature=struct_guidance_scale,
            )
            v_rec = alpha_t * rec_terms["total"].to(torch.float32)
            if adaptive_clean_control:
                v_rec = adaptive_preserve_weight * v_rec
            rec_guidance_norm = v_rec.norm().item()
        else:
            v_rec = torch.zeros_like(v_base)

        # Editing correction: x_hat_0 -> E_edit -> u_edit surrogate.
        edit_energy_terms = None
        v_edit_terms = None
        edit_guidance_norm = 0.0
        clip_edit_energy_terms = None
        clip_guidance_norm = 0.0
        tv_edit_energy = None
        text_edit_energy_terms = None
        text_guidance_norm = 0.0
        dds_guidance_norm = 0.0
        dds_energy_terms = None
        app_guidance_norm = 0.0
        app_edit_energy = None
        color_guidance_norm = 0.0
        color_edit_energy = None
        local_target_formation_norm = 0.0

        if beta_t != 0.0 and (
            (use_rf_diff_edit_field and edit_hedit_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_region_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_target_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_source_guidance_scale > 0.0)
        ):
            with torch.no_grad():
                target_feature_map = None
                source_feature_map = None
                if use_legacy_edit_surrogates and edit_target_guidance_scale > 0.0:
                    target_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_tar,
                        prompt_embeds=tar_prompt_embeds,
                        pooled_embeds=tar_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)
                if use_legacy_edit_surrogates and edit_source_guidance_scale > 0.0:
                    source_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_src_step,
                        prompt_embeds=src_prompt_embeds,
                        pooled_embeds=src_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)
                v_src_edit = calc_cfg_v_sd3(
                    pipe,
                    z_t.to(latents_dtype),
                    src_negative_prompt_embeds,
                    src_prompt_embeds,
                    src_negative_pooled_prompt_embeds,
                    src_pooled_prompt_embeds,
                    edit_src_cfg_scale,
                    t,
                ).to(torch.float32)
                v_edit_terms = editing_velocity_surrogate_total(
                    base_edit_velocity=(v_tar.to(torch.float32) - v_src_edit),
                    x0_tar=x0_tar,
                    x0_src=x0_src_step,
                    t_scalar=t_i,
                    M_edit=None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device),
                    target_feature_map=target_feature_map,
                    source_feature_map=source_feature_map,
                    lambda_base=edit_hedit_guidance_scale if use_rf_diff_edit_field else 0.0,
                    lambda_anchor=edit_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_region=edit_region_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_target=edit_target_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_source=edit_source_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    velocity_t_min=linear_path_t_min,
                )
                edit_energy_terms = editing_energy_total(
                    x0_tar=x0_tar,
                    x0_src=x0_src_step,
                    M_edit=None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device),
                    target_feature_map=target_feature_map,
                    source_feature_map=source_feature_map,
                    lambda_anchor=edit_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_region=edit_region_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_target=edit_target_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_source=edit_source_guidance_scale if use_legacy_edit_surrogates else 0.0,
                )
            edit_guidance = beta_t * v_edit_terms["total"].to(torch.float32)
            edit_guidance_norm = edit_guidance.norm().item()
        else:
            edit_guidance = torch.zeros_like(v_base)

        clip_start_idx = int(clip_start_timestep * len(timesteps))
        clip_stop_idx = min(int(clip_stop_timestep * len(timesteps)), len(timesteps))
        clip_window_active = clip_start_idx <= i < clip_stop_idx
        clip_guidance = torch.zeros_like(v_base)
        text_guidance = torch.zeros_like(v_base)
        dds_guidance = torch.zeros_like(v_base)
        if (
            beta_t != 0.0
            and clip_reward is not None
            and clip_reference is not None
            and (
                (edit_clip_guidance_scale > 0.0 and clip_window_active)
                or (use_text_diff_edit_field and edit_text_guidance_scale > 0.0)
            )
        ):
            zt_for_edit = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                if edit_field_mode in {"text_diff", "rf_text_diff"}:
                    v_reward = _cfg_v_sd3_with_grad(
                        pipe=pipe,
                        latents=zt_for_edit,
                        negative_prompt_embeds=src_negative_prompt_embeds,
                        prompt_embeds=src_prompt_embeds,
                        negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                        pooled_prompt_embeds=src_pooled_prompt_embeds,
                        guidance_scale=base_guidance_scale,
                        t=t,
                    )
                else:
                    v_reward = _cfg_v_sd3_with_grad(
                        pipe=pipe,
                        latents=zt_for_edit,
                        negative_prompt_embeds=tar_negative_prompt_embeds,
                        prompt_embeds=tar_prompt_embeds,
                        negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                        pooled_prompt_embeds=tar_pooled_prompt_embeds,
                        guidance_scale=tar_guidance_scale,
                        t=t,
                    )
                x0_reward = predict_x0_from_linear_rf_path(zt_for_edit, v_reward, t_i)
                current_image = decode_latent_to_unit_image(pipe, x0_reward, vae_override=grad_vae)

                if use_text_diff_edit_field and edit_text_guidance_scale > 0.0:
                    text_edit_energy_terms = clip_reward.semantic_text_loss(
                        current_image=current_image,
                        reference=clip_reference,
                        core_mask=M_core,
                        subject_mask=M_edit,
                        source_scale=edit_text_source_scale,
                        core_weight=edit_text_core_weight,
                        subject_weight=edit_text_subject_weight,
                    )
                    grad_text = torch.autograd.grad(
                        text_edit_energy_terms["total"],
                        zt_for_edit,
                        retain_graph=edit_clip_guidance_scale > 0.0 and clip_window_active,
                    )[0].detach().to(torch.float32)
                    grad_text = smooth_guidance_field(grad_text, kernel_size=5)
                    grad_text_rms = grad_text.square().mean().sqrt() + 1e-8
                    neg_grad_text = -(grad_text / grad_text_rms)
                    text_guidance = (
                        beta_t
                        * edit_text_guidance_scale
                        * identity_break_weight
                        * neg_grad_text
                    )
                    text_guidance_norm = text_guidance.norm().item()

                if use_auxiliary_edit_fields and edit_clip_guidance_scale > 0.0 and clip_window_active:
                    clip_edit_energy_terms = clip_reward.strong_directional_text_loss(
                        current_image=current_image,
                        reference=clip_reference,
                        mask=M_edit,
                    )
                    if edit_image_tv_scale > 0.0:
                        tv_edit_energy = masked_total_variation(current_image, mask=M_edit)
                    clip_total_energy = clip_edit_energy_terms["total"]
                    if tv_edit_energy is not None:
                        clip_total_energy = clip_total_energy + edit_image_tv_scale * tv_edit_energy
                    grad_clip = torch.autograd.grad(clip_total_energy, zt_for_edit)[0].detach().to(torch.float32)
                    grad_clip = smooth_guidance_field(grad_clip, kernel_size=5)
                    grad_clip_rms = grad_clip.square().mean().sqrt() + 1e-8
                    neg_grad_clip = -(grad_clip / grad_clip_rms)
                    if clip_stop_idx - clip_start_idx <= 1:
                        clip_progress = 0.0
                    else:
                        clip_progress = (i - clip_start_idx) / float(clip_stop_idx - clip_start_idx - 1)
                    clip_window_scale = 0.5 * (
                        1.0 + torch.cos(torch.tensor(clip_progress * 3.141592653589793, device=device))
                    ).item()
                    clip_scale = (
                        beta_t
                        * edit_clip_guidance_scale
                        * clip_window_scale
                        * target_attract_weight
                    )
                    if edit_clip_match_base_scale > 0.0:
                        clip_scale = clip_scale + edit_clip_match_base_scale * base_edit_norm * clip_window_scale
                    clip_guidance = clip_scale * neg_grad_clip
                    clip_guidance_norm = clip_guidance.norm().item()

        if beta_t != 0.0 and use_auxiliary_edit_fields and edit_dds_guidance_scale > 0.0:
            zt_for_dds = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_tar_dds = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_dds,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                x0_tar_dds = predict_x0_from_linear_rf_path(zt_for_dds, v_tar_dds, t_i)
                noise = torch.randn_like(x0_tar_dds)
                noisy_x = scale_noise(scheduler, x0_tar_dds, t, noise=noise)
                v_tar_noisy = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=noisy_x,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                v_src_noisy = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=noisy_x,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=edit_src_cfg_scale,
                    t=t,
                )
                dds_target = 0.5 * (v_tar_noisy - noise).pow(2)
                dds_source = 0.5 * (v_src_noisy - noise).pow(2)
                if M_edit is not None:
                    edit_mask = M_edit.to(dtype=torch.float32, device=z_t.device)
                    dds_target = dds_target * edit_mask
                    dds_source = dds_source * edit_mask
                loss_target = dds_target.mean()
                loss_source = dds_source.mean()
                dds_total = loss_target - edit_dds_source_scale * loss_source
                dds_energy_terms = {
                    "target": loss_target.detach(),
                    "source": loss_source.detach(),
                    "total": dds_total.detach(),
                }
            grad_dds = torch.autograd.grad(dds_total, zt_for_dds)[0].detach().to(torch.float32)
            grad_dds = smooth_guidance_field(grad_dds, kernel_size=5)
            grad_dds_rms = grad_dds.square().mean().sqrt() + 1e-8
            neg_grad_dds = -(grad_dds / grad_dds_rms)
            dds_guidance = beta_t * edit_dds_guidance_scale * neg_grad_dds
            dds_guidance_norm = dds_guidance.norm().item()

        app_guidance = torch.zeros_like(v_base)
        if beta_t != 0.0 and use_auxiliary_edit_fields and edit_app_guidance_scale > 0.0 and M_edit is not None:
            zt_for_app = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_tar_app = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_app,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                x0_tar_app = predict_x0_from_linear_rf_path(zt_for_app, v_tar_app, t_i)
                src_changed_words = [w for w in src_prompt.lower().split() if w not in tar_prompt.lower().split()]
                tar_changed_words = [w for w in tar_prompt.lower().split() if w not in src_prompt.lower().split()]
                step_key = int(t.item())
                if step_key not in source_attention_maps:
                    with torch.no_grad():
                        source_attention_maps[step_key] = extract_prompt_attention_map(
                            pipe=pipe,
                            x_latent=x0_src_step,
                            prompt=src_prompt,
                            prompt_embeds=src_prompt_embeds,
                            pooled_embeds=src_pooled_prompt_embeds,
                            t=t,
                            token_words=src_changed_words,
                            detach=True,
                        ).to(dtype=torch.float32, device=z_t.device)
                source_app_attention = source_attention_maps[step_key]
                current_app_attention = extract_prompt_attention_map(
                    pipe=pipe,
                    x_latent=x0_tar_app,
                    prompt=tar_prompt,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_embeds=tar_pooled_prompt_embeds,
                    t=t,
                    token_words=tar_changed_words,
                    detach=False,
                ).to(dtype=torch.float32, device=z_t.device)
                subject_gate = M_edit.to(dtype=torch.float32, device=z_t.device)
                if M_core is not None:
                    core_gate = M_core.to(dtype=torch.float32, device=z_t.device)
                    subject_gate = (subject_gate - core_gate).clamp_min(0.0)
                app_diff = (current_app_attention - source_app_attention) * subject_gate
                app_edit_energy = 0.5 * app_diff.pow(2).mean()
            grad_app = torch.autograd.grad(app_edit_energy, zt_for_app)[0].detach().to(torch.float32)
            grad_app = smooth_guidance_field(grad_app, kernel_size=5)
            grad_app_rms = grad_app.square().mean().sqrt() + 1e-8
            neg_grad_app = -(grad_app / grad_app_rms)
            app_guidance = beta_t * edit_app_guidance_scale * neg_grad_app
            app_guidance_norm = app_guidance.norm().item()

        color_guidance = torch.zeros_like(v_base)
        ref_guidance = torch.zeros_like(v_base)
        if (
            beta_t != 0.0
            and edit_color_guidance_scale > 0.0
            and M_edit is not None
            and color_target is not None
            and color_mask_image is not None
        ):
            zt_for_color = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_color_src = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_color,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=base_guidance_scale,
                    t=t,
                )
                x0_color = predict_x0_from_linear_rf_path(zt_for_color, v_color_src, t_i)
                color_image = decode_latent_to_unit_image(pipe, x0_color, vae_override=grad_vae)
                if edit_color_texture_preserve_scale > 0.0 or edit_color_boundary_chroma_scale > 0.0:
                    color_edit_energy = masked_recolor_texture_boundary_loss(
                        color_image,
                        source_color_reference,
                        color_target[1],
                        color_mask_image,
                        target_chroma_scale=edit_color_target_chroma_scale,
                        luma_preserve_scale=edit_color_luma_preserve_scale,
                        luma_gradient_preserve_scale=edit_color_luma_gradient_preserve_scale,
                        texture_preserve_scale=edit_color_texture_preserve_scale,
                        texture_kernel_size=edit_color_texture_kernel_size,
                        boundary_chroma_scale=edit_color_boundary_chroma_scale,
                        boundary_kernel_size=edit_color_boundary_kernel_size,
                    )
                else:
                    color_edit_energy = masked_chroma_luma_loss(
                        color_image,
                        source_color_reference,
                        color_target[1],
                        color_mask_image,
                        target_chroma_scale=edit_color_target_chroma_scale,
                        luma_preserve_scale=edit_color_luma_preserve_scale,
                        luma_gradient_preserve_scale=edit_color_luma_gradient_preserve_scale,
                    )
            grad_color = torch.autograd.grad(color_edit_energy, zt_for_color)[0].detach().to(torch.float32)
            grad_color = smooth_guidance_field(grad_color, kernel_size=int(edit_color_smooth_kernel))
            grad_color_rms = grad_color.square().mean().sqrt() + 1e-8
            color_guidance = beta_t * edit_color_guidance_scale * (grad_color / grad_color_rms)
            color_guidance_norm = color_guidance.norm().item()

        ref_edit_energy = None
        ref_guidance_norm = 0.0
        ref_guidance_pre_stabilize_norm = 0.0
        ref_guidance_schedule_weight = 1.0
        if (
            beta_t != 0.0
            and edit_ref_guidance_scale > 0.0
            and ref_image_reference is not None
            and ref_mask_image is not None
        ):
            zt_for_ref = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_ref_src = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_ref,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=base_guidance_scale,
                    t=t,
                )
                x0_ref = predict_x0_from_linear_rf_path(zt_for_ref, v_ref_src, t_i)
                ref_current_image = decode_latent_to_unit_image(pipe, x0_ref, vae_override=grad_vae)
                ref_edit_energy = masked_reference_image_loss(
                    ref_current_image,
                    ref_image_reference,
                    ref_mask_image,
                    structure_image=ref_structure_reference,
                    luma_preserve_scale=edit_ref_luma_preserve_scale,
                    gradient_preserve_scale=edit_ref_gradient_preserve_scale,
                    darkness_guard_scale=edit_ref_darkness_guard_scale,
                    darkness_guard_margin=edit_ref_darkness_guard_margin,
                    chroma_mode=edit_ref_chroma_mode,
                    chroma_magnitude_scale=edit_ref_chroma_magnitude_scale,
                )
            grad_ref = torch.autograd.grad(ref_edit_energy, zt_for_ref)[0].detach().to(torch.float32)
            grad_ref = smooth_guidance_field(grad_ref, kernel_size=int(edit_ref_smooth_kernel))
            grad_ref = suppress_low_frequency_guidance(
                grad_ref,
                kernel_size=int(edit_ref_lowfreq_suppress_kernel),
                strength=edit_ref_lowfreq_suppress_scale,
            )
            grad_ref_rms = grad_ref.square().mean().sqrt() + 1e-8
            ref_guidance = beta_t * edit_ref_guidance_scale * (grad_ref / grad_ref_rms)
            progress_t = 1.0 - float(t_i.detach().item())
            ref_guidance_schedule_weight = ramp_schedule_from_progress(
                progress_t,
                start=edit_ref_schedule_start,
                stop=edit_ref_schedule_stop,
                power=edit_ref_schedule_power,
            )
            ref_guidance = ref_guidance * ref_guidance_schedule_weight
            ref_guidance_pre_stabilize_norm = ref_guidance.norm().item()
            struct_anchor = v_base + v_rec
            ref_guidance = remove_opposing_guidance_component(
                ref_guidance,
                struct_anchor,
                strength=edit_ref_project_struct_conflict,
            )
            ref_guidance = limit_guidance_rms_relative_to_anchor(
                ref_guidance,
                struct_anchor,
                max_ratio=edit_ref_max_struct_rms_ratio,
            )
            ref_guidance_norm = ref_guidance.norm().item()

        rec_gate = None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device)
        edit_gate = None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device)
        core_gate = None if M_core is None else M_core.to(dtype=torch.float32, device=z_t.device)
        if rec_gate is not None:
            v_rec = v_rec * rec_gate
            rec_guidance_norm = v_rec.norm().item()
        if (
            adaptive_clean_control
            and adaptive_preserve_clean_correction_scale > 0.0
            and adaptive_preserve_excess > 0.0
        ):
            preserve_error_for_correction = x0_src_step - x_src.to(torch.float32)
            preserve_clean_correction = preserve_error_for_correction / t_i.to(torch.float32).clamp_min(
                float(linear_path_t_min)
            )
            if rec_gate is not None:
                preserve_clean_gate = rec_gate.to(dtype=torch.float32, device=z_t.device)
                if preserve_clean_gate.shape[-2:] != preserve_clean_correction.shape[-2:]:
                    preserve_clean_gate = torch.nn.functional.interpolate(
                        preserve_clean_gate,
                        size=preserve_clean_correction.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                preserve_clean_correction = preserve_clean_correction * preserve_clean_gate.clamp(0.0, 1.0)
            preserve_clean_correction = (
                float(adaptive_preserve_clean_correction_scale)
                * float(adaptive_preserve_excess)
                * preserve_clean_correction
            )
            v_rec = v_rec + preserve_clean_correction
            adaptive_preserve_clean_correction_norm = float(preserve_clean_correction.norm().item())
            rec_guidance_norm = v_rec.norm().item()
        if edit_gate is not None:
            if core_gate is not None:
                subject_ring = (edit_gate - core_gate).clamp_min(0.0)
                spatial_edit_weight = edit_core_scale * core_gate + edit_subject_scale * subject_ring
            else:
                spatial_edit_weight = edit_gate
            edit_guidance = edit_guidance * spatial_edit_weight
            clip_guidance = clip_guidance * spatial_edit_weight
            text_guidance = text_guidance * spatial_edit_weight
            dds_guidance = dds_guidance * spatial_edit_weight
            app_guidance = app_guidance * edit_gate
            color_gate = edit_gate if color_mask_latent is None else color_mask_latent.to(dtype=torch.float32, device=z_t.device)
            color_guidance = color_guidance * color_gate
            ref_gate = edit_gate if ref_mask_latent is None else ref_mask_latent.to(dtype=torch.float32, device=z_t.device)
            ref_guidance = ref_guidance * ref_gate
            edit_guidance_norm = edit_guidance.norm().item()
            clip_guidance_norm = clip_guidance.norm().item()
            text_guidance_norm = text_guidance.norm().item()
            dds_guidance_norm = dds_guidance.norm().item()
            app_guidance_norm = app_guidance.norm().item()
            color_guidance_norm = color_guidance.norm().item()
            ref_guidance_norm = ref_guidance.norm().item()

        v_edit_total = (
            edit_guidance
            + clip_guidance
            + text_guidance
            + dds_guidance
            + app_guidance
            + color_guidance
            + ref_guidance
        )
        recolor_clean_projection_norm = 0.0
        recolor_clean_projection_rms = 0.0
        recolor_clean_projection_refresh = 0.0
        if recolor_projection_enabled:
            should_refresh_projection = recolor_projection_target_latent is None
            if (
                recolor_projection_target_mode == "dynamic"
                and recolor_projection_refresh_interval > 0
                and active_edit_step % recolor_projection_refresh_interval == 0
            ):
                should_refresh_projection = True
            if recolor_projection_target_mode == "dynamic" and recolor_projection_refresh_interval <= 0:
                should_refresh_projection = True
            if should_refresh_projection:
                with torch.no_grad():
                    if recolor_projection_target_mode == "dynamic":
                        projection_current_image = decode_latent_to_unit_image(pipe, x0_src_step)
                    else:
                        projection_current_image = source_color_reference.to(dtype=torch.float32)
                    projection_composite_alpha_image = _prepare_recolor_projection_alpha(
                        alpha=color_projection_alpha_image,
                        size=projection_current_image.shape[-2:],
                        alpha_power=edit_color_clean_projection_alpha_power,
                        boundary_boost=edit_color_clean_projection_boundary_boost,
                        boundary_kernel_size=edit_color_clean_projection_boundary_kernel_size,
                        device=projection_current_image.device,
                    )
                    projection_gate_alpha_image = _prepare_recolor_projection_alpha(
                        alpha=color_mask_image,
                        size=projection_current_image.shape[-2:],
                        alpha_power=edit_color_clean_projection_alpha_power,
                        boundary_boost=edit_color_clean_projection_boundary_boost,
                        boundary_kernel_size=edit_color_clean_projection_boundary_kernel_size,
                        device=projection_current_image.device,
                    )
                    recolor_projection_target_latent, recolor_projection_gate_latent = (
                        _build_recolor_projection_latent_target(
                            pipe=pipe,
                            current_image=projection_current_image,
                            source_image=source_color_reference.to(
                                dtype=torch.float32,
                                device=projection_current_image.device,
                            ),
                            reference_image=ref_image_reference,
                            target_rgb=color_target[1],
                            composite_alpha_image=projection_composite_alpha_image,
                            gate_alpha_image=projection_gate_alpha_image,
                            mode=edit_color_clean_projection_mode,
                            texture_kernel_size=edit_color_clean_projection_texture_kernel_size,
                            luma_texture_scale=edit_color_clean_projection_luma_texture_scale,
                            chroma_texture_scale=edit_color_clean_projection_chroma_texture_scale,
                            composite_mode=edit_color_clean_projection_composite_mode,
                            background_kernel_size=edit_color_clean_projection_background_kernel_size,
                            latent_size=x0_src_step.shape[-2:],
                            latent_device=z_t.device,
                        )
                    )
                recolor_projection_eval_count += 1
                recolor_projection_refresh_count += 1
                recolor_clean_projection_refresh = 1.0
            projection_gate = recolor_projection_gate_latent.to(dtype=torch.float32, device=z_t.device)
            projection_clean_delta = recolor_projection_target_latent.to(dtype=torch.float32, device=z_t.device) - x0_src_step.to(
                torch.float32
            )
            if edit_color_clean_projection_delta_lowpass_kernel > 1:
                projection_clean_delta = _spatial_low_pass(
                    projection_clean_delta,
                    edit_color_clean_projection_delta_lowpass_kernel,
                )
            projection_clean_delta = projection_clean_delta * projection_gate
            projection_velocity = clean_delta_to_velocity(
                projection_clean_delta,
                t_i,
                eps=float(linear_path_t_min),
            )
            recolor_clean_projection_guidance = (
                beta_t * float(edit_color_clean_projection_scale) * projection_velocity
            )
            v_edit_total = v_edit_total + recolor_clean_projection_guidance
            recolor_clean_projection_norm = float(recolor_clean_projection_guidance.norm().item())
            recolor_clean_projection_rms = float(
                masked_rms(projection_clean_delta, projection_gate).item()
            )
        if (
            completion_clean_delta_scale > 0.0
            and completion_clean_target is not None
            and edit_gate is not None
        ):
            completion_gate = (
                edit_gate
                if completion_clean_mask_latent is None
                else completion_clean_mask_latent.to(dtype=torch.float32, device=z_t.device)
            )
            if completion_gate.shape[-2:] != x0_src_step.shape[-2:]:
                completion_gate = torch.nn.functional.interpolate(
                    completion_gate,
                    size=x0_src_step.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            completion_gate = completion_gate.clamp(0.0, 1.0)
            completion_target = completion_clean_target.to(dtype=torch.float32, device=z_t.device)
            if completion_target.shape[-2:] != x0_src_step.shape[-2:]:
                completion_target = torch.nn.functional.interpolate(
                    completion_target,
                    size=x0_src_step.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            completion_delta = (completion_target - x0_src_step) * completion_gate
            completion_velocity = clean_delta_to_velocity(
                completion_delta,
                t_i,
                eps=float(linear_path_t_min),
            )
            progress_t = 1.0 - float(t_i.detach().item())
            completion_clean_delta_schedule_weight = ramp_schedule_from_progress(
                progress_t,
                start=completion_clean_delta_schedule_start,
                stop=completion_clean_delta_schedule_stop,
                power=completion_clean_delta_schedule_power,
            )
            completion_clean_guidance = (
                beta_t
                * float(completion_clean_delta_scale)
                * completion_clean_delta_schedule_weight
                * completion_velocity
            )
            v_edit_total = v_edit_total + completion_clean_guidance
            completion_clean_delta_norm = float(completion_clean_guidance.norm().item())
            completion_clean_delta_rms = float(masked_rms(completion_delta, completion_gate).item())
        if (
            edit_local_target_guidance_scale > 0.0
            and edit_local_target_prompt
            and local_target_prompt_embeds is not None
            and edit_gate is not None
        ):
            with torch.no_grad():
                v_local_target = calc_cfg_v_sd3(
                    pipe,
                    z_t.to(latents_dtype),
                    local_target_negative_prompt_embeds,
                    local_target_prompt_embeds,
                    local_target_negative_pooled_prompt_embeds,
                    local_target_pooled_prompt_embeds,
                    edit_local_target_cfg_scale,
                    t,
                ).to(torch.float32)
                x0_local_target = predict_x0_from_linear_rf_path(z_t, v_local_target, t_i).to(torch.float32)
                formation_gate = edit_gate.to(dtype=torch.float32, device=z_t.device)
                if formation_gate.shape[-2:] != v_base.shape[-2:]:
                    formation_gate = torch.nn.functional.interpolate(
                        formation_gate,
                        size=v_base.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if core_gate is not None:
                    formation_core = core_gate.to(dtype=torch.float32, device=z_t.device)
                    if formation_core.shape[-2:] != v_base.shape[-2:]:
                        formation_core = torch.nn.functional.interpolate(
                            formation_core,
                            size=v_base.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    formation_ring = (formation_gate - formation_core).clamp(0.0, 1.0)
                    formation_gate = (formation_core + 0.35 * formation_ring).clamp(0.0, 1.0)
                else:
                    formation_gate = formation_gate.clamp(0.0, 1.0)
                t_value = float(t_i.detach().item())
                formation_schedule = max(0.0, min(1.0, (t_value - 0.12) / 0.35))
                local_delta = (x0_local_target - x0_src_step) * formation_gate
                local_target_formation = clean_delta_to_velocity(
                    local_delta,
                    t_i,
                    eps=float(linear_path_t_min),
                )
                local_target_formation = (
                    beta_t
                    * float(edit_local_target_guidance_scale)
                    * formation_schedule
                    * local_target_formation
                )
            v_edit_total = v_edit_total + local_target_formation
            local_target_formation_norm = float(local_target_formation.norm().item())
        if region_target_transport_scale > 0.0 and edit_gate is not None:
            t_value = float(t_i.detach().item())
            if t_value > 0.65:
                core_beta_base = 1.0
                core_gamma_base = 0.5
                ring_beta_base = 0.25
            elif t_value > 0.35:
                core_beta_base = 0.8
                core_gamma_base = 0.25
                ring_beta_base = 0.35
            else:
                core_beta_base = 0.35
                core_gamma_base = 0.0
                ring_beta_base = 0.15
            region_target_transport_core_beta = max(
                0.0,
                min(1.0, float(region_target_transport_scale) * core_beta_base),
            )
            region_target_transport_core_gamma = max(
                0.0,
                float(region_target_transport_scale) * core_gamma_base,
            )
            region_target_transport_ring_beta = max(
                0.0,
                min(1.0, float(region_target_transport_scale) * ring_beta_base),
            )
            transport_core_gate = core_gate if core_gate is not None else edit_gate
            transport_edit_gate = edit_gate
            if transport_core_gate.shape[-2:] != v_base.shape[-2:]:
                transport_core_gate = torch.nn.functional.interpolate(
                    transport_core_gate,
                    size=v_base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if transport_edit_gate.shape[-2:] != v_base.shape[-2:]:
                transport_edit_gate = torch.nn.functional.interpolate(
                    transport_edit_gate,
                    size=v_base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            transport_core_gate = transport_core_gate.clamp(0.0, 1.0)
            transport_edit_gate = transport_edit_gate.clamp(0.0, 1.0)
            transport_ring_gate = (transport_edit_gate - transport_core_gate).clamp(0.0, 1.0)
            target_delta_velocity = v_tar - v_src
            region_target_transport = target_delta_velocity * (
                (region_target_transport_core_beta + region_target_transport_core_gamma)
                * transport_core_gate
                + region_target_transport_ring_beta * transport_ring_gate
            )
            v_edit_total = v_edit_total + beta_t * region_target_transport
            region_target_transport_norm = float((beta_t * region_target_transport).norm().item())
        if adaptive_clean_control:
            v_edit_total = adaptive_edit_weight * v_edit_total
            preserve_projection_gate = None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device)
            preserve_error = x0_src_step - x_src.to(torch.float32)
            clean_edit_effect = -t_i.to(torch.float32) * v_edit_total
            clean_effect_after_projection = clean_edit_effect
            if preserve_projection_gate is not None:
                clean_gate = preserve_projection_gate.to(dtype=torch.float32, device=z_t.device)
                if clean_gate.shape[-2:] != preserve_error.shape[-2:]:
                    clean_gate = torch.nn.functional.interpolate(
                        clean_gate,
                        size=preserve_error.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                clean_gate = clean_gate.clamp(0.0, 1.0)
                preserve_error_eval = preserve_error * clean_gate
                clean_effect_eval = clean_edit_effect * clean_gate
            else:
                preserve_error_eval = preserve_error
                clean_effect_eval = clean_edit_effect
            clean_conflict_dot = (preserve_error_eval * clean_effect_eval).sum()
            adaptive_clean_conflict_score = float(clean_conflict_dot.detach().item())
            if float(clean_conflict_dot.detach().item()) > 0.0:
                preserve_error_sq = preserve_error_eval.square().sum().clamp_min(1e-8)
                destructive_effect = (clean_conflict_dot / preserve_error_sq) * preserve_error_eval
                projection_strength = max(0.0, min(1.0, float(adaptive_projection_scale)))
                scaled_destructive_effect = projection_strength * destructive_effect
                adaptive_clean_projection_ratio = float(
                    (
                        scaled_destructive_effect.norm()
                        / clean_effect_eval.norm().clamp_min(1e-8)
                    ).detach().item()
                )
                clean_effect_after_projection = clean_edit_effect - scaled_destructive_effect
                adaptive_projection_dot = float(clean_conflict_dot.detach().item())
                clean_projection_delta = clean_effect_after_projection - clean_edit_effect
                velocity_projection_delta = -clean_projection_delta / t_i.to(torch.float32).clamp_min(1e-6)
                adaptive_projection_norm = float(velocity_projection_delta.norm().detach().item())
            preserve_after_estimate = preserve_error + clean_effect_after_projection
            adaptive_preserve_drift_after_projection_estimate = float(
                masked_rms(preserve_after_estimate, preserve_projection_gate).item()
            )
            v_edit_total = -clean_effect_after_projection / t_i.to(torch.float32).clamp_min(1e-6)
        removal_mode = (removal_controller_mode or "none").strip().lower()
        removal_active = (
            removal_mode not in {"", "none", "off"}
            and (support_edit_operation or "auto").strip().lower() == "remove_object"
            and M_edit is not None
        )
        if removal_active:
            remove_gate = M_edit.to(dtype=torch.float32, device=z_t.device).clamp(0.0, 1.0)
            if remove_gate.shape[-2:] != z_t.shape[-2:]:
                remove_gate = torch.nn.functional.interpolate(
                    remove_gate,
                    size=z_t.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            denom = t_i.to(torch.float32).clamp_min(float(linear_path_t_min))
            x0_noobj = x0_tar
            x0_cur = x0_src_step
            u_fill = -remove_gate * (x0_noobj - x0_cur) / denom
            u_suppress = remove_gate * (v_tar - v_src)
            ring_gate = remove_gate
            kernel = 7
            wide_gate = torch.nn.functional.max_pool2d(
                remove_gate.float(),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            ).to(remove_gate.dtype)
            ring_gate = (wide_gate - remove_gate).clamp(0.0, 1.0)
            u_ring_rec = ring_gate * v_rec
            removal_guidance = beta_t * (
                float(removal_fill_scale) * u_fill
                + float(removal_suppression_scale) * u_suppress
                + float(removal_ring_rec_scale) * u_ring_rec
            )
            v_edit_total = v_edit_total + removal_guidance
            removal_controller_norm = float(removal_guidance.norm().item())
            removal_fill_norm = float((beta_t * float(removal_fill_scale) * u_fill).norm().item())
            removal_suppression_norm = float((beta_t * float(removal_suppression_scale) * u_suppress).norm().item())
            removal_ring_rec_norm = float((beta_t * float(removal_ring_rec_scale) * u_ring_rec).norm().item())
        if edit_bound_scale > 0.0:
            edit_rms = v_edit_total.square().mean().sqrt().clamp_min(1e-8)
            v_edit_total = beta_t * edit_bound_scale * (v_edit_total / edit_rms)
        v_total = v_base + v_rec + v_edit_total
        total_velocity_norm = v_total.norm().item()
        next_z_t = z_t.to(torch.float32) + (t_im1 - t_i).to(torch.float32) * v_total
        trajectory_preserve_norm = 0.0
        trajectory_preserve_weight = 0.0
        if region_target_outside_lock_scale > 0.0 and edit_gate is not None:
            next_timestep_key = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0
            source_next = source_trajectory_by_timestep.get(next_timestep_key)
            if source_next is not None:
                t_value = float(t_i.detach().item())
                if t_value > 0.65:
                    outside_lock_base = 0.5
                    alpha_ring = 0.6
                elif t_value > 0.35:
                    outside_lock_base = 0.75
                    alpha_ring = 0.5
                else:
                    outside_lock_base = 0.9
                    alpha_ring = 0.3
                source_next = source_next.to(device=next_z_t.device, dtype=torch.float32)
                lock_edit_gate = edit_gate.to(device=next_z_t.device, dtype=torch.float32)
                lock_core_gate = core_gate if core_gate is not None else edit_gate
                lock_core_gate = lock_core_gate.to(device=next_z_t.device, dtype=torch.float32)
                if lock_edit_gate.shape[-2:] != next_z_t.shape[-2:]:
                    lock_edit_gate = torch.nn.functional.interpolate(
                        lock_edit_gate,
                        size=next_z_t.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    lock_core_gate = torch.nn.functional.interpolate(
                        lock_core_gate,
                        size=next_z_t.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                lock_edit_gate = lock_edit_gate.clamp(0.0, 1.0)
                lock_core_gate = lock_core_gate.clamp(0.0, 1.0)
                lock_ring_gate = (lock_edit_gate - lock_core_gate).clamp(0.0, 1.0)
                outside_gate = (1.0 - lock_edit_gate).clamp(0.0, 1.0)
                region_target_outside_lock_weight = max(
                    0.0,
                    min(1.0, float(region_target_outside_lock_scale) * outside_lock_base),
                )
                region_target_ring_lock_weight = max(
                    0.0,
                    min(1.0, float(region_target_outside_lock_scale) * (1.0 - alpha_ring)),
                )
                lock_gate = (
                    region_target_outside_lock_weight * outside_gate
                    + region_target_ring_lock_weight * lock_ring_gate
                ).clamp(0.0, 1.0)
                outside_lock_delta = lock_gate * (source_next - next_z_t)
                next_z_t = next_z_t + outside_lock_delta
                region_target_outside_lock_norm = float(outside_lock_delta.norm().item())
        if trajectory_preserve_scale > 0.0 or trajectory_subject_preserve_scale > 0.0:
            next_timestep_key = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0
            source_next = source_trajectory_by_timestep.get(next_timestep_key)
            if source_next is not None:
                trajectory_gate = torch.zeros(
                    next_z_t.shape[0],
                    1,
                    next_z_t.shape[-2],
                    next_z_t.shape[-1],
                    device=next_z_t.device,
                    dtype=torch.float32,
                )
                if rec_gate is None:
                    preserve_gate = torch.ones_like(trajectory_gate)
                else:
                    preserve_gate = rec_gate.to(device=next_z_t.device, dtype=torch.float32)
                    if preserve_gate.shape[-2:] != next_z_t.shape[-2:]:
                        preserve_gate = torch.nn.functional.interpolate(
                            preserve_gate,
                            size=next_z_t.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                if trajectory_preserve_scale > 0.0:
                    preserve_traj_scale = float(trajectory_preserve_scale)
                    if adaptive_clean_control:
                        preserve_traj_scale *= float(adaptive_preserve_weight)
                    trajectory_gate = trajectory_gate + preserve_traj_scale * preserve_gate
                if trajectory_subject_preserve_scale > 0.0 and edit_gate is not None and core_gate is not None:
                    subject_gate_for_traj = edit_gate.to(device=next_z_t.device, dtype=torch.float32)
                    core_gate_for_traj = core_gate.to(device=next_z_t.device, dtype=torch.float32)
                    if subject_gate_for_traj.shape[-2:] != next_z_t.shape[-2:]:
                        subject_gate_for_traj = torch.nn.functional.interpolate(
                            subject_gate_for_traj,
                            size=next_z_t.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        core_gate_for_traj = torch.nn.functional.interpolate(
                            core_gate_for_traj,
                            size=next_z_t.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    subject_ring_for_traj = (subject_gate_for_traj - core_gate_for_traj).clamp_min(0.0)
                    trajectory_gate = (
                        trajectory_gate
                        + float(trajectory_subject_preserve_scale) * subject_ring_for_traj
                    )
                trajectory_weight = float(trajectory_gate.max().item())
                source_next = source_next.to(device=next_z_t.device, dtype=torch.float32)
                trajectory_delta = trajectory_gate.clamp(0.0, 1.0) * (source_next - next_z_t)
                next_z_t = next_z_t + trajectory_delta
                trajectory_preserve_norm = trajectory_delta.norm().item()
                trajectory_preserve_weight = trajectory_weight
        edit_base_for_cos = torch.zeros_like(v_base) if v_edit_terms is None else v_edit_terms["base"]
        edit_anchor_for_cos = torch.zeros_like(v_base) if v_edit_terms is None else v_edit_terms["anchor"]
        edit_region_for_cos = torch.zeros_like(v_base) if v_edit_terms is None else v_edit_terms["region"]
        step_stat = {
            "step": float(i),
            "t": float(t_i.item()),
            "sigma": float(sigma.item()),
            "alpha_max": float(alpha_max),
            "alpha_t": float(alpha_t),
            "beta_max": float(beta_max),
            "beta_t": float(beta_t),
            "adaptive_clean_control": bool(adaptive_clean_control),
            "adaptive_edit_target_progress": float(adaptive_edit_target_progress),
            "adaptive_edit_target_rms": float(adaptive_edit_target_rms),
            "adaptive_rmsgap_mode": adaptive_rmsgap_mode,
            "adaptive_rmsgap_dead_zone": float(adaptive_rmsgap_dead_zone),
            "adaptive_rmsgap_preserve_gate_budget": float(adaptive_rmsgap_preserve_gate_budget),
            "adaptive_hybrid_progress_target": float(adaptive_hybrid_progress_target),
            "adaptive_hybrid_progress_gain": float(adaptive_hybrid_progress_gain),
            "adaptive_hybrid_progress_ema_decay": float(adaptive_hybrid_progress_ema_decay),
            "adaptive_hybrid_preserve_gate_budget": float(adaptive_hybrid_preserve_gate_budget),
            "adaptive_preserve_drift_budget": float(adaptive_preserve_drift_budget),
            "adaptive_edit_gain": float(adaptive_edit_gain),
            "adaptive_preserve_gain": float(adaptive_preserve_gain),
            "adaptive_edit_weight_min": float(adaptive_edit_weight_min),
            "adaptive_edit_weight_max": float(adaptive_edit_weight_max),
            "adaptive_preserve_weight_min": float(adaptive_preserve_weight_min),
            "adaptive_preserve_weight_max": float(adaptive_preserve_weight_max),
            "adaptive_projection_scale": float(adaptive_projection_scale),
            "adaptive_preserve_clean_correction_scale": float(adaptive_preserve_clean_correction_scale),
            "removal_controller_mode": removal_controller_mode,
            "removal_fill_scale": float(removal_fill_scale),
            "removal_suppression_scale": float(removal_suppression_scale),
            "removal_ring_rec_scale": float(removal_ring_rec_scale),
            "region_target_transport_scale": float(region_target_transport_scale),
            "region_target_outside_lock_scale": float(region_target_outside_lock_scale),
            "region_target_transport_norm": float(region_target_transport_norm),
            "local_target_formation_norm": float(local_target_formation_norm),
            "edit_local_target_guidance_scale": float(edit_local_target_guidance_scale),
            "edit_local_target_cfg_scale": float(edit_local_target_cfg_scale),
            "region_target_transport_core_beta": float(region_target_transport_core_beta),
            "region_target_transport_ring_beta": float(region_target_transport_ring_beta),
            "region_target_transport_core_gamma": float(region_target_transport_core_gamma),
            "region_target_outside_lock_norm": float(region_target_outside_lock_norm),
            "region_target_outside_lock_weight": float(region_target_outside_lock_weight),
            "region_target_ring_lock_weight": float(region_target_ring_lock_weight),
            "adaptive_edit_progress": float(adaptive_edit_progress),
            "adaptive_edit_change_rms": float(adaptive_edit_change_rms),
            "adaptive_edit_target_rms_value": float(adaptive_edit_target_rms_value),
            "adaptive_edit_target_gap_rms": float(adaptive_edit_target_gap_rms),
            "adaptive_edit_deficit": float(adaptive_edit_deficit),
            "adaptive_rmsgap_normalized": float(adaptive_rmsgap_normalized),
            "adaptive_rmsgap_active_gap": float(adaptive_rmsgap_active_gap),
            "adaptive_rmsgap_preserve_gate": float(adaptive_rmsgap_preserve_gate),
            "adaptive_hybrid_progress_ema": float(adaptive_hybrid_progress_ema_value),
            "adaptive_hybrid_progress_deficit": float(adaptive_hybrid_progress_deficit),
            "adaptive_hybrid_progress_gate": float(adaptive_hybrid_progress_gate),
            "adaptive_hybrid_progress_boost": float(adaptive_hybrid_progress_boost),
            "adaptive_preserve_drift": float(adaptive_preserve_drift),
            "adaptive_preserve_excess": float(adaptive_preserve_excess),
            "adaptive_edit_weight": float(adaptive_edit_weight),
            "adaptive_preserve_weight": float(adaptive_preserve_weight),
            "adaptive_projection_dot": float(adaptive_projection_dot),
            "adaptive_projection_norm": float(adaptive_projection_norm),
            "adaptive_clean_conflict_score": float(adaptive_clean_conflict_score),
            "adaptive_clean_projection_ratio": float(adaptive_clean_projection_ratio),
            "adaptive_preserve_clean_correction_norm": float(adaptive_preserve_clean_correction_norm),
            "adaptive_preserve_drift_before_projection": float(adaptive_preserve_drift),
            "adaptive_preserve_drift_after_projection_estimate": float(
                adaptive_preserve_drift_after_projection_estimate
            ),
            "velocity_conversion_mode": velocity_conversion_mode,
            "src_guidance_scale": float(src_guidance_scale),
            "inversion_guidance_scale": float(inversion_guidance_scale),
            "base_guidance_scale": float(base_guidance_scale),
            "tar_guidance_scale": float(tar_guidance_scale),
            "edit_src_cfg_scale": float(edit_src_cfg_scale),
            "linear_path_t_min": float(linear_path_t_min),
            "rec_stop_timestep": float(rec_stop_timestep),
            "trajectory_preserve_scale": float(trajectory_preserve_scale),
            "trajectory_subject_preserve_scale": float(trajectory_subject_preserve_scale),
            "source_inject_q_scale": float(source_inject_q_scale),
            "source_inject_k_scale": float(source_inject_k_scale),
            "source_inject_v_scale": float(source_inject_v_scale),
            "source_inject_layer_from": float(source_inject_layer_from),
            "source_inject_layer_to": float(source_inject_layer_to),
            "source_inject_steps": float(source_inject_steps),
            "source_inject_mask_mode": source_inject_mask_mode,
            "source_inject_mask_box": None
            if resolved_source_inject_mask_box is None
            else [float(v) for v in resolved_source_inject_mask_box],
            "source_inject_active_step": float(active_edit_step),
            "source_inject_active": bool(source_inject_active),
            "source_inject_layers": float(source_inject_counts["all"]),
            "source_inject_q_layers": float(source_inject_counts["q"]),
            "source_inject_k_layers": float(source_inject_counts["k"]),
            "source_inject_v_layers": float(source_inject_counts["v"]),
            "edit_initial_noise_scale": float(edit_initial_noise_scale),
            "edit_initial_noise_region": edit_initial_noise_region,
            "trajectory_preserve_weight": float(trajectory_preserve_weight),
            "attention_mask_mode": attention_mask_mode,
            "attention_mask_subject_threshold": float(attention_mask_subject_threshold),
            "attention_mask_core_threshold": float(attention_mask_core_threshold),
            "attention_mask_max_area_ratio": float(attention_mask_max_area_ratio),
            "attention_mask_fallback_threshold": float(attention_mask_fallback_threshold),
            "object_mask_provider": object_mask_provider,
            "semantic_base_mask_path": semantic_base_mask_path,
            "support_score": support_score,
            "support_edit_operation": support_edit_operation,
            "support_relation": support_relation,
            "support_grounding_method": support_grounding_method,
            "save_support_debug_maps": bool(save_support_debug_maps),
            "support_temporal_aggregation": support_temporal_aggregation,
            "support_new_tokens": ",".join(support_new_tokens or []),
            "support_host_tokens": ",".join(support_host_tokens or []),
            "support_removed_tokens": ",".join(support_removed_tokens or []),
            "support_attention_power": float(support_attention_power),
            "support_disagreement_power": float(support_disagreement_power),
            "support_top_percentile": float(support_top_percentile),
            "support_min_area_ratio": float(support_min_area_ratio),
            "support_max_area_ratio": float(support_max_area_ratio),
            "support_keep_components": float(support_keep_components),
            "support_dilate_radius": float(support_dilate_radius),
            "support_blur_kernel": float(support_blur_kernel),
            **generic_support_stats,
            "attention_velocity_support_pad_x": float(attention_velocity_support_pad_x),
            "attention_velocity_support_pad_y": float(attention_velocity_support_pad_y),
            "attention_velocity_support_min_width": float(attention_velocity_support_min_width),
            "attention_velocity_support_min_height": float(attention_velocity_support_min_height),
            "mask_layering_mode": mask_layering_mode,
            "mask_object_threshold": float(mask_object_threshold),
            "mask_contact_dilate_kernel": float(mask_contact_dilate_kernel),
            "mask_contact_scale": float(mask_contact_scale),
            "mask_contact_edge_threshold": float(mask_contact_edge_threshold),
            "mask_contact_edge_protect_scale": float(mask_contact_edge_protect_scale),
            "mask_trimap_inner_erode_kernel": int(mask_trimap_inner_erode_kernel),
            "mask_trimap_outer_dilate_kernel": int(mask_trimap_outer_dilate_kernel),
            "mask_trimap_boundary_edit_scale": float(mask_trimap_boundary_edit_scale),
            "mask_trimap_boundary_preserve_scale": float(mask_trimap_boundary_preserve_scale),
            "mask_area_guard_applied": bool(mask_area_guard_applied),
            "mask_area_before_guard": mask_area_before_guard,
            "mask_area_guard_box": _box_to_list(mask_area_guard_box),
            "edit_mask_dilate_kernel": float(edit_mask_dilate_kernel),
            "edit_mask_erode_kernel": float(edit_mask_erode_kernel),
            "edit_mask_smooth_kernel": float(edit_mask_smooth_kernel),
            "edit_mask_hole_fraction": float(edit_mask_hole_fraction),
            "edit_mask_boundary_noise_scale": float(edit_mask_boundary_noise_scale),
            "edit_mask_component_threshold": float(edit_mask_component_threshold),
            "edit_mask_keep_components": float(edit_mask_keep_components),
            "edit_mask_component_y_min": None
            if edit_mask_component_y_min is None
            else float(edit_mask_component_y_min),
            "edit_mask_component_y_max": None
            if edit_mask_component_y_max is None
            else float(edit_mask_component_y_max),
            "edit_mask_shift_y": float(edit_mask_shift_y),
            "edit_mask_shift_x": float(edit_mask_shift_x),
            "auto_local_boxes": bool(auto_local_boxes),
            "auto_box_threshold": float(auto_box_threshold),
            "auto_anchor_box": _box_to_list(auto_anchor_box),
            "resolved_edit_mask_box": _box_to_list(resolved_edit_mask_box),
            "resolved_edit_mask_exclude_box": _box_to_list(resolved_edit_mask_exclude_box),
            "resolved_source_inject_mask_box": _box_to_list(resolved_source_inject_mask_box),
            "resolved_final_preserve_box": _box_to_list(resolved_final_preserve_box),
            "edit_mask_box_mode": edit_mask_box_mode,
            "edit_mask_exclude_box": None
            if resolved_edit_mask_exclude_box is None
            else [float(v) for v in resolved_edit_mask_exclude_box],
            "edit_mask_use_core_as_subject": bool(edit_mask_use_core_as_subject),
            "external_edit_mask_path": external_edit_mask_path,
            "external_edit_mask_mode": external_edit_mask_mode,
            "mask_blend": bool(mask_blend),
            "mask_blend_mode": mask_blend_mode,
            **mask_stats,
            **core_mask_stats,
            **contact_mask_stats,
            **structure_edge_stats,
            **preserve_mask_stats,
            **source_inject_mask_stats,
            "edit_hedit_guidance_scale": float(edit_hedit_guidance_scale),
            "edit_field_mode": edit_field_mode,
            "edit_guidance_scale": float(edit_guidance_scale),
            "edit_region_guidance_scale": float(edit_region_guidance_scale),
            "edit_clip_guidance_scale": float(edit_clip_guidance_scale),
            "edit_text_guidance_scale": float(edit_text_guidance_scale),
            "edit_text_source_scale": float(edit_text_source_scale),
            "edit_text_core_weight": float(edit_text_core_weight),
            "edit_text_subject_weight": float(edit_text_subject_weight),
            "edit_text_source_prompt": edit_text_source_prompt,
            "edit_text_target_prompt": edit_text_target_prompt,
            "edit_local_target_prompt": edit_local_target_prompt,
            "edit_local_target_guidance_scale": float(edit_local_target_guidance_scale),
            "edit_local_target_cfg_scale": float(edit_local_target_cfg_scale),
            "edit_color_guidance_scale": float(edit_color_guidance_scale),
            "edit_color_source": None if color_source is None else color_source[0],
            "edit_color_target": None if color_target is None else color_target[0],
            "edit_color_mask_path": edit_color_mask_path,
            "edit_color_mask_threshold": float(edit_color_mask_threshold),
            "edit_color_mask_softness": float(edit_color_mask_softness),
            "edit_color_luma_gate_min": float(edit_color_luma_gate_min),
            "edit_color_luma_gate_softness": float(edit_color_luma_gate_softness),
            "edit_color_detail_protect_scale": float(edit_color_detail_protect_scale),
            "edit_color_detail_protect_threshold": float(edit_color_detail_protect_threshold),
            "edit_color_detail_protect_softness": float(edit_color_detail_protect_softness),
            "edit_color_alpha_matte": bool(edit_color_alpha_matte),
            "edit_color_alpha_matte_mode": edit_color_alpha_matte_mode,
            "edit_color_alpha_matte_kernel_size": int(edit_color_alpha_matte_kernel_size),
            "edit_color_alpha_matte_threshold": None
            if edit_color_alpha_matte_threshold is None
            else float(edit_color_alpha_matte_threshold),
            "edit_color_alpha_matte_softness": None
            if edit_color_alpha_matte_softness is None
            else float(edit_color_alpha_matte_softness),
            "edit_color_alpha_matte_max_size": int(edit_color_alpha_matte_max_size),
            "edit_color_alpha_matte_epsilon": float(edit_color_alpha_matte_epsilon),
            "edit_color_alpha_matte_constraint_scale": float(edit_color_alpha_matte_constraint_scale),
            "edit_color_target_chroma_scale": float(edit_color_target_chroma_scale),
            "edit_color_smooth_kernel": int(edit_color_smooth_kernel),
            "edit_color_luma_preserve_scale": float(edit_color_luma_preserve_scale),
            "edit_color_luma_gradient_preserve_scale": float(edit_color_luma_gradient_preserve_scale),
            "edit_color_texture_preserve_scale": float(edit_color_texture_preserve_scale),
            "edit_color_texture_kernel_size": int(edit_color_texture_kernel_size),
            "edit_color_boundary_chroma_scale": float(edit_color_boundary_chroma_scale),
            "edit_color_boundary_kernel_size": int(edit_color_boundary_kernel_size),
            "edit_color_clean_projection_scale": float(edit_color_clean_projection_scale),
            "edit_color_clean_projection_mode": edit_color_clean_projection_mode,
            "edit_color_clean_projection_texture_kernel_size": int(edit_color_clean_projection_texture_kernel_size),
            "edit_color_clean_projection_luma_texture_scale": float(
                edit_color_clean_projection_luma_texture_scale
            ),
            "edit_color_clean_projection_chroma_texture_scale": float(
                edit_color_clean_projection_chroma_texture_scale
            ),
            "edit_color_clean_projection_delta_lowpass_kernel": int(
                edit_color_clean_projection_delta_lowpass_kernel
            ),
            "edit_color_clean_projection_alpha_power": float(edit_color_clean_projection_alpha_power),
            "edit_color_clean_projection_boundary_boost": float(edit_color_clean_projection_boundary_boost),
            "edit_color_clean_projection_boundary_kernel_size": int(edit_color_clean_projection_boundary_kernel_size),
            "edit_color_clean_projection_composite_mode": edit_color_clean_projection_composite_mode,
            "edit_color_clean_projection_background_kernel_size": int(edit_color_clean_projection_background_kernel_size),
            "edit_color_clean_projection_target_mode": recolor_projection_target_mode,
            "edit_color_clean_projection_refresh_interval": int(recolor_projection_refresh_interval),
            "recolor_clean_projection_eval_count": int(recolor_projection_eval_count),
            "recolor_clean_projection_refresh_count": int(recolor_projection_refresh_count),
            **spatial_mask_stats(color_mask_latent, prefix="color_mask"),
            **spatial_mask_stats(
                None
                if color_projection_alpha_image is None
                else torch.nn.functional.interpolate(
                    color_projection_alpha_image,
                    size=x_src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0),
                prefix="color_projection_alpha",
            ),
            "edit_ref_guidance_scale": float(edit_ref_guidance_scale),
            "edit_ref_image_path": edit_ref_image_path,
            "edit_ref_mask_path": edit_ref_mask_path,
            "edit_ref_structure_image_path": edit_ref_structure_image_path,
            "edit_ref_chroma_mode": edit_ref_chroma_mode,
            "edit_ref_chroma_magnitude_scale": float(edit_ref_chroma_magnitude_scale),
            "edit_ref_luma_preserve_scale": float(edit_ref_luma_preserve_scale),
            "edit_ref_gradient_preserve_scale": float(edit_ref_gradient_preserve_scale),
            "edit_ref_darkness_guard_scale": float(edit_ref_darkness_guard_scale),
            "edit_ref_darkness_guard_margin": float(edit_ref_darkness_guard_margin),
            "edit_ref_smooth_kernel": int(edit_ref_smooth_kernel),
            "edit_ref_lowfreq_suppress_kernel": int(edit_ref_lowfreq_suppress_kernel),
            "edit_ref_lowfreq_suppress_scale": float(edit_ref_lowfreq_suppress_scale),
            "edit_ref_schedule_start": float(edit_ref_schedule_start),
            "edit_ref_schedule_stop": float(edit_ref_schedule_stop),
            "edit_ref_schedule_power": float(edit_ref_schedule_power),
            "edit_ref_max_struct_rms_ratio": float(edit_ref_max_struct_rms_ratio),
            "edit_ref_project_struct_conflict": float(edit_ref_project_struct_conflict),
            **spatial_mask_stats(ref_mask_latent, prefix="ref_mask"),
            "completion_clean_delta_scale": float(completion_clean_delta_scale),
            "completion_clean_delta_image_path": completion_clean_delta_image_path,
            "completion_clean_delta_mask_path": completion_clean_delta_mask_path,
            "completion_clean_delta_schedule_start": float(completion_clean_delta_schedule_start),
            "completion_clean_delta_schedule_stop": float(completion_clean_delta_schedule_stop),
            "completion_clean_delta_schedule_power": float(completion_clean_delta_schedule_power),
            **spatial_mask_stats(completion_clean_mask_latent, prefix="completion_clean_mask"),
            "base_edit_norm": float(base_edit_norm),
            "edit_guidance_norm": float(edit_guidance_norm),
            "clip_guidance_norm": float(clip_guidance_norm),
            "text_guidance_norm": float(text_guidance_norm),
            "dds_guidance_norm": float(dds_guidance_norm),
            "app_guidance_norm": float(app_guidance_norm),
            "color_guidance_norm": float(color_guidance_norm),
            "ref_guidance_norm": float(ref_guidance_norm),
            "ref_guidance_pre_stabilize_norm": float(ref_guidance_pre_stabilize_norm),
            "ref_guidance_schedule_weight": float(ref_guidance_schedule_weight),
            "completion_clean_delta_norm": float(completion_clean_delta_norm),
            "completion_clean_delta_schedule_weight": float(completion_clean_delta_schedule_weight),
            "completion_clean_delta_rms": float(completion_clean_delta_rms),
            "recolor_clean_projection_norm": float(recolor_clean_projection_norm),
            "recolor_clean_projection_rms": float(recolor_clean_projection_rms),
            "recolor_clean_projection_refresh": float(recolor_clean_projection_refresh),
            "rec_guidance_norm": float(rec_guidance_norm),
            "removal_controller_norm": float(removal_controller_norm),
            "removal_fill_norm": float(removal_fill_norm),
            "removal_suppression_norm": float(removal_suppression_norm),
            "removal_ring_rec_norm": float(removal_ring_rec_norm),
            "trajectory_preserve_norm": float(trajectory_preserve_norm),
            "total_velocity_norm": float(total_velocity_norm),
            "rec_energy": 0.0 if rec_energy_terms is None else float(rec_energy_terms["latent"].item()),
            "struct_energy": 0.0 if rec_energy_terms is None else float(rec_energy_terms["feature"].item()),
            "edit_anchor_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["anchor"].item()),
            "edit_region_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["region"].item()),
            "edit_target_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["target"].item()),
            "edit_source_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["source"].item()),
            "edit_base_norm": 0.0 if v_edit_terms is None else float(v_edit_terms["base"].norm().item()),
            "edit_tv_energy": 0.0 if tv_edit_energy is None else float(tv_edit_energy.item()),
            "clip_target_energy": 0.0 if clip_edit_energy_terms is None else float(clip_edit_energy_terms["target"].item()),
            "clip_source_energy": 0.0 if clip_edit_energy_terms is None else float(clip_edit_energy_terms["source"].item()),
            "clip_direction_energy": 0.0 if clip_edit_energy_terms is None else float(clip_edit_energy_terms["direction"].item()),
            "text_target_core_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["target_core"].item()),
            "text_source_core_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["source_core"].item()),
            "text_target_subject_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["target_subject"].item()),
            "text_source_subject_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["source_subject"].item()),
            "dds_target_energy": 0.0
            if dds_energy_terms is None
            else float(dds_energy_terms["target"].item()),
            "dds_source_energy": 0.0
            if dds_energy_terms is None
            else float(dds_energy_terms["source"].item()),
            "app_energy": 0.0 if app_edit_energy is None else float(app_edit_energy.item()),
            "color_energy": 0.0 if color_edit_energy is None else float(color_edit_energy.item()),
            "ref_energy": 0.0 if ref_edit_energy is None else float(ref_edit_energy.item()),
            "edit_anchor_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["anchor"].norm().item()),
            "edit_region_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["region"].norm().item()),
            "edit_target_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["target"].norm().item()),
            "edit_source_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["source"].norm().item()),
            "cos_base_anchor": cosine_safe(edit_base_for_cos, edit_anchor_for_cos),
            "cos_base_region": cosine_safe(edit_base_for_cos, edit_region_for_cos),
            "cos_anchor_region": cosine_safe(edit_anchor_for_cos, edit_region_for_cos),
            "cos_rec_base": cosine_safe(v_rec, v_base),
            "cos_rec_edit_total": cosine_safe(v_rec, v_edit_total),
        }
        step_stats.append(step_stat)

        if log_every > 0 and (i == 0 or (i + 1) % log_every == 0 or i == len(timesteps) - 1):
            print(
                "[ode] "
                f"step={i:02d} "
                f"t={step_stat['t']:.4f} "
                f"alpha={alpha_t:.3f} "
                f"beta={beta_t:.3f} "
                f"|v_base|={step_stat['base_edit_norm']:.4f} "
                f"|v_edit|={step_stat['edit_guidance_norm']:.4f} "
                f"|v_clip|={step_stat['clip_guidance_norm']:.4f} "
                f"|v_text|={step_stat['text_guidance_norm']:.4f} "
                f"|v_dds|={step_stat['dds_guidance_norm']:.4f} "
                f"|v_app|={step_stat['app_guidance_norm']:.4f} "
                f"|v_color|={step_stat['color_guidance_norm']:.4f} "
                f"|v_ref|={step_stat['ref_guidance_norm']:.4f} "
                f"|v_cdelta|={step_stat['completion_clean_delta_norm']:.4f} "
                f"w_ref={step_stat['ref_guidance_schedule_weight']:.3f} "
                f"|v_rec|={step_stat['rec_guidance_norm']:.4f} "
                f"|traj|={step_stat['trajectory_preserve_norm']:.4f} "
                f"|v_total|={step_stat['total_velocity_norm']:.4f} "
                f"E_rec={step_stat['rec_energy']:.6f} "
                f"E_struct={step_stat['struct_energy']:.6f} "
                f"E_anchor={step_stat['edit_anchor_energy']:.6f} "
                f"E_region={step_stat['edit_region_energy']:.6f} "
                    f"E_target={step_stat['edit_target_energy']:.6f} "
                    f"E_source={step_stat['edit_source_energy']:.6f} "
                    f"|v_hedit|={step_stat['edit_base_norm']:.4f} "
                    f"E_tv={step_stat['edit_tv_energy']:.6f} "
                    f"E_clip_t={step_stat['clip_target_energy']:.6f} "
                    f"E_clip_s={step_stat['clip_source_energy']:.6f} "
                    f"E_clip_dir={step_stat['clip_direction_energy']:.6f} "
                    f"E_txt_tc={step_stat['text_target_core_energy']:.6f} "
                    f"E_txt_sc={step_stat['text_source_core_energy']:.6f} "
                    f"E_txt_ts={step_stat['text_target_subject_energy']:.6f} "
                    f"E_txt_ss={step_stat['text_source_subject_energy']:.6f} "
                    f"E_dds_t={step_stat['dds_target_energy']:.6f} "
                    f"E_dds_s={step_stat['dds_source_energy']:.6f} "
                    f"E_app={step_stat['app_energy']:.6f} "
                    f"E_color={step_stat['color_energy']:.6f} "
                    f"E_ref={step_stat['ref_energy']:.6f} "
                    f"|v_ref_raw|={step_stat['ref_guidance_pre_stabilize_norm']:.4f}"
            )

        z_t = next_z_t.to(torch.float32)
        if mask_blend and rec_gate is not None and preserve_blend_scale > 0.0:
            blend_start_idx = int(preserve_blend_start_timestep * len(timesteps))
            if i >= blend_start_idx:
                if len(timesteps) - blend_start_idx <= 1:
                    blend_progress = 1.0
                else:
                    blend_progress = (i - blend_start_idx) / float(len(timesteps) - blend_start_idx - 1)
                blend_weight = preserve_blend_scale * blend_progress
                z_t = (1.0 - blend_weight * rec_gate) * z_t + (blend_weight * rec_gate) * x0_src_step.to(torch.float32)
        z_t = z_t.to(latents_dtype)
        active_edit_step += 1

    result = z_t

    if stats_output_path is not None:
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(step_stats, f, indent=2)
        print(f"[stats] saved ODE stats to {stats_output_path}")

    # Optional hard mask blend: paste source background over edited result.
    if mask_blend and M_preserve is not None:
        if mask_blend_mode == "subject":
            blend_edit = (1.0 - M_preserve).to(dtype=result.dtype, device=result.device)
        elif mask_blend_mode == "core":
            if M_core is None:
                raise ValueError("mask_blend_mode='core' requires an extracted core mask.")
            blend_edit = M_core.to(dtype=result.dtype, device=result.device)
        else:
            raise ValueError(f"Unsupported mask_blend_mode: {mask_blend_mode}")
        blend_preserve = (1.0 - blend_edit).clamp(0.0, 1.0)
        x_src_aligned = x_src.to(dtype=result.dtype, device=result.device)
        result = blend_edit * result + blend_preserve * x_src_aligned

    if resolved_final_preserve_box is not None:
        preserve_box_mask = normalized_box_mask_like(result[:, :1], resolved_final_preserve_box).to(
            dtype=result.dtype,
            device=result.device,
        )
        x_src_aligned = x_src.to(dtype=result.dtype, device=result.device)
        result = (1.0 - preserve_box_mask) * result + preserve_box_mask * x_src_aligned

    return result
