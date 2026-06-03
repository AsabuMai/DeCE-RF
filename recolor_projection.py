from __future__ import annotations

import json
import os

import PIL.Image
import torch

from guidance_fields import (
    decode_latent_to_unit_image,
    rgb_to_yuv,
    source_color_similarity_mask,
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


