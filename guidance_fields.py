from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_latent_to_unit_image(
    pipe,
    latent: torch.Tensor,
    vae_override=None,
) -> torch.Tensor:
    vae = pipe.vae if vae_override is None else vae_override
    vae_dtype = next(vae.parameters()).dtype
    latent_denorm = (
        (latent / vae.config.scaling_factor) + vae.config.shift_factor
    ).to(dtype=vae_dtype)
    image = vae.decode(latent_denorm, return_dict=False)[0]
    return ((image / 2.0) + 0.5).clamp(0.0, 1.0)


def masked_total_variation(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    if mask is not None:
        mask = torch.nn.functional.interpolate(
            mask,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        dx = dx * mask[:, :, :, 1:]
        dy = dy * mask[:, :, 1:, :]
    return dx.abs().mean() + dy.abs().mean()


_COLOR_RGB = {
    "black": (0.02, 0.02, 0.02),
    "blue": (0.02, 0.35, 0.95),
    "brown": (0.45, 0.24, 0.08),
    "cyan": (0.0, 0.75, 0.95),
    "gold": (0.95, 0.72, 0.12),
    "golden": (0.95, 0.72, 0.12),
    "gray": (0.5, 0.5, 0.5),
    "green": (0.05, 0.65, 0.12),
    "grey": (0.5, 0.5, 0.5),
    "orange": (0.95, 0.42, 0.05),
    "pink": (0.95, 0.35, 0.65),
    "purple": (0.45, 0.12, 0.8),
    "red": (0.9, 0.05, 0.04),
    "silver": (0.75, 0.75, 0.72),
    "white": (0.95, 0.95, 0.92),
    "yellow": (0.95, 0.82, 0.05),
}


def _prompt_words(text: str) -> list[str]:
    token = []
    words = []
    for char in text.lower():
        if char.isalnum():
            token.append(char)
        elif token:
            words.append("".join(token))
            token = []
    if token:
        words.append("".join(token))
    return words


def infer_target_color_rgb(
    src_prompt: str,
    tar_prompt: str,
    explicit: str | None = None,
) -> tuple[str, torch.Tensor] | None:
    if explicit:
        key = explicit.strip().lower()
        if key not in _COLOR_RGB:
            raise ValueError(f"Unsupported target color: {explicit!r}")
        return key, torch.tensor(_COLOR_RGB[key], dtype=torch.float32)
    src_colors = {word for word in _prompt_words(src_prompt) if word in _COLOR_RGB}
    for word in _prompt_words(tar_prompt):
        if word in _COLOR_RGB and word not in src_colors:
            return word, torch.tensor(_COLOR_RGB[word], dtype=torch.float32)
    for word in _prompt_words(tar_prompt):
        if word in _COLOR_RGB:
            return word, torch.tensor(_COLOR_RGB[word], dtype=torch.float32)
    return None


def infer_source_color_rgb(
    src_prompt: str,
    explicit: str | None = None,
) -> tuple[str, torch.Tensor] | None:
    if explicit:
        key = explicit.strip().lower()
        if key not in _COLOR_RGB:
            raise ValueError(f"Unsupported source color: {explicit!r}")
        return key, torch.tensor(_COLOR_RGB[key], dtype=torch.float32)
    for word in _prompt_words(src_prompt):
        if word in _COLOR_RGB:
            return word, torch.tensor(_COLOR_RGB[word], dtype=torch.float32)
    return None


def source_color_similarity_mask(
    source_image: torch.Tensor,
    source_rgb: torch.Tensor,
    object_mask: torch.Tensor | None,
    threshold: float = 0.38,
    softness: float = 0.10,
    luma_gate_min: float = 0.0,
    luma_gate_softness: float = 0.08,
    detail_protect_scale: float = 0.0,
    detail_protect_threshold: float = 0.35,
    detail_protect_softness: float = 0.08,
) -> torch.Tensor:
    target = source_rgb.to(dtype=source_image.dtype, device=source_image.device).view(1, 3, 1, 1)
    dist = (source_image - target).square().sum(dim=1, keepdim=True).sqrt()
    softness = max(float(softness), 1e-4)
    color_weight = torch.sigmoid((float(threshold) - dist) / softness)
    source_y = None
    if luma_gate_min > 0.0:
        source_y = rgb_to_yuv(source_image)[:, :1]
        gate_softness = max(float(luma_gate_softness), 1e-4)
        luma_gate = torch.sigmoid((source_y - float(luma_gate_min)) / gate_softness)
        color_weight = color_weight * luma_gate
    if detail_protect_scale > 0.0:
        if source_y is None:
            source_y = rgb_to_yuv(source_image)[:, :1]
        grad_x = source_y[..., :, 1:] - source_y[..., :, :-1]
        grad_y = source_y[..., 1:, :] - source_y[..., :-1, :]
        edge = torch.zeros_like(source_y)
        edge[..., :, 1:] = edge[..., :, 1:] + grad_x.abs()
        edge[..., 1:, :] = edge[..., 1:, :] + grad_y.abs()
        edge = edge.float()
        flat = edge.flatten(1)
        lo = torch.quantile(flat, 0.10, dim=1).view(-1, 1, 1, 1)
        hi = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
        edge = ((edge - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        protect_softness = max(float(detail_protect_softness), 1e-4)
        protect = torch.sigmoid((edge - float(detail_protect_threshold)) / protect_softness)
        protect_scale = max(0.0, min(1.0, float(detail_protect_scale)))
        color_weight = color_weight * (1.0 - protect_scale * protect).clamp(0.0, 1.0)
    if object_mask is not None:
        obj = torch.nn.functional.interpolate(
            object_mask.to(dtype=source_image.dtype, device=source_image.device),
            size=source_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        color_weight = color_weight * obj
    return color_weight.clamp(0.0, 1.0)


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r = image[:, 0:1]
    g = image[:, 1:2]
    b = image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def masked_chroma_luma_loss(
    image: torch.Tensor,
    source_image: torch.Tensor,
    target_rgb: torch.Tensor,
    mask: torch.Tensor | None,
    target_chroma_scale: float = 1.0,
    luma_preserve_scale: float = 0.35,
    luma_gradient_preserve_scale: float = 0.0,
) -> torch.Tensor:
    if mask is None:
        weights = torch.ones(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=image.dtype,
        )
    else:
        weights = torch.nn.functional.interpolate(
            mask.to(dtype=image.dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    src = torch.nn.functional.interpolate(
        source_image.to(dtype=image.dtype, device=image.device),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    current_yuv = rgb_to_yuv(image)
    source_yuv = rgb_to_yuv(src)
    target_image = target_rgb.to(dtype=image.dtype, device=image.device).view(1, 3, 1, 1).expand_as(image)
    target_yuv = rgb_to_yuv(target_image)
    target_yuv = torch.cat(
        [
            target_yuv[:, :1],
            target_yuv[:, 1:] * max(0.0, float(target_chroma_scale)),
        ],
        dim=1,
    )
    denom = weights.sum().clamp_min(1e-6)
    chroma_loss = ((current_yuv[:, 1:] - target_yuv[:, 1:]).pow(2) * weights).sum() / denom
    luma_loss = ((current_yuv[:, :1] - source_yuv[:, :1]).pow(2) * weights).sum() / denom
    total = chroma_loss + float(luma_preserve_scale) * luma_loss
    if luma_gradient_preserve_scale > 0.0:
        current_y = current_yuv[:, :1]
        source_y = source_yuv[:, :1]
        grad_x = current_y[..., :, 1:] - current_y[..., :, :-1]
        src_grad_x = source_y[..., :, 1:] - source_y[..., :, :-1]
        grad_y = current_y[..., 1:, :] - current_y[..., :-1, :]
        src_grad_y = source_y[..., 1:, :] - source_y[..., :-1, :]
        weights_x = torch.minimum(weights[..., :, 1:], weights[..., :, :-1])
        weights_y = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
        denom_x = weights_x.sum().clamp_min(1e-6)
        denom_y = weights_y.sum().clamp_min(1e-6)
        grad_loss = (
            ((grad_x - src_grad_x).pow(2) * weights_x).sum() / denom_x
            + ((grad_y - src_grad_y).pow(2) * weights_y).sum() / denom_y
        )
        total = total + float(luma_gradient_preserve_scale) * grad_loss
    return 0.5 * total


def masked_reference_image_loss(
    image: torch.Tensor,
    reference_image: torch.Tensor,
    mask: torch.Tensor | None,
    structure_image: torch.Tensor | None = None,
    luma_preserve_scale: float = 0.35,
    gradient_preserve_scale: float = 0.0,
    darkness_guard_scale: float = 0.0,
    darkness_guard_margin: float = 0.03,
    chroma_mode: str = "yuv",
    chroma_magnitude_scale: float = 1.0,
) -> torch.Tensor:
    # Decode-space color losses can overflow in fp16 when a nearly gray pixel
    # is normalized into a chroma direction. Keep this small objective in fp32.
    loss_dtype = torch.float32
    image_f = image.to(dtype=loss_dtype)
    if mask is None:
        weights = torch.ones(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=loss_dtype,
        )
    else:
        weights = torch.nn.functional.interpolate(
            mask.to(dtype=loss_dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    ref = torch.nn.functional.interpolate(
        reference_image.to(dtype=loss_dtype, device=image.device),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    structure_ref = ref
    if structure_image is not None:
        structure_ref = torch.nn.functional.interpolate(
            structure_image.to(dtype=loss_dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    current_yuv = rgb_to_yuv(image_f)
    reference_yuv = rgb_to_yuv(ref)
    structure_yuv = rgb_to_yuv(structure_ref)
    denom = weights.sum().clamp_min(1e-6)
    chroma_mode = str(chroma_mode).strip().lower()
    if chroma_mode == "yuv":
        chroma_loss = ((current_yuv[:, 1:] - reference_yuv[:, 1:]).pow(2) * weights).sum() / denom
    elif chroma_mode == "yuv_direction":
        cur_uv = current_yuv[:, 1:]
        ref_uv = reference_yuv[:, 1:]
        eps = 1e-2
        cur_norm = (cur_uv.square().sum(dim=1, keepdim=True) + eps * eps).sqrt()
        ref_norm = (ref_uv.square().sum(dim=1, keepdim=True) + eps * eps).sqrt()
        cur_dir = cur_uv / cur_norm
        ref_dir = ref_uv / ref_norm
        direction_loss = ((cur_dir - ref_dir).pow(2) * weights).sum() / denom
        mag_loss = ((cur_norm - ref_norm).pow(2) * weights).sum() / denom
        chroma_loss = direction_loss + max(0.0, float(chroma_magnitude_scale)) * mag_loss
    else:
        raise ValueError(f"Unsupported edit ref chroma mode: {chroma_mode}")
    luma_loss = ((current_yuv[:, :1] - structure_yuv[:, :1]).pow(2) * weights).sum() / denom
    total = chroma_loss + float(luma_preserve_scale) * luma_loss
    if gradient_preserve_scale > 0.0:
        current_y = current_yuv[:, :1]
        reference_y = structure_yuv[:, :1]
        grad_x = current_y[..., :, 1:] - current_y[..., :, :-1]
        ref_grad_x = reference_y[..., :, 1:] - reference_y[..., :, :-1]
        grad_y = current_y[..., 1:, :] - current_y[..., :-1, :]
        ref_grad_y = reference_y[..., 1:, :] - reference_y[..., :-1, :]
        weights_x = torch.minimum(weights[..., :, 1:], weights[..., :, :-1])
        weights_y = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
        denom_x = weights_x.sum().clamp_min(1e-6)
        denom_y = weights_y.sum().clamp_min(1e-6)
        grad_loss = (
            ((grad_x - ref_grad_x).pow(2) * weights_x).sum() / denom_x
            + ((grad_y - ref_grad_y).pow(2) * weights_y).sum() / denom_y
        )
        total = total + float(gradient_preserve_scale) * grad_loss
    if darkness_guard_scale > 0.0:
        margin = max(0.0, float(darkness_guard_margin))
        darkening = torch.relu(structure_yuv[:, :1] - current_yuv[:, :1] - margin)
        darkening_loss = (darkening.pow(2) * weights).sum() / denom
        total = total + float(darkness_guard_scale) * darkening_loss
    return 0.5 * total


def smooth_guidance_field(
    field: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    if kernel_size <= 1:
        return field
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(field, kernel_size=kernel_size, stride=1, padding=pad)


def suppress_low_frequency_guidance(
    field: torch.Tensor,
    kernel_size: int = 0,
    strength: float = 0.0,
) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or kernel_size <= 1:
        return field
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    low = torch.nn.functional.avg_pool2d(field, kernel_size=kernel_size, stride=1, padding=pad)
    return field - strength * low


def ramp_schedule_from_progress(
    progress: float,
    start: float = 0.0,
    stop: float = 0.0,
    power: float = 1.0,
) -> float:
    start = float(start)
    stop = float(stop)
    if stop <= start:
        return 1.0
    x = (float(progress) - start) / max(1e-6, stop - start)
    x = max(0.0, min(1.0, x))
    x = x * x * (3.0 - 2.0 * x)
    if power != 1.0:
        x = x ** max(1e-6, float(power))
    return float(x)


def remove_opposing_guidance_component(
    guidance: torch.Tensor,
    anchor: torch.Tensor | None,
    strength: float = 1.0,
) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or anchor is None:
        return guidance
    anchor = anchor.to(device=guidance.device, dtype=guidance.dtype)
    anchor_sq = anchor.square().sum().clamp_min(1e-8)
    dot = (guidance * anchor).sum()
    if float(dot.detach().item()) >= 0.0:
        return guidance
    opposing_projection = (dot / anchor_sq) * anchor
    return guidance - strength * opposing_projection


def masked_rms(
    tensor: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    value = tensor.detach().float()
    if mask is None:
        return value.square().mean().sqrt()
    gate = mask.detach().float().to(device=value.device)
    if gate.shape[-2:] != value.shape[-2:]:
        gate = torch.nn.functional.interpolate(
            gate,
            size=value.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(1)
    gate = gate.clamp(0.0, 1.0)
    denom = (gate.sum() * value.shape[1]).clamp_min(eps)
    return ((value.square() * gate).sum() / denom).sqrt()


def remove_masked_opposing_guidance_component(
    guidance: torch.Tensor,
    anchor: torch.Tensor | None,
    mask: torch.Tensor | None = None,
    strength: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or anchor is None:
        return guidance, 0.0, 0.0
    anchor = anchor.to(device=guidance.device, dtype=guidance.dtype)
    guidance_eval = guidance
    anchor_eval = anchor
    if mask is not None:
        gate = mask.to(device=guidance.device, dtype=guidance.dtype)
        if gate.shape[-2:] != guidance.shape[-2:]:
            gate = torch.nn.functional.interpolate(
                gate,
                size=guidance.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = gate.clamp(0.0, 1.0)
        guidance_eval = guidance_eval * gate
        anchor_eval = anchor_eval * gate
    anchor_sq = anchor_eval.square().sum().clamp_min(1e-8)
    dot = (guidance_eval * anchor_eval).sum()
    if float(dot.detach().item()) >= 0.0:
        return guidance, float(dot.detach().item()), 0.0
    opposing_projection = (dot / anchor_sq) * anchor
    if mask is not None:
        opposing_projection = opposing_projection * gate
    delta = strength * opposing_projection
    return guidance - delta, float(dot.detach().item()), float(delta.norm().detach().item())


def limit_guidance_rms_relative_to_anchor(
    guidance: torch.Tensor,
    anchor: torch.Tensor | None,
    max_ratio: float = 0.0,
) -> torch.Tensor:
    max_ratio = float(max_ratio)
    if max_ratio <= 0.0 or anchor is None:
        return guidance
    guidance_rms = guidance.square().mean().sqrt()
    anchor_rms = anchor.to(device=guidance.device, dtype=guidance.dtype).square().mean().sqrt()
    max_rms = max_ratio * anchor_rms.clamp_min(1e-8)
    scale = torch.minimum(torch.ones_like(guidance_rms), max_rms / guidance_rms.clamp_min(1e-8))
    return guidance * scale
