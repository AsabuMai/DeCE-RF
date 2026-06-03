from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GenericSupportResult:
    edit_mask: torch.Tensor
    core_mask: torch.Tensor
    attention_map: torch.Tensor
    host_attention_map: torch.Tensor | None
    removed_attention_map: torch.Tensor | None
    clean_disagreement_map: torch.Tensor
    velocity_disagreement_map: torch.Tensor
    support_score: torch.Tensor
    stats: dict[str, float | int | str]


def normalize_spatial_map(x: torch.Tensor, q_low: float = 0.10, q_high: float = 0.98) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim == 3:
        x = x[:, None]
    flat = x.flatten(1)
    lo = torch.quantile(flat, q_low, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, q_high, dim=1).view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def clean_disagreement_map(
    x_t: torch.Tensor,
    t: torch.Tensor,
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    t_view = t.reshape(-1, *([1] * (x_t.ndim - 1))).to(device=x_t.device, dtype=x_t.dtype)
    x0_source = x_t - t_view * source_velocity
    x0_target = x_t - t_view * target_velocity
    diff = torch.linalg.norm((x0_target - x0_source).detach().float(), dim=1, keepdim=True)
    return normalize_spatial_map(diff)


def velocity_disagreement_map(source_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    diff = torch.linalg.norm((target_velocity - source_velocity).detach().float(), dim=1, keepdim=True)
    return normalize_spatial_map(diff)


def build_support_score(
    attention_map: torch.Tensor,
    clean_map: torch.Tensor,
    velocity_map: torch.Tensor,
    host_attention_map: torch.Tensor | None = None,
    removed_attention_map: torch.Tensor | None = None,
    score_mode: str = "attention_x_clean",
    attention_power: float = 1.0,
    disagreement_power: float = 1.0,
) -> torch.Tensor:
    attention = normalize_spatial_map(attention_map).pow(max(0.0, float(attention_power)))
    host = (
        normalize_spatial_map(host_attention_map).pow(max(0.0, float(attention_power)))
        if host_attention_map is not None
        else attention
    )
    removed = (
        normalize_spatial_map(removed_attention_map).pow(max(0.0, float(attention_power)))
        if removed_attention_map is not None
        else attention
    )
    clean = normalize_spatial_map(clean_map).pow(max(0.0, float(disagreement_power)))
    velocity = normalize_spatial_map(velocity_map).pow(max(0.0, float(disagreement_power)))
    if score_mode == "attention_only":
        score = attention
    elif score_mode == "clean_disagreement_only":
        score = clean
    elif score_mode == "velocity_disagreement_only":
        score = velocity
    elif score_mode == "attention_x_velocity":
        score = attention * velocity
    elif score_mode == "attention_x_clean":
        score = attention * clean
    elif score_mode == "host_x_clean":
        score = host * clean
    elif score_mode == "new_x_host_x_clean":
        score = attention * host * clean
    elif score_mode == "new_plus_host_x_clean":
        score = torch.maximum(attention, 0.65 * host) * clean
    elif score_mode == "removed_src_x_clean":
        score = removed * clean
    elif score_mode == "removed_src_x_velocity":
        score = removed * velocity
    else:
        raise ValueError(f"Unsupported generic support score mode: {score_mode}")
    return normalize_spatial_map(score)


def _top_components(
    score: torch.Tensor,
    threshold: float,
    keep_components: int,
) -> tuple[torch.Tensor, int, float]:
    image = score.detach().float().cpu()
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW support score, got {tuple(score.shape)}")
    out = torch.zeros_like(image)
    total_components = 0
    top_area = 0.0
    bsz, _, h, w = image.shape
    for b in range(bsz):
        plane = image[b, 0].clamp(0.0, 1.0)
        binary = plane > threshold
        visited = torch.zeros_like(binary, dtype=torch.bool)
        components: list[tuple[float, int, list[tuple[int, int]]]] = []
        for y in range(h):
            for x in range(w):
                if not bool(binary[y, x]) or bool(visited[y, x]):
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                mass = 0.0
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    mass += float(plane[cy, cx].item())
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                components.append((mass, len(points), points))
        components.sort(key=lambda item: item[0], reverse=True)
        total_components += len(components)
        if components:
            top_area = max(top_area, components[0][1] / max(1, h * w))
        for _, _, points in components[: max(1, int(keep_components))]:
            for py, px in points:
                out[b, 0, py, px] = image[b, 0, py, px]
    return out.to(device=score.device, dtype=score.dtype), total_components, float(top_area)


def _area(mask: torch.Tensor, threshold: float = 0.5) -> float:
    return float((mask.detach().float().clamp(0.0, 1.0) > threshold).float().mean().item())


def postprocess_support_score(
    support_score: torch.Tensor,
    top_percentile: float = 90.0,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.30,
    keep_components: int = 2,
    dilate_radius: int = 5,
    blur_kernel: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    score = normalize_spatial_map(support_score)
    percentiles = [
        top_percentile,
        min(99.0, top_percentile + 5.0),
        min(99.0, top_percentile + 8.0),
        max(50.0, top_percentile - 5.0),
        max(50.0, top_percentile - 10.0),
    ]
    best_core = None
    best_stats: dict[str, float | int] = {}
    for percentile in percentiles:
        threshold = float(torch.quantile(score.flatten(1), percentile / 100.0).mean().item())
        core, num_components, top_component_area = _top_components(score, threshold, keep_components)
        area = _area(core)
        best_core = core
        best_stats = {
            "support_threshold": threshold,
            "support_top_percentile": float(percentile),
            "support_area_core_raw": area,
            "support_num_components": int(num_components),
            "support_top_component_area": float(top_component_area),
        }
        if (min_area_ratio <= 0.0 or area >= min_area_ratio) and (max_area_ratio <= 0.0 or area <= max_area_ratio):
            break
    if best_core is None:
        best_core = score
    core = best_core.clamp(0.0, 1.0)
    if dilate_radius > 1:
        kernel = int(dilate_radius)
        if kernel % 2 == 0:
            kernel += 1
        core = F.max_pool2d(core.float(), kernel_size=kernel, stride=1, padding=kernel // 2).to(core.dtype)
    edit = core
    if blur_kernel > 1:
        kernel = int(blur_kernel)
        if kernel % 2 == 0:
            kernel += 1
        edit = F.avg_pool2d(edit.float(), kernel_size=kernel, stride=1, padding=kernel // 2).to(edit.dtype)
        edit = normalize_spatial_map(edit)
    best_stats["support_area_core"] = _area(core)
    best_stats["support_area_edit"] = _area(edit)
    return edit.clamp(0.0, 1.0), core.clamp(0.0, 1.0), best_stats


def build_generic_support(
    attention_map: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    host_attention_map: torch.Tensor | None = None,
    removed_attention_map: torch.Tensor | None = None,
    edit_operation: str = "auto",
    score_mode: str = "attention_x_clean",
    attention_power: float = 1.0,
    disagreement_power: float = 1.0,
    top_percentile: float = 90.0,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.30,
    keep_components: int = 2,
    dilate_radius: int = 5,
    blur_kernel: int = 5,
) -> GenericSupportResult:
    attention = normalize_spatial_map(attention_map)
    clean = clean_disagreement_map(x_t, t, source_velocity, target_velocity)
    velocity = velocity_disagreement_map(source_velocity, target_velocity)
    score = build_support_score(
        attention,
        clean,
        velocity,
        host_attention_map=host_attention_map,
        removed_attention_map=removed_attention_map,
        score_mode=score_mode,
        attention_power=attention_power,
        disagreement_power=disagreement_power,
    )
    edit, core, stats = postprocess_support_score(
        score,
        top_percentile=top_percentile,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        keep_components=keep_components,
        dilate_radius=dilate_radius,
        blur_kernel=blur_kernel,
    )
    stats.update(
        {
            "support_mode": "generic",
            "support_edit_operation": edit_operation,
            "support_score": score_mode,
            "support_attention_power": float(attention_power),
            "support_disagreement_power": float(disagreement_power),
            "support_min_area_ratio": float(min_area_ratio),
            "support_max_area_ratio": float(max_area_ratio),
            "support_keep_components": int(keep_components),
            "support_dilate_radius": int(dilate_radius),
            "support_blur_kernel": int(blur_kernel),
        }
    )
    return GenericSupportResult(
        edit_mask=edit.to(device=x_t.device, dtype=x_t.dtype),
        core_mask=core.to(device=x_t.device, dtype=x_t.dtype),
        attention_map=attention.to(device=x_t.device, dtype=x_t.dtype),
        host_attention_map=(
            normalize_spatial_map(host_attention_map).to(device=x_t.device, dtype=x_t.dtype)
            if host_attention_map is not None
            else None
        ),
        removed_attention_map=(
            normalize_spatial_map(removed_attention_map).to(device=x_t.device, dtype=x_t.dtype)
            if removed_attention_map is not None
            else None
        ),
        clean_disagreement_map=clean.to(device=x_t.device, dtype=x_t.dtype),
        velocity_disagreement_map=velocity.to(device=x_t.device, dtype=x_t.dtype),
        support_score=score.to(device=x_t.device, dtype=x_t.dtype),
        stats=stats,
    )
