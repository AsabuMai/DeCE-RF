from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from generic_support import (
    clean_disagreement_map,
    normalize_spatial_map,
    velocity_disagreement_map,
)


@dataclass
class OperationSupportV3Result:
    edit_mask: torch.Tensor
    core_mask: torch.Tensor
    attention_map: torch.Tensor
    host_attention_map: torch.Tensor | None
    removed_attention_map: torch.Tensor | None
    grounding_mask: torch.Tensor | None
    relation_map: torch.Tensor | None
    clean_disagreement_map: torch.Tensor
    velocity_disagreement_map: torch.Tensor
    support_score: torch.Tensor
    candidate_scores: dict[str, torch.Tensor]
    stats: dict[str, float | int | str]


def parse_edit_operation(edit_operation: str | None) -> str:
    op = (edit_operation or "auto").strip().lower()
    aliases = {
        "add": "add_object",
        "add_accessory": "add_object",
        "decal": "add_decal",
        "surface": "add_decal",
        "remove": "remove_object",
        "removal": "remove_object",
        "color": "recolor",
        "recolour": "recolor",
        "recolor_object": "recolor",
        "recolour_object": "recolor",
        "replace_object": "replace",
        "replace_attribute": "replace",
    }
    op = aliases.get(op, op)
    valid = {"auto", "add_object", "add_decal", "remove_object", "replace", "recolor"}
    if op not in valid:
        raise ValueError(f"Unsupported edit operation for support v3: {edit_operation}")
    return op


def _match_map(x: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if x is None:
        return None
    out = x.detach().float()
    if out.ndim == 2:
        out = out[None, None]
    elif out.ndim == 3:
        out = out[:, None]
    if out.shape[-2:] != reference.shape[-2:]:
        out = F.interpolate(out, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    if out.shape[0] != reference.shape[0]:
        if out.shape[0] == 1:
            out = out.expand(reference.shape[0], -1, -1, -1)
        else:
            out = out[: reference.shape[0]]
    return normalize_spatial_map(out).to(device=reference.device, dtype=reference.dtype)


def normalize_within_mask(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Min/max normalize a map only over the active mask support."""
    value = value.detach().float()
    if value.ndim == 3:
        value = value[:, None]
    surface = _match_map(mask, value)
    if surface is None:
        return torch.zeros_like(value)
    active = surface.detach().float() > 0.05
    if not bool(active.any().item()):
        return torch.zeros_like(value)
    inf = torch.full_like(value, float("inf"))
    ninf = torch.full_like(value, float("-inf"))
    minv = torch.where(active, value, inf).flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
    maxv = torch.where(active, value, ninf).flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    out = ((value - minv) / (maxv - minv).clamp_min(float(eps))).clamp(0.0, 1.0)
    return (out * active.to(dtype=out.dtype)).to(device=value.device, dtype=value.dtype)


def ground_object_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    """Normalize an externally grounded / segmented mask to latent resolution."""
    return _match_map(mask, reference)


def compute_token_attention(attention_map: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    """Normalize a token-attention map to latent resolution."""
    return _match_map(attention_map, reference)


def compute_clean_disagreement(
    x_t: torch.Tensor,
    t: torch.Tensor,
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    return clean_disagreement_map(x_t, t, source_velocity, target_velocity)


def compute_velocity_disagreement(source_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return velocity_disagreement_map(source_velocity, target_velocity)


def _binary_bbox(mask: torch.Tensor, threshold: float = 0.2) -> tuple[int, int, int, int] | None:
    plane = mask.detach().float()
    if plane.ndim == 4:
        plane = plane[0, 0]
    elif plane.ndim == 3:
        plane = plane[0]
    binary = plane > threshold
    ys, xs = torch.where(binary)
    if ys.numel() == 0:
        return None
    return int(xs.min().item()), int(ys.min().item()), int(xs.max().item()) + 1, int(ys.max().item()) + 1


def _box_mask_like(reference: torch.Tensor, box: tuple[int, int, int, int] | None) -> torch.Tensor:
    mask = torch.zeros(reference.shape[0], 1, reference.shape[-2], reference.shape[-1], device=reference.device)
    if box is None:
        return mask.to(dtype=reference.dtype)
    x0, y0, x1, y1 = box
    h, w = reference.shape[-2:]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        mask[:, :, y0:y1, x0:x1] = 1.0
    return mask.to(dtype=reference.dtype)


def build_above_host_region(
    host_mask: torch.Tensor,
    expand_x: float = 0.20,
    above_height: float = 0.60,
    overlap_height: float = 0.20,
) -> torch.Tensor:
    host = normalize_spatial_map(host_mask)
    h, w = host.shape[-2:]
    bbox = _binary_bbox(host)
    if bbox is None:
        return torch.zeros_like(host)
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    rx0 = int(round(x0 - expand_x * bw))
    rx1 = int(round(x1 + expand_x * bw))
    ry0 = int(round(y0 - above_height * bh))
    ry1 = int(round(y0 + overlap_height * bh))
    return _box_mask_like(host, (rx0, ry0, rx1, ry1)).to(device=host.device, dtype=host.dtype)


def build_surface_region(
    host_mask: torch.Tensor,
    erode_kernel: int = 5,
    center_width: float = 0.55,
    center_height: float = 0.55,
) -> torch.Tensor:
    host = normalize_spatial_map(host_mask)
    bbox = _binary_bbox(host)
    if bbox is None:
        return host
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    sx0 = int(round(cx - 0.5 * center_width * bw))
    sx1 = int(round(cx + 0.5 * center_width * bw))
    sy0 = int(round(cy - 0.5 * center_height * bh))
    sy1 = int(round(cy + 0.5 * center_height * bh))
    center = _box_mask_like(host, (sx0, sy0, sx1, sy1))
    surface = torch.minimum(host, center)
    if erode_kernel > 1:
        kernel = int(erode_kernel)
        if kernel % 2 == 0:
            kernel += 1
        inv = 1.0 - surface.float()
        surface = 1.0 - F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=kernel // 2)
    if float(surface.detach().float().max().item()) <= 1e-6:
        surface = center
    return normalize_spatial_map(surface).to(device=host.device, dtype=host.dtype)


def build_spawn_center_region(
    placement_mask: torch.Tensor,
    center_width: float = 0.45,
    center_height: float = 0.45,
    sigma_scale: float = 0.28,
    center_y: float = 0.50,
) -> torch.Tensor:
    placement = normalize_spatial_map(placement_mask)
    bbox = _binary_bbox(placement)
    if bbox is None:
        return torch.zeros_like(placement)
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = y0 + max(0.0, min(1.0, float(center_y))) * bh
    sx0 = int(round(cx - 0.5 * center_width * bw))
    sx1 = int(round(cx + 0.5 * center_width * bw))
    sy0 = int(round(cy - 0.5 * center_height * bh))
    sy1 = int(round(cy + 0.5 * center_height * bh))
    center_box = _box_mask_like(placement, (sx0, sy0, sx1, sy1))

    h, w = placement.shape[-2:]
    yy = torch.arange(h, device=placement.device, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=placement.device, dtype=torch.float32).view(1, 1, 1, w)
    sigma_x = max(1.0, float(sigma_scale) * float(bw))
    sigma_y = max(1.0, float(sigma_scale) * float(bh))
    gaussian = torch.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
    spawn = placement.float() * center_box.float() * gaussian
    if float(spawn.detach().float().max().item()) <= 1e-6:
        spawn = center_box.float() * gaussian
    return normalize_spatial_map(spawn).to(device=placement.device, dtype=placement.dtype)


def build_host_top_contact_region(
    host_mask: torch.Tensor,
    center_width: float = 0.42,
    above_height: float = 0.07,
    overlap_height: float = 0.12,
    sigma_x_scale: float = 0.22,
    sigma_y_scale: float = 0.09,
) -> torch.Tensor:
    host = normalize_spatial_map(host_mask)
    bbox = _binary_bbox(host)
    if bbox is None:
        return torch.zeros_like(host)
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = y0 + 0.03 * bh
    rx0 = int(round(cx - 0.5 * center_width * bw))
    rx1 = int(round(cx + 0.5 * center_width * bw))
    ry0 = int(round(y0 - above_height * bh))
    ry1 = int(round(y0 + overlap_height * bh))
    contact_box = _box_mask_like(host, (rx0, ry0, rx1, ry1))

    h, w = host.shape[-2:]
    yy = torch.arange(h, device=host.device, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=host.device, dtype=torch.float32).view(1, 1, 1, w)
    sigma_x = max(1.0, float(sigma_x_scale) * float(bw))
    sigma_y = max(1.0, float(sigma_y_scale) * float(bh))
    gaussian = torch.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
    contact = contact_box.float() * gaussian
    return normalize_spatial_map(contact).to(device=host.device, dtype=host.dtype)


def build_face_accessory_region(
    host_mask: torch.Tensor,
    upper_height: float = 0.42,
    center_width: float = 0.74,
) -> torch.Tensor:
    host = normalize_spatial_map(host_mask)
    bbox = _binary_bbox(host)
    if bbox is None:
        return host
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    sx0 = int(round(cx - 0.5 * center_width * bw))
    sx1 = int(round(cx + 0.5 * center_width * bw))
    sy0 = int(round(y0 + 0.10 * bh))
    sy1 = int(round(y0 + max(0.12, upper_height) * bh))
    band = _box_mask_like(host, (sx0, sy0, sx1, sy1))
    region = torch.minimum(host, band)
    if float(region.detach().float().max().item()) <= 1e-6:
        region = band
    return normalize_spatial_map(region).to(device=host.device, dtype=host.dtype)


def build_relation_region(
    relation: str,
    host_mask: torch.Tensor | None,
    grounding_mask: torch.Tensor | None,
    removed_attention_map: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    relation = (relation or "none").strip().lower()
    base_host = grounding_mask if grounding_mask is not None else host_mask
    if relation in {"none", "auto", ""}:
        return None
    if relation in {"above_host", "above"} and base_host is not None:
        return build_above_host_region(
            base_host,
            expand_x=-0.15,
            above_height=0.42,
            overlap_height=0.02,
        )
    if relation in {"on_surface", "surface"} and base_host is not None:
        return build_surface_region(base_host)
    if relation in {"remove_source_object", "removed_object"}:
        if grounding_mask is not None:
            return grounding_mask
        if removed_attention_map is not None:
            return removed_attention_map
    if relation in {"on_face", "face", "eye_band", "eyes"} and base_host is not None:
        return build_face_accessory_region(base_host)
    if relation in {"inside_host", "inside"} and base_host is not None:
        return normalize_spatial_map(base_host)
    return None


def _response(clean: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return normalize_spatial_map(0.65 * normalize_spatial_map(clean) + 0.35 * normalize_spatial_map(velocity))


def attention_localization_confidence(attention_map: torch.Tensor, top_fraction: float = 0.10) -> float:
    attention = normalize_spatial_map(attention_map).detach().float().flatten(1)
    k = max(1, int(round(attention.shape[1] * float(top_fraction))))
    top_mass = attention.topk(k, dim=1).values.sum(dim=1)
    total_mass = attention.sum(dim=1).clamp_min(1e-6)
    return float((top_mass / total_mass).mean().item())


def build_support_candidates(
    attention_map: torch.Tensor,
    clean_map: torch.Tensor,
    velocity_map: torch.Tensor,
    host_attention_map: torch.Tensor | None = None,
    removed_attention_map: torch.Tensor | None = None,
    grounding_mask: torch.Tensor | None = None,
    relation_map: torch.Tensor | None = None,
    attention_power: float = 1.0,
    disagreement_power: float = 1.0,
) -> dict[str, torch.Tensor]:
    attention = normalize_spatial_map(attention_map).pow(max(0.0, float(attention_power)))
    host = normalize_spatial_map(host_attention_map).pow(max(0.0, float(attention_power))) if host_attention_map is not None else attention
    removed = (
        normalize_spatial_map(removed_attention_map).pow(max(0.0, float(attention_power)))
        if removed_attention_map is not None
        else attention
    )
    clean = normalize_spatial_map(clean_map).pow(max(0.0, float(disagreement_power)))
    velocity = normalize_spatial_map(velocity_map).pow(max(0.0, float(disagreement_power)))
    response = _response(clean, velocity)
    candidates: dict[str, torch.Tensor] = {
        "attention_only": attention,
        "clean_disagreement_only": clean,
        "velocity_disagreement_only": velocity,
        "attention_x_clean": attention * clean,
        "attention_x_velocity": attention * velocity,
        "host_x_clean": host * clean,
        "new_x_host_x_clean": attention * host * clean,
        "new_plus_host_x_clean": torch.maximum(attention, 0.65 * host) * clean,
        "removed_src_x_clean": removed * clean,
        "removed_src_x_velocity": removed * velocity,
        "src_tar_attn_x_clean": torch.maximum(attention, removed) * clean,
    }
    if grounding_mask is not None:
        seg = normalize_spatial_map(grounding_mask)
        host_spawn_center = build_spawn_center_region(
            seg,
            center_width=0.75,
            center_height=0.55,
            sigma_scale=0.32,
        )
        host_spawn_wide = build_spawn_center_region(
            seg,
            center_width=0.95,
            center_height=0.72,
            sigma_scale=0.42,
        )
        host_top_contact = build_host_top_contact_region(seg)
        candidates.update(
            {
                "seg_only": seg,
                "seg_x_clean": seg * clean,
                "seg_x_velocity": seg * velocity,
                "seg_x_response": seg * response,
                "host_spawn_center": host_spawn_center,
                "host_spawn_center_x_response": normalize_spatial_map(
                    host_spawn_center * (0.35 + 0.65 * response)
                ),
                "host_spawn_wide": host_spawn_wide,
                "host_spawn_wide_x_response": normalize_spatial_map(
                    host_spawn_wide * (0.30 + 0.70 * response)
                ),
                "host_top_contact": host_top_contact,
                "host_top_contact_x_response": normalize_spatial_map(
                    host_top_contact * (0.35 + 0.65 * response)
                ),
            }
        )
    if relation_map is not None:
        relation = normalize_spatial_map(relation_map)
        candidates.update(
            {
                "relation_only": relation,
                "relation_x_clean": relation * clean,
                "relation_x_velocity": relation * velocity,
                "relation_x_response": relation * response,
            }
        )
    if relation_map is not None or grounding_mask is not None:
        surface = normalize_spatial_map(relation_map if relation_map is not None else grounding_mask)
        spawn_center = build_spawn_center_region(surface)
        spawn_lower_center = build_spawn_center_region(
            surface,
            center_width=0.55,
            center_height=0.35,
            sigma_scale=0.25,
            center_y=0.82,
        )
        surface_attention = normalize_within_mask(attention_map, surface).pow(max(0.0, float(attention_power)))
        surface_clean = normalize_within_mask(clean_map, surface).pow(max(0.0, float(disagreement_power)))
        surface_velocity = normalize_within_mask(velocity_map, surface).pow(max(0.0, float(disagreement_power)))
        surface_response = normalize_spatial_map(
            surface
            * (
                0.30 * surface_attention
                + 0.50 * surface_clean
                + 0.20 * surface_velocity
            )
        )
        candidates.update(
            {
                "host_surface_x_clean": surface * clean,
                "host_surface_x_response": surface * response,
                "new_x_surface_x_clean": attention * surface * clean,
                "surface_local_clean": surface_clean,
                "surface_local_response": surface_response,
                "decal_surface_local_response": surface_response,
                "new_x_surface_local_response": normalize_spatial_map(attention * surface_response),
                "spawn_center": spawn_center,
                "spawn_center_x_response": normalize_spatial_map(spawn_center * (0.35 + 0.65 * surface_response)),
                "new_x_spawn_center": normalize_spatial_map(torch.maximum(attention, 0.45 * spawn_center) * spawn_center),
                "spawn_lower_center": spawn_lower_center,
                "spawn_lower_center_x_response": normalize_spatial_map(
                    spawn_lower_center * (0.35 + 0.65 * surface_response)
                ),
            }
        )
    return {name: normalize_spatial_map(score) for name, score in candidates.items()}


def default_candidate_for_operation(
    edit_operation: str,
    relation: str = "auto",
    has_grounding: bool = False,
    has_relation: bool = False,
) -> str:
    op = parse_edit_operation(edit_operation)
    rel = (relation or "auto").strip().lower()
    if op == "add_object":
        if has_relation and rel in {"inside_host", "inside"}:
            return "surface_local_response"
        if has_relation and rel not in {"none", "auto", ""}:
            return "relation_x_response"
        return "attention_x_clean"
    if op == "add_decal":
        if has_relation or has_grounding:
            return "decal_surface_local_response"
        return "new_x_host_x_clean"
    if op == "remove_object":
        if has_grounding:
            return "seg_only"
        return "removed_src_x_clean"
    if op == "replace":
        if has_grounding:
            return "seg_only"
        return "src_tar_attn_x_clean"
    if op == "recolor":
        if has_relation:
            return "relation_only"
        if has_grounding:
            return "seg_only"
        return "host_x_clean"
    if has_relation:
        return "relation_x_response"
    if has_grounding:
        return "seg_x_clean"
    return "attention_x_clean"


def score_support_candidate(
    candidate: torch.Tensor,
    clean_map: torch.Tensor,
    target_area: float = 0.08,
    grounding_mask: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.15,
    gamma: float = 0.15,
    delta: float = 0.20,
) -> float:
    mask = (candidate.detach().float() > 0.5).float()
    area = float(mask.mean().item())
    clean = normalize_spatial_map(clean_map).detach().float()
    edit_response = float((mask * clean).sum().item() / max(float(mask.sum().item()), 1e-6))
    preserve_leakage = float(((1.0 - mask) * clean).mean().item())
    area_penalty = abs(area - float(target_area))
    grounding_confidence = 0.0
    if grounding_mask is not None:
        grounding = (_match_map(grounding_mask, candidate).detach().float() > 0.5).float()
        grounding_confidence = float((mask * grounding).sum().item() / max(float(mask.sum().item()), 1e-6))
    return (
        float(alpha) * edit_response
        - float(beta) * preserve_leakage
        - float(gamma) * area_penalty
        + float(delta) * grounding_confidence
    )


def postprocess_support(
    support_score: torch.Tensor,
    top_percentile: float = 90.0,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.30,
    keep_components: int = 2,
    dilate_radius: int = 5,
    blur_kernel: int = 5,
    component_score_map: torch.Tensor | None = None,
    relation_map: torch.Tensor | None = None,
    target_area_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    score = normalize_spatial_map(support_score)
    percentiles = sorted(
        {
            min(99.0, top_percentile + 8.0),
            min(99.0, top_percentile + 5.0),
            top_percentile,
            max(25.0, top_percentile - 5.0),
            max(25.0, top_percentile - 10.0),
            max(25.0, top_percentile - 15.0),
            max(25.0, top_percentile - 20.0),
            max(25.0, top_percentile - 30.0),
            max(25.0, top_percentile - 40.0),
            50.0,
            35.0,
            25.0,
        },
        reverse=True,
    )
    best_core = None
    best_stats: dict[str, float | int] = {}
    best_area_penalty = float("inf")
    use_scored_components = component_score_map is not None or relation_map is not None
    target_area = (
        float(target_area_ratio)
        if target_area_ratio is not None
        else 0.5 * (max(0.0, float(min_area_ratio)) + max(0.0, float(max_area_ratio)))
    )
    for percentile in percentiles:
        threshold = float(torch.quantile(score.flatten(1), percentile / 100.0).mean().item())
        score_max = float(score.detach().float().max().item())
        if score_max > 0.0 and threshold >= score_max:
            threshold = max(0.0, score_max - 1e-6)
        if use_scored_components:
            core, num_components, top_component_area, component_score = _top_scored_components(
                score,
                threshold,
                keep_components,
                component_score_map=component_score_map,
                relation_map=relation_map,
                target_area_ratio=target_area,
            )
        else:
            core, num_components, top_component_area, component_score = _top_scored_components(
                score,
                threshold,
                keep_components,
                component_score_map=score,
                relation_map=None,
                target_area_ratio=target_area,
            )
        area = _area(core)
        if min_area_ratio > 0.0 and area < min_area_ratio:
            bound_penalty = float(min_area_ratio - area)
        elif max_area_ratio > 0.0 and area > max_area_ratio:
            bound_penalty = float(area - max_area_ratio)
        else:
            bound_penalty = 0.0
        area_penalty = bound_penalty + 0.10 * abs(float(area) - float(target_area))
        stats = {
            "support_threshold": threshold,
            "support_top_percentile": float(percentile),
            "support_area_core_raw": area,
            "support_num_components": int(num_components),
            "support_top_component_area": float(top_component_area),
            "support_component_score": float(component_score),
            "support_component_scoring": int(use_scored_components),
            "support_target_area_ratio": float(target_area),
        }
        if area_penalty < best_area_penalty:
            best_core = core
            best_stats = stats
            best_area_penalty = area_penalty
        if (min_area_ratio <= 0.0 or area >= min_area_ratio) and (max_area_ratio <= 0.0 or area <= max_area_ratio):
            best_core = core
            best_stats = stats
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


def _area(mask: torch.Tensor, threshold: float = 0.5) -> float:
    return float((mask.detach().float().clamp(0.0, 1.0) > threshold).float().mean().item())


def _top_scored_components(
    score: torch.Tensor,
    threshold: float,
    keep_components: int,
    component_score_map: torch.Tensor | None = None,
    relation_map: torch.Tensor | None = None,
    target_area_ratio: float = 0.05,
) -> tuple[torch.Tensor, int, float, float]:
    image = score.detach().float().cpu()
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW support score, got {tuple(score.shape)}")
    component_ref = _match_map(component_score_map, score).detach().float().cpu() if component_score_map is not None else image
    relation_ref = _match_map(relation_map, score).detach().float().cpu() if relation_map is not None else None
    out = torch.zeros_like(image)
    total_components = 0
    top_area = 0.0
    top_score = 0.0
    bsz, _, h, w = image.shape
    for b in range(bsz):
        plane = image[b, 0].clamp(0.0, 1.0)
        ref_plane = component_ref[b, 0].clamp(0.0, 1.0)
        rel_plane = relation_ref[b, 0].clamp(0.0, 1.0) if relation_ref is not None else None
        binary = plane > threshold
        visited = torch.zeros_like(binary, dtype=torch.bool)
        components: list[tuple[float, float, list[tuple[int, int]]]] = []
        for y in range(h):
            for x in range(w):
                if not bool(binary[y, x]) or bool(visited[y, x]):
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                area = len(points)
                area_ratio = area / max(1, h * w)
                support_mass = sum(float(plane[py, px].item()) for py, px in points) / max(1, area)
                clean_mass = sum(float(ref_plane[py, px].item()) for py, px in points) / max(1, area)
                if rel_plane is None:
                    relation_overlap = 0.0
                else:
                    relation_overlap = sum(float(rel_plane[py, px].item()) for py, px in points) / max(1, area)
                perimeter = 0
                point_set = set(points)
                for py, px in points:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        if (py + dy, px + dx) not in point_set:
                            perimeter += 1
                compactness = (4.0 * math.pi * area) / max(1.0, float(perimeter * perimeter))
                area_penalty = abs(area_ratio - float(target_area_ratio))
                score_value = (
                    1.00 * support_mass
                    + 0.70 * clean_mass
                    - 0.80 * area_penalty
                    + 0.15 * compactness
                    + 0.60 * relation_overlap
                )
                components.append((float(score_value), float(area_ratio), points))
        components.sort(key=lambda item: item[0], reverse=True)
        total_components += len(components)
        if components:
            top_area = max(top_area, components[0][1])
            top_score = max(top_score, components[0][0])
        for _, _, points in components[: max(1, int(keep_components))]:
            for py, px in points:
                out[b, 0, py, px] = image[b, 0, py, px]
    return out.to(device=score.device, dtype=score.dtype), total_components, float(top_area), float(top_score)


def build_core_ring_preserve_masks(
    core_mask: torch.Tensor,
    ring_dilate_radius: int = 7,
    ring_scale: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core = normalize_spatial_map(core_mask)
    wide = core
    if ring_dilate_radius > 1:
        kernel = int(ring_dilate_radius)
        if kernel % 2 == 0:
            kernel += 1
        wide = F.max_pool2d(core.float(), kernel_size=kernel, stride=1, padding=kernel // 2).to(core.dtype)
    ring = (wide - core).clamp(0.0, 1.0) * float(ring_scale)
    preserve = (1.0 - torch.maximum(core, ring)).clamp(0.0, 1.0)
    return core.clamp(0.0, 1.0), ring.clamp(0.0, 1.0), preserve


def support_overlap_metrics(pred_mask: torch.Tensor, reference_mask: torch.Tensor) -> dict[str, float]:
    pred = (_match_map(pred_mask, reference_mask).detach().float() > 0.5).float()
    ref = (normalize_spatial_map(reference_mask).detach().float() > 0.5).float()
    intersection = float((pred * ref).sum().item())
    pred_area = float(pred.sum().item())
    ref_area = float(ref.sum().item())
    union = pred_area + ref_area - intersection
    return {
        "support_iou": intersection / max(union, 1e-6),
        "support_coverage": intersection / max(ref_area, 1e-6),
        "support_leakage": max(0.0, pred_area - intersection) / max(pred_area, 1e-6),
        "support_pred_area": pred_area / max(1, pred.numel()),
        "support_reference_area": ref_area / max(1, ref.numel()),
    }


def _save_mask(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plane = mask.detach().float().clamp(0.0, 1.0)
    if plane.ndim == 4:
        plane = plane[0, 0]
    elif plane.ndim == 3:
        plane = plane[0]
    array = (plane.cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def _bbox_fraction(mask: torch.Tensor, threshold: float = 0.5) -> list[float]:
    bbox = _binary_bbox(mask, threshold=threshold)
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = bbox
    h, w = mask.shape[-2:]
    return [x0 / max(1, w), y0 / max(1, h), x1 / max(1, w), y1 / max(1, h)]


def save_support_debug(result: OperationSupportV3Result, output_dir: str | Path, max_candidates: int = 12) -> None:
    output = Path(output_dir)
    _save_mask(result.support_score, output / "operation_v3_support_score.png")
    _save_mask(result.support_score, output / "support_score_selected.png")
    _save_mask(result.edit_mask, output / "selected_candidate_postprocessed.png")
    _save_mask(result.edit_mask, output / "operation_v3_edit_mask.png")
    _save_mask(result.core_mask, output / "M_core.png")
    _save_mask(result.core_mask, output / "operation_v3_core_mask.png")
    _, ring, preserve = build_core_ring_preserve_masks(result.core_mask)
    _save_mask(ring, output / "M_ring.png")
    _save_mask(ring, output / "operation_v3_ring_mask.png")
    _save_mask(preserve, output / "M_preserve.png")
    _save_mask(preserve, output / "operation_v3_preserve_mask.png")
    selected_name = str(result.stats.get("support_score", ""))
    if selected_name in result.candidate_scores:
        _save_mask(
            result.candidate_scores[selected_name],
            output / f"operation_v3_selected_candidate_{selected_name.replace('/', '_')}.png",
        )
        _save_mask(result.candidate_scores[selected_name], output / "selected_candidate_raw.png")
    if result.grounding_mask is not None:
        _save_mask(result.grounding_mask, output / "operation_v3_grounding_mask.png")
        _save_mask(result.grounding_mask, output / "grounding_mask.png")
    if result.relation_map is not None:
        _save_mask(result.relation_map, output / "operation_v3_relation_map.png")
        _save_mask(result.relation_map, output / "relation_region.png")
        _save_mask(result.relation_map, output / "surface_region.png")
    for idx, (name, score) in enumerate(sorted(result.candidate_scores.items())):
        if idx >= max_candidates:
            break
        _save_mask(score, output / f"operation_v3_candidate_{name.replace('/', '_')}.png")
    metadata = {
        "selected_candidate": selected_name,
        "operation": result.stats.get("support_edit_operation", "auto"),
        "relation": result.stats.get("support_relation", "auto"),
        "support_area": result.stats.get("support_area_edit", _area(result.edit_mask)),
        "support_bbox": _bbox_fraction(result.edit_mask),
        "num_components": result.stats.get("support_num_components", 0),
        "component_score": result.stats.get("support_component_score", 0.0),
        "iou_to_manual": result.stats.get("support_iou", 0.0),
        "coverage_to_manual": result.stats.get("support_coverage", 0.0),
        "leakage_to_manual": result.stats.get("support_leakage", 0.0),
        "stats": result.stats,
    }
    (output / "operation_v3_debug_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def choose_support_candidate(
    candidates: dict[str, torch.Tensor],
    candidate_name: str | None,
    edit_operation: str,
    relation: str,
    has_grounding: bool,
    has_relation: bool,
    clean_map: torch.Tensor,
    grounding_mask: torch.Tensor | None = None,
) -> tuple[str, torch.Tensor]:
    requested = (candidate_name or "auto").strip()
    if requested in {"", "auto", "operation_default"}:
        requested = default_candidate_for_operation(
            edit_operation,
            relation=relation,
            has_grounding=has_grounding,
            has_relation=has_relation,
        )
    if requested == "score_auto":
        scored = [
            (score_support_candidate(score, clean_map, grounding_mask=grounding_mask), name, score)
            for name, score in candidates.items()
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1], scored[0][2]
    if requested not in candidates:
        fallback = default_candidate_for_operation(
            edit_operation,
            relation=relation,
            has_grounding=has_grounding,
            has_relation=has_relation,
        )
        if fallback not in candidates:
            fallback = "attention_x_clean"
        return fallback, candidates[fallback]
    return requested, candidates[requested]


def build_operation_support_v3(
    attention_map: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    host_attention_map: torch.Tensor | None = None,
    removed_attention_map: torch.Tensor | None = None,
    grounding_mask: torch.Tensor | None = None,
    edit_operation: str = "auto",
    relation: str = "auto",
    candidate: str | None = "auto",
    attention_power: float = 1.0,
    disagreement_power: float = 1.0,
    top_percentile: float = 90.0,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.30,
    keep_components: int = 2,
    dilate_radius: int = 5,
    blur_kernel: int = 5,
    clean_map_override: torch.Tensor | None = None,
    velocity_map_override: torch.Tensor | None = None,
    temporal_aggregation: str = "single",
    temporal_steps: int = 1,
) -> OperationSupportV3Result:
    parsed_operation = parse_edit_operation(edit_operation)
    attention = compute_token_attention(attention_map, x_t)
    if attention is None:
        raise ValueError("operation support v3 requires an attention_map")
    host = compute_token_attention(host_attention_map, x_t)
    removed = compute_token_attention(removed_attention_map, x_t)
    grounding = ground_object_mask(grounding_mask, x_t)
    clean = (
        _match_map(clean_map_override, x_t)
        if clean_map_override is not None
        else compute_clean_disagreement(x_t, t, source_velocity, target_velocity)
    )
    velocity = (
        _match_map(velocity_map_override, x_t)
        if velocity_map_override is not None
        else compute_velocity_disagreement(source_velocity, target_velocity)
    )
    relation_map = build_relation_region(
        relation,
        host_mask=host,
        grounding_mask=grounding,
        removed_attention_map=removed,
        reference=x_t,
    )
    candidates = build_support_candidates(
        attention,
        clean,
        velocity,
        host_attention_map=host,
        removed_attention_map=removed,
        grounding_mask=grounding,
        relation_map=relation_map,
        attention_power=attention_power,
        disagreement_power=disagreement_power,
    )
    selected_name, score = choose_support_candidate(
        candidates,
        candidate,
        edit_operation=edit_operation,
        relation=relation,
        has_grounding=grounding is not None,
        has_relation=relation_map is not None,
        clean_map=clean,
        grounding_mask=grounding,
    )
    use_component_scoring = parsed_operation == "add_object" and relation_map is not None
    component_score_ref = score if parsed_operation == "add_object" else clean
    edit, core, stats = postprocess_support(
        score,
        top_percentile=top_percentile,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        keep_components=keep_components,
        dilate_radius=dilate_radius,
        blur_kernel=blur_kernel,
        component_score_map=component_score_ref if use_component_scoring else None,
        relation_map=(relation_map if relation_map is not None else grounding) if use_component_scoring else None,
    )
    stats.update(
        {
            "support_mode": "operation_v3",
            "support_edit_operation": parsed_operation,
            "support_relation": relation,
            "support_score": selected_name,
            "support_requested_candidate": candidate or "auto",
            "support_has_grounding": int(grounding is not None),
            "support_has_relation": int(relation_map is not None),
            "support_candidate_count": int(len(candidates)),
            "support_attention_power": float(attention_power),
            "support_disagreement_power": float(disagreement_power),
            "support_min_area_ratio": float(min_area_ratio),
            "support_max_area_ratio": float(max_area_ratio),
            "support_keep_components": int(keep_components),
            "support_dilate_radius": int(dilate_radius),
            "support_blur_kernel": int(blur_kernel),
            "support_temporal_aggregation": temporal_aggregation,
            "support_temporal_steps": int(temporal_steps),
            "support_attention_local_confidence": attention_localization_confidence(attention),
        }
    )
    if grounding is not None:
        stats.update(support_overlap_metrics(edit, grounding))
    return OperationSupportV3Result(
        edit_mask=edit.to(device=x_t.device, dtype=x_t.dtype),
        core_mask=core.to(device=x_t.device, dtype=x_t.dtype),
        attention_map=attention.to(device=x_t.device, dtype=x_t.dtype),
        host_attention_map=host.to(device=x_t.device, dtype=x_t.dtype) if host is not None else None,
        removed_attention_map=removed.to(device=x_t.device, dtype=x_t.dtype) if removed is not None else None,
        grounding_mask=grounding.to(device=x_t.device, dtype=x_t.dtype) if grounding is not None else None,
        relation_map=relation_map.to(device=x_t.device, dtype=x_t.dtype) if relation_map is not None else None,
        clean_disagreement_map=clean.to(device=x_t.device, dtype=x_t.dtype),
        velocity_disagreement_map=velocity.to(device=x_t.device, dtype=x_t.dtype),
        support_score=score.to(device=x_t.device, dtype=x_t.dtype),
        candidate_scores={name: value.to(device=x_t.device, dtype=x_t.dtype) for name, value in candidates.items()},
        stats=stats,
    )
