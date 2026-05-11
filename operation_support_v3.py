from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from generic_support import (
    clean_disagreement_map,
    normalize_spatial_map,
    postprocess_support_score,
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
        return build_above_host_region(base_host)
    if relation in {"on_surface", "surface"} and base_host is not None:
        return build_surface_region(base_host)
    if relation in {"remove_source_object", "removed_object"}:
        if grounding_mask is not None:
            return grounding_mask
        if removed_attention_map is not None:
            return removed_attention_map
    if relation in {"on_face", "inside_host", "inside"} and base_host is not None:
        return normalize_spatial_map(base_host)
    return None


def _response(clean: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return normalize_spatial_map(0.65 * normalize_spatial_map(clean) + 0.35 * normalize_spatial_map(velocity))


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
        candidates.update(
            {
                "seg_x_clean": seg * clean,
                "seg_x_velocity": seg * velocity,
                "seg_x_response": seg * response,
            }
        )
    if relation_map is not None:
        relation = normalize_spatial_map(relation_map)
        candidates.update(
            {
                "relation_x_clean": relation * clean,
                "relation_x_velocity": relation * velocity,
                "relation_x_response": relation * response,
            }
        )
    if relation_map is not None or grounding_mask is not None:
        surface = normalize_spatial_map(relation_map if relation_map is not None else grounding_mask)
        candidates.update(
            {
                "host_surface_x_clean": surface * clean,
                "host_surface_x_response": surface * response,
                "new_x_surface_x_clean": attention * surface * clean,
            }
        )
    return {name: normalize_spatial_map(score) for name, score in candidates.items()}


def default_candidate_for_operation(
    edit_operation: str,
    relation: str = "auto",
    has_grounding: bool = False,
    has_relation: bool = False,
) -> str:
    op = (edit_operation or "auto").strip().lower()
    rel = (relation or "auto").strip().lower()
    if op == "add_object":
        if has_relation and rel not in {"none", "auto", "", "on_face"}:
            return "relation_x_response"
        return "attention_x_clean"
    if op == "add_decal":
        if has_relation or has_grounding:
            return "host_surface_x_response"
        return "new_x_host_x_clean"
    if op == "remove_object":
        if has_grounding:
            return "seg_x_response"
        return "removed_src_x_clean"
    if op == "replace":
        if has_grounding:
            return "seg_x_clean"
        return "src_tar_attn_x_clean"
    if has_relation:
        return "relation_x_response"
    if has_grounding:
        return "seg_x_clean"
    return "attention_x_clean"


def score_support_candidate(candidate: torch.Tensor, clean_map: torch.Tensor, target_area: float = 0.08) -> float:
    mask = (candidate.detach().float() > 0.5).float()
    area = float(mask.mean().item())
    edit_response = float((mask * normalize_spatial_map(clean_map).detach().float()).mean().item())
    area_penalty = abs(area - float(target_area))
    return edit_response - 0.15 * area_penalty


def choose_support_candidate(
    candidates: dict[str, torch.Tensor],
    candidate_name: str | None,
    edit_operation: str,
    relation: str,
    has_grounding: bool,
    has_relation: bool,
    clean_map: torch.Tensor,
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
        scored = [(score_support_candidate(score, clean_map), name, score) for name, score in candidates.items()]
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
) -> OperationSupportV3Result:
    attention = _match_map(attention_map, x_t)
    if attention is None:
        raise ValueError("operation support v3 requires an attention_map")
    host = _match_map(host_attention_map, x_t)
    removed = _match_map(removed_attention_map, x_t)
    grounding = _match_map(grounding_mask, x_t)
    clean = clean_disagreement_map(x_t, t, source_velocity, target_velocity)
    velocity = velocity_disagreement_map(source_velocity, target_velocity)
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
            "support_mode": "operation_v3",
            "support_edit_operation": edit_operation,
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
        }
    )
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
