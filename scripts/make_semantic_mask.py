from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from attention_mask import _changed_words, _content_edit_words


_SUPPORT_PLAN_RULES: tuple[tuple[set[str], str, str, dict[str, float]], ...] = (
    (
        {"crown", "tiara", "hat", "cap", "helmet", "halo"},
        "head",
        "top_center",
        {"expand_x": 0.05, "band_ratio": 0.42, "overlap_ratio": 0.06},
    ),
    (
        {"sunglasses", "glasses", "eyeglasses", "goggles", "monocle"},
        "eyes",
        "box",
        {"expand_x": 0.45, "expand_y": 0.35},
    ),
    ({"earring", "earrings"}, "ears", "inside", {}),
    ({"scarf", "necklace", "collar", "tie", "bowtie"}, "neck", "around", {"band_ratio": 0.35}),
    ({"shirt", "jacket", "coat", "dress", "sweater", "hoodie"}, "torso", "inside", {}),
    ({"shoes", "shoe", "boots", "boot", "socks", "sock"}, "feet", "inside", {}),
    ({"watch", "bracelet", "glove", "gloves"}, "hands", "inside", {}),
    ({"backpack", "wings", "cape"}, "back", "around", {"band_ratio": 0.40}),
)


_SUBJECT_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "with",
    "of",
    "same",
    "its",
    "wearing",
    "sitting",
    "standing",
    "walking",
    "running",
    "grass",
    "forest",
    "photo",
    "photograph",
    "realistic",
    "high",
    "resolution",
}


_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "golden",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}


def _word_tokens(text: str) -> list[str]:
    token = []
    tokens = []
    for char in text.lower():
        if char.isalnum():
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def _source_subject_prefix(source_prompt: str) -> str:
    for word in _word_tokens(source_prompt):
        if word not in _SUBJECT_STOPWORDS and word not in _COLOR_WORDS:
            return word
    return ""


def _infer_support_plan(
    source_prompt: str,
    target_prompt: str,
    explicit_phrase: str | None,
    requested_relation: str,
) -> tuple[str, str, dict[str, object]]:
    _, tar_changed = _changed_words(source_prompt, target_prompt)
    edit_words = _content_edit_words(tar_changed, max_words=6)
    edit_word_set = set(edit_words)
    relation = requested_relation
    anchor_phrase = explicit_phrase.strip() if explicit_phrase else ""
    matched_rule = None
    rule_params = {}
    for keywords, anchor, rule_relation, params in _SUPPORT_PLAN_RULES:
        if edit_word_set & keywords:
            rule_params = dict(params)
            matched_rule = {
                "keywords": sorted(keywords),
                "anchor": anchor,
                "relation": rule_relation,
                "params": rule_params,
            }
            if not anchor_phrase:
                subject = _source_subject_prefix(source_prompt)
                if subject and anchor in {"head", "eyes", "ears", "neck", "torso", "feet", "hands", "back"}:
                    anchor_phrase = f"{subject} {anchor}"
                else:
                    anchor_phrase = anchor
            if relation == "auto":
                relation = rule_relation
            break
    if matched_rule is None and edit_word_set and edit_word_set <= _COLOR_WORDS:
        subject = _source_subject_prefix(source_prompt)
        if subject:
            anchor_phrase = anchor_phrase or subject
            if relation == "auto":
                relation = "inside"
            matched_rule = {
                "keywords": sorted(edit_word_set),
                "anchor": subject,
                "relation": relation,
                "params": {},
                "type": "color_attribute",
            }
    if not anchor_phrase:
        anchor_phrase = " ".join(edit_words).strip()
    if not anchor_phrase:
        raise ValueError("Could not infer an edit phrase; pass --phrase explicitly.")
    if relation == "auto":
        relation = "inside"
    return anchor_phrase, relation, {
        "support_relation_requested": requested_relation,
        "support_relation_inferred": relation,
        "semantic_phrase_explicit": explicit_phrase,
        "semantic_phrase_inferred": anchor_phrase,
        "semantic_edit_words": edit_words,
        "support_plan_rule": matched_rule,
        "support_plan_params": rule_params,
    }


def _mask_box_area_ratio(mask: np.ndarray, threshold: float = 0.2) -> float:
    bbox = _bbox_from_mask(mask, threshold=threshold)
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    height, width = np.asarray(mask).shape[:2]
    return float(((x1 - x0) * (y1 - y0)) / max(1, height * width))


def _should_fallback_large_eyes_anchor(
    anchor_mask: np.ndarray,
    phrase: str,
    relation: str,
    *,
    area_threshold: float,
    box_area_threshold: float,
    support_threshold: float,
) -> tuple[bool, dict[str, object]]:
    area_ratio = float((np.asarray(anchor_mask) > support_threshold).mean())
    box_area_ratio = _mask_box_area_ratio(anchor_mask, threshold=support_threshold)
    phrase_tokens = set(_word_tokens(phrase))
    should_fallback = (
        relation == "box"
        and "eyes" in phrase_tokens
        and (area_ratio > area_threshold or box_area_ratio > box_area_threshold)
    )
    return should_fallback, {
        "eyes_anchor_area_ratio": area_ratio,
        "eyes_anchor_box_area_ratio": box_area_ratio,
        "eyes_anchor_area_threshold": float(area_threshold),
        "eyes_anchor_box_area_threshold": float(box_area_threshold),
    }


def _bbox_from_mask(mask: np.ndarray, threshold: float = 0.2) -> tuple[int, int, int, int] | None:
    binary = np.asarray(mask) > threshold
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _clamp_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    ix0 = max(0, min(width, int(np.floor(x0))))
    iy0 = max(0, min(height, int(np.floor(y0))))
    ix1 = max(0, min(width, int(np.ceil(x1))))
    iy1 = max(0, min(height, int(np.ceil(y1))))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


def _expanded_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    expand_x: float,
    expand_y: float,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    return _clamp_box(
        x0 - expand_x * bw,
        y0 - expand_y * bh,
        x1 + expand_x * bw,
        y1 + expand_y * bh,
        width,
        height,
    )


def _profile_eye_support_from_anchor(
    anchor: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = anchor.shape[:2]
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx_mid = x0 + 0.5 * bw
    upper_y1 = y0 + int(round(0.62 * bh))
    upper = anchor[y0:upper_y1, x0:x1] > threshold
    left_mass = int(upper[:, : max(1, int(round(0.5 * bw)))].sum())
    right_mass = int(upper[:, max(1, int(round(0.5 * bw))) :].sum())
    side = "right" if right_mass >= left_mass else "left"

    lens_cx = x0 + (0.82 if side == "right" else 0.18) * bw
    lens_cy = y0 + 0.35 * bh
    lens_w = max(4.0, 0.17 * bw)
    lens_h = max(3.0, 0.12 * bh)
    lens_box = _clamp_box(
        lens_cx - 0.5 * lens_w,
        lens_cy - 0.5 * lens_h,
        lens_cx + 0.5 * lens_w,
        lens_cy + 0.5 * lens_h,
        width,
        height,
    )

    mask = np.zeros((height, width), dtype=np.float32)
    if lens_box is not None:
        lx0, ly0, lx1, ly1 = lens_box
        center = (int(round(0.5 * (lx0 + lx1 - 1))), int(round(0.5 * (ly0 + ly1 - 1))))
        axes = (max(1, int(round(0.5 * (lx1 - lx0)))), max(1, int(round(0.5 * (ly1 - ly0)))))
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, thickness=-1)

        temple_len = max(3.0, 0.16 * bw)
        temple_h = max(2.0, 0.035 * bh)
        if side == "right":
            temple_box = _clamp_box(lx1 - 0.10 * lens_w, lens_cy - 0.5 * temple_h, lx1 + temple_len, lens_cy + 0.5 * temple_h, width, height)
        else:
            temple_box = _clamp_box(lx0 - temple_len, lens_cy - 0.5 * temple_h, lx0 + 0.10 * lens_w, lens_cy + 0.5 * temple_h, width, height)
        if temple_box is not None:
            tx0, ty0, tx1, ty1 = temple_box
            mask[ty0:ty1, tx0:tx1] = np.maximum(mask[ty0:ty1, tx0:tx1], 0.85)
    else:
        temple_box = None

    blur = max(3, int(round(min(width, height) * 0.004)) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    if float(mask.max()) > 1e-6:
        mask = mask / float(mask.max())
    support_box = _bbox_from_mask(mask, threshold=0.1)
    return np.clip(mask, 0.0, 1.0), {
        "support_relation": "profile_eye",
        "profile_eye_side": side,
        "profile_eye_left_mass": left_mass,
        "profile_eye_right_mass": right_mass,
        "anchor_box_xyxy": [int(v) for v in bbox],
        "profile_eye_lens_box_xyxy": [] if lens_box is None else [int(v) for v in lens_box],
        "profile_eye_temple_box_xyxy": [] if temple_box is None else [int(v) for v in temple_box],
        "support_box_xyxy": [] if support_box is None else [int(v) for v in support_box],
    }


def _front_glasses_from_head_anchor(
    anchor: np.ndarray,
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    threshold: float,
    auto_eye: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = anchor.shape[:2]
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    eye_centers: list[tuple[float, float]] = []
    candidates: list[dict[str, float]] = []

    if auto_eye:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        head_mask = anchor > threshold
        sx0 = int(round(x0 + 0.20 * bw))
        sx1 = int(round(x1 - 0.20 * bw))
        sy0 = int(round(y0 + 0.22 * bh))
        sy1 = int(round(y0 + 0.58 * bh))
        search = np.zeros_like(head_mask, dtype=np.uint8)
        search[max(0, sy0) : min(height, sy1), max(0, sx0) : min(width, sx1)] = 1
        valid = head_mask & (search > 0)
        if valid.any():
            values = gray[valid]
            dark_threshold = float(np.percentile(values, 22))
            binary = ((gray <= dark_threshold) & valid).astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for label in range(1, count):
                cx, cy = centroids[label]
                px, py, pw, ph, area = stats[label]
                area_ratio = float(area) / float(max(1, bw * bh))
                if area_ratio < 0.0012 or area_ratio > 0.035:
                    continue
                if pw <= 2 or ph <= 2:
                    continue
                nx = (float(cx) - x0) / bw
                ny = (float(cy) - y0) / bh
                if nx < 0.26 or nx > 0.74 or ny < 0.24 or ny > 0.58:
                    continue
                aspect = float(pw) / max(1.0, float(ph))
                if aspect < 0.35 or aspect > 3.6:
                    continue
                score = area_ratio * (1.0 - abs(ny - 0.40)) * (1.0 - 0.25 * abs(nx - 0.5))
                candidates.append(
                    {
                        "cx": float(cx),
                        "cy": float(cy),
                        "nx": nx,
                        "ny": ny,
                        "area_ratio": area_ratio,
                        "score": score,
                    }
                )
            candidates.sort(key=lambda item: item["score"], reverse=True)
            best_pair: tuple[dict[str, float], dict[str, float]] | None = None
            best_pair_score = -1.0
            for i, first in enumerate(candidates):
                for second in candidates[i + 1 :]:
                    left, right = sorted((first, second), key=lambda item: item["cx"])
                    dx = (right["cx"] - left["cx"]) / bw
                    dy = abs(right["cy"] - left["cy"]) / bh
                    if dx < 0.18 or dx > 0.46 or dy > 0.10:
                        continue
                    pair_cx = 0.5 * ((left["cx"] + right["cx"]) / width)
                    pair_nx = 0.5 * (left["nx"] + right["nx"])
                    separation_score = 1.0 - abs(dx - 0.36)
                    symmetry_score = 1.0 - abs(pair_nx - 0.5)
                    pair_score = (
                        left["score"]
                        + right["score"]
                        + 0.15 * separation_score
                        + 0.10 * symmetry_score
                        - 0.15 * abs(pair_cx - 0.5)
                        - 0.35 * dy
                    )
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_pair = (left, right)
            if best_pair is not None:
                eye_centers = [(best_pair[0]["cx"], best_pair[0]["cy"]), (best_pair[1]["cx"], best_pair[1]["cy"])]

    used_auto_eye = len(eye_centers) == 2
    if not used_auto_eye:
        cx = 0.5 * (x0 + x1)
        lens_y = y0 + (0.54 if auto_eye else 0.40) * bh
        half_eye_span = (0.18 if auto_eye else 0.12) * bw
        eye_centers = [(cx - half_eye_span, lens_y), (cx + half_eye_span, lens_y)]

    (left_x, left_y), (right_x, right_y) = sorted(eye_centers, key=lambda point: point[0])
    eye_dist = max(6.0, right_x - left_x)
    lens_w = max(4.0, 0.48 * eye_dist)
    lens_h = max(3.0, 0.62 * lens_w)
    bridge_w = max(2.0, 0.18 * eye_dist)
    bridge_h = max(2.0, 0.12 * lens_h)

    mask = np.zeros((height, width), dtype=np.float32)
    lens_boxes = []
    for lens_cx, lens_cy in ((left_x, left_y), (right_x, right_y)):
        lens_box = _clamp_box(
            lens_cx - 0.5 * lens_w,
            lens_cy - 0.5 * lens_h,
            lens_cx + 0.5 * lens_w,
            lens_cy + 0.5 * lens_h,
            width,
            height,
        )
        if lens_box is None:
            continue
        lx0, ly0, lx1, ly1 = lens_box
        center = (int(round(0.5 * (lx0 + lx1 - 1))), int(round(0.5 * (ly0 + ly1 - 1))))
        axes = (max(1, int(round(0.5 * (lx1 - lx0)))), max(1, int(round(0.5 * (ly1 - ly0)))))
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, thickness=-1)
        lens_boxes.append([int(v) for v in lens_box])
    bridge_cx = 0.5 * (left_x + right_x)
    bridge_cy = 0.5 * (left_y + right_y)
    bridge_box = _clamp_box(
        bridge_cx - 0.5 * bridge_w,
        bridge_cy - 0.5 * bridge_h,
        bridge_cx + 0.5 * bridge_w,
        bridge_cy + 0.5 * bridge_h,
        width,
        height,
    )
    if bridge_box is not None:
        bx0, by0, bx1, by1 = bridge_box
        mask[by0:by1, bx0:bx1] = np.maximum(mask[by0:by1, bx0:bx1], 0.75)

    blur = max(3, int(round(min(width, height) * 0.004)) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    if float(mask.max()) > 1e-6:
        mask = mask / float(mask.max())
    support_box = _bbox_from_mask(mask, threshold=0.1)
    return np.clip(mask, 0.0, 1.0), {
        "support_relation": "front_glasses_auto" if auto_eye else "front_glasses",
        "anchor_box_xyxy": [int(v) for v in bbox],
        "front_glasses_auto_eye_used": bool(used_auto_eye),
        "front_glasses_dark_eye_candidates": len(candidates),
        "front_glasses_dark_eye_candidate_summary": candidates[:8],
        "front_glasses_eye_centers_xy": [[float(x), float(y)] for x, y in eye_centers],
        "front_glasses_lens_boxes_xyxy": lens_boxes,
        "front_glasses_bridge_box_xyxy": [] if bridge_box is None else [int(v) for v in bridge_box],
        "support_box_xyxy": [] if support_box is None else [int(v) for v in support_box],
    }


def support_from_anchor_mask(
    anchor_mask: np.ndarray,
    relation: str,
    image_rgb: np.ndarray | None = None,
    *,
    threshold: float = 0.2,
    expand_x: float = 0.0,
    expand_y: float = 0.0,
    band_ratio: float = 0.55,
    overlap_ratio: float = 0.20,
) -> tuple[np.ndarray, dict[str, object]]:
    """Derive an edit support from a grounded anchor mask.

    The anchor identifies stable source structure; the support identifies where
    generation is allowed. For insertion edits, these are often different.
    """
    anchor = np.asarray(anchor_mask, dtype=np.float32)
    height, width = anchor.shape[:2]
    relation = relation.lower().strip()
    if relation == "inside":
        support = anchor.copy()
    else:
        bbox = _bbox_from_mask(anchor, threshold=threshold)
        if bbox is None:
            raise RuntimeError("Cannot derive support relation from an empty anchor mask.")
        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        band_ratio = max(0.01, float(band_ratio))
        overlap_ratio = max(0.0, float(overlap_ratio))
        if relation == "box":
            box = _expanded_box(bbox, width, height, expand_x, expand_y)
        elif relation == "upper":
            box = _clamp_box(
                x0 - expand_x * bw,
                y0 - expand_y * bh,
                x1 + expand_x * bw,
                y0 + band_ratio * bh,
                width,
                height,
            )
        elif relation == "lower":
            box = _clamp_box(
                x0 - expand_x * bw,
                y1 - band_ratio * bh,
                x1 + expand_x * bw,
                y1 + expand_y * bh,
                width,
                height,
            )
        elif relation == "above":
            box = _clamp_box(
                x0 - expand_x * bw,
                y0 - band_ratio * bh,
                x1 + expand_x * bw,
                y0 + overlap_ratio * bh,
                width,
                height,
            )
        elif relation == "top_center":
            support_w = bw * min(1.0 + 2.0 * expand_x, 0.55 + 2.0 * expand_x)
            cx = 0.5 * (x0 + x1)
            box = _clamp_box(
                cx - 0.5 * support_w,
                y0 - band_ratio * bh,
                cx + 0.5 * support_w,
                y0 + overlap_ratio * bh,
                width,
                height,
            )
        elif relation == "below":
            box = _clamp_box(
                x0 - expand_x * bw,
                y1 - overlap_ratio * bh,
                x1 + expand_x * bw,
                y1 + band_ratio * bh,
                width,
                height,
            )
        elif relation == "left":
            box = _clamp_box(
                x0 - band_ratio * bw,
                y0 - expand_y * bh,
                x0 + overlap_ratio * bw,
                y1 + expand_y * bh,
                width,
                height,
            )
        elif relation == "right":
            box = _clamp_box(
                x1 - overlap_ratio * bw,
                y0 - expand_y * bh,
                x1 + band_ratio * bw,
                y1 + expand_y * bh,
                width,
                height,
            )
        elif relation == "around":
            outer = _expanded_box(bbox, width, height, max(expand_x, band_ratio), max(expand_y, band_ratio))
            inner = _expanded_box(bbox, width, height, 0.0, 0.0)
            support = np.zeros((height, width), dtype=np.float32)
            if outer is not None:
                ox0, oy0, ox1, oy1 = outer
                support[oy0:oy1, ox0:ox1] = 1.0
            if inner is not None:
                ix0, iy0, ix1, iy1 = inner
                support[iy0:iy1, ix0:ix1] *= np.maximum(0.0, 1.0 - anchor[iy0:iy1, ix0:ix1])
            return np.clip(support, 0.0, 1.0), {
                "support_relation": relation,
                "anchor_box_xyxy": [int(v) for v in bbox],
                "support_box_xyxy": [] if outer is None else [int(v) for v in outer],
            }
        elif relation == "profile_eye":
            return _profile_eye_support_from_anchor(anchor, bbox, threshold=threshold)
        elif relation in {"front_glasses", "front_glasses_auto"}:
            if image_rgb is None:
                image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            return _front_glasses_from_head_anchor(
                anchor,
                np.asarray(image_rgb, dtype=np.uint8),
                bbox,
                threshold=threshold,
                auto_eye=relation == "front_glasses_auto",
            )
        else:
            raise ValueError(f"Unsupported --support-relation: {relation}")
        support = np.zeros((height, width), dtype=np.float32)
        if box is not None:
            sx0, sy0, sx1, sy1 = box
            support[sy0:sy1, sx0:sx1] = 1.0
    bbox = _bbox_from_mask(anchor, threshold=threshold)
    support_box = _bbox_from_mask(support, threshold=threshold)
    return np.clip(support, 0.0, 1.0), {
        "support_relation": relation,
        "anchor_box_xyxy": [] if bbox is None else [int(v) for v in bbox],
        "support_box_xyxy": [] if support_box is None else [int(v) for v in support_box],
    }


def _preprocess_image(path: str, max_image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        image = image.resize(
            (max(16, int(round(image.width * scale))), max(16, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    return image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))


def _target_phrase(source_prompt: str, target_prompt: str, explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    _, tar_changed = _changed_words(source_prompt, target_prompt)
    words = _content_edit_words(tar_changed, max_words=4)
    return " ".join(words).strip()


def _mask_from_grounded_sam(
    image: Image.Image,
    phrase: str,
    grounding_model: str,
    sam_model: str,
    box_threshold: float,
    text_threshold: float,
    max_boxes: int,
    max_box_area_ratio: float,
    mask_mode: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor, SamModel, SamProcessor
    except Exception as exc:
        raise RuntimeError(
            "transformers GroundingDINO/SAM classes are unavailable in this environment"
        ) from exc

    try:
        gd_processor = GroundingDinoProcessor.from_pretrained(
            grounding_model,
            local_files_only=local_files_only,
        )
        gd_model = GroundingDinoForObjectDetection.from_pretrained(
            grounding_model,
            local_files_only=local_files_only,
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"GroundingDINO model is not available locally: {grounding_model}. "
            "Download/cache it first or pass --grounding-model to an existing local model."
        ) from exc

    try:
        sam_processor = SamProcessor.from_pretrained(
            sam_model,
            local_files_only=local_files_only,
        )
        sam_model_obj = SamModel.from_pretrained(
            sam_model,
            local_files_only=local_files_only,
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"SAM model is not available locally: {sam_model}. "
            "Download/cache it first or pass --sam-model to an existing local model."
        ) from exc

    gd_model.eval()
    sam_model_obj.eval()
    prompt = phrase if phrase.endswith(".") else f"{phrase}."
    gd_inputs = gd_processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gd_outputs = gd_model(**gd_inputs)
    results = gd_processor.post_process_grounded_object_detection(
        gd_outputs,
        gd_inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels")
    if boxes is None or len(boxes) == 0:
        raise RuntimeError(f"No grounded boxes found for phrase: {phrase!r}")
    width, height = image.size
    if max_box_area_ratio > 0.0:
        box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
        box_area_ratio = (box_wh[:, 0] * box_wh[:, 1]) / float(max(1, width * height))
        keep = box_area_ratio <= max_box_area_ratio
        if bool(keep.any().item()):
            boxes = boxes[keep]
            if scores is not None:
                scores = scores[keep]
            if labels is not None:
                labels = [label for label, flag in zip(labels, keep.detach().cpu().tolist()) if flag]
    if scores is not None and max_boxes > 0 and len(boxes) > max_boxes:
        order = torch.argsort(scores.detach().cpu(), descending=True)[:max_boxes].to(boxes.device)
        boxes = boxes[order]
        scores = scores[order]
        if labels is not None:
            labels = [labels[int(idx)] for idx in order.detach().cpu().tolist()]

    box_mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes.detach().cpu().tolist():
        x0, y0, x1, y1 = box
        x0 = max(0, min(width, int(np.floor(x0))))
        x1 = max(0, min(width, int(np.ceil(x1))))
        y0 = max(0, min(height, int(np.floor(y0))))
        y1 = max(0, min(height, int(np.ceil(y1))))
        if x1 > x0 and y1 > y0:
            box_mask[y0:y1, x0:x1] = 1.0
    if mask_mode == "box":
        union = box_mask
    elif mask_mode not in {"sam", "sam_box_intersect"}:
        raise ValueError(f"Unsupported --mask-mode: {mask_mode}")
    else:
        sam_inputs = sam_processor(image, input_boxes=[boxes.detach().cpu().tolist()], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outputs = sam_model_obj(**sam_inputs)
        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        # Shape is usually (num_boxes, num_masks, H, W). Choose the highest scoring
        # SAM mask for each grounded box, then union boxes.
        masks = masks.float()
        if masks.ndim == 4:
            masks = masks.max(dim=1).values
        union = masks.max(dim=0).values.clamp(0.0, 1.0).numpy()
        if mask_mode == "sam_box_intersect":
            union = union * box_mask
    meta = {
        "phrase": phrase,
        "grounding_model": grounding_model,
        "sam_model": sam_model,
        "mask_mode": mask_mode,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "max_box_area_ratio": float(max_box_area_ratio),
        "num_boxes": int(len(boxes)),
        "boxes_xyxy": [[float(v) for v in box] for box in boxes.detach().cpu().tolist()],
        "scores": [] if scores is None else [float(v) for v in scores.detach().cpu().tolist()],
        "labels": [] if labels is None else [str(v) for v in labels],
    }
    return union, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a semantic base mask for semantic_velocity editing.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-prompt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--anchor-output", default=None)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        default=False,
        help="Allow HuggingFace downloads. By default the script only uses local cached models.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.25)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--blur", type=int, default=0)
    parser.add_argument(
        "--support-relation",
        choices=(
            "auto",
            "inside",
            "box",
            "upper",
            "lower",
            "above",
            "top_center",
            "below",
            "left",
            "right",
            "around",
            "profile_eye",
            "front_glasses",
            "front_glasses_auto",
        ),
        default="auto",
        help="Spatial relation used to derive editable support from the grounded anchor.",
    )
    parser.add_argument("--support-expand-x", type=float, default=0.0)
    parser.add_argument("--support-expand-y", type=float, default=0.0)
    parser.add_argument("--support-band-ratio", type=float, default=0.55)
    parser.add_argument("--support-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--support-threshold", type=float, default=0.2)
    parser.add_argument("--eyes-anchor-max-area-ratio", type=float, default=0.16)
    parser.add_argument("--eyes-anchor-max-box-area-ratio", type=float, default=0.34)
    parser.add_argument(
        "--disable-eyes-anchor-fallback",
        action="store_true",
        help="Disable auto fallback from an oversized eyes anchor to subject head + upper support.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("box", "sam", "sam_box_intersect"),
        default="sam_box_intersect",
    )
    args = parser.parse_args()

    image = _preprocess_image(args.image, args.max_image_size)
    phrase, support_relation, plan_meta = _infer_support_plan(
        args.source_prompt,
        args.prompt,
        args.phrase,
        args.support_relation,
    )
    support_expand_x = args.support_expand_x
    support_expand_y = args.support_expand_y
    support_band_ratio = args.support_band_ratio
    support_overlap_ratio = args.support_overlap_ratio
    if args.support_relation == "auto":
        plan_params = plan_meta.get("support_plan_params") or {}
        support_expand_x = float(plan_params.get("expand_x", support_expand_x))
        support_expand_y = float(plan_params.get("expand_y", support_expand_y))
        support_band_ratio = float(plan_params.get("band_ratio", support_band_ratio))
        support_overlap_ratio = float(plan_params.get("overlap_ratio", support_overlap_ratio))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    anchor_mask, meta = _mask_from_grounded_sam(
        image=image,
        phrase=phrase,
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_boxes=args.max_boxes,
        max_box_area_ratio=args.max_box_area_ratio,
        mask_mode=args.mask_mode,
        device=device,
        local_files_only=not args.allow_download,
    )
    fallback_meta: dict[str, object] = {"eyes_anchor_fallback_used": False}
    if not args.disable_eyes_anchor_fallback:
        use_fallback, fallback_check = _should_fallback_large_eyes_anchor(
            anchor_mask,
            phrase,
            support_relation,
            area_threshold=args.eyes_anchor_max_area_ratio,
            box_area_threshold=args.eyes_anchor_max_box_area_ratio,
            support_threshold=args.support_threshold,
        )
        fallback_meta.update(fallback_check)
        if use_fallback:
            subject = _source_subject_prefix(args.source_prompt)
            fallback_phrase = f"{subject} head" if subject else "head"
            fallback_relation = "profile_eye"
            fallback_expand_x = 0.0
            fallback_expand_y = 0.0
            fallback_band_ratio = 0.22
            anchor_mask, fallback_grounding_meta = _mask_from_grounded_sam(
                image=image,
                phrase=fallback_phrase,
                grounding_model=args.grounding_model,
                sam_model=args.sam_model,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_boxes=args.max_boxes,
                max_box_area_ratio=args.max_box_area_ratio,
                mask_mode=args.mask_mode,
                device=device,
                local_files_only=not args.allow_download,
            )
            meta = fallback_grounding_meta
            phrase = fallback_phrase
            support_relation = fallback_relation
            support_expand_x = fallback_expand_x
            support_expand_y = fallback_expand_y
            support_band_ratio = fallback_band_ratio
            fallback_meta.update(
                {
                    "eyes_anchor_fallback_used": True,
                    "eyes_anchor_fallback_phrase": fallback_phrase,
                    "eyes_anchor_fallback_relation": fallback_relation,
                }
            )
    fallback_meta.update(
        {
            "semantic_phrase_after_quality_gate": phrase,
            "support_relation_after_quality_gate": support_relation,
        }
    )
    mask, support_meta = support_from_anchor_mask(
        anchor_mask,
        support_relation,
        image_rgb=np.asarray(image.convert("RGB"), dtype=np.uint8),
        threshold=args.support_threshold,
        expand_x=support_expand_x,
        expand_y=support_expand_y,
        band_ratio=support_band_ratio,
        overlap_ratio=support_overlap_ratio,
    )
    if args.dilate > 1:
        kernel = args.dilate + 1 if args.dilate % 2 == 0 else args.dilate
        mask = cv2.dilate(mask.astype(np.float32), np.ones((kernel, kernel), dtype=np.uint8), iterations=1)
    if args.blur > 1:
        kernel = args.blur + 1 if args.blur % 2 == 0 else args.blur
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
        mask = np.clip(mask, 0.0, 1.0)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.anchor_output:
        os.makedirs(os.path.dirname(args.anchor_output), exist_ok=True)
        Image.fromarray((anchor_mask * 255.0).round().astype("uint8"), mode="L").save(args.anchor_output)
    Image.fromarray((mask * 255.0).round().astype("uint8"), mode="L").save(args.output)
    meta.update(
        {
            "image": args.image,
            "source_prompt": args.source_prompt,
            "target_prompt": args.prompt,
            "output": args.output,
            "anchor_output": args.anchor_output,
            "anchor_mask_area_ratio": float((anchor_mask > 0.5).mean()),
            "mask_area_ratio": float((mask > 0.5).mean()),
            "dilate": int(args.dilate),
            "blur": int(args.blur),
            "support_expand_x": float(support_expand_x),
            "support_expand_y": float(support_expand_y),
            "support_band_ratio": float(support_band_ratio),
            "support_overlap_ratio": float(support_overlap_ratio),
            "support_threshold": float(args.support_threshold),
            "eyes_anchor_max_area_ratio": float(args.eyes_anchor_max_area_ratio),
            "eyes_anchor_max_box_area_ratio": float(args.eyes_anchor_max_box_area_ratio),
        }
    )
    meta.update(plan_meta)
    meta.update(fallback_meta)
    meta.update(support_meta)
    if args.metadata_output:
        os.makedirs(os.path.dirname(args.metadata_output), exist_ok=True)
        with open(args.metadata_output, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
