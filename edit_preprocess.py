from __future__ import annotations

import os

import cv2
import numpy as np
import PIL.Image


NormalizedBox = tuple[float, float, float, float]


def parse_normalized_box(value: str | None, arg_name: str = "--edit-mask-box") -> NormalizedBox | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{arg_name} must be formatted as x0,y0,x1,y1")
    box = tuple(float(part) for part in parts)
    if any(coord < 0.0 or coord > 1.0 for coord in box):
        raise ValueError(f"{arg_name} coordinates must be in [0, 1]")
    return box


def parse_word_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    words = [word.strip().lower() for word in value.split(",") if word.strip()]
    return words if words else None


def clamp_normalized_box(box: NormalizedBox) -> NormalizedBox:
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    return x0, y0, x1, y1


def expand_normalized_box(
    box: NormalizedBox,
    min_width: float,
    min_height: float,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> NormalizedBox:
    x0, y0, x1, y1 = clamp_normalized_box(box)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    width = max(x1 - x0 + 2.0 * pad_x, min_width)
    height = max(y1 - y0 + 2.0 * pad_y, min_height)
    return clamp_normalized_box((cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height))


def preprocess_source_image(image_path: str, max_image_size: int) -> PIL.Image.Image:
    image = PIL.Image.open(image_path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        resized_w = max(16, int(round(image.width * scale)))
        resized_h = max(16, int(round(image.height * scale)))
        image = image.resize((resized_w, resized_h), PIL.Image.Resampling.LANCZOS)
    return image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))


def estimate_dark_eye_structure_boxes(
    image_path: str,
    max_image_size: int,
) -> tuple[dict[str, NormalizedBox | None], dict[str, object]]:
    image = preprocess_source_image(image_path, max_image_size)
    rgb = np.asarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y0_lim = int(0.18 * h)
    y1_lim = int(0.72 * h)
    x0_lim = int(0.08 * w)
    x1_lim = int(0.92 * w)
    roi = gray[y0_lim:y1_lim, x0_lim:x1_lim]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    dark_threshold = min(95.0, max(25.0, float(np.percentile(blur, 18))))
    binary = (blur <= dark_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates: list[dict[str, float]] = []
    image_area = float(h * w)
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        area_ratio = float(area) / image_area
        if area_ratio < 0.00025 or area_ratio > 0.018:
            continue
        if bw <= 2 or bh <= 2:
            continue
        cx = (x0_lim + float(centroids[idx][0])) / w
        cy = (y0_lim + float(centroids[idx][1])) / h
        if cy < 0.33 or cy > 0.62:
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 0.25 or aspect > 4.5:
            continue
        candidates.append(
            {
                "x0": (x0_lim + x) / w,
                "y0": (y0_lim + y) / h,
                "x1": (x0_lim + x + bw) / w,
                "y1": (y0_lim + y + bh) / h,
                "cx": cx,
                "cy": cy,
                "area_ratio": area_ratio,
                "score": area_ratio * (1.0 - abs(cy - 0.45)),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_pair = None
    best_pair_score = -1.0
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            a, b = sorted((left, right), key=lambda item: item["cx"])
            dx = b["cx"] - a["cx"]
            dy = abs(b["cy"] - a["cy"])
            if dx < 0.10 or dx > 0.48 or dy > 0.16:
                continue
            center_y = 0.5 * (a["cy"] + b["cy"])
            pair_score = a["score"] + b["score"] - 0.4 * dy - 0.2 * abs(center_y - 0.46)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (a, b)
    if best_pair is not None:
        selected = list(best_pair)
    else:
        selected = candidates[:1]
    if not selected:
        fallback_boxes, fallback_meta = estimate_foreground_head_structure_boxes(image_path, max_image_size)
        fallback_meta.update(
            {
                "structure_dark_eye_candidates": len(candidates),
                "structure_primary_mode": "dark_eyes",
            }
        )
        return fallback_boxes, fallback_meta
    anchor = clamp_normalized_box(
        (
            min(item["x0"] for item in selected),
            min(item["y0"] for item in selected),
            max(item["x1"] for item in selected),
            max(item["y1"] for item in selected),
        )
    )
    accessory_anchor = clamp_normalized_box(
        (
            anchor[0] - 0.11,
            anchor[1] - 0.050,
            anchor[2] - 0.025,
            anchor[3] - 0.005,
        )
    )
    edit = expand_normalized_box(accessory_anchor, min_width=0.44, min_height=0.14, pad_x=0.02, pad_y=0.008)
    source_inject = expand_normalized_box(accessory_anchor, min_width=0.72, min_height=0.30, pad_x=0.12, pad_y=0.08)
    ex0, ey0, ex1, ey1 = edit
    width = ex1 - ex0
    preserve = clamp_normalized_box(
        (
            ex0 + 0.08 * width,
            min(1.0, min(ey1 + 0.005, accessory_anchor[1] + 0.12)),
            ex1 - 0.02 * width,
            min(1.0, min(ey1 + 0.005, accessory_anchor[1] + 0.12) + 0.30),
        )
    )
    meta = {
        "structure_found": True,
        "structure_mode": "dark_eyes",
        "structure_candidates": len(candidates),
        "structure_selected": len(selected),
        "structure_dark_threshold": dark_threshold,
        "structure_anchor_box": list(anchor),
        "structure_accessory_anchor_box": list(accessory_anchor),
        "structure_edit_mask_box": list(edit),
        "structure_source_inject_mask_box": list(source_inject),
        "structure_preserve_box": list(preserve),
    }
    if len(selected) >= 2:
        left, right = sorted(selected[:2], key=lambda item: item["cx"])
        angle = np.degrees(np.arctan2((right["cy"] - left["cy"]) * h, (right["cx"] - left["cx"]) * w))
        meta["structure_eye_line_angle_deg"] = float(angle)
    else:
        meta["structure_eye_line_angle_deg"] = 0.0
    return {"edit": edit, "source_inject": source_inject, "preserve": preserve, "anchor": anchor}, meta


def estimate_foreground_head_structure_boxes(
    image_path: str,
    max_image_size: int,
) -> tuple[dict[str, NormalizedBox | None], dict[str, object]]:
    image = preprocess_source_image(image_path, max_image_size)
    rgb = np.asarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)

    greenish = (hue >= 35.0) & (hue <= 95.0) & (sat >= 35.0)
    foreground = (((val < 135.0) & ~greenish) | ((sat > 45.0) & ~greenish)).astype(np.uint8)
    foreground[: int(0.05 * h), :] = 0
    foreground[int(0.92 * h) :, :] = 0
    kernel = np.ones((5, 5), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    components: list[dict[str, float]] = []
    image_area = float(h * w)
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        area_ratio = float(area) / image_area
        if area_ratio < 0.015 or area_ratio > 0.75:
            continue
        if bw < 0.08 * w or bh < 0.08 * h:
            continue
        components.append(
            {
                "x0": x / w,
                "y0": y / h,
                "x1": (x + bw) / w,
                "y1": (y + bh) / h,
                "area_ratio": area_ratio,
                "label": float(idx),
            }
        )
    components.sort(key=lambda item: item["area_ratio"], reverse=True)
    if not components:
        return (
            {"edit": None, "source_inject": None, "preserve": None, "anchor": None},
            {"structure_found": False, "structure_candidates": 0, "structure_mode": "foreground_head"},
        )

    subject = components[0]
    label_id = int(subject["label"])
    x0 = int(round(subject["x0"] * w))
    y0 = int(round(subject["y0"] * h))
    x1 = int(round(subject["x1"] * w))
    y1 = int(round(subject["y1"] * h))
    subject_mask = labels == label_id
    top_y0 = y0
    top_y1 = min(y1, y0 + int(round(0.55 * max(1, y1 - y0))))
    mid_x = x0 + int(round(0.5 * max(1, x1 - x0)))
    left_mass = int(subject_mask[top_y0:top_y1, x0:mid_x].sum())
    right_mass = int(subject_mask[top_y0:top_y1, mid_x:x1].sum())
    head_on_left = left_mass >= right_mass

    sx0, sy0, sx1, sy1 = clamp_normalized_box((subject["x0"], subject["y0"], subject["x1"], subject["y1"]))
    sw = max(1e-6, sx1 - sx0)
    sh = max(1e-6, sy1 - sy0)
    if head_on_left:
        head = clamp_normalized_box((sx0, sy0 + 0.20 * sh, sx0 + 0.46 * sw, sy0 + 0.52 * sh))
    else:
        head = clamp_normalized_box((sx1 - 0.46 * sw, sy0 + 0.20 * sh, sx1, sy0 + 0.52 * sh))

    edit = expand_normalized_box(head, min_width=0.28, min_height=0.13, pad_x=0.015, pad_y=0.01)
    source_inject = expand_normalized_box(head, min_width=0.46, min_height=0.24, pad_x=0.08, pad_y=0.06)
    ex0, ey0, ex1, ey1 = edit
    ew = max(1e-6, ex1 - ex0)
    preserve = clamp_normalized_box(
        (
            ex0 + 0.08 * ew,
            min(1.0, ey1 + 0.035),
            ex1 - 0.02 * ew,
            min(1.0, ey1 + 0.30),
        )
    )
    meta = {
        "structure_found": True,
        "structure_mode": "foreground_head",
        "structure_candidates": len(components),
        "structure_selected": 1,
        "structure_subject_box": [float(subject["x0"]), float(subject["y0"]), float(subject["x1"]), float(subject["y1"])],
        "structure_head_side": "left" if head_on_left else "right",
        "structure_head_box": list(head),
        "structure_edit_mask_box": list(edit),
        "structure_source_inject_mask_box": list(source_inject),
        "structure_preserve_box": list(preserve),
    }
    return {"edit": edit, "source_inject": source_inject, "preserve": preserve, "anchor": head}, meta


def save_structure_glasses_mask(
    image_path: str,
    max_image_size: int,
    edit_box: NormalizedBox,
    output_path: str,
    angle_deg: float = 0.0,
) -> str:
    image = preprocess_source_image(image_path, max_image_size)
    w, h = image.size
    x0, y0, x1, y1 = clamp_normalized_box(edit_box)
    width = x1 - x0
    height = y1 - y0
    mask = np.zeros((h, w), dtype=np.float32)

    def box_to_pixels(box: NormalizedBox) -> tuple[int, int, int, int]:
        bx0, by0, bx1, by1 = clamp_normalized_box(box)
        ix0 = max(0, min(w, int(round(bx0 * w))))
        ix1 = max(0, min(w, int(round(bx1 * w))))
        iy0 = max(0, min(h, int(round(by0 * h))))
        iy1 = max(0, min(h, int(round(by1 * h))))
        return ix0, iy0, ix1, iy1

    def add_ellipse(box: NormalizedBox, value: float = 1.0) -> None:
        ix0, iy0, ix1, iy1 = box_to_pixels(box)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        center = (int(round(0.5 * (ix0 + ix1 - 1))), int(round(0.5 * (iy0 + iy1 - 1))))
        axes = (max(1, int(round(0.5 * (ix1 - ix0)))), max(1, int(round(0.5 * (iy1 - iy0)))))
        layer = np.zeros_like(mask)
        cv2.ellipse(layer, center, axes, float(angle_deg), 0, 360, float(value), thickness=-1)
        np.maximum(mask, layer, out=mask)

    def add_rotated_bridge(box: NormalizedBox, value: float = 0.85) -> None:
        ix0, iy0, ix1, iy1 = box_to_pixels(box)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        rect = (
            (0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1)),
            (max(1.0, float(ix1 - ix0)), max(1.0, float(iy1 - iy0))),
            float(angle_deg),
        )
        points = cv2.boxPoints(rect).round().astype(np.int32)
        layer = np.zeros_like(mask)
        cv2.fillConvexPoly(layer, points, float(value))
        np.maximum(mask, layer, out=mask)

    split = x0 + 0.5 * width
    gap = 0.035 * width
    lens_y0 = y0 + 0.12 * height
    lens_y1 = y1 - 0.02 * height
    add_ellipse((x0 + 0.02 * width, lens_y0, split - gap, lens_y1))
    add_ellipse((split + gap, lens_y0, x1 - 0.02 * width, lens_y1))
    add_rotated_bridge((split - 0.055 * width, y0 + 0.30 * height, split + 0.055 * width, y0 + 0.52 * height), 0.85)

    blur = max(3, int(round(min(w, h) * 0.006)) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    if float(mask.max()) > 1e-6:
        mask = mask / float(mask.max())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PIL.Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype("uint8"), mode="L").save(output_path)
    return output_path


def _normalize01_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - float(values.min())) / max(float(values.max() - values.min()), 1e-6)


def _otsu_threshold(values: np.ndarray, floor: float) -> float:
    values_u8 = np.clip(values.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(threshold) / 255.0, float(floor))


def _keep_top_mask_components(
    binary: np.ndarray,
    score: np.ndarray,
    keep: int,
    min_area: int,
) -> np.ndarray:
    if keep <= 0:
        return binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    components: list[tuple[float, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        components.append((float(score[component].mean()) * np.sqrt(float(area)), label))
    components.sort(reverse=True)
    out = np.zeros_like(binary, dtype=np.uint8)
    for _, label in components[:keep]:
        out[labels == label] = 1
    return out


def build_proposal_diff_mask(
    source_image_path: str,
    proposal_image_path: str,
    max_image_size: int,
    output_path: str,
    threshold: float = 0.22,
    keep_components: int = 2,
    min_area: int = 24,
    dilate: int = 9,
    erode: int = 0,
    blur: int = 17,
    dark_bias: float = 1.0,
) -> tuple[str, dict[str, object]]:
    source = preprocess_source_image(source_image_path, max_image_size)
    proposal_paths = [path.strip() for path in proposal_image_path.split(",") if path.strip()]
    if not proposal_paths:
        raise ValueError("--proposal-edit-image must contain at least one image path")
    src = np.asarray(source, dtype=np.uint8)
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_luma = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    proposal_scores: list[np.ndarray] = []
    for proposal_path in proposal_paths:
        proposal = PIL.Image.open(proposal_path).convert("RGB").resize(source.size, PIL.Image.Resampling.LANCZOS)
        prop = np.asarray(proposal, dtype=np.uint8)
        prop_lab = cv2.cvtColor(prop, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_diff = _normalize01_array(np.linalg.norm(prop_lab - src_lab, axis=-1))
        prop_luma = cv2.cvtColor(prop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        darker = np.clip(src_luma - prop_luma, 0.0, 1.0)
        proposal_scores.append(_normalize01_array(lab_diff + float(dark_bias) * darker))
    if len(proposal_scores) == 1:
        score = proposal_scores[0]
    else:
        score = np.minimum.reduce(proposal_scores)

    auto_threshold = _otsu_threshold(score.reshape(-1), threshold)
    binary = (score >= auto_threshold).astype(np.uint8)
    if erode > 0:
        binary = cv2.erode(binary, np.ones((erode, erode), dtype=np.uint8), iterations=1)
    binary = _keep_top_mask_components(binary, score, keep=keep_components, min_area=min_area)
    if dilate > 0:
        binary = cv2.dilate(binary, np.ones((dilate, dilate), dtype=np.uint8), iterations=1)

    soft = binary.astype(np.float32)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        soft = cv2.GaussianBlur(soft, (blur, blur), 0)
        soft = np.clip(soft, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PIL.Image.fromarray((soft * 255.0).round().astype("uint8"), mode="L").save(output_path)
    meta = {
        "proposal_diff_found": bool(float(soft.max()) > 1e-6),
        "proposal_diff_mask": output_path,
        "proposal_edit_image": proposal_image_path,
        "proposal_edit_images": proposal_paths,
        "proposal_diff_consensus_count": len(proposal_paths),
        "proposal_mask_auto_threshold": float(auto_threshold),
        "proposal_mask_mean": float(soft.mean()),
        "proposal_mask_max": float(soft.max()),
        "proposal_mask_keep_components": int(keep_components),
        "proposal_mask_min_area": int(min_area),
        "proposal_mask_dilate": int(dilate),
        "proposal_mask_erode": int(erode),
        "proposal_mask_blur": int(blur),
        "proposal_mask_dark_bias": float(dark_bias),
    }
    return output_path, meta
