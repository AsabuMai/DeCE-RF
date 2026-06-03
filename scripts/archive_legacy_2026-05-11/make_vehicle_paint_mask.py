from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


COLOR_RGB = {
    "black": (0.02, 0.02, 0.02),
    "blue": (0.05, 0.22, 0.9),
    "brown": (0.42, 0.22, 0.08),
    "gray": (0.5, 0.5, 0.5),
    "green": (0.05, 0.55, 0.18),
    "grey": (0.5, 0.5, 0.5),
    "orange": (0.95, 0.42, 0.05),
    "red": (0.9, 0.05, 0.04),
    "silver": (0.75, 0.75, 0.72),
    "white": (0.95, 0.95, 0.92),
    "yellow": (0.95, 0.82, 0.05),
}


DEFAULT_EXCLUDE_PROMPTS = (
    "wheel",
    "tire",
    "windshield",
    "window",
    "headlight",
    "license plate",
    "mirror",
    "grille",
)


def preprocess_image(path: Path, max_image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        image = image.resize(
            (max(16, int(round(image.width * scale))), max(16, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    return image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(path)


def mask_area(mask: np.ndarray, threshold: float = 0.5) -> float:
    return float((mask > threshold).mean())


def keep_components(mask: np.ndarray, keep: int, min_area: int) -> np.ndarray:
    if keep <= 0:
        return mask
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    out = np.zeros_like(mask, dtype=np.float32)
    for _, label in components[:keep]:
        out[labels == label] = mask[labels == label]
    return out


def color_confidence_mask(
    image: Image.Image,
    source_color: str,
    *,
    hue_threshold: float,
    lab_threshold: float,
    min_saturation: float,
    min_value: float,
    max_value: float,
    mask_threshold: float,
) -> np.ndarray:
    key = source_color.lower().strip()
    if key not in COLOR_RGB:
        raise ValueError(f"Unsupported --source-color {source_color!r}; known={sorted(COLOR_RGB)}")
    rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] / 180.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    target_rgb = np.array(COLOR_RGB[key], dtype=np.float32).reshape(1, 1, 3)
    target_u8 = (target_rgb * 255.0).round().astype(np.uint8)
    target_hsv = cv2.cvtColor(target_u8, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
    target_hue = float(target_hsv[0] / 180.0)
    hue_dist = np.minimum(np.abs(hue - target_hue), 1.0 - np.abs(hue - target_hue))
    hue_weight = 1.0 / (1.0 + np.exp((hue_dist - hue_threshold) / 0.025))

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    target_lab = cv2.cvtColor(target_u8, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0] / 255.0
    lab_dist = np.linalg.norm(lab - target_lab.reshape(1, 1, 3), axis=2)
    lab_weight = 1.0 / (1.0 + np.exp((lab_dist - lab_threshold) / 0.055))

    chroma_gate = (sat >= min_saturation) & (val >= min_value) & (val <= max_value)
    mask = np.minimum(hue_weight, lab_weight).astype(np.float32) * chroma_gate.astype(np.float32)
    if mask_threshold > 0.0:
        mask = np.where(mask >= mask_threshold, mask, 0.0).astype(np.float32)
    return mask.clip(0.0, 1.0)


def make_overlay(image: Image.Image, paint: np.ndarray, exclude: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    paint_color = np.array([0.0, 155.0, 255.0], dtype=np.float32)
    exclude_color = np.array([255.0, 65.0, 45.0], dtype=np.float32)
    paint_alpha = np.clip(paint[..., None], 0.0, 1.0) * 0.52
    out = rgb * (1.0 - paint_alpha) + paint_color * paint_alpha
    exclude_alpha = (exclude[..., None] > 0.2).astype(np.float32) * 0.42
    out = out * (1.0 - exclude_alpha) + exclude_color * exclude_alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(path)


def vehicle_detail_exclude_mask(
    image: Image.Image,
    vehicle_mask: np.ndarray,
    *,
    edge_low: int = 55,
    edge_high: int = 145,
    edge_dilate: int = 3,
    low_saturation: float = 0.32,
    bright_value: float = 0.70,
    dark_value: float = 0.26,
) -> np.ndarray:
    rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    edges = cv2.Canny(gray, int(edge_low), int(edge_high)).astype(np.float32) / 255.0
    if edge_dilate > 1:
        kernel_size = edge_dilate + 1 if edge_dilate % 2 == 0 else edge_dilate
        edges = cv2.dilate(edges, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)

    achromatic_highlight = (sat < low_saturation) & (val > bright_value)
    dark_detail = val < dark_value
    structured_nonpaint = (edges > 0.0) & ((sat < low_saturation) | (val > bright_value) | dark_detail)
    detail = (achromatic_highlight | dark_detail | structured_nonpaint).astype(np.float32)
    return (detail * vehicle_mask).clip(0.0, 1.0)


class GroundedSam:
    def __init__(
        self,
        grounding_model: str,
        sam_model: str,
        device: torch.device,
        local_files_only: bool,
    ) -> None:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor, SamModel, SamProcessor

        self.gd_processor = GroundingDinoProcessor.from_pretrained(
            grounding_model,
            local_files_only=local_files_only,
        )
        self.gd_model = GroundingDinoForObjectDetection.from_pretrained(
            grounding_model,
            local_files_only=local_files_only,
        ).to(device)
        self.sam_processor = SamProcessor.from_pretrained(
            sam_model,
            local_files_only=local_files_only,
        )
        self.sam_model = SamModel.from_pretrained(
            sam_model,
            local_files_only=local_files_only,
        ).to(device)
        self.gd_model.eval()
        self.sam_model.eval()
        self.grounding_model = grounding_model
        self.sam_model_name = sam_model
        self.device = device

    def segment(
        self,
        image: Image.Image,
        phrase: str,
        *,
        box_threshold: float,
        text_threshold: float,
        max_boxes: int,
        max_box_area_ratio: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        width, height = image.size
        prompt = phrase if phrase.endswith(".") else f"{phrase}."
        inputs = self.gd_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None or len(boxes) == 0:
            return np.zeros((height, width), dtype=np.float32), {
                "phrase": phrase,
                "accepted": False,
                "reject_reason": "no_boxes",
                "num_boxes": 0,
            }

        box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
        box_area_ratio = (box_wh[:, 0] * box_wh[:, 1]) / float(max(1, width * height))
        if max_box_area_ratio > 0.0:
            keep = box_area_ratio <= max_box_area_ratio
            if not bool(keep.any().item()):
                return np.zeros((height, width), dtype=np.float32), {
                    "phrase": phrase,
                    "accepted": False,
                    "reject_reason": "all_boxes_too_large",
                    "num_boxes": 0,
                    "filtered_box_area_ratios": [float(v) for v in box_area_ratio.detach().cpu().tolist()],
                }
            boxes = boxes[keep]
            box_area_ratio = box_area_ratio[keep]
            if scores is not None:
                scores = scores[keep]
            if labels is not None:
                labels = [label for label, flag in zip(labels, keep.detach().cpu().tolist()) if flag]
        if scores is not None and max_boxes > 0 and len(boxes) > max_boxes:
            order = torch.argsort(scores.detach().cpu(), descending=True)[:max_boxes].to(boxes.device)
            boxes = boxes[order]
            box_area_ratio = box_area_ratio[order]
            scores = scores[order]
            if labels is not None:
                labels = [labels[int(idx)] for idx in order.detach().cpu().tolist()]

        if len(boxes) == 0:
            return np.zeros((height, width), dtype=np.float32), {
                "phrase": phrase,
                "accepted": False,
                "reject_reason": "all_boxes_filtered",
                "num_boxes": 0,
            }

        box_mask = np.zeros((height, width), dtype=np.float32)
        for box in boxes.detach().cpu().tolist():
            x0, y0, x1, y1 = box
            x0 = max(0, min(width, int(np.floor(x0))))
            x1 = max(0, min(width, int(np.ceil(x1))))
            y0 = max(0, min(height, int(np.floor(y0))))
            y1 = max(0, min(height, int(np.ceil(y1))))
            if x1 > x0 and y1 > y0:
                box_mask[y0:y1, x0:x1] = 1.0

        sam_inputs = self.sam_processor(image, input_boxes=[boxes.detach().cpu().tolist()], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu(),
        )[0].float()
        if masks.ndim == 4:
            masks = masks.max(dim=1).values
        union = masks.max(dim=0).values.clamp(0.0, 1.0).numpy() * box_mask
        return union.astype(np.float32), {
            "phrase": phrase,
            "accepted": True,
            "num_boxes": int(len(boxes)),
            "boxes_xyxy": [[float(v) for v in box] for box in boxes.detach().cpu().tolist()],
            "box_area_ratios": [float(v) for v in box_area_ratio.detach().cpu().tolist()],
            "scores": [] if scores is None else [float(v) for v in scores.detach().cpu().tolist()],
            "labels": [] if labels is None else [str(v) for v in labels],
            "mask_area_gt_0_5": mask_area(union),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable vehicle paint-panel mask with SAM part exclusions.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vehicle-phrase", default="car")
    parser.add_argument("--exclude-prompts", default=",".join(DEFAULT_EXCLUDE_PROMPTS))
    parser.add_argument("--source-color", default=None)
    parser.add_argument("--color-mode", choices=("none", "soft", "hard"), default="soft")
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--allow-download", action="store_true", default=False)
    parser.add_argument("--vehicle-box-threshold", type=float, default=0.25)
    parser.add_argument("--vehicle-text-threshold", type=float, default=0.20)
    parser.add_argument("--part-box-threshold", type=float, default=0.18)
    parser.add_argument("--part-text-threshold", type=float, default=0.16)
    parser.add_argument("--vehicle-max-box-area-ratio", type=float, default=0.65)
    parser.add_argument("--part-max-box-area-ratio", type=float, default=0.08)
    parser.add_argument("--part-max-vehicle-area-ratio", type=float, default=0.45)
    parser.add_argument("--part-min-vehicle-overlap", type=float, default=0.45)
    parser.add_argument("--part-max-boxes", type=int, default=4)
    parser.add_argument("--exclude-dilate", type=int, default=5)
    parser.add_argument("--detail-exclude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detail-edge-dilate", type=int, default=3)
    parser.add_argument("--detail-low-saturation", type=float, default=0.32)
    parser.add_argument("--detail-bright-value", type=float, default=0.70)
    parser.add_argument("--detail-dark-value", type=float, default=0.26)
    parser.add_argument("--paint-erode", type=int, default=0)
    parser.add_argument("--keep-components", type=int, default=8)
    parser.add_argument("--min-component-area", type=int, default=40)
    parser.add_argument("--hue-threshold", type=float, default=0.12)
    parser.add_argument("--lab-threshold", type=float, default=0.60)
    parser.add_argument("--min-saturation", type=float, default=0.20)
    parser.add_argument("--min-value", type=float, default=0.18)
    parser.add_argument("--max-value", type=float, default=0.98)
    parser.add_argument("--color-mask-threshold", type=float, default=0.36)
    args = parser.parse_args()

    image = preprocess_image(args.image, args.max_image_size)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    segmenter = GroundedSam(
        args.grounding_model,
        args.sam_model,
        device=device,
        local_files_only=not args.allow_download,
    )
    print(f"[vehicle-paint] models loaded on {device}", flush=True)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vehicle-paint] segment vehicle: {args.vehicle_phrase}", flush=True)
    vehicle_mask, vehicle_meta = segmenter.segment(
        image,
        args.vehicle_phrase,
        box_threshold=args.vehicle_box_threshold,
        text_threshold=args.vehicle_text_threshold,
        max_boxes=1,
        max_box_area_ratio=args.vehicle_max_box_area_ratio,
    )
    if mask_area(vehicle_mask) <= 0.0:
        raise RuntimeError(f"Could not segment vehicle phrase {args.vehicle_phrase!r}: {vehicle_meta}")
    save_mask(vehicle_mask, out_dir / "vehicle_mask.png")

    exclude_union = np.zeros_like(vehicle_mask, dtype=np.float32)
    part_meta = []
    vehicle_area_pixels = float((vehicle_mask > 0.5).sum())
    prompts = [item.strip() for item in args.exclude_prompts.split(",") if item.strip()]
    for prompt in prompts:
        print(f"[vehicle-paint] segment part: {prompt}", flush=True)
        part_mask, meta = segmenter.segment(
            image,
            prompt,
            box_threshold=args.part_box_threshold,
            text_threshold=args.part_text_threshold,
            max_boxes=args.part_max_boxes,
            max_box_area_ratio=args.part_max_box_area_ratio,
        )
        part_inside = (part_mask * vehicle_mask).clip(0.0, 1.0)
        part_pixels = float((part_mask > 0.5).sum())
        inside_pixels = float((part_inside > 0.5).sum())
        overlap = 0.0 if part_pixels <= 0 else inside_pixels / part_pixels
        vehicle_ratio = 0.0 if vehicle_area_pixels <= 0 else inside_pixels / vehicle_area_pixels
        reject_reason = None
        if part_pixels <= 0:
            reject_reason = "empty_mask"
        elif overlap < args.part_min_vehicle_overlap:
            reject_reason = "low_vehicle_overlap"
        elif vehicle_ratio > args.part_max_vehicle_area_ratio:
            reject_reason = "too_large_vs_vehicle"
        accepted = reject_reason is None
        if accepted:
            exclude_union = np.maximum(exclude_union, part_inside)
            save_mask(part_inside, out_dir / f"exclude_{prompt.replace(' ', '_')}.png")
        print(
            "[vehicle-paint] part "
            f"{prompt}: accepted={accepted} "
            f"overlap={overlap:.3f} vehicle_ratio={vehicle_ratio:.3f} "
            f"reason={reject_reason}",
            flush=True,
        )
        meta.update(
            {
                "accepted": bool(accepted),
                "reject_reason": reject_reason,
                "vehicle_overlap_ratio": float(overlap),
                "part_area_over_vehicle_area": float(vehicle_ratio),
            }
        )
        part_meta.append(meta)

    if args.exclude_dilate > 1:
        kernel_size = args.exclude_dilate + 1 if args.exclude_dilate % 2 == 0 else args.exclude_dilate
        exclude_union = cv2.dilate(exclude_union, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    exclude_union = (exclude_union * vehicle_mask).clip(0.0, 1.0)
    detail_exclude = np.zeros_like(exclude_union, dtype=np.float32)
    if args.detail_exclude:
        detail_exclude = vehicle_detail_exclude_mask(
            image,
            vehicle_mask,
            edge_dilate=args.detail_edge_dilate,
            low_saturation=args.detail_low_saturation,
            bright_value=args.detail_bright_value,
            dark_value=args.detail_dark_value,
        )
        save_mask(detail_exclude, out_dir / "detail_exclude_mask.png")
        exclude_union = np.maximum(exclude_union, detail_exclude).clip(0.0, 1.0)
    save_mask(exclude_union, out_dir / "exclude_mask.png")

    paint_mask = (vehicle_mask * (1.0 - exclude_union)).clip(0.0, 1.0)
    color_meta = None
    if args.source_color and args.color_mode != "none":
        color_mask = color_confidence_mask(
            image,
            args.source_color,
            hue_threshold=args.hue_threshold,
            lab_threshold=args.lab_threshold,
            min_saturation=args.min_saturation,
            min_value=args.min_value,
            max_value=args.max_value,
            mask_threshold=args.color_mask_threshold,
        )
        color_mask = (color_mask * vehicle_mask).clip(0.0, 1.0)
        save_mask(color_mask, out_dir / "source_color_mask.png")
        if args.color_mode == "hard":
            paint_mask = (paint_mask * (color_mask > 0.0).astype(np.float32)).clip(0.0, 1.0)
        else:
            paint_mask = (paint_mask * color_mask).clip(0.0, 1.0)
        color_meta = {
            "source_color": args.source_color,
            "color_mode": args.color_mode,
            "source_color_area_gt_0_5": mask_area(color_mask),
            "hue_threshold": float(args.hue_threshold),
            "lab_threshold": float(args.lab_threshold),
            "color_mask_threshold": float(args.color_mask_threshold),
        }

    if args.paint_erode > 1:
        kernel_size = args.paint_erode + 1 if args.paint_erode % 2 == 0 else args.paint_erode
        paint_mask = cv2.erode(paint_mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    paint_mask = keep_components(paint_mask, args.keep_components, args.min_component_area)
    save_mask(paint_mask, out_dir / "paint_mask.png")
    make_overlay(image, paint_mask, exclude_union, out_dir / "overlay.png")

    metadata = {
        "image": str(args.image),
        "vehicle_phrase": args.vehicle_phrase,
        "exclude_prompts": prompts,
        "grounding_model": args.grounding_model,
        "sam_model": args.sam_model,
        "device": str(device),
        "vehicle": vehicle_meta,
        "parts": part_meta,
        "color": color_meta,
        "vehicle_area_gt_0_5": mask_area(vehicle_mask),
        "detail_exclude_area_gt_0_5": mask_area(detail_exclude),
        "exclude_area_gt_0_5": mask_area(exclude_union),
        "paint_area_gt_0_5": mask_area(paint_mask),
        "outputs": {
            "vehicle_mask": str(out_dir / "vehicle_mask.png"),
            "exclude_mask": str(out_dir / "exclude_mask.png"),
            "detail_exclude_mask": str(out_dir / "detail_exclude_mask.png") if args.detail_exclude else None,
            "paint_mask": str(out_dir / "paint_mask.png"),
            "overlay": str(out_dir / "overlay.png"),
        },
        "quality_gates": {
            "part_max_vehicle_area_ratio": float(args.part_max_vehicle_area_ratio),
            "part_min_vehicle_overlap": float(args.part_min_vehicle_overlap),
            "part_max_box_area_ratio": float(args.part_max_box_area_ratio),
            "detail_exclude": bool(args.detail_exclude),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
