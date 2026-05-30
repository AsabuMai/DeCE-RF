from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)

DEFAULT_TASKS = (
    "cat_crown",
    "dog_sunglasses",
    "mug_heart",
    "backpack_remove_toy_charm",
)
DEFAULT_DIFF_METHODS = (
    "direct_target",
    "adaptive_full_generic_support",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
)
SUPPORT_FALLBACKS = (
    "support_v3_controller_rmsgap/seed_{seed}/masks/operation_v3_edit_mask.png",
    "support_v3_controller_rmsgap/seed_{seed}/masks/operation_v3_core_mask.png",
    "support_v3_controller_rmsgap/seed_{seed}/masks/selected_candidate_postprocessed.png",
    "support_v3_fixed/seed_{seed}/masks/operation_v3_edit_mask.png",
    "support_v3_fixed/seed_{seed}/masks/operation_v3_core_mask.png",
    "support_v3_fixed/seed_{seed}/masks/selected_candidate_postprocessed.png",
)
TASK_KEEP_COMPONENTS = {
    "cat_crown": 1,
    "dog_sunglasses": 2,
    "mug_heart": 1,
    "backpack_remove_toy_charm": 1,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, RESAMPLE_BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_gray(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, RESAMPLE_BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    image.save(path)


def smooth(mask: np.ndarray, blur_radius: float) -> np.ndarray:
    if blur_radius <= 0:
        return mask.astype(np.float32)
    image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    size = radius * 2 + 1
    image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
    image = image.filter(ImageFilter.MaxFilter(size=size))
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def keep_components(mask: np.ndarray, keep: int, min_pixels: int) -> np.ndarray:
    binary = mask > 0.5
    height, width = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or seen[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if binary[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            if len(pixels) >= min_pixels:
                components.append((len(pixels), pixels))
    components.sort(key=lambda item: item[0], reverse=True)
    out = np.zeros_like(mask, dtype=np.float32)
    for _, pixels in components[:keep]:
        for y, x in pixels:
            out[y, x] = 1.0
    return out


def threshold_to_area(score: np.ndarray, min_area: float, max_area: float) -> tuple[np.ndarray, float]:
    positive = score[score > 1e-8]
    if positive.size == 0:
        return np.zeros_like(score, dtype=np.float32), 0.0
    height, width = score.shape
    total = float(height * width)
    target_area = min(max(float(np.mean(score > np.percentile(positive, 92.0))), min_area), max_area)
    target_pixels = max(1, int(round(target_area * total)))
    flat = score.reshape(-1)
    order = np.argsort(flat)[::-1]
    out = np.zeros(flat.shape, dtype=np.float32)
    out[order[:target_pixels]] = 1.0
    return out.reshape(score.shape), float(out.mean())


def build_diff_mask(task: str, task_dir: Path, seed: str, methods: tuple[str, ...]) -> tuple[np.ndarray | None, dict[str, Any]]:
    base_dir = task_dir / "base_only" / f"seed_{seed}"
    if not (base_dir / "metadata.json").exists() or not (base_dir / "result.png").exists():
        return None, {"reason": "missing_base"}
    metadata = load_json(base_dir / "metadata.json")
    source_path = Path(metadata.get("image", ""))
    if not source_path.exists():
        return None, {"reason": "missing_source", "source_image": str(source_path)}
    base_result = Image.open(base_dir / "result.png").convert("RGB")
    size = base_result.size
    source = load_rgb(source_path, size=size)
    base = np.asarray(base_result, dtype=np.float32) / 255.0
    base_drift = np.abs(base - source).mean(axis=2)

    maps = []
    used_methods = []
    for method in methods:
        result_path = task_dir / method / f"seed_{seed}" / "result.png"
        if not result_path.exists():
            continue
        result = load_rgb(result_path, size=size)
        edit_diff = np.abs(result - source).mean(axis=2)
        maps.append(np.maximum(edit_diff - base_drift, 0.0))
        used_methods.append(method)
    if not maps:
        return None, {"reason": "missing_edit_results"}
    score = np.percentile(np.stack(maps, axis=0), 75.0, axis=0)
    score = smooth(score / max(float(score.max()), 1e-8), 1.5)
    binary, raw_area = threshold_to_area(score, min_area=0.008, max_area=0.12)
    keep = TASK_KEEP_COMPONENTS.get(task, 1)
    binary = keep_components(binary, keep=keep, min_pixels=max(16, int(binary.size * 0.0004)))
    binary = dilate(binary, radius=3)
    mask = smooth(binary, 1.2)
    area = float((mask > 0.2).mean())
    if area < 0.004 or area > 0.18:
        return None, {"reason": "diff_area_out_of_range", "area": area, "raw_area": raw_area}
    return mask, {"reason": "diff_minus_base", "area": area, "methods": used_methods}


def build_support_mask(task_dir: Path, seed: str, size: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    for template in SUPPORT_FALLBACKS:
        path = task_dir / template.format(seed=seed)
        mask = load_gray(path, size=size)
        if mask is None:
            continue
        mask = smooth(dilate(mask > 0.2, radius=2), 1.2)
        return mask, {"reason": "support_fallback", "source": str(path)}
    return None, {"reason": "missing_support_fallback"}


def build_task_mask(
    outputs_dir: Path,
    manual_dir: Path,
    task: str,
    seed: str,
    methods: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    task_dir = outputs_dir / task
    base_dir = task_dir / "base_only" / f"seed_{seed}"
    metadata = load_json(base_dir / "metadata.json")
    source_path = Path(metadata.get("image", ""))
    size = Image.open(base_dir / "result.png").size

    manual_path = manual_dir / f"{task}_eval_mask.png"
    manual = load_gray(manual_path, size=size)
    if manual is not None:
        return manual, {"source_type": "manual", "source": str(manual_path), "source_image": str(source_path)}

    diff_mask, diff_meta = build_diff_mask(task, task_dir, seed, methods)
    if diff_mask is not None:
        diff_meta.update({"source_type": "diff_minus_base", "source_image": str(source_path)})
        return diff_mask, diff_meta

    support_mask, support_meta = build_support_mask(task_dir, seed, size=size)
    if support_mask is not None:
        support_meta.update({"source_type": "support_fallback", "diff_failure": diff_meta, "source_image": str(source_path)})
        return support_mask, support_meta

    raise FileNotFoundError(f"Could not build eval mask for task={task}: {diff_meta}; {support_meta}")


def parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.replace(",", " ").split() if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one fixed evaluation mask per task.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--eval-mask-dir", type=Path, default=Path("experiments/support_v3_2026-05-11/eval_masks"))
    parser.add_argument("--manual-dir", type=Path, default=None)
    parser.add_argument("--tasks", default=" ".join(DEFAULT_TASKS))
    parser.add_argument("--diff-methods", default=" ".join(DEFAULT_DIFF_METHODS))
    parser.add_argument("--seed", default="10")
    args = parser.parse_args()

    tasks = parse_list(args.tasks)
    methods = parse_list(args.diff_methods)
    manual_dir = args.manual_dir or (args.eval_mask_dir / "manual")
    args.eval_mask_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {}
    for task in tasks:
        mask, meta = build_task_mask(args.outputs_dir, manual_dir, task, args.seed, methods)
        out_path = args.eval_mask_dir / f"{task}_eval_mask.png"
        save_mask(mask, out_path)
        meta.update(
            {
                "task": task,
                "seed": args.seed,
                "eval_mask": str(out_path),
                "fixed_eval_mask_area": float((mask > 0.2).mean()),
            }
        )
        metadata[task] = meta
        print(f"{task}: area={meta['fixed_eval_mask_area']:.4f} source={meta['source_type']} -> {out_path}")

    meta_path = args.eval_mask_dir / "eval_masks_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
