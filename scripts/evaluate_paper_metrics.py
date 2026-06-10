from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
REQUIRED_FILES = ("result.png", "stats.json", "metadata.json", "command.txt")
MAIN_TASKS = (
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
)
MAIN_METHODS = (
    "base_only",
    "direct_target",
    "adaptive_full_generic_support",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
)
MAIN_SEEDS = ("10", "11", "12")
METHOD_DISPLAY_NAMES = {
    "base_only": "RF reconstruction / base reconstruction",
    "direct_target": "Direct target guidance",
    "adaptive_full_generic_support": "Generic support control",
    "support_v3_fixed": "Fixed DeCE displacement",
    "support_v3_controller_rmsgap": "DeCE-RF",
}
MASK_CANDIDATES = (
    "masks/operation_v3_edit_mask.png",
    "masks/operation_v3_core_mask.png",
    "masks/M_core.png",
    "masks/selected_candidate_postprocessed.png",
    "masks/semantic_base_generated.png",
    "masks/semantic_base.png",
    "masks/final_used_mask.png",
    "masks/combined.png",
    "masks/subject_final.png",
    "masks/subject.png",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, RESAMPLE_BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, RESAMPLE_BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def scalar(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(scalar(row, key) for row in rows) / len(rows) if rows else 0.0


def max_value(rows: list[dict[str, Any]], key: str) -> float:
    return max((scalar(row, key) for row in rows), default=0.0)


def simple_ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    wa = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    ya = (a * wa).sum(axis=2)
    yb = (b * wa).sum(axis=2)
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(ya, yb, data_range=1.0))
    except Exception:
        pass

    # Dependency-light fallback used only when skimage is unavailable.
    mu_a = float(ya.mean())
    mu_b = float(yb.mean())
    var_a = float(((ya - mu_a) ** 2).mean())
    var_b = float(((yb - mu_b) ** 2).mean())
    cov = float(((ya - mu_a) * (yb - mu_b)).mean())
    c1 = 0.01**2
    c2 = 0.03**2
    denom = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denom <= 0:
        return 1.0
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denom


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def blue_ratio(image: np.ndarray, mask: np.ndarray | None) -> float:
    region = np.ones(image.shape[:2], dtype=bool) if mask is None else mask > 0.2
    if not np.any(region):
        return 0.0
    rgb = image[region]
    return float(((rgb[:, 2] > rgb[:, 0] + 0.05) & (rgb[:, 2] > rgb[:, 1] + 0.02)).mean())


def mask_bbox(mask: np.ndarray | None, size: tuple[int, int], pad_fraction: float = 0.35) -> tuple[int, int, int, int]:
    width, height = size
    if mask is None or not np.any(mask > 0.2):
        return 0, 0, width, height
    ys, xs = np.where(mask > 0.2)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * pad_fraction))
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


class ClipScorer:
    def __init__(self, model_name: str, device: str, allow_download: bool):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=not allow_download)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=not allow_download).to(self.device)
        self.model.eval()

    def score(self, image_path: Path, texts: list[str]) -> list[float]:
        return self.score_image(Image.open(image_path).convert("RGB"), texts)

    def score_image(self, image: Image.Image, texts: list[str]) -> list[float]:
        if not texts:
            return []
        inputs = self.processor(text=texts, images=image.convert("RGB"), return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scores = image_features @ text_features.T
        return [float(value) for value in scores.squeeze(0).detach().cpu().tolist()]

    def image_embedding(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return image_features

    def image_similarity(self, image_a: Path, image_b: Path) -> float:
        emb_a = self.image_embedding(image_a)
        emb_b = self.image_embedding(image_b)
        return float((emb_a * emb_b).sum(dim=-1).detach().cpu().item())

    def direction_similarity(
        self,
        source_image: Path,
        result_image: Path,
        source_prompt: str,
        target_prompt: str,
    ) -> float:
        source_inputs = self.processor(images=Image.open(source_image).convert("RGB"), return_tensors="pt").to(self.device)
        result_inputs = self.processor(images=Image.open(result_image).convert("RGB"), return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=[source_prompt, target_prompt], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            source_img = self.model.get_image_features(pixel_values=source_inputs["pixel_values"])
            result_img = self.model.get_image_features(pixel_values=result_inputs["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            image_dir = result_img - source_img
            text_dir = text_features[1:2] - text_features[0:1]
            image_dir = image_dir / image_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            text_dir = text_dir / text_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return float((image_dir * text_dir).sum(dim=-1).detach().cpu().item())


class LpipsScorer:
    def __init__(self, device: str):
        import lpips
        import torch

        self.torch = torch
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = lpips.LPIPS(net="alex").to(self.device)
        self.model.eval()

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        with self.torch.no_grad():
            ta = self.torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            tb = self.torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            ta = ta.float() * 2.0 - 1.0
            tb = tb.float() * 2.0 - 1.0
            return float(self.model(ta, tb).detach().cpu().item())


class DinoScorer:
    def __init__(self, model_name: str, device: str, allow_download: bool):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=not allow_download)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=not allow_download).to(self.device)
        self.model.eval()

    def embedding(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return features

    def similarity(self, image_a: Path, image_b: Path) -> float:
        emb_a = self.embedding(image_a)
        emb_b = self.embedding(image_b)
        return float((emb_a * emb_b).sum(dim=-1).detach().cpu().item())


def find_run_dirs(
    outputs_dir: Path,
    task_names: set[str] | None = None,
    method_names: set[str] | None = None,
    seeds: set[str] | None = None,
) -> list[Path]:
    run_dirs: list[Path] = []
    for path in outputs_dir.rglob("*"):
        if not path.is_dir() or "archive" in path.parts:
            continue
        rel_parts = path.relative_to(outputs_dir).parts
        if len(rel_parts) != 3 or not rel_parts[-1].startswith("seed_"):
            continue
        if task_names is not None and rel_parts[0] not in task_names:
            continue
        if method_names is not None and rel_parts[1] not in method_names:
            continue
        seed = rel_parts[2].removeprefix("seed_")
        if seeds is not None and seed not in seeds:
            continue
        if any((path / name).exists() for name in REQUIRED_FILES):
            run_dirs.append(path)
    return sorted(run_dirs)


def split_matrix_path(run_dir: Path, outputs_dir: Path) -> tuple[str, str, str]:
    parts = run_dir.relative_to(outputs_dir).parts
    if len(parts) >= 3 and parts[-1].startswith("seed_"):
        return parts[-3], parts[-2], parts[-1].removeprefix("seed_")
    return "", run_dir.name, ""


def evaluate_run(
    run_dir: Path,
    outputs_dir: Path,
    clip_scorer: ClipScorer | None = None,
    dino_scorer: DinoScorer | None = None,
    lpips_scorer: LpipsScorer | None = None,
    failure_annotations: dict[str, Any] | None = None,
    mask_candidates: tuple[str, ...] = MASK_CANDIDATES,
    eval_mask_dir: Path | None = None,
) -> dict[str, Any]:
    present = {name: (run_dir / name).exists() for name in REQUIRED_FILES}
    missing = [name for name, exists in present.items() if not exists]
    task, method, seed = split_matrix_path(run_dir, outputs_dir)
    run_key = run_dir.relative_to(outputs_dir).as_posix()
    record: dict[str, Any] = {
        "run": run_key,
        "task": task,
        "method": method,
        "seed": seed,
        "complete": not missing,
        "missing": ";".join(missing),
        "method_display": METHOD_DISPLAY_NAMES.get(method, method),
    }
    failure_annotation = {}
    if failure_annotations is not None:
        failure_annotation = failure_annotations.get(record["run"], {})
        if not failure_annotation:
            compact_key = "/".join(part for part in (task, method, seed) if part)
            failure_annotation = failure_annotations.get(compact_key, {})
    if missing:
        return record

    metadata = load_json(run_dir / "metadata.json")
    stats = load_json(run_dir / "stats.json")
    rows = stats if isinstance(stats, list) else stats.get("steps", [])
    final = rows[-1] if rows else {}

    source_path = Path(metadata.get("image", ""))
    result_path = run_dir / "result.png"
    record.update(
        {
            "source_prompt": metadata.get("source_prompt") or metadata.get("effective_source_prompt"),
            "target_prompt": metadata.get("target_prompt") or metadata.get("effective_target_prompt"),
            "source_image": str(source_path),
            "result_image": str(result_path),
            "avg_rec_energy": avg(rows, "rec_energy"),
            "max_rec_energy": max_value(rows, "rec_energy"),
            "avg_rec_norm": avg(rows, "rec_guidance_norm"),
            "max_rec_norm": max_value(rows, "rec_guidance_norm"),
            "avg_edit_norm": avg(rows, "edit_guidance_norm"),
            "max_edit_norm": max_value(rows, "edit_guidance_norm"),
            "avg_cos_rec_base": avg(rows, "cos_rec_base"),
            "avg_cos_rec_edit_total": avg(rows, "cos_rec_edit_total"),
            "avg_cos_base_anchor": avg(rows, "cos_base_anchor"),
            "avg_cos_base_region": avg(rows, "cos_base_region"),
            "avg_cos_anchor_region": avg(rows, "cos_anchor_region"),
            "avg_mask_area": avg(rows, "mask_area_ratio"),
            "final_mask_area": scalar(final, "mask_area_ratio"),
            "avg_beta_t": avg(rows, "beta_t"),
            "max_beta_t": max_value(rows, "beta_t"),
            "runtime_seconds": metadata.get("runtime_seconds"),
            "peak_gpu_memory_gb": metadata.get("peak_gpu_memory_gb"),
            "failure_flag": failure_annotation.get("failure_flag", ""),
            "failure_note": failure_annotation.get("failure_note", ""),
        }
    )

    if not source_path.is_file():
        record["missing"] = "source_image"
        record["complete"] = False
        return record

    result_image = Image.open(result_path).convert("RGB")
    size = result_image.size
    result = np.asarray(result_image, dtype=np.float32) / 255.0
    source = load_rgb(source_path, size=size)
    diff = np.abs(result - source)
    sq_diff = (result - source) ** 2

    mask = None
    mask_path = ""
    mask_source = ""
    if eval_mask_dir is not None and task:
        fixed_mask_path = eval_mask_dir / f"{task}_eval_mask.png"
        mask = load_mask(fixed_mask_path, size=size)
        if mask is not None:
            mask_path = str(fixed_mask_path)
            mask_source = "fixed_eval_mask"
    if mask is None:
        for candidate in mask_candidates:
            candidate_path = run_dir / candidate
            mask = load_mask(candidate_path, size=size)
            if mask is not None:
                mask_path = str(candidate_path)
                mask_source = "run_mask"
                break
    outside = np.ones(size[::-1], dtype=bool) if mask is None else mask <= 0.2
    inside = None if mask is None else mask > 0.2

    record.update(
        {
            "mask_path": mask_path,
            "mask_source": mask_source,
            "eval_mask_area": float((mask > 0.2).mean()) if mask is not None else 1.0,
            "fixed_eval_mask_area": float((mask > 0.2).mean())
            if mask_source == "fixed_eval_mask" and mask is not None
            else "",
            "source_l1": float(diff.mean()),
            "source_rmse": float(math.sqrt(float((diff**2).mean()))),
            "source_psnr": psnr_from_mse(float(sq_diff.mean())),
            "source_ssim_luma": simple_ssim_luma(source, result),
            "source_ssim": simple_ssim_luma(source, result),
            "outside_mask_l1": float(diff[outside].mean()) if np.any(outside) else 0.0,
            "outside_mask_rmse": float(math.sqrt(float(sq_diff[outside].mean()))) if np.any(outside) else 0.0,
            "outside_mask_psnr": psnr_from_mse(float(sq_diff[outside].mean())) if np.any(outside) else 0.0,
            "inside_mask_l1": float(diff[inside].mean()) if inside is not None and np.any(inside) else 0.0,
            "inside_mask_rmse": float(math.sqrt(float(sq_diff[inside].mean()))) if inside is not None and np.any(inside) else 0.0,
            "inside_mask_psnr": psnr_from_mse(float(sq_diff[inside].mean())) if inside is not None and np.any(inside) else 0.0,
            "inside_blue_ratio": blue_ratio(result, mask),
        }
    )
    if clip_scorer is not None:
        source_prompt = str(record.get("source_prompt") or "")
        target_prompt = str(record.get("target_prompt") or "")
        scores = clip_scorer.score(result_path, [source_prompt, target_prompt])
        if len(scores) == 2:
            record["clip_source_score"] = scores[0]
            record["clip_target_score"] = scores[1]
            record["clip_target_minus_source"] = scores[1] - scores[0]
            record["edit_score"] = scores[1] - scores[0]
        record["clip_image_source_similarity"] = clip_scorer.image_similarity(source_path, result_path)
        record["clip_direction_similarity"] = clip_scorer.direction_similarity(
            source_path,
            result_path,
            source_prompt,
            target_prompt,
        )
        bbox = mask_bbox(mask, size)
        local_scores = clip_scorer.score_image(result_image.crop(bbox), [source_prompt, target_prompt])
        if len(local_scores) == 2:
            record["local_clip_source_score"] = local_scores[0]
            record["local_clip_target_score"] = local_scores[1]
            record["local_clip_target_minus_source"] = local_scores[1] - local_scores[0]
            record["local_crop_box"] = ",".join(str(value) for value in bbox)
    if dino_scorer is not None:
        record["dino_source_similarity"] = dino_scorer.similarity(source_path, result_path)
    if lpips_scorer is not None:
        record["lpips_full"] = lpips_scorer.score(source, result)
        if mask is None:
            outside_source = source
            outside_result = result
        else:
            outside_mask = outside.astype(np.float32)[:, :, None]
            outside_source = source
            outside_result = result * outside_mask + source * (1.0 - outside_mask)
        record["lpips_outside"] = lpips_scorer.score(outside_source, outside_result)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate paper-level RF h-Edit metrics.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/main_matrix"))
    parser.add_argument("--csv-output", type=Path, default=Path("experiments/main_metrics.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("experiments/main_metrics.json"))
    parser.add_argument(
        "--clip-model",
        default=None,
        help="Optional CLIP model name/path for text-image edit-success scoring.",
    )
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Optional DINO/DINOv2 model name/path for source-preservation embedding similarity.",
    )
    parser.add_argument("--dino-device", default="auto")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow metric models to be downloaded. By default only local cache is used.",
    )
    parser.add_argument(
        "--failure-annotations",
        type=Path,
        default=Path("experiments/failure_flags.json"),
        help=(
            "Optional JSON mapping run paths such as task/method/seed_10 or "
            "task/method/10 to failure_flag/failure_note annotations."
        ),
    )
    parser.add_argument(
        "--task-names",
        default=" ".join(MAIN_TASKS),
        help="Space/comma-separated task directory names to include, or 'all' for every detected run.",
    )
    parser.add_argument(
        "--method-names",
        default=" ".join(MAIN_METHODS),
        help="Space/comma-separated method directory names to include, or 'all' for every detected run.",
    )
    parser.add_argument(
        "--seeds",
        default=" ".join(MAIN_SEEDS),
        help="Space/comma-separated seed values to include, or 'all' for every detected run.",
    )
    parser.add_argument(
        "--mask-candidates",
        default=" ".join(MASK_CANDIDATES),
        help=(
            "Space/comma-separated run-relative mask paths to try for inside/outside metrics. "
            "Defaults to operation-v3 edit/core masks before legacy combined masks."
        ),
    )
    parser.add_argument(
        "--eval-mask-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing fixed per-task masks named "
            "{task}_eval_mask.png. When present, these masks override run-local masks."
        ),
    )
    parser.add_argument(
        "--preserve-floor-csv",
        type=Path,
        default=None,
        help="Optional preserve-floor reference table kept for compatibility with experiment runners.",
    )
    args = parser.parse_args()

    clip_scorer = None
    if args.clip_model:
        try:
            clip_scorer = ClipScorer(args.clip_model, args.clip_device, args.allow_download)
        except Exception as exc:
            print(f"WARNING: CLIP scorer unavailable: {exc}")
    dino_scorer = None
    if args.dino_model:
        try:
            dino_scorer = DinoScorer(args.dino_model, args.dino_device, args.allow_download)
        except Exception as exc:
            print(f"WARNING: DINO scorer unavailable: {exc}")
    lpips_scorer = None
    if args.compute_lpips:
        try:
            lpips_scorer = LpipsScorer(args.lpips_device)
        except Exception as exc:
            print(f"WARNING: LPIPS scorer unavailable: {exc}")

    failure_annotations: dict[str, Any] = {}
    if args.failure_annotations.exists():
        try:
            loaded = load_json(args.failure_annotations)
            if isinstance(loaded, dict):
                failure_annotations = loaded
            else:
                print(f"WARNING: failure annotations must be a JSON object: {args.failure_annotations}")
        except Exception as exc:
            print(f"WARNING: failure annotations unavailable: {exc}")

    task_names = None
    if args.task_names.strip().lower() != "all":
        task_names = {item.strip() for item in args.task_names.replace(",", " ").split() if item.strip()}
    method_names = None
    if args.method_names.strip().lower() != "all":
        method_names = {item.strip() for item in args.method_names.replace(",", " ").split() if item.strip()}
    seeds = None
    if args.seeds.strip().lower() != "all":
        seeds = {item.strip().removeprefix("seed_") for item in args.seeds.replace(",", " ").split() if item.strip()}
    mask_candidates = tuple(
        item.strip()
        for item in args.mask_candidates.replace(",", " ").split()
        if item.strip()
    )

    records = [
        evaluate_run(
            path,
            args.outputs_dir,
            clip_scorer=clip_scorer,
            dino_scorer=dino_scorer,
            lpips_scorer=lpips_scorer,
            failure_annotations=failure_annotations,
            mask_candidates=mask_candidates,
            eval_mask_dir=args.eval_mask_dir,
        )
        for path in find_run_dirs(
            args.outputs_dir,
            task_names=task_names,
            method_names=method_names,
            seeds=seeds,
        )
    ]
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    fieldnames = sorted({key for record in records for key in record.keys()})
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    complete = sum(1 for record in records if record.get("complete"))
    print(f"runs: {len(records)}")
    print(f"complete: {complete}")
    print(f"incomplete: {len(records) - complete}")
    print(f"csv: {args.csv_output}")
    print(f"json: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
