from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluate_paper_metrics import MASK_CANDIDATES, load_json, load_mask, load_rgb, simple_ssim_luma


RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
BASE_METHODS = ("support_v3_fixed", "support_v3_controller_rmsgap")


@dataclass
class MethodInfo:
    family: str
    base_method: str
    condition: str
    edit_scale: float | None
    perturbation: str


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def parse_method(method: str, edit_scales: set[str], perturbations: set[str]) -> MethodInfo | None:
    for base in BASE_METHODS:
        if method == base:
            return MethodInfo("baseline", base, "baseline", None, "")
        prefix = f"{base}_edit"
        if method.startswith(prefix):
            tag = method.removeprefix(prefix)
            if tag in edit_scales:
                return MethodInfo("edit_strength", base, tag, int(tag) / 100.0, "")
        prefix = f"{base}_pert_"
        if method.startswith(prefix):
            perturb = method.removeprefix(prefix)
            if perturb in perturbations:
                return MethodInfo("support_perturb", base, perturb, None, perturb)
    return None


def scale_tag(value: str) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def find_runs(outputs_dir: Path, tasks: set[str], seeds: set[str]) -> list[Path]:
    runs: list[Path] = []
    for task_dir in sorted(outputs_dir.iterdir() if outputs_dir.exists() else []):
        if not task_dir.is_dir() or task_dir.name not in tasks:
            continue
        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            for seed_dir in sorted(method_dir.glob("seed_*")):
                if not seed_dir.is_dir():
                    continue
                seed = seed_dir.name.removeprefix("seed_")
                if seed in seeds and (seed_dir / "metadata.json").exists():
                    runs.append(seed_dir)
    return runs


def split_run(run_dir: Path, outputs_dir: Path) -> tuple[str, str, str]:
    task, method, seed = run_dir.relative_to(outputs_dir).parts[-3:]
    return task, method, seed.removeprefix("seed_")


def first_mask(run_dir: Path, size: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    for candidate in MASK_CANDIDATES:
        path = run_dir / candidate
        mask = load_mask(path, size)
        if mask is not None:
            return mask, str(path)
    return None, ""


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


def evaluate_run(
    run_dir: Path,
    outputs_dir: Path,
    info: MethodInfo,
    clip_scorer: ClipScorer | None,
    lpips_scorer: LpipsScorer | None,
) -> dict[str, Any]:
    task, method, seed = split_run(run_dir, outputs_dir)
    result_path = run_dir / "result.png"
    metadata_path = run_dir / "metadata.json"
    stats_path = run_dir / "stats.json"
    if not result_path.exists() or not stats_path.exists():
        return {
            "task": task,
            "method": method,
            "seed": seed,
            "base_method": info.base_method,
            "family": info.family,
            "condition": info.condition,
            "complete": False,
            "missing": "result_or_stats",
        }

    metadata = load_json(metadata_path)
    stats = load_json(stats_path)
    rows = stats if isinstance(stats, list) else stats.get("steps", [])
    final = rows[-1] if rows else {}
    source_path = Path(metadata.get("image", ""))
    if not source_path.exists():
        return {
            "task": task,
            "method": method,
            "seed": seed,
            "base_method": info.base_method,
            "family": info.family,
            "condition": info.condition,
            "complete": False,
            "missing": "source_image",
        }

    result_image = Image.open(result_path).convert("RGB")
    size = result_image.size
    result = np.asarray(result_image, dtype=np.float32) / 255.0
    source = load_rgb(source_path, size=size)
    diff = np.abs(result - source)
    mask, mask_path = first_mask(run_dir, size)
    outside = np.ones(size[::-1], dtype=bool) if mask is None else mask <= 0.2
    inside = None if mask is None else mask > 0.2
    ssim = simple_ssim_luma(source, result)

    record: dict[str, Any] = {
        "task": task,
        "method": method,
        "seed": seed,
        "base_method": info.base_method,
        "family": info.family,
        "condition": info.condition,
        "edit_scale": "" if info.edit_scale is None else info.edit_scale,
        "perturbation": info.perturbation,
        "complete": True,
        "missing": "",
        "source_image": str(source_path),
        "result_image": str(result_path),
        "mask_path": mask_path,
        "eval_mask_area": float((mask > 0.2).mean()) if mask is not None else 1.0,
        "final_mask_area": float(final.get("mask_area_ratio", 0.0) or 0.0),
        "outside_mask_l1": float(diff[outside].mean()) if np.any(outside) else 0.0,
        "inside_mask_l1": float(diff[inside].mean()) if inside is not None and np.any(inside) else 0.0,
        "source_l1": float(diff.mean()),
        "source_ssim_luma": ssim,
        "ssim_loss": 1.0 - ssim,
        "runtime_seconds": metadata.get("runtime_seconds", ""),
        "source_prompt": metadata.get("source_prompt") or metadata.get("effective_source_prompt") or "",
        "target_prompt": metadata.get("target_prompt") or metadata.get("effective_target_prompt") or "",
    }

    if clip_scorer is not None:
        source_prompt = str(record["source_prompt"])
        target_prompt = str(record["target_prompt"])
        scores = clip_scorer.score_image(result_image, [source_prompt, target_prompt])
        if len(scores) == 2:
            record["clip_source_score"] = scores[0]
            record["clip_target_score"] = scores[1]
            record["clip_target_minus_source"] = scores[1] - scores[0]
        bbox = mask_bbox(mask, size)
        crop = result_image.crop(bbox)
        local_scores = clip_scorer.score_image(crop, [source_prompt, target_prompt])
        if len(local_scores) == 2:
            record["local_clip_source_score"] = local_scores[0]
            record["local_clip_target_score"] = local_scores[1]
            record["local_clip_target_minus_source"] = local_scores[1] - local_scores[0]
            record["local_crop_box"] = ",".join(str(v) for v in bbox)

    if lpips_scorer is not None:
        record["lpips_full"] = lpips_scorer.score(source, result)
        if mask is None:
            outside_source = source
            outside_result = result
        else:
            outside_mask = (mask <= 0.2).astype(np.float32)[:, :, None]
            outside_source = source
            outside_result = result * outside_mask + source * (1.0 - outside_mask)
        record["lpips_outside"] = lpips_scorer.score(outside_source, outside_result)

    return record


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    if value == "" or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def grouped_summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        record = {key: value for key, value in zip(keys, group_key)}
        record["n"] = len(group_rows)
        for metric in metrics:
            values = [value for value in (numeric(row, metric) for row in group_rows) if value is not None]
            if values:
                record[f"{metric}_mean"] = mean(values)
                record[f"{metric}_min"] = min(values)
                record[f"{metric}_max"] = max(values)
        out.append(record)
    return out


def pareto_auc(points: list[dict[str, Any]], x_key: str, y_key: str, x_min: float, x_max: float) -> float:
    xy = []
    for point in points:
        x = numeric(point, x_key)
        y = numeric(point, y_key)
        if x is not None and y is not None:
            xy.append((x, y))
    if not xy:
        return 0.0
    xy.sort(key=lambda item: item[0])
    if x_max <= x_min:
        return max(y for _, y in xy)
    frontier = []
    best_y = -float("inf")
    for x, y in xy:
        best_y = max(best_y, y)
        frontier.append(((x - x_min) / (x_max - x_min), best_y))
    if frontier[0][0] > 0.0:
        frontier.insert(0, (0.0, frontier[0][1]))
    if frontier[-1][0] < 1.0:
        frontier.append((1.0, frontier[-1][1]))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(frontier, frontier[1:]):
        area += (x1 - x0) * 0.5 * (y0 + y1)
    return area


def baseline_budget(rows: list[dict[str, Any]], task: str) -> tuple[float, float]:
    fixed = [
        row for row in rows
        if row.get("task") == task and row.get("base_method") == "support_v3_fixed" and row.get("family") == "baseline"
    ]
    if not fixed:
        fixed = [
            row for row in rows
            if row.get("task") == task
            and row.get("base_method") == "support_v3_fixed"
            and row.get("family") == "edit_strength"
            and str(row.get("condition")) == "100"
        ]
    outside_values = [value for value in (numeric(row, "outside_mask_l1") for row in fixed) if value is not None]
    ssim_values = [value for value in (numeric(row, "ssim_loss") for row in fixed) if value is not None]
    return mean(outside_values), mean(ssim_values)


def pareto_summary(
    rows: list[dict[str, Any]],
    success_key: str,
    outside_budget_delta: float,
    ssim_budget_delta: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("family") not in {"edit_strength", "support_perturb"}:
            continue
        groups.setdefault((str(row.get("task")), str(row.get("family"))), []).append(row)

    out: list[dict[str, Any]] = []
    for (task, family), group_rows in sorted(groups.items()):
        outside_values = [value for value in (numeric(row, "outside_mask_l1") for row in group_rows) if value is not None]
        ssim_values = [value for value in (numeric(row, "ssim_loss") for row in group_rows) if value is not None]
        if not outside_values or not ssim_values:
            continue
        outside_min, outside_max = min(outside_values), max(outside_values)
        ssim_min, ssim_max = min(ssim_values), max(ssim_values)
        outside_base, ssim_base = baseline_budget(rows, task)
        outside_budget = outside_base + outside_budget_delta
        ssim_budget = ssim_base + ssim_budget_delta
        for base_method in BASE_METHODS:
            method_rows = [row for row in group_rows if row.get("base_method") == base_method]
            outside_ok = [
                row for row in method_rows
                if (numeric(row, "outside_mask_l1") is not None and numeric(row, "outside_mask_l1") <= outside_budget)
            ]
            ssim_ok = [
                row for row in method_rows
                if (numeric(row, "ssim_loss") is not None and numeric(row, "ssim_loss") <= ssim_budget)
            ]
            joint_ok = [
                row for row in method_rows
                if (
                    numeric(row, "outside_mask_l1") is not None
                    and numeric(row, "outside_mask_l1") <= outside_budget
                    and numeric(row, "ssim_loss") is not None
                    and numeric(row, "ssim_loss") <= ssim_budget
                )
            ]
            record = {
                "task": task,
                "family": family,
                "base_method": base_method,
                "n": len(method_rows),
                "success_key": success_key,
                "outside_budget": outside_budget,
                "ssim_loss_budget": ssim_budget,
                "pareto_auc_outside": pareto_auc(method_rows, "outside_mask_l1", success_key, outside_min, outside_max),
                "pareto_auc_ssim_loss": pareto_auc(method_rows, "ssim_loss", success_key, ssim_min, ssim_max),
                "best_success_under_outside_budget": best_success(outside_ok, success_key),
                "best_success_under_ssim_budget": best_success(ssim_ok, success_key),
                "best_success_under_joint_budget": best_success(joint_ok, success_key),
                "outside_budget_eligible": len(outside_ok),
                "ssim_budget_eligible": len(ssim_ok),
                "joint_budget_eligible": len(joint_ok),
            }
            out.append(record)
    return out


def best_success(rows: list[dict[str, Any]], success_key: str) -> float:
    values = [value for value in (numeric(row, success_key) for row in rows) if value is not None]
    return max(values) if values else 0.0


def paired_delta_summary(rows: list[dict[str, Any]], success_key: str) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("family") not in {"edit_strength", "support_perturb"}:
            continue
        key = (
            str(row.get("task")),
            str(row.get("family")),
            str(row.get("condition")),
            str(row.get("seed")),
        )
        indexed.setdefault(key, {})[str(row.get("base_method"))] = row
    out = []
    for (task, family, condition, seed), pair in sorted(indexed.items()):
        fixed = pair.get("support_v3_fixed")
        rmsgap = pair.get("support_v3_controller_rmsgap")
        if not fixed or not rmsgap:
            continue
        record = {
            "task": task,
            "family": family,
            "condition": condition,
            "seed": seed,
        }
        for metric in (success_key, "outside_mask_l1", "inside_mask_l1", "ssim_loss", "source_ssim_luma", "lpips_outside"):
            r_value = numeric(rmsgap, metric)
            f_value = numeric(fixed, metric)
            if r_value is not None and f_value is not None:
                record[f"delta_{metric}"] = r_value - f_value
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def plot_tradeoffs(rows: list[dict[str, Any]], output_dir: Path, success_key: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable: {exc}")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "support_v3_fixed": "#3b82f6",
        "support_v3_controller_rmsgap": "#dc2626",
    }
    labels = {
        "support_v3_fixed": "fixed",
        "support_v3_controller_rmsgap": "rmsgap",
    }
    for family in ("edit_strength", "support_perturb"):
        family_rows = [row for row in rows if row.get("family") == family]
        for task in sorted({str(row.get("task")) for row in family_rows}):
            task_rows = [row for row in family_rows if row.get("task") == task]
            if not task_rows:
                continue
            for x_key, x_label in (
                ("outside_mask_l1", "outside L1"),
                ("ssim_loss", "SSIM loss"),
                ("lpips_outside", "outside LPIPS"),
            ):
                if not any(numeric(row, x_key) is not None for row in task_rows):
                    continue
                fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=160)
                for base in BASE_METHODS:
                    method_rows = [row for row in task_rows if row.get("base_method") == base]
                    if family == "edit_strength":
                        method_rows.sort(key=lambda row: float(row.get("edit_scale") or 0.0))
                    else:
                        method_rows.sort(key=lambda row: str(row.get("condition")))
                    xs = [numeric(row, x_key) for row in method_rows]
                    ys = [numeric(row, success_key) for row in method_rows]
                    valid = [(x, y, row) for x, y, row in zip(xs, ys, method_rows) if x is not None and y is not None]
                    if not valid:
                        continue
                    ax.plot(
                        [x for x, _, _ in valid],
                        [y for _, y, _ in valid],
                        marker="o",
                        linewidth=1.5,
                        markersize=4,
                        color=colors[base],
                        label=labels[base],
                    )
                    for x, y, row in valid:
                        text = str(row.get("edit_scale") or row.get("condition"))
                        ax.annotate(text, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
                if not ax.lines:
                    plt.close(fig)
                    continue
                ax.set_title(f"{task}: {family}")
                ax.set_xlabel(x_label)
                ax.set_ylabel(success_key)
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(plot_dir / f"{task}_{family}_{success_key}_vs_{x_key}.png")
                plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze fixed vs rmsgap controller stress sweeps.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/support_v3_2026-05-11/controller_stress"))
    parser.add_argument("--tasks", default="cat_crown dog_sunglasses mug_heart")
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--edit-scales", default="0.5 0.75 1.0 1.25 1.5 2.0")
    parser.add_argument("--perturbations", default="erode dilate shift boundary_noise holes")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="auto")
    parser.add_argument("--outside-budget-delta", type=float, default=0.001)
    parser.add_argument("--ssim-budget-delta", type=float, default=0.002)
    args = parser.parse_args()

    tasks = set(parse_list(args.tasks))
    seeds = {seed.removeprefix("seed_") for seed in parse_list(args.seeds)}
    edit_tags = {scale_tag(value) for value in parse_list(args.edit_scales)}
    perturbations = set(parse_list(args.perturbations))

    clip_scorer = None
    if args.clip_model:
        try:
            clip_scorer = ClipScorer(args.clip_model, args.clip_device, args.allow_download)
        except Exception as exc:
            print(f"WARNING: CLIP scorer unavailable: {exc}")
    lpips_scorer = None
    if args.compute_lpips:
        try:
            lpips_scorer = LpipsScorer(args.lpips_device)
        except Exception as exc:
            print(f"WARNING: LPIPS scorer unavailable: {exc}")

    records = []
    for run_dir in find_runs(args.outputs_dir, tasks, seeds):
        _, method, _ = split_run(run_dir, args.outputs_dir)
        info = parse_method(method, edit_tags, perturbations)
        if info is None:
            continue
        records.append(evaluate_run(run_dir, args.outputs_dir, info, clip_scorer, lpips_scorer))

    success_key = "local_clip_target_minus_source"
    if not any(numeric(row, success_key) is not None for row in records):
        success_key = "clip_target_minus_source"

    metrics = [
        "clip_target_minus_source",
        "local_clip_target_minus_source",
        "outside_mask_l1",
        "inside_mask_l1",
        "source_ssim_luma",
        "ssim_loss",
        "lpips_outside",
        "lpips_full",
    ]
    edit_rows = [row for row in records if row.get("family") == "edit_strength"]
    perturb_rows = [row for row in records if row.get("family") == "support_perturb"]
    edit_summary = grouped_summary(edit_rows, ["task", "base_method", "edit_scale"], metrics)
    perturb_summary = grouped_summary(perturb_rows, ["task", "base_method", "perturbation"], metrics)
    pareto_rows = pareto_summary(records, success_key, args.outside_budget_delta, args.ssim_budget_delta)
    paired_rows = paired_delta_summary(records, success_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stress_raw_metrics.csv", records)
    write_json(args.output_dir / "stress_raw_metrics.json", records)
    write_csv(args.output_dir / "edit_strength_summary.csv", edit_summary)
    write_csv(args.output_dir / "support_perturb_summary.csv", perturb_summary)
    write_csv(args.output_dir / "pareto_budget_summary.csv", pareto_rows)
    write_csv(args.output_dir / "paired_rmsgap_minus_fixed.csv", paired_rows)
    plot_tradeoffs(records, args.output_dir, success_key)

    print(f"records: {len(records)}")
    print(f"success_key: {success_key}")
    print(f"raw: {args.output_dir / 'stress_raw_metrics.csv'}")
    print(f"pareto: {args.output_dir / 'pareto_budget_summary.csv'}")
    print(f"paired: {args.output_dir / 'paired_rmsgap_minus_fixed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
