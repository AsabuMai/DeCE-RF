#!/usr/bin/env python3
"""Audit downloaded E2 baselines and run lightweight command smoke checks.

This script separates three states that are easy to conflate in paper text:
downloaded source, runnable command surface, and validated strict Core-6 output.
Only the last state should enter a reduced RF comparison table.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


STRICT_TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    paper_name: str
    family: str
    priority: str
    candidate_entrypoints: tuple[str, ...]
    legacy_runner: str = ""
    notes: str = ""


SPECS: dict[str, BaselineSpec] = {
    "FlowEdit": BaselineSpec(
        "FlowEdit",
        "flowedit",
        "rf_flow",
        "high_rf",
        ("run_script.py",),
        "scripts/archive_legacy_2026-05-11/run_flowedit_baseline.py",
        "RF-flow baseline; old adapter exists but revised strict Core-6 adapter is not validated.",
    ),
    "SplitFlow": BaselineSpec(
        "SplitFlow",
        "splitflow",
        "rf_flow",
        "high_rf",
        ("run_script.py",),
        "scripts/archive_legacy_2026-05-11/run_splitflow_baseline.py",
        "RF-flow baseline; old adapter exists but revised strict Core-6 adapter is not validated.",
    ),
    "FireFlow": BaselineSpec(
        "FireFlow",
        "fireflow",
        "rf_flow",
        "high_rf",
        ("src/edit.py", "src/gradio_demo.py", "fireflow/run.py", "inference.py", "run.py"),
        "scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py",
        "RF-flow baseline with package-style repo; entrypoint may require project-specific invocation.",
    ),
    "RF-Solver-Edit": BaselineSpec(
        "RF-Solver-Edit",
        "rf_solver_edit",
        "rf_solver",
        "high_rf",
        (
            "FLUX_Image_Edit/src/edit.py",
            "FLUX_Image_Edit/src/gradio_demo.py",
            "Hunyuanvideo_Video_Edit/edit_video.py",
            "scripts/inference.py",
            "inference.py",
            "run.py",
        ),
        "scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py",
        "High-priority RF baseline; repo has no obvious root-level Python entrypoint.",
    ),
    "ReFlex": BaselineSpec(
        "ReFlex",
        "reflex",
        "rf_rectified_flow",
        "high_rf",
        ("img_edit.py",),
        "scripts/archive_legacy_2026-05-11/run_reflex_baseline.py",
        "High-priority RF-style image editing baseline.",
    ),
    "FlowAlign": BaselineSpec(
        "FlowAlign",
        "flowalign",
        "rf_flow",
        "high_rf",
        ("run_edit.py", "run_t2i.py"),
        "",
        "High-priority flow baseline; needs Core-6 image-edit adapter.",
    ),
    "stable-flow": BaselineSpec(
        "stable-flow",
        "stable_flow",
        "rf_flow",
        "high_rf",
        ("run_stable_flow.py",),
        "",
        "High-priority flow baseline; needs Core-6 image-edit adapter.",
    ),
    "ZONE": BaselineSpec(
        "ZONE",
        "zone",
        "diffusion_inversion_attention",
        "medium_non_rf",
        ("inference.py",),
        "",
        "External editing baseline; useful audit row but not RF-native.",
    ),
    "instruct-pix2pix": BaselineSpec(
        "instruct-pix2pix",
        "instruct_pix2pix",
        "diffusion_instruction_editing",
        "medium_non_rf",
        ("edit_cli.py", "main.py"),
        "",
        "Instruction editing baseline; not RF-native.",
    ),
    "pix2pix-zero": BaselineSpec(
        "pix2pix-zero",
        "pix2pix_zero",
        "diffusion_inversion_direction",
        "medium_non_rf",
        ("app_gradio.py",),
        "",
        "Diffusion inversion/direction baseline; not RF-native.",
    ),
    "MasaCtrl": BaselineSpec(
        "MasaCtrl",
        "masactrl",
        "diffusion_attention_control",
        "medium_non_rf",
        ("run_synthesis_sdxl.py", "app.py"),
        "",
        "Attention-control baseline; not RF-native.",
    ),
    "prompt-to-prompt": BaselineSpec(
        "prompt-to-prompt",
        "prompt_to_prompt",
        "diffusion_attention_control",
        "medium_non_rf",
        ("prompt-to-prompt_stable.ipynb",),
        "",
        "Notebook/library-style baseline; needs adapter before runnable comparison.",
    ),
    "h-edit": BaselineSpec(
        "h-edit",
        "h_edit_r_p2p",
        "diffusion_bridge_p2p",
        "medium_non_rf",
        ("text-guided/main_demo.py",),
        "",
        "H-edit bridge baseline; old manifest references a Core-6 adapter, revised strict validation pending.",
    ),
    "ledits_pp": BaselineSpec(
        "ledits_pp",
        "ledits_pp",
        "diffusion_editing",
        "medium_non_rf",
        ("examples/LEdits.ipynb", "demo.py", "app.py"),
        "scripts/run_leditspp_baseline.py",
        "LEDITS++ has legacy Core-6 artifacts, but revised strict T2/T5 validation is pending.",
    ),
}


def parse_download_status(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[row["name"]] = row
    return rows


def parse_env_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["repo_name"]] = row
    return rows


def parse_reduced_methods(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row.get("method", "")
            for row in csv.DictReader(handle)
            if row.get("complete") == "True"
        }


def parse_manifest_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    status: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            baseline = row.get("baseline", "")
            if not baseline:
                continue
            rec = status.setdefault(
                baseline,
                {"complete": "0", "failed": "0", "pending": "0", "failure_examples": ""},
            )
            state = row.get("status", "pending") or "pending"
            if state not in {"complete", "failed", "pending"}:
                state = "pending"
            rec[state] = str(int(rec[state]) + 1)
            if state == "failed" and not rec["failure_examples"]:
                reason = row.get("failure_reason", "")
                rec["failure_examples"] = f"{row.get('task', '')}/seed{row.get('seed', '')}: {reason}"
    return status


def list_dependency_files(repo: Path) -> list[str]:
    names = []
    for pattern in ("requirements*.txt", "environment*.yml", "environment*.yaml", "pyproject.toml", "setup.py"):
        names.extend(str(path.relative_to(repo)) for path in repo.glob(pattern))
    return sorted(set(names))


def list_readmes(repo: Path) -> list[str]:
    return sorted(str(path.relative_to(repo)) for path in repo.glob("README*"))


def existing_entries(repo: Path, spec: BaselineSpec) -> list[str]:
    found = []
    for rel in spec.candidate_entrypoints:
        path = repo / rel
        if path.exists():
            found.append(rel)
    return found


def smoke_help(repo: Path, entry: str, python_bin: str, timeout_s: int) -> tuple[str, str, str]:
    path = repo / entry
    if not path.exists() or path.suffix != ".py":
        return "not_run", "", "no Python command entrypoint"
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    cmd = [python_bin, str(path), "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        snippet = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[:300].replace("\n", " ")
        return "help_timeout", " ".join(cmd), snippet
    except Exception as exc:  # noqa: BLE001 - audit output should keep moving.
        return "smoke_exception", " ".join(cmd), repr(exc)

    output = (proc.stdout + "\n" + proc.stderr).strip()
    snippet = output[:500].replace("\n", " ")
    if proc.returncode == 0:
        return "help_ok", " ".join(cmd), snippet
    if "ModuleNotFoundError" in output or "No module named" in output:
        return "needs_env", " ".join(cmd), snippet
    if "CUDA" in output or "torch" in output or "diffusers" in output or "transformers" in output:
        return "env_or_model_error", " ".join(cmd), snippet
    return f"help_exit_{proc.returncode}", " ".join(cmd), snippet


def classify_gate(family: str, smoke_status: str, entries: list[str], legacy_strict: bool) -> tuple[str, str]:
    if legacy_strict:
        return (
            "baseline_audit",
            "Legacy Core-6 artifacts exist, but revised strict T2/T5 coverage is not validated.",
        )
    if not entries:
        return "baseline_audit", "No runnable command entrypoint found; adapter required."
    if smoke_status == "help_ok":
        if family.startswith("rf"):
            return (
                "adapter_validation_queue",
                "Command smoke passed; still needs revised strict Core-6 generation and human/metric gate.",
            )
        return (
            "non_rf_external_queue",
            "Command smoke passed, but this is not RF-native; use as secondary external baseline only.",
        )
    if smoke_status in {"needs_env", "env_or_model_error", "help_timeout"}:
        return "baseline_audit", "Environment/checkpoint/entrypoint validation failed at smoke stage."
    return "baseline_audit", "Runnable validation did not pass."


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], strict_tasks: list[str]) -> None:
    downloaded = sum(1 for row in rows if row["download_status"] == "ok")
    smoke_ok = sum(1 for row in rows if row["smoke_status"] == "help_ok")
    rf_queue = [row for row in rows if row["e2_bucket"] == "adapter_validation_queue"]
    reduced = [row for row in rows if row["reduced_rf_comparison_entry"] == "yes"]
    audit = [row for row in rows if row["e2_bucket"] == "baseline_audit"]

    lines = [
        "# E2 Baseline Runnable Validation Audit",
        "",
        "Date: 2026-06-02",
        "",
        "Claim boundary: downloaded source, command-level smoke, and strict Core-6 output validation are separate states. A method enters the reduced RF comparison only after revised strict Core-6 generation is stable and is evaluated with the same human/metric gates as DeCE-RF.",
        "",
        f"Revised strict tasks: {', '.join(strict_tasks)}.",
        "",
        "## Summary",
        "",
        f"- Downloaded repositories: {downloaded}/{len(rows)}.",
        f"- Command smoke `--help` passed: {smoke_ok}/{len(rows)}.",
        f"- RF-family baselines queued for adapter/generation validation: {len(rf_queue)}.",
        f"- Reduced RF comparison entries now: {len(reduced)}.",
        f"- Baseline audit rows not claimable now: {len(audit)}.",
        "",
    ]

    if rf_queue:
        lines += ["## RF Queue", ""]
        for row in rf_queue:
            lines.append(
                f"- {row['baseline']}: {row['smoke_status']}; next step: {row['next_action']}"
            )
        lines.append("")

    lines += ["## Audit Table", ""]
    header = [
        "baseline",
        "family",
        "download_status",
        "entrypoints_found",
        "smoke_status",
        "strict_complete",
        "strict_failed",
        "e2_bucket",
        "strict_failure_example",
        "claim_status",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[key] or "-" for key in header) + " |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- `adapter_validation_queue` means the repository has a command surface worth adapting, not that it is already comparable.",
        "- `baseline_audit` means the method remains useful for transparency, but it must not support a claim that DeCE-RF beats that baseline.",
        "- LEDITS++ remains legacy evidence until it is rerun on the revised strict task set including `bowl_apple_inside` and `pillow_vertical_fabric_strip`.",
        "- SteerFlow and DiffEdit are not included in the downloaded-source count: SteerFlow has no confirmed public runnable repo; DiffEdit should be registered separately through a chosen diffusers implementation if used.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/rf_h_edit")
    parser.add_argument("--baseline-root", default="/workspace/baselines")
    parser.add_argument("--out-dir", default="/workspace/rf_h_edit/experiments/support_v3_2026-06-02")
    parser.add_argument("--python-bin", default="/workspace/rf_h_edit/.venv/bin/python")
    parser.add_argument("--env-registry", default="/workspace/baselines/e2_baseline_env_registry.csv")
    parser.add_argument(
        "--reduced-metrics-csv",
        default="/workspace/rf_h_edit/experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv",
    )
    parser.add_argument(
        "--strict-manifest",
        default="/workspace/rf_h_edit/experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv",
    )
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    baseline_root = Path(args.baseline_root)
    src_root = baseline_root / "src"
    out_dir = Path(args.out_dir)
    python_bin = args.python_bin if Path(args.python_bin).exists() else sys.executable
    env_registry = parse_env_registry(Path(args.env_registry))
    reduced_methods = parse_reduced_methods(Path(args.reduced_metrics_csv))
    manifest_status = parse_manifest_status(Path(args.strict_manifest))

    downloads = parse_download_status(baseline_root / "download_status.tsv")
    rows: list[dict[str, str]] = []

    for name in sorted(downloads):
        row = downloads[name]
        spec = SPECS.get(
            name,
            BaselineSpec(name, name, "unknown", "unknown", tuple(), "", "Unregistered baseline spec."),
        )
        repo = Path(row["path"])
        deps = list_dependency_files(repo) if repo.exists() else []
        readmes = list_readmes(repo) if repo.exists() else []
        entries = existing_entries(repo, spec) if repo.exists() else []
        legacy_runner_exists = bool(spec.legacy_runner and (repo_root / spec.legacy_runner).exists())

        smoke_status = "not_run"
        smoke_command = ""
        smoke_detail = ""
        row_python = python_bin
        env_row = env_registry.get(name, {})
        if env_row.get("install_status") == "ok" and Path(env_row.get("env_python", "")).exists():
            row_python = env_row["env_python"]

        if not args.no_smoke and entries:
            smoke_status, smoke_command, smoke_detail = smoke_help(
                repo, entries[0], row_python, args.timeout_s
            )
        elif not entries:
            smoke_status = "no_entrypoint"
            smoke_detail = "No candidate entrypoint exists on disk."

        legacy_artifacts = False
        if spec.name == "ledits_pp":
            legacy_artifacts = (
                repo_root
                / "experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.csv"
            ).exists()

        e2_bucket, claim_detail = classify_gate(spec.family, smoke_status, entries, legacy_artifacts)
        strict_status = manifest_status.get(spec.paper_name, {})
        reduced_entry = "no"
        claim_status = "do_not_claim_beat"
        next_action = "keep in baseline audit"
        if spec.paper_name in reduced_methods:
            e2_bucket = "reduced_rf_comparison"
            reduced_entry = "yes"
            next_action = "included in reduced target-mode RF comparison; keep remaining RF baselines in adapter/generation validation"
            claim_detail = "Completed revised strict Core-6 generation, fixed-mask metrics, and internal visual audit for seeds 10/11/12."
        elif int(strict_status.get("failed", "0")) > 0:
            e2_bucket = "baseline_audit"
            claim_detail = "Strict target-mode generation smoke failed; do not use for reduced comparison."
            next_action = strict_status.get("failure_examples", "inspect strict generation logs")
        elif e2_bucket == "adapter_validation_queue":
            next_action = "build revised strict Core-6 adapter, run seed-10 smoke generation, then seeds 10/11/12"
        elif e2_bucket == "non_rf_external_queue":
            next_action = "optional secondary external comparison after strict generation validation"
        elif spec.name == "ledits_pp":
            next_action = "rerun LEDITS++ on revised strict Core-6 before using in paper-facing comparison"

        rows.append(
            {
                "baseline": spec.paper_name,
                "repo_name": spec.name,
                "family": spec.family,
                "priority": spec.priority,
                "download_status": row["status"],
                "repo_path": row["path"],
                "commit_or_note": row["commit_or_note"],
                "readmes": ";".join(readmes),
                "dependency_files": ";".join(deps),
                "candidate_entrypoints": ";".join(spec.candidate_entrypoints),
                "entrypoints_found": ";".join(entries),
                "legacy_runner": spec.legacy_runner,
                "legacy_runner_exists": "yes" if legacy_runner_exists else "no",
                "env_name": env_row.get("env_name", ""),
                "env_python": env_row.get("env_python", ""),
                "env_install_status": env_row.get("install_status", ""),
                "env_log_path": env_row.get("log_path", ""),
                "smoke_status": smoke_status,
                "smoke_command": smoke_command,
                "smoke_detail": smoke_detail,
                "strict_complete": strict_status.get("complete", "0"),
                "strict_failed": strict_status.get("failed", "0"),
                "strict_pending": strict_status.get("pending", "0"),
                "strict_failure_example": strict_status.get("failure_examples", ""),
                "e2_bucket": e2_bucket,
                "reduced_rf_comparison_entry": reduced_entry,
                "claim_status": claim_status,
                "claim_detail": claim_detail,
                "next_action": next_action,
                "notes": spec.notes,
            }
        )

    write_csv(out_dir / "e2_baseline_runnable_validation.csv", rows)
    registry_rows = [
        {
            "baseline": row["baseline"],
            "repo_name": row["repo_name"],
            "family": row["family"],
            "download_status": row["download_status"],
            "repo_path": row["repo_path"],
            "commit_or_note": row["commit_or_note"],
        }
        for row in rows
    ]
    write_csv(out_dir / "e2_baseline_download_registry.csv", registry_rows)
    write_markdown(out_dir / "e2_baseline_audit.md", rows, STRICT_TASKS)

    print(f"Wrote {out_dir / 'e2_baseline_download_registry.csv'}")
    print(f"Wrote {out_dir / 'e2_baseline_runnable_validation.csv'}")
    print(f"Wrote {out_dir / 'e2_baseline_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
