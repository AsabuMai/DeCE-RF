#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shlex
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edit_cli_args import build_parser as build_edit_parser
try:
    from run_edit_sd3 import load_sd3_pipeline, run_edit
except ImportError as exc:  # Fallback for CLI-only run_edit_sd3.py snapshots.
    load_sd3_pipeline = None
    run_edit = None
    BATCH_API_IMPORT_ERROR = exc
else:
    BATCH_API_IMPORT_ERROR = None


def _read_command_entries(paths: list[str], manifest: str | None) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    if manifest:
        manifest_path = Path(manifest)
        for raw in manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            command_path = Path(line)
            if not command_path.is_absolute():
                command_path = (manifest_path.parent / command_path).resolve()
            entries.append((command_path, command_path.read_text(encoding="utf-8").strip()))
    for raw_path in paths:
        path = Path(raw_path)
        entries.append((path, path.read_text(encoding="utf-8").strip()))
    return entries


def _extract_edit_argv(command: str) -> list[str]:
    tokens = shlex.split(command)
    script_index = None
    for idx, token in enumerate(tokens):
        if token.endswith("run_edit_sd3.py"):
            script_index = idx
            break
    if script_index is None:
        raise ValueError("command does not contain run_edit_sd3.py")
    return tokens[script_index + 1 :]


def _parse_edit_args(command: str):
    parser = build_edit_parser()
    return parser.parse_args(_extract_edit_argv(command))


def _complete(args) -> bool:
    metadata_output = args.metadata_output
    if metadata_output is None:
        root, _ = os.path.splitext(args.output)
        metadata_output = f"{root}_metadata.json"
    required = [args.output, args.stats_output, metadata_output]
    return all(path and Path(path).is_file() and Path(path).stat().st_size > 0 for path in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple SD3 edit commands while reusing one loaded pipeline.")
    parser.add_argument("command_files", nargs="*", help="command.txt files produced by run_pretty_matrix.sh")
    parser.add_argument("--manifest", help="Text file listing command.txt files, one per line")
    parser.add_argument("--skip-existing", action="store_true", help="Skip commands whose output/stats/metadata already exist")
    parser.add_argument("--summary-output", help="Optional JSON summary path")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop at the first failed command")
    args = parser.parse_args()

    entries = _read_command_entries(args.command_files, args.manifest)
    if not entries:
        raise SystemExit("No command files provided.")

    parsed = []
    for command_path, command in entries:
        edit_args = _parse_edit_args(command)
        parsed.append((command_path, command, edit_args))

    if load_sd3_pipeline is None or run_edit is None:
        print(
            "[sd3-batch] batch API unavailable; falling back to subprocess mode: "
            f"{BATCH_API_IMPORT_ERROR!r}"
        )
        summary = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command_count": len(parsed),
            "mode": "subprocess_fallback",
            "records": [],
        }
        for index, (command_path, command, edit_args) in enumerate(parsed, start=1):
            record = {
                "index": index,
                "command_file": str(command_path),
                "output": edit_args.output,
                "stats_output": edit_args.stats_output,
                "metadata_output": edit_args.metadata_output,
                "status": "pending",
            }
            started = time.perf_counter()
            print(f"[sd3-batch] {index}/{len(parsed)} {command_path}")
            try:
                if args.skip_existing and _complete(edit_args):
                    record["status"] = "skipped_existing"
                    print(f"[sd3-batch] skip existing: {edit_args.output}")
                else:
                    subprocess.run(command, shell=True, executable="/bin/bash", cwd=ROOT, check=True)
                    record["status"] = "complete"
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = repr(exc)
                print(f"[sd3-batch] failed: {exc!r}", file=sys.stderr)
                if args.stop_on_failure:
                    summary["records"].append(record)
                    raise
            finally:
                record["runtime_seconds"] = time.perf_counter() - started
                summary["records"].append(record)
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if args.summary_output:
            out = Path(args.summary_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        complete = sum(1 for r in summary["records"] if r["status"] == "complete")
        skipped = sum(1 for r in summary["records"] if r["status"] == "skipped_existing")
        failed = sum(1 for r in summary["records"] if r["status"] == "failed")
        print(f"[sd3-batch] done complete={complete} skipped={skipped} failed={failed}")
        if failed:
            raise SystemExit(1)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[sd3-batch] loading pipeline once on {device} for {len(parsed)} commands")
    pipe = load_sd3_pipeline(parsed[0][2], device)
    scheduler = pipe.scheduler

    summary = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_count": len(parsed),
        "records": [],
    }
    for index, (command_path, command, edit_args) in enumerate(parsed, start=1):
        record = {
            "index": index,
            "command_file": str(command_path),
            "output": edit_args.output,
            "stats_output": edit_args.stats_output,
            "metadata_output": edit_args.metadata_output,
            "status": "pending",
        }
        started = time.perf_counter()
        print(f"[sd3-batch] {index}/{len(parsed)} {command_path}")
        try:
            if args.skip_existing and _complete(edit_args):
                record["status"] = "skipped_existing"
                print(f"[sd3-batch] skip existing: {edit_args.output}")
            else:
                run_edit(edit_args, pipe=pipe, scheduler=scheduler)
                record["status"] = "complete"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            print(f"[sd3-batch] failed: {exc!r}", file=sys.stderr)
            if args.stop_on_failure:
                summary["records"].append(record)
                raise
        finally:
            record["runtime_seconds"] = time.perf_counter() - started
            summary["records"].append(record)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.summary_output:
        out = Path(args.summary_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete = sum(1 for r in summary["records"] if r["status"] == "complete")
    skipped = sum(1 for r in summary["records"] if r["status"] == "skipped_existing")
    failed = sum(1 for r in summary["records"] if r["status"] == "failed")
    print(f"[sd3-batch] done complete={complete} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
