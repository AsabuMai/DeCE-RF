from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = ("result.png", "stats.json", "metadata.json", "command.txt")


def _has_experiment_file(path: Path) -> bool:
    return any((path / name).exists() for name in REQUIRED_FILES)


def _load_metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def audit_outputs(outputs_dir: Path) -> list[dict]:
    records: list[dict] = []
    run_dirs = [
        path
        for path in outputs_dir.rglob("*")
        if path.is_dir()
        and path.name != "archive"
        and len(path.relative_to(outputs_dir).parts) == 3
        and path.relative_to(outputs_dir).parts[-1].startswith("seed_")
        and _has_experiment_file(path)
    ]
    for run_dir in sorted(run_dirs):
        if "archive" in run_dir.relative_to(outputs_dir).parts:
            continue

        present = {name: (run_dir / name).exists() for name in REQUIRED_FILES}
        missing = [name for name, exists in present.items() if not exists]
        metadata = _load_metadata(run_dir / "metadata.json") if present["metadata.json"] else {}
        rel_run = str(run_dir.relative_to(outputs_dir))
        records.append(
            {
                "run": rel_run,
                "path": str(run_dir),
                "missing": missing,
                "git_commit": metadata.get("git_commit"),
                "source_prompt": metadata.get("source_prompt"),
                "target_prompt": metadata.get("target_prompt"),
                "seed": metadata.get("seed"),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit RF h-Edit experiment folders for paper reproducibility records."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing per-run output folders.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the full audit JSON.",
    )
    args = parser.parse_args()

    records = audit_outputs(args.outputs_dir)
    complete = [record for record in records if not record["missing"]]
    incomplete = [record for record in records if record["missing"]]

    print(f"runs: {len(records)}")
    print(f"complete: {len(complete)}")
    print(f"incomplete: {len(incomplete)}")
    if incomplete:
        print()
        print("incomplete runs:")
        for record in incomplete:
            print(f"- {record['run']}: missing {', '.join(record['missing'])}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(records, indent=2) + "\n")

    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
