from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = {
    "T1": "cat_crown",
    "T2": "backpack_blue",
    "T3": "yellow_car_blue",
    "T4": "rabbit_sunglasses",
}
METHODS = {
    "M0": "base_only",
    "M1": "direct_target",
    "M2": "anchor_only",
    "M3": "decoupled_rec",
    "M4": "full",
}
REQUIRED_FILES = ("result.png", "stats.json", "metadata.json", "command.txt")


def parse_list(value: str, allowed: dict[str, str] | None = None) -> list[str]:
    items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if allowed is not None:
        unknown = [item for item in items if item not in allowed and item not in allowed.values()]
        if unknown:
            raise ValueError(f"Unknown values: {', '.join(unknown)}")
    return items


def resolve(item: str, mapping: dict[str, str]) -> tuple[str, str]:
    if item in mapping:
        return item, mapping[item]
    for key, value in mapping.items():
        if item == value:
            return key, value
    raise ValueError(f"Unknown item: {item}")


def audit(outputs_dir: Path, tasks: list[str], methods: list[str], seeds: list[str]) -> list[dict]:
    records: list[dict] = []
    for task_item in tasks:
        task_id, task_name = resolve(task_item, TASKS)
        for method_item in methods:
            method_id, method_name = resolve(method_item, METHODS)
            for seed in seeds:
                run_dir = outputs_dir / task_name / method_name / f"seed_{seed}"
                present = {name: (run_dir / name).exists() for name in REQUIRED_FILES}
                missing = [name for name, exists in present.items() if not exists]
                records.append(
                    {
                        "task_id": task_id,
                        "task": task_name,
                        "method_id": method_id,
                        "method": method_name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "exists": run_dir.exists(),
                        "complete": run_dir.exists() and not missing,
                        "missing": missing,
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit planned main-matrix coverage, not just existing run records.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/main_matrix"))
    parser.add_argument("--tasks", default="T1 T2 T3 T4")
    parser.add_argument("--methods", default="M0 M1 M2 M3 M4")
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--json-output", type=Path, default=Path("experiments/main_matrix_coverage_audit.json"))
    args = parser.parse_args()

    tasks = parse_list(args.tasks, TASKS)
    methods = parse_list(args.methods, METHODS)
    seeds = parse_list(args.seeds)
    records = audit(args.outputs_dir, tasks, methods, seeds)
    complete = [record for record in records if record["complete"]]
    missing = [record for record in records if not record["complete"]]

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    print(f"planned: {len(records)}")
    print(f"complete: {len(complete)}")
    print(f"missing_or_incomplete: {len(missing)}")
    if missing:
        print()
        print("missing_or_incomplete runs:")
        for record in missing:
            reason = "missing run_dir" if not record["exists"] else f"missing {', '.join(record['missing'])}"
            print(
                f"- {record['task_id']}/{record['task']} "
                f"{record['method_id']}/{record['method']} seed_{record['seed']}: {reason}"
            )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
