from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pretty_matrix"

BASE_TASKS = [
    "cat_crown",
    "tshirt_star",
    "pillow_vertical_fabric_strip",
]
BASE_METHODS = [
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
]
SEEDS = [10, 11, 12]
REQUIRED = ["result.png", "metadata.json", "stats.json"]

STRESS_LEVELS = ["050", "075", "125", "150", "200"]


def check_row(task: str, method: str, seed: int) -> list[str]:
    row_dir = OUT / task / method / f"seed_{seed}"
    return [
        str((row_dir / name).relative_to(ROOT))
        for name in REQUIRED
        if not (row_dir / name).is_file()
    ]


def main() -> int:
    missing: list[str] = []
    base_total = 0
    for task in BASE_TASKS:
        for method in BASE_METHODS:
            for seed in SEEDS:
                base_total += 1
                missing.extend(check_row(task, method, seed))

    stress_total = 0
    for task in BASE_TASKS:
        for method in BASE_METHODS:
            for level in STRESS_LEVELS:
                stress_total += 1
                missing.extend(check_row(task, f"{method}_e4x{level}", 10))

    present = base_total + stress_total - len({Path(m).parent for m in missing})
    print(f"base_rows_expected={base_total}")
    print(f"stress_rows_expected={stress_total}")
    print(f"rows_with_required_files_present_at_least={present}")
    print(f"missing_file_count={len(missing)}")
    if missing:
        print("missing:")
        for item in missing[:120]:
            print(item)
        return 1
    print("E4 file check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
