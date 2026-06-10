from __future__ import annotations

import importlib.util
from pathlib import Path


TASKS = [
    "cat_crown",
    "dog_bow_tie_phase2",
    "dog_front_sunglasses_phase2",
    "bowl_apple_inside",
    "white_bowl_orange_tabletop_phase2",
    "brown_bowl_lemon_phase2",
    "tshirt_star",
    "mug_heart",
    "tote_leaf",
    "red_office_chair_to_blue_office_chair",
    "green_mug_orange_phase2",
    "yellow_vase_blue_phase2",
]


def load_base_module():
    path = Path(__file__).with_name("summarize_e4_controller_ablation.py")
    spec = importlib.util.spec_from_file_location("e4_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_base_module()
    exp = module.EXP
    out = exp / "e4_controller_ablation_t1_t4"
    original_read_csv = module.read_csv

    def read_csv_mapped(path: Path):
        if path.name == "e4_controller_base_metrics.csv":
            return original_read_csv(exp / "e4_t1_t4_controller_base_metrics.csv")
        if path.name == "e4_edit_strength_metrics.csv":
            return original_read_csv(exp / "e4_t1_t4_edit_strength_metrics.csv")
        return original_read_csv(path)

    module.read_csv = read_csv_mapped
    module.OUT = out
    module.TASKS = TASKS
    result = module.main()

    summary = out / "e4_controller_ablation_summary.md"
    if summary.exists():
        text = summary.read_text(encoding="utf-8")
        text = text.replace(
            "Scope: cat_crown, tshirt_star, and pillow_vertical_fabric_strip.",
            "Scope: Phase2 T1-T4, 12 tasks.",
        )
        summary.write_text(text, encoding="utf-8")

    base_rows = max(0, sum(1 for _ in (exp / "e4_t1_t4_controller_base_metrics.csv").open(encoding="utf-8")) - 1)
    stress_rows = max(0, sum(1 for _ in (exp / "e4_t1_t4_edit_strength_metrics.csv").open(encoding="utf-8")) - 1)
    traj_rows = max(0, sum(1 for _ in (out / "e4_controller_trajectory_stats.csv").open(encoding="utf-8")) - 1)
    complete = out / "e4_controller_ablation_t1_t4_complete_2026-06-09.md"
    complete.write_text(
        "\n".join(
            [
                "# E4 T1-T4 Controller Ablation Completion",
                "",
                "Date: 2026-06-09",
                "",
                "Scope: Phase2 T1-T4, 12 tasks.",
                f"Base fixed-vs-feedback metric rows: {base_rows}/72",
                f"Edit-strength stress metric rows: {stress_rows}/144",
                f"Controller trajectory rows: {traj_rows}/72",
                "",
                "Artifacts:",
                f"- `{exp / 'e4_t1_t4_controller_base_metrics.csv'}`",
                f"- `{exp / 'e4_t1_t4_edit_strength_metrics.csv'}`",
                f"- `{out / 'e4_controller_ablation_summary.md'}`",
                f"- `{out / 'e4_figure5_edit_strength_pareto.png'}`",
                f"- `{out / 'e4_controller_trajectory_tshirt_star_seed10.png'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {complete}")
    return result if base_rows == 72 and stress_rows == 144 and traj_rows == 72 else 1


if __name__ == "__main__":
    raise SystemExit(main())
