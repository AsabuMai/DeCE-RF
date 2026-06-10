# E4 Controller Ablation Completion

Date: 2026-06-04

E4 compares fixed DeCE displacement against DeCE-RF feedback-updated controller behavior on the three E4 tasks: cat_crown, tshirt_star, and pillow_vertical_fabric_strip.

## Completion Audit

- Base fixed-vs-feedback rows: 18 / 18
- Edit-strength stress metric rows: 36 / 36
- Base summary rows: 2 / 2
- Stress summary rows: 12 / 12
- Controller trajectory rows: 18 / 18

## Primary Artifacts

- experiments/support_v3_2026-06-02/e4_controller_base_metrics.csv
- experiments/support_v3_2026-06-02/e4_edit_strength_metrics.csv
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_base_summary.csv
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_edit_strength_summary.csv
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_stats.csv
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_summary.csv
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_figure5_edit_strength_pareto.png
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_trajectory_tshirt_star_seed10.png
- experiments/support_v3_2026-06-02/e4_controller_ablation/e4_controller_ablation_summary.md

Interpretation policy: E4 is a contextual controller ablation. It supports a robustness/stabilization claim for feedback-updated clean-estimate control, not the headline E2.2 algorithmic claim by itself.
