# Gated Clean-Delta Cross-Seed Diagnosis

Scope: six removal tasks, seeds `10/11/12`, methods `M0 default`, `M1 ungated clean_delta`, and `M2 reliability-gated clean_delta`.

Outputs:

- Cross-seed grid: `experiments/support_v3_2026-05-11/visual_gates/removal_completion_clean_delta_gated_seeds10_11_12_grid.png`.
- Reliability-vs-gain CSV: `experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_seed10_11_12.csv`.
- Reliability-vs-gain scatter: `experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_scatter.png`.
- Gate protocol: `experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_seeds10_11_12_protocol.json`.

Fixed task-wise gate:

| Task | R | Gate | Effective scale |
| --- | ---: | ---: | ---: |
| laptop_remove_sticker | 0.75 | 1.0 | 0.550 |
| fridge_remove_yellow_magnet | 0.21 | 0.5 | 0.275 |
| fridge_remove_peach_magnet | 0.23 | 0.5 | 0.275 |
| whiteboard_remove_yellow_letter | 0.21 | 0.5 | 0.275 |
| backpack_remove_toy_charm | 0.16 | 0.0 | 0.000 |
| dog_remove_tennis_ball | 0.09 | 0.0 | 0.000 |

Aggregate mask-MAE diagnostics over seeds `10/11/12`:

| Task | Gain M1 to Telea | GatedGain M2 to Telea | M2-M0 | M2-M1 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| laptop_remove_sticker | 0.0876 | 0.0871 | 0.0930 | 0.0018 | Gated preserves clean_delta behavior. |
| fridge_remove_yellow_magnet | 0.0689 | 0.0573 | 0.0675 | 0.0140 | Gated is conservative, between default and ungated. |
| fridge_remove_peach_magnet | 0.1425 | 0.1134 | 0.1194 | 0.0334 | Gated is conservative, between default and ungated. |
| whiteboard_remove_yellow_letter | 0.0891 | 0.0716 | 0.0776 | 0.0210 | Gated is conservative, between default and ungated. |
| backpack_remove_toy_charm | 0.0776 | 0.0012 | 0.0123 | 0.0814 | Gated falls back near default and avoids bad prior pursuit. |
| dog_remove_tennis_ball | 0.0917 | -0.0003 | 0.0128 | 0.0987 | Gated falls back near default and avoids bad prior pursuit. |

Interpretation:

- Laptop is the high-reliability positive case: `M2 ~= M1` across all three seeds.
- Backpack and dog are low-reliability hard cases: `M2 ~= M0`, while `M1` moves toward the completion prior.
- Fridge and whiteboard are medium-low reliability cases: `M2` is consistently a conservative injection rather than full prior pursuit.
- Raw `Gain = MAE(default, Telea) - MAE(clean_delta, Telea)` is not by itself a safety metric: low-R tasks can move closer to a bad Telea prior. The useful diagnostic is reliability-aware gain plus distance from default. The cross-seed scatter therefore shows weak `R` vs ungated gain but a clearer positive relationship for gated gain.

Paper-ready statement: reliability score provides diagnostic evidence for when completion clean-delta should be trusted, and a simple fixed gate preserves high-R gains while suppressing low-R prior chasing.
