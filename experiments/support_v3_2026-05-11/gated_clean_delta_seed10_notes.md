# Gated Clean-Delta, Seed 10

Experiment: reliability-aware completion control for six removal tasks.

Methods:

- `M0 default`: `support_v3_controller_rmsgap`.
- `M1 ungated clean_delta`: `support_v3_controller_rmsgap_completion_clean_delta`, scale `0.55`.
- `M2 gated clean_delta`: `support_v3_controller_rmsgap_completion_clean_delta_gated`.

Gate:

- Reliability source: `experiments/support_v3_2026-05-11/prior_reliability/completion_prior_reliability_seed10.csv`.
- Tiered rule: `R >= 0.50 -> gate=1.0`; `0.18 <= R < 0.50 -> gate=0.5`; `R < 0.18 -> gate=0.0`.
- Effective clean-delta scale: `0.55 * gate`.
- Gate-off tasks reuse the default source command rather than running a zero-scale clean_delta variant with altered text-guidance settings.

| Task | R | Gate | Scale | First-pass read |
| --- | ---: | ---: | ---: | --- |
| laptop_remove_sticker | 0.75 | 1.0 | 0.550 | Gated matches ungated clean_delta; the sticker is removed and the surface is smooth. |
| fridge_remove_yellow_magnet | 0.21 | 0.5 | 0.275 | Gated weakens prior pursuit; less aggressive than ungated, useful as a risk-control setting. |
| fridge_remove_peach_magnet | 0.23 | 0.5 | 0.275 | Gated weakens prior pursuit; behavior remains close to the surface subset rather than hard failures. |
| whiteboard_remove_yellow_letter | 0.21 | 0.5 | 0.275 | Gated keeps the partial removal behavior but reduces the strength of the completion pull. |
| backpack_remove_toy_charm | 0.16 | 0.0 | 0.000 | Gated falls back to default and avoids chasing the implausible Telea prior. |
| dog_remove_tennis_ball | 0.09 | 0.0 | 0.000 | Gated falls back to default and avoids the occluded-mouth prior. |

Sanity checks:

- Low-R gate-off commands show `|v_cdelta|=0.0000` in logs.
- Laptop gated is numerically very close to ungated clean_delta (`M1` vs `M2` mean absolute pixel difference `0.056`).
- Dog gated is nearly identical to default (`M0` vs `M2` mean absolute pixel difference `0.146`); backpack is visually default-like, with small deterministic-run drift (`0.847` mean absolute pixel difference).

Takeaway: this seed supports the intended paper claim at the control level. The gate preserves the high-reliability laptop improvement, attenuates medium-low surface priors, and disables unreliable hard priors for backpack/dog.
