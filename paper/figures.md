# Figure Plan

Use only complete runs with `result.png`, `stats.json`, `metadata.json`, and
`command.txt`.

## Main Figures

- T1 cat crown: usable as the current positive qualitative result. Prefer
  seed 11 or seed 12 over seed 10 because the crown placement/scale is cleaner.
- T2 backpack blue: not usable as a success figure. The full method creates a
  large blue object beside a still-red backpack, so this is a hybrid-object
  failure unless a new run fixes the edit.
- T3 yellow car blue: not usable as a success figure. The full method preserves
  structure but fails to turn the car body blue.
- T4 rabbit sunglasses: not usable as a success figure. The sunglasses-like
  edit is small and poorly localized on the side-profile rabbit; keep only as a
  robustness/failure example.

Generate draft panels with:

```bash
scripts/make_paper_figures.sh
```

Current generated panels:

```text
outputs/paper_figures/cat_crown_seed_10.png
outputs/paper_figures/cat_crown_seed_11.png
outputs/paper_figures/cat_crown_seed_12.png
outputs/paper_figures/backpack_blue_seed_10.png
outputs/paper_figures/backpack_blue_seed_11.png
outputs/paper_figures/backpack_blue_seed_12.png
outputs/paper_figures/yellow_car_blue_seed_10.png
outputs/paper_figures/yellow_car_blue_seed_11.png
outputs/paper_figures/yellow_car_blue_seed_12.png
outputs/paper_figures/rabbit_sunglasses_seed_10.png
outputs/paper_figures/rabbit_sunglasses_seed_11.png
outputs/paper_figures/rabbit_sunglasses_seed_12.png
```

## Paper-Use Decision

Current success-quality figure candidates:

```text
outputs/paper_figures/cat_crown_seed_11.png
outputs/paper_figures/cat_crown_seed_12.png
```

Current failure/limitation figures only:

```text
outputs/paper_figures/backpack_blue_seed_10.png
outputs/paper_figures/backpack_blue_seed_11.png
outputs/paper_figures/backpack_blue_seed_12.png
outputs/paper_figures/yellow_car_blue_seed_10.png
outputs/paper_figures/yellow_car_blue_seed_11.png
outputs/paper_figures/yellow_car_blue_seed_12.png
outputs/paper_figures/rabbit_sunglasses_seed_10.png
outputs/paper_figures/rabbit_sunglasses_seed_11.png
outputs/paper_figures/rabbit_sunglasses_seed_12.png
```

Do not frame T2-T4 as successful paper results unless new outputs visibly fix
the listed failure modes.

## Ablation Figures

```text
outputs/paper_figures/cat_crown_ablation_seed_10.png
outputs/paper_figures/backpack_blue_ablation_seed_10.png
outputs/paper_figures/yellow_car_blue_ablation_seed_10.png
```
