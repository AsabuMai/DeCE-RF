# Main Experiment Matrix

This is the minimum fixed matrix for a submission-oriented RF h-Edit paper. Do
not add qualitative probes to the paper until this matrix is populated.

## Tasks

| Task ID | Edit type | Source prompt | Target prompt | Notes |
| --- | --- | --- | --- | --- |
| T1 | Headwear insertion | A photo of a cat sitting in grass. | A photo of the same cat sitting in the same grass, wearing a small golden crown on its head. | Main local accessory / relation-aware support task. |
| T2 | Color / attribute edit | A photo of a burgundy backpack sitting on rocks outdoors. | A photo of the same backpack sitting on the same rocks outdoors, with the backpack fabric changed to clean blue. | Main non-vehicle surface recolor task. |
| T3 | Color / attribute edit | A photo of a yellow car parked on a street. | A photo of the same car parked on the same street, with the car body changed to clean blue. | Vehicle surface recolor task with paint-mask support. |
| T4 | Side-profile accessory | A photo of a rabbit sitting outdoors in side profile. | A photo of the same rabbit sitting outdoors in side profile, wearing small black sunglasses. | Known robustness / failure case; keep visible. |
| T5 | Color / attribute edit | A photo of a red chair indoors. | A photo of the same chair indoors, with the chair changed to clean blue. | Simple licensed color-edit sanity check. |
| T6 | Headwear insertion | A photo of a dog sitting. | A photo of the same dog sitting, wearing a small golden crown on its head. | Backup local accessory task if cat crown is unstable. |

## Method Variants

| Variant ID | Name | Purpose |
| --- | --- | --- |
| M0 | base only | Tests source-conditioned reconstruction without edit pressure |
| M1 | direct target | Strong edit baseline; expected to drift |
| M2 | anchor only | Tests clean-space target anchor without reconstruction branch |
| M3 | decoupled reconstruction | Tests whether preservation branch improves source faithfulness |
| M4 | full method | Uses the current best mask/support and edit dynamics |

## Required Outputs Per Run

Each `Task ID x Variant ID x seed` run must create:

```text
outputs/main_matrix/<task_name>/<method_name>/seed_<seed>/
  result.png
  stats.json
  metadata.json
  command.txt
  masks/
```

## Seeds

Use at least:

```text
10, 11, 12
```

For a full submission, expand to 5 or 10 seeds after the small matrix is stable.

## Paper Tables

| Table | Rows | Required fields |
| --- | --- | --- |
| Main comparison | T1-T4 x M0-M4 | edit success, preservation, mask-outside drift, runtime |
| Ablation | T1-T3 x M4 variants | no trajectory preserve, no structure mask, broad mask, changed-token mask |
| Failure analysis | T4 variants | visible edit success, source drift, failure mode note |

The current runner source of truth is `scripts/run_main_table.sh`. The
publication-safe task set is backed by `experiments/image_manifest.md`; older
panda and bus prompts remain debug-only continuity cases.

## Acceptance Criteria

The matrix is ready for writing only when:

1. Every planned run passes `scripts/audit_experiment_records.py`.
2. Every paper table cell has a metric value or an explicit missing-value note.
3. Every qualitative figure has a matching command and metadata file.
4. The failure case is described as a limitation, not hidden.
