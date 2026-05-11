# Table Plan

## Main Comparison

Rows: T1-T4 x M0-M4 x seeds 10, 11, 12.

Fields:

- edit success
- source preservation
- outside-mask drift
- runtime
- failure flag

Generate draft metrics with:

```bash
scripts/evaluate_paper_metrics.py
```

Current artifact:

```text
experiments/main_metrics.csv
```

Current status: 60 rows, 60 complete for T1-T4 x M0-M4 x seeds 10, 11,
12. Runtime and peak GPU memory are populated for all rows.

Manual visual failure labels are read from:

```text
experiments/failure_flags.json
```

Current status: all 60 main-matrix rows have a visual-review note; 57 rows have
non-empty failure flags and the three blank flags are the cat-crown full-method
rows judged visually successful enough for the current table.

Before treating the table as complete, verify planned coverage with:

```bash
scripts/audit_main_matrix_coverage.py
```

The generic record audit only checks existing run directories. It does not
prove that every planned task/method/seed cell exists.

## Ablation Table

Rows: T1-T3 x full-method ablations.

Required ablations:

- no reconstruction correction
- no trajectory preserve
- attention mask versus semantic/SAM support
- source-reference Q/K/V injection

The source-reference Q/K/V rows should not be treated as the main method unless
their outputs and metrics show a consistent benefit under the same T1-T3 task
conditions.

Current artifact:

```text
experiments/ablation_metrics.csv
experiments/ablation_summary.md
```

Current status: 21 rows, 21 complete for T1-T3 seed 10. This includes the
reference `full` row plus six full-method variants. The source V injection row
has favorable preservation proxies on this seed-10 slice, but the qualitative
figures still show weak T3 color editing; keep it as an experimental ablation,
not as the main method.

## External Baseline Table

Artifacts:

```text
experiments/baseline_parity_manifest.csv
experiments/baseline_summary.csv
experiments/baseline_summary.md
experiments/baseline_visual_scores_seed10_12.csv
experiments/baseline_visual_score_template.csv
paper/stage2_5_integrity_precheck.md
```

Rows:

```text
6 external baselines x 4 tasks x seeds 10,11,12 = 72 manifest rows
```

Status:

```text
FlowEdit:       12 complete
SplitFlow:      12 complete
FireFlow:       12 complete
RF-Solver-Edit: 12 complete
ReFlex:         12 failed with resource/code compatibility reason
SteerFlow:      12 failed because no public runnable code was found
```

Recommended paper tables:

- Baseline availability/status table: report runnable status, complete rows,
  failed rows, model family, seed-matching caveat, and notes.
- Internal visual-score table: report edit success, preservation, locality,
  artifact, and overall on a 1-5 scale. Label this explicitly as an internal
  visual audit, not a user study.

Recommended qualitative figures:

```text
outputs/paper_figures/baseline_backpack_remove_toy_charm_seed10.png
outputs/paper_figures/baseline_dog_sunglasses_seed10.png
```

Use `backpack_remove_toy_charm` as the main preservation success case and
`dog_sunglasses` as the edit-strength versus preservation tradeoff case.
