# Next Steps Completion Audit

Date: 2026-05-08

Objective: verify implementation and experiment readiness against
`task/rf_h_edit_next_steps.md`.

## Current Verified State

| Requirement | Artifact / evidence | Status |
| --- | --- | --- |
| Use `xdot_t = v_src + u_rec + u_edit` as the implementation convention. | `sd3_hrec.py` uses `v_total = v_base + v_rec + v_edit_total`; summary docs describe the source-conditioned RF velocity plus reconstruction and editing corrections. | Done |
| Keep reproducible run records with `result.png`, `stats.json`, `metadata.json`, `command.txt`, and masks. | `scripts/run_main_table.sh`; `scripts/audit_experiment_records.py --outputs-dir outputs/main_matrix` reports complete existing run records with no incomplete detected. The paper matrix subset is separately verified below. | Done |
| Separate fixed paper task matrix from debug cases. | `experiments/main_matrix.md`; `scripts/evaluate_paper_metrics.py` defaults to T1-T4 task names so older `red_chair_blue` exploratory runs do not enter `experiments/main_metrics.csv`. | Done |
| Verify planned main-matrix coverage, not just existing run completeness. | `scripts/audit_main_matrix_coverage.py --outputs-dir outputs/main_matrix` reports 60 planned, 60 complete, 0 missing/incomplete for T1-T4 x M0-M4 x seeds 10, 11, 12. | Done |
| Run fixed matrix for T1-T4 x M0-M4 x seeds 10, 11, 12. | `outputs/main_matrix/{cat_crown,backpack_blue,yellow_car_blue,rabbit_sunglasses}/{base_only,direct_target,anchor_only,decoupled_rec,full}/seed_{10,11,12}`. | Done |
| Record runtime and peak GPU memory for paper rows. | All 60 T1-T4 main-matrix metadata files contain `runtime_seconds` and `peak_gpu_memory_gb`. | Done |
| Add paper-grade metric evaluator. | `scripts/evaluate_paper_metrics.py` writes `experiments/main_metrics.csv` and `experiments/main_metrics.json`; current output is 60 rows, 60 complete. Includes preservation/drift proxies, trajectory summaries, runtime, peak memory, CLIP target/source scores, DINO source similarity, and manual failure annotations. | Done |
| Fix evaluator/audit run discovery. | Evaluator now only treats three-level `task/method/seed_*` directories as runs and can filter task names, method names, and seeds; tests cover nested support-artifact exclusion and all filters. | Done |
| Add trajectory-level statistics summary. | `scripts/summarize_stats.py` and metric evaluator summarize rec/edit norms, cosine fields, mask area, beta schedule, and related step statistics. | Done |
| Add SAM-supported relation-aware support for local edits. | `scripts/make_semantic_mask.py`; crown/headwear and side-profile support masks are generated in main runs. | Done |
| Add surface-reference mode for color tasks. | `scripts/run_main_table.sh` generates backpack surface references and yellow-car vehicle-paint references for full-method runs. | Done |
| Add experimental SD3 source-reference Q/K/V injection. | `sd3_hrec.py`, `run_edit_sd3.py`, and `scripts/run_main_table.sh` expose the source Q/K/V injection path. `outputs/main_matrix/*/full_source_v_inject/seed_10` validates source V injection on T1-T3. | Implemented and smoke-validated |
| Add one-command runners. | `scripts/run_main_table.sh`, `scripts/run_missing_main_matrix.sh`, `scripts/run_ablation_table.sh`, `scripts/make_paper_figures.sh`. | Done |
| Add paper figures. | `outputs/paper_figures/*_seed_{10,11,12}.png`; 12 T1-T4 comparison grids generated. `outputs/paper_figures/*_ablation_seed_10.png`; 3 ablation grids generated. | Done |
| Add environment dependency file. | `requirements.txt` pinned from the active environment. | Done |
| Add manuscript package skeleton. | `paper/outline.md`, `paper/draft.md`, `paper/results.md`, `paper/references.bib`, `paper/figures.md`, `paper/tables.md`, `paper/limitations.md`. Results, figures, tables, and limitations are populated with current metrics and artifacts. | Draft package done |
| Add manual failure labels. | `experiments/failure_flags.json` has 60 reviewed main-matrix entries. `experiments/main_metrics.csv` has failure notes for all 60 rows and non-empty failure flags for 57 rows. | Done |
| Add external baseline parity results. | `experiments/baseline_parity.md` records that current FireFlow/ReFlex/other artifacts are exploratory and not matched to T1-T4 x seed 10/11/12. No runnable matched external-baseline script exists in the project, so these are explicitly excluded from paper tables. | Documented unavailable |
| Populate ablation table, including source Q/K/V validation. | T1-T3 seed-10 ablations are complete: 18 ablation runs plus 3 reference `full` rows in `experiments/ablation_metrics.csv`; summary in `experiments/ablation_summary.md`. | Seed-10 done |
| Fill manuscript with final claims, citations, tables, and limitations. | Draft is still a careful skeleton and should not be treated as a complete paper. | Not done |

## Verification Commands Run

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/audit_main_matrix_coverage.py \
  --outputs-dir outputs/main_matrix \
  --json-output experiments/main_matrix_coverage_audit.json
```

Result:

```text
planned: 60
complete: 60
missing_or_incomplete: 0
```

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/audit_experiment_records.py \
  --outputs-dir outputs/main_matrix \
  --json-output experiments/main_matrix_record_audit.json
```

Result:

```text
runs: 82
complete: 82
incomplete: 0
```

The 82 existing records include 4 older `red_chair_blue` exploratory runs and
18 T1-T3 seed-10 ablation runs. The paper metric table filters to T1-T4,
M0-M4, seeds 10/11/12.

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir outputs/main_matrix \
  --csv-output experiments/main_metrics.csv \
  --json-output experiments/main_metrics.json \
  --clip-model openai/clip-vit-base-patch32 \
  --dino-model facebook/dinov2-small \
  --allow-download
```

Result:

```text
runs: 60
complete: 60
incomplete: 0
```

Failure annotation coverage after metric regeneration:

```text
rows 60
with_failure_flag 57
with_note 60
flags: background_drift=18, color_miss=18, semantic_miss=15, hybrid_object=3, localization_error=3, success/blank=3
```

Semantic/perceptual metric coverage:

```text
clip_source_score 60/60
clip_target_score 60/60
clip_target_minus_source 60/60
dino_source_similarity 60/60
```

Matched ablation metrics were generated with:

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/evaluate_paper_metrics.py \
  --outputs-dir outputs/main_matrix \
  --task-names "cat_crown backpack_blue yellow_car_blue" \
  --method-names "full full_full_ref full_no_rec full_no_traj full_attention_velocity full_semantic_velocity full_source_v_inject" \
  --seeds 10 \
  --csv-output experiments/ablation_metrics.csv \
  --json-output experiments/ablation_metrics.json
```

Result:

```text
runs: 21
complete: 21
incomplete: 0
```

Runtime and peak-memory check:

```text
missing runtime/peak 0
```

```bash
TASKS="cat_crown backpack_blue yellow_car_blue rabbit_sunglasses" \
SEEDS="10 11 12" \
bash scripts/make_paper_figures.sh
```

Result: 12 comparison grids under `outputs/paper_figures/`.

Additional ablation grids:

```text
outputs/paper_figures/cat_crown_ablation_seed_10.png
outputs/paper_figures/backpack_blue_ablation_seed_10.png
outputs/paper_figures/yellow_car_blue_ablation_seed_10.png
```

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python -m pytest \
  tests/test_mask_and_box_helpers.py \
  tests/test_linear_path_velocity_conversion.py \
  tests/test_evaluate_paper_metrics.py
```

Result:

```text
45 passed, 2 warnings
```

## Metric Snapshot

Current uncalibrated averages over the 60 paper rows:

| Method | Rows | Outside-mask L1 | Inside-mask L1 | Luma SSIM | Runtime s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_only | 12 | 0.0864 | 0.0789 | 0.8520 | 23.80 | 12.23 |
| direct_target | 12 | 0.1201 | 0.1312 | 0.6385 | 25.17 | 12.23 |
| anchor_only | 12 | 0.1199 | 0.1313 | 0.6371 | 27.29 | 12.23 |
| decoupled_rec | 12 | 0.1040 | 0.1102 | 0.7109 | 27.53 | 12.23 |
| full | 12 | 0.0590 | 0.1182 | 0.8745 | 54.98 | 13.73 |

Current CLIP/DINO averages over the 60 paper rows:

| Method | CLIP target-source | DINO source sim |
| --- | ---: | ---: |
| base_only | -0.0373 | 0.5834 |
| direct_target | 0.0177 | 0.4793 |
| anchor_only | 0.0199 | 0.4656 |
| decoupled_rec | 0.0189 | 0.5498 |
| full | 0.0058 | 0.8738 |

These are proxy metrics and should be interpreted with the generated figures
and failure labels. They do not yet replace semantic edit-success scoring.

Current ablation snapshot over T1-T3 seed 10:

| Method | Rows | Outside-mask L1 | Inside-mask L1 | Luma SSIM | Blue proxy | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 3 | 0.0650 | 0.1295 | 0.8716 | 0.2671 | 54.28 |
| full_full_ref | 3 | 0.0614 | 0.1266 | 0.8956 | 0.2559 | 54.41 |
| full_no_rec | 3 | 0.0623 | 0.1502 | 0.8932 | 0.2512 | 50.69 |
| full_no_traj | 3 | 0.0712 | 0.1386 | 0.8852 | 0.2611 | 54.70 |
| full_attention_velocity | 3 | 0.0834 | 0.0902 | 0.7877 | 0.1029 | 52.07 |
| full_semantic_velocity | 3 | 0.0613 | 0.1283 | 0.8948 | 0.2600 | 51.65 |
| full_source_v_inject | 3 | 0.0579 | 0.1134 | 0.9074 | 0.2713 | 53.68 |

The ablation metric file also has CLIP/DINO fields populated for all 21 rows.
On this seed-10 slice, `full_source_v_inject` has the strongest DINO source
similarity (0.9292) and lowest outside-mask L1 (0.0579), but a negative CLIP
target-source delta (-0.0078), so it remains a preservation-biased ablation
rather than a stronger edit method.

Visual review of the ablation grids shows that attention support is visibly
worse on T1/T2, source V injection improves preservation proxies but does not
solve weak T3 color editing, and T3 should remain a failure/limitation case.

## Remaining Work Before Claiming Completion

1. Expand ablations beyond seed 10 if statistical evidence is required.
2. LPIPS/VLM scoring remains absent; add only if needed for the target venue.
3. Convert `paper/draft.md` into a camera-ready manuscript if a submission
   venue is chosen; current package is a results-backed draft, not final
   prose.
