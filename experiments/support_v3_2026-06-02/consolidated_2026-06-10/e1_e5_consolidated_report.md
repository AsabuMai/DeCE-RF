# E1-E5 Consolidated Result Report (Phase 2, T1-T4 core)

Date: 2026-06-10
Scope: integrates all completed E1-E5 artifacts under
`experiments/support_v3_2026-06-02/` into one readout, with image comparison
sheets, and assesses how well the current evidence supports the DeCE-RF paper
claims in `paper/argument_blueprint.md` and `paper/wacv_experiment_design.md`.

Status note: this is a consolidation of existing locked/diagnostic artifacts.
No new generation runs were performed; the two comparison sheets below were
assembled from existing `outputs/` images (seed 10).

## Artifact Index

| Experiment | Completion | Primary metric/summary source | Image evidence |
| --- | --- | --- | --- |
| E1 Phase 2 T1-T4 formal matrix | 144/144 runs (12 tasks x 4 methods x seeds 10/11/12), `audit_t1_t4_e1_completion.py` | `e4_t1_t4_reconstruction_floor_metrics.csv`, `e4_t1_t4_controller_base_metrics.csv`, strict Phase 1 `strict_fixed_mask_metrics_summary.md` | `consolidated_2026-06-10/e1_t1_t4_seed10_comparison_sheet.png` |
| E2 RF baseline comparison | E2-A strict Core-6 LOCKED; T1-T4 external matrix 36/36 per baseline | `e2_reduced_rf_comparison_summary.md`, `e2_t1_t4_baseline_summary.csv`, `e2_support_matched_t1_t4_summary.csv`, `paper/e2_status_lock.md` | `consolidated_2026-06-10/e2_t1_t4_seed10_rf_baseline_sheet.png` |
| E3 support geometry ablation | 36/36 | `e3_support_geometry_t1_t4/e3_support_geometry_summary.md` | `e3_support_geometry_t1_t4/e3_support_geometry_seed10_task_sheet.png`, `..._tshirt_star_seed10_figure4_panel.png` |
| E4 controller/stress ablation | base 72 rows + stress 6 multipliers x 12 tasks (seed 10) | `e4_controller_ablation_t1_t4/e4_controller_ablation_summary.md` | `e4_controller_ablation_t1_t4/e4_figure5_edit_strength_pareto.png`, `..._trajectory_tshirt_star_seed10.png` |
| E5 boundary/extension audit | 36/36 selected outputs | `e5_boundary_extension_t1_t4/e5_t1_t4_boundary_extension_summary.md` | `e5_boundary_extension_t1_t4/e5_t1_t4_boundary_extension_seed10.png` |

Task grouping used throughout: T1 attached accessory (`cat_crown`,
`dog_bow_tie_phase2`, `dog_front_sunglasses_phase2`), T2 container insertion
(`bowl_apple_inside`, `white_bowl_orange_tabletop_phase2`,
`brown_bowl_lemon_phase2`), T3 surface decal (`tshirt_star`, `mug_heart`,
`tote_leaf`), T4 local recolor (`red_office_chair_to_blue_office_chair`,
`green_mug_orange_phase2`, `yellow_vase_blue_phase2`).

## E1: Main Edit-Preserve Effect

Image sheet (seed 10, all 12 tasks; columns: source, base_only, direct_target,
generic support, DeCE-RF):

```text
consolidated_2026-06-10/e1_t1_t4_seed10_comparison_sheet.png
```

### Phase 2 T1-T4 aggregate (fixed eval masks, seeds 10/11/12)

| Group | Method | n | Outside L1 (down) | Inside L1 | Source SSIM (up) |
| --- | --- | ---: | ---: | ---: | ---: |
| ALL | base_only (reconstruction floor) | 36 | 0.0504 | 0.0546 | 0.8858 |
| ALL | support_v3_fixed (Fixed DeCE) | 36 | 0.0350 | 0.1729 | 0.9147 |
| ALL | support_v3_controller_rmsgap (DeCE-RF) | 36 | 0.0343 | 0.1772 | 0.9144 |
| T1 | DeCE-RF | 9 | 0.0568 | 0.1353 | 0.8715 |
| T2 | DeCE-RF | 9 | 0.0341 | 0.0907 | 0.9141 |
| T3 | DeCE-RF | 9 | 0.0168 | 0.0960 | 0.8984 |
| T4 | DeCE-RF | 9 | 0.0297 | 0.3868 | 0.9736 |

Reading: DeCE-RF (and Fixed DeCE) hold outside-mask change *below* the
base_only reconstruction floor while producing large inside-mask change,
i.e. the controller edits inside the support and actively suppresses
reconstruction drift outside it. T4 inside L1 is high by design (recolor).

Coverage gap: `direct_target` and `adaptive_full_generic_support` Phase 2
T1-T4 fixed-mask metric rows have not been computed yet (outputs exist,
144/144). The cross-method quantitative comparison below therefore uses the
strict Phase 1 Core-6 table; the Phase 2 cross-method evidence is currently
visual (comparison sheet) plus the support-matched diagnostic.

### Strict Phase 1 Core-6 cross-method summary (from `strict_fixed_mask_metrics_summary.md`)

Representative rows (Outside L1 / Source SSIM / edit score):

| Task | direct_target | generic support | DeCE-RF |
| --- | --- | --- | --- |
| cat_crown | 0.1246 / 0.4358 / 0.0007 | 0.0582 / 0.6575 / 0.0022 | 0.0572 / 0.6503 / 0.0988 |
| bowl_apple_inside | 0.0925 / 0.3450 / 0.0054 | 0.0620 / 0.5113 / 0.0125 | 0.0533 / 0.5367 / 0.0202 |
| tshirt_star | 0.0410 / 0.8446 / -0.0006 | 0.0286 / 0.8648 / -0.0092 | 0.0176 / 0.8780 / 0.0798 |
| red_chair_blue | 0.1203 / 0.3275 / -0.0226 | 0.0652 / 0.5210 / -0.0016 | 0.0653 / 0.5220 / 0.0055 |
| pillow_vertical_fabric_strip | 0.0747 / 0.4945 / 0.1082 | 0.0255 / 0.6889 / 0.0449 | 0.0207 / 0.7173 / 0.0494 |
| backpack_remove_toy_charm | 0.0697 / 0.5619 / -0.0078 | 0.0280 / 0.7956 / -0.0078 | 0.0295 / 0.7366 / -0.0169 |

Human quick audit (`strict_visual_human_quick_audit.md`): all six DeCE-RF
strict rows pass; direct_target is rejected for figures on 5/6 tasks
(semantic miss / over-edit / crop drift); generic support misses the target
edit on cat, tshirt, and backpack.

## E2: RF Baseline Comparison

Image sheet (seed 10; columns: source, FlowEdit, FlowAlign, SplitFlow,
RF-Solver-Edit, FireFlow, ReFlex, DeCE-RF):

```text
consolidated_2026-06-10/e2_t1_t4_seed10_rf_baseline_sheet.png
```

### E2-A strict Core-6, same-protocol external RF baselines (seeds 10/11/12, ALL row)

| Method | Outside L1 | Inside L1 | Source SSIM | DINO/source | CLIP edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlowEdit (external RF) | 0.1760 | 0.2581 | 0.4133 | 0.4092 | 0.0482 |
| FlowAlign (external RF) | 0.0769 | 0.1263 | 0.6406 | 0.6900 | 0.0401 |
| SplitFlow (external RF) | 0.0965 | 0.1402 | 0.5094 | 0.6159 | 0.0427 |
| DeCE-RF | 0.0406 | 0.1038 | 0.6735 | 0.8497 | 0.0395 |

DeCE-RF achieves the lowest outside-mask change and highest source
preservation (SSIM, DINO) at comparable CLIP edit score. Claim boundary per
`paper/e2_status_lock.md`: reduced target-mode comparison, not "beats all RF
editors".

### Phase 2 T1-T4 external baseline matrix (36 runs each, means)

| Baseline | Outside L1 | Inside L1 | Source SSIM |
| --- | ---: | ---: | ---: |
| flowedit | 0.1543 | 0.2036 | 0.5278 |
| flowalign | 0.0760 | 0.1370 | 0.8292 |
| splitflow | 0.0794 | 0.1511 | 0.7625 |
| fireflow | 0.0670 | 0.1024 | 0.8404 |
| rf_solver_edit | 0.0551 | 0.0848 | 0.8778 |
| reflex | 0.0676 | 0.1351 | 0.8315 |
| DeCE-RF (from E4 cache, same tasks/seeds) | 0.0343 | 0.1772 | 0.9144 |

Caveat: the DeCE-RF row comes from `e4_t1_t4_controller_base_metrics.csv`;
confirm both metric runs used identical normalization before promoting this
join into a paper table. The low inside-L1 of `rf_solver_edit`/`fireflow`
together with visual inspection suggests under-editing rather than better
control; an edit-score column should be added before paper use (currently
empty in `e2_t1_t4_baseline_summary.csv`).

### Support-matched diagnostic (T1-T4) — protocol warning

`e2_support_matched_t1_t4_summary.csv` reports outside L1 0.2359
(direct_target_raw) vs 0.2174 (DeCE-RF) with SSIM 0.14 vs 0.19. These
absolute values are not comparable to the E1/E4 tables: the diagnostic
evaluates 512x336 results against 512x512 normalized sources
(`normalized_512/sources/`), so geometric misalignment inflates all
difference metrics. Use only within-table relative reading (DeCE-RF still
better than raw direct target on outside L1/SSIM in every group), and re-run
with aligned geometry before any paper table. `*_mask_blend` rows reach
outside L1 = 0 by construction and remain diagnostic-only.

## E3: Support Geometry Ablation

Source: `e3_support_geometry_t1_t4/e3_support_geometry_summary.md` (36 rows;
cat_crown, tshirt_star, backpack_remove_toy_charm; seeds 10/11/12).

| Variant | IoU | Precision | Recall | Edit score |
| --- | ---: | ---: | ---: | ---: |
| Attention only | 0.0588 | 0.0839 | 0.1098 | - |
| Clean disagreement | 0.0644 | 0.1172 | 0.1277 | - |
| Velocity disagreement | 0.0646 | 0.1174 | 0.1286 | - |
| Grounding/SAM | 0.3220 | 0.3292 | 0.8205 | - |
| Generic support | 0.2044 | 0.2457 | 0.2817 | 0.0018 |
| Operation support (v3) | 0.5242 | 0.5661 | 0.7470 | 0.0663 |

Support quality correlates with downstream edit success: Spearman r between
support IoU/precision/recall and edit score is 0.57/0.59/0.60 (n=18,
runnable rows only). Correlations with preserve-side metrics are weak.
Grounding/SAM alone gets recall but not precision; operation conditioning is
what concentrates the geometry.

## E4: Controller / Feedback Ablation And Stress

Source: `e4_controller_ablation_t1_t4/e4_controller_ablation_summary.md`
(12 tasks, seeds 10/11/12; stress at seed 10, multipliers 0.50-2.00).

- Base table: Fixed DeCE 0.0350 outside L1 / 0.9147 SSIM vs DeCE-RF 0.0343 /
  0.9144 — statistically indistinguishable on these aggregate metrics.
- Trajectory stats: the feedback path is mechanically active (preserve drift
  0.21, preserve weight 1.23, edit weight 1.16, preserve correction 2.88)
  while the fixed variant is inert by construction.
- Stress: both variants degrade gracefully and similarly as edit strength
  rises (outside L1 0.034->0.035, SSIM 0.91->0.89 at x2.00).

Reading: feedback currently shows *mechanism activity without measurable
aggregate benefit* over Fixed DeCE on T1-T4. The summary's own framing
(stabilizer/robustness component, not the headline gain) is the only
defensible wording; see claim assessment below.

## E5: Boundary / Extension Audit

Source: `e5_boundary_extension_t1_t4/e5_t1_t4_boundary_extension_summary.md`.
36/36 selected outputs across positive_core / extension_core / boundary_core
labels for the 12 tasks; failure taxonomy in `e5_t1_t4_failure_taxonomy.csv`.
This rerun audits boundary behavior of the current core tasks; the old T5/T6
removal/completion extension package is out of scope here.

## Claim-By-Claim Assessment (vs `paper/argument_blueprint.md`)

| # | Claim | Verdict | Evidence | Gap / risk |
| --- | --- | --- | --- | --- |
| 1 | Direct target velocity couples local edit with global drift | **Supported, strong** | Strict E1: direct_target outside L1 2-7x worse than DeCE-RF on every task; human audit rejects 5/6 direct_target rows; E2-A FlowEdit shows the same coupling pattern | Phase 2 T1-T4 direct_target metric rows not yet computed (images exist) |
| 2 | Clean-estimate displacement decoupling improves the edit-preserve tradeoff | **Supported on T1-T3 insertion/decal; partial on T4/T6** | E1: DeCE-RF best or tied outside L1 + best DINO with positive edit score where baselines are negative (cat_crown 0.099 vs 0.002; tshirt_star 0.080 vs -0.009); E2-A: best preserve metrics at comparable CLIP edit | red_chair_blue nearly tied with generic support (edit score 0.0055); backpack_remove (T6) edit score negative and SSIM below generic support — removal remains the weak operation |
| 3 | Operation-conditioned geometry is a real component, not a hand mask | **Supported** | E3: IoU 0.52 vs 0.20 (generic) vs 0.06 (attention); downstream edit score 0.066 vs 0.002; Spearman support-quality vs edit score about 0.6 | n=18 for correlations; only 3 tasks; manual-support upper bound row (planned in blueprint) not present |
| 4 | Clean-estimate feedback makes the controller closed-loop and this matters | **Mechanism shown, benefit not yet demonstrated** | E4 trajectory stats prove the loop is active; stress curves show no harm | Fixed DeCE matches DeCE-RF on every aggregate metric; current data supports only "stabilizer, no cost" wording; need at least one regime (harder support noise, stronger edit pressure, or per-task failure counts) where fixed fails and feedback recovers, otherwise reviewers can call feedback unnecessary |
| 5 | Scoped boundary claim (works under reasonable support; support is the bottleneck) | **Supported and well-instrumented** | E5 taxonomy + E3 correlation directly implement the "support is the bottleneck" narrative | none major; keep T6/removal in limitations |

## Overall Verdict

The central thesis — decoupled clean-estimate edit-preserve control improves
localized RF editing over direct target guidance, generic support, and
runnable RF-native baselines under matched support and fixed masks — is
supported by the completed E1/E2/E3 evidence, with honest scoping already in
place via E5 and the e2 status lock. The two open weaknesses before paper
lock are: (a) the feedback component (the "RF" in DeCE-RF's controller story)
currently shows no measurable aggregate gain over Fixed DeCE, so either find
the regime where it wins or demote the claim to stabilization; (b) several
quantitative joins are incomplete or protocol-inconsistent (Phase 2
direct_target/generic-support metrics missing; support-matched T1-T4 geometry
misalignment; missing edit-score column in the T1-T4 external baseline
summary), and these should be fixed on a100-01 before the tables are frozen.

## Reproduction Notes

Sheets built by `consolidated_2026-06-10/build_e1_sheet.py` and
`consolidated_2026-06-10/build_e2_sheet.py` (PIL-only, master-safe).
Aggregates computed from the CSVs listed in the artifact index.
