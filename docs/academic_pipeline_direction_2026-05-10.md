# Academic Pipeline Direction

Date: 2026-05-10

## Pipeline Stage Diagnosis

Current stage: **Stage 1 RESEARCH / method validation**, not Stage 2 WRITE.

The repository already contains draft-like paper materials, experiment scripts,
metrics, and visual audits, but the central research claim is not stable enough
for the writing and integrity-review stages. The correct next step is not to
polish the existing draft. It is to narrow the claim and run a decisive
go/no-go validation matrix.

## Current Project Position

The original broad framing:

```text
general RF/ODE image editing via h-Edit-style reconstruction/edit decoupling
```

is no longer competitive enough. Recent RF editing papers already cover
inversion-free editing, faithful inversion, trajectory interpolation, adaptive
masking, feature/attention injection, and geometric/proximal formulations.

The project should be repositioned as:

```text
clean-estimate-space RF control for localized image editing
```

Core contribution candidate:

```text
x0_hat = x_t - t v_theta(x_t, t)
u = -delta_x0 / t
```

with an automatic local support/reference interface:

```text
prompt diff -> edit type -> anchor/support -> local reference prior
-> clean-estimate correction -> RF velocity update
```

## Recommended Paper Scope

Use a narrow localized-editing claim:

> A clean-estimate-space control interface for localized rectified-flow editing
> can improve source preservation for local semantic insertion, decal editing,
> and object removal when paired with automatic support/reference construction.

Avoid these claims:

- General-purpose RF image editing.
- State-of-the-art RF editing.
- Robust pure recoloring.
- Strict object replacement success.

## Main Qualitative Set

Keep these four tasks as the main set:

| Task | Role | Current status |
| --- | --- | --- |
| `cat_crown` | local accessory insertion | strongest success |
| `dog_sunglasses` | accessory insertion with automatic support/reference | weak success, usable if framed honestly |
| `mug_heart` | local decal/symbol edit | success |
| `backpack_remove_toy_charm` | localized semantic removal | success |

Use these only as supplemental or limitation cases:

| Task | Use |
| --- | --- |
| `backpack_replace_patch_blue` | supplemental attribute-local replacement |
| `dog_replace_tennis_ball_star` | replacement candidate, not yet validated |
| `red_chair_blue` | limitation: deterministic recolor is stronger |
| `tshirt_star` | exclude unless star becomes sharp |
| `tote_leaf` | exclude unless reference shape is fixed |
| old `backpack_blue`, `yellow_car_blue`, `rabbit_sunglasses` | failure analysis |

## Required Go/No-Go Matrix

Before writing the paper, run one compact matrix:

```text
Tasks:   cat_crown dog_sunglasses mug_heart backpack_remove_toy_charm
Methods: base_only direct_target full_no_ref full_no_rec full_no_traj full
Seeds:   10 11 12
```

Minimum success gate:

1. `full` succeeds visually on at least 9/12 task-seed pairs.
2. `full` has lower outside-mask drift than `direct_target` on most rows.
3. `full_no_ref`, `full_no_rec`, and `full_no_traj` each expose a meaningful
   failure mode or measurable degradation.
4. At least one matched external baseline is run on the same images/prompts.
5. The evaluation includes functional correctness, not only CLIP/DINO.

If the gate fails, the project should become a workshop/technical report and
failure-analysis artifact rather than a conference paper.

## External Baseline Priority

Minimum:

1. FlowEdit
2. FireFlow or RF-Solver/RF-Edit

If feasible:

3. ReFlex
4. SteerFlow

If a baseline cannot be run because of memory, framework mismatch, or missing
code, record it explicitly in `experiments/baseline_parity.md` with the exact
reason. Do not use unmatched exploratory examples as paper baselines.

## Evaluation Upgrade

The current metric files are useful but insufficient. Add:

- visual audit CSV: task/method/seed/success/failure_type/note;
- outside-mask preservation: L1, SSIM, DINO, optionally LPIPS;
- edit-region functional correctness: VLM or question-based scoring;
- per-task binary checks, for example:
  - "Is there a crown on the cat's head?"
  - "Is the dog wearing sunglasses?"
  - "Is there a red heart on the mug?"
  - "Was the yellow toy charm removed while backpack details remain?"

GIE-Bench-style functional correctness plus preservation is a better model than
CLIP-only scoring.

## Next Implementation Steps

1. Freeze the main task set in `scripts/run_pretty_matrix.sh`.
2. Add method variants:
   - `full_no_ref`
   - `full_no_rec`
   - `full_no_traj`
3. Add a visual-audit CSV or JSON schema.
4. Generate review grids for every task across methods/seeds.
5. Run the go/no-go matrix.
6. Try FlowEdit matched baseline.
7. Decide:
   - if gate passes: enter Stage 2 WRITE with a narrowed paper;
   - if gate fails: stop paper push and produce a workshop-style report.

## Pipeline Recommendation

Do **not** proceed to Stage 2 WRITE yet.

Recommended current workflow:

```text
Stage 1 RESEARCH补强
  -> method repositioning
  -> compact validation matrix
  -> matched baseline attempt
  -> go/no-go checkpoint
```

Only after the go/no-go checkpoint passes should the project enter:

```text
Stage 2 WRITE
  -> Stage 2.5 INTEGRITY
  -> Stage 3 REVIEW
```

