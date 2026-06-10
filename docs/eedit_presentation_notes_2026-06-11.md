# EEdit-Inspired Experiment Presentation Notes (2026-06-11)

Source: user analysis of EEdit's table/metric organization applied to DeCE-RF.
Decision: borrow presentation discipline, do NOT borrow the experiment center
(DeCE-RF stays an edit-preserve control paper, not an efficiency paper, not a
wide-task benchmark).

## The One-Line Lesson

Organize tables by the claim; every metric must answer a specific question,
not pad an appearance of thoroughness.

## Ten Points (condensed)

1. Group metrics by question, not by metric name. Table 1 header:
   `Method | Edit ↑ | Relation ↑ | Preserve ↓ | Locality ↑ | Artifact ↓ | Wins ↑`
   with group composition defined in the caption/supplement, never a flat
   CLIP/LPIPS/SSIM/L1/DINO row.
2. One table answers one question: Table 1 = main effect vs direct target /
   generic support; Table 2a = same-backbone SD3 RF rows; Table 2b =
   calibration + native preservation-aware context; Table 3 = support /
   displacement / feedback component attribution.
3. CLIP alone cannot prove localized edit success. Task-specific success
   checks (crown present AND on the cat's head; apple truly inside the bowl;
   star on the shirt surface with body contour intact; blue ratio up inside
   the chair mask only; charm gone with plausible backpack) are a core feature
   of the evaluation, not a supplement detail.
   Setup sentence: "Since localized editing requires relation-sensitive
   success beyond global prompt alignment, we report task-specific success
   checks and a blind internal audit in addition to CLIP-style scores."
4. Closed-loop ablations: E3 must feed each support variant into the SAME
   controller and report downstream edit success / outside drift (not only
   support IoU); E4 must show the edit-preserve Pareto/stress curve, not a
   single fixed-vs-feedback mean.
5. Rhythm: compact main table + RF baseline table + one Pareto/ablation
   figure in the main paper; per-task/per-seed grids, audit protocol, mask
   sensitivity, failure taxonomy in supplement. Main-paper cells 50-75.
6. Task-coverage wording: "controlled localized edit-preserve diagnostic
   suite" — after the 2026-06-10 rescope this reads: insertion / surface
   editing / appearance-material editing (Core-5), with exposed-object
   removal as an E5 boundary probe. Never "large-scale benchmark" or
   "general-purpose editing".
7. Report cost transparently even though efficiency is not the claim: NFE,
   runtime, resolution, GPU, source/target forward passes, inversion use —
   one context block for ALL methods (pre-empts "is the gain bought with
   compute?").
8. Fixed evaluation masks are stricter than EEdit's background-consistency
   habit: never the method's own support, never derived from outputs, never
   adjusted after seeing results — and this statement belongs in the MAIN
   TEXT experimental setup, not the appendix.
9. Five concrete borrowings: grouped headers; a "question answered" line per
   metric group; blind internal audit (3 raters, randomized method order,
   hidden method names, source+instruction visible, 1-5 rubric — call it
   "blind internal audit", never "user study"); main-text one-liner "ranking
   is stable under eroded/base/dilated evaluation masks" with the sweep in
   supplement; structured failure taxonomy in E5 that never enters main
   means.
10. Priority of presentation upgrades: (a) Table 1 grouping + task-specific
    success columns; (b) blind internal audit + mask sensitivity; (c) E3/E4
    from point ablations to downstream/Pareto evidence.

## Reconciliation With The Core-5 Rescope

The source analysis referenced the pre-rescope Core-6 wording (including
"simple exposed-object removal" in the suite sentence). Under the 2026-06-10
scope revision the suite sentence drops removal; the removal probe keeps the
same evaluation discipline (task-specific success = removal-aware metrics) in
E5. No other point is affected.

## Where Encoded

`paper/wacv_experiment_design.md` (2026-06-11 presentation-pass edits):
grouped Table 1 header spec, setup wording block, blind-audit protocol, mask
sensitivity line, efficiency context block. This file is the rationale
record.
