# E2 Status Lock

Date: 2026-06-05

Purpose: synchronize the E2 paper wording with the revised WACV experiment
design. This file records status and paper use only; it is not a new result
report.

Primary evidence lock:

```text
experiments/support_v3_2026-06-02/e2_evidence_lock_2026-06-04.md
```

## Status Levels

| Status | Meaning | Paper use |
| --- | --- | --- |
| LOCKED | completed under the current protocol with fixed masks, metrics, and audit artifacts | main or supplement evidence, within the stated claim boundary |
| DIAGNOSTIC | useful for mechanism or boundary interpretation, but not a fair headline row | supplement, ablation, or limitation |
| PLANNED | designed but not complete | run plan only |
| BLOCKED | access, repo, environment, or adapter is not runnable/verified | transparent status disclosure only |

## E2.1 Calibration / Protocol

| Row | Backbone | Status | Paper use | Notes |
| --- | --- | --- | --- | --- |
| `base_only` | SD3 | LOCKED | reconstruction floor | reused from strict E1 |
| `direct_target` | SD3 | LOCKED | naive edit-preserve floor | reused from strict E1 and Table 2a |
| native reconstruction/direct floor | FLUX/native | LOCKED where present in native context artifacts; otherwise PLANNED/BLOCKED by method | contextual calibration only | do not merge into same-backbone SD3 algorithm table |

## E2.2 Same-Backbone SD3 Algorithm Comparison

| Row | Backbone | Status | Paper use | Notes |
| --- | --- | --- | --- | --- |
| `direct_target-SD3` | SD3 | LOCKED | same-backbone algorithm/control row | coupled target baseline |
| `FlowEdit-SD3` | SD3 | LOCKED | same-backbone RF-native baseline | algorithmic evidence within SD3 only |
| `FlowAlign-SD3` | SD3 | LOCKED | same-backbone RF-native baseline | algorithmic evidence within SD3 only |
| `SplitFlow-SD3` | SD3 | LOCKED | same-backbone RF-native baseline | algorithmic evidence within SD3 only |
| `Fixed DeCE-SD3` / `support_v3_fixed` | SD3 | LOCKED as component cache; join into Table 2a as preservation-control row | same-backbone preservation-control row and E4 ablation | not an external baseline, not E1 headline |
| `DeCE-RF-SD3` / `support_v3_controller_rmsgap` | SD3 | LOCKED | full method | main algorithm row |
| OT-RF/OTIP-SD3, RF-Edit-SD3, DVRF-SD3 | SD3 | PLANNED unless a real SD3 adapter is verified | optional same-backbone row only | otherwise move to E2.3/status table |

## E2.3 Native Preservation-Aware RF Context

| Row | Native backbone/interface | Status | Paper use | Notes |
| --- | --- | --- | --- | --- |
| RF-Solver-Edit / RF-Edit | native/public route | LOCKED when present in `e2_native_flux_*` artifacts; otherwise disclose blocker | contextual native baseline | do not claim pure algorithmic superiority |
| ReFlex | native/public route | LOCKED when present in `e2_native_flux_*` artifacts; otherwise disclose blocker | contextual native baseline | do not claim SD3-DeCE beats ReFlex/FLUX as an algorithm |
| FireFlow | native/public route | LOCKED when present in `e2_native_flux_*` artifacts; otherwise disclose blocker | contextual native baseline | native RF-flow context |
| stable-flow | native/public route | PLANNED/BLOCKED unless strict adapter is complete | status/context only | include exact adapter blocker if not runnable |
| OT-RF / OTIP | repo-dependent | PLANNED/BLOCKED unless runnable | status/context only, or E2.2 if same-SD3 verified | no invented comparison |
| DVRF / Delta Velocity RF | repo-dependent | PLANNED/BLOCKED unless runnable | status/context only, or E2.2 if same-SD3 verified | no invented comparison |

Paper-safe wording:

```text
Native preservation-aware RF editors are reported as implementation-context
baselines because their public routes use different backbones or interfaces.
```

## E2.4 Support-Matched Diagnostic

| Row | Support condition | Status | Paper use | Notes |
| --- | --- | --- | --- | --- |
| `direct_target_raw` | none | LOCKED | diagnostic anchor | aggressive raw baseline |
| `direct_target_mask_blend` | same binary mask, post-hoc | LOCKED | diagnostic only | preservation improves by construction |
| `flowedit_mask_blend` | same binary mask, post-hoc | LOCKED | diagnostic only | not a fair main baseline |
| `support_v3_controller_rmsgap` | operation support + controller | LOCKED | full-method diagnostic comparator | same task subset |
| `direct_target + same M_edit gating` | inference-time same support | PLANNED | stronger support-matched diagnostic | run if feasible |
| `FlowEdit + same M_edit gating` | inference-time same support | PLANNED | stronger support-matched diagnostic | only if wrapper is stable |
| `Fixed DeCE` | operation support, no feedback | LOCKED/PLANNED depending joined subset | support + fixed displacement comparator | do not label as support-only |

Paper-safe wording:

```text
Binary localization/output blending improves preservation metrics by
construction but does not recover target correctness or boundary coherence.
Therefore, localization alone is insufficient to explain the DeCE-RF result.
```

## Forbidden E2 Claims

Do not write:

```text
DeCE-RF beats all RF editors.
DeCE-RF beats FLUX.
SD3-DeCE is directly superior to ReFlex-FLUX or RF-Edit-FLUX as an algorithm.
```

Use:

```text
Under the same SD3 backbone and fixed evaluation masks, DeCE-RF improves the
localized edit-preserve tradeoff over completed SD3 RF-native baselines and
fixed decoupled preservation controls.
```
