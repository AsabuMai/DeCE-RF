# replace_editor_v1 probe audit

Date: 2026-05-12

## Scope

Goal: test the current replacement-edit hypothesis without leaving the existing RF control logic.

Method tested: `M26/support_v3_controller_rmsgap_replace_editor_v1`.

Core change:

- Keep the existing RF decomposition: `v_total = v_base + v_rec + v_edit`.
- Keep rmsgap as the preservation controller.
- Add a local target-formation field inside the edit support:
  - compute a target-prompt CFG velocity,
  - convert its predicted clean target delta into RF velocity with `clean_delta_to_velocity`,
  - gate it by edit/core support,
  - schedule it mostly in the high-noise / mid-noise part of the trajectory.

Implementation entry points:

- `sd3_hrec.py`: `edit_local_target_prompt`, `edit_local_target_guidance_scale`, `edit_local_target_cfg_scale`.
- `run_edit_sd3.py`: CLI args `--edit-local-target-*`.
- `scripts/run_pretty_matrix.sh`: `M26/support_v3_controller_rmsgap_replace_editor_v1`.

## Probe

Command family:

```bash
TASKS="P8 P11" METHODS="M26" SEEDS="10" DEVICE=7 METHOD_NAME_SUFFIX="_probe" EDIT_STRENGTH_MULTIPLIER=1.0 \
  SKIP_EXISTING=0 REGENERATE_MASKS=0 REUSE_SEMANTIC_MASKS=1 \
  SEMANTIC_MASK_CACHE_METHOD=support_v3_fixed_policyv1_debug ALLOW_MASK_DOWNLOAD=0 \
  bash scripts/run_pretty_matrix.sh
```

Cases:

- `backpack_replace_patch_blue`: colorful cartoon patch -> plain blue fabric patch.
- `dog_replace_tennis_ball_star`: green tennis ball -> red star-shaped dog toy.

Preview:

- `experiments/support_v3_2026-05-11/replace_editor_v1_probe/policyv1_edit_gate_scale100.png`

## Runtime Signal

The new local target field was active:

| case | avg local target formation norm | max local target formation norm |
| --- | ---: | ---: |
| `backpack_replace_patch_blue` | 5.6071 | 19.2311 |
| `dog_replace_tennis_ball_star` | 10.5970 | 21.3591 |

Prompts:

- `A close-up photo of plain blue fabric patch.`
- `A close-up photo of red star-shaped dog toy.`

Scale:

- `edit_local_target_guidance_scale = 0.30`
- `edit_local_target_cfg_scale = 8.0`

## Visual Finding

This version is not a clear improvement over `replace_editor_v0`.

- P8: `replace_v1` creates a light/yellowish local fabric-like disturbance, but does not form a believable blue fabric patch. It still behaves mostly like local suppression plus surface distortion.
- P11: `replace_v1` produces a red/green blob in the mouth region. The edit fires, but the new object is less clean than `replace_v0` and not clearly better than the simpler opfield/fixed variants for this seed.

## Conclusion

The target-formation RF prior is mathematically wired in and produces nonzero force, but this exact formulation should not become the main line. It adds target pressure, yet the pressure is still too prompt-global and not enough of an object-level formation mechanism. It can increase local disturbance without reliably creating the replacement object's geometry/material.

The current evidence favors keeping rmsgap/opfield as the main line and treating replacement as a separate formation problem: old-object suppression + support-local target synthesis + preservation coupling, not only stronger target CFG inside the support.

## Recommended Next Step

Do not keep tuning only `edit_local_target_guidance_scale`.

Next replace-specific change should separate the target field into two terms:

1. source-removal term: suppress old-object clean response inside the source core;
2. target-construction term: inject a localized clean target prior anchored to support geometry, preferably with a task-level target mask/shape proposal or a local synthesized clean prototype.

Then evaluate on the same P8/P11 seed10 preview before expanding.

## Validation

- `python3 -m py_compile run_edit_sd3.py sd3_hrec.py scripts/make_policyv1_edit_gate_preview.py`
- `bash -n scripts/run_pretty_matrix.sh`
- `/home/Wu_25R8111/ENTER/envs/flowedit/bin/python -m pytest tests/test_operation_support_v3.py -q`

Result: all passed.
