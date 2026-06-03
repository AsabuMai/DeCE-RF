# Limitations

- The method depends on reliable spatial support for local edits.
- Current implementation is SD3-specific.
- The revised strict Phase 1 evidence covers all six current Core-6 rows:
  attached accessory, container-constrained insertion, surface decal, local
  recolor, surface material strip editing, and exposed-object removal. The
  suite is still a controlled diagnostic set, not a large-scale benchmark.
- Container-constrained insertion is covered by `bowl_apple_inside`, but the
  current `inside_container` relation is still a simple geometry prior derived
  from the grounded bowl mask. Do not overstate it as general 3D container
  reasoning or robust free-space placement.
- Surface material strip editing is covered by
  `pillow_vertical_fabric_strip`, which passed the strict visual gate. It
  should be described as a local surface-pattern/material-strip edit, not as
  broad material transfer.
- Removal is only partially covered. `backpack_remove_toy_charm` removes the
  intended dangling charm, but the zipper/fabric region occluded by the charm is
  locally smoothed. More difficult occluded-object removal requiring host or
  background completion, such as `dog_remove_tennis_ball`, remains outside the
  main claim.
- Accurate support does not guarantee object erasure or surface completion.
  Sticker, magnet, and letter-removal probes show residual marks, transformed
  objects, or nearby-object damage even when localization is plausible.
- High-confidence completion prior helps only under restricted conditions. The
  `laptop_remove_sticker` probe supports planar-surface completion, while
  cluttered or semantic hosts should be gated off.
- Replacement target formation is not broadly solved. The whiteboard red-star
  probe shows that a strong non-glyph color/shape target can work, but precise
  glyph replacement, small tags, and dog-ball replacement remain unreliable.
- Recolor/appearance editing is represented by `red_chair_blue`, which passed
  the strict Phase 1 visual audit. Describe it as a localized recolor probe,
  not evidence of general appearance editing.
- `support_v3_fixed` covers the earlier component-ablation set, but its gap to
  DeCE-RF is modest. Feedback control should be presented as component
  evidence and strengthened with stress/Pareto curves before making a headline
  robustness claim.
- External baselines must be compared only under matched prompts, seeds,
  resolution, backbone assumptions, and mask inputs. Older core-4 baseline
  artifacts should be labeled as contextual evidence unless rerun under the
  current evidence protocol.
