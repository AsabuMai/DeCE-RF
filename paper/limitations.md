# Limitations

- The method depends on reliable spatial support for local edits.
- Current implementation is SD3-specific.
- The completed server evidence package covers selected localized add-object,
  decal, exposed-object removal, and recolor/attribute-editing tasks. It maps
  into the updated WACV taxonomy as T1/T3/T4/T6 evidence plus supplementary
  T1/T3 rows; it does not yet cover the updated strict T2/T5 definitions.
- Container-constrained insertion is not solved yet. Server add-editor probes
  show that bowl/apple placement can be pulled toward the bowl rim or external
  shadows, so the updated T2 needs a geometry-aware interior placement prior.
- Surface pattern editing is not yet separately established. `tshirt_star`
  supports clothing/surface decal editing, but it should not be overstated as a
  general medium-area pattern-edit result unless the paper explicitly narrows
  T5 to clothing-decal evidence.
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
- Recolor/appearance editing is represented by the promoted `red_chair_blue`
  server evidence task, which passed the seed-10 visual gate and was rerun over
  seeds 10/11/12.
- `support_v3_fixed` now covers the promoted server evidence task set, but its
  gap to DeCE-RF is modest. Feedback control should be presented as component
  evidence, not as a standalone headline gain.
- External baselines must be compared only under matched prompts, seeds,
  resolution, backbone assumptions, and mask inputs. Older core-4 baseline
  artifacts should be labeled as contextual evidence unless rerun under the
  current evidence protocol.
