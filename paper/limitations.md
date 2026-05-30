# Limitations

- The method depends on reliable spatial support for local edits.
- Current implementation is SD3-specific.
- The current completed evidence is a core-5 matrix over selected localized
  add-object, decal, and exposed-object removal tasks. The final planned main
  benchmark is Core-6 after adding one recolor / attribute-editing task. It
  does not establish broad arbitrary editing.
- Removal is only partially covered. `backpack_remove_toy_charm` is a successful
  exposed-object removal case under visual audit, but occluded-object removal
  requiring host or background completion, such as `dog_remove_tennis_ball`,
  remains outside the main claim.
- Accurate support does not guarantee object erasure or surface completion.
  Sticker, magnet, and letter-removal probes show residual marks, transformed
  objects, or nearby-object damage even when localization is plausible.
- High-confidence completion prior helps only under restricted conditions. The
  `laptop_remove_sticker` probe supports planar-surface completion, while
  cluttered or semantic hosts should be gated off.
- Replacement target formation is not broadly solved. The whiteboard red-star
  probe shows that a strong non-glyph color/shape target can work, but precise
  glyph replacement, small tags, and dog-ball replacement remain unreliable.
- Recolor/appearance editing is the only planned main-matrix expansion. It
  should be added only after a seed-10 visual gate passes.
- `support_v3_fixed` currently covers the original core-4 ablation cache. It
  should be completed for any promoted final task set before making strong
  feedback-control claims.
- External baselines must be compared only under matched prompts, seeds,
  resolution, backbone assumptions, and mask inputs. Older core-4 baseline
  artifacts should be labeled as contextual evidence unless rerun under the
  core-5 protocol.
