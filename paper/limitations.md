# Limitations

- The method depends on reliable spatial support for local edits.
- Current implementation is SD3-specific.
- Source-reference Q/K/V injection is experimental and requires systematic
  multi-seed ablation before it can be claimed as part of the main method. The
  seed-10 source V row improves preservation proxies but does not improve CLIP
  target alignment.
- External baselines must be compared only under matched prompts, seeds,
  resolution, and mask conditions. Current FireFlow/ReFlex artifacts are
  exploratory only and should not enter the main table.
- Known failure cases, including side-profile sunglasses and object-level
  replacement overlays, should remain visible in the paper.
- The current full method is preservation-biased: it has high DINO source
  similarity and low outside-mask drift, but weaker CLIP target-source gain.
- T3 yellow-car recoloring remains a color-miss case across the current main
  and ablation grids.
- T2 backpack recoloring can create a hybrid object rather than a clean surface
  recolor.
