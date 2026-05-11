# Surface Recolor Reference

This is the generic reference-building path for masked appearance recoloring.
It is not vehicle-specific.

## Concept

```text
source image + surface mask + target color
  -> source-luma-preserving surface recolor reference
  -> EDIT_REF_IMAGE for ODE guidance
```

The ODE edit remains responsible for the final result. The generated image is
a reference target, not the final postprocess output.

## Main Entry

```bash
python scripts/make_surface_recolor_reference.py \
  --image input.png \
  --surface-mask mask.png \
  --target-color blue \
  --luma-image input.png \
  --mode hsv-target \
  --output reference.png \
  --metadata-output metadata.json
```

`--surface-mask` can come from any provider: SAM, GroundingDINO, a clothing
segmenter, a user mask, or a composed mask.

Available modes:

```text
hsv-target   default; target-color H/S injection, source V/luma preservation
hsv-hue      legacy/probe mode; preserve value/luma and interpolate hue/saturation
lab-chroma   useful for some materials, but can shift yellow surfaces toward purple
yuv-chroma   conservative chroma change
```

For yellow-to-blue surfaces, prefer `hsv-target` when the reference itself
shows yellow/green residue near semi-transparent mask boundaries.

## ODE Interface

Use the generated reference with:

```text
EDIT_REF_IMAGE=/path/to/reference.png
EDIT_REF_MASK=/path/to/surface_mask.png
EDIT_REF_GUIDANCE_SCALE=...
EDIT_REF_SCHEDULE_START=...
EDIT_REF_SCHEDULE_STOP=...
EDIT_REF_MAX_STRUCT_RMS_RATIO=...
EDIT_REF_PROJECT_STRUCT_CONFLICT=...
```

Color matching modes:

```text
EDIT_REF_CHROMA_MODE=yuv            stable default for complex/small masks
EDIT_REF_CHROMA_MODE=yuv_direction  stronger hue direction for simple large color regions
```

`yuv_direction` is computed in fp32 with soft chroma normalization. It fixed
dark blue collapse on the simple stop-sign test, but it can over-amplify local
artifacts on small reflective regions. Keep `yuv` as the default unless the
absolute YUV match is too dark or too weak.

## Naming Rule

Use `surface` / `reference` names in shared interfaces. Domain-specific scripts
may still exist as mask providers, but they should output a generic surface
mask consumed by the ODE path.
