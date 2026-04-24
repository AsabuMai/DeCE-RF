# RF h-Edit Prototype

This is a standalone research sandbox for migrating the core h-Edit idea to
rectified-flow image editing.

The goal is not to reproduce the full h-Edit paper inside FLUX in one step.
The goal is to own a separate codebase where we can test:

- source trajectory extraction
- path-based clean estimates `x_hat_0 = x_t - t v_theta`
- reconstruction/editing decoupling
- mask-aware reconstruction terms from SD3 cross-attention
- fidelity vs editability tradeoffs

## Project Layout

- `hrec_rf_pipeline.py`
  Standalone pipeline subclass that adds:
  - source trajectory extraction during inversion
  - optional reconstruction drift during editing
- `run_edit.py`
  CLI entrypoint for running a single edit
- `sd3_hrec.py`
  Standalone SD3-based research path for testing the document's ODE structure
- `attention_mask.py`
  Extracts soft editing masks from SD3 transformer cross-attention
- `energies.py`
  Reconstruction/editing energy surrogates and reconstruction-space alignment

## Current SD3 Research Direction

The current SD3 prototype is organized around the document's structure:

```text
x_hat_0(x_t, t) = x_t - t * v_theta(x_t, t)
M_edit = editing-region mask from SD3 cross-attention
M_preserve = 1 - M_edit

E_rec_total = lambda_latent * E_rec_latent + lambda_struct * E_rec_struct
E_edit_total = lambda_anchor * E_edit_anchor + lambda_region * E_edit_region

v_rec_total  ~= surrogate_grad(E_rec_total)
v_edit_total ~= surrogate_grad(E_edit_total)

dx_t/dt = beta(t) * v_edit_total + alpha(t) * v_rec_total

E_rec currently uses:
- a mask-aware multiscale latent term
- a source-conditioned attention structure term

E_edit currently uses:
- a target-anchor clean-estimate term
- a reserved region term placeholder
```

This is still a research prototype. It is not yet a full bridge-based RF
h-Edit method.

## Current Method Convention

We use the linear RF path:

```text
x_t = (1 - t) x_0 + t x_1
x0_hat = x_t - t * v_theta(x_t, t)
```

The controlled editing dynamics are implemented as:

```text
v_total = v_src + u_rec + u_edit
```

where `u_rec` is a reconstruction-aware preserve correction and `u_edit` is a
sum of target-seeking editing velocity surrogates. The implementation uses
energy-inspired surrogate velocity fields: some branches are exact autograd
gradients, while others are manually constructed velocity surrogates derived
from clean-space displacement or feature-space differences.

Source-side CFG is split because reconstruction fidelity and edit strength
need different operating points. By default the source inversion/base field
uses low CFG (`--src-guidance-scale 1.0`) to avoid prompt-driven re-generation,
while the target field can stay high (`--tar-guidance-scale 10.5`) to provide a
strong edit direction. Use `--inversion-guidance-scale` and
`--base-guidance-scale` when those two source roles need to be tuned
separately.

Attention masks are the main spatial control path. By default,
`--attention-mask-mode changed_union` builds the editable region from changed
source/target tokens rather than from the full prompt subject. This keeps local
edits such as `sunglasses over its eyes` focused, while object replacement
prompts such as panda to tiger still use the union of the changed source and
target concepts. Use `--mask-output-dir` to save `source_subject`,
`target_changed`, `combined`, and final mask PNGs for inspection.

For small local insertions, the changed-token union can still be too broad if
the prompt contains placement words that attend to the whole face or object.
Use `--attention-mask-target-words` plus stricter
`--attention-mask-subject-threshold` / `--attention-mask-core-threshold` to
make the attention mask follow the edited concept instead of the full subject.
Changed-token masks use cross-attention only; self-attention is still used for
full-subject masks, but it tends to spread local edit masks into visually
salient source regions.

For accessory-like edits, optional connected-component filtering can remove
attention spillover before dilation/smoothing. This is still attention-derived
spatial control, not a hand-written edit box:

```text
--edit-mask-component-y-max 0.45
--edit-mask-component-threshold 0.5
```

For local edits, the source inversion trajectory can also be used as an
explicit preserve-region anchor:

```text
z_next = z_next + s * M_preserve * (z_source_next - z_next)
```

Use `--trajectory-preserve-scale` to enable this. For the panda sunglasses
probe, the best pure-SD3 result so far is `target_changed` attention over
`sunglasses,eyes`, thresholds `0.48/0.72`, component filtering at
`y_max=0.45`, a small downward mask shift `0.10`, and target CFG `10.5`.
A wide unfiltered mask makes sunglasses strong, but it also lets the target
prompt re-generate the panda face.

## External Mask Diagnostics

The server also contains FLUX editing methods such as RF-Solver and FireFlow.
For the panda sunglasses prompt, those methods place the sunglasses correctly
but change the image too much. They should not replace the RF h-Edit
formulation above. They can be used only as diagnostics for the spatial support
of `M_edit`: extract a mask from `proposal - source`, then feed that mask back
into the same ODE/velocity formulation.

This keeps the mathematical objects unchanged:

```text
M_edit = external or attention-derived edit support
M_preserve = 1 - M_edit
v_total = v_src + u_rec + u_edit
u = -Delta x0 / t
```

The `proposal_local_composite.py` script is therefore a mask/debug utility. Its
composited image is useful for inspecting localization quality, but it is not
the method result.

```bash
cd /home/Wu_25R8111/rf_h_edit_project
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/proposal_local_composite.py \
  --source /home/Wu_25R8111/rf_editing_compare/panda_512.jpg \
  --proposal /home/Wu_25R8111/rf_editing_compare_sunglasses/outputs/fireflow/panda_sunglasses_inject_1_start_layer_index_0_end_layer_index_37_img_0.jpg \
  --output-dir /home/Wu_25R8111/rf_h_edit_project/outputs/proposal_composite_fireflow_v2 \
  --name fireflow_face_v2 \
  --roi 0.22,0.16,0.78,0.62 \
  --threshold 0.12 \
  --keep-components 5 \
  --min-area 8 \
  --dilate 11 \
  --blur 21 \
  --dark-bias 2.0 \
  --mode both
```

The extracted mask can then be used inside the unchanged SD3 RF dynamics:

```bash
CUDA_VISIBLE_DEVICES=2 /home/Wu_25R8111/ENTER/envs/flowedit/bin/python run_edit_sd3.py \
  --image /home/Wu_25R8111/h-edit/text-guided/assets/demo/panda.jpg \
  --source-prompt "A panda is walking in a forest." \
  --prompt "A panda wearing sunglasses over its eyes is walking in a forest." \
  --output /home/Wu_25R8111/rf_h_edit_project/results/panda_sunglasses_external_mask_sd3.png \
  --mask-output-dir /home/Wu_25R8111/rf_h_edit_project/results/panda_sunglasses_external_mask_masks \
  --external-edit-mask /home/Wu_25R8111/rf_h_edit_project/outputs/proposal_composite_fireflow_v2/fireflow_face_v2_mask.png \
  --external-edit-mask-mode replace \
  --src-guidance-scale 1.0 \
  --tar-guidance-scale 10.5 \
  --edit-hedit-guidance-scale 1.0 \
  --rec-guidance-scale 0 \
  --photo-prompt-mode both
```

Current diagnostic output:

```text
outputs/proposal_composite_final_comparison.png
outputs/proposal_composite_fireflow_v2/fireflow_face_v2_mask.png
```

ReFlex is still the most relevant algorithmic reference because it combines
attention adaptation, feature injection, and latent blending. In this local
environment it OOMs during FLUX inversion on a 24 GB RTX 3090 even at
`--image_size 256`, so the next research step is to port the mechanism rather
than depend on running ReFlex directly.

For clean-space displacement branches, the code uses the linear-path
conversion:

```text
u = -delta_x0 / t
```

because adding a velocity correction changes the clean estimate as:

```text
x0_hat' - x0_hat = -t * u
```

## Example

```bash
cd /home/Wu_25R8111/rf_h_edit_project
CUDA_VISIBLE_DEVICES=2 /home/Wu_25R8111/ENTER/envs/flowedit/bin/python run_edit_sd3.py \
  --image /home/Wu_25R8111/h-edit/text-guided/assets/demo/panda.jpg \
  --source-prompt "A panda is walking in a forest." \
  --prompt "A panda wearing sunglasses over its eyes is walking in a forest." \
  --output /home/Wu_25R8111/rf_h_edit_project/results/panda_sunglasses_sd3.png \
  --mask-output-dir /home/Wu_25R8111/rf_h_edit_project/results/panda_sunglasses_masks \
  --src-guidance-scale 1.0 \
  --tar-guidance-scale 10.5 \
  --attention-mask-mode target_changed \
  --attention-mask-target-words sunglasses,eyes \
  --attention-mask-subject-threshold 0.48 \
  --attention-mask-core-threshold 0.72 \
  --edit-mask-component-y-max 0.45 \
  --edit-mask-component-threshold 0.5 \
  --edit-mask-dilate-kernel 5 \
  --edit-mask-smooth-kernel 5 \
  --edit-mask-shift-y 0.10 \
  --edit-hedit-guidance-scale 1.0 \
  --rec-guidance-scale 0 \
  --trajectory-preserve-scale 0.15 \
  --photo-prompt-mode both
```

## Notes

- This project is independent from future edits to `RF-Inversion`.
- It still reuses the local `RF-Inversion` checkout as a read-only dependency
  for base FLUX RF inversion helpers.
- The SD3 entrypoint `run_edit_sd3.py` now downsizes the source image to a
  maximum long side of `512` by default before VAE encoding. Use
  `--max-image-size` to override this.
- See `docs/worklog_2026-04-24.md` for the detailed implementation and
  experiment log behind the current SD3 RF h-Edit cleanup.
