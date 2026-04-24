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

## Example

```bash
cd /home/Wu_25R8111/rf_h_edit_project
CUDA_VISIBLE_DEVICES=2 /home/Wu_25R8111/ENTER/envs/flowedit/bin/python run_edit_sd3.py \
  --image /home/Wu_25R8111/h-edit/text-guided/assets/demo/panda.jpg \
  --source-prompt "A panda is walking in a forest." \
  --prompt "A polar bear is walking in a forest." \
  --output /home/Wu_25R8111/rf_h_edit_project/results/panda_to_polar_bear_sd3.png \
  --rec-guidance-scale 0.25
```

## Notes

- This project is independent from future edits to `RF-Inversion`.
- It still reuses the local `RF-Inversion` checkout as a read-only dependency
  for base FLUX RF inversion helpers.
- The SD3 entrypoint `run_edit_sd3.py` now downsizes the source image to a
  maximum long side of `512` by default before VAE encoding. Use
  `--max-image-size` to override this.
