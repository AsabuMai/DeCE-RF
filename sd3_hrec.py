from __future__ import annotations

import copy
import json
import os
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from clip_text_reward import CLIPReferenceState, LocalCLIPTextReward
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from attention_mask import extract_attention_masks, extract_prompt_attention_map
from energies import (
    cosine_safe,
    editing_energy_total,
    editing_velocity_surrogate_total,
    reconstruction_energy_total,
    reconstruction_velocity_surrogate_total,
)
from schedules import get_schedule_value


class _SD3FeatureStore:
    def __init__(self):
        self.features = []
        self.handles = []

    def register(self, transformer, layer_indices, detach: bool):
        for i in layer_indices:
            block = transformer.transformer_blocks[i]

            def _make_hook():
                def _hook(_module, _inputs, output):
                    hidden_states = output[1]
                    feat = hidden_states.detach() if detach else hidden_states
                    self.features.append(feat)

                return _hook

            self.handles.append(block.register_forward_hook(_make_hook()))

    def restore(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _feature_tokens_to_map(feature_tokens: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    token_norm = torch.linalg.norm(feature_tokens, dim=-1)  # (B, seq)
    fmap = token_norm[0].reshape(1, 1, img_h, img_w)
    mn, mx = fmap.min(), fmap.max()
    return (fmap - mn) / (mx - mn + 1e-6)


def extract_sd3_feature_structure_map(
    pipe,
    x_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    detach: bool = True,
    max_latent_side: int = 128,
) -> torch.Tensor:
    """
    Capture a stronger structure representation from SD3 mid-block visual features.

    Unlike attention heatmaps, this uses the transformer block outputs
    themselves, which carry more local geometry/layout information.
    """
    orig_size = x_latent.shape[-2:]
    transformer_dtype = next(pipe.transformer.parameters()).dtype

    h, w = x_latent.shape[-2], x_latent.shape[-1]
    if h > max_latent_side or w > max_latent_side:
        scale = max_latent_side / max(h, w)
        new_h = max(2, int(h * scale) // 2 * 2)
        new_w = max(2, int(w * scale) // 2 * 2)
        x_small = torch.nn.functional.interpolate(
            x_latent.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).to(dtype=transformer_dtype)
    else:
        x_small = x_latent.to(dtype=transformer_dtype)

    n_blocks = len(pipe.transformer.transformer_blocks)
    lo, hi = n_blocks // 4, 3 * n_blocks // 4
    layer_indices = list(range(lo, hi))
    img_h = x_small.shape[-2] // 2
    img_w = x_small.shape[-1] // 2
    timestep = t.expand(x_small.shape[0])
    prompt_embeds = prompt_embeds.to(device=x_small.device, dtype=transformer_dtype)
    pooled_embeds = pooled_embeds.to(device=x_small.device, dtype=transformer_dtype)

    store = _SD3FeatureStore()
    try:
        store.register(pipe.transformer, layer_indices, detach=detach)
        pipe.transformer(
            hidden_states=x_small,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )
    finally:
        store.restore()

    if not store.features:
        return torch.zeros(1, 1, *orig_size, device=x_latent.device, dtype=x_latent.dtype)

    maps = [_feature_tokens_to_map(feat, img_h, img_w) for feat in store.features]
    fmap = torch.stack(maps, dim=0).mean(dim=0).to(x_latent.device, dtype=x_latent.dtype)
    fmap = torch.nn.functional.interpolate(fmap, size=orig_size, mode="bilinear", align_corners=False)
    return fmap


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


@torch.no_grad()
def calc_cfg_v_sd3(
    pipe,
    latents: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    t: torch.Tensor,
) -> torch.Tensor:
    if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None:
        timestep = t.expand(latents.shape[0])
        return pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    timestep = t.expand(latents.shape[0] * 2)
    latent_model_input = torch.cat([latents, latents], dim=0)
    prompt_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_input,
        pooled_projections=pooled_input,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


def predict_x0_from_linear_rf_path(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t_scalar: Union[float, torch.Tensor],
) -> torch.Tensor:
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=x_t.device, dtype=x_t.dtype)
    t_scalar = t_scalar.to(device=x_t.device, dtype=x_t.dtype)
    while t_scalar.ndim < x_t.ndim:
        t_scalar = t_scalar.view(*t_scalar.shape, 1)
    return x_t - t_scalar * v_t


def decode_latent_to_unit_image(
    pipe,
    latent: torch.Tensor,
    vae_override=None,
) -> torch.Tensor:
    vae = pipe.vae if vae_override is None else vae_override
    latent_denorm = ((latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor).clone()
    image = vae.decode(latent_denorm, return_dict=False)[0]
    return ((image / 2.0) + 0.5).clamp(0.0, 1.0)


def masked_total_variation(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    if mask is not None:
        mask = torch.nn.functional.interpolate(
            mask,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        dx = dx * mask[:, :, :, 1:]
        dy = dy * mask[:, :, 1:, :]
    return dx.abs().mean() + dy.abs().mean()


def smooth_guidance_field(
    field: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    if kernel_size <= 1:
        return field
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(field, kernel_size=kernel_size, stride=1, padding=pad)


def translate_spatial_mask(mask: torch.Tensor, shift_y: float = 0.0, shift_x: float = 0.0) -> torch.Tensor:
    """
    Translate a BCHW mask without wraparound.

    `shift_y` and `shift_x` are fractions of mask height/width. Positive y
    moves the mask down; positive x moves it right.
    """
    if shift_y == 0.0 and shift_x == 0.0:
        return mask
    h, w = mask.shape[-2:]
    dy = int(round(shift_y * h))
    dx = int(round(shift_x * w))
    if dy == 0 and dx == 0:
        return mask

    out = torch.zeros_like(mask)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = mask[..., src_y0:src_y1, src_x0:src_x1]
    return out


def dilate_spatial_mask(mask: torch.Tensor, kernel_size: int = 0) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return torch.nn.functional.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad).to(mask.dtype)


def smooth_spatial_mask(mask: torch.Tensor, kernel_size: int = 0) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad).to(mask.dtype)


def filter_spatial_mask_components(
    mask: torch.Tensor,
    threshold: float = 0.5,
    keep_components: int = 0,
    center_y_min: float | None = None,
    center_y_max: float | None = None,
) -> torch.Tensor:
    if keep_components <= 0 and center_y_min is None and center_y_max is None:
        return mask
    threshold = max(0.0, min(1.0, threshold))
    out = torch.zeros_like(mask)
    mask_cpu = mask.detach().float().cpu()
    bsz, _, h, w = mask_cpu.shape
    for b in range(bsz):
        binary = mask_cpu[b, 0] > threshold
        visited = torch.zeros_like(binary, dtype=torch.bool)
        components: list[tuple[float, list[tuple[int, int]]]] = []
        for y in range(h):
            for x in range(w):
                if not bool(binary[y, x]) or bool(visited[y, x]):
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                cy_norm = sum(p[0] for p in points) / max(1, len(points)) / h
                if center_y_min is not None and cy_norm < center_y_min:
                    continue
                if center_y_max is not None and cy_norm > center_y_max:
                    continue
                mass = float(sum(float(mask_cpu[b, 0, py, px]) for py, px in points))
                components.append((mass, points))
        components.sort(key=lambda item: item[0], reverse=True)
        selected = components if keep_components <= 0 else components[:keep_components]
        for _, points in selected:
            for py, px in points:
                out[b, 0, py, px] = mask[b, 0, py, px]
    return out


def normalized_box_mask_like(
    reference: torch.Tensor,
    box: tuple[float, float, float, float],
    feather: float = 0.025,
) -> torch.Tensor:
    """
    Build a soft BCHW mask from normalized image coordinates.

    Box format is (x0, y0, x1, y1), each in [0, 1].
    """
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    h, w = reference.shape[-2:]
    ys = torch.linspace(0.0, 1.0, h, device=reference.device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=reference.device, dtype=torch.float32).view(1, 1, 1, w)
    feather = max(feather, 1e-4)
    inside_x = torch.sigmoid((xs - x0) / feather) * torch.sigmoid((x1 - xs) / feather)
    inside_y = torch.sigmoid((ys - y0) / feather) * torch.sigmoid((y1 - ys) / feather)
    mask = (inside_x * inside_y).clamp(0.0, 1.0)
    return mask.expand(reference.shape[0], 1, h, w).to(dtype=reference.dtype)


def load_external_mask_like(reference: torch.Tensor, mask_path: str) -> torch.Tensor:
    """
    Load an external grayscale edit mask as BCHW in the same spatial space as `reference`.

    This only changes the support of M_edit. It does not alter the
    reconstruction/editing velocity formulation.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = torch.from_numpy(np.array(mask_img, dtype="float32") / 255.0)
    mask = mask_arr[None, None].to(device=reference.device, dtype=reference.dtype)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = torch.nn.functional.interpolate(
            mask,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return mask.expand(reference.shape[0], 1, reference.shape[-2], reference.shape[-1]).clamp(0.0, 1.0)


def save_mask_image(mask: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = mask.detach().float()[0, 0].clamp(0.0, 1.0)
    array = (image.cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def _cfg_v_sd3_with_grad(
    pipe,
    latents: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    t: torch.Tensor,
) -> torch.Tensor:
    if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None:
        timestep = t.expand(latents.shape[0])
        return pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    timestep = t.expand(latents.shape[0] * 2)
    latent_model_input = torch.cat([latents, latents], dim=0)
    prompt_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_input,
        pooled_projections=pooled_input,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


@torch.no_grad()
def invert_source_sd3(
    pipe,
    x_src: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    timesteps,
    T_steps: int,
    n_max: int,
    return_trajectory: bool = False,
):
    """
    Forward ODE (source inversion): integrate v_src from sigma=0 → sigma_max.

    Traverses the active timesteps in reverse (ascending sigma) so the
    inverted latent z_T lives at the same noise level where editing starts.

    If return_trajectory=True, also returns a timestep-keyed dictionary of
    intermediate source latents. Key 0 is the original source latent, and each
    active scheduler timestep key stores the latent reached by source
    inversion at that noise level.
    """
    active_ts = [t for i, t in enumerate(timesteps) if T_steps - i <= n_max]
    inv_ts = list(reversed(active_ts))

    z_t = x_src.clone().to(torch.float32)
    latents_dtype = x_src.dtype

    trajectory: list[torch.Tensor] = []
    trajectory_by_timestep: dict[int, torch.Tensor] = {}
    if return_trajectory:
        source_latent = z_t.clone().to(latents_dtype)
        trajectory.append(source_latent)  # index 0: x_src
        trajectory_by_timestep[0] = source_latent

    for i, t in enumerate(inv_ts[:-1]):
        t_curr = t / 1000
        t_next = inv_ts[i + 1] / 1000
        v = calc_cfg_v_sd3(
            pipe, z_t.to(latents_dtype),
            negative_prompt_embeds, prompt_embeds,
            negative_pooled_prompt_embeds, pooled_prompt_embeds,
            guidance_scale, t,
        )
        z_t = z_t + (t_next - t_curr).to(torch.float32) * v.to(torch.float32)
        if return_trajectory:
            source_latent = z_t.clone().to(latents_dtype)
            trajectory.append(source_latent)
            trajectory_by_timestep[int(inv_ts[i + 1].item())] = source_latent

    z_T = z_t.to(latents_dtype)
    if return_trajectory:
        if active_ts:
            trajectory_by_timestep[int(active_ts[0].item())] = z_T
        return z_T, trajectory_by_timestep
    return z_T


def HRecSD3Edit(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    T_steps: int = 28,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 10.5,
    inversion_guidance_scale: Optional[float] = None,
    base_guidance_scale: Optional[float] = None,
    n_max: int = 24,
    eta: float = 0.8,
    rec_guidance_scale: float = 0.0,
    struct_guidance_scale: float = 0.5,
    edit_hedit_guidance_scale: float = 0.0,
    edit_src_cfg_scale: Optional[float] = None,
    edit_guidance_scale: float = 0.0,
    edit_region_guidance_scale: float = 0.0,
    edit_target_guidance_scale: float = 0.0,
    edit_source_guidance_scale: float = 0.0,
    edit_clip_guidance_scale: float = 0.0,
    edit_clip_match_base_scale: float = 0.0,
    edit_image_tv_scale: float = 0.0,
    edit_text_guidance_scale: float = 0.0,
    edit_text_source_scale: float = 0.8,
    edit_text_core_weight: float = 1.0,
    edit_text_subject_weight: float = 0.3,
    identity_break_stop: float = 0.55,
    target_attract_start: float = 0.25,
    edit_dds_guidance_scale: float = 0.0,
    edit_dds_source_scale: float = 0.8,
    edit_app_guidance_scale: float = 0.0,
    edit_core_scale: float = 1.35,
    edit_subject_scale: float = 0.35,
    edit_bound_scale: float = 0.0,
    clip_start_timestep: float = 0.0,
    clip_stop_timestep: float = 0.6,
    preserve_blend_scale: float = 0.0,
    preserve_blend_start_timestep: float = 0.5,
    alpha_max: Optional[float] = None,
    alpha_schedule: str = "constant",
    beta_max: Optional[float] = None,
    beta_schedule: str = "constant",
    velocity_conversion_mode: str = "linear_path",
    linear_path_t_min: float = 0.05,
    rec_stop_timestep: float = 0.08,
    trajectory_preserve_scale: float = 0.0,
    trajectory_subject_preserve_scale: float = 0.0,
    edit_initial_noise_scale: float = 0.0,
    edit_initial_noise_region: str = "core",
    attention_mask_mode: str = "changed_union",
    attention_mask_target_words: list[str] | None = None,
    attention_mask_source_words: list[str] | None = None,
    attention_mask_subject_threshold: float = 0.48,
    attention_mask_core_threshold: float = 0.72,
    mask_output_dir: str | None = None,
    edit_mask_dilate_kernel: int = 0,
    edit_mask_smooth_kernel: int = 0,
    edit_mask_component_threshold: float = 0.0,
    edit_mask_keep_components: int = 0,
    edit_mask_component_y_min: float | None = None,
    edit_mask_component_y_max: float | None = None,
    edit_mask_shift_y: float = 0.0,
    edit_mask_shift_x: float = 0.0,
    edit_mask_box: tuple[float, float, float, float] | None = None,
    edit_mask_box_mode: str = "replace",
    external_edit_mask_path: str | None = None,
    external_edit_mask_mode: str = "replace",
    mask_blend: bool = False,
    log_every: int = 0,
    stats_output_path: Optional[str] = None,
):
    """
    RF image editing via source inversion + controlled reverse ODE.

    The active implementation follows the project logic:
    - a source-conditioned RF base prior
    - a separate reconstruction guidance branch
    - a separate editing guidance branch
    both defined from the same clean estimate x_hat_0.

    The current implementation uses energy-inspired surrogate velocity fields.
    Some branches are exact autograd gradients, while others are manually
    constructed velocity surrogates derived from clean-space displacement or
    feature-space differences.
    """
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    if inversion_guidance_scale is None:
        inversion_guidance_scale = src_guidance_scale
    if base_guidance_scale is None:
        base_guidance_scale = src_guidance_scale
    if edit_src_cfg_scale is None:
        edit_src_cfg_scale = base_guidance_scale
    source_encode_guidance_scale = max(
        float(inversion_guidance_scale),
        float(base_guidance_scale),
        float(edit_src_cfg_scale),
        1.0,
    )

    with torch.no_grad():
        pipe._guidance_scale = source_encode_guidance_scale
        (
            src_prompt_embeds,
            src_negative_prompt_embeds,
            src_pooled_prompt_embeds,
            src_negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=src_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            device=device,
        )

        pipe._guidance_scale = tar_guidance_scale
        (
            tar_prompt_embeds,
            tar_negative_prompt_embeds,
            tar_pooled_prompt_embeds,
            tar_negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=tar_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            device=device,
        )

    # --- Source inversion: forward ODE x_src → z_T ---
    print("[inversion] running source forward ODE ...")
    inversion_result = invert_source_sd3(
        pipe=pipe,
        x_src=x_src,
        negative_prompt_embeds=src_negative_prompt_embeds,
        prompt_embeds=src_prompt_embeds,
        negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
        pooled_prompt_embeds=src_pooled_prompt_embeds,
        guidance_scale=inversion_guidance_scale,
        timesteps=timesteps,
        T_steps=T_steps,
        n_max=n_max,
        return_trajectory=trajectory_preserve_scale > 0.0 or trajectory_subject_preserve_scale > 0.0,
    )
    if trajectory_preserve_scale > 0.0 or trajectory_subject_preserve_scale > 0.0:
        z_T, source_trajectory_by_timestep = inversion_result
    else:
        z_T = inversion_result
        source_trajectory_by_timestep = {}
    print("[inversion] done.")

    if alpha_max is None:
        alpha_max = rec_guidance_scale
    if beta_max is None:
        beta_max = 1.0

    # Extract attention mask for soft E_rec (optional)
    M_edit = None
    M_core = None
    M_preserve = None
    source_feature_maps: dict[int, torch.Tensor] = {}
    source_attention_maps: dict[int, torch.Tensor] = {}
    clip_reward: LocalCLIPTextReward | None = None
    clip_reference: CLIPReferenceState | None = None
    grad_vae = None
    if (
        rec_guidance_scale > 0.0
        or edit_hedit_guidance_scale > 0.0
        or edit_guidance_scale > 0.0
        or edit_region_guidance_scale > 0.0
        or edit_target_guidance_scale > 0.0
        or edit_source_guidance_scale > 0.0
        or edit_clip_guidance_scale > 0.0
        or edit_text_guidance_scale > 0.0
        or edit_dds_guidance_scale > 0.0
        or edit_app_guidance_scale > 0.0
        or trajectory_preserve_scale > 0.0
        or trajectory_subject_preserve_scale > 0.0
        or edit_initial_noise_scale > 0.0
        or external_edit_mask_path is not None
    ):
        with torch.no_grad():
            t_mid = timesteps[len(timesteps) // 2]
            masks = extract_attention_masks(
                pipe=pipe,
                x_src=x_src,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                src_prompt_embeds=src_prompt_embeds,
                src_pooled_embeds=src_pooled_prompt_embeds,
                tar_prompt_embeds=tar_prompt_embeds,
                tar_pooled_embeds=tar_pooled_prompt_embeds,
                t=t_mid,
                mode=attention_mask_mode,
                target_token_words=attention_mask_target_words,
                source_token_words=attention_mask_source_words,
                subject_threshold=attention_mask_subject_threshold,
                core_threshold=attention_mask_core_threshold,
            )
            M_edit = masks["subject"].to(dtype=x_src.dtype)
            M_core = masks["core"].to(dtype=x_src.dtype)
            M_preserve = masks["preserve"].to(dtype=x_src.dtype)
            if (
                edit_mask_keep_components > 0
                or edit_mask_component_y_min is not None
                or edit_mask_component_y_max is not None
            ):
                component_threshold = edit_mask_component_threshold if edit_mask_component_threshold > 0.0 else 0.5
                M_edit = filter_spatial_mask_components(
                    M_edit,
                    threshold=component_threshold,
                    keep_components=edit_mask_keep_components,
                    center_y_min=edit_mask_component_y_min,
                    center_y_max=edit_mask_component_y_max,
                ).clamp(0.0, 1.0)
                M_core = filter_spatial_mask_components(
                    M_core,
                    threshold=component_threshold,
                    keep_components=edit_mask_keep_components,
                    center_y_min=edit_mask_component_y_min,
                    center_y_max=edit_mask_component_y_max,
                ).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_shift_y != 0.0 or edit_mask_shift_x != 0.0:
                M_edit = translate_spatial_mask(M_edit, shift_y=edit_mask_shift_y, shift_x=edit_mask_shift_x)
                M_core = translate_spatial_mask(M_core, shift_y=edit_mask_shift_y, shift_x=edit_mask_shift_x)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_dilate_kernel > 1:
                M_edit = dilate_spatial_mask(M_edit, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                M_core = dilate_spatial_mask(M_core, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_smooth_kernel > 1:
                M_edit = smooth_spatial_mask(M_edit, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                M_core = smooth_spatial_mask(M_core, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if edit_mask_box is not None:
                box_mask = normalized_box_mask_like(M_edit, edit_mask_box)
                if edit_mask_box_mode == "replace":
                    M_edit = box_mask
                    M_core = box_mask
                elif edit_mask_box_mode == "intersect":
                    M_edit = M_edit * box_mask
                    M_core = M_core * box_mask
                elif edit_mask_box_mode == "union":
                    M_edit = torch.maximum(M_edit, box_mask)
                    M_core = torch.maximum(M_core, box_mask)
                else:
                    raise ValueError(f"Unsupported edit_mask_box_mode: {edit_mask_box_mode}")
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            external_mask = None
            if external_edit_mask_path is not None:
                external_mask = load_external_mask_like(M_edit, external_edit_mask_path)
                if external_edit_mask_mode == "replace":
                    M_edit = external_mask
                    M_core = external_mask
                elif external_edit_mask_mode == "intersect":
                    M_edit = M_edit * external_mask
                    M_core = M_core * external_mask
                elif external_edit_mask_mode == "union":
                    M_edit = torch.maximum(M_edit, external_mask)
                    M_core = torch.maximum(M_core, external_mask)
                else:
                    raise ValueError(f"Unsupported external_edit_mask_mode: {external_edit_mask_mode}")
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                for name, mask in masks.items():
                    save_mask_image(mask.to(dtype=torch.float32), os.path.join(mask_output_dir, f"{name}.png"))
                save_mask_image(M_edit.to(dtype=torch.float32), os.path.join(mask_output_dir, "subject_final.png"))
                save_mask_image(M_core.to(dtype=torch.float32), os.path.join(mask_output_dir, "core_final.png"))
                save_mask_image(M_preserve.to(dtype=torch.float32), os.path.join(mask_output_dir, "preserve_final.png"))
                if external_mask is not None:
                    save_mask_image(
                        external_mask.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "external_edit_mask.png"),
                    )
            if edit_initial_noise_scale > 0.0:
                if edit_initial_noise_region == "core":
                    noise_gate = M_core
                elif edit_initial_noise_region == "subject":
                    noise_gate = M_edit
                else:
                    raise ValueError(f"Unsupported edit_initial_noise_region: {edit_initial_noise_region}")
                noise_gate = noise_gate.to(device=z_T.device, dtype=torch.float32)
                if noise_gate.shape[-2:] != z_T.shape[-2:]:
                    noise_gate = torch.nn.functional.interpolate(
                        noise_gate,
                        size=z_T.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                noise_gate = (float(edit_initial_noise_scale) * noise_gate).clamp(0.0, 1.0)
                edit_noise = torch.randn_like(z_T, dtype=torch.float32)
                z_T = ((1.0 - noise_gate) * z_T.to(torch.float32) + noise_gate * edit_noise).to(z_T.dtype)
        m_subject = M_edit.squeeze()
        m_core = M_core.squeeze() if M_core is not None else None
        print(
            f"[mask] min={m_subject.min():.3f} "
            f"max={m_subject.max():.3f} "
            f"mean={m_subject.mean():.3f} "
            f"edit(>0.5)={(m_subject > 0.5).float().mean():.2%} "
            f"preserve(<0.2)={(M_preserve.squeeze() < 0.2).float().mean():.2%}"
            + (
                ""
                if m_core is None
                else f" core(>0.5)={(m_core > 0.5).float().mean():.2%}"
            )
        )
    if edit_clip_guidance_scale > 0.0 or edit_text_guidance_scale > 0.0:
        # Offloaded diffusers modules wrap the pipeline VAE with accelerate
        # hooks that produce inference tensors. A dedicated copy keeps the VAE
        # decode path differentiable for CLIP-based edit rewards.
        grad_vae = copy.deepcopy(pipe.vae).to(device)
        grad_vae.eval()
        for param in grad_vae.parameters():
            param.requires_grad_(False)
        clip_reward = LocalCLIPTextReward(device=device)
        with torch.no_grad():
            source_image = decode_latent_to_unit_image(pipe, x_src, vae_override=grad_vae)
            clip_reference = clip_reward.prepare_reference(
                source_image=source_image,
                source_prompt=src_prompt,
                target_prompt=tar_prompt,
                mask=M_edit,
            )

    # --- Controlled reverse ODE ---
    z_t = z_T.clone()
    step_stats: list[dict[str, float]] = []

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        sigma = t_i.clamp(min=1e-4)
        alpha_t = get_schedule_value(alpha_schedule, i, len(timesteps), alpha_max)
        if rec_stop_timestep > 0.0 and float(t_i.item()) < rec_stop_timestep:
            alpha_t = 0.0
        beta_t = get_schedule_value(beta_schedule, i, len(timesteps), beta_max)
        if len(timesteps) <= 1:
            edit_stage_progress = 1.0
        else:
            edit_stage_progress = i / float(len(timesteps) - 1)
        identity_break_weight = max(0.0, 1.0 - (edit_stage_progress / max(identity_break_stop, 1e-6)))
        if edit_stage_progress <= target_attract_start:
            target_attract_weight = 0.0
        else:
            target_attract_weight = min(
                1.0,
                (edit_stage_progress - target_attract_start) / max(1.0 - target_attract_start, 1e-6),
            )
        latents_dtype = z_t.dtype

        # Base prior: source-conditioned reference field, closer to h-edit's rec_term.
        with torch.no_grad():
            v_tar = calc_cfg_v_sd3(
                pipe=pipe,
                latents=z_t,
                negative_prompt_embeds=tar_negative_prompt_embeds,
                prompt_embeds=tar_prompt_embeds,
                negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                guidance_scale=tar_guidance_scale,
                t=t,
            )
            v_src = calc_cfg_v_sd3(
                pipe=pipe,
                latents=z_t,
                negative_prompt_embeds=src_negative_prompt_embeds,
                prompt_embeds=src_prompt_embeds,
                negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                pooled_prompt_embeds=src_pooled_prompt_embeds,
                guidance_scale=base_guidance_scale,
                t=t,
            )

        v_tar = v_tar.to(torch.float32)
        v_src = v_src.to(torch.float32)
        v_base = v_src
        x0_tar = predict_x0_from_linear_rf_path(z_t, v_tar, t_i).to(torch.float32)
        x0_src_step = predict_x0_from_linear_rf_path(z_t, v_src, t_i).to(torch.float32)
        base_edit_norm = v_base.norm().item()

        # Reconstruction guidance: its own explicit path from x_hat_0 -> E_rec -> v_rec.
        rec_energy_terms = None
        rec_terms = None
        rec_guidance_norm = 0.0
        current_rec_feature_map = None
        source_rec_feature_map = None
        if alpha_t != 0.0 and rec_guidance_scale > 0.0:
            step_key = int(t.item())
            if struct_guidance_scale != 0.0:
                if step_key not in source_feature_maps:
                    with torch.no_grad():
                        source_feature_maps[step_key] = extract_sd3_feature_structure_map(
                            pipe=pipe,
                            x_latent=x_src,
                            prompt_embeds=src_prompt_embeds,
                            pooled_embeds=src_pooled_prompt_embeds,
                            t=t,
                            detach=True,
                        ).to(dtype=torch.float32, device=z_t.device)
                source_rec_feature_map = source_feature_maps[step_key]
                with torch.no_grad():
                    current_rec_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_src_step,
                        prompt_embeds=src_prompt_embeds,
                        pooled_embeds=src_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)

            rec_terms = reconstruction_velocity_surrogate_total(
                x0_pred=x0_src_step,
                x_src=x_src.to(torch.float32),
                t_scalar=t_i,
                M_preserve=None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device),
                current_feature_map=current_rec_feature_map,
                source_feature_map=source_rec_feature_map,
                lambda_latent=1.0,
                lambda_struct=0.0,
                lambda_feature=struct_guidance_scale,
                velocity_conversion_mode=velocity_conversion_mode,
                velocity_t_min=linear_path_t_min,
            )
            rec_energy_terms = reconstruction_energy_total(
                x0_pred=x0_src_step,
                x_src=x_src.to(torch.float32),
                M_preserve=None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device),
                current_feature_map=current_rec_feature_map,
                source_feature_map=source_rec_feature_map,
                lambda_latent=1.0,
                lambda_struct=0.0,
                lambda_feature=struct_guidance_scale,
            )
            v_rec = alpha_t * rec_terms["total"].to(torch.float32)
            rec_guidance_norm = v_rec.norm().item()
        else:
            v_rec = torch.zeros_like(v_base)

        # Editing guidance: its own explicit path from x_hat_0 -> E_edit -> v_edit.
        edit_energy_terms = None
        v_edit_terms = None
        edit_guidance_norm = 0.0
        clip_edit_energy_terms = None
        clip_guidance_norm = 0.0
        tv_edit_energy = None
        text_edit_energy_terms = None
        text_guidance_norm = 0.0
        dds_guidance_norm = 0.0
        dds_energy_terms = None
        app_guidance_norm = 0.0
        app_edit_energy = None

        if beta_t != 0.0 and (
            edit_hedit_guidance_scale > 0.0
            or edit_guidance_scale > 0.0
            or edit_region_guidance_scale > 0.0
            or edit_target_guidance_scale > 0.0
            or edit_source_guidance_scale > 0.0
        ):
            with torch.no_grad():
                target_feature_map = None
                source_feature_map = None
                if edit_target_guidance_scale > 0.0:
                    target_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_tar,
                        prompt_embeds=tar_prompt_embeds,
                        pooled_embeds=tar_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)
                if edit_source_guidance_scale > 0.0:
                    source_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_src_step,
                        prompt_embeds=src_prompt_embeds,
                        pooled_embeds=src_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)
                v_src_edit = calc_cfg_v_sd3(
                    pipe,
                    z_t.to(latents_dtype),
                    src_negative_prompt_embeds,
                    src_prompt_embeds,
                    src_negative_pooled_prompt_embeds,
                    src_pooled_prompt_embeds,
                    edit_src_cfg_scale,
                    t,
                ).to(torch.float32)
                v_edit_terms = editing_velocity_surrogate_total(
                    base_edit_velocity=(v_tar.to(torch.float32) - v_src_edit),
                    x0_tar=x0_tar,
                    x0_src=x0_src_step,
                    t_scalar=t_i,
                    M_edit=None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device),
                    target_feature_map=target_feature_map,
                    source_feature_map=source_feature_map,
                    lambda_base=edit_hedit_guidance_scale,
                    lambda_anchor=edit_guidance_scale,
                    lambda_region=edit_region_guidance_scale,
                    lambda_target=edit_target_guidance_scale,
                    lambda_source=edit_source_guidance_scale,
                    velocity_t_min=linear_path_t_min,
                )
                edit_energy_terms = editing_energy_total(
                    x0_tar=x0_tar,
                    x0_src=x0_src_step,
                    M_edit=None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device),
                    target_feature_map=target_feature_map,
                    source_feature_map=source_feature_map,
                    lambda_anchor=edit_guidance_scale,
                    lambda_region=edit_region_guidance_scale,
                    lambda_target=edit_target_guidance_scale,
                    lambda_source=edit_source_guidance_scale,
                )
            edit_guidance = beta_t * v_edit_terms["total"].to(torch.float32)
            edit_guidance_norm = edit_guidance.norm().item()
        else:
            edit_guidance = torch.zeros_like(v_base)

        clip_start_idx = int(clip_start_timestep * len(timesteps))
        clip_stop_idx = min(int(clip_stop_timestep * len(timesteps)), len(timesteps))
        clip_window_active = clip_start_idx <= i < clip_stop_idx
        clip_guidance = torch.zeros_like(v_base)
        text_guidance = torch.zeros_like(v_base)
        dds_guidance = torch.zeros_like(v_base)
        if (
            beta_t != 0.0
            and clip_reward is not None
            and clip_reference is not None
            and (
                (edit_clip_guidance_scale > 0.0 and clip_window_active)
                or edit_text_guidance_scale > 0.0
            )
        ):
            zt_for_edit = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_tar_clip = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_edit,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                x0_tar_clip = predict_x0_from_linear_rf_path(zt_for_edit, v_tar_clip, t_i)
                current_image = decode_latent_to_unit_image(pipe, x0_tar_clip, vae_override=grad_vae)

                if edit_text_guidance_scale > 0.0:
                    text_edit_energy_terms = clip_reward.semantic_text_loss(
                        current_image=current_image,
                        reference=clip_reference,
                        core_mask=M_core,
                        subject_mask=M_edit,
                        source_scale=edit_text_source_scale,
                        core_weight=edit_text_core_weight,
                        subject_weight=edit_text_subject_weight,
                    )
                    grad_text = torch.autograd.grad(
                        text_edit_energy_terms["total"],
                        zt_for_edit,
                        retain_graph=edit_clip_guidance_scale > 0.0 and clip_window_active,
                    )[0].detach().to(torch.float32)
                    grad_text = smooth_guidance_field(grad_text, kernel_size=5)
                    grad_text_rms = grad_text.square().mean().sqrt() + 1e-8
                    neg_grad_text = -(grad_text / grad_text_rms)
                    text_guidance = (
                        beta_t
                        * edit_text_guidance_scale
                        * identity_break_weight
                        * neg_grad_text
                    )
                    text_guidance_norm = text_guidance.norm().item()

                if edit_clip_guidance_scale > 0.0 and clip_window_active:
                    clip_edit_energy_terms = clip_reward.strong_directional_text_loss(
                        current_image=current_image,
                        reference=clip_reference,
                        mask=M_edit,
                    )
                    if edit_image_tv_scale > 0.0:
                        tv_edit_energy = masked_total_variation(current_image, mask=M_edit)
                    clip_total_energy = clip_edit_energy_terms["total"]
                    if tv_edit_energy is not None:
                        clip_total_energy = clip_total_energy + edit_image_tv_scale * tv_edit_energy
                    grad_clip = torch.autograd.grad(clip_total_energy, zt_for_edit)[0].detach().to(torch.float32)
                    grad_clip = smooth_guidance_field(grad_clip, kernel_size=5)
                    grad_clip_rms = grad_clip.square().mean().sqrt() + 1e-8
                    neg_grad_clip = -(grad_clip / grad_clip_rms)
                    if clip_stop_idx - clip_start_idx <= 1:
                        clip_progress = 0.0
                    else:
                        clip_progress = (i - clip_start_idx) / float(clip_stop_idx - clip_start_idx - 1)
                    clip_window_scale = 0.5 * (
                        1.0 + torch.cos(torch.tensor(clip_progress * 3.141592653589793, device=device))
                    ).item()
                    clip_scale = (
                        beta_t
                        * edit_clip_guidance_scale
                        * clip_window_scale
                        * target_attract_weight
                    )
                    if edit_clip_match_base_scale > 0.0:
                        clip_scale = clip_scale + edit_clip_match_base_scale * base_edit_norm * clip_window_scale
                    clip_guidance = clip_scale * neg_grad_clip
                    clip_guidance_norm = clip_guidance.norm().item()

        if beta_t != 0.0 and edit_dds_guidance_scale > 0.0:
            zt_for_dds = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_tar_dds = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_dds,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                x0_tar_dds = predict_x0_from_linear_rf_path(zt_for_dds, v_tar_dds, t_i)
                noise = torch.randn_like(x0_tar_dds)
                noisy_x = scale_noise(scheduler, x0_tar_dds, t, noise=noise)
                v_tar_noisy = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=noisy_x,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                v_src_noisy = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=noisy_x,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=edit_src_cfg_scale,
                    t=t,
                )
                dds_target = 0.5 * (v_tar_noisy - noise).pow(2)
                dds_source = 0.5 * (v_src_noisy - noise).pow(2)
                if M_edit is not None:
                    edit_mask = M_edit.to(dtype=torch.float32, device=z_t.device)
                    dds_target = dds_target * edit_mask
                    dds_source = dds_source * edit_mask
                loss_target = dds_target.mean()
                loss_source = dds_source.mean()
                dds_total = loss_target - edit_dds_source_scale * loss_source
                dds_energy_terms = {
                    "target": loss_target.detach(),
                    "source": loss_source.detach(),
                    "total": dds_total.detach(),
                }
            grad_dds = torch.autograd.grad(dds_total, zt_for_dds)[0].detach().to(torch.float32)
            grad_dds = smooth_guidance_field(grad_dds, kernel_size=5)
            grad_dds_rms = grad_dds.square().mean().sqrt() + 1e-8
            neg_grad_dds = -(grad_dds / grad_dds_rms)
            dds_guidance = beta_t * edit_dds_guidance_scale * neg_grad_dds
            dds_guidance_norm = dds_guidance.norm().item()

        app_guidance = torch.zeros_like(v_base)
        if beta_t != 0.0 and edit_app_guidance_scale > 0.0 and M_edit is not None:
            zt_for_app = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_tar_app = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_app,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                )
                x0_tar_app = predict_x0_from_linear_rf_path(zt_for_app, v_tar_app, t_i)
                src_changed_words = [w for w in src_prompt.lower().split() if w not in tar_prompt.lower().split()]
                tar_changed_words = [w for w in tar_prompt.lower().split() if w not in src_prompt.lower().split()]
                step_key = int(t.item())
                if step_key not in source_attention_maps:
                    with torch.no_grad():
                        source_attention_maps[step_key] = extract_prompt_attention_map(
                            pipe=pipe,
                            x_latent=x0_src_step,
                            prompt=src_prompt,
                            prompt_embeds=src_prompt_embeds,
                            pooled_embeds=src_pooled_prompt_embeds,
                            t=t,
                            token_words=src_changed_words,
                            detach=True,
                        ).to(dtype=torch.float32, device=z_t.device)
                source_app_attention = source_attention_maps[step_key]
                current_app_attention = extract_prompt_attention_map(
                    pipe=pipe,
                    x_latent=x0_tar_app,
                    prompt=tar_prompt,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_embeds=tar_pooled_prompt_embeds,
                    t=t,
                    token_words=tar_changed_words,
                    detach=False,
                ).to(dtype=torch.float32, device=z_t.device)
                subject_gate = M_edit.to(dtype=torch.float32, device=z_t.device)
                if M_core is not None:
                    core_gate = M_core.to(dtype=torch.float32, device=z_t.device)
                    subject_gate = (subject_gate - core_gate).clamp_min(0.0)
                app_diff = (current_app_attention - source_app_attention) * subject_gate
                app_edit_energy = 0.5 * app_diff.pow(2).mean()
            grad_app = torch.autograd.grad(app_edit_energy, zt_for_app)[0].detach().to(torch.float32)
            grad_app = smooth_guidance_field(grad_app, kernel_size=5)
            grad_app_rms = grad_app.square().mean().sqrt() + 1e-8
            neg_grad_app = -(grad_app / grad_app_rms)
            app_guidance = beta_t * edit_app_guidance_scale * neg_grad_app
            app_guidance_norm = app_guidance.norm().item()

        rec_gate = None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device)
        edit_gate = None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device)
        core_gate = None if M_core is None else M_core.to(dtype=torch.float32, device=z_t.device)
        if rec_gate is not None:
            v_rec = v_rec * rec_gate
            rec_guidance_norm = v_rec.norm().item()
        if edit_gate is not None:
            if core_gate is not None:
                subject_ring = (edit_gate - core_gate).clamp_min(0.0)
                spatial_edit_weight = edit_core_scale * core_gate + edit_subject_scale * subject_ring
            else:
                spatial_edit_weight = edit_gate
            edit_guidance = edit_guidance * spatial_edit_weight
            clip_guidance = clip_guidance * spatial_edit_weight
            text_guidance = text_guidance * spatial_edit_weight
            dds_guidance = dds_guidance * spatial_edit_weight
            app_guidance = app_guidance * edit_gate
            edit_guidance_norm = edit_guidance.norm().item()
            clip_guidance_norm = clip_guidance.norm().item()
            text_guidance_norm = text_guidance.norm().item()
            dds_guidance_norm = dds_guidance.norm().item()
            app_guidance_norm = app_guidance.norm().item()

        v_edit_total = edit_guidance + clip_guidance + text_guidance + dds_guidance + app_guidance
        if edit_bound_scale > 0.0:
            edit_rms = v_edit_total.square().mean().sqrt().clamp_min(1e-8)
            v_edit_total = beta_t * edit_bound_scale * (v_edit_total / edit_rms)
        v_total = v_base + v_rec + v_edit_total
        total_velocity_norm = v_total.norm().item()
        next_z_t = z_t.to(torch.float32) + (t_im1 - t_i).to(torch.float32) * v_total
        trajectory_preserve_norm = 0.0
        trajectory_preserve_weight = 0.0
        if trajectory_preserve_scale > 0.0 or trajectory_subject_preserve_scale > 0.0:
            next_timestep_key = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0
            source_next = source_trajectory_by_timestep.get(next_timestep_key)
            if source_next is not None:
                trajectory_gate = torch.zeros(
                    next_z_t.shape[0],
                    1,
                    next_z_t.shape[-2],
                    next_z_t.shape[-1],
                    device=next_z_t.device,
                    dtype=torch.float32,
                )
                if rec_gate is None:
                    preserve_gate = torch.ones_like(trajectory_gate)
                else:
                    preserve_gate = rec_gate.to(device=next_z_t.device, dtype=torch.float32)
                    if preserve_gate.shape[-2:] != next_z_t.shape[-2:]:
                        preserve_gate = torch.nn.functional.interpolate(
                            preserve_gate,
                            size=next_z_t.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                if trajectory_preserve_scale > 0.0:
                    trajectory_gate = trajectory_gate + float(trajectory_preserve_scale) * preserve_gate
                if trajectory_subject_preserve_scale > 0.0 and edit_gate is not None and core_gate is not None:
                    subject_gate_for_traj = edit_gate.to(device=next_z_t.device, dtype=torch.float32)
                    core_gate_for_traj = core_gate.to(device=next_z_t.device, dtype=torch.float32)
                    if subject_gate_for_traj.shape[-2:] != next_z_t.shape[-2:]:
                        subject_gate_for_traj = torch.nn.functional.interpolate(
                            subject_gate_for_traj,
                            size=next_z_t.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        core_gate_for_traj = torch.nn.functional.interpolate(
                            core_gate_for_traj,
                            size=next_z_t.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    subject_ring_for_traj = (subject_gate_for_traj - core_gate_for_traj).clamp_min(0.0)
                    trajectory_gate = (
                        trajectory_gate
                        + float(trajectory_subject_preserve_scale) * subject_ring_for_traj
                    )
                trajectory_weight = float(trajectory_gate.max().item())
                source_next = source_next.to(device=next_z_t.device, dtype=torch.float32)
                trajectory_delta = trajectory_gate.clamp(0.0, 1.0) * (source_next - next_z_t)
                next_z_t = next_z_t + trajectory_delta
                trajectory_preserve_norm = trajectory_delta.norm().item()
                trajectory_preserve_weight = trajectory_weight
        edit_base_for_cos = torch.zeros_like(v_base) if v_edit_terms is None else v_edit_terms["base"]
        edit_anchor_for_cos = torch.zeros_like(v_base) if v_edit_terms is None else v_edit_terms["anchor"]
        edit_region_for_cos = torch.zeros_like(v_base) if v_edit_terms is None else v_edit_terms["region"]
        step_stat = {
            "step": float(i),
            "t": float(t_i.item()),
            "sigma": float(sigma.item()),
            "alpha_max": float(alpha_max),
            "alpha_t": float(alpha_t),
            "beta_max": float(beta_max),
            "beta_t": float(beta_t),
            "velocity_conversion_mode": velocity_conversion_mode,
            "src_guidance_scale": float(src_guidance_scale),
            "inversion_guidance_scale": float(inversion_guidance_scale),
            "base_guidance_scale": float(base_guidance_scale),
            "tar_guidance_scale": float(tar_guidance_scale),
            "edit_src_cfg_scale": float(edit_src_cfg_scale),
            "linear_path_t_min": float(linear_path_t_min),
            "rec_stop_timestep": float(rec_stop_timestep),
            "trajectory_preserve_scale": float(trajectory_preserve_scale),
            "trajectory_subject_preserve_scale": float(trajectory_subject_preserve_scale),
            "edit_initial_noise_scale": float(edit_initial_noise_scale),
            "edit_initial_noise_region": edit_initial_noise_region,
            "trajectory_preserve_weight": float(trajectory_preserve_weight),
            "attention_mask_mode": attention_mask_mode,
            "attention_mask_subject_threshold": float(attention_mask_subject_threshold),
            "attention_mask_core_threshold": float(attention_mask_core_threshold),
            "edit_mask_dilate_kernel": float(edit_mask_dilate_kernel),
            "edit_mask_smooth_kernel": float(edit_mask_smooth_kernel),
            "edit_mask_component_threshold": float(edit_mask_component_threshold),
            "edit_mask_keep_components": float(edit_mask_keep_components),
            "edit_mask_component_y_min": None
            if edit_mask_component_y_min is None
            else float(edit_mask_component_y_min),
            "edit_mask_component_y_max": None
            if edit_mask_component_y_max is None
            else float(edit_mask_component_y_max),
            "edit_mask_shift_y": float(edit_mask_shift_y),
            "edit_mask_shift_x": float(edit_mask_shift_x),
            "edit_mask_box_mode": edit_mask_box_mode,
            "external_edit_mask_path": external_edit_mask_path,
            "external_edit_mask_mode": external_edit_mask_mode,
            "edit_hedit_guidance_scale": float(edit_hedit_guidance_scale),
            "edit_guidance_scale": float(edit_guidance_scale),
            "edit_region_guidance_scale": float(edit_region_guidance_scale),
            "edit_clip_guidance_scale": float(edit_clip_guidance_scale),
            "edit_text_guidance_scale": float(edit_text_guidance_scale),
            "base_edit_norm": float(base_edit_norm),
            "edit_guidance_norm": float(edit_guidance_norm),
            "clip_guidance_norm": float(clip_guidance_norm),
            "text_guidance_norm": float(text_guidance_norm),
            "dds_guidance_norm": float(dds_guidance_norm),
            "app_guidance_norm": float(app_guidance_norm),
            "rec_guidance_norm": float(rec_guidance_norm),
            "trajectory_preserve_norm": float(trajectory_preserve_norm),
            "total_velocity_norm": float(total_velocity_norm),
            "rec_energy": 0.0 if rec_energy_terms is None else float(rec_energy_terms["latent"].item()),
            "struct_energy": 0.0 if rec_energy_terms is None else float(rec_energy_terms["feature"].item()),
            "edit_anchor_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["anchor"].item()),
            "edit_region_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["region"].item()),
            "edit_target_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["target"].item()),
            "edit_source_energy": 0.0 if edit_energy_terms is None else float(edit_energy_terms["source"].item()),
            "edit_base_norm": 0.0 if v_edit_terms is None else float(v_edit_terms["base"].norm().item()),
            "edit_tv_energy": 0.0 if tv_edit_energy is None else float(tv_edit_energy.item()),
            "clip_target_energy": 0.0 if clip_edit_energy_terms is None else float(clip_edit_energy_terms["target"].item()),
            "clip_source_energy": 0.0 if clip_edit_energy_terms is None else float(clip_edit_energy_terms["source"].item()),
            "clip_direction_energy": 0.0 if clip_edit_energy_terms is None else float(clip_edit_energy_terms["direction"].item()),
            "text_target_core_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["target_core"].item()),
            "text_source_core_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["source_core"].item()),
            "text_target_subject_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["target_subject"].item()),
            "text_source_subject_energy": 0.0
            if text_edit_energy_terms is None
            else float(text_edit_energy_terms["source_subject"].item()),
            "dds_target_energy": 0.0
            if dds_energy_terms is None
            else float(dds_energy_terms["target"].item()),
            "dds_source_energy": 0.0
            if dds_energy_terms is None
            else float(dds_energy_terms["source"].item()),
            "app_energy": 0.0 if app_edit_energy is None else float(app_edit_energy.item()),
            "edit_anchor_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["anchor"].norm().item()),
            "edit_region_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["region"].norm().item()),
            "edit_target_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["target"].norm().item()),
            "edit_source_velocity_norm": 0.0
            if v_edit_terms is None
            else float(v_edit_terms["source"].norm().item()),
            "cos_base_anchor": cosine_safe(edit_base_for_cos, edit_anchor_for_cos),
            "cos_base_region": cosine_safe(edit_base_for_cos, edit_region_for_cos),
            "cos_anchor_region": cosine_safe(edit_anchor_for_cos, edit_region_for_cos),
            "cos_rec_base": cosine_safe(v_rec, v_base),
            "cos_rec_edit_total": cosine_safe(v_rec, v_edit_total),
        }
        step_stats.append(step_stat)

        if log_every > 0 and (i == 0 or (i + 1) % log_every == 0 or i == len(timesteps) - 1):
            print(
                "[ode] "
                f"step={i:02d} "
                f"t={step_stat['t']:.4f} "
                f"alpha={alpha_t:.3f} "
                f"beta={beta_t:.3f} "
                f"|v_base|={step_stat['base_edit_norm']:.4f} "
                f"|v_edit|={step_stat['edit_guidance_norm']:.4f} "
                f"|v_clip|={step_stat['clip_guidance_norm']:.4f} "
                f"|v_text|={step_stat['text_guidance_norm']:.4f} "
                f"|v_dds|={step_stat['dds_guidance_norm']:.4f} "
                f"|v_app|={step_stat['app_guidance_norm']:.4f} "
                f"|v_rec|={step_stat['rec_guidance_norm']:.4f} "
                f"|traj|={step_stat['trajectory_preserve_norm']:.4f} "
                f"|v_total|={step_stat['total_velocity_norm']:.4f} "
                f"E_rec={step_stat['rec_energy']:.6f} "
                f"E_struct={step_stat['struct_energy']:.6f} "
                f"E_anchor={step_stat['edit_anchor_energy']:.6f} "
                f"E_region={step_stat['edit_region_energy']:.6f} "
                    f"E_target={step_stat['edit_target_energy']:.6f} "
                    f"E_source={step_stat['edit_source_energy']:.6f} "
                    f"|v_hedit|={step_stat['edit_base_norm']:.4f} "
                    f"E_tv={step_stat['edit_tv_energy']:.6f} "
                    f"E_clip_t={step_stat['clip_target_energy']:.6f} "
                    f"E_clip_s={step_stat['clip_source_energy']:.6f} "
                    f"E_clip_dir={step_stat['clip_direction_energy']:.6f} "
                    f"E_txt_tc={step_stat['text_target_core_energy']:.6f} "
                    f"E_txt_sc={step_stat['text_source_core_energy']:.6f} "
                    f"E_txt_ts={step_stat['text_target_subject_energy']:.6f} "
                    f"E_txt_ss={step_stat['text_source_subject_energy']:.6f} "
                    f"E_dds_t={step_stat['dds_target_energy']:.6f} "
                    f"E_dds_s={step_stat['dds_source_energy']:.6f} "
                    f"E_app={step_stat['app_energy']:.6f}"
            )

        z_t = next_z_t.to(torch.float32)
        if mask_blend and rec_gate is not None and preserve_blend_scale > 0.0:
            blend_start_idx = int(preserve_blend_start_timestep * len(timesteps))
            if i >= blend_start_idx:
                if len(timesteps) - blend_start_idx <= 1:
                    blend_progress = 1.0
                else:
                    blend_progress = (i - blend_start_idx) / float(len(timesteps) - blend_start_idx - 1)
                blend_weight = preserve_blend_scale * blend_progress
                z_t = (1.0 - blend_weight * rec_gate) * z_t + (blend_weight * rec_gate) * x0_src_step.to(torch.float32)
        z_t = z_t.to(latents_dtype)

    result = z_t

    if stats_output_path is not None:
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(step_stats, f, indent=2)
        print(f"[stats] saved ODE stats to {stats_output_path}")

    # Optional hard mask blend: paste source background over edited result.
    if mask_blend and M_preserve is not None:
        M_edit = (1.0 - M_preserve).to(dtype=result.dtype, device=result.device)
        x_src_aligned = x_src.to(dtype=result.dtype, device=result.device)
        result = M_edit * result + M_preserve.to(dtype=result.dtype) * x_src_aligned

    return result
