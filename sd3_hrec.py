from __future__ import annotations

import copy
import contextlib
import json
import os
import re
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
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
from generic_support import build_generic_support
from operation_support_v3 import (
    build_operation_support_v3,
    compute_clean_disagreement,
    compute_velocity_disagreement,
    save_support_debug,
)
from schedules import get_schedule_value


class _SD3SourceQKVInjectionController:
    def __init__(
        self,
        q_strength: float,
        k_strength: float,
        v_strength: float,
        layer_from: int,
        layer_to: int,
        spatial_gate: torch.Tensor | None = None,
    ):
        self.q_strength = float(q_strength)
        self.k_strength = float(k_strength)
        self.v_strength = float(v_strength)
        self.layer_from = int(layer_from)
        self.layer_to = int(layer_to)
        self.spatial_gate = spatial_gate
        self.mode = "store"
        self.queries: dict[int, torch.Tensor] = {}
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.injected_layers: set[int] = set()
        self.injected_q_layers: set[int] = set()
        self.injected_k_layers: set[int] = set()
        self.injected_v_layers: set[int] = set()

    def reset(self) -> None:
        self.queries.clear()
        self.keys.clear()
        self.values.clear()
        self.injected_layers.clear()
        self.injected_q_layers.clear()
        self.injected_k_layers.clear()
        self.injected_v_layers.clear()

    def layer_enabled(self, layer_index: int | None) -> bool:
        if layer_index is None or max(self.q_strength, self.k_strength, self.v_strength) <= 0.0:
            return False
        return self.layer_from <= layer_index < self.layer_to

    def store_qkv(self, layer_index: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        self.queries[layer_index] = query.detach()
        self.keys[layer_index] = key.detach()
        self.values[layer_index] = value.detach()

    def _match_batch(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        source = source.to(device=target.device, dtype=target.dtype)
        if source.shape[0] != target.shape[0]:
            if source.shape[0] == 1:
                source = source.expand(target.shape[0], -1, -1, -1)
            elif target.shape[0] % source.shape[0] == 0:
                repeat = target.shape[0] // source.shape[0]
                source = source.repeat_interleave(repeat, dim=0)
            else:
                return None
        if source.shape != target.shape:
            return None
        return source

    def _token_gate(self, target: torch.Tensor) -> torch.Tensor | None:
        if self.spatial_gate is None:
            return None
        token_count = target.shape[2]
        side = int(round(token_count ** 0.5))
        if side * side != token_count:
            return None
        gate = self.spatial_gate.to(device=target.device, dtype=target.dtype)
        gate = F.interpolate(gate, size=(side, side), mode="bilinear", align_corners=False)
        gate = gate.flatten(2).unsqueeze(1).transpose(2, 3)
        if gate.shape[0] != target.shape[0]:
            if gate.shape[0] == 1:
                gate = gate.expand(target.shape[0], -1, -1, -1)
            elif target.shape[0] % gate.shape[0] == 0:
                gate = gate.repeat_interleave(target.shape[0] // gate.shape[0], dim=0)
            else:
                return None
        return gate.clamp(0.0, 1.0)

    def add_query(self, layer_index: int, query: torch.Tensor) -> torch.Tensor:
        source_query = self.queries.get(layer_index)
        if source_query is None or self.q_strength <= 0.0:
            return query
        source_query = self._match_batch(source_query, query)
        if source_query is None:
            return query
        gate = self._token_gate(query)
        if gate is not None:
            source_query = source_query * gate
        self.injected_layers.add(layer_index)
        self.injected_q_layers.add(layer_index)
        return query + self.q_strength * source_query

    def add_key(self, layer_index: int, key: torch.Tensor) -> torch.Tensor:
        source_key = self.keys.get(layer_index)
        if source_key is None or self.k_strength <= 0.0:
            return key
        source_key = self._match_batch(source_key, key)
        if source_key is None:
            return key
        gate = self._token_gate(key)
        if gate is not None:
            source_key = source_key * gate
        self.injected_layers.add(layer_index)
        self.injected_k_layers.add(layer_index)
        return key + self.k_strength * source_key

    def blend_value(self, layer_index: int, value: torch.Tensor) -> torch.Tensor:
        source_value = self.values.get(layer_index)
        if source_value is None or self.v_strength <= 0.0:
            return value
        source_value = self._match_batch(source_value, value)
        if source_value is None:
            return value
        gate = self._token_gate(value)
        if gate is None:
            gate = self.v_strength
        else:
            gate = self.v_strength * gate
        self.injected_layers.add(layer_index)
        self.injected_v_layers.add(layer_index)
        return value * (1.0 - gate) + source_value * gate


class _SD3SourceQKVInjectProcessor:
    def __init__(self, controller: _SD3SourceQKVInjectionController, layer_index: int | None):
        self.controller = controller
        self.layer_index = layer_index

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.controller.layer_enabled(self.layer_index):
            assert self.layer_index is not None
            if self.controller.mode == "store":
                self.controller.store_qkv(self.layer_index, query, key, value)
            elif self.controller.mode == "inject":
                query = self.controller.add_query(self.layer_index, query)
                key = self.controller.add_key(self.layer_index, key)
                value = self.controller.blend_value(self.layer_index, value)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def _parse_sd3_attn_layer_index(processor_name: str) -> int | None:
    match = re.search(r"transformer_blocks\.(\d+)\.attn\.processor$", processor_name)
    if match is None:
        return None
    return int(match.group(1))


@contextlib.contextmanager
def _sd3_source_qkv_injection(transformer, controller: _SD3SourceQKVInjectionController):
    original_processors = dict(transformer.attn_processors)
    injected_processors = {
        name: _SD3SourceQKVInjectProcessor(controller, _parse_sd3_attn_layer_index(name))
        for name in original_processors
    }
    transformer.set_attn_processor(dict(injected_processors))
    try:
        yield controller
    finally:
        transformer.set_attn_processor(dict(original_processors))


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


@torch.no_grad()
def calc_cfg_v_sd3_with_source_qkv_injection(
    pipe,
    latents: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    t: torch.Tensor,
    source_latents: torch.Tensor,
    source_prompt_embeds: torch.Tensor,
    source_pooled_prompt_embeds: torch.Tensor,
    q_strength: float,
    k_strength: float,
    v_strength: float,
    layer_from: int,
    layer_to: int,
    spatial_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    controller = _SD3SourceQKVInjectionController(
        q_strength=q_strength,
        k_strength=k_strength,
        v_strength=v_strength,
        layer_from=layer_from,
        layer_to=layer_to,
        spatial_gate=spatial_gate,
    )
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    timestep_source = t.expand(source_latents.shape[0])
    source_latents = source_latents.to(device=latents.device, dtype=transformer_dtype)
    source_prompt_embeds = source_prompt_embeds.to(device=latents.device, dtype=transformer_dtype)
    source_pooled_prompt_embeds = source_pooled_prompt_embeds.to(device=latents.device, dtype=transformer_dtype)

    with _sd3_source_qkv_injection(pipe.transformer, controller):
        controller.mode = "store"
        controller.reset()
        pipe.transformer(
            hidden_states=source_latents,
            timestep=timestep_source,
            encoder_hidden_states=source_prompt_embeds,
            pooled_projections=source_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        controller.mode = "inject"
        velocity = calc_cfg_v_sd3(
            pipe=pipe,
            latents=latents,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            t=t,
        )
        injected_counts = {
            "all": len(controller.injected_layers),
            "q": len(controller.injected_q_layers),
            "k": len(controller.injected_k_layers),
            "v": len(controller.injected_v_layers),
        }

    return velocity, injected_counts


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


_COLOR_RGB = {
    "black": (0.02, 0.02, 0.02),
    "blue": (0.02, 0.35, 0.95),
    "brown": (0.45, 0.24, 0.08),
    "cyan": (0.0, 0.75, 0.95),
    "gold": (0.95, 0.72, 0.12),
    "golden": (0.95, 0.72, 0.12),
    "gray": (0.5, 0.5, 0.5),
    "green": (0.05, 0.65, 0.12),
    "grey": (0.5, 0.5, 0.5),
    "orange": (0.95, 0.42, 0.05),
    "pink": (0.95, 0.35, 0.65),
    "purple": (0.45, 0.12, 0.8),
    "red": (0.9, 0.05, 0.04),
    "silver": (0.75, 0.75, 0.72),
    "white": (0.95, 0.95, 0.92),
    "yellow": (0.95, 0.82, 0.05),
}


def _prompt_words(text: str) -> list[str]:
    token = []
    words = []
    for char in text.lower():
        if char.isalnum():
            token.append(char)
        elif token:
            words.append("".join(token))
            token = []
    if token:
        words.append("".join(token))
    return words


def infer_target_color_rgb(
    src_prompt: str,
    tar_prompt: str,
    explicit: str | None = None,
) -> tuple[str, torch.Tensor] | None:
    if explicit:
        key = explicit.strip().lower()
        if key not in _COLOR_RGB:
            raise ValueError(f"Unsupported target color: {explicit!r}")
        return key, torch.tensor(_COLOR_RGB[key], dtype=torch.float32)
    src_colors = {word for word in _prompt_words(src_prompt) if word in _COLOR_RGB}
    for word in _prompt_words(tar_prompt):
        if word in _COLOR_RGB and word not in src_colors:
            return word, torch.tensor(_COLOR_RGB[word], dtype=torch.float32)
    for word in _prompt_words(tar_prompt):
        if word in _COLOR_RGB:
            return word, torch.tensor(_COLOR_RGB[word], dtype=torch.float32)
    return None


def infer_source_color_rgb(
    src_prompt: str,
    explicit: str | None = None,
) -> tuple[str, torch.Tensor] | None:
    if explicit:
        key = explicit.strip().lower()
        if key not in _COLOR_RGB:
            raise ValueError(f"Unsupported source color: {explicit!r}")
        return key, torch.tensor(_COLOR_RGB[key], dtype=torch.float32)
    for word in _prompt_words(src_prompt):
        if word in _COLOR_RGB:
            return word, torch.tensor(_COLOR_RGB[word], dtype=torch.float32)
    return None


def source_color_similarity_mask(
    source_image: torch.Tensor,
    source_rgb: torch.Tensor,
    object_mask: torch.Tensor | None,
    threshold: float = 0.38,
    softness: float = 0.10,
    luma_gate_min: float = 0.0,
    luma_gate_softness: float = 0.08,
    detail_protect_scale: float = 0.0,
    detail_protect_threshold: float = 0.35,
    detail_protect_softness: float = 0.08,
) -> torch.Tensor:
    target = source_rgb.to(dtype=source_image.dtype, device=source_image.device).view(1, 3, 1, 1)
    dist = (source_image - target).square().sum(dim=1, keepdim=True).sqrt()
    softness = max(float(softness), 1e-4)
    color_weight = torch.sigmoid((float(threshold) - dist) / softness)
    source_y = None
    if luma_gate_min > 0.0:
        source_y = rgb_to_yuv(source_image)[:, :1]
        gate_softness = max(float(luma_gate_softness), 1e-4)
        luma_gate = torch.sigmoid((source_y - float(luma_gate_min)) / gate_softness)
        color_weight = color_weight * luma_gate
    if detail_protect_scale > 0.0:
        if source_y is None:
            source_y = rgb_to_yuv(source_image)[:, :1]
        grad_x = source_y[..., :, 1:] - source_y[..., :, :-1]
        grad_y = source_y[..., 1:, :] - source_y[..., :-1, :]
        edge = torch.zeros_like(source_y)
        edge[..., :, 1:] = edge[..., :, 1:] + grad_x.abs()
        edge[..., 1:, :] = edge[..., 1:, :] + grad_y.abs()
        edge = edge.float()
        flat = edge.flatten(1)
        lo = torch.quantile(flat, 0.10, dim=1).view(-1, 1, 1, 1)
        hi = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
        edge = ((edge - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        protect_softness = max(float(detail_protect_softness), 1e-4)
        protect = torch.sigmoid((edge - float(detail_protect_threshold)) / protect_softness)
        protect_scale = max(0.0, min(1.0, float(detail_protect_scale)))
        color_weight = color_weight * (1.0 - protect_scale * protect).clamp(0.0, 1.0)
    if object_mask is not None:
        obj = torch.nn.functional.interpolate(
            object_mask.to(dtype=source_image.dtype, device=source_image.device),
            size=source_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        color_weight = color_weight * obj
    return color_weight.clamp(0.0, 1.0)


def masked_mean_rgb_loss(
    image: torch.Tensor,
    target_rgb: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        weights = torch.ones(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=image.dtype,
        )
    else:
        weights = torch.nn.functional.interpolate(
            mask.to(dtype=image.dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    denom = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    mean_rgb = (image * weights).sum(dim=(2, 3)) / denom
    target = target_rgb.to(dtype=image.dtype, device=image.device).view(1, 3)
    return 0.5 * (mean_rgb - target).pow(2).mean()


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r = image[:, 0:1]
    g = image[:, 1:2]
    b = image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def masked_chroma_luma_loss(
    image: torch.Tensor,
    source_image: torch.Tensor,
    target_rgb: torch.Tensor,
    mask: torch.Tensor | None,
    target_chroma_scale: float = 1.0,
    luma_preserve_scale: float = 0.35,
    luma_gradient_preserve_scale: float = 0.0,
) -> torch.Tensor:
    if mask is None:
        weights = torch.ones(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=image.dtype,
        )
    else:
        weights = torch.nn.functional.interpolate(
            mask.to(dtype=image.dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    src = torch.nn.functional.interpolate(
        source_image.to(dtype=image.dtype, device=image.device),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    current_yuv = rgb_to_yuv(image)
    source_yuv = rgb_to_yuv(src)
    target_image = target_rgb.to(dtype=image.dtype, device=image.device).view(1, 3, 1, 1).expand_as(image)
    target_yuv = rgb_to_yuv(target_image)
    target_yuv = torch.cat(
        [
            target_yuv[:, :1],
            target_yuv[:, 1:] * max(0.0, float(target_chroma_scale)),
        ],
        dim=1,
    )
    denom = weights.sum().clamp_min(1e-6)
    chroma_loss = ((current_yuv[:, 1:] - target_yuv[:, 1:]).pow(2) * weights).sum() / denom
    luma_loss = ((current_yuv[:, :1] - source_yuv[:, :1]).pow(2) * weights).sum() / denom
    total = chroma_loss + float(luma_preserve_scale) * luma_loss
    if luma_gradient_preserve_scale > 0.0:
        current_y = current_yuv[:, :1]
        source_y = source_yuv[:, :1]
        grad_x = current_y[..., :, 1:] - current_y[..., :, :-1]
        src_grad_x = source_y[..., :, 1:] - source_y[..., :, :-1]
        grad_y = current_y[..., 1:, :] - current_y[..., :-1, :]
        src_grad_y = source_y[..., 1:, :] - source_y[..., :-1, :]
        weights_x = torch.minimum(weights[..., :, 1:], weights[..., :, :-1])
        weights_y = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
        denom_x = weights_x.sum().clamp_min(1e-6)
        denom_y = weights_y.sum().clamp_min(1e-6)
        grad_loss = (
            ((grad_x - src_grad_x).pow(2) * weights_x).sum() / denom_x
            + ((grad_y - src_grad_y).pow(2) * weights_y).sum() / denom_y
        )
        total = total + float(luma_gradient_preserve_scale) * grad_loss
    return 0.5 * total


def masked_reference_image_loss(
    image: torch.Tensor,
    reference_image: torch.Tensor,
    mask: torch.Tensor | None,
    structure_image: torch.Tensor | None = None,
    luma_preserve_scale: float = 0.35,
    gradient_preserve_scale: float = 0.0,
    darkness_guard_scale: float = 0.0,
    darkness_guard_margin: float = 0.03,
    chroma_mode: str = "yuv",
    chroma_magnitude_scale: float = 1.0,
) -> torch.Tensor:
    # Decode-space color losses can overflow in fp16 when a nearly gray pixel
    # is normalized into a chroma direction. Keep this small objective in fp32.
    loss_dtype = torch.float32
    image_f = image.to(dtype=loss_dtype)
    if mask is None:
        weights = torch.ones(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=loss_dtype,
        )
    else:
        weights = torch.nn.functional.interpolate(
            mask.to(dtype=loss_dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    ref = torch.nn.functional.interpolate(
        reference_image.to(dtype=loss_dtype, device=image.device),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    structure_ref = ref
    if structure_image is not None:
        structure_ref = torch.nn.functional.interpolate(
            structure_image.to(dtype=loss_dtype, device=image.device),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    current_yuv = rgb_to_yuv(image_f)
    reference_yuv = rgb_to_yuv(ref)
    structure_yuv = rgb_to_yuv(structure_ref)
    denom = weights.sum().clamp_min(1e-6)
    chroma_mode = str(chroma_mode).strip().lower()
    if chroma_mode == "yuv":
        chroma_loss = ((current_yuv[:, 1:] - reference_yuv[:, 1:]).pow(2) * weights).sum() / denom
    elif chroma_mode == "yuv_direction":
        cur_uv = current_yuv[:, 1:]
        ref_uv = reference_yuv[:, 1:]
        eps = 1e-2
        cur_norm = (cur_uv.square().sum(dim=1, keepdim=True) + eps * eps).sqrt()
        ref_norm = (ref_uv.square().sum(dim=1, keepdim=True) + eps * eps).sqrt()
        cur_dir = cur_uv / cur_norm
        ref_dir = ref_uv / ref_norm
        direction_loss = ((cur_dir - ref_dir).pow(2) * weights).sum() / denom
        mag_loss = ((cur_norm - ref_norm).pow(2) * weights).sum() / denom
        chroma_loss = direction_loss + max(0.0, float(chroma_magnitude_scale)) * mag_loss
    else:
        raise ValueError(f"Unsupported edit ref chroma mode: {chroma_mode}")
    luma_loss = ((current_yuv[:, :1] - structure_yuv[:, :1]).pow(2) * weights).sum() / denom
    total = chroma_loss + float(luma_preserve_scale) * luma_loss
    if gradient_preserve_scale > 0.0:
        current_y = current_yuv[:, :1]
        reference_y = structure_yuv[:, :1]
        grad_x = current_y[..., :, 1:] - current_y[..., :, :-1]
        ref_grad_x = reference_y[..., :, 1:] - reference_y[..., :, :-1]
        grad_y = current_y[..., 1:, :] - current_y[..., :-1, :]
        ref_grad_y = reference_y[..., 1:, :] - reference_y[..., :-1, :]
        weights_x = torch.minimum(weights[..., :, 1:], weights[..., :, :-1])
        weights_y = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
        denom_x = weights_x.sum().clamp_min(1e-6)
        denom_y = weights_y.sum().clamp_min(1e-6)
        grad_loss = (
            ((grad_x - ref_grad_x).pow(2) * weights_x).sum() / denom_x
            + ((grad_y - ref_grad_y).pow(2) * weights_y).sum() / denom_y
        )
        total = total + float(gradient_preserve_scale) * grad_loss
    if darkness_guard_scale > 0.0:
        margin = max(0.0, float(darkness_guard_margin))
        darkening = torch.relu(structure_yuv[:, :1] - current_yuv[:, :1] - margin)
        darkening_loss = (darkening.pow(2) * weights).sum() / denom
        total = total + float(darkness_guard_scale) * darkening_loss
    return 0.5 * total


def smooth_guidance_field(
    field: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    if kernel_size <= 1:
        return field
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(field, kernel_size=kernel_size, stride=1, padding=pad)


def suppress_low_frequency_guidance(
    field: torch.Tensor,
    kernel_size: int = 0,
    strength: float = 0.0,
) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or kernel_size <= 1:
        return field
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    low = torch.nn.functional.avg_pool2d(field, kernel_size=kernel_size, stride=1, padding=pad)
    return field - strength * low


def ramp_schedule_from_progress(
    progress: float,
    start: float = 0.0,
    stop: float = 0.0,
    power: float = 1.0,
) -> float:
    start = float(start)
    stop = float(stop)
    if stop <= start:
        return 1.0
    x = (float(progress) - start) / max(1e-6, stop - start)
    x = max(0.0, min(1.0, x))
    x = x * x * (3.0 - 2.0 * x)
    if power != 1.0:
        x = x ** max(1e-6, float(power))
    return float(x)


def remove_opposing_guidance_component(
    guidance: torch.Tensor,
    anchor: torch.Tensor | None,
    strength: float = 1.0,
) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or anchor is None:
        return guidance
    anchor = anchor.to(device=guidance.device, dtype=guidance.dtype)
    anchor_sq = anchor.square().sum().clamp_min(1e-8)
    dot = (guidance * anchor).sum()
    if float(dot.detach().item()) >= 0.0:
        return guidance
    opposing_projection = (dot / anchor_sq) * anchor
    return guidance - strength * opposing_projection


def masked_rms(
    tensor: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    value = tensor.detach().float()
    if mask is None:
        return value.square().mean().sqrt()
    gate = mask.detach().float().to(device=value.device)
    if gate.shape[-2:] != value.shape[-2:]:
        gate = torch.nn.functional.interpolate(
            gate,
            size=value.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(1)
    gate = gate.clamp(0.0, 1.0)
    denom = (gate.sum() * value.shape[1]).clamp_min(eps)
    return ((value.square() * gate).sum() / denom).sqrt()


def remove_masked_opposing_guidance_component(
    guidance: torch.Tensor,
    anchor: torch.Tensor | None,
    mask: torch.Tensor | None = None,
    strength: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or anchor is None:
        return guidance, 0.0, 0.0
    anchor = anchor.to(device=guidance.device, dtype=guidance.dtype)
    guidance_eval = guidance
    anchor_eval = anchor
    if mask is not None:
        gate = mask.to(device=guidance.device, dtype=guidance.dtype)
        if gate.shape[-2:] != guidance.shape[-2:]:
            gate = torch.nn.functional.interpolate(
                gate,
                size=guidance.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = gate.clamp(0.0, 1.0)
        guidance_eval = guidance_eval * gate
        anchor_eval = anchor_eval * gate
    anchor_sq = anchor_eval.square().sum().clamp_min(1e-8)
    dot = (guidance_eval * anchor_eval).sum()
    if float(dot.detach().item()) >= 0.0:
        return guidance, float(dot.detach().item()), 0.0
    opposing_projection = (dot / anchor_sq) * anchor
    if mask is not None:
        opposing_projection = opposing_projection * gate
    delta = strength * opposing_projection
    return guidance - delta, float(dot.detach().item()), float(delta.norm().detach().item())


def limit_guidance_rms_relative_to_anchor(
    guidance: torch.Tensor,
    anchor: torch.Tensor | None,
    max_ratio: float = 0.0,
) -> torch.Tensor:
    max_ratio = float(max_ratio)
    if max_ratio <= 0.0 or anchor is None:
        return guidance
    guidance_rms = guidance.square().mean().sqrt()
    anchor_rms = anchor.to(device=guidance.device, dtype=guidance.dtype).square().mean().sqrt()
    max_rms = max_ratio * anchor_rms.clamp_min(1e-8)
    scale = torch.minimum(torch.ones_like(guidance_rms), max_rms / guidance_rms.clamp_min(1e-8))
    return guidance * scale


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


def latent_structure_edge_mask(
    reference: torch.Tensor,
    threshold: float = 0.55,
    soften_kernel: int = 3,
) -> torch.Tensor:
    """
    Estimate strong source structure boundaries from latent-space gradients.

    This is weight-free and image-agnostic. It protects high-gradient source
    structure inside the contact ring, such as fur boundaries and identity
    markings, without requiring a hand-written exclude box.
    """
    ref = reference.detach().float()
    dx = ref[..., :, 1:] - ref[..., :, :-1]
    dy = ref[..., 1:, :] - ref[..., :-1, :]
    edge = torch.zeros(ref.shape[0], 1, ref.shape[-2], ref.shape[-1], device=ref.device, dtype=torch.float32)
    edge[..., :, 1:] = edge[..., :, 1:] + dx.abs().mean(dim=1, keepdim=True)
    edge[..., 1:, :] = edge[..., 1:, :] + dy.abs().mean(dim=1, keepdim=True)
    flat = edge.flatten(1)
    lo = torch.quantile(flat, 0.10, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
    edge = ((edge - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    threshold = max(0.0, min(1.0, float(threshold)))
    if threshold > 0.0:
        edge = ((edge - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
    if soften_kernel > 1:
        edge = smooth_spatial_mask(edge, kernel_size=soften_kernel).clamp(0.0, 1.0)
    return edge.to(dtype=reference.dtype)


def build_object_contact_masks(
    edit_mask: torch.Tensor,
    core_mask: torch.Tensor | None = None,
    structure_reference: torch.Tensor | None = None,
    object_threshold: float = 0.45,
    contact_dilate_kernel: int = 7,
    contact_scale: float = 0.25,
    contact_edge_threshold: float = 0.55,
    contact_edge_protect_scale: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Split a candidate edit support into object/contact/preserve layers.

    `object_mask` is the target object support where editing is strongest.
    `contact_mask` is a soft ring around the object that allows weak blending.
    `subject_mask` is the effective M_edit used by edit velocities.
    `preserve_mask` is everything outside the object/contact support.
    """
    base = edit_mask if core_mask is None else torch.minimum(core_mask, edit_mask)
    object_threshold = max(0.0, min(1.0, float(object_threshold)))
    if object_threshold > 0.0:
        object_mask = (base.detach().float() > object_threshold).to(dtype=edit_mask.dtype, device=edit_mask.device)
        object_mask = object_mask * edit_mask.clamp(0.0, 1.0)
    else:
        object_mask = base.clamp(0.0, 1.0)
    if float(object_mask.detach().float().max().item()) <= 1e-6:
        object_mask = base.clamp(0.0, 1.0)
    support_mask = dilate_spatial_mask(object_mask, kernel_size=contact_dilate_kernel).clamp(0.0, 1.0)
    contact_mask = (support_mask - object_mask).clamp(0.0, 1.0)
    edge_mask = None
    if structure_reference is not None and contact_edge_protect_scale > 0.0:
        edge_mask = latent_structure_edge_mask(
            structure_reference,
            threshold=contact_edge_threshold,
        )
        if edge_mask.shape[-2:] != contact_mask.shape[-2:]:
            edge_mask = torch.nn.functional.interpolate(
                edge_mask.float(),
                size=contact_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=contact_mask.dtype, device=contact_mask.device)
        protect = (float(contact_edge_protect_scale) * edge_mask).clamp(0.0, 1.0)
        contact_mask = (contact_mask * (1.0 - protect)).clamp(0.0, 1.0)
    subject_mask = (object_mask + float(contact_scale) * contact_mask).clamp(0.0, 1.0)
    preserve_mask = (1.0 - torch.maximum(object_mask, contact_mask)).clamp(0.0, 1.0)
    return subject_mask, object_mask, contact_mask, preserve_mask, edge_mask


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


def load_external_image_like(reference: torch.Tensor, image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = torch.from_numpy(np.array(image, dtype="float32") / 255.0).permute(2, 0, 1)
    tensor = array[None].to(device=reference.device, dtype=reference.dtype)
    if tensor.shape[-2:] != reference.shape[-2:]:
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return tensor.expand(reference.shape[0], 3, reference.shape[-2], reference.shape[-1]).clamp(0.0, 1.0)


def save_mask_image(mask: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = mask.detach().float()[0, 0].clamp(0.0, 1.0)
    array = (image.cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def spatial_mask_stats(mask: torch.Tensor | None, prefix: str = "mask") -> dict[str, float | None]:
    if mask is None:
        return {
            f"{prefix}_area_ratio": 0.0,
            f"{prefix}_soft_mean": 0.0,
            f"{prefix}_soft_max": 0.0,
            f"{prefix}_center_x": None,
            f"{prefix}_center_y": None,
            f"{prefix}_bbox_x0": None,
            f"{prefix}_bbox_y0": None,
            f"{prefix}_bbox_x1": None,
            f"{prefix}_bbox_y1": None,
        }

    image = mask.detach().float()
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 3:
        image = image[0]
    image = image.clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    binary = image > 0.5
    area_ratio = float(binary.float().mean().item())
    soft_mean = float(image.mean().item())
    soft_max = float(image.max().item())

    mass = image.sum()
    if float(mass.item()) <= 1e-8:
        center_x = center_y = None
    else:
        ys = torch.linspace(0.0, 1.0, h, device=image.device, dtype=image.dtype).view(h, 1)
        xs = torch.linspace(0.0, 1.0, w, device=image.device, dtype=image.dtype).view(1, w)
        center_x = float((image * xs).sum().div(mass).item())
        center_y = float((image * ys).sum().div(mass).item())

    if bool(binary.any().item()):
        coords = torch.nonzero(binary, as_tuple=False)
        y0 = float(coords[:, 0].min().item() / max(1, h - 1))
        y1 = float(coords[:, 0].max().item() / max(1, h - 1))
        x0 = float(coords[:, 1].min().item() / max(1, w - 1))
        x1 = float(coords[:, 1].max().item() / max(1, w - 1))
    else:
        x0 = y0 = x1 = y1 = None

    return {
        f"{prefix}_area_ratio": area_ratio,
        f"{prefix}_soft_mean": soft_mean,
        f"{prefix}_soft_max": soft_max,
        f"{prefix}_center_x": center_x,
        f"{prefix}_center_y": center_y,
        f"{prefix}_bbox_x0": x0,
        f"{prefix}_bbox_y0": y0,
        f"{prefix}_bbox_x1": x1,
        f"{prefix}_bbox_y1": y1,
    }


def _clamp_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    return x0, y0, x1, y1


def _expand_box(
    box: tuple[float, float, float, float],
    pad_x: float,
    pad_y_top: float,
    pad_y_bottom: float | None = None,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _clamp_box(box)
    if pad_y_bottom is None:
        pad_y_bottom = pad_y_top
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    target_width = max(width + 2.0 * pad_x, min_width)
    target_height = max(height + pad_y_top + pad_y_bottom, min_height)
    expanded = (
        cx - 0.5 * target_width,
        cy - 0.5 * target_height,
        cx + 0.5 * target_width,
        cy + 0.5 * target_height,
    )
    return _clamp_box(expanded)


def _box_from_mask(
    mask: torch.Tensor | None,
    threshold: float = 0.35,
    fallback: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    if mask is None:
        return fallback
    image = mask.detach().float()
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 3:
        image = image[0]
    image = image.clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    binary = image > threshold
    if not bool(binary.any().item()):
        return fallback
    coords = torch.nonzero(binary, as_tuple=False)
    y0 = float(coords[:, 0].min().item() / max(1, h - 1))
    y1 = float(coords[:, 0].max().item() / max(1, h - 1))
    x0 = float(coords[:, 1].min().item() / max(1, w - 1))
    x1 = float(coords[:, 1].max().item() / max(1, w - 1))
    return _clamp_box((x0, y0, x1, y1))


def _mask_binary_area_ratio(mask: torch.Tensor | None, threshold: float = 0.5) -> float:
    if mask is None:
        return 0.0
    image = mask.detach().float()
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 3:
        image = image[0]
    return float((image.clamp(0.0, 1.0) > threshold).float().mean().item())


def _largest_component_box_from_mask(
    mask: torch.Tensor | None,
    threshold: float = 0.5,
) -> tuple[float, float, float, float] | None:
    if mask is None:
        return None
    image = mask.detach().float().cpu()
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 3:
        image = image[0]
    image = image.clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    binary = image > threshold
    visited = torch.zeros_like(binary, dtype=torch.bool)
    best_mass = -1.0
    best_points: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not bool(binary[y, x]) or bool(visited[y, x]):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            mass = 0.0
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                mass += float(image[cy, cx].item())
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if mass > best_mass:
                best_mass = mass
                best_points = points
    if not best_points:
        return None
    ys = [p[0] for p in best_points]
    xs = [p[1] for p in best_points]
    return _clamp_box(
        (
            float(min(xs) / max(1, w - 1)),
            float(min(ys) / max(1, h - 1)),
            float(max(xs) / max(1, w - 1)),
            float(max(ys) / max(1, h - 1)),
        )
    )


def _largest_component_mask_from_mask(
    mask: torch.Tensor | None,
    threshold: float = 0.5,
) -> torch.Tensor | None:
    if mask is None:
        return None
    image = mask.detach().float().cpu()
    original_ndim = image.ndim
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[:, None] if image.shape[0] != 1 else image[None]
    elif image.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D, or 4D mask, got shape {tuple(mask.shape)}")
    out = torch.zeros_like(image)
    bsz, _, h, w = image.shape
    threshold = max(0.0, min(1.0, float(threshold)))
    for b in range(bsz):
        plane = image[b, 0].clamp(0.0, 1.0)
        binary = plane > threshold
        visited = torch.zeros_like(binary, dtype=torch.bool)
        best_mass = -1.0
        best_points: list[tuple[int, int]] = []
        for y in range(h):
            for x in range(w):
                if not bool(binary[y, x]) or bool(visited[y, x]):
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                mass = 0.0
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    mass += float(plane[cy, cx].item())
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                if mass > best_mass:
                    best_mass = mass
                    best_points = points
        for py, px in best_points:
            out[b, 0, py, px] = image[b, 0, py, px]
    out = out.to(device=mask.device, dtype=mask.dtype)
    if original_ndim == 2:
        return out[0, 0]
    if original_ndim == 3:
        return out[0]
    return out


def _top_components_mask_from_mask(
    mask: torch.Tensor | None,
    threshold: float = 0.5,
    keep_components: int = 1,
) -> torch.Tensor | None:
    if mask is None:
        return None
    keep_components = max(1, int(keep_components))
    image = mask.detach().float().cpu()
    original_ndim = image.ndim
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[:, None] if image.shape[0] != 1 else image[None]
    elif image.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D, or 4D mask, got shape {tuple(mask.shape)}")
    out = torch.zeros_like(image)
    bsz, _, h, w = image.shape
    threshold = max(0.0, min(1.0, float(threshold)))
    for b in range(bsz):
        plane = image[b, 0].clamp(0.0, 1.0)
        binary = plane > threshold
        visited = torch.zeros_like(binary, dtype=torch.bool)
        components: list[tuple[float, list[tuple[int, int]]]] = []
        for y in range(h):
            for x in range(w):
                if not bool(binary[y, x]) or bool(visited[y, x]):
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                mass = 0.0
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    mass += float(plane[cy, cx].item())
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                components.append((mass, points))
        components.sort(key=lambda item: item[0], reverse=True)
        for _, points in components[:keep_components]:
            for py, px in points:
                out[b, 0, py, px] = image[b, 0, py, px]
    out = out.to(device=mask.device, dtype=mask.dtype)
    if original_ndim == 2:
        return out[0, 0]
    if original_ndim == 3:
        return out[0]
    return out


def _attention_object_mask_from_map(
    attention_map: torch.Tensor | None,
    threshold: float = 0.72,
    quantile: float = 0.9,
    fallback: torch.Tensor | None = None,
    keep_components: int = 1,
) -> torch.Tensor | None:
    if attention_map is None:
        return fallback
    image = attention_map.detach().float().clamp(0.0, 1.0)
    if float(image.max().item()) <= 1e-6:
        return fallback
    flat = image.flatten(start_dim=max(0, image.ndim - 2))
    quantile_threshold = float(torch.quantile(flat, quantile).item())
    threshold = max(float(threshold), quantile_threshold)
    for candidate_threshold in (threshold, 0.68, 0.55, 0.40):
        component = _top_components_mask_from_mask(
            image,
            threshold=candidate_threshold,
            keep_components=keep_components,
        )
        if component is not None and float(component.detach().float().max().item()) > 1e-6:
            max_value = component.detach().float().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return (component.float() / max_value).to(device=attention_map.device, dtype=attention_map.dtype)
    return fallback


def _velocity_diff_object_mask(
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    threshold: float = 0.72,
    quantile: float = 0.9,
    fallback: torch.Tensor | None = None,
    keep_components: int = 1,
) -> torch.Tensor | None:
    diff = (target_velocity.detach().float() - source_velocity.detach().float()).abs().mean(dim=1, keepdim=True)
    if float(diff.max().item()) <= 1e-6:
        return fallback
    flat = diff.flatten(1)
    lo = torch.quantile(flat, 0.10, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
    score = ((diff - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    if score.shape[-2:] != source_velocity.shape[-2:]:
        score = F.interpolate(score, size=source_velocity.shape[-2:], mode="bilinear", align_corners=False)
    return _attention_object_mask_from_map(
        score.to(device=source_velocity.device, dtype=source_velocity.dtype),
        threshold=threshold,
        quantile=quantile,
        fallback=fallback,
        keep_components=keep_components,
    )


def _attention_velocity_object_mask(
    attention_object: torch.Tensor | None,
    velocity_object: torch.Tensor | None,
    fallback: torch.Tensor | None = None,
    min_velocity_area: float = 0.001,
    max_attention_area: float = 0.18,
    continuous_support: bool = True,
    support_pad_x: float = 0.04,
    support_pad_y: float = 0.025,
    support_min_width: float = 0.28,
    support_min_height: float = 0.10,
) -> torch.Tensor | None:
    """
    Fuse semantic changed-token attention with RF velocity-difference support.

    Attention usually localizes the requested object better, while velocity
    difference suppresses broad prompt-attention regions. The fusion keeps the
    generic path oracle-free: if attention is compact, union it with velocity;
    if attention is too broad, trust velocity when it has non-trivial support.

    The final generic support is intentionally continuous rather than a sparse
    set of pixels. Attention and RF response are noisy at this resolution; a
    compact soft support prevents the edit ODE from being spatially gated away.
    """
    if attention_object is None and velocity_object is None:
        return fallback
    if attention_object is None:
        fused = velocity_object if velocity_object is not None else fallback
        return _continuous_support_mask(
            fused,
            pad_x=support_pad_x,
            pad_y=support_pad_y,
            min_width=support_min_width,
            min_height=support_min_height,
        ) if continuous_support else fused
    if velocity_object is None:
        return _continuous_support_mask(
            attention_object,
            pad_x=support_pad_x,
            pad_y=support_pad_y,
            min_width=support_min_width,
            min_height=support_min_height,
        ) if continuous_support else attention_object

    attention = attention_object.detach().float().clamp(0.0, 1.0)
    velocity = velocity_object.detach().float().clamp(0.0, 1.0)
    if velocity.shape[-2:] != attention.shape[-2:]:
        velocity = F.interpolate(velocity, size=attention.shape[-2:], mode="bilinear", align_corners=False)
        velocity_object = velocity.to(device=attention_object.device, dtype=attention_object.dtype)
    attention_area = _mask_binary_area_ratio(attention, threshold=0.5)
    velocity_area = _mask_binary_area_ratio(velocity, threshold=0.5)

    if attention_area > max_attention_area and velocity_area >= min_velocity_area:
        fused = velocity_object
    elif velocity_area < min_velocity_area:
        fused = attention_object
    else:
        velocity_neighborhood = dilate_spatial_mask(velocity, kernel_size=5).clamp(0.0, 1.0)
        attention_near_velocity = (attention * velocity_neighborhood).to(
            device=attention_object.device,
            dtype=attention_object.dtype,
        )
        if float(attention_near_velocity.detach().float().max().item()) > 1e-6:
            fused = torch.maximum(attention_near_velocity, velocity_object).clamp(0.0, 1.0)
        else:
            fused = velocity_object
    if not continuous_support:
        return fused
    return _continuous_support_mask(
        fused,
        pad_x=support_pad_x,
        pad_y=support_pad_y,
        min_width=support_min_width,
        min_height=support_min_height,
    )


def _continuous_support_mask(
    mask: torch.Tensor | None,
    threshold: float = 0.35,
    pad_x: float = 0.04,
    pad_y: float = 0.025,
    min_width: float = 0.28,
    min_height: float = 0.10,
) -> torch.Tensor | None:
    if mask is None:
        return None
    box = _box_from_mask(mask, threshold=threshold)
    if box is None:
        return mask
    box = _expand_box(
        box,
        pad_x=pad_x,
        pad_y_top=pad_y,
        pad_y_bottom=pad_y,
        min_width=min_width,
        min_height=min_height,
    )
    return normalized_box_mask_like(mask, box, feather=0.018).to(device=mask.device, dtype=mask.dtype)


def _semantic_velocity_object_mask(
    semantic_base: torch.Tensor | None,
    velocity_object: torch.Tensor | None,
    fallback: torch.Tensor | None = None,
    velocity_neighborhood_kernel: int = 9,
) -> torch.Tensor | None:
    """
    Refine an off-the-shelf semantic support with RF source-target response.

    The semantic mask supplies concept-level localization. The velocity mask is
    only trusted near that semantic support so unrelated prompt-response blobs do
    not leak into the edit region.
    """
    if semantic_base is None and velocity_object is None:
        return fallback
    if semantic_base is None:
        return velocity_object if velocity_object is not None else fallback
    base = semantic_base.detach().float().clamp(0.0, 1.0)
    if velocity_object is None:
        return semantic_base
    velocity = velocity_object.detach().float().clamp(0.0, 1.0)
    if velocity.shape[-2:] != base.shape[-2:]:
        velocity = F.interpolate(velocity, size=base.shape[-2:], mode="bilinear", align_corners=False)
    base_neighborhood = dilate_spatial_mask(base, kernel_size=velocity_neighborhood_kernel).clamp(0.0, 1.0)
    velocity_near_base = velocity * base_neighborhood
    fused = torch.maximum(base, velocity_near_base).clamp(0.0, 1.0)
    return fused.to(device=semantic_base.device, dtype=semantic_base.dtype)


def _conservative_attention_box(
    attention_map: torch.Tensor | None,
    threshold: float = 0.72,
    quantile: float = 0.9,
    fallback: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    if attention_map is None:
        return fallback
    image = attention_map.detach().float()
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 3:
        image = image[0]
    image = image.clamp(0.0, 1.0)
    if float(image.max().item()) <= 1e-6:
        return fallback
    threshold = max(float(threshold), float(torch.quantile(image.flatten(), quantile).item()))
    for candidate_threshold in (threshold, 0.68, 0.55, 0.40):
        box = _largest_component_box_from_mask(image, threshold=candidate_threshold)
        if box is not None:
            return box
    return fallback


def _box_to_list(box: tuple[float, float, float, float] | None) -> list[float] | None:
    if box is None:
        return None
    return [float(v) for v in box]


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
    edit_field_mode: str = "surrogate",
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
    edit_color_guidance_scale: float = 0.0,
    edit_color_source: str | None = None,
    edit_color_target: str | None = None,
    edit_color_mask_path: str | None = None,
    edit_color_mask_threshold: float = 0.38,
    edit_color_mask_softness: float = 0.10,
    edit_color_luma_gate_min: float = 0.0,
    edit_color_luma_gate_softness: float = 0.08,
    edit_color_detail_protect_scale: float = 0.0,
    edit_color_detail_protect_threshold: float = 0.35,
    edit_color_detail_protect_softness: float = 0.08,
    edit_color_target_chroma_scale: float = 1.0,
    edit_color_smooth_kernel: int = 5,
    edit_color_luma_preserve_scale: float = 0.35,
    edit_color_luma_gradient_preserve_scale: float = 0.15,
    edit_ref_guidance_scale: float = 0.0,
    edit_ref_image_path: str | None = None,
    edit_ref_mask_path: str | None = None,
    edit_ref_structure_image_path: str | None = None,
    edit_ref_chroma_mode: str = "yuv",
    edit_ref_chroma_magnitude_scale: float = 1.0,
    edit_ref_luma_preserve_scale: float = 0.35,
    edit_ref_gradient_preserve_scale: float = 0.15,
    edit_ref_darkness_guard_scale: float = 0.0,
    edit_ref_darkness_guard_margin: float = 0.03,
    edit_ref_smooth_kernel: int = 1,
    edit_ref_lowfreq_suppress_kernel: int = 0,
    edit_ref_lowfreq_suppress_scale: float = 0.0,
    edit_ref_schedule_start: float = 0.0,
    edit_ref_schedule_stop: float = 0.0,
    edit_ref_schedule_power: float = 1.0,
    edit_ref_max_struct_rms_ratio: float = 0.0,
    edit_ref_project_struct_conflict: float = 0.0,
    edit_core_scale: float = 1.35,
    edit_subject_scale: float = 0.35,
    source_inject_q_scale: float = 0.0,
    source_inject_k_scale: float = 0.0,
    source_inject_v_scale: float = 0.0,
    source_inject_layer_from: int = -1,
    source_inject_layer_to: int = -1,
    source_inject_steps: int = 8,
    source_inject_mask_mode: str = "none",
    source_inject_mask_box: tuple[float, float, float, float] | None = None,
    edit_bound_scale: float = 0.0,
    clip_start_timestep: float = 0.0,
    clip_stop_timestep: float = 0.6,
    preserve_blend_scale: float = 0.0,
    preserve_blend_start_timestep: float = 0.5,
    alpha_max: Optional[float] = None,
    alpha_schedule: str = "constant",
    beta_max: Optional[float] = None,
    beta_schedule: str = "constant",
    adaptive_clean_control: bool = False,
    adaptive_edit_target_progress: float = 0.0,
    adaptive_edit_target_rms: float = 0.0,
    adaptive_preserve_drift_budget: float = 0.0,
    adaptive_edit_gain: float = 0.0,
    adaptive_preserve_gain: float = 0.0,
    adaptive_edit_weight_min: float = 0.7,
    adaptive_edit_weight_max: float = 1.8,
    adaptive_preserve_weight_min: float = 1.0,
    adaptive_preserve_weight_max: float = 2.0,
    adaptive_projection_scale: float = 0.0,
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
    attention_mask_max_area_ratio: float = 0.25,
    attention_mask_fallback_threshold: float = 0.72,
    object_mask_provider: str = "attention",
    semantic_base_mask_path: str | None = None,
    support_score: str = "attention_x_clean",
    support_edit_operation: str = "auto",
    support_relation: str = "auto",
    support_grounding_method: str = "external_mask",
    save_support_debug_maps: bool = False,
    support_temporal_aggregation: str = "single",
    support_new_tokens: list[str] | None = None,
    support_host_tokens: list[str] | None = None,
    support_removed_tokens: list[str] | None = None,
    support_attention_power: float = 1.0,
    support_disagreement_power: float = 1.0,
    support_top_percentile: float = 90.0,
    support_min_area_ratio: float = 0.02,
    support_max_area_ratio: float = 0.30,
    support_keep_components: int = 2,
    support_dilate_radius: int = 5,
    support_blur_kernel: int = 5,
    attention_velocity_support_pad_x: float = 0.015,
    attention_velocity_support_pad_y: float = 0.010,
    attention_velocity_support_min_width: float = 0.18,
    attention_velocity_support_min_height: float = 0.065,
    mask_layering_mode: str = "object_contact",
    mask_object_threshold: float = 0.45,
    mask_contact_dilate_kernel: int = 7,
    mask_contact_scale: float = 0.25,
    mask_contact_edge_threshold: float = 0.55,
    mask_contact_edge_protect_scale: float = 0.75,
    mask_output_dir: str | None = None,
    edit_mask_dilate_kernel: int = 0,
    edit_mask_smooth_kernel: int = 0,
    edit_mask_component_threshold: float = 0.0,
    edit_mask_keep_components: int = 0,
    edit_mask_component_y_min: float | None = None,
    edit_mask_component_y_max: float | None = None,
    edit_mask_shift_y: float = 0.0,
    edit_mask_shift_x: float = 0.0,
    auto_local_boxes: bool = False,
    auto_box_threshold: float = 0.35,
    auto_edit_pad_x: float = 0.08,
    auto_edit_pad_y: float = 0.04,
    auto_edit_min_width: float = 0.28,
    auto_edit_min_height: float = 0.10,
    auto_source_pad_x: float = 0.14,
    auto_source_pad_y: float = 0.08,
    auto_preserve_pad_x: float = 0.04,
    auto_preserve_start_offset: float = 0.06,
    auto_preserve_height: float = 0.24,
    edit_mask_box: tuple[float, float, float, float] | None = None,
    edit_mask_box_mode: str = "replace",
    edit_mask_exclude_box: tuple[float, float, float, float] | None = None,
    edit_mask_use_core_as_subject: bool = False,
    external_edit_mask_path: str | None = None,
    external_edit_mask_mode: str = "replace",
    mask_blend: bool = False,
    mask_blend_mode: str = "subject",
    final_preserve_box: tuple[float, float, float, float] | None = None,
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
    valid_source_inject_mask_modes = {"none", "edit", "core", "preserve", "box"}
    if source_inject_mask_mode not in valid_source_inject_mask_modes:
        raise ValueError(f"Unsupported source_inject_mask_mode: {source_inject_mask_mode}")
    valid_object_mask_providers = {
        "attention",
        "velocity_diff",
        "attention_velocity",
        "semantic",
        "semantic_velocity",
        "generic_support",
        "operation_support_v3",
        "structure",
        "proposal_diff",
        "auto",
    }
    if object_mask_provider not in valid_object_mask_providers:
        raise ValueError(f"Unsupported object_mask_provider: {object_mask_provider}")
    valid_edit_field_modes = {"surrogate", "rf_diff", "text_diff", "rf_text_diff"}
    if edit_field_mode not in valid_edit_field_modes:
        raise ValueError(f"Unsupported edit_field_mode: {edit_field_mode}")
    use_legacy_edit_surrogates = edit_field_mode == "surrogate"
    use_rf_diff_edit_field = edit_field_mode in {"surrogate", "rf_diff", "rf_text_diff"}
    use_text_diff_edit_field = edit_field_mode in {"surrogate", "text_diff", "rf_text_diff"}
    use_auxiliary_edit_fields = edit_field_mode == "surrogate"
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
    source_inject_enabled = max(source_inject_q_scale, source_inject_k_scale, source_inject_v_scale) > 0.0
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
        return_trajectory=(
            trajectory_preserve_scale > 0.0
            or trajectory_subject_preserve_scale > 0.0
            or source_inject_enabled
        ),
    )
    if trajectory_preserve_scale > 0.0 or trajectory_subject_preserve_scale > 0.0 or source_inject_enabled:
        z_T, source_trajectory_by_timestep = inversion_result
    else:
        z_T = inversion_result
        source_trajectory_by_timestep = {}
    print("[inversion] done.")

    if alpha_max is None:
        alpha_max = rec_guidance_scale
    if beta_max is None:
        beta_max = 1.0
    color_source = infer_source_color_rgb(src_prompt, edit_color_source)
    color_target = infer_target_color_rgb(src_prompt, tar_prompt, edit_color_target)
    n_transformer_blocks = len(pipe.transformer.transformer_blocks)
    if source_inject_layer_from < 0:
        source_inject_layer_from = max(0, (2 * n_transformer_blocks) // 3)
    if source_inject_layer_to < 0:
        source_inject_layer_to = n_transformer_blocks
    source_inject_layer_from = max(0, min(int(source_inject_layer_from), n_transformer_blocks))
    source_inject_layer_to = max(source_inject_layer_from, min(int(source_inject_layer_to), n_transformer_blocks))

    # Extract attention mask for soft E_rec (optional)
    M_edit = None
    M_core = None
    M_preserve = None
    M_contact = None
    M_structure_edge = None
    M_attention_object = None
    M_velocity_object = None
    M_attention_velocity_object = None
    M_semantic_base = None
    M_semantic_velocity_object = None
    M_generic_support_attention = None
    M_generic_support_host = None
    M_generic_support_removed = None
    M_generic_support_clean = None
    M_generic_support_velocity = None
    M_generic_support_score = None
    M_operation_support_grounding = None
    M_operation_support_relation = None
    generic_support_stats: dict[str, float | int | str] = {}
    auto_anchor_mask = None
    auto_anchor_box = None
    mask_area_guard_applied = False
    mask_area_before_guard = None
    mask_area_guard_box = None
    resolved_edit_mask_box = edit_mask_box
    resolved_edit_mask_exclude_box = edit_mask_exclude_box
    resolved_source_inject_mask_box = source_inject_mask_box
    resolved_final_preserve_box = final_preserve_box
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
        or edit_color_guidance_scale > 0.0
        or adaptive_clean_control
        or trajectory_preserve_scale > 0.0
        or trajectory_subject_preserve_scale > 0.0
        or edit_initial_noise_scale > 0.0
        or external_edit_mask_path is not None
        or auto_local_boxes
        or object_mask_provider
        in {
            "attention",
            "velocity_diff",
            "attention_velocity",
            "semantic",
            "semantic_velocity",
            "generic_support",
            "operation_support_v3",
            "auto",
        }
        or source_inject_mask_mode in {"edit", "core", "preserve"}
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
            if object_mask_provider in {
                "velocity_diff",
                "attention_velocity",
                "semantic_velocity",
                "generic_support",
                "operation_support_v3",
            }:
                v_src_mid = calc_cfg_v_sd3(
                    pipe=pipe,
                    latents=x_src,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=base_guidance_scale,
                    t=t_mid,
                )
                v_tar_mid = calc_cfg_v_sd3(
                    pipe=pipe,
                    latents=x_src,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t_mid,
                )
                M_velocity_object = _velocity_diff_object_mask(
                    source_velocity=v_src_mid,
                    target_velocity=v_tar_mid,
                    threshold=attention_mask_fallback_threshold,
                    fallback=M_core,
                    keep_components=3 if object_mask_provider == "attention_velocity" else 1,
                )
                if M_velocity_object is not None:
                    M_edit = M_velocity_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if object_mask_provider in {"generic_support", "operation_support_v3"}:
                object_source = masks.get("target_changed")
                if support_new_tokens is not None:
                    new_masks = extract_attention_masks(
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
                        target_token_words=support_new_tokens,
                        source_token_words=None,
                        subject_threshold=attention_mask_subject_threshold,
                        core_threshold=attention_mask_core_threshold,
                    )
                    object_source = new_masks.get("target_changed")
                if object_source is None or float(object_source.detach().float().max().item()) <= 1e-6:
                    object_source = masks.get("combined")
                host_source = None
                if support_host_tokens is not None:
                    host_masks = extract_attention_masks(
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
                        target_token_words=support_host_tokens,
                        source_token_words=support_host_tokens,
                        subject_threshold=attention_mask_subject_threshold,
                        core_threshold=attention_mask_core_threshold,
                    )
                    host_source = torch.maximum(host_masks["source_changed"], host_masks["target_changed"])
                removed_source = None
                if support_removed_tokens is not None:
                    removed_masks = extract_attention_masks(
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
                        target_token_words=None,
                        source_token_words=support_removed_tokens,
                        subject_threshold=attention_mask_subject_threshold,
                        core_threshold=attention_mask_core_threshold,
                    )
                    removed_source = removed_masks["source_changed"]
                support_grounding = None
                if (
                    object_mask_provider == "operation_support_v3"
                    and semantic_base_mask_path is not None
                    and (support_grounding_method or "external_mask").strip().lower() != "none"
                ):
                    support_grounding = load_external_mask_like(M_edit, semantic_base_mask_path)
                    M_operation_support_grounding = support_grounding
                if object_mask_provider == "operation_support_v3":
                    temporal_mode = (support_temporal_aggregation or "single").strip().lower()
                    clean_map_override = None
                    velocity_map_override = None
                    temporal_step_count = 1
                    if temporal_mode in {"mean", "max"}:
                        support_indices = sorted(
                            {
                                max(0, min(len(timesteps) - 1, len(timesteps) // 4)),
                                max(0, min(len(timesteps) - 1, len(timesteps) // 2)),
                                max(0, min(len(timesteps) - 1, (3 * len(timesteps)) // 4)),
                            }
                        )
                        clean_maps = []
                        velocity_maps = []
                        mid_index = len(timesteps) // 2
                        for support_index in support_indices:
                            support_t = timesteps[support_index]
                            if support_index == mid_index:
                                v_src_support = v_src_mid
                                v_tar_support = v_tar_mid
                            else:
                                v_src_support = calc_cfg_v_sd3(
                                    pipe=pipe,
                                    latents=x_src,
                                    negative_prompt_embeds=src_negative_prompt_embeds,
                                    prompt_embeds=src_prompt_embeds,
                                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                                    guidance_scale=base_guidance_scale,
                                    t=support_t,
                                )
                                v_tar_support = calc_cfg_v_sd3(
                                    pipe=pipe,
                                    latents=x_src,
                                    negative_prompt_embeds=tar_negative_prompt_embeds,
                                    prompt_embeds=tar_prompt_embeds,
                                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                    guidance_scale=tar_guidance_scale,
                                    t=support_t,
                                )
                            clean_maps.append(compute_clean_disagreement(x_src, support_t, v_src_support, v_tar_support))
                            velocity_maps.append(compute_velocity_disagreement(v_src_support, v_tar_support))
                        temporal_step_count = len(support_indices)
                        clean_stack = torch.stack(clean_maps, dim=0)
                        velocity_stack = torch.stack(velocity_maps, dim=0)
                        if temporal_mode == "max":
                            clean_map_override = clean_stack.max(dim=0).values
                            velocity_map_override = velocity_stack.max(dim=0).values
                        else:
                            clean_map_override = clean_stack.mean(dim=0)
                            velocity_map_override = velocity_stack.mean(dim=0)
                    elif temporal_mode != "single":
                        raise ValueError(f"Unsupported support_temporal_aggregation: {support_temporal_aggregation}")
                    generic = build_operation_support_v3(
                        attention_map=object_source,
                        x_t=x_src,
                        t=t_mid,
                        source_velocity=v_src_mid,
                        target_velocity=v_tar_mid,
                        host_attention_map=host_source,
                        removed_attention_map=removed_source,
                        grounding_mask=support_grounding,
                        edit_operation=support_edit_operation,
                        relation=support_relation,
                        candidate=support_score,
                        attention_power=support_attention_power,
                        disagreement_power=support_disagreement_power,
                        top_percentile=support_top_percentile,
                        min_area_ratio=support_min_area_ratio,
                        max_area_ratio=support_max_area_ratio,
                        keep_components=support_keep_components,
                        dilate_radius=support_dilate_radius,
                        blur_kernel=support_blur_kernel,
                        clean_map_override=clean_map_override,
                        velocity_map_override=velocity_map_override,
                        temporal_aggregation=temporal_mode,
                        temporal_steps=temporal_step_count,
                    )
                    M_operation_support_relation = generic.relation_map
                    if save_support_debug_maps and mask_output_dir is not None:
                        save_support_debug(generic, mask_output_dir)
                else:
                    generic = build_generic_support(
                        attention_map=object_source,
                        x_t=x_src,
                        t=t_mid,
                        source_velocity=v_src_mid,
                        target_velocity=v_tar_mid,
                        host_attention_map=host_source,
                        removed_attention_map=removed_source,
                        edit_operation=support_edit_operation,
                        score_mode=support_score,
                        attention_power=support_attention_power,
                        disagreement_power=support_disagreement_power,
                        top_percentile=support_top_percentile,
                        min_area_ratio=support_min_area_ratio,
                        max_area_ratio=support_max_area_ratio,
                        keep_components=support_keep_components,
                        dilate_radius=support_dilate_radius,
                        blur_kernel=support_blur_kernel,
                    )
                M_edit = generic.edit_mask.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                M_core = torch.minimum(generic.core_mask.to(dtype=x_src.dtype).clamp(0.0, 1.0), M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                M_generic_support_attention = generic.attention_map
                M_generic_support_host = generic.host_attention_map
                M_generic_support_removed = generic.removed_attention_map
                M_generic_support_clean = generic.clean_disagreement_map
                M_generic_support_velocity = generic.velocity_disagreement_map
                M_generic_support_score = generic.support_score
                generic_support_stats = dict(generic.stats)
            if object_mask_provider in {"attention", "attention_velocity", "auto"}:
                object_source = masks.get("target_changed")
                if object_source is None or float(object_source.detach().float().max().item()) <= 1e-6:
                    object_source = masks.get("combined")
                M_attention_object = _attention_object_mask_from_map(
                    object_source,
                    threshold=attention_mask_fallback_threshold,
                    fallback=M_core,
                    keep_components=3 if object_mask_provider == "attention_velocity" else 1,
                )
                if M_attention_object is not None:
                    M_edit = M_attention_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if object_mask_provider == "attention_velocity":
                M_attention_velocity_object = _attention_velocity_object_mask(
                    M_attention_object,
                    M_velocity_object,
                    fallback=M_core,
                    support_pad_x=attention_velocity_support_pad_x,
                    support_pad_y=attention_velocity_support_pad_y,
                    support_min_width=attention_velocity_support_min_width,
                    support_min_height=attention_velocity_support_min_height,
                )
                if M_attention_velocity_object is not None:
                    M_edit = M_attention_velocity_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if object_mask_provider in {"semantic", "semantic_velocity"}:
                if semantic_base_mask_path is None:
                    raise ValueError("--object-mask-provider semantic(_velocity) requires --semantic-base-mask")
                M_semantic_base = load_external_mask_like(M_edit, semantic_base_mask_path)
                if object_mask_provider == "semantic_velocity":
                    M_semantic_velocity_object = _semantic_velocity_object_mask(
                        M_semantic_base,
                        M_velocity_object,
                        fallback=M_semantic_base,
                    )
                    M_edit = M_semantic_velocity_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                else:
                    M_edit = M_semantic_base.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                M_core = M_edit
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            auto_anchor_mask = M_core
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
                auto_anchor_mask = M_core
                if (
                    object_mask_provider in {"attention", "attention_velocity", "auto"}
                    and M_attention_object is not None
                    and float(M_edit.detach().float().max().item()) <= 1e-6
                    and float(M_attention_object.detach().float().max().item()) > 1e-6
                ):
                    M_edit = M_attention_object.to(dtype=x_src.dtype).clamp(0.0, 1.0)
                    M_core = M_edit
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                    auto_anchor_mask = M_core
                    print("[mask] component filters removed the attention object; restored generic attention object mask")
            if edit_mask_shift_y != 0.0 or edit_mask_shift_x != 0.0:
                M_edit = translate_spatial_mask(M_edit, shift_y=edit_mask_shift_y, shift_x=edit_mask_shift_x)
                M_core = translate_spatial_mask(M_core, shift_y=edit_mask_shift_y, shift_x=edit_mask_shift_x)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                auto_anchor_mask = M_core
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
            mask_area_before_guard = _mask_binary_area_ratio(M_edit, threshold=0.5)
            skip_attention_velocity_soft_guard = (
                object_mask_provider == "attention_velocity"
                and M_attention_velocity_object is not None
                and mask_area_before_guard <= 1.30 * attention_mask_max_area_ratio
            )
            if (
                attention_mask_max_area_ratio > 0.0
                and mask_area_before_guard > attention_mask_max_area_ratio
                and not skip_attention_velocity_soft_guard
            ):
                if object_mask_provider == "velocity_diff" and M_velocity_object is not None:
                    guard_source = M_velocity_object
                else:
                    guard_source = masks.get("target_changed")
                    if guard_source is None or float(guard_source.detach().float().max().item()) <= 1e-6:
                        guard_source = masks.get("combined")
                guard_anchor_box = _conservative_attention_box(
                    guard_source,
                    threshold=attention_mask_fallback_threshold,
                    fallback=_box_from_mask(M_core, threshold=auto_box_threshold),
                )
                if guard_anchor_box is not None:
                    mask_area_guard_box = _expand_box(
                        guard_anchor_box,
                        pad_x=auto_edit_pad_x,
                        pad_y_top=auto_edit_pad_y,
                        pad_y_bottom=auto_edit_pad_y,
                        min_width=0.28,
                        min_height=0.12,
                    )
                    guard_mask = normalized_box_mask_like(M_edit, mask_area_guard_box)
                    M_edit = (M_edit * guard_mask).clamp(0.0, 1.0)
                    M_core = (M_core * guard_mask).clamp(0.0, 1.0)
                    M_core = torch.minimum(M_core, M_edit)
                    M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
                    auto_anchor_mask = guard_mask
                    auto_anchor_box = mask_area_guard_box
                    mask_area_guard_applied = True
                    if resolved_edit_mask_box is None:
                        resolved_edit_mask_box = mask_area_guard_box
                    print(
                        "[mask_guard] "
                        f"area={mask_area_before_guard:.2%} "
                        f"limit={attention_mask_max_area_ratio:.2%} "
                        f"box={_box_to_list(mask_area_guard_box)}"
                    )
            if auto_local_boxes:
                if auto_anchor_box is None:
                    auto_anchor_box = _box_from_mask(
                        auto_anchor_mask,
                        threshold=auto_box_threshold,
                        fallback=_box_from_mask(M_edit, threshold=auto_box_threshold),
                    )
                if auto_anchor_box is not None:
                    if resolved_edit_mask_box is None:
                        resolved_edit_mask_box = _expand_box(
                            auto_anchor_box,
                            pad_x=auto_edit_pad_x,
                            pad_y_top=auto_edit_pad_y,
                            pad_y_bottom=auto_edit_pad_y,
                            min_width=auto_edit_min_width,
                            min_height=auto_edit_min_height,
                        )
                    if resolved_source_inject_mask_box is None and source_inject_mask_mode == "box":
                        resolved_source_inject_mask_box = _expand_box(
                            auto_anchor_box,
                            pad_x=auto_source_pad_x,
                            pad_y_top=auto_source_pad_y,
                            pad_y_bottom=auto_source_pad_y,
                            min_width=0.56,
                            min_height=0.24,
                        )
                    if resolved_edit_mask_box is None:
                        preserve_x0, preserve_y0, preserve_x1, preserve_y1 = auto_anchor_box
                    else:
                        preserve_x0, preserve_y0, preserve_x1, preserve_y1 = resolved_edit_mask_box
                    preserve_width = max(preserve_x1 - preserve_x0, 1e-6)
                    px0 = max(0.0, preserve_x0 + 0.15 * preserve_width - auto_preserve_pad_x)
                    px1 = min(1.0, preserve_x1 - 0.05 * preserve_width + auto_preserve_pad_x)
                    y1 = preserve_y1
                    py0 = min(1.0, y1 + auto_preserve_start_offset)
                    py1 = min(1.0, py0 + auto_preserve_height)
                    if py1 > py0:
                        preserve_box = _clamp_box((px0, py0, px1, py1))
                        if resolved_edit_mask_exclude_box is None:
                            resolved_edit_mask_exclude_box = preserve_box
                        if resolved_final_preserve_box is None:
                            resolved_final_preserve_box = preserve_box
            if resolved_edit_mask_box is not None:
                box_mask = normalized_box_mask_like(M_edit, resolved_edit_mask_box)
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
            if resolved_edit_mask_exclude_box is not None:
                exclude_mask = normalized_box_mask_like(M_edit, resolved_edit_mask_exclude_box)
                keep_mask = (1.0 - exclude_mask).clamp(0.0, 1.0)
                M_edit = (M_edit * keep_mask).clamp(0.0, 1.0)
                M_core = (M_core * keep_mask).clamp(0.0, 1.0)
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
                if edit_mask_dilate_kernel > 1:
                    M_edit = dilate_spatial_mask(M_edit, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                    M_core = dilate_spatial_mask(M_core, kernel_size=edit_mask_dilate_kernel).clamp(0.0, 1.0)
                if edit_mask_smooth_kernel > 1:
                    M_edit = smooth_spatial_mask(M_edit, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                    M_core = smooth_spatial_mask(M_core, kernel_size=edit_mask_smooth_kernel).clamp(0.0, 1.0)
                M_core = torch.minimum(M_core, M_edit)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if mask_layering_mode == "object_contact":
                M_edit, M_core, M_contact, M_preserve, M_structure_edge = build_object_contact_masks(
                    edit_mask=M_edit,
                    core_mask=M_core,
                    structure_reference=x_src,
                    object_threshold=mask_object_threshold,
                    contact_dilate_kernel=mask_contact_dilate_kernel,
                    contact_scale=mask_contact_scale,
                    contact_edge_threshold=mask_contact_edge_threshold,
                    contact_edge_protect_scale=mask_contact_edge_protect_scale,
                )
            elif mask_layering_mode != "none":
                raise ValueError(f"Unsupported mask_layering_mode: {mask_layering_mode}")
            if edit_mask_use_core_as_subject:
                M_edit = M_core.clamp(0.0, 1.0)
                M_preserve = (1.0 - M_edit).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                for name, mask in masks.items():
                    save_mask_image(mask.to(dtype=torch.float32), os.path.join(mask_output_dir, f"{name}.png"))
                if M_attention_object is not None:
                    save_mask_image(
                        M_attention_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "attention_object.png"),
                    )
                if M_velocity_object is not None:
                    save_mask_image(
                        M_velocity_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "velocity_diff_object.png"),
                    )
                if M_attention_velocity_object is not None:
                    save_mask_image(
                        M_attention_velocity_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "attention_velocity_object.png"),
                    )
                if M_semantic_base is not None:
                    save_mask_image(
                        M_semantic_base.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "semantic_base.png"),
                    )
                if M_semantic_velocity_object is not None:
                    save_mask_image(
                        M_semantic_velocity_object.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "semantic_velocity_object.png"),
                    )
                if M_generic_support_attention is not None:
                    save_mask_image(
                        M_generic_support_attention.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_attention_map.png"),
                    )
                if M_generic_support_host is not None:
                    save_mask_image(
                        M_generic_support_host.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_host_attention_map.png"),
                    )
                if M_generic_support_removed is not None:
                    save_mask_image(
                        M_generic_support_removed.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_removed_attention_map.png"),
                    )
                if M_generic_support_clean is not None:
                    save_mask_image(
                        M_generic_support_clean.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_clean_disagreement_map.png"),
                    )
                if M_generic_support_velocity is not None:
                    save_mask_image(
                        M_generic_support_velocity.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_velocity_disagreement_map.png"),
                    )
                if M_generic_support_score is not None:
                    save_mask_image(
                        M_generic_support_score.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "generic_support_score.png"),
                    )
                if M_operation_support_grounding is not None:
                    save_mask_image(
                        M_operation_support_grounding.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "operation_v3_grounding_mask.png"),
                    )
                if M_operation_support_relation is not None:
                    save_mask_image(
                        M_operation_support_relation.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "operation_v3_relation_map.png"),
                    )
                save_mask_image(M_edit.to(dtype=torch.float32), os.path.join(mask_output_dir, "subject_final.png"))
                save_mask_image(M_core.to(dtype=torch.float32), os.path.join(mask_output_dir, "core_final.png"))
                save_mask_image(M_preserve.to(dtype=torch.float32), os.path.join(mask_output_dir, "preserve_final.png"))
                if M_contact is not None:
                    save_mask_image(
                        M_contact.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "contact_final.png"),
                    )
                if M_structure_edge is not None:
                    save_mask_image(
                        M_structure_edge.to(dtype=torch.float32),
                        os.path.join(mask_output_dir, "structure_edge.png"),
                    )
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
    source_inject_mask = None
    if source_inject_mask_mode == "box":
        if resolved_source_inject_mask_box is None:
            raise ValueError("--source-inject-mask-mode box requires --source-inject-mask-box")
        ref_mask = M_edit if M_edit is not None else x_src[:, :1]
        source_inject_mask = normalized_box_mask_like(ref_mask, resolved_source_inject_mask_box)
    elif source_inject_mask_mode == "edit":
        if M_edit is None:
            raise ValueError("--source-inject-mask-mode edit requires an extracted edit mask")
        source_inject_mask = M_edit
    elif source_inject_mask_mode == "core":
        if M_core is None:
            raise ValueError("--source-inject-mask-mode core requires an extracted core mask")
        source_inject_mask = M_core
    elif source_inject_mask_mode == "preserve":
        if M_preserve is None:
            raise ValueError("--source-inject-mask-mode preserve requires an extracted preserve mask")
        source_inject_mask = M_preserve

    if source_inject_mask is not None:
        source_inject_mask = source_inject_mask.to(device=x_src.device, dtype=x_src.dtype).clamp(0.0, 1.0)
        if mask_output_dir is not None:
            os.makedirs(mask_output_dir, exist_ok=True)
            save_mask_image(
                source_inject_mask.to(dtype=torch.float32),
                os.path.join(mask_output_dir, "source_inject_mask.png"),
            )
    mask_stats = spatial_mask_stats(M_edit, prefix="mask")
    core_mask_stats = spatial_mask_stats(M_core, prefix="core_mask")
    contact_mask_stats = spatial_mask_stats(M_contact, prefix="contact_mask")
    structure_edge_stats = spatial_mask_stats(M_structure_edge, prefix="structure_edge")
    preserve_mask_stats = spatial_mask_stats(M_preserve, prefix="preserve_mask")
    source_inject_mask_stats = spatial_mask_stats(source_inject_mask, prefix="source_inject_mask")

    color_mask_image = None
    color_mask_latent = None
    source_color_reference = None
    ref_mask_image = None
    ref_mask_latent = None
    ref_image_reference = None
    ref_structure_reference = None
    needs_clip_reward = (
        (use_auxiliary_edit_fields and edit_clip_guidance_scale > 0.0)
        or (use_text_diff_edit_field and edit_text_guidance_scale > 0.0)
    )
    if needs_clip_reward or edit_color_guidance_scale > 0.0 or edit_ref_guidance_scale > 0.0:
        # Offloaded diffusers modules wrap the pipeline VAE with accelerate
        # hooks that produce inference tensors. A dedicated copy keeps the VAE
        # decode path differentiable for image-space edit rewards.
        grad_vae = copy.deepcopy(pipe.vae).to(device)
        grad_vae.eval()
        for param in grad_vae.parameters():
            param.requires_grad_(False)
    if edit_color_guidance_scale > 0.0 and color_source is not None and M_edit is not None:
        with torch.no_grad():
            source_color_image = decode_latent_to_unit_image(pipe, x_src, vae_override=grad_vae)
            source_color_reference = source_color_image.detach()
            if edit_color_mask_path is not None:
                external_color_mask = load_external_mask_like(source_color_image[:, :1], edit_color_mask_path)
                edit_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_color_image.dtype, device=source_color_image.device),
                    size=source_color_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                color_mask_image = (external_color_mask * edit_mask_image).clamp(0.0, 1.0)
            else:
                color_mask_image = source_color_similarity_mask(
                    source_color_image,
                    color_source[1],
                    M_edit,
                    threshold=edit_color_mask_threshold,
                    softness=edit_color_mask_softness,
                    luma_gate_min=edit_color_luma_gate_min,
                    luma_gate_softness=edit_color_luma_gate_softness,
                    detail_protect_scale=edit_color_detail_protect_scale,
                    detail_protect_threshold=edit_color_detail_protect_threshold,
                    detail_protect_softness=edit_color_detail_protect_softness,
                )
            color_mask_latent = torch.nn.functional.interpolate(
                color_mask_image,
                size=x_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                save_mask_image(color_mask_image, os.path.join(mask_output_dir, "color_edit_mask.png"))
    if edit_ref_guidance_scale > 0.0:
        if edit_ref_image_path is None:
            raise ValueError("--edit-ref-guidance-scale requires --edit-ref-image")
        with torch.no_grad():
            source_ref_image = decode_latent_to_unit_image(pipe, x_src, vae_override=grad_vae)
            ref_image_reference = load_external_image_like(source_ref_image, edit_ref_image_path).detach()
            if edit_ref_structure_image_path is not None:
                ref_structure_reference = load_external_image_like(source_ref_image, edit_ref_structure_image_path).detach()
            if edit_ref_mask_path is not None:
                ref_mask_image = load_external_mask_like(source_ref_image[:, :1], edit_ref_mask_path)
            elif color_mask_image is not None:
                ref_mask_image = color_mask_image.detach()
            elif M_edit is not None:
                ref_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_ref_image.dtype, device=source_ref_image.device),
                    size=source_ref_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            else:
                ref_mask_image = torch.ones_like(source_ref_image[:, :1])
            if M_edit is not None:
                edit_mask_image = torch.nn.functional.interpolate(
                    M_edit.to(dtype=source_ref_image.dtype, device=source_ref_image.device),
                    size=source_ref_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                ref_mask_image = (ref_mask_image * edit_mask_image).clamp(0.0, 1.0)
            ref_mask_latent = torch.nn.functional.interpolate(
                ref_mask_image,
                size=x_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            if mask_output_dir is not None:
                save_mask_image(ref_mask_image, os.path.join(mask_output_dir, "ref_edit_mask.png"))
    if needs_clip_reward:
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
    step_stats: list[dict[str, float | str | None]] = []
    active_edit_step = 0

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
        source_inject_active = (
            source_inject_enabled
            and source_inject_layer_to > source_inject_layer_from
            and (source_inject_steps <= 0 or active_edit_step < source_inject_steps)
        )
        source_inject_counts = {"all": 0, "q": 0, "k": 0, "v": 0}
        with torch.no_grad():
            if source_inject_active:
                source_latent_for_inject = source_trajectory_by_timestep.get(int(t.item()), x_src)
                v_tar, source_inject_counts = calc_cfg_v_sd3_with_source_qkv_injection(
                    pipe=pipe,
                    latents=z_t,
                    negative_prompt_embeds=tar_negative_prompt_embeds,
                    prompt_embeds=tar_prompt_embeds,
                    negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance_scale=tar_guidance_scale,
                    t=t,
                    source_latents=source_latent_for_inject.to(latents_dtype),
                    source_prompt_embeds=src_prompt_embeds,
                    source_pooled_prompt_embeds=src_pooled_prompt_embeds,
                    q_strength=source_inject_q_scale,
                    k_strength=source_inject_k_scale,
                    v_strength=source_inject_v_scale,
                    layer_from=source_inject_layer_from,
                    layer_to=source_inject_layer_to,
                    spatial_gate=source_inject_mask,
                )
            else:
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
        adaptive_edit_progress = 0.0
        adaptive_edit_change_rms = 0.0
        adaptive_edit_target_rms_value = 0.0
        adaptive_edit_target_gap_rms = 0.0
        adaptive_edit_deficit = 0.0
        adaptive_preserve_drift = 0.0
        adaptive_preserve_excess = 0.0
        adaptive_edit_weight = 1.0
        adaptive_preserve_weight = 1.0
        adaptive_projection_dot = 0.0
        adaptive_projection_norm = 0.0
        adaptive_clean_conflict_score = 0.0
        adaptive_clean_projection_ratio = 0.0
        adaptive_preserve_drift_after_projection_estimate = 0.0
        if adaptive_clean_control:
            edit_diag_gate = None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device)
            preserve_diag_gate = None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device)
            source_clean = x_src.to(torch.float32)
            current_delta = x0_src_step - source_clean
            target_delta = x0_tar - source_clean
            target_gap = x0_tar - x0_src_step
            adaptive_edit_change_rms = float(masked_rms(current_delta, edit_diag_gate).item())
            adaptive_edit_target_rms_value = float(masked_rms(target_delta, edit_diag_gate).item())
            adaptive_edit_target_gap_rms = float(masked_rms(target_gap, edit_diag_gate).item())
            if edit_diag_gate is None:
                edit_progress_num = (current_delta * target_delta).sum()
                edit_progress_den = target_delta.square().sum().clamp_min(1e-8)
            else:
                edit_gate = edit_diag_gate.to(dtype=torch.float32, device=z_t.device)
                if edit_gate.shape[-2:] != current_delta.shape[-2:]:
                    edit_gate = torch.nn.functional.interpolate(
                        edit_gate,
                        size=current_delta.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                edit_gate = edit_gate.clamp(0.0, 1.0)
                edit_progress_num = (current_delta * target_delta * edit_gate).sum()
                edit_progress_den = (target_delta.square() * edit_gate).sum().clamp_min(1e-8)
            adaptive_edit_progress = float((edit_progress_num / edit_progress_den).detach().item())
            adaptive_preserve_drift = float(masked_rms(x0_src_step - source_clean, preserve_diag_gate).item())
            if adaptive_edit_target_progress > 0.0:
                adaptive_edit_deficit = max(0.0, float(adaptive_edit_target_progress) - adaptive_edit_progress)
            elif adaptive_edit_target_rms > 0.0:
                adaptive_edit_deficit = max(0.0, float(adaptive_edit_target_rms) - adaptive_edit_target_gap_rms)
            if adaptive_preserve_drift_budget > 0.0:
                adaptive_preserve_excess = max(0.0, adaptive_preserve_drift - float(adaptive_preserve_drift_budget))
            adaptive_edit_weight = 1.0 + float(adaptive_edit_gain) * adaptive_edit_deficit
            adaptive_preserve_weight = 1.0 + float(adaptive_preserve_gain) * adaptive_preserve_excess
            adaptive_edit_weight = max(
                float(adaptive_edit_weight_min),
                min(float(adaptive_edit_weight_max), adaptive_edit_weight),
            )
            adaptive_preserve_weight = max(
                float(adaptive_preserve_weight_min),
                min(float(adaptive_preserve_weight_max), adaptive_preserve_weight),
            )

        # Reconstruction correction: x_hat_0 -> E_rec -> u_rec surrogate.
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
            if adaptive_clean_control:
                v_rec = adaptive_preserve_weight * v_rec
            rec_guidance_norm = v_rec.norm().item()
        else:
            v_rec = torch.zeros_like(v_base)

        # Editing correction: x_hat_0 -> E_edit -> u_edit surrogate.
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
        color_guidance_norm = 0.0
        color_edit_energy = None

        if beta_t != 0.0 and (
            (use_rf_diff_edit_field and edit_hedit_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_region_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_target_guidance_scale > 0.0)
            or (use_legacy_edit_surrogates and edit_source_guidance_scale > 0.0)
        ):
            with torch.no_grad():
                target_feature_map = None
                source_feature_map = None
                if use_legacy_edit_surrogates and edit_target_guidance_scale > 0.0:
                    target_feature_map = extract_sd3_feature_structure_map(
                        pipe=pipe,
                        x_latent=x0_tar,
                        prompt_embeds=tar_prompt_embeds,
                        pooled_embeds=tar_pooled_prompt_embeds,
                        t=t,
                        detach=True,
                    ).to(dtype=torch.float32, device=z_t.device)
                if use_legacy_edit_surrogates and edit_source_guidance_scale > 0.0:
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
                    lambda_base=edit_hedit_guidance_scale if use_rf_diff_edit_field else 0.0,
                    lambda_anchor=edit_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_region=edit_region_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_target=edit_target_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_source=edit_source_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    velocity_t_min=linear_path_t_min,
                )
                edit_energy_terms = editing_energy_total(
                    x0_tar=x0_tar,
                    x0_src=x0_src_step,
                    M_edit=None if M_edit is None else M_edit.to(dtype=torch.float32, device=z_t.device),
                    target_feature_map=target_feature_map,
                    source_feature_map=source_feature_map,
                    lambda_anchor=edit_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_region=edit_region_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_target=edit_target_guidance_scale if use_legacy_edit_surrogates else 0.0,
                    lambda_source=edit_source_guidance_scale if use_legacy_edit_surrogates else 0.0,
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
                or (use_text_diff_edit_field and edit_text_guidance_scale > 0.0)
            )
        ):
            zt_for_edit = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                if edit_field_mode in {"text_diff", "rf_text_diff"}:
                    v_reward = _cfg_v_sd3_with_grad(
                        pipe=pipe,
                        latents=zt_for_edit,
                        negative_prompt_embeds=src_negative_prompt_embeds,
                        prompt_embeds=src_prompt_embeds,
                        negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                        pooled_prompt_embeds=src_pooled_prompt_embeds,
                        guidance_scale=base_guidance_scale,
                        t=t,
                    )
                else:
                    v_reward = _cfg_v_sd3_with_grad(
                        pipe=pipe,
                        latents=zt_for_edit,
                        negative_prompt_embeds=tar_negative_prompt_embeds,
                        prompt_embeds=tar_prompt_embeds,
                        negative_pooled_prompt_embeds=tar_negative_pooled_prompt_embeds,
                        pooled_prompt_embeds=tar_pooled_prompt_embeds,
                        guidance_scale=tar_guidance_scale,
                        t=t,
                    )
                x0_reward = predict_x0_from_linear_rf_path(zt_for_edit, v_reward, t_i)
                current_image = decode_latent_to_unit_image(pipe, x0_reward, vae_override=grad_vae)

                if use_text_diff_edit_field and edit_text_guidance_scale > 0.0:
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

                if use_auxiliary_edit_fields and edit_clip_guidance_scale > 0.0 and clip_window_active:
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

        if beta_t != 0.0 and use_auxiliary_edit_fields and edit_dds_guidance_scale > 0.0:
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
        if beta_t != 0.0 and use_auxiliary_edit_fields and edit_app_guidance_scale > 0.0 and M_edit is not None:
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

        color_guidance = torch.zeros_like(v_base)
        ref_guidance = torch.zeros_like(v_base)
        if (
            beta_t != 0.0
            and edit_color_guidance_scale > 0.0
            and M_edit is not None
            and color_target is not None
            and color_mask_image is not None
        ):
            zt_for_color = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_color_src = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_color,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=base_guidance_scale,
                    t=t,
                )
                x0_color = predict_x0_from_linear_rf_path(zt_for_color, v_color_src, t_i)
                color_image = decode_latent_to_unit_image(pipe, x0_color, vae_override=grad_vae)
                color_edit_energy = masked_chroma_luma_loss(
                    color_image,
                    source_color_reference,
                color_target[1],
                color_mask_image,
                target_chroma_scale=edit_color_target_chroma_scale,
                luma_preserve_scale=edit_color_luma_preserve_scale,
                luma_gradient_preserve_scale=edit_color_luma_gradient_preserve_scale,
            )
            grad_color = torch.autograd.grad(color_edit_energy, zt_for_color)[0].detach().to(torch.float32)
            grad_color = smooth_guidance_field(grad_color, kernel_size=int(edit_color_smooth_kernel))
            grad_color_rms = grad_color.square().mean().sqrt() + 1e-8
            color_guidance = beta_t * edit_color_guidance_scale * (grad_color / grad_color_rms)
            color_guidance_norm = color_guidance.norm().item()

        ref_edit_energy = None
        ref_guidance_norm = 0.0
        ref_guidance_pre_stabilize_norm = 0.0
        ref_guidance_schedule_weight = 1.0
        if (
            beta_t != 0.0
            and edit_ref_guidance_scale > 0.0
            and ref_image_reference is not None
            and ref_mask_image is not None
        ):
            zt_for_ref = z_t.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v_ref_src = _cfg_v_sd3_with_grad(
                    pipe=pipe,
                    latents=zt_for_ref,
                    negative_prompt_embeds=src_negative_prompt_embeds,
                    prompt_embeds=src_prompt_embeds,
                    negative_pooled_prompt_embeds=src_negative_pooled_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance_scale=base_guidance_scale,
                    t=t,
                )
                x0_ref = predict_x0_from_linear_rf_path(zt_for_ref, v_ref_src, t_i)
                ref_current_image = decode_latent_to_unit_image(pipe, x0_ref, vae_override=grad_vae)
                ref_edit_energy = masked_reference_image_loss(
                    ref_current_image,
                    ref_image_reference,
                    ref_mask_image,
                    structure_image=ref_structure_reference,
                    luma_preserve_scale=edit_ref_luma_preserve_scale,
                    gradient_preserve_scale=edit_ref_gradient_preserve_scale,
                    darkness_guard_scale=edit_ref_darkness_guard_scale,
                    darkness_guard_margin=edit_ref_darkness_guard_margin,
                    chroma_mode=edit_ref_chroma_mode,
                    chroma_magnitude_scale=edit_ref_chroma_magnitude_scale,
                )
            grad_ref = torch.autograd.grad(ref_edit_energy, zt_for_ref)[0].detach().to(torch.float32)
            grad_ref = smooth_guidance_field(grad_ref, kernel_size=int(edit_ref_smooth_kernel))
            grad_ref = suppress_low_frequency_guidance(
                grad_ref,
                kernel_size=int(edit_ref_lowfreq_suppress_kernel),
                strength=edit_ref_lowfreq_suppress_scale,
            )
            grad_ref_rms = grad_ref.square().mean().sqrt() + 1e-8
            ref_guidance = beta_t * edit_ref_guidance_scale * (grad_ref / grad_ref_rms)
            progress_t = 1.0 - float(t_i.detach().item())
            ref_guidance_schedule_weight = ramp_schedule_from_progress(
                progress_t,
                start=edit_ref_schedule_start,
                stop=edit_ref_schedule_stop,
                power=edit_ref_schedule_power,
            )
            ref_guidance = ref_guidance * ref_guidance_schedule_weight
            ref_guidance_pre_stabilize_norm = ref_guidance.norm().item()
            struct_anchor = v_base + v_rec
            ref_guidance = remove_opposing_guidance_component(
                ref_guidance,
                struct_anchor,
                strength=edit_ref_project_struct_conflict,
            )
            ref_guidance = limit_guidance_rms_relative_to_anchor(
                ref_guidance,
                struct_anchor,
                max_ratio=edit_ref_max_struct_rms_ratio,
            )
            ref_guidance_norm = ref_guidance.norm().item()

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
            color_gate = edit_gate if color_mask_latent is None else color_mask_latent.to(dtype=torch.float32, device=z_t.device)
            color_guidance = color_guidance * color_gate
            ref_gate = edit_gate if ref_mask_latent is None else ref_mask_latent.to(dtype=torch.float32, device=z_t.device)
            ref_guidance = ref_guidance * ref_gate
            edit_guidance_norm = edit_guidance.norm().item()
            clip_guidance_norm = clip_guidance.norm().item()
            text_guidance_norm = text_guidance.norm().item()
            dds_guidance_norm = dds_guidance.norm().item()
            app_guidance_norm = app_guidance.norm().item()
            color_guidance_norm = color_guidance.norm().item()
            ref_guidance_norm = ref_guidance.norm().item()

        v_edit_total = (
            edit_guidance
            + clip_guidance
            + text_guidance
            + dds_guidance
            + app_guidance
            + color_guidance
            + ref_guidance
        )
        if adaptive_clean_control:
            v_edit_total = adaptive_edit_weight * v_edit_total
            preserve_projection_gate = None if M_preserve is None else M_preserve.to(dtype=torch.float32, device=z_t.device)
            preserve_error = x0_src_step - x_src.to(torch.float32)
            clean_edit_effect = -t_i.to(torch.float32) * v_edit_total
            if preserve_projection_gate is not None:
                clean_gate = preserve_projection_gate.to(dtype=torch.float32, device=z_t.device)
                if clean_gate.shape[-2:] != preserve_error.shape[-2:]:
                    clean_gate = torch.nn.functional.interpolate(
                        clean_gate,
                        size=preserve_error.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                clean_gate = clean_gate.clamp(0.0, 1.0)
                preserve_error_eval = preserve_error * clean_gate
                clean_effect_eval = clean_edit_effect * clean_gate
            else:
                preserve_error_eval = preserve_error
                clean_effect_eval = clean_edit_effect
            clean_conflict_dot = (preserve_error_eval * clean_effect_eval).sum()
            adaptive_clean_conflict_score = float(clean_conflict_dot.detach().item())
            if float(clean_conflict_dot.detach().item()) > 0.0:
                preserve_error_sq = preserve_error_eval.square().sum().clamp_min(1e-8)
                destructive_effect = (clean_conflict_dot / preserve_error_sq) * preserve_error_eval
                adaptive_clean_projection_ratio = float(
                    (
                        destructive_effect.norm()
                        / clean_effect_eval.norm().clamp_min(1e-8)
                    ).detach().item()
                )
                clean_effect_projected = clean_effect_eval - destructive_effect
                if preserve_projection_gate is not None:
                    clean_effect_estimate = clean_edit_effect * (1.0 - clean_gate) + clean_effect_projected
                else:
                    clean_effect_estimate = clean_effect_projected
            else:
                clean_effect_estimate = clean_edit_effect
            preserve_after_estimate = preserve_error + clean_effect_estimate
            adaptive_preserve_drift_after_projection_estimate = float(
                masked_rms(preserve_after_estimate, preserve_projection_gate).item()
            )
            v_edit_total, adaptive_projection_dot, adaptive_projection_norm = remove_masked_opposing_guidance_component(
                guidance=v_edit_total,
                anchor=v_base + v_rec,
                mask=preserve_projection_gate,
                strength=adaptive_projection_scale,
            )
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
            "adaptive_clean_control": bool(adaptive_clean_control),
            "adaptive_edit_target_progress": float(adaptive_edit_target_progress),
            "adaptive_edit_target_rms": float(adaptive_edit_target_rms),
            "adaptive_preserve_drift_budget": float(adaptive_preserve_drift_budget),
            "adaptive_edit_gain": float(adaptive_edit_gain),
            "adaptive_preserve_gain": float(adaptive_preserve_gain),
            "adaptive_edit_weight_min": float(adaptive_edit_weight_min),
            "adaptive_edit_weight_max": float(adaptive_edit_weight_max),
            "adaptive_preserve_weight_min": float(adaptive_preserve_weight_min),
            "adaptive_preserve_weight_max": float(adaptive_preserve_weight_max),
            "adaptive_projection_scale": float(adaptive_projection_scale),
            "adaptive_edit_progress": float(adaptive_edit_progress),
            "adaptive_edit_change_rms": float(adaptive_edit_change_rms),
            "adaptive_edit_target_rms_value": float(adaptive_edit_target_rms_value),
            "adaptive_edit_target_gap_rms": float(adaptive_edit_target_gap_rms),
            "adaptive_edit_deficit": float(adaptive_edit_deficit),
            "adaptive_preserve_drift": float(adaptive_preserve_drift),
            "adaptive_preserve_excess": float(adaptive_preserve_excess),
            "adaptive_edit_weight": float(adaptive_edit_weight),
            "adaptive_preserve_weight": float(adaptive_preserve_weight),
            "adaptive_projection_dot": float(adaptive_projection_dot),
            "adaptive_projection_norm": float(adaptive_projection_norm),
            "adaptive_clean_conflict_score": float(adaptive_clean_conflict_score),
            "adaptive_clean_projection_ratio": float(adaptive_clean_projection_ratio),
            "adaptive_preserve_drift_before_projection": float(adaptive_preserve_drift),
            "adaptive_preserve_drift_after_projection_estimate": float(
                adaptive_preserve_drift_after_projection_estimate
            ),
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
            "source_inject_q_scale": float(source_inject_q_scale),
            "source_inject_k_scale": float(source_inject_k_scale),
            "source_inject_v_scale": float(source_inject_v_scale),
            "source_inject_layer_from": float(source_inject_layer_from),
            "source_inject_layer_to": float(source_inject_layer_to),
            "source_inject_steps": float(source_inject_steps),
            "source_inject_mask_mode": source_inject_mask_mode,
            "source_inject_mask_box": None
            if resolved_source_inject_mask_box is None
            else [float(v) for v in resolved_source_inject_mask_box],
            "source_inject_active_step": float(active_edit_step),
            "source_inject_active": bool(source_inject_active),
            "source_inject_layers": float(source_inject_counts["all"]),
            "source_inject_q_layers": float(source_inject_counts["q"]),
            "source_inject_k_layers": float(source_inject_counts["k"]),
            "source_inject_v_layers": float(source_inject_counts["v"]),
            "edit_initial_noise_scale": float(edit_initial_noise_scale),
            "edit_initial_noise_region": edit_initial_noise_region,
            "trajectory_preserve_weight": float(trajectory_preserve_weight),
            "attention_mask_mode": attention_mask_mode,
            "attention_mask_subject_threshold": float(attention_mask_subject_threshold),
            "attention_mask_core_threshold": float(attention_mask_core_threshold),
            "attention_mask_max_area_ratio": float(attention_mask_max_area_ratio),
            "attention_mask_fallback_threshold": float(attention_mask_fallback_threshold),
            "object_mask_provider": object_mask_provider,
            "semantic_base_mask_path": semantic_base_mask_path,
            "support_score": support_score,
            "support_edit_operation": support_edit_operation,
            "support_relation": support_relation,
            "support_grounding_method": support_grounding_method,
            "save_support_debug_maps": bool(save_support_debug_maps),
            "support_temporal_aggregation": support_temporal_aggregation,
            "support_new_tokens": ",".join(support_new_tokens or []),
            "support_host_tokens": ",".join(support_host_tokens or []),
            "support_removed_tokens": ",".join(support_removed_tokens or []),
            "support_attention_power": float(support_attention_power),
            "support_disagreement_power": float(support_disagreement_power),
            "support_top_percentile": float(support_top_percentile),
            "support_min_area_ratio": float(support_min_area_ratio),
            "support_max_area_ratio": float(support_max_area_ratio),
            "support_keep_components": float(support_keep_components),
            "support_dilate_radius": float(support_dilate_radius),
            "support_blur_kernel": float(support_blur_kernel),
            **generic_support_stats,
            "attention_velocity_support_pad_x": float(attention_velocity_support_pad_x),
            "attention_velocity_support_pad_y": float(attention_velocity_support_pad_y),
            "attention_velocity_support_min_width": float(attention_velocity_support_min_width),
            "attention_velocity_support_min_height": float(attention_velocity_support_min_height),
            "mask_layering_mode": mask_layering_mode,
            "mask_object_threshold": float(mask_object_threshold),
            "mask_contact_dilate_kernel": float(mask_contact_dilate_kernel),
            "mask_contact_scale": float(mask_contact_scale),
            "mask_contact_edge_threshold": float(mask_contact_edge_threshold),
            "mask_contact_edge_protect_scale": float(mask_contact_edge_protect_scale),
            "mask_area_guard_applied": bool(mask_area_guard_applied),
            "mask_area_before_guard": mask_area_before_guard,
            "mask_area_guard_box": _box_to_list(mask_area_guard_box),
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
            "auto_local_boxes": bool(auto_local_boxes),
            "auto_box_threshold": float(auto_box_threshold),
            "auto_anchor_box": _box_to_list(auto_anchor_box),
            "resolved_edit_mask_box": _box_to_list(resolved_edit_mask_box),
            "resolved_edit_mask_exclude_box": _box_to_list(resolved_edit_mask_exclude_box),
            "resolved_source_inject_mask_box": _box_to_list(resolved_source_inject_mask_box),
            "resolved_final_preserve_box": _box_to_list(resolved_final_preserve_box),
            "edit_mask_box_mode": edit_mask_box_mode,
            "edit_mask_exclude_box": None
            if resolved_edit_mask_exclude_box is None
            else [float(v) for v in resolved_edit_mask_exclude_box],
            "edit_mask_use_core_as_subject": bool(edit_mask_use_core_as_subject),
            "external_edit_mask_path": external_edit_mask_path,
            "external_edit_mask_mode": external_edit_mask_mode,
            "mask_blend": bool(mask_blend),
            "mask_blend_mode": mask_blend_mode,
            **mask_stats,
            **core_mask_stats,
            **contact_mask_stats,
            **structure_edge_stats,
            **preserve_mask_stats,
            **source_inject_mask_stats,
            "edit_hedit_guidance_scale": float(edit_hedit_guidance_scale),
            "edit_field_mode": edit_field_mode,
            "edit_guidance_scale": float(edit_guidance_scale),
            "edit_region_guidance_scale": float(edit_region_guidance_scale),
            "edit_clip_guidance_scale": float(edit_clip_guidance_scale),
            "edit_text_guidance_scale": float(edit_text_guidance_scale),
            "edit_color_guidance_scale": float(edit_color_guidance_scale),
            "edit_color_source": None if color_source is None else color_source[0],
            "edit_color_target": None if color_target is None else color_target[0],
            "edit_color_mask_path": edit_color_mask_path,
            "edit_color_mask_threshold": float(edit_color_mask_threshold),
            "edit_color_mask_softness": float(edit_color_mask_softness),
            "edit_color_luma_gate_min": float(edit_color_luma_gate_min),
            "edit_color_luma_gate_softness": float(edit_color_luma_gate_softness),
            "edit_color_detail_protect_scale": float(edit_color_detail_protect_scale),
            "edit_color_detail_protect_threshold": float(edit_color_detail_protect_threshold),
            "edit_color_detail_protect_softness": float(edit_color_detail_protect_softness),
            "edit_color_target_chroma_scale": float(edit_color_target_chroma_scale),
            "edit_color_smooth_kernel": int(edit_color_smooth_kernel),
            "edit_color_luma_preserve_scale": float(edit_color_luma_preserve_scale),
            "edit_color_luma_gradient_preserve_scale": float(edit_color_luma_gradient_preserve_scale),
            **spatial_mask_stats(color_mask_latent, prefix="color_mask"),
            "edit_ref_guidance_scale": float(edit_ref_guidance_scale),
            "edit_ref_image_path": edit_ref_image_path,
            "edit_ref_mask_path": edit_ref_mask_path,
            "edit_ref_structure_image_path": edit_ref_structure_image_path,
            "edit_ref_chroma_mode": edit_ref_chroma_mode,
            "edit_ref_chroma_magnitude_scale": float(edit_ref_chroma_magnitude_scale),
            "edit_ref_luma_preserve_scale": float(edit_ref_luma_preserve_scale),
            "edit_ref_gradient_preserve_scale": float(edit_ref_gradient_preserve_scale),
            "edit_ref_darkness_guard_scale": float(edit_ref_darkness_guard_scale),
            "edit_ref_darkness_guard_margin": float(edit_ref_darkness_guard_margin),
            "edit_ref_smooth_kernel": int(edit_ref_smooth_kernel),
            "edit_ref_lowfreq_suppress_kernel": int(edit_ref_lowfreq_suppress_kernel),
            "edit_ref_lowfreq_suppress_scale": float(edit_ref_lowfreq_suppress_scale),
            "edit_ref_schedule_start": float(edit_ref_schedule_start),
            "edit_ref_schedule_stop": float(edit_ref_schedule_stop),
            "edit_ref_schedule_power": float(edit_ref_schedule_power),
            "edit_ref_max_struct_rms_ratio": float(edit_ref_max_struct_rms_ratio),
            "edit_ref_project_struct_conflict": float(edit_ref_project_struct_conflict),
            **spatial_mask_stats(ref_mask_latent, prefix="ref_mask"),
            "base_edit_norm": float(base_edit_norm),
            "edit_guidance_norm": float(edit_guidance_norm),
            "clip_guidance_norm": float(clip_guidance_norm),
            "text_guidance_norm": float(text_guidance_norm),
            "dds_guidance_norm": float(dds_guidance_norm),
            "app_guidance_norm": float(app_guidance_norm),
            "color_guidance_norm": float(color_guidance_norm),
            "ref_guidance_norm": float(ref_guidance_norm),
            "ref_guidance_pre_stabilize_norm": float(ref_guidance_pre_stabilize_norm),
            "ref_guidance_schedule_weight": float(ref_guidance_schedule_weight),
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
            "color_energy": 0.0 if color_edit_energy is None else float(color_edit_energy.item()),
            "ref_energy": 0.0 if ref_edit_energy is None else float(ref_edit_energy.item()),
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
                f"|v_color|={step_stat['color_guidance_norm']:.4f} "
                f"|v_ref|={step_stat['ref_guidance_norm']:.4f} "
                f"w_ref={step_stat['ref_guidance_schedule_weight']:.3f} "
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
                    f"E_app={step_stat['app_energy']:.6f} "
                    f"E_color={step_stat['color_energy']:.6f} "
                    f"E_ref={step_stat['ref_energy']:.6f} "
                    f"|v_ref_raw|={step_stat['ref_guidance_pre_stabilize_norm']:.4f}"
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
        active_edit_step += 1

    result = z_t

    if stats_output_path is not None:
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(step_stats, f, indent=2)
        print(f"[stats] saved ODE stats to {stats_output_path}")

    # Optional hard mask blend: paste source background over edited result.
    if mask_blend and M_preserve is not None:
        if mask_blend_mode == "subject":
            blend_edit = (1.0 - M_preserve).to(dtype=result.dtype, device=result.device)
        elif mask_blend_mode == "core":
            if M_core is None:
                raise ValueError("mask_blend_mode='core' requires an extracted core mask.")
            blend_edit = M_core.to(dtype=result.dtype, device=result.device)
        else:
            raise ValueError(f"Unsupported mask_blend_mode: {mask_blend_mode}")
        blend_preserve = (1.0 - blend_edit).clamp(0.0, 1.0)
        x_src_aligned = x_src.to(dtype=result.dtype, device=result.device)
        result = blend_edit * result + blend_preserve * x_src_aligned

    if resolved_final_preserve_box is not None:
        preserve_box_mask = normalized_box_mask_like(result[:, :1], resolved_final_preserve_box).to(
            dtype=result.dtype,
            device=result.device,
        )
        x_src_aligned = x_src.to(dtype=result.dtype, device=result.device)
        result = (1.0 - preserve_box_mask) * result + preserve_box_mask * x_src_aligned

    return result
