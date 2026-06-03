from __future__ import annotations

import contextlib
import re
from typing import Optional, Union

import torch
import torch.nn.functional as F


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

