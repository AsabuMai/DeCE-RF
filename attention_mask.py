from __future__ import annotations

import difflib
import re
from typing import Optional

import torch
import torch.nn.functional as F


class _CapturingJointAttnProcessor:
    """
    Drop-in replacement for JointAttnProcessor2_0 that captures image→text
    cross-attention maps and image self-attention maps without flash attention.
    """

    def __init__(self, store: "SD3AttentionStore", layer_idx: int, detach: bool = True):
        self.store = store
        self.layer_idx = layer_idx
        self.detach = detach

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]

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

        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([query, enc_q], dim=2)
            key = torch.cat([key, enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        # Manual softmax attention to capture weights (bypasses SDPA)
        scale = head_dim ** -0.5
        attn_weight = torch.softmax(
            torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale,
            dim=-1,
        ).to(query.dtype)  # (B, heads, total_seq, total_seq)

        if encoder_hidden_states is not None:
            # img tokens attending to txt tokens: (B, heads, img_seq, txt_seq)
            cross = attn_weight[:, :, :img_seq_len, img_seq_len:]
            cross = cross.mean(dim=1)
            if self.detach:
                cross = cross.detach()
            self.store.add_cross(self.layer_idx, cross.float())
            # img tokens attending to img tokens: (B, heads, img_seq, img_seq)
            self_attn = attn_weight[:, :, :img_seq_len, :img_seq_len]
            self_attn = self_attn.mean(dim=1)
            if self.detach:
                self_attn = self_attn.detach()
            self.store.add_self(self.layer_idx, self_attn.float())

        hidden_states = torch.matmul(attn_weight, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :img_seq_len],
                hidden_states[:, img_seq_len:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class SD3AttentionStore:
    """Temporarily replaces SD3 joint attention processors to capture attention maps."""

    def __init__(self):
        self._cross_maps: dict[int, list[torch.Tensor]] = {}
        self._self_maps: dict[int, list[torch.Tensor]] = {}
        self._orig_processors: dict[int, object] = {}

    def add_cross(self, layer_idx: int, attn_map: torch.Tensor):
        self._cross_maps.setdefault(layer_idx, []).append(attn_map)

    def add_self(self, layer_idx: int, attn_map: torch.Tensor):
        self._self_maps.setdefault(layer_idx, []).append(attn_map)

    def register(self, transformer, layer_indices: list[int], detach: bool = True):
        for i in layer_indices:
            block = transformer.transformer_blocks[i]
            self._orig_processors[i] = block.attn.processor
            block.attn.processor = _CapturingJointAttnProcessor(self, i, detach=detach)

    def restore(self, transformer):
        for i, proc in self._orig_processors.items():
            transformer.transformer_blocks[i].attn.processor = proc
        self._orig_processors.clear()

    def clear(self):
        self._cross_maps.clear()
        self._self_maps.clear()

    def aggregate_cross_spatial(
        self,
        img_h: int,
        img_w: int,
        token_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Average all captured cross-attention maps over layers and txt tokens.
        Returns (1, 1, img_h, img_w) float tensor in [0, 1].
        """
        all_maps = []
        for maps_per_layer in self._cross_maps.values():
            for m in maps_per_layer:
                if token_indices:
                    valid = [idx for idx in token_indices if 0 <= idx < m.shape[-1]]
                    if valid:
                        all_maps.append(m[..., valid].mean(dim=-1))
                        continue
                # m: (B, img_seq, txt_seq) → mean over txt → (B, img_seq)
                all_maps.append(m.mean(dim=-1))

        if not all_maps:
            return torch.zeros(1, 1, img_h, img_w)

        avg = torch.stack(all_maps, dim=0).mean(dim=0)  # (B, img_seq)
        # Use uncond or first batch element
        avg = avg[0]  # (img_seq,)
        avg = avg.reshape(1, 1, img_h, img_w)

        mn, mx = avg.min(), avg.max()
        return (avg - mn) / (mx - mn + 1e-6)

    def aggregate_self_spatial(self, img_h: int, img_w: int) -> torch.Tensor:
        """
        Average all captured image self-attention maps over layers.

        We collapse the query dimension to obtain a token saliency map over the
        image grid. This is still cheap, but carries more geometry/layout
        information than cross-attention alone.
        """
        all_maps = []
        for maps_per_layer in self._self_maps.values():
            for m in maps_per_layer:
                # m: (B, img_seq, img_seq) -> mean over query positions -> (B, img_seq)
                all_maps.append(m.mean(dim=-2))

        if not all_maps:
            return torch.zeros(1, 1, img_h, img_w)

        avg = torch.stack(all_maps, dim=0).mean(dim=0)
        avg = avg[0]
        avg = avg.reshape(1, 1, img_h, img_w)

        mn, mx = avg.min(), avg.max()
        return (avg - mn) / (mx - mn + 1e-6)


class SD3FeatureStore:
    """Captures mid-block image token features from SD3 transformer blocks."""

    def __init__(self):
        self._features: dict[int, list[torch.Tensor]] = {}
        self._handles = []

    def add(self, layer_idx: int, feature_map: torch.Tensor):
        self._features.setdefault(layer_idx, []).append(feature_map)

    def register(self, transformer, layer_indices: list[int]):
        for i in layer_indices:
            block = transformer.transformer_blocks[i]

            def _make_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    # JointTransformerBlock returns (encoder_hidden_states, hidden_states)
                    hidden_states = output[1]
                    self.add(layer_idx, hidden_states.detach().float())

                return _hook

            handle = block.register_forward_hook(_make_hook(i))
            self._handles.append(handle)

    def restore(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def aggregate_spatial(self, img_h: int, img_w: int) -> torch.Tensor:
        """
        Convert captured token features into a spatial saliency map.

        We use the per-token L2 norm of the hidden state as a cheap visual
        structure proxy, then average across layers.
        """
        all_maps = []
        for feats_per_layer in self._features.values():
            for feat in feats_per_layer:
                token_norm = torch.linalg.norm(feat, dim=-1)  # (B, img_seq)
                all_maps.append(token_norm)

        if not all_maps:
            return torch.zeros(1, 1, img_h, img_w)

        avg = torch.stack(all_maps, dim=0).mean(dim=0)
        avg = avg[0].reshape(1, 1, img_h, img_w)
        mn, mx = avg.min(), avg.max()
        return (avg - mn) / (mx - mn + 1e-6)

def _changed_words(src_prompt: str, tar_prompt: str) -> tuple[list[str], list[str]]:
    src_words = re.findall(r"\w+", src_prompt.lower())
    tar_words = re.findall(r"\w+", tar_prompt.lower())
    matcher = difflib.SequenceMatcher(None, src_words, tar_words)
    src_changed: list[str] = []
    tar_changed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            src_changed.extend(src_words[i1:i2])
        if tag in {"replace", "insert"}:
            tar_changed.extend(tar_words[j1:j2])
    return src_changed, tar_changed


def _token_indices_for_words(pipe, prompt: str, words: list[str]) -> list[int]:
    if not words:
        return []
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return []
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].tolist()
    token_texts = [tokenizer.decode([tok]).strip().lower() for tok in token_ids]
    indices: list[int] = []
    for idx, piece in enumerate(token_texts):
        normalized = piece.replace("</w>", "").replace("Ġ", "").replace("▁", "").strip()
        if not normalized:
            continue
        if any(word in normalized or normalized in word for word in words):
            indices.append(idx)
    return sorted(set(indices))


def _extract_single_prompt_attention_maps(
    pipe,
    x_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    token_indices: list[int] | None = None,
    max_latent_side: int = 128,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Capture source-conditioned cross/self attention structure maps.
    """
    orig_size = x_latent.shape[-2:]
    transformer_dtype = next(pipe.transformer.parameters()).dtype

    h, w = x_latent.shape[-2], x_latent.shape[-1]
    if h > max_latent_side or w > max_latent_side:
        scale = max_latent_side / max(h, w)
        new_h = max(2, int(h * scale) // 2 * 2)
        new_w = max(2, int(w * scale) // 2 * 2)
        x_small = F.interpolate(
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

    store = SD3AttentionStore()
    grad_ctx = torch.no_grad() if detach else torch.enable_grad()
    with grad_ctx:
        try:
            store.register(pipe.transformer, layer_indices, detach=detach)
            pipe.transformer(
                hidden_states=x_small,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )
        finally:
            store.restore(pipe.transformer)

    cross_map = store.aggregate_cross_spatial(img_h, img_w, token_indices=token_indices).to(x_latent.device)
    self_map = store.aggregate_self_spatial(img_h, img_w).to(x_latent.device)
    cross_map = F.interpolate(cross_map, size=orig_size, mode="bilinear", align_corners=False)
    self_map = F.interpolate(self_map, size=orig_size, mode="bilinear", align_corners=False)
    return cross_map, self_map


def extract_prompt_attention_map(
    pipe,
    x_latent: torch.Tensor,
    prompt: str,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    token_words: list[str] | None = None,
    max_latent_side: int = 128,
    detach: bool = True,
) -> torch.Tensor:
    token_indices = _token_indices_for_words(pipe, prompt, token_words or [])
    cross_map, _ = _extract_single_prompt_attention_maps(
        pipe=pipe,
        x_latent=x_latent,
        prompt_embeds=prompt_embeds,
        pooled_embeds=pooled_embeds,
        t=t,
        token_indices=token_indices,
        max_latent_side=max_latent_side,
        detach=detach,
    )
    return cross_map


def extract_source_feature_structure_map(
    pipe,
    x_latent: torch.Tensor,
    src_prompt_embeds: torch.Tensor,
    src_pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    max_latent_side: int = 128,
) -> torch.Tensor:
    """
    Capture a stronger structure map from SD3 mid-block visual features.

    Compared with attention-only maps, block output features carry more local
    geometry/layout information and are therefore a better reconstruction-side
    structure anchor.
    """
    orig_size = x_latent.shape[-2:]

    h, w = x_latent.shape[-2], x_latent.shape[-1]
    if h > max_latent_side or w > max_latent_side:
        scale = max_latent_side / max(h, w)
        new_h = max(2, int(h * scale) // 2 * 2)
        new_w = max(2, int(w * scale) // 2 * 2)
        x_small = F.interpolate(
            x_latent.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).to(x_latent.dtype)
    else:
        x_small = x_latent

    n_blocks = len(pipe.transformer.transformer_blocks)
    lo, hi = n_blocks // 4, 3 * n_blocks // 4
    layer_indices = list(range(lo, hi))
    img_h = x_small.shape[-2] // 2
    img_w = x_small.shape[-1] // 2
    timestep = t.expand(x_small.shape[0])

    store = SD3FeatureStore()
    with torch.no_grad():
        try:
            store.register(pipe.transformer, layer_indices)
            pipe.transformer(
                hidden_states=x_small,
                timestep=timestep,
                encoder_hidden_states=src_prompt_embeds,
                pooled_projections=src_pooled_embeds,
                return_dict=False,
            )
        finally:
            store.restore()

    feature_map = store.aggregate_spatial(img_h, img_w).to(x_latent.device)
    feature_map = F.interpolate(feature_map, size=orig_size, mode="bilinear", align_corners=False)
    return feature_map


def extract_source_structure_map(
    pipe,
    x_latent: torch.Tensor,
    src_prompt_embeds: torch.Tensor,
    src_pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    max_latent_side: int = 128,
) -> torch.Tensor:
    cross_map, self_map = _extract_single_prompt_attention_maps(
        pipe=pipe,
        x_latent=x_latent,
        prompt_embeds=src_prompt_embeds,
        pooled_embeds=src_pooled_embeds,
        t=t,
        max_latent_side=max_latent_side,
    )
    fused = 0.5 * cross_map + 0.5 * self_map
    mn, mx = fused.min(), fused.max()
    return (fused - mn) / (mx - mn + 1e-6)


def extract_attention_masks(
    pipe,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    src_prompt_embeds: torch.Tensor,
    src_pooled_embeds: torch.Tensor,
    tar_prompt_embeds: torch.Tensor,
    tar_pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    sharpness: float = 10.0,
    max_latent_side: int = 128,
) -> dict[str, torch.Tensor]:
    """
    Extract a soft editing mask M from SD3 cross-attention maps.

    Runs two lightweight forward passes (src prompt, tar prompt) with manual
    attention to capture img→txt attention maps from the middle transformer
    blocks. The editing mask M is derived from A_src + A_tar: both prompts
    attend to the subject region, while background has low attention from both.

    max_latent_side: latent is downscaled to at most this size before the
        attention pass to avoid OOM on large images. The mask is upsampled
        back to the original latent resolution afterward.

    Returns a soft spatial decomposition where `subject` is the editable mask.
    `core` is a higher-confidence semantic subset inside the editable subject.
    """
    src_changed_words, tar_changed_words = _changed_words(src_prompt, tar_prompt)
    src_token_indices = _token_indices_for_words(pipe, src_prompt, src_changed_words)
    tar_token_indices = _token_indices_for_words(pipe, tar_prompt, tar_changed_words)

    A_src_cross, A_src_self = _extract_single_prompt_attention_maps(
        pipe=pipe,
        x_latent=x_src,
        prompt_embeds=src_prompt_embeds,
        pooled_embeds=src_pooled_embeds,
        t=t,
        token_indices=src_token_indices,
        max_latent_side=max_latent_side,
    )
    A_tar_cross, A_tar_self = _extract_single_prompt_attention_maps(
        pipe=pipe,
        x_latent=x_src,
        prompt_embeds=tar_prompt_embeds,
        pooled_embeds=tar_pooled_embeds,
        t=t,
        token_indices=tar_token_indices,
        max_latent_side=max_latent_side,
    )
    A_src = 0.8 * A_src_cross + 0.2 * A_src_self
    A_tar = 0.8 * A_tar_cross + 0.2 * A_tar_self

    combined = torch.maximum(A_src, A_tar)
    mn, mx = combined.min(), combined.max()
    combined = (combined - mn) / (mx - mn + 1e-6)
    combined = F.max_pool2d(combined, kernel_size=3, stride=1, padding=1)
    combined = combined / (combined.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # Keep the same attention-derived mask logic, but tighten the editable
    # region so the subject mask does not swallow nearly half the image.
    # This is intended to reduce background/style drift without changing the
    # reconstruction/editing math downstream.
    hard_subject = (combined > 0.48).to(combined.dtype)
    soft_subject = torch.sigmoid((sharpness + 1.0) * (combined - 0.52))
    M_subject = (0.8 * hard_subject + 0.2 * soft_subject).clamp(0.0, 1.0)
    hard_core = (combined > 0.72).to(combined.dtype)
    soft_core = torch.sigmoid((sharpness + 3.0) * (combined - 0.76))
    M_core = (0.85 * hard_core + 0.15 * soft_core).clamp(0.0, 1.0)
    M_core = torch.minimum(M_core, M_subject)
    M_preserve = (1.0 - M_subject).clamp(0.0, 1.0)

    return {
        "subject": M_subject,
        "core": M_core,
        "preserve": M_preserve,
    }


def extract_attention_mask(
    pipe,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    src_prompt_embeds: torch.Tensor,
    src_pooled_embeds: torch.Tensor,
    tar_prompt_embeds: torch.Tensor,
    tar_pooled_embeds: torch.Tensor,
    t: torch.Tensor,
    sharpness: float = 10.0,
    max_latent_side: int = 128,
) -> torch.Tensor:
    masks = extract_attention_masks(
        pipe=pipe,
        x_src=x_src,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        src_prompt_embeds=src_prompt_embeds,
        src_pooled_embeds=src_pooled_embeds,
        tar_prompt_embeds=tar_prompt_embeds,
        tar_pooled_embeds=tar_pooled_embeds,
        t=t,
        sharpness=sharpness,
        max_latent_side=max_latent_side,
    )
    return masks["subject"]
