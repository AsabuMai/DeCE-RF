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


_EDIT_WORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "black",
    "blue",
    "brown",
    "by",
    "for",
    "gray",
    "green",
    "grey",
    "in",
    "is",
    "large",
    "little",
    "on",
    "orange",
    "over",
    "pink",
    "purple",
    "red",
    "small",
    "the",
    "to",
    "under",
    "wearing",
    "white",
    "with",
    "yellow",
}


def _content_edit_words(words: list[str], max_words: int = 3) -> list[str]:
    content = [word for word in words if word and word not in _EDIT_WORD_STOPWORDS and len(word) > 1]
    if not content:
        return words[-max_words:]
    return content[-max_words:]


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


def _normalize_map(attn_map: torch.Tensor) -> torch.Tensor:
    mn, mx = attn_map.min(), attn_map.max()
    return (attn_map - mn) / (mx - mn + 1e-6)


def _fuse_cross_self(
    cross_map: torch.Tensor,
    self_map: torch.Tensor,
    self_weight: float = 0.2,
) -> torch.Tensor:
    self_weight = max(0.0, min(1.0, self_weight))
    return _normalize_map((1.0 - self_weight) * cross_map + self_weight * self_map)


def _mask_from_attention_map(
    attn_map: torch.Tensor,
    sharpness: float,
    subject_threshold: float = 0.48,
    core_threshold: float = 0.72,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = _normalize_map(attn_map)
    combined = F.max_pool2d(combined, kernel_size=3, stride=1, padding=1)
    combined = combined / (combined.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    subject_threshold = max(0.0, min(1.0, subject_threshold))
    core_threshold = max(subject_threshold, min(1.0, core_threshold))
    subject_soft_threshold = min(1.0, subject_threshold + 0.04)
    core_soft_threshold = min(1.0, core_threshold + 0.04)

    hard_subject = (combined > subject_threshold).to(combined.dtype)
    soft_subject = torch.sigmoid((sharpness + 1.0) * (combined - subject_soft_threshold))
    subject = (0.8 * hard_subject + 0.2 * soft_subject).clamp(0.0, 1.0)
    hard_core = (combined > core_threshold).to(combined.dtype)
    soft_core = torch.sigmoid((sharpness + 3.0) * (combined - core_soft_threshold))
    core = (0.85 * hard_core + 0.15 * soft_core).clamp(0.0, 1.0)
    core = torch.minimum(core, subject)
    return subject, core


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
    mode: str = "changed_union",
    target_token_words: list[str] | None = None,
    source_token_words: list[str] | None = None,
    subject_threshold: float = 0.48,
    core_threshold: float = 0.72,
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

    `mode` controls which attention map becomes the editable mask:
    - changed_union: source changed-token and target changed-token union.
    - target_changed: target changed-token map only.
    - subject_union: source subject and target subject union.
    - source_subject: source subject map only.
    - target_subject: target subject map only.

    Returns a soft spatial decomposition where `subject` is the editable mask.
    `core` is a higher-confidence semantic subset inside the editable subject.
    """
    src_changed_words, tar_changed_words = _changed_words(src_prompt, tar_prompt)
    if source_token_words is not None:
        src_changed_words = source_token_words
    else:
        src_changed_words = _content_edit_words(src_changed_words)
    if target_token_words is not None:
        tar_changed_words = target_token_words
    else:
        tar_changed_words = _content_edit_words(tar_changed_words)
    src_token_indices = _token_indices_for_words(pipe, src_prompt, src_changed_words)
    tar_token_indices = _token_indices_for_words(pipe, tar_prompt, tar_changed_words)

    A_src_subject_cross, A_src_self = _extract_single_prompt_attention_maps(
        pipe=pipe,
        x_latent=x_src,
        prompt_embeds=src_prompt_embeds,
        pooled_embeds=src_pooled_embeds,
        t=t,
        token_indices=None,
        max_latent_side=max_latent_side,
    )
    A_tar_subject_cross, A_tar_self = _extract_single_prompt_attention_maps(
        pipe=pipe,
        x_latent=x_src,
        prompt_embeds=tar_prompt_embeds,
        pooled_embeds=tar_pooled_embeds,
        t=t,
        token_indices=None,
        max_latent_side=max_latent_side,
    )
    A_src_subject = _fuse_cross_self(A_src_subject_cross, A_src_self)
    A_tar_subject = _fuse_cross_self(A_tar_subject_cross, A_tar_self)

    if src_token_indices:
        A_src_changed_cross, _ = _extract_single_prompt_attention_maps(
            pipe=pipe,
            x_latent=x_src,
            prompt_embeds=src_prompt_embeds,
            pooled_embeds=src_pooled_embeds,
            t=t,
            token_indices=src_token_indices,
            max_latent_side=max_latent_side,
        )
        A_src_changed = _fuse_cross_self(A_src_changed_cross, A_src_self, self_weight=0.0)
    else:
        A_src_changed = torch.zeros_like(A_src_subject)

    if tar_token_indices:
        A_tar_changed_cross, _ = _extract_single_prompt_attention_maps(
            pipe=pipe,
            x_latent=x_src,
            prompt_embeds=tar_prompt_embeds,
            pooled_embeds=tar_pooled_embeds,
            t=t,
            token_indices=tar_token_indices,
            max_latent_side=max_latent_side,
        )
        A_tar_changed = _fuse_cross_self(A_tar_changed_cross, A_tar_self, self_weight=0.0)
    else:
        A_tar_changed = torch.zeros_like(A_tar_subject)

    if mode == "changed_union":
        combined = torch.maximum(A_src_changed, A_tar_changed)
        if combined.amax() <= 1e-6:
            combined = torch.maximum(A_src_subject, A_tar_subject)
    elif mode == "target_changed":
        combined = A_tar_changed if A_tar_changed.amax() > 1e-6 else A_tar_subject
    elif mode == "subject_union":
        combined = torch.maximum(A_src_subject, A_tar_subject)
    elif mode == "source_subject":
        combined = A_src_subject
    elif mode == "target_subject":
        combined = A_tar_subject
    else:
        raise ValueError(f"Unsupported attention mask mode: {mode}")

    M_subject, M_core = _mask_from_attention_map(
        combined,
        sharpness=sharpness,
        subject_threshold=subject_threshold,
        core_threshold=core_threshold,
    )
    M_preserve = (1.0 - M_subject).clamp(0.0, 1.0)

    return {
        "subject": M_subject,
        "core": M_core,
        "preserve": M_preserve,
        "source_subject": A_src_subject,
        "target_subject": A_tar_subject,
        "source_changed": A_src_changed,
        "target_changed": A_tar_changed,
        "combined": _normalize_map(combined),
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
