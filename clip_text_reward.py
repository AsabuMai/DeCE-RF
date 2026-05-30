from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


@dataclass
class CLIPReferenceState:
    source_image_features: torch.Tensor
    source_text_features: torch.Tensor
    target_text_features: torch.Tensor
    bbox: tuple[int, int, int, int] | None
    source_global_image_features: torch.Tensor


class LocalCLIPTextReward:
    def __init__(
        self,
        device: torch.device,
        model_name: str = "openai/clip-vit-large-patch14",
    ):
        self.device = device
        local_files_only = os.environ.get("RF_H_EDIT_ALLOW_CLIP_DOWNLOAD", "0") != "1"
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.image_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1)
        self._text_cache: dict[str, torch.Tensor] = {}

    def _encode_text(self, prompt: str) -> torch.Tensor:
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        tokens = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_cache[prompt] = text_features.detach()
        return self._text_cache[prompt]

    @staticmethod
    def _bbox_from_mask(mask: torch.Tensor, threshold: float = 0.35, pad: int = 8) -> tuple[int, int, int, int] | None:
        mask_2d = mask[0, 0].detach()
        active = mask_2d > threshold
        if not active.any():
            return None
        ys, xs = active.nonzero(as_tuple=True)
        y0 = max(int(ys.min().item()) - pad, 0)
        y1 = min(int(ys.max().item()) + pad + 1, mask_2d.shape[0])
        x0 = max(int(xs.min().item()) - pad, 0)
        x1 = min(int(xs.max().item()) + pad + 1, mask_2d.shape[1])
        return (y0, y1, x0, x1)

    def _mask_and_crop(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None,
        bbox: tuple[int, int, int, int] | None,
    ) -> torch.Tensor:
        if mask is not None:
            mask_img = F.interpolate(mask, size=images.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            images = images * mask_img
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            images = images[:, :, y0:y1, x0:x1]
        return images

    def _extract_patch_batch(
        self,
        images: torch.Tensor,
        grid_size: int = 2,
        min_patch: int = 64,
    ) -> torch.Tensor:
        _, _, h, w = images.shape
        if h < min_patch or w < min_patch:
            return images

        patches = []
        patch_h = max(h // grid_size, min_patch)
        patch_w = max(w // grid_size, min_patch)
        if patch_h >= h or patch_w >= w:
            return images

        ys = torch.linspace(0, h - patch_h, steps=grid_size, device=images.device).round().long()
        xs = torch.linspace(0, w - patch_w, steps=grid_size, device=images.device).round().long()
        for y in ys.tolist():
            for x in xs.tolist():
                patches.append(images[:, :, y : y + patch_h, x : x + patch_w])

        return torch.cat(patches, dim=0) if patches else images

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        image_features = self.model.get_image_features(pixel_values=images)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def prepare_reference(
        self,
        source_image: torch.Tensor,
        source_prompt: str,
        target_prompt: str,
        mask: torch.Tensor | None = None,
    ) -> CLIPReferenceState:
        mask_img = None
        if mask is not None:
            mask_img = F.interpolate(mask, size=source_image.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        bbox = self._bbox_from_mask(mask_img) if mask_img is not None else None
        source_cropped = self._mask_and_crop(source_image.detach(), mask_img, bbox)
        source_cropped = self._extract_patch_batch(source_cropped)
        with torch.no_grad():
            source_image_features = self._encode_image(source_cropped)
            source_global_image_features = self._encode_image(source_image.detach())
        return CLIPReferenceState(
            source_image_features=source_image_features.detach(),
            source_text_features=self._encode_text(source_prompt),
            target_text_features=self._encode_text(target_prompt),
            bbox=bbox,
            source_global_image_features=source_global_image_features.detach(),
        )

    def directional_text_loss(
        self,
        current_image: torch.Tensor,
        reference: CLIPReferenceState,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask_img = None
        if mask is not None:
            mask_img = F.interpolate(mask, size=current_image.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        current_cropped = self._mask_and_crop(current_image, mask_img, reference.bbox)
        current_cropped = self._extract_patch_batch(current_cropped)
        current_image_features = self._encode_image(current_cropped)

        source_ref = reference.source_image_features
        if source_ref.shape[0] != current_image_features.shape[0]:
            if source_ref.shape[0] == 1:
                source_ref = source_ref.expand(current_image_features.shape[0], -1)
            else:
                repeats = current_image_features.shape[0] // source_ref.shape[0]
                source_ref = source_ref.repeat(repeats, 1)

        image_direction = current_image_features - source_ref
        image_direction = image_direction / (image_direction.norm(dim=-1, keepdim=True) + 1e-6)

        text_direction = reference.target_text_features - reference.source_text_features
        text_direction = text_direction / (text_direction.norm(dim=-1, keepdim=True) + 1e-6)

        target_sim = (current_image_features * reference.target_text_features).sum(dim=-1)
        source_sim = (current_image_features * reference.source_text_features).sum(dim=-1)
        directional_sim = (image_direction * text_direction).sum(dim=-1)

        loss_target = 1.0 - target_sim
        loss_source = source_sim
        loss_direction = 1.0 - directional_sim
        total = loss_target.mean() + 0.5 * loss_source.mean() + loss_direction.mean()

        return {
            "total": total,
            "target": loss_target.mean(),
            "source": loss_source.mean(),
            "direction": loss_direction.mean(),
        }

    def semantic_text_loss(
        self,
        current_image: torch.Tensor,
        reference: CLIPReferenceState,
        core_mask: torch.Tensor | None = None,
        subject_mask: torch.Tensor | None = None,
        source_scale: float = 0.8,
        core_weight: float = 1.0,
        subject_weight: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        def _masked_similarity_terms(mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            mask_img = None
            if mask is not None:
                mask_img = F.interpolate(
                    mask,
                    size=current_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            current_cropped = self._mask_and_crop(current_image, mask_img, reference.bbox)
            current_cropped = self._extract_patch_batch(current_cropped)
            current_image_features = self._encode_image(current_cropped)
            target_sim = (current_image_features * reference.target_text_features).sum(dim=-1).mean()
            source_sim = (current_image_features * reference.source_text_features).sum(dim=-1).mean()
            return target_sim, source_sim

        target_core, source_core = _masked_similarity_terms(core_mask)
        target_subject, source_subject = _masked_similarity_terms(subject_mask)

        loss_core = (1.0 - target_core) + source_scale * source_core
        loss_subject = (1.0 - target_subject) + source_scale * source_subject
        total = core_weight * loss_core + subject_weight * loss_subject
        return {
            "total": total,
            "target_core": 1.0 - target_core,
            "source_core": source_core,
            "target_subject": 1.0 - target_subject,
            "source_subject": source_subject,
        }

    def strong_directional_text_loss(
        self,
        current_image: torch.Tensor,
        reference: CLIPReferenceState,
        mask: torch.Tensor | None = None,
        source_scale: float = 0.5,
        direction_scale: float = 1.0,
        global_target_scale: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        mask_img = None
        if mask is not None:
            mask_img = F.interpolate(mask, size=current_image.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        current_cropped = self._mask_and_crop(current_image, mask_img, reference.bbox)
        current_cropped = self._extract_patch_batch(current_cropped)
        current_local_features = self._encode_image(current_cropped)
        current_global_features = self._encode_image(current_image)

        source_local = reference.source_image_features
        if source_local.shape[0] != current_local_features.shape[0]:
            if source_local.shape[0] == 1:
                source_local = source_local.expand(current_local_features.shape[0], -1)
            else:
                repeats = current_local_features.shape[0] // source_local.shape[0]
                source_local = source_local.repeat(repeats, 1)

        target_text = reference.target_text_features
        source_text = reference.source_text_features

        local_target_sim = (current_local_features * target_text).sum(dim=-1)
        local_source_sim = (current_local_features * source_text).sum(dim=-1)
        global_target_sim = (current_global_features * target_text).sum(dim=-1)

        image_direction = current_local_features - source_local
        image_direction = image_direction / (image_direction.norm(dim=-1, keepdim=True) + 1e-6)
        text_direction = target_text - source_text
        text_direction = text_direction / (text_direction.norm(dim=-1, keepdim=True) + 1e-6)
        directional_sim = (image_direction * text_direction).sum(dim=-1)

        loss_local_target = 1.0 - local_target_sim
        loss_local_source = local_source_sim
        loss_direction = 1.0 - directional_sim
        loss_global_target = 1.0 - global_target_sim

        total = (
            loss_local_target.mean()
            + source_scale * loss_local_source.mean()
            + direction_scale * loss_direction.mean()
            + global_target_scale * loss_global_target.mean()
        )
        return {
            "total": total,
            "target": loss_local_target.mean(),
            "source": loss_local_source.mean(),
            "direction": loss_direction.mean(),
            "global_target": loss_global_target.mean(),
        }
