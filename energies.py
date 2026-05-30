from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


VelocityConversionMode = Literal["legacy", "linear_path"]


def expand_t_like(t_scalar: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=target.device, dtype=target.dtype)
    t_scalar = t_scalar.to(device=target.device, dtype=target.dtype).clamp_min(eps)
    while t_scalar.ndim < target.ndim:
        t_scalar = t_scalar.view(*t_scalar.shape, 1)
    return t_scalar


def clean_delta_to_velocity(
    delta_x0: torch.Tensor,
    t_scalar: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Convert a desired clean-estimate displacement into a velocity correction.

    RF linear path convention:
        x0_hat = x_t - t * v

    If v' = v + u, then:
        x0_hat' - x0_hat = -t * u

    Therefore:
        u = -delta_x0 / t
    """
    t = expand_t_like(t_scalar, delta_x0, eps=eps)
    return -delta_x0 / t


def cosine_safe(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    denom = a_flat.norm() * b_flat.norm() + eps
    return float(torch.dot(a_flat, b_flat) / denom)


def _align_like(reference: torch.Tensor, tensor: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """
    Match the spatial size of `tensor` to `reference`.

    SD3 latent tensors can end up with different packed spatial layouts across
    different stages of the pipeline. For the current surrogate reconstruction
    term, we only need a stable geometry-aware comparison space, so spatial
    alignment is sufficient.
    """
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    return F.interpolate(tensor, size=reference.shape[-2:], mode=mode)


def align_rec_space(
    x0_pred: torch.Tensor,
    x_src: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Canonicalize all reconstruction-side tensors into the same spatial space.

    We always treat `x0_pred` as the reference reconstruction space because it
    is the clean estimate produced by the current RF path step. Both the source
    latent and the optional preservation mask are resized to match it.
    """
    x_src_aligned = _align_like(x0_pred, x_src, mode="nearest")
    mask_aligned = None if mask is None else _align_like(x0_pred, mask, mode="bilinear")
    return x0_pred, x_src_aligned, mask_aligned


def align_energy_space(
    reference: torch.Tensor,
    x_src: torch.Tensor | None = None,
    M_edit: torch.Tensor | None = None,
    M_preserve: torch.Tensor | None = None,
    current_structure_map: torch.Tensor | None = None,
    source_structure_map: torch.Tensor | None = None,
    current_feature_map: torch.Tensor | None = None,
    source_feature_map: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """
    Canonicalize all energy-side tensors into one shared spatial space.

    This makes the mathematical objects explicit: the same reconstruction/edit
    space is used for x_hat_0, source latent, masks, and optional structure maps.
    """
    return {
        "reference": reference,
        "x_src": None if x_src is None else _align_like(reference, x_src, mode="nearest"),
        "M_edit": None if M_edit is None else _align_like(reference, M_edit, mode="bilinear"),
        "M_preserve": None if M_preserve is None else _align_like(reference, M_preserve, mode="bilinear"),
        "current_structure_map": None
        if current_structure_map is None
        else _align_like(reference, current_structure_map, mode="bilinear"),
        "source_structure_map": None
        if source_structure_map is None
        else _align_like(reference, source_structure_map, mode="bilinear"),
        "current_feature_map": None
        if current_feature_map is None
        else _align_like(reference, current_feature_map, mode="bilinear"),
        "source_feature_map": None
        if source_feature_map is None
        else _align_like(reference, source_feature_map, mode="bilinear"),
    }


def reconstruction_energy_multiscale(
    x0_pred: torch.Tensor,
    x_src: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    A structure-aware latent reconstruction surrogate.

    mask: (B, 1, H, W) preservation weight in [0,1].
          1 = preserve (full E_rec), 0 = editing region (E_rec suppressed).
          If None, full-image reconstruction is applied.
    """
    total = 0.0
    count = 0
    x0_pred, x_src, mask = align_rec_space(x0_pred, x_src, mask)

    for scale in scales:
        if scale == 1:
            pred_level = x0_pred
            src_level = x_src
            w = mask
        else:
            target_h = max(1, x0_pred.shape[-2] // scale)
            target_w = max(1, x0_pred.shape[-1] // scale)
            pred_level = F.adaptive_avg_pool2d(x0_pred, output_size=(target_h, target_w))
            src_level = F.adaptive_avg_pool2d(x_src, output_size=(target_h, target_w))
            w = F.adaptive_avg_pool2d(mask, output_size=(target_h, target_w)) if mask is not None else None

        diff = pred_level - src_level
        if w is not None:
            diff = diff * w
        level_loss = 0.5 * diff.pow(2).mean()
        total = total + level_loss
        count += 1

    total = total / count
    return total


def reconstruction_velocity_surrogate_multiscale(
    x0_pred: torch.Tensor,
    x_src: torch.Tensor,
    t_scalar: torch.Tensor | None = None,
    scales: tuple[int, ...] = (1, 2, 4),
    upsample_mode: str = "nearest",
    mask: torch.Tensor = None,
    velocity_conversion_mode: VelocityConversionMode = "linear_path",
    velocity_t_min: float = 1e-3,
) -> torch.Tensor:
    """
    Cheap surrogate for -∇E_rec in x_hat_0 space.

    mask: (B, 1, H, W) preservation weight in [0,1].
          1 = preserve (full correction), 0 = editing region (correction suppressed).
    """
    residuals = []
    x0_pred, x_src, mask = align_rec_space(x0_pred, x_src, mask)

    for scale in scales:
        if scale == 1:
            residuals.append(x0_pred - x_src)
            continue

        target_h = max(1, x0_pred.shape[-2] // scale)
        target_w = max(1, x0_pred.shape[-1] // scale)
        pred_level = F.adaptive_avg_pool2d(x0_pred, output_size=(target_h, target_w))
        src_level = F.adaptive_avg_pool2d(x_src, output_size=(target_h, target_w))
        coarse_residual = pred_level - src_level
        residuals.append(
            F.interpolate(
                coarse_residual,
                size=x0_pred.shape[-2:],
                mode=upsample_mode,
            )
        )

    residual = torch.stack(residuals, dim=0).mean(dim=0)
    if velocity_conversion_mode == "legacy":
        v = -residual
    elif velocity_conversion_mode == "linear_path":
        if t_scalar is None:
            raise ValueError("t_scalar is required for linear_path reconstruction velocity conversion")
        # Desired clean-space displacement is x_src - x0_pred == -residual.
        v = clean_delta_to_velocity(-residual, t_scalar, eps=velocity_t_min)
    else:
        raise ValueError(f"Unsupported velocity_conversion_mode: {velocity_conversion_mode}")
    if mask is not None:
        v = v * mask
    return v


def reconstruction_energy_structure_map(
    current_map: torch.Tensor,
    source_map: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Structure-preserving reconstruction term based on source-conditioned
    attention structure maps.
    """
    source_map = _align_like(current_map, source_map, mode="bilinear")
    if mask is not None:
        mask = _align_like(current_map, mask, mode="bilinear")
    diff = current_map - source_map
    if mask is not None:
        diff = diff * mask
    return 0.5 * diff.pow(2).mean()


def reconstruction_velocity_surrogate_structure(
    current_map: torch.Tensor,
    source_map: torch.Tensor,
    reference_latent: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cheap spatial structure correction in x_t space.

    The attention-map difference is expanded across latent channels so it can
    act as an additional spatial reconstruction bias.
    """
    source_map = _align_like(current_map, source_map, mode="bilinear")
    if mask is not None:
        mask = _align_like(current_map, mask, mode="bilinear")

    diff = current_map - source_map
    if mask is not None:
        diff = diff * mask

    diff = _align_like(reference_latent, diff, mode="bilinear")
    return -diff.expand_as(reference_latent)


def reconstruction_energy_total(
    x0_pred: torch.Tensor,
    x_src: torch.Tensor,
    M_preserve: torch.Tensor | None = None,
    current_structure_map: torch.Tensor | None = None,
    source_structure_map: torch.Tensor | None = None,
    current_feature_map: torch.Tensor | None = None,
    source_feature_map: torch.Tensor | None = None,
    lambda_latent: float = 1.0,
    lambda_struct: float = 0.5,
    lambda_feature: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Unified reconstruction-side energy decomposition.

    Returns the total energy plus named sub-terms so the caller can log or
    ablate them without re-implementing the composition logic.
    """
    aligned = align_energy_space(
        reference=x0_pred,
        x_src=x_src,
        M_preserve=M_preserve,
        current_structure_map=current_structure_map,
        source_structure_map=source_structure_map,
        current_feature_map=current_feature_map,
        source_feature_map=source_feature_map,
    )
    x0_pred = aligned["reference"]
    x_src = aligned["x_src"]
    M_preserve = aligned["M_preserve"]
    current_structure_map = aligned["current_structure_map"]
    source_structure_map = aligned["source_structure_map"]
    current_feature_map = aligned["current_feature_map"]
    source_feature_map = aligned["source_feature_map"]

    e_latent = reconstruction_energy_multiscale(x0_pred, x_src, mask=M_preserve)
    e_struct = torch.zeros_like(e_latent)
    e_feature = torch.zeros_like(e_latent)
    if current_structure_map is not None and source_structure_map is not None:
        e_struct = reconstruction_energy_structure_map(
            current_map=current_structure_map,
            source_map=source_structure_map,
            mask=M_preserve,
        )
    if current_feature_map is not None and source_feature_map is not None:
        e_feature = reconstruction_energy_structure_map(
            current_map=current_feature_map,
            source_map=source_feature_map,
            mask=M_preserve,
        )

    total = lambda_latent * e_latent + lambda_struct * e_struct + lambda_feature * e_feature
    return {
        "total": total,
        "latent": e_latent,
        "struct": e_struct,
        "feature": e_feature,
    }


def reconstruction_velocity_surrogate_total(
    x0_pred: torch.Tensor,
    x_src: torch.Tensor,
    t_scalar: torch.Tensor | None = None,
    M_preserve: torch.Tensor | None = None,
    current_structure_map: torch.Tensor | None = None,
    source_structure_map: torch.Tensor | None = None,
    current_feature_map: torch.Tensor | None = None,
    source_feature_map: torch.Tensor | None = None,
    lambda_latent: float = 1.0,
    lambda_struct: float = 0.5,
    lambda_feature: float = 0.5,
    velocity_conversion_mode: VelocityConversionMode = "linear_path",
    velocity_t_min: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """
    Unified reconstruction-side surrogate velocity decomposition.
    """
    aligned = align_energy_space(
        reference=x0_pred,
        x_src=x_src,
        M_preserve=M_preserve,
        current_structure_map=current_structure_map,
        source_structure_map=source_structure_map,
        current_feature_map=current_feature_map,
        source_feature_map=source_feature_map,
    )
    x0_pred = aligned["reference"]
    x_src = aligned["x_src"]
    M_preserve = aligned["M_preserve"]
    current_structure_map = aligned["current_structure_map"]
    source_structure_map = aligned["source_structure_map"]
    current_feature_map = aligned["current_feature_map"]
    source_feature_map = aligned["source_feature_map"]

    v_latent = reconstruction_velocity_surrogate_multiscale(
        x0_pred,
        x_src,
        t_scalar=t_scalar,
        mask=M_preserve,
        velocity_conversion_mode=velocity_conversion_mode,
        velocity_t_min=velocity_t_min,
    )
    v_struct = torch.zeros_like(v_latent)
    v_feature = torch.zeros_like(v_latent)
    if current_structure_map is not None and source_structure_map is not None:
        v_struct = reconstruction_velocity_surrogate_structure(
            current_map=current_structure_map,
            source_map=source_structure_map,
            reference_latent=x0_pred,
            mask=M_preserve,
        )
    if current_feature_map is not None and source_feature_map is not None:
        v_feature = reconstruction_velocity_surrogate_structure(
            current_map=current_feature_map,
            source_map=source_feature_map,
            reference_latent=x0_pred,
            mask=M_preserve,
        )

    total = lambda_latent * v_latent + lambda_struct * v_struct + lambda_feature * v_feature
    return {
        "total": total,
        "latent": v_latent,
        "struct": v_struct,
        "feature": v_feature,
    }


def editing_energy_target_anchor(
    x0_tar: torch.Tensor,
    x0_src: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Minimal target-conditioned edit-side surrogate energy.

    Measures how much the target-conditioned clean estimate separates from the
    source-conditioned clean estimate inside the editing region. This is still
    a surrogate, but it is explicitly target-conditioned in x_hat_0 space.
    """
    x0_tar, x0_src, mask = align_rec_space(x0_tar, x0_src, mask)
    diff = x0_tar - x0_src
    if mask is not None:
        diff = diff * mask
    return 0.5 * diff.pow(2).mean()


def editing_energy_region_delta(
    x0_tar: torch.Tensor,
    x0_src: torch.Tensor,
    M_edit: torch.Tensor | None = None,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """
    Region-specific edit energy in x_hat_0 space.

    This uses the same multiscale layout as reconstruction, but focuses on the
    editing region instead of the preserved region.
    """
    aligned = align_energy_space(reference=x0_tar, x_src=x0_src, M_edit=M_edit)
    x0_tar = aligned["reference"]
    x0_src = aligned["x_src"]
    M_edit = aligned["M_edit"]

    total = 0.0
    count = 0
    for scale in scales:
        if scale == 1:
            tar_level = x0_tar
            src_level = x0_src
            w = M_edit
        else:
            target_h = max(1, x0_tar.shape[-2] // scale)
            target_w = max(1, x0_tar.shape[-1] // scale)
            tar_level = F.adaptive_avg_pool2d(x0_tar, output_size=(target_h, target_w))
            src_level = F.adaptive_avg_pool2d(x0_src, output_size=(target_h, target_w))
            w = F.adaptive_avg_pool2d(M_edit, output_size=(target_h, target_w)) if M_edit is not None else None

        diff = tar_level - src_level
        if w is not None:
            diff = diff * w
        total = total + 0.5 * diff.pow(2).mean()
        count += 1

    return total / max(count, 1)


def editing_energy_target_attract(
    target_feature_map: torch.Tensor,
    M_edit: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Region-specific target attraction energy from prompt-conditioned features.

    Inside the edit region, target-conditioned responses should be strong.
    """
    if M_edit is not None:
        M_edit = _align_like(target_feature_map, M_edit, mode="bilinear")

    target_term = (1.0 - target_feature_map).pow(2)
    if M_edit is not None:
        target_term = target_term * M_edit
    return 0.5 * target_term.mean()


def editing_energy_source_suppress(
    source_feature_map: torch.Tensor,
    M_edit: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Region-specific source suppression energy from prompt-conditioned features.

    Inside the edit region, source-conditioned responses should be weak.
    """
    if M_edit is not None:
        M_edit = _align_like(source_feature_map, M_edit, mode="bilinear")

    source_term = source_feature_map.pow(2)
    if M_edit is not None:
        source_term = source_term * M_edit
    return 0.5 * source_term.mean()


def editing_velocity_surrogate_target_anchor(
    x0_tar: torch.Tensor,
    x0_src: torch.Tensor,
    t_scalar: torch.Tensor,
    mask: torch.Tensor | None = None,
    boost: float = 1.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Map a target-vs-source clean-estimate gap back into x_t-space.

    Under the linear RF path x_hat_0 = x_t - t * v, a clean-estimate gap can be
    approximately converted into a velocity correction by dividing by t.
    """
    x0_tar, x0_src, mask = align_rec_space(x0_tar, x0_src, mask)
    delta = x0_tar - x0_src
    if mask is not None:
        delta = delta * mask

    return boost * clean_delta_to_velocity(delta, t_scalar, eps=eps)


def editing_velocity_surrogate_region_delta(
    x0_tar: torch.Tensor,
    x0_src: torch.Tensor,
    t_scalar: torch.Tensor,
    M_edit: torch.Tensor | None = None,
    scales: tuple[int, ...] = (1, 2, 4),
    upsample_mode: str = "nearest",
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Region-specific edit surrogate derived from masked multiscale x_hat_0 gaps.
    """
    aligned = align_energy_space(reference=x0_tar, x_src=x0_src, M_edit=M_edit)
    x0_tar = aligned["reference"]
    x0_src = aligned["x_src"]
    M_edit = aligned["M_edit"]

    residuals = []
    for scale in scales:
        if scale == 1:
            diff = x0_tar - x0_src
        else:
            target_h = max(1, x0_tar.shape[-2] // scale)
            target_w = max(1, x0_tar.shape[-1] // scale)
            tar_level = F.adaptive_avg_pool2d(x0_tar, output_size=(target_h, target_w))
            src_level = F.adaptive_avg_pool2d(x0_src, output_size=(target_h, target_w))
            diff = tar_level - src_level
            diff = F.interpolate(diff, size=x0_tar.shape[-2:], mode=upsample_mode)
        residuals.append(diff)

    delta = torch.stack(residuals, dim=0).mean(dim=0)
    if M_edit is not None:
        delta = delta * M_edit

    return clean_delta_to_velocity(delta, t_scalar, eps=eps)


def editing_velocity_surrogate_target_attract(
    target_feature_map: torch.Tensor,
    reference_latent: torch.Tensor,
    M_edit: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cheap edit-side surrogate from target-conditioned feature attraction.
    """
    if M_edit is not None:
        M_edit = _align_like(target_feature_map, M_edit, mode="bilinear")

    diff = target_feature_map
    if M_edit is not None:
        diff = diff * M_edit

    diff = _align_like(reference_latent, diff, mode="bilinear")
    return diff.expand_as(reference_latent)


def editing_velocity_surrogate_source_suppress(
    source_feature_map: torch.Tensor,
    reference_latent: torch.Tensor,
    M_edit: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cheap edit-side surrogate from source-conditioned feature suppression.
    """
    if M_edit is not None:
        M_edit = _align_like(source_feature_map, M_edit, mode="bilinear")

    diff = -source_feature_map
    if M_edit is not None:
        diff = diff * M_edit

    diff = _align_like(reference_latent, diff, mode="bilinear")
    return diff.expand_as(reference_latent)


def editing_energy_total(
    x0_tar: torch.Tensor,
    x0_src: torch.Tensor,
    M_edit: torch.Tensor | None = None,
    target_feature_map: torch.Tensor | None = None,
    source_feature_map: torch.Tensor | None = None,
    lambda_anchor: float = 1.0,
    lambda_region: float = 0.0,
    lambda_target: float = 0.0,
    lambda_source: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Unified editing-side energy decomposition.

    The editing term now has two explicit components:
    - anchor: source-vs-target clean estimate separation
    - region: multiscale edit-region delta
    - target: target-attract feature term in M_edit
    - source: source-suppress feature term in M_edit
    """
    aligned = align_energy_space(reference=x0_tar, x_src=x0_src, M_edit=M_edit)
    x0_tar = aligned["reference"]
    x0_src = aligned["x_src"]
    M_edit = aligned["M_edit"]

    e_anchor = editing_energy_target_anchor(x0_tar, x0_src, mask=M_edit)
    e_region = editing_energy_region_delta(x0_tar, x0_src, M_edit=M_edit)
    e_target = torch.zeros_like(e_anchor)
    e_source = torch.zeros_like(e_anchor)
    if target_feature_map is not None:
        e_target = editing_energy_target_attract(
            target_feature_map=target_feature_map,
            M_edit=M_edit,
        )
    if source_feature_map is not None:
        e_source = editing_energy_source_suppress(
            source_feature_map=source_feature_map,
            M_edit=M_edit,
        )
    total = (
        lambda_anchor * e_anchor
        + lambda_region * e_region
        + lambda_target * e_target
        + lambda_source * e_source
    )
    return {
        "total": total,
        "anchor": e_anchor,
        "region": e_region,
        "target": e_target,
        "source": e_source,
    }


def editing_velocity_surrogate_total(
    base_edit_velocity: torch.Tensor,
    x0_tar: torch.Tensor,
    x0_src: torch.Tensor,
    t_scalar: torch.Tensor,
    M_edit: torch.Tensor | None = None,
    target_feature_map: torch.Tensor | None = None,
    source_feature_map: torch.Tensor | None = None,
    lambda_base: float = 1.0,
    lambda_anchor: float = 1.0,
    lambda_region: float = 0.0,
    lambda_target: float = 0.0,
    lambda_source: float = 0.0,
    velocity_t_min: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """
    Unified editing-side surrogate velocity decomposition.

    The current method has three concrete edit components:
    - base target-vs-source RF velocity difference
    - target-anchor clean-estimate correction
    - region-specific masked clean-estimate correction
    - target-attract feature correction
    - source-suppress feature correction
    """
    aligned = align_energy_space(reference=x0_tar, x_src=x0_src, M_edit=M_edit)
    x0_tar = aligned["reference"]
    x0_src = aligned["x_src"]
    M_edit = aligned["M_edit"]
    if M_edit is not None:
        base_edit_velocity = base_edit_velocity * _align_like(base_edit_velocity, M_edit, mode="bilinear")

    v_anchor = editing_velocity_surrogate_target_anchor(
        x0_tar=x0_tar,
        x0_src=x0_src,
        t_scalar=t_scalar,
        mask=M_edit,
        eps=velocity_t_min,
    )
    v_region = editing_velocity_surrogate_region_delta(
        x0_tar=x0_tar,
        x0_src=x0_src,
        t_scalar=t_scalar,
        M_edit=M_edit,
        eps=velocity_t_min,
    )
    v_target = torch.zeros_like(base_edit_velocity)
    v_source = torch.zeros_like(base_edit_velocity)
    if target_feature_map is not None:
        v_target = editing_velocity_surrogate_target_attract(
            target_feature_map=target_feature_map,
            reference_latent=x0_tar,
            M_edit=M_edit,
        )
    if source_feature_map is not None:
        v_source = editing_velocity_surrogate_source_suppress(
            source_feature_map=source_feature_map,
            reference_latent=x0_tar,
            M_edit=M_edit,
        )
    total = (
        lambda_base * base_edit_velocity
        + lambda_anchor * v_anchor
        + lambda_region * v_region
        + lambda_target * v_target
        + lambda_source * v_source
    )
    return {
        "total": total,
        "base": base_edit_velocity,
        "anchor": v_anchor,
        "region": v_region,
        "target": v_target,
        "source": v_source,
    }
