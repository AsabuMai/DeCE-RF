from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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


def build_recolor_trimap_masks(
    edit_mask: torch.Tensor,
    core_mask: torch.Tensor | None = None,
    object_threshold: float = 0.45,
    inner_erode_kernel: int = 3,
    outer_dilate_kernel: int = 5,
    boundary_edit_scale: float = 0.8,
    boundary_preserve_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a recolor support into inner object, boundary band, and outer preserve.

    Unlike the generic object/contact split, the recolor band remains editable:
    it should receive target color pressure while avoiding source-color
    reconstruction pressure. `preserve_mask` therefore defaults to the strict
    outside region only, with optional weak boundary preservation for sweeps.
    """
    base = edit_mask if core_mask is None else torch.minimum(core_mask, edit_mask)
    object_threshold = max(0.0, min(1.0, float(object_threshold)))
    if object_threshold > 0.0:
        host = (base.detach().float() > object_threshold).to(dtype=edit_mask.dtype, device=edit_mask.device)
        host = host * edit_mask.clamp(0.0, 1.0)
    else:
        host = base.clamp(0.0, 1.0)
    if float(host.detach().float().max().item()) <= 1e-6:
        host = base.clamp(0.0, 1.0)

    inner = host
    inner_erode_kernel = int(inner_erode_kernel)
    if inner_erode_kernel > 1:
        if inner_erode_kernel % 2 == 0:
            inner_erode_kernel += 1
        inner = (
            1.0
            - dilate_spatial_mask((1.0 - host).clamp(0.0, 1.0), kernel_size=inner_erode_kernel)
        ).clamp(0.0, 1.0)

    support = host
    outer_dilate_kernel = int(outer_dilate_kernel)
    if outer_dilate_kernel > 1:
        if outer_dilate_kernel % 2 == 0:
            outer_dilate_kernel += 1
        support = dilate_spatial_mask(host, kernel_size=outer_dilate_kernel).clamp(0.0, 1.0)

    boundary = (support - inner).clamp(0.0, 1.0)
    edit = (inner + float(boundary_edit_scale) * boundary).clamp(0.0, 1.0)
    preserve = (
        (1.0 - support).clamp(0.0, 1.0)
        + float(boundary_preserve_scale) * boundary
    ).clamp(0.0, 1.0)
    return edit, inner.clamp(0.0, 1.0), boundary, preserve


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
