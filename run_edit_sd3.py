from __future__ import annotations

import argparse
import json
import os
import subprocess

import PIL.Image
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import set_seed

from sd3_hrec import HRecSD3Edit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone SD3 h-Edit prototype.")
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--source-prompt", required=True, help="Source prompt.")
    parser.add_argument("--prompt", required=True, help="Target prompt.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=512,
        help="Resize the source image so its longest side is at most this size before VAE encoding.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument(
        "--src-guidance-scale",
        type=float,
        default=1.0,
        help="Legacy source guidance default used by inversion/base unless the split source CFG args are set.",
    )
    parser.add_argument("--tar-guidance-scale", type=float, default=10.5)
    parser.add_argument(
        "--inversion-guidance-scale",
        type=float,
        default=None,
        help="Optional source CFG used only for source inversion. Defaults to --src-guidance-scale.",
    )
    parser.add_argument(
        "--base-guidance-scale",
        type=float,
        default=None,
        help="Optional source CFG used only for the reverse ODE base prior. Defaults to --src-guidance-scale.",
    )
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--rec-guidance-scale", type=float, default=0.0)
    parser.add_argument("--struct-guidance-scale", type=float, default=0.5)
    parser.add_argument("--edit-hedit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-src-cfg-scale", type=float, default=None)
    parser.add_argument("--edit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-region-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-source-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-clip-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-clip-match-base-scale", type=float, default=0.0)
    parser.add_argument("--edit-image-tv-scale", type=float, default=0.0)
    parser.add_argument("--edit-text-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-text-source-scale", type=float, default=0.8)
    parser.add_argument("--edit-text-core-weight", type=float, default=1.0)
    parser.add_argument("--edit-text-subject-weight", type=float, default=0.3)
    parser.add_argument("--identity-break-stop", type=float, default=0.55)
    parser.add_argument("--target-attract-start", type=float, default=0.25)
    parser.add_argument("--edit-dds-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-dds-source-scale", type=float, default=0.8)
    parser.add_argument("--edit-app-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-core-scale", type=float, default=1.35)
    parser.add_argument("--edit-subject-scale", type=float, default=0.35)
    parser.add_argument("--edit-bound-scale", type=float, default=0.0)
    parser.add_argument("--clip-start-timestep", type=float, default=0.0)
    parser.add_argument("--clip-stop-timestep", type=float, default=0.6)
    parser.add_argument("--preserve-blend-scale", type=float, default=0.0)
    parser.add_argument("--preserve-blend-start-timestep", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--alpha-schedule", type=str, default="constant")
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--beta-schedule", type=str, default="constant")
    parser.add_argument(
        "--velocity-conversion-mode",
        type=str,
        choices=("legacy", "linear_path"),
        default="linear_path",
        help="How clean-space displacement surrogates are converted into velocity corrections.",
    )
    parser.add_argument(
        "--linear-path-t-min",
        type=float,
        default=0.05,
        help="Minimum denominator for linear-path clean-delta-to-velocity conversion in image runs.",
    )
    parser.add_argument(
        "--rec-stop-timestep",
        type=float,
        default=0.08,
        help="Disable reconstruction guidance below this normalized timestep to avoid low-t blow-up.",
    )
    parser.add_argument(
        "--trajectory-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "Blend preserve regions toward the saved source inversion trajectory "
            "after each reverse ODE step. 0 disables trajectory anchoring."
        ),
    )
    parser.add_argument(
        "--trajectory-subject-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "Additionally blend the attention subject ring (subject minus core) "
            "toward the source trajectory. This helps local edits keep identity "
            "while leaving the high-confidence core editable."
        ),
    )
    parser.add_argument(
        "--edit-initial-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Blend the initial inverted latent toward random noise inside the "
            "attention edit region. Useful for local insertions that need more "
            "generation freedom than source inversion provides."
        ),
    )
    parser.add_argument(
        "--edit-initial-noise-region",
        type=str,
        choices=("core", "subject"),
        default="core",
        help="Which attention mask region receives --edit-initial-noise-scale.",
    )
    parser.add_argument(
        "--attention-mask-mode",
        type=str,
        choices=("changed_union", "target_changed", "subject_union", "source_subject", "target_subject"),
        default="changed_union",
        help="Which attention map is converted into the editable mask.",
    )
    parser.add_argument(
        "--attention-mask-target-words",
        type=str,
        default=None,
        help=(
            "Optional comma-separated target prompt words used for token attention. "
            "Defaults to the changed words inferred from source/target prompts."
        ),
    )
    parser.add_argument(
        "--attention-mask-source-words",
        type=str,
        default=None,
        help=(
            "Optional comma-separated source prompt words used for token attention. "
            "Defaults to the changed words inferred from source/target prompts."
        ),
    )
    parser.add_argument(
        "--attention-mask-subject-threshold",
        type=float,
        default=0.48,
        help="Threshold on the normalized attention map for the editable subject mask.",
    )
    parser.add_argument(
        "--attention-mask-core-threshold",
        type=float,
        default=0.72,
        help="Threshold on the normalized attention map for the high-confidence edit core.",
    )
    parser.add_argument(
        "--mask-output-dir",
        type=str,
        default=None,
        help="Optional directory for saving attention-mask debug PNGs.",
    )
    parser.add_argument(
        "--edit-mask-dilate-kernel",
        type=int,
        default=0,
        help="Optionally dilate the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-smooth-kernel",
        type=int,
        default=0,
        help="Optionally smooth the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-component-threshold",
        type=float,
        default=0.0,
        help="Threshold used by optional attention-mask connected-component filtering. 0 uses 0.5.",
    )
    parser.add_argument(
        "--edit-mask-keep-components",
        type=int,
        default=0,
        help="Keep only the top-N attention-mask connected components by mask mass. 0 keeps all.",
    )
    parser.add_argument(
        "--edit-mask-component-y-min",
        type=float,
        default=None,
        help="Optional minimum normalized component center y for attention-mask component filtering.",
    )
    parser.add_argument(
        "--edit-mask-component-y-max",
        type=float,
        default=None,
        help="Optional maximum normalized component center y for attention-mask component filtering.",
    )
    parser.add_argument(
        "--edit-mask-shift-y",
        type=float,
        default=0.0,
        help="Translate the attention-derived edit/core masks vertically as a fraction of latent height.",
    )
    parser.add_argument(
        "--edit-mask-shift-x",
        type=float,
        default=0.0,
        help="Translate the attention-derived edit/core masks horizontally as a fraction of latent width.",
    )
    parser.add_argument(
        "--edit-mask-box",
        type=str,
        default=None,
        help="Optional normalized edit box as x0,y0,x1,y1. Useful for local accessory placement.",
    )
    parser.add_argument(
        "--edit-mask-box-mode",
        type=str,
        choices=("replace", "intersect", "union"),
        default="replace",
        help="How --edit-mask-box combines with the attention-derived edit mask.",
    )
    parser.add_argument(
        "--external-edit-mask",
        type=str,
        default=None,
        help=(
            "Optional grayscale image used as an external M_edit support. "
            "This changes only the mask source; the RF/ODE edit dynamics are unchanged."
        ),
    )
    parser.add_argument(
        "--external-edit-mask-mode",
        type=str,
        choices=("replace", "intersect", "union"),
        default="replace",
        help="How --external-edit-mask combines with the attention-derived edit mask.",
    )
    parser.add_argument(
        "--edit-field-mode",
        type=str,
        choices=("surrogate", "autograd"),
        default="surrogate",
        help="Current implementation supports the surrogate edit field mode.",
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--stats-output", type=str, default=None)
    parser.add_argument("--metadata-output", type=str, default=None)
    parser.add_argument("--mask-blend", action="store_true", default=False)
    parser.add_argument(
        "--photo-prompt-mode",
        type=str,
        choices=("off", "source", "both"),
        default="source",
        help=(
            "Optionally rewrite prompts into an explicit photorealistic style "
            "without changing the editing math. 'source' only stabilizes the "
            "base prior; 'both' applies the same style prior to source and target."
        ),
    )
    return parser


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def make_photorealistic_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    style_cues = (
        "photo",
        "photograph",
        "photorealistic",
        "realistic",
        "wildlife",
        "natural lighting",
        "detailed fur",
        "high-resolution",
    )
    if any(cue in lowered for cue in style_cues):
        return prompt
    return (
        f"A realistic wildlife photograph of {prompt.rstrip('.').lower()}, "
        "natural lighting, detailed fur, high-resolution photo."
    )


def parse_normalized_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--edit-mask-box must be formatted as x0,y0,x1,y1")
    box = tuple(float(part) for part in parts)
    if any(coord < 0.0 or coord > 1.0 for coord in box):
        raise ValueError("--edit-mask-box coordinates must be in [0, 1]")
    return box


def parse_word_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    words = [word.strip().lower() for word in value.split(",") if word.strip()]
    return words if words else None


def load_image_latent(
    pipe,
    image_path: str,
    device: torch.device,
    max_image_size: int,
) -> torch.Tensor:
    image = PIL.Image.open(image_path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        resized_w = max(16, int(round(image.width * scale)))
        resized_h = max(16, int(round(image.height * scale)))
        image = image.resize((resized_w, resized_h), PIL.Image.Resampling.LANCZOS)
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_src = pipe.image_processor.preprocess(image).to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    return (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def main() -> None:
    args = build_parser().parse_args()
    if args.edit_field_mode != "surrogate":
        raise ValueError("Only --edit-field-mode surrogate is currently implemented.")
    edit_mask_box = parse_normalized_box(args.edit_mask_box)
    attention_mask_target_words = parse_word_list(args.attention_mask_target_words)
    attention_mask_source_words = parse_word_list(args.attention_mask_source_words)
    set_seed(args.seed)

    source_prompt = args.source_prompt
    target_prompt = args.prompt
    if args.photo_prompt_mode in {"source", "both"}:
        source_prompt = make_photorealistic_prompt(source_prompt)
    if args.photo_prompt_mode == "both":
        target_prompt = make_photorealistic_prompt(target_prompt)
    if source_prompt != args.source_prompt or target_prompt != args.prompt:
        print(f"[prompt] source: {source_prompt}")
        print(f"[prompt] target: {target_prompt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    scheduler = pipe.scheduler
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()

    x_src = load_image_latent(
        pipe=pipe,
        image_path=args.image,
        device=device,
        max_image_size=args.max_image_size,
    ).to(device)
    x_tar = HRecSD3Edit(
        pipe=pipe,
        scheduler=scheduler,
        x_src=x_src,
        src_prompt=source_prompt,
        tar_prompt=target_prompt,
        negative_prompt="",
        T_steps=args.num_inference_steps,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        inversion_guidance_scale=args.inversion_guidance_scale,
        base_guidance_scale=args.base_guidance_scale,
        n_max=args.n_max,
        eta=args.eta,
        rec_guidance_scale=args.rec_guidance_scale,
        struct_guidance_scale=args.struct_guidance_scale,
        edit_hedit_guidance_scale=args.edit_hedit_guidance_scale,
        edit_src_cfg_scale=args.edit_src_cfg_scale,
        edit_guidance_scale=args.edit_guidance_scale,
        edit_region_guidance_scale=args.edit_region_guidance_scale,
        edit_target_guidance_scale=args.edit_target_guidance_scale,
        edit_source_guidance_scale=args.edit_source_guidance_scale,
        edit_clip_guidance_scale=args.edit_clip_guidance_scale,
        edit_clip_match_base_scale=args.edit_clip_match_base_scale,
        edit_image_tv_scale=args.edit_image_tv_scale,
        edit_text_guidance_scale=args.edit_text_guidance_scale,
        edit_text_source_scale=args.edit_text_source_scale,
        edit_text_core_weight=args.edit_text_core_weight,
        edit_text_subject_weight=args.edit_text_subject_weight,
        identity_break_stop=args.identity_break_stop,
        target_attract_start=args.target_attract_start,
        edit_dds_guidance_scale=args.edit_dds_guidance_scale,
        edit_dds_source_scale=args.edit_dds_source_scale,
        edit_app_guidance_scale=args.edit_app_guidance_scale,
        edit_core_scale=args.edit_core_scale,
        edit_subject_scale=args.edit_subject_scale,
        edit_bound_scale=args.edit_bound_scale,
        clip_start_timestep=args.clip_start_timestep,
        clip_stop_timestep=args.clip_stop_timestep,
        preserve_blend_scale=args.preserve_blend_scale,
        preserve_blend_start_timestep=args.preserve_blend_start_timestep,
        alpha_max=args.alpha_max,
        alpha_schedule=args.alpha_schedule,
        beta_max=args.beta_max,
        beta_schedule=args.beta_schedule,
        velocity_conversion_mode=args.velocity_conversion_mode,
        linear_path_t_min=args.linear_path_t_min,
        rec_stop_timestep=args.rec_stop_timestep,
        trajectory_preserve_scale=args.trajectory_preserve_scale,
        trajectory_subject_preserve_scale=args.trajectory_subject_preserve_scale,
        edit_initial_noise_scale=args.edit_initial_noise_scale,
        edit_initial_noise_region=args.edit_initial_noise_region,
        attention_mask_mode=args.attention_mask_mode,
        attention_mask_target_words=attention_mask_target_words,
        attention_mask_source_words=attention_mask_source_words,
        attention_mask_subject_threshold=args.attention_mask_subject_threshold,
        attention_mask_core_threshold=args.attention_mask_core_threshold,
        mask_output_dir=args.mask_output_dir,
        edit_mask_dilate_kernel=args.edit_mask_dilate_kernel,
        edit_mask_smooth_kernel=args.edit_mask_smooth_kernel,
        edit_mask_component_threshold=args.edit_mask_component_threshold,
        edit_mask_keep_components=args.edit_mask_keep_components,
        edit_mask_component_y_min=args.edit_mask_component_y_min,
        edit_mask_component_y_max=args.edit_mask_component_y_max,
        edit_mask_shift_y=args.edit_mask_shift_y,
        edit_mask_shift_x=args.edit_mask_shift_x,
        edit_mask_box=edit_mask_box,
        edit_mask_box_mode=args.edit_mask_box_mode,
        external_edit_mask_path=args.external_edit_mask,
        external_edit_mask_mode=args.external_edit_mask_mode,
        log_every=args.log_every,
        stats_output_path=args.stats_output,
        mask_blend=args.mask_blend,
    )

    x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image_tar[0].save(args.output)
    metadata_output = args.metadata_output
    if metadata_output is None:
        root, _ = os.path.splitext(args.output)
        metadata_output = f"{root}_metadata.json"
    metadata = {
        "image": args.image,
        "source_prompt": args.source_prompt,
        "effective_source_prompt": source_prompt,
        "target_prompt": args.prompt,
        "effective_target_prompt": target_prompt,
        "output": args.output,
        "stats_output": args.stats_output,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "n_max": args.n_max,
        "src_guidance_scale": args.src_guidance_scale,
        "tar_guidance_scale": args.tar_guidance_scale,
        "inversion_guidance_scale": (
            args.inversion_guidance_scale
            if args.inversion_guidance_scale is not None
            else args.src_guidance_scale
        ),
        "base_guidance_scale": (
            args.base_guidance_scale
            if args.base_guidance_scale is not None
            else args.src_guidance_scale
        ),
        "edit_src_cfg_scale": (
            args.edit_src_cfg_scale
            if args.edit_src_cfg_scale is not None
            else (
                args.base_guidance_scale
                if args.base_guidance_scale is not None
                else args.src_guidance_scale
            )
        ),
        "rec_guidance_scale": args.rec_guidance_scale,
        "struct_guidance_scale": args.struct_guidance_scale,
        "edit_hedit_guidance_scale": args.edit_hedit_guidance_scale,
        "edit_guidance_scale": args.edit_guidance_scale,
        "edit_region_guidance_scale": args.edit_region_guidance_scale,
        "edit_target_guidance_scale": args.edit_target_guidance_scale,
        "edit_source_guidance_scale": args.edit_source_guidance_scale,
        "edit_clip_guidance_scale": args.edit_clip_guidance_scale,
        "edit_text_guidance_scale": args.edit_text_guidance_scale,
        "edit_dds_guidance_scale": args.edit_dds_guidance_scale,
        "edit_app_guidance_scale": args.edit_app_guidance_scale,
        "alpha_max": args.alpha_max if args.alpha_max is not None else args.rec_guidance_scale,
        "alpha_schedule": args.alpha_schedule,
        "beta_max": args.beta_max if args.beta_max is not None else 1.0,
        "beta_schedule": args.beta_schedule,
        "velocity_conversion_mode": args.velocity_conversion_mode,
        "linear_path_t_min": args.linear_path_t_min,
        "rec_stop_timestep": args.rec_stop_timestep,
        "trajectory_preserve_scale": args.trajectory_preserve_scale,
        "trajectory_subject_preserve_scale": args.trajectory_subject_preserve_scale,
        "edit_initial_noise_scale": args.edit_initial_noise_scale,
        "edit_initial_noise_region": args.edit_initial_noise_region,
        "attention_mask_mode": args.attention_mask_mode,
        "attention_mask_target_words": attention_mask_target_words,
        "attention_mask_source_words": attention_mask_source_words,
        "attention_mask_subject_threshold": args.attention_mask_subject_threshold,
        "attention_mask_core_threshold": args.attention_mask_core_threshold,
        "mask_output_dir": args.mask_output_dir,
        "edit_mask_dilate_kernel": args.edit_mask_dilate_kernel,
        "edit_mask_smooth_kernel": args.edit_mask_smooth_kernel,
        "edit_mask_component_threshold": args.edit_mask_component_threshold,
        "edit_mask_keep_components": args.edit_mask_keep_components,
        "edit_mask_component_y_min": args.edit_mask_component_y_min,
        "edit_mask_component_y_max": args.edit_mask_component_y_max,
        "edit_mask_shift_y": args.edit_mask_shift_y,
        "edit_mask_shift_x": args.edit_mask_shift_x,
        "edit_mask_box": edit_mask_box,
        "edit_mask_box_mode": args.edit_mask_box_mode,
        "external_edit_mask": args.external_edit_mask,
        "external_edit_mask_mode": args.external_edit_mask_mode,
        "edit_field_mode": args.edit_field_mode,
        "photo_prompt_mode": args.photo_prompt_mode,
        "git_commit": current_git_commit(),
    }
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved result to {args.output}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
