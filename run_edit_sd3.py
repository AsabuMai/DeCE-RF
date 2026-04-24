from __future__ import annotations

import argparse
import os

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
    parser.add_argument("--src-guidance-scale", type=float, default=3.5)
    parser.add_argument("--tar-guidance-scale", type=float, default=10.5)
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
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--stats-output", type=str, default=None)
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
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
