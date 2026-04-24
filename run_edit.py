from __future__ import annotations

import argparse
import os

import PIL.Image
import torch
from diffusers import FluxPipeline
from diffusers.training_utils import set_seed

from hrec_rf_pipeline import HRecRFInversionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone RF h-Edit prototype.")
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--prompt", required=True, help="Target prompt.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--num-inversion-steps", type=int, default=28)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--rec-guidance-scale", type=float, default=0.0)
    parser.add_argument("--start-timestep", type=float, default=0.0)
    parser.add_argument("--stop-timestep", type=float, default=7 / 28)
    parser.add_argument("--enable-sde", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    set_seed(args.seed)
    image = PIL.Image.open(args.image).convert("RGB")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    try:
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.to("cuda")

    pipeline = HRecRFInversionPipeline.from_pipe(pipe)

    invert_outputs = pipeline.invert(
        image=image,
        num_inversion_steps=args.num_inversion_steps,
        gamma=args.gamma,
        return_source_trajectory=args.rec_guidance_scale > 0.0,
    )

    if args.rec_guidance_scale > 0.0:
        inverted_latents, image_latents, latent_image_ids, source_trajectory = invert_outputs
    else:
        inverted_latents, image_latents, latent_image_ids = invert_outputs
        source_trajectory = None

    edited_image = pipeline(
        prompt=args.prompt,
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        source_trajectory=source_trajectory,
        rec_guidance_scale=args.rec_guidance_scale,
        start_timestep=args.start_timestep,
        stop_timestep=args.stop_timestep,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        enable_sde=args.enable_sde,
    ).images[0]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    edited_image.save(args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
