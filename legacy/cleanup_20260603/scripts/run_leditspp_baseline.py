#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


TASKS: dict[str, dict[str, Any]] = {
    "cat_crown": {
        "image": "data/paper_images/cat_sitting_in_grass.jpg",
        "source_prompt": "A photo of a cat sitting in grass.",
        "target_prompt": "A photo of the same cat sitting in the same grass, wearing a small golden crown on its head.",
        "editing_prompt": ["small golden crown"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale": [8.0],
        "edit_threshold": [0.75],
    },
    "dog_sunglasses": {
        "image": "data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg",
        "source_prompt": "A front-facing portrait of a dog in snow.",
        "target_prompt": "A front-facing portrait of the same dog wearing black sunglasses in snow.",
        "editing_prompt": ["black sunglasses"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale": [8.0],
        "edit_threshold": [0.75],
    },
    "mug_heart": {
        "image": "data/pretty_free_candidates/pexels_white_mug_6312107.jpg",
        "source_prompt": "A minimalist photo of a plain white ceramic mug on a grey background.",
        "target_prompt": "A minimalist photo of the same white ceramic mug with a small red heart printed on the front, on the same grey background.",
        "editing_prompt": ["small red heart printed on the mug"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale": [8.0],
        "edit_threshold": [0.75],
    },
    "tshirt_star": {
        "image": "data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg",
        "source_prompt": "A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting.",
        "target_prompt": "The same person wearing the same white t-shirt and blue jeans, with a clearly visible medium-sized bright red star printed on the center chest, while preserving the fabric folds, shadows, jeans, pose, and background.",
        "editing_prompt": ["bright red star printed on the t-shirt"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale": [8.0],
        "edit_threshold": [0.75],
    },
    "backpack_remove_toy_charm": {
        "image": "data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg",
        "source_prompt": "A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap.",
        "target_prompt": "A close-up photo of the same grey backpack with the yellow dangling toy charm removed, pink strap, zipper, and fabric preserved.",
        "editing_prompt": ["yellow dangling toy charm"],
        "reverse_editing_direction": [True],
        "edit_guidance_scale": [8.0],
        "edit_threshold": [0.75],
    },
    "red_chair_blue": {
        "image": "data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg",
        "source_prompt": "A photo of a red armless rounded upholstered chair in a stylish room.",
        "target_prompt": "A photo of the same armless rounded upholstered chair in the same stylish room, with only the fabric color changed to deep blue, no armrests added.",
        "editing_prompt": ["deep blue upholstered chair"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale": [8.0],
        "edit_threshold": [0.75],
    },
}


def split_items(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        value = " ".join(value)
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def load_image(path: Path, size: int | None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size and max(image.size) != size:
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        left = (size - image.width) // 2
        top = (size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LEDITS++ external baseline on Core-6 tasks.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/external_baselines"))
    parser.add_argument("--tasks", nargs="+", default=["mug_heart"])
    parser.add_argument("--seeds", nargs="+", default=["10"])
    parser.add_argument("--model", default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inversion-steps", type=int, default=30)
    parser.add_argument("--skip", type=float, default=0.15)
    parser.add_argument("--source-guidance-scale", type=float, default=3.5)
    parser.add_argument("--edit-warmup-steps", type=int, default=0)
    parser.add_argument("--use-cross-attn-mask", action="store_true")
    parser.add_argument("--use-intersect-mask", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else root / args.outputs_dir
    tasks = split_items(args.tasks)
    seeds = [int(seed) for seed in split_items(args.seeds)]
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown tasks: {', '.join(unknown)}")

    if args.dry_run:
        for task in tasks:
            for seed in seeds:
                print(outputs_dir / task / "ledits_pp" / f"seed_{seed}")
        return 0

    from diffusers import LEditsPPPipelineStableDiffusion

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=not args.allow_download,
    )
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    pipe = pipe.to(args.device)

    for task in tasks:
        cfg = TASKS[task]
        image_path = (root / cfg["image"]).resolve()
        source_prompt = cfg["source_prompt"]
        target_prompt = cfg["target_prompt"]
        editing_prompt = cfg["editing_prompt"]
        reverse_editing_direction = cfg["reverse_editing_direction"]
        edit_guidance_scale = cfg["edit_guidance_scale"]
        edit_threshold = cfg["edit_threshold"]
        image = load_image(image_path, args.image_size)

        for seed in seeds:
            run_dir = outputs_dir / task / "ledits_pp" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = time.time()

            _ = pipe.invert(
                image=image,
                source_prompt=source_prompt,
                source_guidance_scale=args.source_guidance_scale,
                num_inversion_steps=args.num_inversion_steps,
                skip=args.skip,
                generator=generator,
                height=args.image_size,
                width=args.image_size,
            )
            edited = pipe(
                editing_prompt=editing_prompt,
                reverse_editing_direction=reverse_editing_direction,
                edit_guidance_scale=edit_guidance_scale,
                edit_threshold=edit_threshold,
                edit_warmup_steps=args.edit_warmup_steps,
                use_cross_attn_mask=args.use_cross_attn_mask,
                use_intersect_mask=args.use_intersect_mask,
                generator=generator,
                output_type="pil",
            ).images[0]
            edited.save(run_dir / "result.png")

            runtime = time.time() - start
            peak_gb = None
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            metadata = {
                "baseline": "ledits_pp",
                "baseline_family": "diffusion_text_editing",
                "baseline_repo": "/workspace/baselines/src/ledits_pp",
                "baseline_runner": str(Path(__file__).resolve()),
                "model": args.model,
                "image": str(image_path),
                "source_prompt": source_prompt,
                "effective_source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "effective_target_prompt": target_prompt,
                "editing_prompt": editing_prompt,
                "reverse_editing_direction": reverse_editing_direction,
                "output": str(run_dir / "result.png"),
                "stats_output": str(run_dir / "stats.json"),
                "seed": seed,
                "num_inversion_steps": args.num_inversion_steps,
                "skip": args.skip,
                "source_guidance_scale": args.source_guidance_scale,
                "edit_guidance_scale": edit_guidance_scale,
                "edit_threshold": edit_threshold,
                "edit_warmup_steps": args.edit_warmup_steps,
                "use_cross_attn_mask": args.use_cross_attn_mask,
                "use_intersect_mask": args.use_intersect_mask,
                "image_size": args.image_size,
                "runtime_seconds": runtime,
                "peak_gpu_memory_gb": peak_gb,
                "mask_usage": "no fixed eval mask passed to baseline; fixed masks are for evaluation only",
            }
            stats = {
                "steps": [
                    {
                        "step": "ledits_pp_complete",
                        "mask_area_ratio": 0.0,
                        "rec_energy": 0.0,
                        "rec_guidance_norm": 0.0,
                        "edit_guidance_norm": 0.0,
                    }
                ]
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (run_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
            command = " ".join(shlex.quote(part) for part in ["python", str(Path(__file__).resolve()), *(__import__('sys').argv[1:])])
            (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
            print(f"[ledits_pp] complete task={task} seed={seed} output={run_dir / 'result.png'} runtime={runtime:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
