#!/usr/bin/env python3
"""
SDXL Lightning + Refiner Script
Fast generation with SDXL Lightning (8-step) + quality refinement
https://huggingface.co/ByteDance/SDXL-Lightning
https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
"""

import argparse
import sys

import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def generate_image(
    prompt,
    negative_prompt="low quality",
    n_steps=8,
    guidance_scale=0,
    refine=True,
    refine_prompt="higher definition, hyper realistic",
    output_base="output.png",
    output_refined="output_refined.png",
):
    """
    Generate an image using SDXL Lightning (fast) + optional refiner.

    Args:
        prompt (str): Text prompt for image generation
        negative_prompt (str): Negative prompt
        n_steps (int): Number of inference steps for Lightning model
        guidance_scale (float): Guidance scale (Lightning works best with 0)
        refine (bool): Whether to use refiner
        refine_prompt (str): Prompt for refinement
        output_base (str): Filename for base model output
        output_refined (str): Filename for refined output
    """
    print(f"Loading SDXL Lightning model...")

    # Model configurations
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"

    # Detect device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("Warning: Lightning model works best on GPU (MPS/CUDA)")

    print(f"Using device: {device}")

    # Load Lightning UNet
    print("Loading Lightning UNet...")
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
        device, torch.float16
    )
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

    # Create Lightning pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    # Configure scheduler for Lightning
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    print(f"Generating image with Lightning model...")
    print(f"Prompt: '{prompt}'")
    print(f"Steps: {n_steps}, Guidance scale: {guidance_scale}")

    # Generate with Lightning (fast)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Save base output
    image.save(output_base)
    print(f"Lightning output saved to: {output_base}")

    if refine:
        print(f"Loading refiner model...")

        # Load refiner pipeline
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        print(f"Refining image...")
        print(f"Refine prompt: '{refine_prompt}'")

        # Refine the image
        refined_image = refiner_pipe(refine_prompt, image=image).images[0]

        # Save refined output
        refined_image.save(output_refined)
        print(f"Refined output saved to: {output_refined}")
    else:
        print("Skipping refinement step")


def main():
    """Main function with command-line argument parsing.

    Example:
        ./main.py --prompt "A majestic dragon" --refine
    """
    parser = argparse.ArgumentParser(
        description="Generate images using SDXL Lightning + optional Refiner"
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="A fish that looks like a hamburger, swimming in the ocean with its fish friends",
        help="Text prompt for image generation",
    )

    parser.add_argument(
        "--negative-prompt",
        "-n",
        type=str,
        default="low quality",
        help="Negative prompt (default: 'low quality')",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=8,
        help="Number of inference steps for Lightning (default: 8)",
    )

    parser.add_argument(
        "--guidance-scale",
        "-g",
        type=float,
        default=0,
        help="Guidance scale (Lightning works best with 0, default: 0)",
    )

    parser.add_argument(
        "--refine",
        "-r",
        action="store_true",
        default=True,
        help="Use refiner for higher quality (slower)",
    )

    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip refinement step (faster)",
    )

    parser.add_argument(
        "--refine-prompt",
        "-rp",
        type=str,
        default="higher definition, hyper realistic",
        help="Prompt for refinement step",
    )

    parser.add_argument(
        "--output-base",
        "-ob",
        type=str,
        default="output.png",
        help="Output filename for Lightning model (default: output.png)",
    )

    parser.add_argument(
        "--output-refined",
        "-or",
        type=str,
        default="output_refined.png",
        help="Output filename for refined model (default: output_refined.png)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.steps <= 0:
        print("Error: --steps must be a positive integer")
        sys.exit(1)

    # Handle refine logic
    use_refine = args.refine if not args.no_refine else False
    if args.no_refine and args.refine:
        print("Warning: Both --refine and --no-refine specified, skipping refinement")
        use_refine = False

    try:
        generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            n_steps=args.steps,
            guidance_scale=args.guidance_scale,
            refine=use_refine,
            refine_prompt=args.refine_prompt,
            output_base=args.output_base,
            output_refined=args.output_refined,
        )
        print("Image generation completed successfully!")

    except Exception as e:
        print(f"Error during image generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
