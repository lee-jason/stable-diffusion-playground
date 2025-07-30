#!/usr/bin/env python3
"""
Stable Diffusion XL with Refiner Script
https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
"""

import argparse
import os
import sys

import torch
from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_pil_image


def generate_image(
    prompt,
    n_steps=40,
    high_noise_frac=0.8,
    output_base="output_base.png",
    output_refined="output.png",
):
    """
    Generate an image using Stable Diffusion XL base and refiner models.

    Args:
        prompt (str): Text prompt for image generation
        n_steps (int): Number of inference steps
        high_noise_frac (float): Fraction of steps to run on base model (0.0-1.0)
        output_base (str): Filename for base model output
        output_refined (str): Filename for refined output
    """
    print(f"Loading models...")

    # Load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Detect device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    base.to(device)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to(device)

    print(f"Generating image with prompt: '{prompt}'")
    print(f"Steps: {n_steps}, Base model fraction: {high_noise_frac}")

    # Run base model
    print("Running base model...")
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="pil",
    ).images[0]

    # # Save base output
    image.save(output_base)
    # pil_image = to_pil_image(image[0].cpu())
    # pil_image.save(output_base)
    # print(f"Base model output saved to: {output_base}")

    # Run refiner
    print("Running refiner...")
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    # Save refined output
    image.save(output_refined)
    print(f"Refined output saved to: {output_refined}")


def main():
    """Main function with command-line argument parsing.

    Example:
        ./main.py --p 'A beautiful landscape with moutntains and a river' -s 50
    """
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion XL with Refiner"
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Cartoon style sleepy girl lying on couch with laptop opened on a running video.",
        help="Text prompt for image generation",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=40,
        help="Number of inference steps (default: 40)",
    )

    parser.add_argument(
        "--base-fraction",
        "-f",
        type=float,
        default=0.8,
        help="Fraction of steps to run on base model (0.0-1.0, default: 0.8). Smaller values means more variability in refinement output",
    )

    parser.add_argument(
        "--output-base",
        "-ob",
        type=str,
        default="output_base.png",
        help="Output filename for base model (default: output_base.png)",
    )

    parser.add_argument(
        "--output-refined",
        "-or",
        type=str,
        default="output.png",
        help="Output filename for refined model (default: output.png)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.base_fraction <= 1.0:
        print("Error: --base-fraction must be between 0.0 and 1.0")
        sys.exit(1)

    if args.steps <= 0:
        print("Error: --steps must be a positive integer")
        sys.exit(1)

    try:
        generate_image(
            prompt=args.prompt,
            n_steps=args.steps,
            high_noise_frac=args.base_fraction,
            output_base=args.output_base,
            output_refined=args.output_refined,
        )
        print("Image generation completed successfully!")

    except Exception as e:
        print(f"Error during image generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
