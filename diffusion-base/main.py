#!/usr/bin/env python3
"""
Stable Diffusion 1.5 Base Model Script
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
"""

import argparse
import sys

import torch
from diffusers import DiffusionPipeline


def generate_image(
    prompt,
    n_steps=50,
    guidance_scale=7.5,
    output="output.png",
):
    """
    Generate an image using Stable Diffusion 1.5 base model.

    Args:
        prompt (str): Text prompt for image generation
        n_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        output (str): Output filename
    """
    print(f"Loading Stable Diffusion 1.5 base model...")

    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    # Detect device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    pipe.to(device)

    print(f"Generating image with prompt: '{prompt}'")
    print(f"Steps: {n_steps}, Guidance scale: {guidance_scale}")

    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Save output
    image.save(output)
    print(f"Image saved to: {output}")


def main():
    """Main function with command-line argument parsing.

    Example:
        ./main.py --prompt "A beautiful landscape with mountains and a river" --steps 40
    """
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion XL Base Model"
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="An astronaut riding a green horse",
        help="Text prompt for image generation",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )

    parser.add_argument(
        "--guidance-scale",
        "-g",
        type=float,
        default=7.5,
        help="Guidance scale for generation (default: 7.5)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.png",
        help="Output filename (default: output.png)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.steps <= 0:
        print("Error: --steps must be a positive integer")
        sys.exit(1)

    if args.guidance_scale <= 0:
        print("Error: --guidance-scale must be positive")
        sys.exit(1)

    try:
        generate_image(
            prompt=args.prompt,
            n_steps=args.steps,
            guidance_scale=args.guidance_scale,
            output=args.output,
        )
        print("Image generation completed successfully!")

    except Exception as e:
        print(f"Error during image generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
