#!/usr/bin/env python3
"""
ControlNet Script with Canny and Depth Map
https://huggingface.co/lllyasviel/sd-controlnet-canny
https://huggingface.co/lllyasviel/sd-controlnet-depth
"""

import argparse
import sys

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from PIL import Image
from transformers import pipeline


class Preprocessors:
    @classmethod
    def canny(cls, image):
        image = np.array(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        image.save("canny.png")
        return image

    @classmethod
    def depth_map(cls, image):
        depth_estimator = pipeline("depth-estimation")
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        image.save("depth_map.png")
        return image


def generate_image(
    prompt,
    input_image_path="input.jpeg",
    negative_prompt="low quality",
    n_steps=20,
    canny_scale=0.8,
    depth_scale=0.5,
    control_guidance_end=0.8,
    output_file="output.png",
):
    """
    Generate an image using ControlNet with Canny and Depth Map controls.

    Args:
        prompt (str): Text prompt for image generation
        input_image_path (str): Path to input image for control
        negative_prompt (str): Negative prompt
        n_steps (int): Number of inference steps
        canny_scale (float): Canny control strength (0.0-1.0)
        depth_scale (float): Depth map control strength (0.0-1.0)
        control_guidance_end (float): When to stop control guidance (0.0-1.0)
        output_file (str): Output filename
    """
    print(f"Loading input image: {input_image_path}")

    # Load and resize input image
    image = load_image(input_image_path)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)

    # Generate control images
    print("Generating Canny edge map...")
    image_canny = Preprocessors.canny(image)

    print("Generating depth map...")
    image_depth_map = Preprocessors.depth_map(image)

    print("Loading ControlNet models...")

    # Load ControlNet models
    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )

    controlnet_depth_map = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Create pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V4.0_noVAE",
        # "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_canny, controlnet_depth_map],
        safety_checker=None,
        torch_dtype=torch.float16,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    print(f"Generating image with prompt: '{prompt}'")
    print(f"Steps: {n_steps}, Canny scale: {canny_scale}, Depth scale: {depth_scale}")

    # Generate image
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=[image_canny, image_depth_map],
        controlnet_conditioning_scale=[canny_scale, depth_scale],
        control_guidance_end=[control_guidance_end, control_guidance_end],
        num_inference_steps=n_steps,
    ).images[0]

    # Save output
    image.save(output_file)
    print(f"Output saved to: {output_file}")


def main():
    """Main function with command-line argument parsing.

    Example:
        ./main.py -p "Mountain landscape at night" -i input.jpg
    """
    parser = argparse.ArgumentParser(
        description="Generate images using ControlNet with Canny and Depth Map"
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Backdrop mountain landscape. Forested trees. Photorealistic. Foreground lake. Starry moonlit night. Night time scene. High quality",
        help="Text prompt for image generation",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="input.jpeg",
        help="Input image path for control (default: input.jpeg)",
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
        default=20,
        help="Number of inference steps (default: 20)",
    )

    parser.add_argument(
        "--canny-scale",
        "-cs",
        type=float,
        default=0.8,
        help="Canny control strength (0.0-1.0, default: 0.8)",
    )

    parser.add_argument(
        "--depth-scale",
        "-ds",
        type=float,
        default=0.5,
        help="Depth map control strength (0.0-1.0, default: 0.5)",
    )

    parser.add_argument(
        "--control-end",
        "-ce",
        type=float,
        default=0.8,
        help="When to stop control guidance (0.0-1.0, default: 0.8)",
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

    if not (0.0 <= args.canny_scale <= 1.0):
        print("Error: --canny-scale must be between 0.0 and 1.0")
        sys.exit(1)

    if not (0.0 <= args.depth_scale <= 1.0):
        print("Error: --depth-scale must be between 0.0 and 1.0")
        sys.exit(1)

    if not (0.0 <= args.control_end <= 1.0):
        print("Error: --control-end must be between 0.0 and 1.0")
        sys.exit(1)

    try:
        generate_image(
            prompt=args.prompt,
            input_image_path=args.input,
            negative_prompt=args.negative_prompt,
            n_steps=args.steps,
            canny_scale=args.canny_scale,
            depth_scale=args.depth_scale,
            control_guidance_end=args.control_end,
            output_file=args.output,
        )
        print("Image generation completed successfully!")

    except Exception as e:
        print(f"Error during image generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
