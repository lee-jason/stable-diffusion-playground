#!/usr/bin/env python3
"""
Fitting Room AI - IP-Adapter with ControlNet
Generate images using inspiration image with depth map control.
"""

import argparse
import sys
from random import randint
from types import MethodType

import cv2
import face_recognition
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from ip_adapter import IPAdapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from PIL import Image
from transformers import pipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def noise_scheduler():
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def VAE():
    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
        dtype=torch.float16
    )


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

    @classmethod
    def openpose(cls, image):
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        return openpose(image)


def generate_fitting_room_image(
    inspo_image_path,
    source_image_path,
    face_image_path,
    prompt="fashion model. photo realistic. high definition. Clean clear face. high quality accurate face.",
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
    num_inference_steps=100,
    output_file="output.png",
):
    """
    Generate fitting room image using IP-Adapter and ControlNet.

    Args:
        inspo_image_path (str): Path to inspiration image
        source_image_path (str): Path to source image for depth control
        face_image_path (str): Path to face image for swapping
        prompt (str): Text prompt for generation
        negative_prompt (str): Negative prompt
        num_inference_steps (int): Number of inference steps
        output_file (str): Output filename
    """
    image = swap_clothes(
        inspo_image_path=inspo_image_path,
        source_image_path=source_image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        output_file=output_file,
    )
    image.save("clothes_swap_image.png")
    image = Image.open("clothes_swap_image.png")
    image = face_swap(image, face_image_path)
    image.save(output_file)


def swap_clothes(
    inspo_image_path,
    source_image_path,
    prompt,
    negative_prompt,
    num_inference_steps,
    output_file,
):
    print(f"Loading inspiration image: {inspo_image_path}")
    print(f"Loading source image: {source_image_path}")

    # Load images
    inspo_image = load_image(inspo_image_path)
    source_image = load_image(source_image_path)

    # Generate control images
    print("Generating control maps...")
    source_image_depth_map = Preprocessors.depth_map(source_image)

    # load controlnet
    controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path, torch_dtype=torch.float16
    )

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # load SD pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler(),
        vae=VAE(),
        feature_extractor=None,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # load ip-adapter
    print("Loading IP-Adapter...")
    ip_model = IPAdapter(
        pipe,
        "InvokeAI/ip_adapter_sd_image_encoder",
        "../models/ip-adapter_sd15.bin",
        device,
    )

    print(f"Generating image with prompt: '{prompt}'")

    # Generate image
    image = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        pil_image=inspo_image,
        image=source_image,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        seed=randint(0, 100000000),
    )[0]

    # Save output
    image.save(output_file)
    print(f"Output saved to: {output_file}")
    return image


def face_swap(source_image, face_image_path):
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = "../models/ip-adapter_sd15.bin"
    device = "mps"

    def create_simple_face_mask(image_path, padding=15):
        """
        Create a simple rectangular face mask using face_recognition library

        Args:
            image_path: Path to input image
            padding: Extra padding around detected face

        Returns:
            PIL Image of the face mask
        """
        # Load image
        image = face_recognition.load_image_file(image_path)

        # Find face locations
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print("No face detected!")
            return None

        # Get image dimensions
        height, width = image.shape[:2]

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Use the first detected face
        top, right, bottom, left = face_locations[0]

        # Add padding
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)

        # Fill the face region
        mask[top:bottom, left:right] = 255
        image = Image.fromarray(mask)

        image.save("source_masked_image.png")
        return image

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    source_image = source_image
    source_masked_image = create_simple_face_mask("temp_source_image.png")
    face_image = Image.open(face_image_path)

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    # generate image
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    )

    image = ip_model.generate(
        prompt="high quality face, clear face, clean face",
        negative_prompt=negative_prompt,
        pil_image=face_image,
        num_samples=1,
        num_inference_steps=100,
        seed=randint(0, 10000000),
        image=source_image,
        mask_image=source_masked_image,
        strength=0.7,
    )[0]
    return image


def main():
    """Main function with command-line argument parsing.

    Example:
        ./main.py -s source.jpg -i inspo.jpg -o output.png
    """
    parser = argparse.ArgumentParser(
        description="Generate fitting room images using IP-Adapter and ControlNet"
    )

    parser.add_argument(
        "--inspo-image",
        "-i",
        type=str,
        required=True,
        help="Path to inspiration image",
    )

    parser.add_argument(
        "--source-image",
        "-s",
        type=str,
        required=True,
        help="Path to source image for control",
    )

    parser.add_argument(
        "--face-image",
        "-f",
        type=str,
        required=True,
        help="Path to face image for swapping",
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="photo realistic. high definition",
        help="Text prompt for generation",
    )

    parser.add_argument(
        "--negative-prompt",
        "-n",
        type=str,
        default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
        help="Negative prompt",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of inference steps (default: 100)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.png",
        help="Output filename (default: output.png)",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        print("Error: --steps must be a positive integer")
        sys.exit(1)

    try:
        generate_fitting_room_image(
            inspo_image_path=args.inspo_image,
            source_image_path=args.source_image,
            face_image_path=args.face_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            output_file=args.output,
        )
        print("Image generation completed successfully!")

    except Exception as e:
        print(f"Error during image generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
