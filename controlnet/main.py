# https://huggingface.co/lllyasviel/sd-controlnet-canny
# https://huggingface.co/lllyasviel/sd-controlnet-depth

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


image = load_image("input.jpeg")
image = image.resize((512, 512), Image.Resampling.LANCZOS)

image_canny = Preprocessors.canny(image)
image_depth_map = Preprocessors.depth_map(image)

controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

controlnet_depth_map = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V4.0_noVAE",
    # "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_depth_map],
    safety_checker=None,
    torch_dtype=torch.float16,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("mps")

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()

image = pipe(
    "Backdrop mountain landscape. Forested trees. Photorealistic. Foreground lake. Starry moonlit night. Night time scene. High quality",
    negative_promp="low quality",
    image=[image_canny, image_depth_map],
    controlnet_conditioning_scale=[0.8, 0.5],  # Control strength for each
    control_guidance_end=[0.8, 0.8],  # When to stop each control
    num_inference_steps=20,
).images[0]

image.save("output.png")
