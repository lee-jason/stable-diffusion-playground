# https://huggingface.co/h94/IP-Adapter-FaceID
# https://huggingface.co/h94/IP-Adapter

from random import randint
from types import MethodType

import cv2
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
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


inspo_image = load_image("inspo.avif")
source_image = load_image("source.avif")
source_face_image = cv2.imread("source_face_image.jpg")

image_canny = Preprocessors.canny(source_image)
source_image_depth_map = Preprocessors.depth_map(source_image)
image_openpose = Preprocessors.openpose(source_image)

# source_face_embed = face_embed(source_face_image)


controlnet_depth_map = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

# load controlnet
controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path, torch_dtype=torch.float16
)
# load SD pipeline
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
pipe.to("mps")

# load ip-adapter
ip_model = IPAdapter(
    pipe, "InvokeAI/ip_adapter_sd_image_encoder", "../models/ip-adapter_sd15.bin", "mps"
)

# # load ip-adapter
# # load ip-adapter
# ip_model = IPAdapterFaceID(
#     "InvokeAI/ip_adapter_sd_image_encoder",
#     "../models/ip-adapter-faceid_sd15.bin",
#     "mps",
# )


image = ip_model.generate(
    prompt="photo realistic. high definition",
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
    pil_image=inspo_image,
    image=source_image_depth_map,
    num_samples=1,
    num_inference_steps=100,
    seed=randint(0, 100000000),
)[0]


image.save("output.png")
