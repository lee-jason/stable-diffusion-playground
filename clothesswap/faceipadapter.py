# https://huggingface.co/h94/IP-Adapter-FaceID

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
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from PIL import Image
from transformers import pipeline

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "ip-adapter-faceid-portrait_sd15.bin"
device = "mps"


def face_embed(image):
    app = FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # image = cv2.imread("person.jpg")
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    return faceid_embeds


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
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
)


# load ip-adapter
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

# generate image
prompt = "photo of a woman in red dress in a garden"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    faceid_embeds=faceid_embeds,
    num_samples=4,
    num_inference_steps=30,
    seed=2023,
)
