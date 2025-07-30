# https://huggingface.co/h94/IP-Adapter-FaceID

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

# from insightface.app import FaceAnalysis
from ip_adapter import IPAdapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from PIL import Image
from transformers import pipeline

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "../models/ip-adapter_sd15.bin"
device = "mps"


def create_simple_face_mask(image_path, padding=30):
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


# def face_embed(image):
#     app = FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider"])
#     app.prepare(ctx_id=0, det_size=(640, 640))

#     # image = cv2.imread("person.jpg")
#     faces = app.get(image)

#     faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
#     return faceid_embeds


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

source_image = Image.open("output.png")
source_masked_image = create_simple_face_mask("output.png")
face_image = Image.open("source_face_image.jpg")
# faceid_embeds = face_embed(face_image)

# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

# generate image
# prompt = "photo of a woman in red dress in a garden"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

image = ip_model.generate(
    negative_prompt=negative_prompt,
    pil_image=face_image,
    num_samples=1,
    num_inference_steps=100,
    seed=randint(0, 10000000),
    image=source_image,
    mask_image=source_masked_image,
    strength=0.7,
)[0]

image.save("face_output.png")
