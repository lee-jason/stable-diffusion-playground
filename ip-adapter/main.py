from random import randint

import torch
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterXL
from PIL import Image

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
ip_ckpt = "./models/ip-adapter_sdxl.bin"
device = "mps"
# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
# read image prompt
image = Image.open("input.jpg")
image.resize((512, 512))

# generate image variations with only image prompt
# Note takes 30 minutes. Didn't have patience to wait
image = ip_model.generate(
    pil_image=image, num_inference_steps=30, seed=randint(0, 10000000)
)[0]
image.save("output.png")
