# https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
# https://huggingface.co/ByteDance/SDXL-Lightning

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

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_8step_unet.safetensors"  # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("mps", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="mps"))
pipe = StableDiffusionXLPipeline.from_pretrained(
    base, unet=unet, torch_dtype=torch.float16, variant="fp16"
).to("mps")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# Ensure using the same inference steps as the loaded model and CFG set to 0.
image = pipe(
    "A fish that looks like a hamburger, swimming in the ocean with its fish friends",
    negative_prompt="low quality",
    num_inference_steps=8,
    guidance_scale=0,
).images[0]
image.save("output.png")


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("mps")

init_image = load_image("output.png").convert("RGB")
prompt = "higher definition, hyper realistic"
image = pipe(prompt, image=init_image).images[0]
image.save("output_refined.png")
