# https://huggingface.co/black-forest-labs/FLUX.1-dev
# https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0
import os

import torch
from diffusers import DiffusionPipeline, FluxControlNetModel, FluxControlNetPipeline
from diffusers.utils import load_image
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.environ.get("HF_TOKEN"))


# base_model = "black-forest-labs/FLUX.1-dev"
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model_union, torch_dtype=torch.float16
)
# pipe = FluxControlNetPipeline.from_pretrained(
#     base_model, controlnet=controlnet, torch_dtype=torch.float16
# )

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("mps")
# optimizations
# pipe.enable_model_cpu_offload()
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()

# replace with other conds
control_image = load_image("../refiner/output.png")
width, height = control_image.size
# width, height = 512, 512
num_inference_steps = 30

prompt = "A man works on his laptop on his lap in front of a tiny black table from ikea, only large enough to hold a monitor. There is another monitor on the table playing a video. He is sitting in a herman miller chair"

image = pipe(
    prompt,
    control_image=control_image,
    width=width,
    height=height,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.8,
    num_inference_steps=num_inference_steps,
    guidance_scale=3.5,
    generator=torch.Generator(device="mps").manual_seed(42),
).images[0]

image.save("output.png")
