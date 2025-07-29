# https://huggingface.co/black-forest-labs/FLUX.1-dev
# https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0

import torch

# use local files for this moment
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.utils import load_image

from utils import setup

# https://github.com/huggingface/diffusers/pull/11350
# You can directly import from diffusers by install the laster version from source
# from diffusers import FluxControlNetPipeline, FluxControlNetModel


base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model_union, torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=[controlnet], torch_dtype=torch.bfloat16
)  # use [] to enable multi-CNs
pipe.to("cuda")

# replace with other conds
control_image = load_image("./conds/canny.png")
width, height = control_image.size

prompt = "A young girl stands gracefully at the edge of a serene beach, her long, flowing hair gently tousled by the sea breeze. She wears a soft, pastel-colored dress that complements the tranquil blues and greens of the coastal scenery. The golden hues of the setting sun cast a warm glow on her face, highlighting her serene expression. The background features a vast, azure ocean with gentle waves lapping at the shore, surrounded by distant cliffs and a clear, cloudless sky. The composition emphasizes the girl's serene presence amidst the natural beauty, with a balanced blend of warm and cool tones."

image = pipe(
    prompt,
    control_image=[
        control_image,
        control_image,
    ],  # try with different conds such as canny&depth, pose&depth
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.35, 0.35],
    control_guidance_end=[0.8, 0.8],
    num_inference_steps=30,
    guidance_scale=3.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
