"""Juggernaut XL (SDXL fine-tune) stills. Runs inside a GPU container."""
from __future__ import annotations

import io


class Painter:
    def __init__(self, device: str = "cuda"):
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        from .. import config

        kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(config.IMAGE_MODEL, variant="fp16", **kwargs)
        except (OSError, ValueError):  # repo without an fp16 variant
            pipe = StableDiffusionXLPipeline.from_pretrained(config.IMAGE_MODEL, **kwargs)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        self.pipe = pipe.to(device)
        self.torch = torch
        self.device = device

    def paint(self, prompt: str, negative: str, seed: int, width: int, height: int, steps: int = 30) -> bytes:
        g = self.torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(prompt=prompt, negative_prompt=negative, width=width, height=height,
                          num_inference_steps=steps, guidance_scale=5.0, generator=g).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
