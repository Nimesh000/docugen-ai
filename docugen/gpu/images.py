"""Z-Image-Turbo stills (6B DiT, 8 denoising steps, photorealistic). Runs inside a GPU container."""
from __future__ import annotations

import io


class Painter:
    def __init__(self, device: str = "cuda"):
        import torch
        from diffusers import ZImagePipeline

        from .. import config

        self.pipe = ZImagePipeline.from_pretrained(config.IMAGE_MODEL, torch_dtype=torch.bfloat16).to(device)
        self.torch = torch
        self.device = device
        self.cfg = config

    def paint(self, prompt: str, seed: int, width: int = 1536, height: int = 864) -> bytes:
        cfg = self.cfg
        g = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            image = self.pipe(prompt=prompt, width=width, height=height,
                              num_inference_steps=cfg.IMAGE_STEPS, guidance_scale=0.0, generator=g).images[0]
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=94)
        return buf.getvalue()
