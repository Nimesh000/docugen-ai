"""LTX-Video image-to-video for the 'hero' scenes. Runs inside a GPU container."""
from __future__ import annotations

import io
import os
import tempfile


class Animator:
    def __init__(self, device: str = "cuda"):
        import torch
        from diffusers import LTXImageToVideoPipeline

        from .. import config

        self.pipe = LTXImageToVideoPipeline.from_pretrained(config.VIDEO_MODEL, torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        self.torch = torch
        self.device = device
        self.cfg = config

    def animate(self, image_png: bytes, prompt: str, seed: int, steps: int = 40) -> bytes:
        from diffusers.utils import export_to_video
        from PIL import Image

        cfg = self.cfg
        image = Image.open(io.BytesIO(image_png)).convert("RGB").resize((cfg.VIDEO_W, cfg.VIDEO_H), Image.LANCZOS)
        g = self.torch.Generator(device=self.device).manual_seed(seed)
        frames = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted, morphing, text",
            width=cfg.VIDEO_W, height=cfg.VIDEO_H, num_frames=cfg.VIDEO_FRAMES,
            num_inference_steps=steps, guidance_scale=3.0, generator=g,
        ).frames[0]
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            export_to_video(frames, path, fps=24)
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            os.unlink(path)
