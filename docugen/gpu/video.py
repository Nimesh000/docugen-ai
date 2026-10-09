"""Wan 2.2 image-to-video (A14B MoE with Lightning 4-step distillation). Runs inside a GPU container."""
from __future__ import annotations

import io
import os
import tempfile


class Animator:
    def __init__(self, device: str = "cuda"):
        import torch
        from diffusers import UniPCMultistepScheduler, WanImageToVideoPipeline

        from .. import config

        pipe = WanImageToVideoPipeline.from_pretrained(config.VIDEO_MODEL, torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        self.pipe = pipe.to(device)
        self.torch = torch
        self.device = device
        self.cfg = config

    def animate(self, image_bytes: bytes, prompt: str, seed: int, width: int = 960, height: int = 544) -> bytes:
        from diffusers.utils import export_to_video
        from PIL import Image, ImageOps

        cfg = self.cfg
        image = ImageOps.fit(Image.open(io.BytesIO(image_bytes)).convert("RGB"), (width, height),
                             Image.LANCZOS)
        g = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            frames = self.pipe(
                image=image, prompt=prompt + ", smooth natural motion, stable consistent subject, cinematic",
                negative_prompt="static, frozen, blurry, distorted, morphing, extra limbs, flicker, scene cut, text, watermark",
                width=width, height=height, num_frames=cfg.VIDEO_FRAMES,
                num_inference_steps=cfg.VIDEO_STEPS, guidance_scale=1.0, guidance_scale_2=1.0, generator=g,
            ).frames[0]
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            export_to_video(frames, path, fps=cfg.VIDEO_FPS, quality=9)
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            os.unlink(path)
