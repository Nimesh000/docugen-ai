"""A CPU-only fake backend so the whole pipeline (FFmpeg editing, captions, cards, UI) can be tested without GPUs."""
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from docugen.script import normalize


class FakeBackend:
    def __init__(self, fail_motion_index: int | None = 1):
        self.fail_motion_index = fail_motion_index

    def write_script(self, topic, seconds, heroes, log):
        scenes = [{"narration": f"Scene {i + 1}. " + "India built a new way to pay, simple and instant. " * 2,
                   "visual": f"street market scene number {i + 1}", "motion": "slow push in", "hero": i in (1, 3)}
                  for i in range(5)]
        log("fake script written")
        return normalize({"title": f"The Story of {topic}", "logline": "How a nation learned to pay with a tap.",
                          "scenes": scenes}, topic, heroes)

    def voice(self, texts, speaker):
        for t in texts:
            secs = max(2.0, len(t.split()) / 2.6)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "v.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                                f"sine=frequency=220:sample_rate=24000:duration={secs:.2f}",
                                "-af", "volume=0.2", "-ac", "1", str(p)], check=True)
                yield p.read_bytes()

    def images(self, prompts, seeds):
        for i, prompt in enumerate(prompts):
            img = Image.new("RGB", (1344, 768))
            d = ImageDraw.Draw(img)
            for y in range(768):
                d.line((0, y, 1344, y), fill=(30 + i * 40 % 200, 60 + y // 6, 120 + i * 25 % 120))
            for k in range(0, 1344, 96):
                d.ellipse((k, 300 + (k * 7 + i * 50) % 300, k + 60, 360 + (k * 7 + i * 50) % 300), fill=(240, 200, 120))
            d.text((40, 40), f"SCENE {i + 1}", fill=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            yield buf.getvalue()

    def animate(self, items):
        for k, (png, prompt, seed) in enumerate(items):
            if k == self.fail_motion_index:
                yield RuntimeError("simulated GPU out of memory")
                continue
            with tempfile.TemporaryDirectory() as d:
                src, p = Path(d) / "in.png", Path(d) / "m.mp4"
                src.write_bytes(png)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(src), "-vf",
                                "scale=768:448,zoompan=z='1+0.002*on':d=121:s=768x448:fps=24", "-frames:v", "121",
                                "-pix_fmt", "yuv420p", str(p)], check=True)
                yield p.read_bytes()
