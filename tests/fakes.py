"""A CPU-only fake backend so the whole pipeline (FFmpeg editing, captions, cards, UI) can be tested without GPUs."""
from __future__ import annotations

import io
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from docugen.script import normalize

_POOL = ThreadPoolExecutor(max_workers=4)


class _Failed:
    def result(self):
        raise RuntimeError("simulated GPU out of memory")


class FakeBackend:
    def __init__(self, fail_motion_index: int | None = 1):
        self.fail_motion_index = fail_motion_index
        self.animations = 0
        self.prewarmed = None

    def prewarm(self, images, motions):
        self.prewarmed = (images, motions)

    def write_script(self, topic, seconds, heroes, log, scenes=5):
        sc = [{"narration": f"Scene {i + 1}. In 2016 India launched a new way to pay, simple and instant for everyone.",
               "shots": [f"wide market scene {i + 1}", f"medium shot {i + 1}", f"close-up {i + 1}", f"angle {i + 1}"],
               "motion": "people walk past, slow dolly in", "hero": False} for i in range(scenes)]
        log("fake script written")
        return normalize({"title": f"The Story of {topic}", "logline": "How a nation learned to pay with a tap.",
                          "scenes": sc}, topic, heroes, scenes)

    def narrate(self, texts, voice, speed):
        for t in texts:
            toks = t.split()
            secs = max(2.0, len(toks) / (2.6 * speed / 1.05))
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "v.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                                f"sine=frequency=220:sample_rate=24000:duration={secs:.2f}",
                                "-af", "volume=0.2", "-ac", "1", str(p)], check=True)
                step = secs / len(toks)
                words = [{"w": w, "s": round(k * step, 3), "e": round((k + 0.9) * step, 3)} for k, w in enumerate(toks)]
                yield p.read_bytes(), words

    def images(self, prompts, seeds):
        for i, prompt in enumerate(prompts):
            img = Image.new("RGB", (1536, 864))
            d = ImageDraw.Draw(img)
            for y in range(864):
                d.line((0, y, 1536, y), fill=(30 + i * 40 % 200, 60 + y // 7, 120 + i * 25 % 120))
            for k in range(0, 1536, 96):
                d.ellipse((k, 300 + (k * 7 + i * 50) % 300, k + 60, 360 + (k * 7 + i * 50) % 300), fill=(240, 200, 120))
            d.text((40, 40), prompt[:40], fill=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            yield buf.getvalue()

    def start_animate(self, image, prompt, seed):
        k = self.animations
        self.animations += 1
        if k == self.fail_motion_index:
            return _Failed()
        return _POOL.submit(self._animate, image)

    @staticmethod
    def _animate(image):
        with tempfile.TemporaryDirectory() as d:
            src, p = Path(d) / "in.jpg", Path(d) / "m.mp4"
            src.write_bytes(image)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(src), "-vf",
                            "scale=960:544,zoompan=z='1+0.002*on':d=81:s=960x544:fps=16", "-frames:v", "81",
                            "-pix_fmt", "yuv420p", str(p)], check=True)
            return p.read_bytes()
