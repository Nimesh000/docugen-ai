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

    def prewarm(self, images, motions, render="economy"):
        self.prewarmed = (images, motions)

    def usage(self):
        return {"image": {"gpu": "NVIDIA L40S", "busy": 12.0, "load": 30.0, "containers": 1}}

    def fact_check(self, script, topic, log):
        log("fake fact-check")
        return ["Corrected a launch year"]

    def direct(self, script, counts, brief, log):
        self.counts = counts
        self.brief = brief
        return {"bible": {"era": "2016-2024", "places": ["Mumbai"]}, "stats": {"shots": sum(counts)},
                "scenes": [{"shots": [f"director shot {i}.{j} in Mumbai, India, 2019" for j in range(c)],
                            "motion": "vendor waves, slow push in", "from_agent": True}
                           for i, c in enumerate(counts)]}

    def write_script(self, brief, log):
        topic, heroes, scenes = brief["topic"], brief["heroes"], brief["scenes"]
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

    def images(self, prompts, seeds, size=(1536, 864), render="economy"):
        self.image_size = size
        self.prompts = list(prompts)
        w, h = size
        for i, prompt in enumerate(prompts):
            img = Image.new("RGB", (w, h))
            d = ImageDraw.Draw(img)
            for y in range(h):
                d.line((0, y, w, y), fill=(30 + i * 40 % 200, 60 + y // 7 % 120, 120 + i * 25 % 120))
            for k in range(0, w, 96):
                d.ellipse((k, 300 + (k * 7 + i * 50) % 300, k + 60, 360 + (k * 7 + i * 50) % 300), fill=(240, 200, 120))
            d.text((40, 40), prompt[:40], fill=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            yield buf.getvalue()

    def start_animate(self, image, prompt, seed, size=(960, 544), render="economy"):
        k = self.animations
        self.animations += 1
        self.motion_prompt = prompt
        if k == self.fail_motion_index:
            return _Failed()
        return _POOL.submit(self._animate, image, size)

    @staticmethod
    def _animate(image, size):
        w, h = size
        with tempfile.TemporaryDirectory() as d:
            src, p = Path(d) / "in.jpg", Path(d) / "m.mp4"
            src.write_bytes(image)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(src), "-vf",
                            f"scale={w}:{h},zoompan=z='1+0.002*on':d=81:s={w}x{h}:fps=16", "-frames:v", "81",
                            "-pix_fmt", "yuv420p", str(p)], check=True)
            return p.read_bytes()
