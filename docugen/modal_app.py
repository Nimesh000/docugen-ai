"""Modal deployment: public web app + CPU orchestrator + three GPU model services.

Deploy:   modal deploy docugen/modal_app.py
CLI run:  modal run docugen/modal_app.py --topic "The rise of India's UPI"

Everything heavy is imported inside functions, so deploying only needs the `modal` package locally.
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "docugen"
app = modal.App(APP_NAME)

secrets = [modal.Secret.from_name("docugen-secrets")]           # GROQ_API_KEY (+ optional ACCESS_CODE, DAILY_LIMIT)
cache_vol = modal.Volume.from_name("docugen-cache", create_if_missing=True)   # model weights
jobs_vol = modal.Volume.from_name("docugen-jobs", create_if_missing=True)     # generated films
STATE = modal.Dict.from_name("docugen-state", create_if_missing=True)          # live job status

CACHE_ENV = {
    "HF_HOME": "/cache/hf",
    "TTS_HOME": "/cache/tts",
    "XDG_DATA_HOME": "/cache/xdg",
    "COQUI_TOS_AGREED": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
SRC_IGNORE = ["**/__pycache__/**", "**/*.pyc"]

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "unzip", "fonts-dejavu-core")
    .run_commands(
        "mkdir -p /root/fonts && curl -L -o /tmp/inter.zip "
        "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip && "
        "unzip -j -o /tmp/inter.zip 'extras/otf/Inter-Bold.otf' 'extras/otf/Inter-Regular.otf' -d /root/fonts && "
        "rm /tmp/inter.zip"
    )
    .pip_install("fastapi==0.115.12", "openai>=1.40", "pillow>=10.4", "numpy>=1.26", "faster-whisper==1.2.1",
                 "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("docugen", ignore=SRC_IGNORE)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install("coqui-tts==0.27.5", "diffusers==0.35.2", "transformers==4.57.6", "accelerate>=1.0",
                 "sentencepiece", "protobuf", "imageio", "imageio-ffmpeg", "pillow", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("docugen", ignore=SRC_IGNORE)
)


# --------------------------------------------------------------------------- GPU services
@app.cls(image=gpu_image, gpu="T4", volumes={"/cache": cache_vol}, timeout=900, scaledown_window=60,
         max_containers=1, memory=16384)
class VoiceGPU:
    @modal.enter()
    def load(self):
        from docugen.gpu.tts import Narrator

        self.model = Narrator()
        cache_vol.commit()

    @modal.method()
    def speak(self, text: str, speaker: str) -> bytes:
        return self.model.speak(text, speaker)


@app.cls(image=gpu_image, gpu="L4", volumes={"/cache": cache_vol}, timeout=900, scaledown_window=60,
         max_containers=1, memory=24576)
class ImageGPU:
    @modal.enter()
    def load(self):
        from docugen.gpu.images import Painter

        self.model = Painter()
        cache_vol.commit()

    @modal.method()
    def paint(self, prompt: str, seed: int) -> bytes:
        from docugen import config

        return self.model.paint(prompt, config.NEGATIVE_PROMPT, seed, config.IMAGE_W, config.IMAGE_H)


@app.cls(image=gpu_image, gpu="L40S", volumes={"/cache": cache_vol}, timeout=1200, scaledown_window=60,
         max_containers=1, memory=32768)
class MotionGPU:
    @modal.enter()
    def load(self):
        from docugen.gpu.video import Animator

        self.model = Animator()
        cache_vol.commit()

    @modal.method()
    def animate(self, image_png: bytes, prompt: str, seed: int) -> bytes:
        return self.model.animate(image_png, prompt, seed)


# --------------------------------------------------------------------------- orchestrator
class ModalBackend:
    def write_script(self, topic, seconds, heroes, log):
        from docugen.script import write_script

        return write_script(topic, seconds, heroes, log)

    def voice(self, texts, speaker):
        return VoiceGPU().speak.map(texts, kwargs={"speaker": speaker})

    def images(self, prompts, seeds):
        return ImageGPU().paint.map(prompts, seeds)

    def animate(self, items):
        pngs, prompts, seeds = zip(*items)
        return MotionGPU().animate.map(list(pngs), list(prompts), list(seeds), return_exceptions=True)


@app.function(image=base_image, secrets=secrets, volumes={"/jobs": jobs_vol, "/cache": cache_vol},
              cpu=8.0, memory=8192, timeout=3600)
def run_job(job_id: str, params: dict) -> bool:
    from docugen import pipeline
    from docugen.web.server import remember_recent

    state = STATE.get(job_id) or pipeline.new_state(job_id, params)

    def publish() -> None:
        try:
            jobs_vol.commit()
        except Exception as exc:  # never fail a job because of a sync hiccup
            print("volume commit failed:", exc)

    status = pipeline.Status(STATE, job_id, state, publish=publish)
    ok = pipeline.run(params, Path("/jobs") / job_id, ModalBackend(), status)
    if ok:
        remember_recent(STATE, job_id)
    publish()
    return ok


# --------------------------------------------------------------------------- public web app
@app.function(image=base_image, secrets=secrets, volumes={"/jobs": jobs_vol}, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label=APP_NAME)
def web():
    from docugen.web.server import create_app

    def spawn(job_id: str, params: dict) -> None:
        run_job.spawn(job_id, params)

    def refresh() -> None:
        try:
            jobs_vol.reload()
        except Exception as exc:
            print("volume reload skipped:", exc)

    return create_app(STATE, spawn, Path("/jobs"), refresh)


# --------------------------------------------------------------------------- terminal entry point
@app.local_entrypoint()
def main(topic: str = "The rise of India's UPI", seconds: int = 120, style: str = "cinematic",
         voice: str = "Damien Black", motion: int = 3):
    """modal run docugen/modal_app.py --topic "..."  -> generates a film and prints where to download it."""
    import time

    job_id = time.strftime("%Y%m%d-%H%M%S-cli")
    params = {"topic": topic, "seconds": seconds, "style": style, "voice": voice, "motion": motion,
              "music": True, "subtitles": True}
    ok = run_job.remote(job_id, params)
    print(("Done" if ok else "Failed") + f": job {job_id}. Download with:\n"
          f"  modal volume get docugen-jobs {job_id}/output ./{job_id}")
