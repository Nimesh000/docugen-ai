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
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}
SRC_IGNORE = ["**/__pycache__/**", "**/*.pyc"]
SPACY_EN = ("https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")

# CPU image: web app + orchestrator (Groq client, Kokoro TTS on CPU, FFmpeg editing)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "unzip", "fonts-dejavu-core", "espeak-ng")
    .run_commands(
        "mkdir -p /root/fonts && curl -L -o /tmp/inter.zip "
        "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip && "
        "unzip -j -o /tmp/inter.zip 'extras/otf/Inter-Bold.otf' 'extras/otf/Inter-Regular.otf' -d /root/fonts && "
        "rm /tmp/inter.zip"
    )
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.57.6", "soundfile", "fastapi==0.115.12",
                 "openai>=1.40", "pillow>=10.4", "numpy>=1.26", "hf_transfer", SPACY_EN)
    .env(CACHE_ENV)
    .add_local_python_source("docugen", ignore=SRC_IGNORE)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
    .pip_install("diffusers==0.37.0", "transformers==4.57.6", "accelerate>=1.0", "sentencepiece", "protobuf",
                 "ftfy", "imageio", "imageio-ffmpeg", "pillow", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("docugen", ignore=SRC_IGNORE)
)

MAX_IMAGE_GPUS = 4
MAX_MOTION_GPUS = 4


# --------------------------------------------------------------------------- GPU services
@app.cls(image=gpu_image, gpu="L40S", volumes={"/cache": cache_vol}, timeout=900, scaledown_window=120,
         max_containers=MAX_IMAGE_GPUS, memory=32768)
class ImageGPU:
    @modal.enter()
    def load(self):
        from docugen.gpu.images import Painter

        self.model = Painter()

    @modal.method()
    def warm(self) -> bool:
        return True

    @modal.method()
    def paint(self, prompt: str, seed: int) -> bytes:
        return self.model.paint(prompt, seed)


@app.cls(image=gpu_image, gpu="H200", volumes={"/cache": cache_vol}, timeout=1200, scaledown_window=150,
         max_containers=MAX_MOTION_GPUS, memory=98304)
class MotionGPU:
    @modal.enter()
    def load(self):
        from docugen.gpu.video import Animator

        self.model = Animator()

    @modal.method()
    def warm(self) -> bool:
        return True

    @modal.method()
    def animate(self, image: bytes, prompt: str, seed: int) -> bytes:
        return self.model.animate(image, prompt, seed)


# --------------------------------------------------------------------------- orchestrator
_NARRATOR = None


class _Call:
    def __init__(self, call):
        self.call = call

    def result(self) -> bytes:
        return self.call.get(timeout=1200)


class ModalBackend:
    def prewarm(self, images: int, motions: int) -> None:
        for _ in range(max(1, min(MAX_IMAGE_GPUS, -(-images // 6)))):
            ImageGPU().warm.spawn()
        for _ in range(min(MAX_MOTION_GPUS, motions)):
            MotionGPU().warm.spawn()

    def write_script(self, topic, seconds, heroes, log, scenes):
        from docugen.script import write_script

        return write_script(topic, seconds, heroes, log, scenes)

    def narrate(self, texts, voice, speed):
        global _NARRATOR
        from docugen.tts import Narrator

        if _NARRATOR is None:
            _NARRATOR = Narrator()
        for t in texts:
            yield _NARRATOR.speak(t, voice, speed)

    def images(self, prompts, seeds):
        return ImageGPU().paint.map(prompts, seeds, return_exceptions=True)

    def start_animate(self, image, prompt, seed):
        return _Call(MotionGPU().animate.spawn(image, prompt, seed))


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
    elif status.state.get("stage") == "script":
        refund_daily_quota(STATE, status.state.get("created_at"))
    publish()
    return ok


def refund_daily_quota(store, created_at) -> None:
    """A job that fails before any GPU work should not use up one of the day's films."""
    from datetime import datetime, timezone

    try:
        day = datetime.fromtimestamp(created_at or 0, timezone.utc).strftime("%Y-%m-%d")
        key = f"count:{day}"
        store[key] = max(0, int(store.get(key, 0) or 0) - 1)
    except Exception as exc:
        print("quota refund skipped:", exc)


@app.function(image=base_image, volumes={"/cache": cache_vol}, cpu=4.0, memory=8192, timeout=3600)
def prefetch() -> dict:
    """Download every model into the cache volume once (run before the first film)."""
    import time

    from huggingface_hub import snapshot_download

    from docugen import config

    sizes = {}
    for repo in (config.TTS_REPO, config.IMAGE_MODEL, config.VIDEO_MODEL):
        t = time.time()
        path = Path(snapshot_download(repo, ignore_patterns=["*.md", "*.gif", "*.mp4", "assets/*", "examples/*"]))
        gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        sizes[repo] = round(gb, 1)
        print(f"{repo}: {gb:.1f} GB in {time.time() - t:.0f} s", flush=True)
        cache_vol.commit()
    # warm the TTS + spaCy path too
    from docugen.tts import Narrator

    wav, words = Narrator().speak("Prefetch complete. The narrator is ready.", "bm_george", 1.05)
    print(f"TTS ok: {len(wav)} bytes, {len(words)} timed words: {words[:3]}", flush=True)
    cache_vol.commit()
    return sizes


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
def main(topic: str = "The rise of India's UPI", seconds: int = 60, scenes: int = 6, motion: int = 2,
         style: str = "cinematic", voice: str = "bm_george"):
    """modal run docugen/modal_app.py --topic "..."  -> generates a film and prints where to download it."""
    import json
    import time

    job_id = time.strftime("%Y%m%d-%H%M%S-cli")
    params = {"topic": topic, "seconds": seconds, "scenes": scenes, "style": style, "voice": voice,
              "motion": motion, "music": True, "subtitles": True}
    t = time.time()
    ok = run_job.remote(job_id, params)
    state = STATE.get(job_id) or {}
    print("\n".join(state.get("log", [])))
    print(json.dumps({k: {"status": v["status"], "seconds": v.get("seconds")} for k, v in
                      state.get("stages", {}).items()}, indent=1))
    print(("Done" if ok else f"Failed: {state.get('error')}") + f" in {time.time() - t:.0f} s: job {job_id}. "
          f"Download with:\n  modal volume get docugen-jobs {job_id}/output ./{job_id}")
    if not ok:
        raise SystemExit(1)
