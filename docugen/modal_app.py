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

# CPU image: web app + orchestrator (Groq client, FFmpeg editing)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "unzip", "fonts-dejavu-core")
    .run_commands(
        "mkdir -p /root/fonts && curl -L -o /tmp/inter.zip "
        "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip && "
        "unzip -j -o /tmp/inter.zip 'extras/otf/Inter-Bold.otf' 'extras/otf/Inter-Regular.otf' -d /root/fonts && "
        "rm /tmp/inter.zip"
    )
    .pip_install("fastapi==0.115.12", "openai>=1.40", "pillow>=10.4", "numpy>=1.26", "huggingface_hub<1.0",
                 "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("docugen", ignore=SRC_IGNORE)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "espeak-ng")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
    .pip_install("diffusers==0.37.0", "transformers==4.57.6", "accelerate>=1.0", "sentencepiece", "protobuf",
                 "ftfy", "imageio", "imageio-ffmpeg", "pillow", "numpy", "hf_transfer",
                 "kokoro==0.9.4", "misaki[en]==0.9.4", "soundfile", SPACY_EN)
    .env(CACHE_ENV)
    .add_local_python_source("docugen", ignore=SRC_IGNORE)
)

MAX_IMAGE_GPUS = 4
MAX_MOTION_GPUS = 4


class _Meter:
    """Mixin for GPU classes: measures model-load time once and the busy time of every call, so each film can
    report what it really cost."""

    def _start_load(self):
        import time

        self._t0 = time.time()

    def _end_load(self):
        import os
        import time

        import torch

        self._load = time.time() - self._t0
        self._gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self._cid = os.environ.get("MODAL_TASK_ID") or f"{os.getpid()}-{self._t0}"

    def _meta(self, busy: float) -> dict:
        load, self._load = self._load, 0.0      # report the load only once per container
        return {"gpu": self._gpu, "cid": self._cid, "busy": round(busy, 2), "load": round(load, 2)}


# --------------------------------------------------------------------------- GPU services
@app.cls(image=gpu_image, gpu=["T4", "L4", "A10G"], volumes={"/cache": cache_vol}, timeout=600, scaledown_window=45,
         max_containers=2, memory=16384)
class VoiceGPU(_Meter):
    @modal.enter()
    def load(self):
        from docugen.tts import Narrator

        self._start_load()
        self.model = Narrator(device="cuda")
        self.model.speak("Ready.", "bm_george", 1.0)   # warm up G2P + CUDA kernels
        self._end_load()

    @modal.method()
    def warm(self) -> dict:
        return self._meta(0.0)

    @modal.method()
    def speak_all(self, texts: list[str], voice: str, speed: float) -> dict:
        import time

        t = time.time()
        items = [self.model.speak(x, voice, speed) for x in texts]
        return {"items": items, **self._meta(time.time() - t)}


@app.cls(image=gpu_image, gpu=["L40S", "A100-40GB", "A100-80GB", "H100"], volumes={"/cache": cache_vol}, timeout=900,
         scaledown_window=45, max_containers=MAX_IMAGE_GPUS, memory=32768)
class ImageGPU(_Meter):
    @modal.enter()
    def load(self):
        from docugen.gpu.images import Painter

        self._start_load()
        self.model = Painter()
        self._end_load()

    @modal.method()
    def warm(self) -> dict:
        return self._meta(0.0)

    @modal.method()
    def paint(self, prompt: str, seed: int, width: int = 1536, height: int = 864) -> dict:
        import time

        t = time.time()
        img = self.model.paint(prompt, seed, width, height)
        return {"data": img, **self._meta(time.time() - t)}


@app.cls(image=gpu_image, gpu=["H200", "B200"], volumes={"/cache": cache_vol}, timeout=1200, scaledown_window=40,
         max_containers=MAX_MOTION_GPUS, memory=98304)
class MotionGPU(_Meter):
    @modal.enter()
    def load(self):
        from docugen.gpu.video import Animator

        self._start_load()
        self.model = Animator()
        self._end_load()

    @modal.method()
    def warm(self) -> dict:
        return self._meta(0.0)

    @modal.method()
    def animate(self, image: bytes, prompt: str, seed: int, width: int = 960, height: int = 544) -> dict:
        import time

        t = time.time()
        clip = self.model.animate(image, prompt, seed, width, height)
        return {"data": clip, **self._meta(time.time() - t)}


# --------------------------------------------------------------------------- orchestrator
class _Call:
    def __init__(self, future, meter):
        self.future, self.meter = future, meter

    def result(self) -> bytes:
        return self.meter("motion", self.future.result(timeout=1500))


class ModalBackend:
    """Talks to the GPU classes. Render mode decides how many GPU containers run at once: every extra container
    loads the model again (paid GPU seconds), so 'economy' keeps it to the minimum."""

    def __init__(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self._lock = threading.Lock()
        self._usage: dict = {}
        self._warm: list = []
        self._pools: dict = {}
        self._Pool = ThreadPoolExecutor

    def _pool(self, kind: str, workers: int):
        key = (kind, workers)
        if key not in self._pools:
            self._pools[key] = self._Pool(max_workers=workers)
        return self._pools[key]

    def _meter(self, kind: str, res):
        if isinstance(res, dict) and "gpu" in res:
            with self._lock:
                u = self._usage.setdefault(kind, {"gpu": res["gpu"], "busy": 0.0, "load": 0.0, "cids": set()})
                u["busy"] += res.get("busy", 0.0)
                u["load"] += res.get("load", 0.0)
                u["cids"].add(res.get("cid"))
            return res.get("data", res.get("items"))
        return res

    def prewarm(self, images: int, motions: int, render: str = "economy") -> None:
        from docugen import config

        mode = config.RENDER_MODES.get(render, config.RENDER_MODES["economy"])
        self._warm.append(("voice", VoiceGPU().warm.spawn()))
        for _ in range(max(1, min(mode["image_gpus"], -(-images // 6)))):
            self._warm.append(("image", ImageGPU().warm.spawn()))
        for _ in range(min(mode["motion_gpus"], motions)):
            self._warm.append(("motion", MotionGPU().warm.spawn()))

    def write_script(self, brief, log):
        from docugen.script import write_script

        return write_script(brief, log)

    def fact_check(self, script, topic, log):
        from docugen.script import fact_check

        return fact_check(script, topic, log)

    def direct(self, script, counts, brief, log):
        from docugen.director import direct

        return direct(script, counts, brief, log)

    def narrate(self, texts, voice, speed):
        return self._meter("voice", VoiceGPU().speak_all.remote(texts, voice, speed))

    def images(self, prompts, seeds, size, render="economy"):
        from docugen import config

        workers = config.RENDER_MODES.get(render, config.RENDER_MODES["economy"])["image_gpus"]
        w, h = size

        def one(args):
            try:
                return self._meter("image", ImageGPU().paint.remote(args[0], args[1], w, h))
            except Exception as exc:  # keep going; the pipeline reuses the previous picture
                return exc

        return self._pool("image", workers).map(one, list(zip(prompts, seeds)))

    def start_animate(self, image, prompt, seed, size, render="economy"):
        from docugen import config

        workers = config.RENDER_MODES.get(render, config.RENDER_MODES["economy"])["motion_gpus"]
        w, h = size
        fut = self._pool("motion", workers).submit(lambda: MotionGPU().animate.remote(image, prompt, seed, w, h))
        return _Call(fut, self._meter)

    def usage(self) -> dict:
        for kind, call in self._warm:          # pre-warmed containers count too (their model load was paid)
            try:
                self._meter(kind, call.get(timeout=0))
            except Exception:
                pass
        self._warm = []
        with self._lock:
            return {k: {"gpu": u["gpu"], "busy": round(u["busy"], 1), "load": round(u["load"], 1),
                        "containers": len(u["cids"])} for k, u in self._usage.items()}


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
    wav, words = VoiceGPU().speak_all.remote(["Prefetch complete. The narrator is ready."], "bm_george", 1.1)["items"][0]
    print(f"TTS ok: {len(wav)} bytes, {len(words)} timed words: {words[:3]}", flush=True)
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
def main(topic: str = "The rise of India's UPI", format: str = "long", seconds: int = 60, scenes: int = 0,
         motion: int = 2, style: str = "cinematic", voice: str = "bm_george", render: str = "economy"):
    """modal run docugen/modal_app.py --topic "..." [--format reel]  -> generates a film and prints the timings."""
    import json
    import time

    from docugen import config

    job_id = time.strftime("%Y%m%d-%H%M%S-cli")
    params = {"topic": topic, "format": format, "seconds": seconds,
              "scenes": scenes or config.default_scenes(seconds, format), "style": style, "voice": voice,
              "motion": motion, "render": render}
    t = time.time()
    ok = run_job.remote(job_id, params)
    state = STATE.get(job_id) or {}
    print("\n".join(state.get("log", [])))
    print(json.dumps({k: {"status": v["status"], "seconds": v.get("seconds")} for k, v in
                      state.get("stages", {}).items()}, indent=1))
    print("Estimate:", json.dumps(state.get("estimate")), "\nMeasured cost:", json.dumps(state.get("cost")))
    print(("Done" if ok else f"Failed: {state.get('error')}") + f" in {time.time() - t:.0f} s: job {job_id}. "
          f"Download with:\n  modal volume get docugen-jobs {job_id}/output ./{job_id}")
    if not ok:
        raise SystemExit(1)
