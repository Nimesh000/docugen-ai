"""What a film costs on Modal: an up-front estimate for the web page, and the measured bill after the run.

Prices are Modal's published per-second rates (modal.com/pricing). GPU time is measured inside the GPU containers
(model load + every call), idle time before a container scales down is added on top, and the CPU orchestrator is
billed for its wall-clock time. Groq LLM calls are not included (free tier / fractions of a cent).
"""
from __future__ import annotations

from . import config

GPU_PRICE = {  # USD per second
    "B200": 0.001736, "H200": 0.001261, "H100": 0.001097, "A100-80": 0.000694, "A100-40": 0.000583,
    "L40S": 0.000542, "A10": 0.000306, "L4": 0.000222, "T4": 0.000164,
}
CPU_CORE_PRICE = 0.0000131        # per physical core per second
MEM_GIB_PRICE = 0.00000222        # per GiB per second
ORCHESTRATOR_CORES, ORCHESTRATOR_GIB = 8.0, 8.0

# scale-down windows used in modal_app.py (idle time that is still billed after the last call)
IDLE = {"voice": 45, "image": 30, "motion": 40}

# measured on Modal and used for the up-front estimate
ASSUME = {
    "voice_gpu": "T4", "voice_load": 15.0, "voice_per_min": 4.0,
    "image_gpu": "L40S", "image_load": 30.0, "image_each": 6.0,
    "motion_gpu": "H200", "motion_load": 60.0, "motion_each": 38.0,
    "cpu_base": 45.0, "cpu_per_shot": 1.6,
}


def gpu_key(name: str) -> str:
    """Map a CUDA device name ('NVIDIA L40S', 'Tesla T4', 'NVIDIA A100-SXM4-80GB', ...) to a price key."""
    n = (name or "").upper()
    if "A100" in n:
        return "A100-80" if "80" in n else "A100-40"
    for k in ("B200", "H200", "H100", "L40S", "A10", "L4", "T4"):
        if k in n:
            return k
    return "H200" if "H" in n[:2] else "L40S"


def price(name: str) -> float:
    return GPU_PRICE[gpu_key(name)]


def estimate(params: dict) -> dict:
    """Rough cost and time before the run, from the user's choices."""
    f = config.fmt(params.get("format"))
    seconds = int(params.get("seconds") or f["default_seconds"])
    heroes = int(params.get("motion") or 0)
    mode = config.RENDER_MODES.get(params.get("render") or "economy", config.RENDER_MODES["economy"])
    shot = config.shot_seconds(params.get("format"), params.get("pacing"))
    narr = config.narration_seconds(seconds, params.get("format"))
    stills = max(int(params.get("scenes") or 3), round(max(0.0, narr - heroes * config.MOTION_SECONDS) / shot))
    shots = stills + heroes
    a = ASSUME
    per = 16 if (params.get("render") or "economy") == "economy" else 6
    img_containers = max(1, min(mode["image_gpus"], -(-stills // per)))
    mot_containers = min(mode["motion_gpus"], heroes)
    voice = (a["voice_load"] + a["voice_per_min"] * narr / 60 + IDLE["voice"]) * GPU_PRICE[a["voice_gpu"]]
    images = (img_containers * (a["image_load"] + IDLE["image"]) + stills * a["image_each"]) * GPU_PRICE[a["image_gpu"]]
    motion = (mot_containers * (a["motion_load"] + IDLE["motion"]) + heroes * a["motion_each"]) * GPU_PRICE[a["motion_gpu"]]
    image_wall = a["image_load"] + stills * a["image_each"] / img_containers
    motion_wall = (a["motion_load"] + heroes * a["motion_each"] / max(1, mot_containers)) if heroes else 0
    wall = 20 + max(image_wall + 25, motion_wall + 15) + a["cpu_base"] + a["cpu_per_shot"] * shots * seconds / 60
    cpu = wall * (ORCHESTRATOR_CORES * CPU_CORE_PRICE + ORCHESTRATOR_GIB * MEM_GIB_PRICE)
    total = voice + images + motion + cpu
    return {"usd": round(total, 3), "minutes": round(wall / 60, 1), "shots": shots, "stills": stills,
            "breakdown": {"voice": round(voice, 4), "images": round(images, 4), "motion": round(motion, 4),
                          "cpu": round(cpu, 4)}}


def bill(usage: dict, wall_seconds: float) -> dict:
    """Measured cost. usage = {"voice"|"image"|"motion": {"gpu": name, "busy": s, "load": s, "containers": n}}."""
    parts = {}
    for kind, u in usage.items():
        if not u or not u.get("gpu"):
            continue
        secs = u.get("busy", 0) + u.get("load", 0) + u.get("containers", 0) * IDLE.get(kind, 0)
        parts[kind] = round(secs * price(u["gpu"]), 4)
    parts["cpu"] = round(wall_seconds * (ORCHESTRATOR_CORES * CPU_CORE_PRICE + ORCHESTRATOR_GIB * MEM_GIB_PRICE), 4)
    return {"usd": round(sum(parts.values()), 3), "breakdown": parts,
            "gpus": {k: u.get("gpu") for k, u in usage.items() if u and u.get("gpu")}}
