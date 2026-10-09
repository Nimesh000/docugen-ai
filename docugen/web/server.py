"""FastAPI web app. The same app is served by Modal in production and by `python -m docugen.dev` locally."""
from __future__ import annotations

import hmac
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import config, cost
from ..pipeline import new_state

STATIC = Path(__file__).parent / "static"
ALLOWED_EXT = {".png", ".jpg", ".mp4", ".wav", ".srt", ".md", ".json"}


class JobRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=config.MAX_TOPIC_CHARS)
    format: str = config.DEFAULT_FORMAT
    seconds: int = 60
    scenes: int = Field(default=5, ge=config.MIN_SCENES, le=config.MAX_SCENES)
    motion: int = Field(default=2, ge=0, le=config.MAX_HEROES)
    style: str = "cinematic"
    voice: str = config.DEFAULT_VOICE
    tone: str = "informative"
    audience: str = "general"
    pacing: str = "balanced"
    captions: str = ""
    music: str = "ambient"
    render: str = "economy"
    fact_check: bool = True
    key_points: str = Field(default="", max_length=config.MAX_KEY_POINTS_CHARS)
    code: str = ""


def validate(req: JobRequest) -> str | None:
    """Return an error message, or None if the request is valid."""
    if req.format not in config.FORMATS:
        return "Unknown format."
    f = config.FORMATS[req.format]
    if req.seconds not in f["lengths"]:
        return "Unsupported length for this format."
    lo, hi = f["scenes"]
    if not lo <= req.scenes <= hi:
        return f"Choose between {lo} and {hi} scenes for this format."
    if req.motion > min(req.scenes, f["max_heroes"]):
        return "Too many animated shots for the number of scenes."
    checks = [(req.style, config.STYLES, "style"), (req.voice, config.VOICES, "voice"), (req.tone, config.TONES, "tone"),
              (req.audience, config.AUDIENCES, "audience"), (req.pacing, config.PACING, "pacing"),
              (req.music, config.MUSIC, "music"), (req.render, config.RENDER_MODES, "render mode")]
    if req.captions:
        checks.append((req.captions, config.CAPTIONS, "caption style"))
    for value, allowed, name in checks:
        if value not in allowed:
            return f"Unknown {name}."
    return None


def create_app(store, spawn: Callable[[str, dict], None], jobs_root: Path,
               refresh_files: Callable[[], None] = lambda: None) -> FastAPI:
    """store: dict-like (modal.Dict); spawn(job_id, params) starts the pipeline in the background."""
    app = FastAPI(title="DocuGen AI")

    def today_key() -> str:
        return "count:" + datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @app.get("/api/config")
    def get_config() -> dict:
        used = store.get(today_key(), 0) or 0
        formats = {k: {"label": f["label"], "aspect": f["aspect"], "hint": f["hint"], "lengths": f["lengths"],
                       "default_seconds": f["default_seconds"], "scenes": {"min": f["scenes"][0], "max": f["scenes"][1]},
                       "default_scenes": {s: config.default_scenes(s, k) for s in f["lengths"]},
                       "max_heroes": f["max_heroes"], "captions": f["captions"]} for k, f in config.FORMATS.items()}
        return {
            "formats": formats, "default_format": config.DEFAULT_FORMAT,
            "styles": {k: v["label"] for k, v in config.STYLES.items()},
            "voices": config.VOICES, "default_voice": config.DEFAULT_VOICE,
            "tones": {k: v[0] for k, v in config.TONES.items()},
            "audiences": {k: v[0] for k, v in config.AUDIENCES.items()},
            "pacing": {k: v[0] for k, v in config.PACING.items()},
            "captions": config.CAPTIONS, "music": config.MUSIC,
            "render": {k: {"label": v["label"], "hint": v["hint"]} for k, v in config.RENDER_MODES.items()},
            "max_key_points": config.MAX_KEY_POINTS_CHARS,
            "access_required": bool(config.ACCESS_CODE),
            "daily_limit": config.DAILY_LIMIT,
            "used_today": used,
        }

    @app.post("/api/estimate")
    def estimate(req: JobRequest) -> dict:
        return cost.estimate(req.model_dump(exclude={"code"}))

    @app.post("/api/jobs")
    def create_job(req: JobRequest) -> dict:
        if config.ACCESS_CODE and not hmac.compare_digest(req.code.strip().encode(), config.ACCESS_CODE.encode()):
            raise HTTPException(403, "Wrong access code. Ask the owner of this demo for the code.")
        problem = validate(req)
        if problem:
            raise HTTPException(400, problem)
        key = today_key()
        used = store.get(key, 0) or 0
        if used >= config.DAILY_LIMIT:
            raise HTTPException(429, f"Today's limit of {config.DAILY_LIMIT} films is used up. "
                                     "Please try again tomorrow (UTC).")
        store[key] = used + 1
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        params = req.model_dump(exclude={"code"})
        params["topic"] = " ".join(params["topic"].split())
        params["key_points"] = " ".join(params["key_points"].split())
        params["captions"] = params["captions"] or config.FORMATS[req.format]["captions"]
        store[job_id] = new_state(job_id, params)
        spawn(job_id, params)
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        state = store.get(job_id)
        if not state or not isinstance(state, dict):
            raise HTTPException(404, "Job not found")
        return state

    @app.get("/api/recent")
    def recent() -> list[dict]:
        out = []
        for jid in (store.get("recent", []) or [])[:12]:
            s = store.get(jid)
            if s and s.get("status") == "done":
                out.append({"id": jid, "title": s.get("title"), "logline": s.get("logline"),
                            "duration": s.get("duration"), "created_at": s.get("created_at"),
                            "format": s.get("format") or "long", "cost": (s.get("cost") or {}).get("usd")})
        return out

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str, download: bool = False):
        if ".." in path or "\\" in path or "/" in job_id or ".." in job_id:
            raise HTTPException(400, "bad path")
        p = jobs_root / job_id / path
        if p.suffix.lower() not in ALLOWED_EXT:
            raise HTTPException(404, "not allowed")
        if not p.exists():
            refresh_files()
        if not p.exists():
            raise HTTPException(404, "file not ready")
        return FileResponse(p, filename=p.name if download else None,
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "time": time.time()}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def remember_recent(store, job_id: str) -> None:
    ids = [j for j in (store.get("recent", []) or []) if j != job_id]
    store["recent"] = [job_id] + ids[:30]
