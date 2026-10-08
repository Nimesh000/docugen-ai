"""Thin, safe wrappers around the ffmpeg / ffprobe command-line tools."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(
                f"'{tool}' was not found on PATH. Install FFmpeg first "
                "(Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`, "
                "Linux: `sudo apt install ffmpeg`) and restart the terminal."
            )


_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def run(
    args: Sequence[str],
    cwd: Path | None = None,
    total_seconds: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Run ffmpeg, streaming stderr so we can report progress and keep the tail for errors."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", *map(str, args)]
    if log:
        log("$ " + " ".join(_short(a) for a in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail: list[str] = []
    assert proc.stderr is not None
    buf = ""
    while True:
        chunk = proc.stderr.read(256)
        if not chunk:
            break
        buf += chunk
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for line in parts:
            if not line.strip():
                continue
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            if on_progress and total_seconds:
                m = _TIME_RE.search(line)
                if m:
                    secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                    on_progress(max(0.0, min(1.0, secs / total_seconds)))
    proc.wait()
    if proc.returncode != 0:
        raise FFmpegError("ffmpeg failed:\n" + "\n".join(tail[-15:]))


def _short(arg: str, limit: int = 300) -> str:
    arg = str(arg)
    return arg if len(arg) <= limit else arg[:limit] + "...<truncated>"


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {path.name}: {out.stderr.strip()[:500]}")
    return json.loads(out.stdout)


def media_info(path: Path) -> dict:
    data = probe(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0)
    return {
        "duration": duration,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": int(video["width"]) if video else None,
        "height": int(video["height"]) if video else None,
    }
