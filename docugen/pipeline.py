"""The documentary pipeline. Each stage starts automatically when the previous one finishes.

    script  -> Groq LLM (GPT-OSS 120B) writes title, logline and scenes (JSON, validated)
    voice   -> XTTS-v2 narrates every scene            } run in parallel
    images  -> Juggernaut XL paints every scene        }
    motion  -> LTX-Video animates the 'hero' scenes (falls back to Ken Burns on failure)
    captions-> faster-whisper aligns words on the narration
    assemble-> FFmpeg: Ken Burns / motion clips, title & end cards, subtitles, ducked music

The GPU work goes through a `Backend`, so the same pipeline runs on Modal or with a fake backend in tests.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Protocol

from . import cards, config, media
from .captions import build_cues, transcribe_words, words_from_script, write_ass, write_srt
from .music import make_music
from .script import to_markdown

STAGES = [
    ("script", "Writing the script (Groq LLM)", 6),
    ("voice", "Recording narration (XTTS-v2)", 14),
    ("images", "Painting the scenes (Juggernaut XL)", 25),
    ("motion", "Animating hero shots (LTX-Video)", 25),
    ("captions", "Aligning captions (faster-whisper)", 8),
    ("assemble", "Editing the film (FFmpeg)", 22),
]


class Backend(Protocol):
    def write_script(self, topic: str, seconds: int, heroes: int, log: Callable) -> dict: ...
    def voice(self, texts: list[str], speaker: str) -> Iterable[bytes]: ...
    def images(self, prompts: list[str], seeds: list[int]) -> Iterable[bytes]: ...
    def animate(self, items: list[tuple[bytes, str, int]]) -> Iterable[bytes | Exception]: ...


class Status:
    """Live job state, written to a dict-like store (a modal.Dict in production)."""

    def __init__(self, store, job_id: str, state: dict, publish: Callable[[], None] = lambda: None):
        self.store, self.id, self.state, self.publish = store, job_id, state, publish
        self._last = 0.0
        self._lock = threading.Lock()

    def save(self, force: bool = False) -> None:
        with self._lock:
            if force or time.time() - self._last > 1.0:
                self.store[self.id] = self.state
                self._last = time.time()

    def log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(f"[{self.id}] {msg}", flush=True)
        self.state.setdefault("log", []).append(line)
        self.state["log"] = self.state["log"][-60:]
        self.state["message"] = msg
        self.save()

    def stage(self, key: str, status: str, progress: float | None = None) -> None:
        st = self.state["stages"][key]
        if status == "running" and st["status"] != "running":
            st["started"] = time.time()
            self.state["stage"] = key
        if status in ("done", "skipped", "error") and st.get("started"):
            st["seconds"] = round(time.time() - st["started"], 1)
        st["status"] = status
        if progress is not None:
            st["progress"] = round(progress, 3)
        if status in ("done", "skipped"):
            st["progress"] = 1.0
        total = sum(w for _, _, w in STAGES)
        done = sum(w * (1.0 if self.state["stages"][k]["status"] in ("done", "skipped")
                        else self.state["stages"][k].get("progress", 0) if self.state["stages"][k]["status"] == "running"
                        else 0) for k, _, w in STAGES)
        self.state["progress"] = round(done / total, 3)
        self.save(force=status != "running" or progress is None)


def new_state(job_id: str, params: dict) -> dict:
    return {
        "id": job_id, "created_at": time.time(), "status": "queued", "params": params,
        "stage": None, "progress": 0.0, "message": "Waiting for a worker", "error": None,
        "stages": {k: {"status": "pending", "progress": 0.0, "seconds": None, "label": label} for k, label, _ in STAGES},
        "title": None, "logline": None, "scenes": [], "outputs": [], "log": [],
    }


def run(params: dict, root: Path, backend: Backend, status: Status) -> bool:
    """Run one job end to end. Returns True on success; errors are captured in the status."""
    st = status.state
    work = root / "work"
    scenes_dir = root / "scenes"
    out = root / "output"
    for d in (work, scenes_dir, out):
        d.mkdir(parents=True, exist_ok=True)
    st["status"] = "running"
    status.save(force=True)
    t_start = time.time()
    try:
        _run(params, root, work, scenes_dir, out, backend, status)
        st["status"] = "done"
        st["stage"] = None
        st["progress"] = 1.0
        st["seconds"] = round(time.time() - t_start, 1)
        status.log("Your documentary is ready")
        status.publish()
        status.save(force=True)
        return True
    except Exception as exc:
        stage = st.get("stage")
        if stage:
            st["stages"][stage]["status"] = "error"
        st["status"] = "error"
        st["error"] = f"{type(exc).__name__}: {exc}"
        status.log(f"Failed during '{stage}': {exc}")
        print(traceback.format_exc(), flush=True)
        status.publish()
        status.save(force=True)
        return False


def _run(params, root: Path, work: Path, scenes_dir: Path, out: Path, backend: Backend, status: Status) -> None:
    st = status.state
    seconds, heroes = int(params["seconds"]), int(params["motion"])
    style = config.STYLES[params["style"]]["prompt"]

    # ------------------------------------------------------------------ 1. script
    status.stage("script", "running")
    status.log(f"Writing a {seconds}s documentary about: {params['topic']}")
    script = backend.write_script(params["topic"], seconds, heroes, status.log)
    scenes = script["scenes"]
    (root / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "script.md").write_text(to_markdown(script), encoding="utf-8")
    st["title"], st["logline"] = script["title"], script["logline"]
    st["scenes"] = [{"id": s["id"], "narration": s["narration"], "visual": s["visual"], "hero": s["hero"],
                     "image": None, "motion": None} for s in scenes]
    status.stage("script", "done")
    status.publish()

    # ------------------------------------------------------------------ 2+3. voice and images in parallel
    status.stage("voice", "running")
    status.stage("images", "running")
    n = len(scenes)
    voice_err: list[BaseException] = []

    def do_voice() -> None:
        try:
            for i, wav in enumerate(backend.voice([s["narration"] for s in scenes], params["voice"])):
                (scenes_dir / f"scene_{i:02d}.wav").write_bytes(wav)
                status.stage("voice", "running", (i + 1) / n)
            status.stage("voice", "done")
        except BaseException as exc:  # surfaced after images finish
            voice_err.append(exc)

    vt = threading.Thread(target=do_voice, daemon=True)
    vt.start()
    prompts = [f"{s['visual']}, {style}" for s in scenes]
    seeds = [1000 + 17 * i for i in range(n)]
    for i, png in enumerate(backend.images(prompts, seeds)):
        (scenes_dir / f"scene_{i:02d}.png").write_bytes(png)
        st["scenes"][i]["image"] = f"scenes/scene_{i:02d}.png"
        status.stage("images", "running", (i + 1) / n)
        status.log(f"Painted scene {i + 1}/{n}")
        status.publish()
    status.stage("images", "done")
    vt.join()
    if voice_err:
        status.state["stage"] = "voice"
        raise voice_err[0]

    # ------------------------------------------------------------------ 4. motion
    hero_ids = [s["id"] for s in scenes if s["hero"]]
    motion_paths: dict[int, Path] = {}
    if hero_ids:
        status.stage("motion", "running")
        items = [((scenes_dir / f"scene_{i:02d}.png").read_bytes(), f"{scenes[i]['motion']}, {scenes[i]['visual']}",
                  42 + i) for i in hero_ids]
        for k, (i, result) in enumerate(zip(hero_ids, backend.animate(items))):
            if isinstance(result, (bytes, bytearray)) and len(result) > 1000:
                p = scenes_dir / f"motion_{i:02d}.mp4"
                p.write_bytes(result)
                motion_paths[i] = p
                st["scenes"][i]["motion"] = f"scenes/motion_{i:02d}.mp4"
                status.log(f"Animated scene {i + 1}")
            else:
                status.log(f"Scene {i + 1}: animation failed ({str(result)[:120]}) - using Ken Burns instead")
            status.stage("motion", "running", (k + 1) / len(hero_ids))
            status.publish()
        status.stage("motion", "done")
    else:
        status.stage("motion", "skipped")

    # ------------------------------------------------------------------ 5. narration timeline + captions
    status.stage("captions", "running")
    intro_s, outro_s = media.frames(4.5), media.frames(4.5)
    padded, spans, t = [], [], intro_s
    for i in range(n):
        dst = work / f"narr_{i:02d}.wav"
        d = media.pad_narration(scenes_dir / f"scene_{i:02d}.wav", dst)
        padded.append((dst, d))
        spans.append((t + 0.35, t + d - 0.6, scenes[i]["narration"]))
        t += d
    total = t + outro_s
    media.silence(work / "sil_intro.wav", intro_s)
    media.silence(work / "sil_outro.wav", outro_s)
    audio_list = media.concat_list([work / "sil_intro.wav"] + [p for p, _ in padded] + [work / "sil_outro.wav"],
                                   work / "audio.txt")
    run_ffmpeg_concat_audio(audio_list, out / "narration.wav")
    try:
        words = transcribe_words(out / "narration.wav", config.WHISPER_MODEL, script["title"], status.log)
        if len(words) < 0.5 * sum(len(s[2].split()) for s in spans):
            raise RuntimeError("too few words recognised")
    except Exception as exc:
        status.log(f"Whisper alignment unavailable ({exc}); using script timing")
        words = words_from_script(spans)
    cues = build_cues([words], max_words=7, max_seconds=2.8)
    write_srt(cues, out / "subtitles.srt")
    fonts_dir = work / "fonts"
    fonts_dir.mkdir(exist_ok=True)
    for weight in ("Bold", "Regular"):
        fp = cards.font_path(weight)
        if fp:
            shutil.copy2(fp, fonts_dir / Path(fp).name)
    write_ass(cues, work / "subs.ass", config.WIDTH, config.HEIGHT, cards.font_family(), 50)
    status.stage("captions", "done")

    # ------------------------------------------------------------------ 6. assemble
    status.stage("assemble", "running")
    first_png = (scenes_dir / "scene_00.png").read_bytes()
    last_png = (scenes_dir / f"scene_{n - 1:02d}.png").read_bytes()
    cards.title_card(work / "title.png", config.WIDTH, config.HEIGHT, script["title"], script["logline"], first_png)
    cards.end_card(work / "end.png", config.WIDTH, config.HEIGHT, last_png)

    jobs = [("title", work / "title.png", None, intro_s, 0)]
    for i, (_, d) in enumerate(padded):
        jobs.append((f"scene{i}", scenes_dir / f"scene_{i:02d}.png", motion_paths.get(i), d, i))
    jobs.append(("end", work / "end.png", None, outro_s, 1))
    clip_paths = [work / f"clip_{k:02d}.mp4" for k in range(len(jobs))]
    done = [0]

    def render(k: int) -> None:
        name, img, motion, d, move = jobs[k]
        if motion is not None:
            try:
                media.motion_clip(motion, clip_paths[k], d)
            except Exception as exc:
                status.log(f"{name}: motion clip failed ({exc}); using still")
                media.ken_burns(img, clip_paths[k], d, move)
        else:
            media.ken_burns(img, clip_paths[k], d, move if name.startswith("scene") else 0)
        done[0] += 1
        status.stage("assemble", "running", 0.6 * done[0] / len(jobs))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(render, range(len(jobs))))
    status.log("All shots rendered - final mix")
    video_list = media.concat_list(clip_paths, work / "video.txt")
    music = None
    if params.get("music", True):
        try:
            music = make_music(work / "music.mp3")
        except Exception as exc:
            status.log(f"Music skipped: {exc}")
    media.final_mix(work, video_list, audio_list, music, total, [(0, intro_s), (total - outro_s, total)],
                    work / "subs.ass" if params.get("subtitles", True) and cues else None,
                    out / "documentary.mp4", out / "documentary_audio.wav",
                    on_progress=lambda f: status.stage("assemble", "running", 0.6 + 0.38 * f))
    cards.thumbnail(first_png, out / "thumbnail.jpg", script["title"])
    shutil.copy2(root / "script.md", out / "script.md")
    shutil.copy2(root / "script.json", out / "script.json")
    st["duration"] = round(total, 1)
    st["outputs"] = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    status.stage("assemble", "done")


def run_ffmpeg_concat_audio(audio_list: Path, dst: Path) -> None:
    from .ffmpeg import run as ff
    ff(["-f", "concat", "-safe", "0", "-i", str(audio_list), "-c:a", "pcm_s16le", str(dst)])
