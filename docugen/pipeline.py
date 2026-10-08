"""The documentary pipeline. Each stage starts automatically when the previous one finishes.

    script   -> Groq LLM (GPT-OSS 120B) writes title, logline and scenes with 4 shot ideas each (JSON, validated)
    voice    -> Kokoro-82M narrates every scene on CPU, with word timestamps (seconds, not minutes)
    plan     -> each scene is cut into ~3 s shots; animated "hero" scenes open with a 5 s video clip
    images   -> Z-Image-Turbo paints every shot (8 steps, several GPUs in parallel)
    motion   -> Wan 2.2 Lightning animates each hero still as soon as it is painted (parallel GPUs)
    captions -> word-level captions straight from the TTS timestamps
    assemble -> FFmpeg: eased Ken Burns moves, cross-dissolves, title/end cards, captions, ducked music

GPU work goes through a `Backend`, so the same pipeline runs on Modal or with a fake backend in tests.
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
from .captions import build_cues, words_from_script, write_ass, write_srt
from .music import make_music
from .script import to_markdown

STAGES = [
    ("script", "Writing the script (GPT-OSS 120B)", 8),
    ("voice", "Recording narration (Kokoro TTS)", 6),
    ("images", "Painting the shots (Z-Image Turbo)", 30),
    ("motion", "Animating hero shots (Wan 2.2)", 30),
    ("captions", "Timing captions", 2),
    ("assemble", "Editing the film (FFmpeg)", 24),
]


class Handle(Protocol):
    def result(self) -> bytes: ...


class Backend(Protocol):
    def prewarm(self, images: int, motions: int) -> None: ...
    def write_script(self, topic: str, seconds: int, heroes: int, log: Callable, scenes: int) -> dict: ...
    def narrate(self, texts: list[str], voice: str, speed: float) -> Iterable[tuple[bytes, list[dict]]]: ...
    def images(self, prompts: list[str], seeds: list[int]) -> Iterable[bytes]: ...
    def start_animate(self, image: bytes, prompt: str, seed: int) -> Handle: ...


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
        self.state["log"] = self.state["log"][-80:]
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
    for d in ("work", "scenes", "output"):
        (root / d).mkdir(parents=True, exist_ok=True)
    st["status"] = "running"
    status.save(force=True)
    t_start = time.time()
    try:
        _run(params, root, backend, status)
        st["status"] = "done"
        st["stage"] = None
        st["progress"] = 1.0
        st["seconds"] = round(time.time() - t_start, 1)
        status.log(f"Your documentary is ready ({st['seconds']:.0f} s from start to finish)")
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


# --------------------------------------------------------------------------- shot planning
def plan_scene(seconds: float, hero: bool) -> list[dict]:
    """Cut one scene into shots. A hero scene opens with the generated video clip, the rest are ~3 s stills."""
    shots, rem = [], seconds
    if hero:
        m = config.MOTION_SECONDS
        if rem - m < 1.8:          # not enough left for a still: let the clip carry the whole scene
            m = rem
        shots.append({"kind": "motion", "seconds": m})
        rem -= m
    if rem > 0.05:
        k = max(1, round(rem / config.SHOT_SECONDS))
        k = min(k, config.MAX_SHOTS_PER_SCENE - len(shots))
        shots += [{"kind": "still", "seconds": rem / k} for _ in range(k)]
    return shots


def plan_parameters(params: dict) -> tuple[int, int, int]:
    seconds = int(params["seconds"])
    scenes = int(params.get("scenes") or config.default_scenes(seconds))
    scenes = max(config.MIN_SCENES, min(config.MAX_SCENES, scenes))
    heroes = max(0, min(int(params.get("motion", 0)), config.MAX_HEROES, scenes))
    return seconds, scenes, heroes


def _run(params, root: Path, backend: Backend, status: Status) -> None:
    st = status.state
    work, scenes_dir, out = root / "work", root / "scenes", root / "output"
    seconds, n_scenes, heroes = plan_parameters(params)
    style = config.STYLES[params["style"]]["prompt"]
    try:  # start GPU containers now so model loading overlaps with writing and narration
        backend.prewarm(max(1, round(seconds / config.SHOT_SECONDS)), heroes)
    except Exception as exc:
        status.log(f"GPU pre-warm skipped: {exc}")

    # ------------------------------------------------------------------ 1. script
    status.stage("script", "running")
    status.log(f"Writing a {seconds}s documentary in {n_scenes} scenes ({heroes} animated) about: {params['topic']}")
    script = backend.write_script(params["topic"], seconds, heroes, status.log, n_scenes)
    scenes = script["scenes"]
    n = len(scenes)
    (root / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "script.md").write_text(to_markdown(script), encoding="utf-8")
    st["title"], st["logline"] = script["title"], script["logline"]
    st["scenes"] = [{"id": s["id"], "narration": s["narration"], "visual": s["visual"], "hero": s["hero"],
                     "image": None, "motion": None, "shots": []} for s in scenes]
    status.stage("script", "done")
    status.publish()

    # ------------------------------------------------------------------ 2. narration (CPU, fast)
    status.stage("voice", "running")
    target = max(10.0, seconds - 2 * config.CARD_SECONDS)
    speed = config.SPEECH_SPEED
    for attempt in range(2):
        padded, scene_words = [], []
        for i, (wav, words) in enumerate(backend.narrate([s["narration"] for s in scenes], params["voice"], speed)):
            raw = scenes_dir / f"scene_{i:02d}.wav"
            raw.write_bytes(wav)
            dst = work / f"narr_{i:02d}.wav"
            padded.append((dst, media.pad_narration(raw, dst)))
            scene_words.append(words)
            status.stage("voice", "running", (i + 1) / n)
        narration_len = sum(d for _, d in padded)
        status.log(f"Narration: {narration_len:.1f} s for {n} scenes at speed {speed:.2f} (target {target:.0f} s)")
        if attempt or narration_len <= target * 1.12 or speed >= config.MAX_SPEECH_SPEED:
            break
        speed = min(config.MAX_SPEECH_SPEED, speed * narration_len / target)   # keep the film to its length
    status.stage("voice", "done")

    # ------------------------------------------------------------------ 3. shot plan
    jobs: list[dict] = []          # image generation jobs
    timeline: list[dict] = []      # shots in screen order (scenes only; cards are added at assembly)
    for i, s in enumerate(scenes):
        shots = plan_scene(padded[i][1], s["hero"])
        prompts = s["shots"]
        still_k = 1 if s["hero"] else 0      # hero clip uses prompt 0; stills continue from prompt 1
        for j, shot in enumerate(shots):
            if shot["kind"] == "motion":
                prompt_idx = 0
            else:
                prompt_idx = still_k % len(prompts)
                still_k += 1
            job = {"scene": i, "shot": j, "prompt": f"{prompts[prompt_idx]}, {style}, {config.CLEAN_SUFFIX}", "seed": 1000 + 97 * i + 13 * j,
                   "motion": shot["kind"] == "motion", "file": scenes_dir / f"shot_{i:02d}_{j}.jpg"}
            jobs.append(job)
            timeline.append({**shot, "job": job, "move": (i * 3 + j) % 7})
    status.log(f"Shot plan: {len(timeline)} shots, {sum(j['motion'] for j in jobs)} animated")

    # ------------------------------------------------------------------ 4+5. images, motion starts as heroes land
    status.stage("images", "running")
    if heroes:
        status.stage("motion", "running", 0.0)
    order = sorted(jobs, key=lambda j: (not j["motion"], j["scene"], j["shot"]))  # hero stills first
    handles: dict[int, Handle] = {}
    last_good: bytes | None = None
    for k, (job, img) in enumerate(zip(order, backend.images([j["prompt"] for j in order],
                                                             [j["seed"] for j in order]))):
        if not isinstance(img, (bytes, bytearray)):
            status.log(f"Shot {k + 1}: image failed ({str(img)[:100]}) - reusing the previous picture")
            if last_good is None:
                raise RuntimeError(f"image generation failed: {img}")
            img = last_good
        last_good = img
        job["file"].write_bytes(img)
        rel = f"scenes/{job['file'].name}"
        sc = st["scenes"][job["scene"]]
        sc["shots"].append(rel)
        if job["shot"] == 0:
            sc["image"] = rel
        if job["motion"]:
            s = scenes[job["scene"]]
            handles[job["scene"]] = backend.start_animate(img, f"{s['motion']}. {s['shots'][0]}", 42 + job["scene"])
            status.log(f"Animating scene {job['scene'] + 1}")
        status.stage("images", "running", (k + 1) / len(order))
        if k % 3 == 2 or k == len(order) - 1:
            status.publish()
    status.log(f"Painted {len(order)} shots")
    status.stage("images", "done")

    motion_files: dict[int, Path] = {}
    if handles:
        for k, (i, h) in enumerate(handles.items()):
            try:
                data = h.result()
                if not data or len(data) < 1000:
                    raise RuntimeError("empty clip")
                p = scenes_dir / f"motion_{i:02d}.mp4"
                p.write_bytes(data)
                motion_files[i] = p
                st["scenes"][i]["motion"] = f"scenes/{p.name}"
                status.log(f"Animated scene {i + 1}")
            except Exception as exc:
                status.log(f"Scene {i + 1}: animation failed ({str(exc)[:120]}) - using a Ken Burns move instead")
            status.stage("motion", "running", (k + 1) / len(handles))
            status.publish()
        status.stage("motion", "done")
    else:
        status.stage("motion", "skipped")

    # ------------------------------------------------------------------ 6. captions from TTS timestamps
    status.stage("captions", "running")
    card = media.frames(config.CARD_SECONDS)
    words_all, spans, t = [], [], card
    for i, (_, d) in enumerate(padded):
        lead = 0.12
        words_all += [{"w": w["w"], "s": round(t + lead + w["s"], 3), "e": round(t + lead + w["e"], 3)}
                      for w in scene_words[i] if w.get("w")]
        spans.append((t + lead, t + d - 0.3, scenes[i]["narration"]))
        t += d
    total = t + card
    if len(words_all) < 0.5 * sum(len(s[2].split()) for s in spans):
        status.log("TTS timestamps incomplete - spreading caption words evenly")
        words_all = words_from_script(spans)
    cues = build_cues([words_all], max_words=6, max_seconds=2.4)
    write_srt(cues, out / "subtitles.srt")
    fonts_dir = work / "fonts"
    fonts_dir.mkdir(exist_ok=True)
    for weight in ("Bold", "Regular"):
        fp = cards.font_path(weight)
        if fp:
            shutil.copy2(fp, fonts_dir / Path(fp).name)
    write_ass(cues, work / "subs.ass", config.WIDTH, config.HEIGHT, cards.font_family(), 50)
    status.stage("captions", "done")

    # ------------------------------------------------------------------ 7. assemble
    status.stage("assemble", "running")
    media.silence(work / "sil_card.wav", card)
    audio_list = media.concat_list([work / "sil_card.wav"] + [p for p, _ in padded] + [work / "sil_card.wav"],
                                   work / "audio.txt")
    media.run_concat_audio(audio_list, out / "narration.wav")

    first_img = jobs[0]["file"].read_bytes()
    last_img = jobs[-1]["file"].read_bytes()
    cards.title_card(work / "title.png", config.WIDTH, config.HEIGHT, script["title"], script["logline"], first_img)
    cards.end_card(work / "end.png", config.WIDTH, config.HEIGHT, last_img)
    shots = [{"kind": "card", "seconds": card, "image": work / "title.png", "move": -1}]
    for sh in timeline:
        i = sh["job"]["scene"]
        if sh["kind"] == "motion" and i in motion_files:
            shots.append({"kind": "motion", "seconds": sh["seconds"], "clip": motion_files[i]})
        else:
            shots.append({"kind": "still", "seconds": sh["seconds"], "image": sh["job"]["file"], "move": sh["move"]})
    shots.append({"kind": "card", "seconds": card, "image": work / "end.png", "move": -1})

    x = config.XFADE
    paths = [work / f"shot_{k:03d}.mp4" for k in range(len(shots))]
    done = [0]

    def render(k: int) -> None:
        sh = shots[k]
        length = sh["seconds"] + (x if k < len(shots) - 1 else 0)
        if sh["kind"] == "motion":
            try:
                media.motion_clip(sh["clip"], paths[k], length)
            except Exception as exc:
                status.log(f"shot {k}: motion clip failed ({exc}); using a still")
                media.ken_burns(timeline[k - 1]["job"]["file"], paths[k], length, k)
        else:
            media.ken_burns(sh["image"], paths[k], length, sh["move"])
        done[0] += 1
        status.stage("assemble", "running", 0.5 * done[0] / len(shots))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(render, range(len(shots))))
    status.log(f"Rendered {len(shots)} shots - final cut with dissolves, captions and music")
    music = None
    if params.get("music", True):
        try:
            music = make_music(work / "music.mp3")
        except Exception as exc:
            status.log(f"Music skipped: {exc}")
    media.final_mix(work, [(p, sh["seconds"]) for p, sh in zip(paths, shots)], x, audio_list, music, total,
                    [(0, card), (total - card, total)],
                    work / "subs.ass" if params.get("subtitles", True) and cues else None,
                    out / "documentary.mp4", out / "documentary_audio.wav",
                    on_progress=lambda f: status.stage("assemble", "running", 0.5 + 0.48 * f))
    cards.thumbnail(first_img, out / "thumbnail.jpg", script["title"])
    shutil.copy2(root / "script.md", out / "script.md")
    shutil.copy2(root / "script.json", out / "script.json")
    st["duration"] = round(total, 1)
    st["outputs"] = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    status.stage("assemble", "done")
