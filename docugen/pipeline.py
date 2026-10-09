"""The documentary pipeline. Each stage starts automatically when the previous one finishes.

    script   -> Groq LLM (GPT-OSS 120B) writes title, logline and scenes (JSON, validated), then fact-checks itself
    voice    -> Kokoro-82M narrates every scene on a small GPU, with word timestamps
    direct   -> the Visual Director agent writes the image and motion prompts (runs while the voice is recorded)
    plan     -> each scene is cut into short shots; animated "hero" scenes open with a 5 s video clip
    images   -> Z-Image-Turbo paints every shot (8 steps, GPUs in parallel)
    motion   -> Wan 2.2 Lightning animates each hero still as soon as it is painted
    captions -> word-level captions straight from the TTS timestamps (clean subtitles or word-by-word "pop")
    assemble -> FFmpeg: eased Ken Burns moves, cross-dissolves, cards, captions, ducked music

Two formats share the whole pipeline: 16:9 long videos (1-5 min) and 9:16 reels (15-60 s).
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

from . import cards, config, cost, director, media
from .captions import build_cues, words_from_script, write_ass, write_srt
from .music import make_music
from .script import to_markdown

STAGES = [
    ("script", "Writing & fact-checking the script", 10),
    ("voice", "Recording narration (Kokoro TTS)", 6),
    ("direct", "Visual Director agent designing shots", 8),
    ("images", "Painting the shots (Z-Image Turbo)", 30),
    ("motion", "Animating hero shots (Wan 2.2)", 30),
    ("captions", "Timing captions", 2),
    ("assemble", "Editing the film (FFmpeg)", 24),
]


class Handle(Protocol):
    def result(self) -> bytes: ...


class Backend(Protocol):
    def prewarm(self, images: int, motions: int, render: str) -> None: ...
    def write_script(self, brief: dict, log: Callable) -> dict: ...
    def fact_check(self, script: dict, topic: str, log: Callable) -> list[str]: ...
    def direct(self, script: dict, counts: list[int], brief: dict, log: Callable) -> dict: ...
    def narrate(self, texts: list[str], voice: str, speed: float) -> Iterable[tuple[bytes, list[dict]]]: ...
    def images(self, prompts: list[str], seeds: list[int], size: tuple[int, int], render: str) -> Iterable[bytes]: ...
    def start_animate(self, image: bytes, prompt: str, seed: int, size: tuple[int, int], render: str) -> Handle: ...
    def usage(self) -> dict: ...


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
        "format": params.get("format", config.DEFAULT_FORMAT), "estimate": cost.estimate(params), "cost": None,
        "fact_check": None, "director": None,
    }


def normalize_params(params: dict) -> dict:
    """Fill defaults and accept older parameter names (music/subtitles booleans)."""
    p = dict(params)
    f = config.fmt(p.get("format"))
    p["format"] = p.get("format") if p.get("format") in config.FORMATS else config.DEFAULT_FORMAT
    p["seconds"] = int(p.get("seconds") or f["default_seconds"])
    p.setdefault("style", "cinematic")
    p.setdefault("voice", config.DEFAULT_VOICE)
    p.setdefault("tone", "informative")
    p.setdefault("audience", "general")
    p.setdefault("pacing", "balanced")
    p.setdefault("render", "economy")
    p.setdefault("fact_check", True)
    p.setdefault("key_points", "")
    music = p.get("music", "ambient")
    p["music"] = "ambient" if music is True else "none" if music in (False, None) else music
    if "captions" not in p:
        p["captions"] = f["captions"] if p.get("subtitles", True) else "none"
    return p


def run(params: dict, root: Path, backend: Backend, status: Status) -> bool:
    """Run one job end to end. Returns True on success; errors are captured in the status."""
    st = status.state
    for d in ("work", "scenes", "output"):
        (root / d).mkdir(parents=True, exist_ok=True)
    st["status"] = "running"
    status.save(force=True)
    t_start = time.time()
    ok = False
    try:
        _run(normalize_params(params), root, backend, status)
        st["status"] = "done"
        st["stage"] = None
        st["progress"] = 1.0
        ok = True
    except Exception as exc:
        stage = st.get("stage")
        if stage:
            st["stages"][stage]["status"] = "error"
        st["status"] = "error"
        st["error"] = f"{type(exc).__name__}: {exc}"
        status.log(f"Failed during '{stage}': {exc}")
        print(traceback.format_exc(), flush=True)
    st["seconds"] = round(time.time() - t_start, 1)
    try:
        st["cost"] = cost.bill(backend.usage() if hasattr(backend, "usage") else {}, st["seconds"])
    except Exception as exc:
        print("cost accounting skipped:", exc)
    if ok:
        c = st.get("cost") or {}
        status.log(f"Your film is ready: {st['seconds']:.0f} s from start to finish"
                   + (f", about ${c['usd']:.2f} of compute" if c.get("usd") else ""))
    status.publish()
    status.save(force=True)
    return ok


# --------------------------------------------------------------------------- shot planning
def plan_scene(seconds: float, hero: bool, shot_len: float = 3.2) -> list[dict]:
    """Cut one scene into shots. A hero scene opens with the generated video clip, the rest are short stills."""
    shots, rem = [], seconds
    if hero:
        m = config.MOTION_SECONDS
        if rem - m < 1.6:          # not enough left for a still: let the clip carry the whole scene
            m = rem
        shots.append({"kind": "motion", "seconds": m})
        rem -= m
    if rem > 0.05:
        k = max(1, round(rem / shot_len))
        k = min(k, config.MAX_SHOTS_PER_SCENE - len(shots))
        shots += [{"kind": "still", "seconds": rem / k} for _ in range(k)]
    return shots


def plan_parameters(params: dict) -> tuple[int, int, int]:
    f = config.fmt(params.get("format"))
    seconds = int(params.get("seconds") or f["default_seconds"])
    lo, hi = f["scenes"]
    scenes = int(params.get("scenes") or config.default_scenes(seconds, params.get("format")))
    scenes = max(lo, min(hi, scenes))
    heroes = max(0, min(int(params.get("motion", 0)), f["max_heroes"], scenes))
    return seconds, scenes, heroes


def _run(params, root: Path, backend: Backend, status: Status) -> None:
    st = status.state
    work, scenes_dir, out = root / "work", root / "scenes", root / "output"
    fmt_name = params["format"]
    f = config.fmt(fmt_name)
    W, H = f["size"]
    seconds, n_scenes, heroes = plan_parameters(params)
    shot_len = config.shot_seconds(fmt_name, params["pacing"])
    style = config.STYLES.get(params["style"], config.STYLES["cinematic"])["prompt"]
    render_mode = params["render"] if params["render"] in config.RENDER_MODES else "economy"
    try:  # start GPU containers now so model loading overlaps with writing and narration
        backend.prewarm(max(1, round(seconds / shot_len)), heroes, render_mode)
    except Exception as exc:
        status.log(f"GPU pre-warm skipped: {exc}")

    # ------------------------------------------------------------------ 1. script + fact-check
    status.stage("script", "running")
    status.log(f"Writing a {seconds}s {f['label'].lower()} in {n_scenes} scenes ({heroes} animated) about: "
               f"{params['topic']}")
    brief = {"topic": params["topic"], "seconds": seconds, "scenes": n_scenes, "heroes": heroes, "format": fmt_name,
             "tone": params["tone"], "audience": params["audience"], "key_points": params["key_points"],
             "style": params["style"], "pacing": params["pacing"]}
    script = backend.write_script(brief, status.log)
    if params.get("fact_check", True):
        status.stage("script", "running", 0.6)
        st["fact_check"] = backend.fact_check(script, params["topic"], status.log)
    scenes = script["scenes"]
    n = len(scenes)
    st["title"], st["logline"] = script["title"], script["logline"]
    st["scenes"] = [{"id": s["id"], "narration": s["narration"], "visual": s["visual"], "hero": s["hero"],
                     "image": None, "motion": None, "shots": [], "prompts": []} for s in scenes]
    status.stage("script", "done")
    status.publish()

    # ------------------------------------------------------------------ 2. narration || Visual Director
    counts = director.plan_counts(script, fmt_name, params["pacing"], plan_scene)
    plan_box: dict = {}

    def run_director() -> None:
        try:
            plan_box["plan"] = backend.direct(script, counts, brief, status.log)
        except Exception as exc:  # fall back to the script's own shot ideas
            status.log(f"Visual Director failed ({str(exc)[:140]}) - using the script's shot ideas")
            plan_box["plan"] = None
        status.stage("direct", "done")

    status.stage("direct", "running", 0.1)
    director_thread = threading.Thread(target=run_director, daemon=True)
    director_thread.start()

    status.stage("voice", "running")
    target = config.narration_seconds(seconds, fmt_name)
    speed = config.SPEECH_SPEED * (1.06 if fmt_name == "reel" else 1.0)
    for attempt in range(2):
        padded, scene_words = [], []
        for i, (wav, words) in enumerate(backend.narrate([s["narration"] for s in scenes], params["voice"], speed)):
            raw = scenes_dir / f"scene_{i:02d}.wav"
            raw.write_bytes(wav)
            dst = work / f"narr_{i:02d}.wav"
            padded.append((dst, media.pad_narration(raw, dst, tail=0.2 if fmt_name == "reel" else 0.28)))
            scene_words.append(words)
            status.stage("voice", "running", (i + 1) / n)
        narration_len = sum(d for _, d in padded)
        status.log(f"Narration: {narration_len:.1f} s for {n} scenes at speed {speed:.2f} (target {target:.0f} s)")
        if attempt or narration_len <= target * 1.12 or speed >= config.MAX_SPEECH_SPEED:
            break
        speed = min(config.MAX_SPEECH_SPEED, speed * narration_len / target)   # keep the film to its length
    status.stage("voice", "done")
    director_thread.join(timeout=240)
    visual = plan_box.get("plan")
    if visual:
        st["director"] = {"bible": visual.get("bible"), "stats": visual.get("stats")}
    if st["stages"]["direct"]["status"] != "done":
        status.stage("direct", "done")

    # ------------------------------------------------------------------ 3. shot plan
    jobs: list[dict] = []          # image generation jobs
    timeline: list[dict] = []      # shots in screen order (scenes only; cards are added at assembly)
    motion_prompts: dict[int, str] = {}
    for i, s in enumerate(scenes):
        shots = plan_scene(padded[i][1], s["hero"], shot_len)
        vp = visual["scenes"][i] if visual and i < len(visual["scenes"]) else None
        prompts = (vp["shots"] if vp and vp["shots"] else None) or s["shots"]
        motion_prompts[i] = (vp["motion"] if vp and vp.get("motion") else "") or s["motion"]
        for j, shot in enumerate(shots):
            text = prompts[j % len(prompts)]
            job = {"scene": i, "shot": j, "prompt": f"{text}, {f['framing']}, {style}, {config.CLEAN_SUFFIX}",
                   "seed": 1000 + 97 * i + 13 * j, "motion": shot["kind"] == "motion",
                   "file": scenes_dir / f"shot_{i:02d}_{j}.jpg"}
            jobs.append(job)
            st["scenes"][i]["prompts"].append(text)
            timeline.append({**shot, "job": job, "move": (i * 3 + j) % 7})
    status.log(f"Shot plan: {len(timeline)} shots of about {shot_len:.1f} s, {sum(j['motion'] for j in jobs)} animated")
    status.publish()

    # ------------------------------------------------------------------ 4+5. images, motion starts as heroes land
    status.stage("images", "running")
    if heroes:
        status.stage("motion", "running", 0.0)
    order = sorted(jobs, key=lambda j: (not j["motion"], j["scene"], j["shot"]))  # hero stills first
    handles: dict[int, Handle] = {}
    last_good: bytes | None = None
    results = backend.images([j["prompt"] for j in order], [j["seed"] for j in order], f["image"], render_mode)
    for k, (job, img) in enumerate(zip(order, results)):
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
            i = job["scene"]
            still = " ".join(st["scenes"][i]["prompts"][0].split()[:45])
            handles[i] = backend.start_animate(img, f"{motion_prompts[i]}. {still}", 42 + i, f["video"], render_mode)
            status.log(f"Animating scene {i + 1}")
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
    title_len = media.frames(f["title_card"]) if f["title_card"] else 0.0
    end_len = media.frames(f["end_card"])
    words_all, spans, t = [], [], title_len
    for i, (_, d) in enumerate(padded):
        lead = 0.12
        words_all += [{"w": w["w"], "s": round(t + lead + w["s"], 3), "e": round(t + lead + w["e"], 3)}
                      for w in scene_words[i] if w.get("w")]
        spans.append((t + lead, t + d - 0.3, scenes[i]["narration"]))
        t += d
    total = t + end_len
    if len(words_all) < 0.5 * sum(len(s[2].split()) for s in spans):
        status.log("TTS timestamps incomplete - spreading caption words evenly")
        words_all = words_from_script(spans)
    cap_style = params["captions"] if params["captions"] in config.CAPTIONS else f["captions"]
    pop_words = 3 if fmt_name == "reel" else 4
    cues = build_cues([words_all], max_words=pop_words if cap_style == "pop" else 6,
                      max_seconds=1.6 if cap_style == "pop" else 2.4)
    write_srt(cues, out / "subtitles.srt")
    fonts_dir = work / "fonts"
    fonts_dir.mkdir(exist_ok=True)
    for weight in ("Bold", "Regular"):
        fp = cards.font_path(weight)
        if fp:
            shutil.copy2(fp, fonts_dir / Path(fp).name)
    hook = (script["title"], 0.15, min(3.0, total - end_len)) if not title_len else None
    burn = cap_style != "none" and bool(cues)
    write_ass(cues if burn else [], work / "subs.ass", W, H, cards.font_family(), 50, style=cap_style, hook=hook)
    status.stage("captions", "done")

    # ------------------------------------------------------------------ 7. assemble
    status.stage("assemble", "running")
    sil = []
    if title_len:
        media.silence(work / "sil_title.wav", title_len)
        sil.append(work / "sil_title.wav")
    media.silence(work / "sil_end.wav", end_len)
    audio_list = media.concat_list(sil + [p for p, _ in padded] + [work / "sil_end.wav"], work / "audio.txt")
    media.run_concat_audio(audio_list, out / "narration.wav")

    first_img = jobs[0]["file"].read_bytes()
    last_img = jobs[-1]["file"].read_bytes()
    shots = []
    if title_len:
        cards.title_card(work / "title.png", W, H, script["title"], script["logline"], first_img)
        shots.append({"kind": "card", "seconds": title_len, "image": work / "title.png", "move": -1})
    cards.end_card(work / "end.png", W, H, last_img)
    for sh in timeline:
        i = sh["job"]["scene"]
        if sh["kind"] == "motion" and i in motion_files:
            shots.append({"kind": "motion", "seconds": sh["seconds"], "clip": motion_files[i],
                          "image": sh["job"]["file"]})
        else:
            shots.append({"kind": "still", "seconds": sh["seconds"], "image": sh["job"]["file"], "move": sh["move"]})
    shots.append({"kind": "card", "seconds": end_len, "image": work / "end.png", "move": -1})

    x = config.XFADE
    paths = [work / f"shot_{k:03d}.mp4" for k in range(len(shots))]
    done = [0]
    lock = threading.Lock()

    def render(k: int) -> None:
        sh = shots[k]
        length = sh["seconds"] + (x if k < len(shots) - 1 else 0)
        if sh["kind"] == "motion":
            try:
                media.motion_clip(sh["clip"], paths[k], length, f["size"])
            except Exception as exc:
                status.log(f"shot {k}: motion clip failed ({exc}); using a still")
                media.ken_burns(sh["image"], paths[k], length, k, f["size"], f["kb"])
        else:
            media.ken_burns(sh["image"], paths[k], length, sh["move"], f["size"], f["kb"])
        with lock:
            done[0] += 1
            status.stage("assemble", "running", 0.5 * done[0] / len(shots))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(render, range(len(shots))))
    status.log(f"Rendered {len(shots)} shots - final cut with dissolves, captions and music")
    music = None
    if params["music"] in config.MUSIC and params["music"] != "none":
        try:
            music = make_music(work / f"music_{params['music']}.mp3", params["music"])
        except Exception as exc:
            status.log(f"Music skipped: {exc}")
    swell = [(a, b) for a, b in ((0.0, title_len), (total - end_len, total)) if b - a > 0.05]
    media.final_mix(work, [(p, sh["seconds"]) for p, sh in zip(paths, shots)], x, audio_list, music, total, swell,
                    work / "subs.ass" if burn or hook else None,
                    out / "documentary.mp4", out / "documentary_audio.wav",
                    on_progress=lambda fr: status.stage("assemble", "running", 0.5 + 0.48 * fr))
    cards.thumbnail(first_img, out / "thumbnail.jpg", script["title"], (720, 1280) if H > W else (1280, 720))
    script_out = {**script, "format": fmt_name, "fact_check": st.get("fact_check"),
                  "visual_bible": (visual or {}).get("bible"),
                  "shot_prompts": [sc["prompts"] for sc in st["scenes"]]}
    (out / "script.json").write_text(json.dumps(script_out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = {**script, "scenes": [{**sc, "shots": st["scenes"][i]["prompts"] or sc["shots"],
                                "motion": motion_prompts.get(i) or sc["motion"]} for i, sc in enumerate(scenes)]}
    (out / "script.md").write_text(to_markdown(md), encoding="utf-8")
    st["duration"] = round(total, 1)
    st["outputs"] = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    status.stage("assemble", "done")
