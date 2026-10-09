"""Stage 1 - the LLM writes the documentary as structured JSON; Python validates and repairs it."""
from __future__ import annotations

import json
import os
import re
import time

from . import config

SYSTEM_PROMPT = """You are an award-winning documentary writer and director.
Write a narrated documentary as JSON. Rules:
- Narration is vivid, confident and factual, with concrete, well-established details: key years, places,
  institutions, inventions, turning points. No vague filler ("a quiet revolution began"). Only state facts you are
  sure of; if unsure of a number, name or date, leave it out rather than guess.
- Scene 1 is a strong hook. Each scene moves the story forward. The last scene closes with a memorable line.
- Each scene has one narration paragraph of plain spoken sentences (no headings, stage directions, emojis or lists).
- "shots" are 3 different photographs that illustrate that scene's narration, in order (wide, medium, close-up).
  Each is ONE concrete image description: subject, action, setting with country/city and era, lighting, camera angle.
  NEVER include screens, signs, posters, papers, maps, charts, writing, numbers, logos, brand names or real people.
- "motion" describes how the FIRST shot comes alive as a 5 second video (what moves + camera move). One sentence.
- Mark exactly the requested number of the most visually dynamic scenes with "hero": true.
Return ONLY this JSON:
{"title": "...", "logline": "one sentence", "scenes": [
  {"narration": "...", "shots": ["...", "...", "..."], "motion": "...", "hero": false}
]}"""

REEL_RULES = """This is a vertical short-form REEL. The first sentence must hook the viewer within 3 seconds
(a surprising fact, a question or a bold claim). Use short punchy sentences, no slow introduction, and end with a
line that makes people want to share it. The title must be at most 6 words."""

FACT_CHECK_PROMPT = """You are a meticulous fact-checker for a documentary channel.
You receive the narration of each scene. Check every factual claim (dates, numbers, names, places, firsts, causes).
- If a claim is wrong, correct it. If it is doubtful or unverifiable, soften it or remove that detail.
- Keep the storytelling voice, the order and roughly the same length of every scene (within 10 percent).
- Do not add new facts you are not certain of. Do not add citations.
Return ONLY JSON: {"scenes": [{"narration": "..."}], "changes": ["one short note per correction"]}
with exactly as many scenes as you received. If everything is accurate, return the narration unchanged and an
empty "changes" list."""


def targets(seconds: int, scenes: int | None = None, format: str | None = None) -> tuple[int, int]:
    """(number of scenes, total narration words) for a target length."""
    lo, hi = config.fmt(format)["scenes"]
    n = int(scenes or config.default_scenes(seconds, format))
    n = max(lo, min(hi, n))
    return n, int(config.narration_seconds(seconds, format) * config.WORDS_PER_SECOND)


def user_prompt(brief: dict) -> str:
    fmt = brief.get("format") or config.DEFAULT_FORMAT
    n, words = targets(brief["seconds"], brief.get("scenes"), fmt)
    tone = config.TONES.get(brief.get("tone") or "informative", config.TONES["informative"])
    audience = config.AUDIENCES.get(brief.get("audience") or "general", config.AUDIENCES["general"])
    lines = [f"Topic: {brief['topic']}",
             f"Length: about {brief['seconds']} seconds = about {words} words of narration in total.",
             f"Write exactly {n} scenes of roughly {max(8, words // n)} words each.",
             f"Tone: {tone[1]}.", f"Audience: {audience[1]}.",
             f"Mark {brief.get('heroes', 0)} scenes as hero."]
    if brief.get("key_points"):
        lines.append(f"The viewer asked you to cover these points (only if they are accurate): {brief['key_points']}")
    if fmt == "reel":
        lines.append(REEL_RULES)
    return "\n".join(lines)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^\(?(scene|narrator)\s*\d*\)?\s*[:\-]\s*", "", text, flags=re.I)
    return text.replace("*", "").replace("#", "")


def normalize(data: dict, topic: str, heroes: int, max_scenes: int = 16) -> dict:
    """Validate the LLM output and repair what can be repaired. Raises ValueError if unusable."""
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise ValueError("JSON has no 'scenes' list")
    scenes = []
    for s in data["scenes"]:
        if not isinstance(s, dict):
            continue
        narration = _clean(s.get("narration"))
        raw_shots = s.get("shots") if isinstance(s.get("shots"), list) else []
        shots = [_clean(v)[:400] for v in raw_shots + [s.get("visual")] if isinstance(v, str) and _clean(v)]
        if len(narration.split()) < 4 or not shots:
            continue
        scenes.append({
            "narration": narration[:900],
            "shots": shots[:config.MAX_SHOTS_PER_SCENE],
            "visual": shots[0],
            "motion": _clean(s.get("motion")) or "subtle natural movement, slow cinematic dolly in",
            "hero": bool(s.get("hero", False)),
        })
    if len(scenes) < 3:
        raise ValueError(f"only {len(scenes)} usable scenes")
    scenes = scenes[:max_scenes]
    heroes = max(0, min(heroes, len(scenes)))

    # exactly `heroes` hero scenes: keep the LLM's picks, then fill evenly spaced ones
    flagged = [i for i, s in enumerate(scenes) if s["hero"]][:heroes]
    if len(flagged) < heroes:
        step = len(scenes) / (heroes + 1)
        for k in range(1, heroes + 1):
            idx = min(len(scenes) - 1, round(k * step))
            if idx not in flagged and len(flagged) < heroes:
                flagged.append(idx)
    for i, s in enumerate(scenes):
        s["hero"] = i in flagged
        s["id"] = i

    title = _clean(data.get("title")) or topic.title()
    return {"title": title[:90], "logline": _clean(data.get("logline"))[:240], "scenes": scenes}


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("no JSON object in LLM response")
        return json.loads(m.group(0))


PREFERRED_MODELS = [  # first one that the Groq account can use wins
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]


def pick_model(client, log=print) -> str:
    """Groq retires models from time to time - choose the best one this key can use right now."""
    wanted = [config.LLM_MODEL] + [m for m in PREFERRED_MODELS if m != config.LLM_MODEL]
    try:
        available = {m.id for m in client.models.list().data}
    except Exception as exc:  # listing failed - just try the configured model
        log(f"Could not list Groq models ({exc}); using {config.LLM_MODEL}")
        return config.LLM_MODEL
    for m in wanted:
        if m in available:
            return m
    chat = sorted(m for m in available if not any(x in m for x in ("whisper", "tts", "guard", "embed")))
    if not chat:
        raise RuntimeError("No chat model is available for this Groq API key.")
    return chat[0]


def _complete(client, model, messages, temperature: float = 0.7, effort: str = "medium", max_tokens: int = 12000):
    """One chat call. Reasoning models get an effort level; JSON mode is dropped if the model rejects it."""
    import openai

    kwargs = dict(model=model, temperature=temperature, messages=messages, max_tokens=max_tokens)
    if "gpt-oss" in model or "qwen3" in model:
        kwargs["extra_body"] = {"reasoning_effort": effort if "gpt-oss" in model else "none"}
    try:
        return client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except openai.BadRequestError:
        kwargs.pop("extra_body", None)
        return client.chat.completions.create(**kwargs)


_LLM: dict = {}


def llm(log=print):
    """(client, model) - created once per process and shared by the writer, fact-checker and director."""
    if "client" not in _LLM:
        from openai import OpenAI

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GROQ_API_KEY is missing - add it to the 'docugen-secrets' Modal secret.")
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", max_retries=6, timeout=120)
        _LLM["client"], _LLM["model"] = client, pick_model(client, log)
        log(f"LLM: {_LLM['model']} (Groq)")
    return _LLM["client"], _LLM["model"]


def chat_json(messages: list[dict], log=print, temperature: float = 0.7, effort: str = "medium",
              max_tokens: int = 12000) -> dict:
    client, model = llm(log)
    resp = _complete(client, model, messages, temperature, effort, max_tokens)
    return parse_json(resp.choices[0].message.content)


def write_script(brief: dict, log=print) -> dict:
    """brief: topic, seconds, scenes, heroes, format, tone, audience, key_points."""
    n, words = targets(brief["seconds"], brief.get("scenes"), brief.get("format"))
    heroes = int(brief.get("heroes", 0))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(brief)}]
    last_err, script = None, None
    for attempt in range(1, 4):
        try:
            script = normalize(chat_json(messages, log), brief["topic"], heroes, n)
            total = sum(len(s["narration"].split()) for s in script["scenes"])
            log(f"Script attempt {attempt}: {len(script['scenes'])} scenes, {total} words (target {words})")
            if total < words * 0.6 and attempt < 3:
                messages += [{"role": "assistant", "content": json.dumps(script)},
                             {"role": "user", "content": f"Too short ({total} words). Rewrite to about {words} words."}]
                continue
            return script
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            log(f"Script attempt {attempt} invalid: {exc}")
            time.sleep(2)
    if script is None:
        raise RuntimeError(f"The LLM did not return a usable script: {last_err}")
    return script


def fact_check(script: dict, topic: str, log=print) -> list[str]:
    """Second LLM pass that corrects or softens doubtful claims in place. Returns the list of changes."""
    scenes = script["scenes"]
    payload = {"topic": topic, "scenes": [{"narration": s["narration"]} for s in scenes]}
    try:
        data = chat_json([{"role": "system", "content": FACT_CHECK_PROMPT},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                         log, temperature=0.2, effort="high")
        fixed = data.get("scenes") if isinstance(data, dict) else None
        if not isinstance(fixed, list) or len(fixed) != len(scenes):
            raise ValueError("fact-checker changed the number of scenes")
        applied = 0
        for s, f in zip(scenes, fixed):
            new = _clean(f.get("narration") if isinstance(f, dict) else "")
            old_n, new_n = len(s["narration"].split()), len(new.split())
            if new and new != s["narration"] and 0.6 * old_n <= new_n <= 1.4 * old_n:
                s["narration"] = new[:900]
                applied += 1
        changes = [_clean(c)[:200] for c in (data.get("changes") or []) if isinstance(c, str) and _clean(c)]
        log(f"Fact-check: {len(changes)} correction(s), {applied} scene(s) edited")
        return changes[:12] if applied else []
    except Exception as exc:  # the film can still be made; just say the check did not run
        log(f"Fact-check skipped: {str(exc)[:160]}")
        return []


def to_markdown(script: dict) -> str:
    lines = [f"# {script['title']}", "", f"_{script['logline']}_", ""]
    for s in script["scenes"]:
        lines += [f"## Scene {s['id'] + 1}" + (" (animated)" if s["hero"] else ""), "", s["narration"], ""]
        lines += [f"- **Shot {k + 1}:** {v}" for k, v in enumerate(s.get("shots") or [s["visual"]])]
        if s["hero"]:
            lines.append(f"- **Motion:** {s['motion']}")
        lines.append("")
    return "\n".join(lines)
