"""Stage 1 - the LLM writes the documentary as structured JSON; Python validates and repairs it."""
from __future__ import annotations

import json
import os
import re
import time

from . import config

SYSTEM_PROMPT = """You are an award-winning documentary writer and director.
Write a short narrated documentary as JSON. Rules:
- Narration is vivid, confident and factual, with concrete, well-established details: key years, places,
  institutions, inventions, turning points. No vague filler ("a quiet revolution began"). If unsure of a number, leave it out.
- Scene 1 is a strong hook. Each scene moves the story forward. The last scene closes with a memorable line.
- Each scene has one narration paragraph of plain spoken sentences (no headings, stage directions, emojis or lists).
- "shots" are 4 different photographs that illustrate that scene's narration, in order: wide establishing shot,
  medium shot of people, close-up of hands / faces / textures, and another angle. Each is ONE concrete image
  description for an image generator: subject, action, setting, era, lighting, camera angle.
  Every shot is generated on its own, so EACH shot must restate the country/city, era and who is in it
  (e.g. "Indian shopkeeper in a Mumbai market, 2018"), never just "the vendor" or "people".
  The image model renders any writing it sees in the prompt, so NEVER include phone or computer screens, monitors,
  dashboards, signs, banners, posters, papers, documents, notebooks, handwriting, numbers, charts, QR codes,
  banknotes, price tags, logos or brand names, and no recognisable real people. Show the idea through people,
  hands, gestures, places, objects and light instead (e.g. "a vendor smiling as a customer holds up a phone to
  pay, phone seen from behind", not "a phone screen showing a payment").
- "motion" describes how the FIRST shot comes alive as a 5 second video: what moves (people walking, steam rising,
  hands tapping a phone) and the camera move (slow dolly in, pan left). One sentence.
- Mark exactly the requested number of the most visually dynamic scenes with "hero": true.
Return ONLY this JSON:
{"title": "...", "logline": "one sentence", "scenes": [
  {"narration": "...", "shots": ["...", "...", "...", "..."], "motion": "...", "hero": false}
]}"""


def targets(seconds: int, scenes: int | None = None) -> tuple[int, int]:
    """(number of scenes, total narration words) for a target length."""
    n = scenes or config.default_scenes(seconds)
    n = max(config.MIN_SCENES, min(config.MAX_SCENES, int(n)))
    narration_seconds = max(10, seconds - 2 * config.CARD_SECONDS)
    return n, int(narration_seconds * config.WORDS_PER_SECOND)


def user_prompt(topic: str, seconds: int, heroes: int, scenes: int | None = None) -> str:
    n, words = targets(seconds, scenes)
    return (f"Topic: {topic}\n"
            f"Length: about {seconds} seconds = about {words} words of narration in total.\n"
            f"Write exactly {n} scenes of roughly {max(8, words // n)} words each.\n"
            f"Mark {heroes} scenes as hero.")


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
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
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


def _complete(client, model, messages):
    """One chat call. Reasoning models get low effort; JSON mode is dropped if the model rejects it."""
    import openai

    kwargs = dict(model=model, temperature=0.7, messages=messages, max_tokens=12000)
    if "gpt-oss" in model or "qwen3" in model:
        kwargs["extra_body"] = {"reasoning_effort": "medium" if "gpt-oss" in model else "none"}
    try:
        return client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except openai.BadRequestError:
        kwargs.pop("extra_body", None)
        return client.chat.completions.create(**kwargs)


def write_script(topic: str, seconds: int, heroes: int, log=print, scenes: int | None = None) -> dict:
    from openai import OpenAI

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing - add it to the 'docugen-secrets' Modal secret.")
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", max_retries=5, timeout=90)
    model = pick_model(client, log)
    log(f"Script model: {model} (Groq)")
    n, words = targets(seconds, scenes)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(topic, seconds, heroes, n)}]
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = _complete(client, model, messages)
            script = normalize(parse_json(resp.choices[0].message.content), topic, heroes, n)
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
    if last_err:
        raise RuntimeError(f"The LLM did not return a usable script: {last_err}")
    return script  # type: ignore[possibly-undefined]


def to_markdown(script: dict) -> str:
    lines = [f"# {script['title']}", "", f"_{script['logline']}_", ""]
    for s in script["scenes"]:
        lines += [f"## Scene {s['id'] + 1}" + (" (animated)" if s["hero"] else ""), "", s["narration"], ""]
        lines += [f"- **Shot {k + 1}:** {v}" for k, v in enumerate(s.get("shots") or [s["visual"]])]
        if s["hero"]:
            lines.append(f"- **Motion:** {s['motion']}")
        lines.append("")
    return "\n".join(lines)
