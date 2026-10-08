"""Stage 1 - the LLM writes the documentary as structured JSON; Python validates and repairs it."""
from __future__ import annotations

import json
import os
import re
import time

from . import config

SYSTEM_PROMPT = """You are an award-winning documentary writer and director.
Write a short narrated documentary as JSON. Rules:
- Narration is calm, vivid and factual. Use only well-established facts; avoid precise numbers unless you are sure.
- Scene 1 is a hook that makes the viewer curious. The last scene closes the story with a reflective line.
- Each scene has one continuous narration paragraph (no headings, no stage directions, no emojis).
- "visual" describes ONE concrete photograph for an image generator: subject, setting, lighting, camera angle.
  Never ask for text, logos, brand names, screens with readable words, or recognisable real people.
- "motion" describes subtle camera or subject movement for that shot (e.g. "slow push in, people walking past").
- Mark exactly the requested number of the most visually dynamic scenes with "hero": true.
Return ONLY this JSON:
{"title": "...", "logline": "one sentence", "scenes": [
  {"narration": "...", "visual": "...", "motion": "...", "hero": false}
]}"""


def targets(seconds: int) -> tuple[int, int]:
    """(number of scenes, total narration words) for a target length."""
    scenes = max(5, min(14, round(seconds / 13)))
    words = int(seconds * config.WORDS_PER_SECOND)
    return scenes, words


def user_prompt(topic: str, seconds: int, heroes: int) -> str:
    n, words = targets(seconds)
    return (f"Topic: {topic}\n"
            f"Length: about {seconds} seconds of narration = about {words} words in total.\n"
            f"Write exactly {n} scenes of roughly {words // n} words each.\n"
            f"Mark {heroes} scenes as hero.")


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^\(?(scene|narrator)\s*\d*\)?\s*[:\-]\s*", "", text, flags=re.I)
    return text.replace("*", "").replace("#", "")


def normalize(data: dict, topic: str, heroes: int) -> dict:
    """Validate the LLM output and repair what can be repaired. Raises ValueError if unusable."""
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise ValueError("JSON has no 'scenes' list")
    scenes = []
    for s in data["scenes"]:
        if not isinstance(s, dict):
            continue
        narration = _clean(s.get("narration"))
        visual = _clean(s.get("visual"))
        if len(narration.split()) < 4 or not visual:
            continue
        scenes.append({
            "narration": narration[:900],
            "visual": visual[:400],
            "motion": _clean(s.get("motion")) or "slow cinematic push in",
            "hero": bool(s.get("hero", False)),
        })
    if len(scenes) < 3:
        raise ValueError(f"only {len(scenes)} usable scenes")
    scenes = scenes[:16]

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


def write_script(topic: str, seconds: int, heroes: int, log=print) -> dict:
    from openai import OpenAI

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing - add it to the 'docugen-secrets' Modal secret.")
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", max_retries=5, timeout=90)
    _, words = targets(seconds)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(topic, seconds, heroes)}]
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL, temperature=0.7, response_format={"type": "json_object"},
                messages=messages, max_tokens=4000)
            script = normalize(parse_json(resp.choices[0].message.content), topic, heroes)
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
        lines += [f"## Scene {s['id'] + 1}" + (" (motion)" if s["hero"] else ""), "", s["narration"], "",
                  f"**Visual:** {s['visual']}", ""]
    return "\n".join(lines)
