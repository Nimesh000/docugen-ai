"""Visual Director agent: turns the finished narration into engaging, accurate image and motion prompts.

How it works (all LLM calls go to the same Groq model as the script writer):
  1. Visual bible   - one call fixes the look of the whole film: places, era, palette, recurring characters with a
                      fixed description and a recurring motif, so every shot feels like the same documentary.
  2. Shot design    - scenes are sent in batches. Each scene gets exactly the number of shots the edit needs,
                      following a shot grammar (wide -> medium -> detail -> reaction, never the same framing twice
                      in a row) and a prompt recipe tuned for Z-Image Turbo. Hero scenes also get a Wan 2.2 motion
                      prompt (one clear action + ambient motion + one camera move).
  3. Critic         - a rule-based linter checks every prompt (forbidden things that render as garbled text, too
                      short / too long, missing place or era, duplicates). Only flagged prompts go back to the LLM
                      for one repair round; anything still bad is sanitised in code.
If the agent fails for any reason, the pipeline falls back to the shot ideas in the script.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from . import config
from .script import _clean, chat_json

DIRECTOR_PROMPT = """You are the Visual Director of an award-winning documentary studio: a senior cinematographer and
an expert prompt engineer for two models.

IMAGE MODEL - Z-Image Turbo: it follows long, precise natural-language descriptions very literally. It renders ANY
writing that a prompt implies, so text-bearing objects ruin the image. Each prompt is generated on its own with no
memory of other prompts, so every prompt must be complete.

VIDEO MODEL - Wan 2.2 image-to-video: it animates the first still of a hero scene for 5 seconds. It works best with
one clear main action, some secondary ambient motion, and exactly one smooth camera move. It fails with scene cuts,
new objects appearing, fast complex choreography, or text.

YOUR JOB
A) Read the whole narration and fix a VISUAL BIBLE: the era(s), real places, a 3-colour palette, the light, the
   recurring characters with a fixed physical description (age, build, clothing, hair; period-accurate), and one
   recurring visual motif that ties the film together. The motif is a real, photographable object or element
   (a brass bell, a red scarf, monsoon rain), never graphics: no glowing lines, icons, holograms or UI overlays.
B) For every scene, write exactly the requested number of shot prompts, in the order the narration is spoken, so each
   shot shows what the voice is saying at that moment.

SHOT GRAMMAR
- Vary framing: establishing wide / aerial, medium, close-up detail of hands or objects, reaction or portrait,
  over-the-shoulder or point of view, low or high angle. Never two of the same framing in a row.
- The very first shot of the film is the most striking image of the whole film (it is the hook).
- One clear focal subject per shot, caught at the peak of an action (mid-stride, steam rising, sparks flying,
  water splashing). Faces show emotion. Use depth: something in the foreground, the subject, a background.
- Accuracy matters: period-correct clothing, tools, vehicles and architecture for the exact place and year.

PROMPT RECIPE (55 to 90 words, one paragraph, in this order)
[shot type and camera angle], [main subject using the bible description], [the specific action],
[setting: city/region + country + year or era, restated in EVERY prompt], [foreground and background details],
[lighting and time of day], [lens and depth of field], [mood and palette].
Do not add style words like "photorealistic" or "oil painting": the studio appends the house style itself.

NEVER SHOW (they render as gibberish text or cause trouble)
screens, monitors, phones seen from the front, dashboards, signs, banners, posters, billboards, books, newspapers,
letters, papers, documents, maps, charts, graphs, diagrams, whiteboards, handwriting, numbers, price tags, money
notes, logos, brand names, flags with emblems, any named real person or celebrity (show "a scientist", "a leader"
generically), gore. Show ideas through people, hands, gestures, places, objects, nature and light instead.
Example: NOT "a phone screen showing a UPI payment" BUT "a street vendor in Mumbai, 2019, smiling as a customer holds
up a phone toward a small stand, the phone seen from behind".

MOTION PROMPT for hero scenes (25 to 50 words): describes how shot 1 of that scene moves: the main subject's action,
ambient motion (smoke, cloth, crowd, water, dust), and one camera move (slow push in, slow orbit, gentle pan, crane up,
tracking alongside). Physically plausible, continuous, no cuts. Shot 1 of a hero scene must show something that moves
well on video (people working or walking, vehicles, water, fire, crowds, weather).

GOOD EXAMPLE (one shot):
"Low-angle medium shot of a young Indian railway engineer in a khaki uniform and white turban, sleeves rolled,
hammering a steel rail spike at the Bori Bunder yard in Bombay, India, 1853, sparks leaping from the hammer, a black
steam locomotive venting white steam behind him, workers blurred in the background, hard morning sunlight through
coal smoke, 35mm lens, shallow depth of field, determined mood, palette of soot black, brass and dusty ochre"
BAD EXAMPLE: "The railway is built. Workers. Old times." (vague, no place, no era, no light, no lens)
"""

FORMAT_NOTE = {
    "long": "FRAME: wide 16:9 landscape. Use the width: wide vistas, subjects placed on thirds, layered depth.",
    "reel": ("FRAME: tall vertical 9:16 for phones. Keep the main subject large and centred in the middle of the frame, "
             "favour close and medium shots, low angles and strong vertical lines (towers, trees, standing people); "
             "avoid tiny distant subjects."),
}

BIBLE_SCHEMA = """Return ONLY JSON:
{"bible": {"era": "...", "places": ["..."], "palette": "...", "light": "...",
           "characters": [{"name": "role, e.g. the engineer", "look": "fixed physical description"}],
           "motif": "..."},
 "scenes": [{"scene": <number>, "shots": ["...", "..."], "motion": "... (hero scenes only, else empty)"}]}"""

BATCH_SCHEMA = """Return ONLY JSON:
{"scenes": [{"scene": <number>, "shots": ["...", "..."], "motion": "... (hero scenes only, else empty)"}]}"""

REPAIR_PROMPT = """You are the Visual Director's critic. Rewrite each listed prompt so it fixes the listed problems
while keeping its subject, framing and meaning. Follow the same recipe: 55-90 words; shot type, subject, action,
place + country + era, foreground/background, light, lens, mood. No screens, signs, papers, maps, charts, writing,
numbers, logos or named real people. Return ONLY JSON: {"fixes": [{"id": "<id>", "prompt": "..."}]}"""

# things that make Z-Image draw fake text, or real-person likenesses
FORBIDDEN = re.compile(
    r"\b(screens?|monitors?|displays?|dashboards?|laptop screen|computer screen|phone screen|smartphone screen|"
    r"tablet screen|signs?|signage|signboards?|banners?|posters?|billboards?|placards?|books?|newspapers?|"
    r"magazines?|letters?|papers?|documents?|notebooks?|manuscripts?|scrolls?|maps?|charts?|graphs?|diagrams?|"
    r"whiteboards?|blackboards?|chalkboards?|handwriting|writing|written|text|caption|labels?|logos?|brand|"
    r"price tags?|banknotes?|bank notes?|currency notes?|receipts?|headlines?|slogans?|typography|inscriptions?|"
    r"numbers?|digits?|qr codes?)\b", re.I)
SAFE_REPLACEMENTS = [  # last-resort sanitiser for anything the critic could not fix: swap for safe nouns
    (re.compile(r"\b(phone|smartphone|laptop|computer|tablet)(?:'s)? (screens?|displays?|monitors?)\b", re.I), r"\1"),
    (re.compile(r"\b(screens?|displays?|monitors?|dashboards?)\b", re.I), "device"),
    (re.compile(r"\b(signs?|signage|signboards?|banners?|posters?|billboards?|placards?)\b", re.I), "stall awning"),
    (re.compile(r"\b(maps?|charts?|graphs?|diagrams?|whiteboards?|blackboards?|chalkboards?)\b", re.I), "tool rack"),
    (re.compile(r"\b(documents?|papers?|newspapers?|magazines?|books?|letters?|notebooks?|manuscripts?|scrolls?|"
                r"receipts?)\b", re.I), "wooden crate"),
    (re.compile(r"\b(banknotes?|bank notes?|currency notes?|price tags?|qr codes?|logos?|labels?)\b", re.I), "objects"),
    (FORBIDDEN, ""),
]


def plan_counts(script: dict, format: str, pacing: str, plan_scene) -> list[int]:
    """Shots each scene will need, estimated from its word count before the narration is recorded."""
    shot_len = config.shot_seconds(format, pacing)
    counts = []
    for s in script["scenes"]:
        est = len(s["narration"].split()) / config.WORDS_PER_SECOND + 0.4
        n = len(plan_scene(est, s["hero"], shot_len))
        counts.append(min(config.MAX_SHOTS_PER_SCENE, n + 1))   # one spare in case the voice runs long
    return counts


def lint(prompt: str, bible: dict | None = None) -> list[str]:
    problems = []
    words = len(prompt.split())
    if words < 30:
        problems.append(f"too short ({words} words), add setting, light and lens")
    if words > 130:
        problems.append(f"too long ({words} words)")
    bad = sorted({m.group(0).lower() for m in FORBIDDEN.finditer(prompt)})
    if bad:
        problems.append("mentions things that render as text: " + ", ".join(bad))
    if not re.search(r"\b(1[0-9]{3}s?|20[0-9]{2}s?|[0-9]{1,2}(st|nd|rd|th).century|ancient|medieval|prehistoric|"
                     r"modern|present.day|today|contemporary|era|age|period|bc|bce|ad|ce)\b", prompt, re.I):
        problems.append("no era or year")
    return problems


def sanitize(prompt: str) -> str:
    for pat, rep in SAFE_REPLACEMENTS:
        prompt = pat.sub(rep, prompt)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+,", ",", prompt)).strip(" ,")


def _scene_brief(i: int, s: dict, count: int) -> dict:
    return {"scene": i + 1, "narration": s["narration"], "shots_needed": count, "hero": bool(s["hero"])}


def _brief_text(script: dict, brief: dict) -> str:
    style = config.STYLES.get(brief.get("style") or "cinematic", config.STYLES["cinematic"])["label"]
    tone = config.TONES.get(brief.get("tone") or "informative", config.TONES["informative"])[1]
    return (f"Film: {script['title']} - {script.get('logline', '')}\nTopic: {brief.get('topic', '')}\n"
            f"House style (appended automatically): {style}. Tone: {tone}.\n"
            f"{FORMAT_NOTE.get(brief.get('format') or 'long', FORMAT_NOTE['long'])}")


def _parse_scenes(data: dict, wanted: dict[int, int]) -> dict[int, dict]:
    out = {}
    for item in (data.get("scenes") or []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("scene")) - 1
        except (TypeError, ValueError):
            continue
        if idx not in wanted:
            continue
        shots = [_clean(p)[:900] for p in (item.get("shots") or []) if isinstance(p, str) and len(_clean(p)) > 10]
        if shots:
            out[idx] = {"shots": shots[:config.MAX_SHOTS_PER_SCENE], "motion": _clean(item.get("motion"))[:500]}
    return out


def direct(script: dict, counts: list[int], brief: dict, log=print, batch_shots: int = 24) -> dict:
    """Returns {"bible": {...}, "scenes": [{"shots": [...], "motion": "..."}], "stats": {...}} (one per scene)."""
    scenes = script["scenes"]
    head = _brief_text(script, brief)
    full_story = "\n".join(f"Scene {i + 1}: {s['narration']}" for i, s in enumerate(scenes))

    # batches of scenes with at most `batch_shots` shots each
    batches, cur, n = [], [], 0
    for i, c in enumerate(counts):
        if cur and n + c > batch_shots:
            batches.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += c
    if cur:
        batches.append(cur)

    def ask(idxs: list[int], bible: dict | None) -> dict:
        want = [_scene_brief(i, scenes[i], counts[i]) for i in idxs]
        user = [head, "", "FULL NARRATION (for context):", full_story, ""]
        if bible:
            user += ["VISUAL BIBLE (follow it exactly, reuse character descriptions word for word):",
                     json.dumps(bible, ensure_ascii=False), ""]
        user += ["WRITE SHOTS FOR THESE SCENES:", json.dumps(want, ensure_ascii=False), "",
                 BATCH_SCHEMA if bible else BIBLE_SCHEMA]
        return chat_json([{"role": "system", "content": DIRECTOR_PROMPT}, {"role": "user", "content": "\n".join(user)}],
                         log, temperature=0.8, effort="medium")

    first = ask(batches[0], None)
    bible = first.get("bible") if isinstance(first.get("bible"), dict) else {}
    result = _parse_scenes(first, {i: counts[i] for i in batches[0]})
    if len(batches) > 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for data in pool.map(lambda b: ask(b, bible or {"note": "keep the look consistent"}), batches[1:]):
                result.update(_parse_scenes(data, {i: counts[i] for b in batches[1:] for i in b}))

    # fill gaps from the script's own shot ideas
    plan = []
    for i, s in enumerate(scenes):
        got = result.get(i) or {"shots": [], "motion": ""}
        shots = got["shots"] or list(s.get("shots") or [s.get("visual", "")])
        plan.append({"shots": shots, "motion": got["motion"] or s.get("motion", ""), "from_agent": i in result})

    # critic: lint everything, send only the flagged prompts back for one repair round
    flagged = {}
    for i, p in enumerate(plan):
        for j, prompt in enumerate(p["shots"]):
            probs = lint(prompt, bible)
            if probs:
                flagged[f"{i}.{j}"] = {"prompt": prompt, "problems": probs}
    repaired = 0
    if flagged:
        try:
            items = [{"id": k, **v} for k, v in list(flagged.items())[:40]]
            ctx = f"{head}\nVISUAL BIBLE: {json.dumps(bible, ensure_ascii=False)}"
            data = chat_json([{"role": "system", "content": REPAIR_PROMPT},
                              {"role": "user", "content": ctx + "\n\nFIX THESE:\n" + json.dumps(items, ensure_ascii=False)}],
                             log, temperature=0.4, effort="low")
            for fx in data.get("fixes") or []:
                key, new = str(fx.get("id", "")), _clean(fx.get("prompt", ""))
                if key in flagged and new and len(lint(new)) < len(flagged[key]["problems"]):
                    i, j = map(int, key.split("."))
                    plan[i]["shots"][j] = new
                    repaired += 1
        except Exception as exc:
            log(f"Director critic skipped: {str(exc)[:120]}")
    sanitized = 0
    for p in plan:
        for j, prompt in enumerate(p["shots"]):
            if FORBIDDEN.search(prompt):
                p["shots"][j] = sanitize(prompt)
                sanitized += 1
        p["motion"] = sanitize(p["motion"]) if p["motion"] else p["motion"]
    stats = {"shots": sum(len(p["shots"]) for p in plan), "agent_scenes": sum(p["from_agent"] for p in plan),
             "flagged": len(flagged), "repaired": repaired, "sanitized": sanitized, "batches": len(batches)}
    log(f"Visual Director: {stats['shots']} shot prompts for {len(plan)} scenes, {stats['flagged']} flagged by the "
        f"critic, {repaired} repaired")
    return {"bible": bible, "scenes": plan, "stats": stats}
