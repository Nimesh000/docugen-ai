"""Captions: group timed words into readable cues -> SRT + styled ASS (optional faster-whisper alignment)."""
from __future__ import annotations

from pathlib import Path

SENTENCE_END = (".", "?", "!", "।")  # includes Devanagari danda


def fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def build_cues(groups: list[list[dict]], max_words: int = 7, max_seconds: float = 2.8) -> list[dict]:
    cues: list[dict] = []
    for words in groups:
        current: list[dict] = []

        def flush() -> None:
            if current:
                text = " ".join(w["w"] for w in current if w["w"]).strip()
                if text:
                    cues.append({"start": current[0]["s"], "end": current[-1]["e"], "text": text})
                current.clear()

        for i, w in enumerate(words):
            if current:
                gap = w["s"] - current[-1]["e"]
                too_long = w["e"] - current[0]["s"] > max_seconds
                if gap > 0.6 or too_long or len(current) >= max_words:
                    flush()
            current.append(w)
            if w["w"].endswith(SENTENCE_END) and len(current) >= 2:
                flush()
        flush()

    # hold each cue a little longer (helps readability) without overlapping the next one
    for i, cue in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else cue["end"] + 1.0
        cue["end"] = min(cue["end"] + 0.25, max(cue["end"], nxt - 0.02))
        cue["end"] = max(cue["end"], cue["start"] + 0.3)
    return cues


def write_srt(cues: list[dict], path: Path) -> None:
    blocks = [f"{i}\n{fmt_srt_time(c['start'])} --> {fmt_srt_time(c['end'])}\n{c['text']}\n"
              for i, c in enumerate(cues, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_ass(cues: list[dict], path: Path, width: int, height: int, font: str = "Inter",
              font_size: int = 54) -> None:
    scale = height / 1080
    size = int(font_size * scale)
    margin_v = int(80 * scale)
    outline = max(2, int(3 * scale))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00101010,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,{int(160 * scale)},{int(160 * scale)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for c in cues:
        text = c["text"].replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")
        lines.append(f"Dialogue: 0,{fmt_ass_time(c['start'])},{fmt_ass_time(c['end'])},Default,,0,0,0,,{text}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def transcribe_words(wav_path: Path, model_name: str = "base", prompt: str = "", log=print) -> list[dict]:
    """Word-level timestamps for the narration (CPU, int8)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav_path), language="en", word_timestamps=True, vad_filter=False,
                                   initial_prompt=prompt[:200] or None, beam_size=5)
    words = []
    for seg in segments:
        for w in seg.words or []:
            if w.end > w.start and w.word.strip():
                words.append({"s": round(w.start, 3), "e": round(w.end, 3), "w": w.word.strip()})
    log(f"Captions: {len(words)} words aligned")
    return words


def words_from_script(scene_spans: list[tuple[float, float, str]]) -> list[dict]:
    """Fallback alignment: spread each scene's words evenly over its narration span."""
    words = []
    for start, end, text in scene_spans:
        toks = text.split()
        if not toks:
            continue
        step = (end - start) / len(toks)
        for i, t in enumerate(toks):
            words.append({"s": round(start + i * step, 3), "e": round(start + (i + 0.9) * step, 3), "w": t})
    return words
