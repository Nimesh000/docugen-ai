"""Kokoro-82M narration on a small GPU (~50x faster than real time). It also returns word timestamps,
so captions need no separate speech recognition pass."""
from __future__ import annotations

import io
import re
import wave

SAMPLE_RATE = 24000
SENTENCE_GAP = 0.22          # seconds of silence between sentences


def trim_silence(audio, threshold: float = 0.02, keep: float = 0.04):
    """Cut leading/trailing silence. Returns (trimmed audio, seconds removed from the start)."""
    import numpy as np

    loud = np.flatnonzero(np.abs(audio) > threshold * max(1e-6, float(np.abs(audio).max())))
    if loud.size == 0:
        return audio, 0.0
    pad = int(keep * SAMPLE_RATE)
    start, end = max(0, loud[0] - pad), min(len(audio), loud[-1] + pad)
    return audio[start:end], start / SAMPLE_RATE


class Narrator:
    def __init__(self, device: str = "cpu"):
        import torch
        from kokoro import KModel

        from . import config

        self.torch = torch
        self.model = KModel(repo_id=config.TTS_REPO).to(device).eval()
        self.pipelines: dict = {}

    def _pipeline(self, voice: str):
        from kokoro import KPipeline

        from . import config

        lang = "b" if voice.startswith("b") else "a"
        if lang not in self.pipelines:
            self.pipelines[lang] = KPipeline(lang_code=lang, repo_id=config.TTS_REPO, model=self.model)
        return self.pipelines[lang]

    def speak(self, text: str, voice: str, speed: float = 1.0) -> tuple[bytes, list[dict]]:
        """-> (16-bit mono WAV, [{"w": word, "s": start, "e": end}, ...]) with times in seconds."""
        import numpy as np

        chunks, words, offset = [], [], 0.0
        with self.torch.inference_mode():
            for result in self._pipeline(voice)(text, voice=voice, speed=speed, split_pattern=r"(?<=[.!?])\s+"):
                if result.audio is None:
                    continue
                audio = result.audio.detach().cpu().numpy().astype(np.float32)
                audio, cut = trim_silence(audio)          # Kokoro pads every sentence with silence
                if chunks:                                 # keep a short, natural pause between sentences
                    chunks.append(np.zeros(int(SENTENCE_GAP * SAMPLE_RATE), np.float32))
                    offset += SENTENCE_GAP
                for t in result.tokens or []:
                    s, e = getattr(t, "start_ts", None), getattr(t, "end_ts", None)
                    if s is None or e is None or not re.search(r"\w", t.text or ""):
                        continue
                    s, e = max(0.0, s - cut), max(0.0, e - cut)
                    words.append({"w": t.text.strip(), "s": round(offset + s, 3),
                                  "e": round(offset + max(e, s + 0.05), 3)})
                chunks.append(audio)
                offset += len(audio) / SAMPLE_RATE
        _attach_punctuation(words, text)
        pcm = np.clip(np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE // 2, np.float32), -1, 1)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(SAMPLE_RATE)
            fh.writeframes((pcm * 32767).astype("<i2").tobytes())
        return buf.getvalue(), words


def _attach_punctuation(words: list[dict], text: str) -> None:
    """Kokoro tokens drop trailing punctuation; restore it from the source text so captions read naturally."""
    src = text.split()
    j = 0
    for w in words:
        core = re.sub(r"\W+", "", w["w"]).lower()
        for k in range(j, min(j + 4, len(src))):
            if re.sub(r"\W+", "", src[k]).lower() == core:
                w["w"] = src[k]
                j = k + 1
                break
