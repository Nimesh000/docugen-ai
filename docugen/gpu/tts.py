"""XTTS-v2 narration (Coqui TTS). Runs inside a GPU container."""
from __future__ import annotations

import io
import wave

import numpy as np


class Narrator:
    SAMPLE_RATE = 24000

    def __init__(self, device: str = "cuda"):
        import os

        os.environ.setdefault("COQUI_TOS_AGREED", "1")  # XTTS-v2 is under the Coqui Public Model License
        from TTS.api import TTS

        from .. import config

        self.tts = TTS(config.TTS_MODEL).to(device)

    def speak(self, text: str, speaker: str, language: str = "en") -> bytes:
        audio = self.tts.tts(text=text, speaker=speaker, language=language, split_sentences=True)
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.SAMPLE_RATE)
            fh.writeframes((pcm * 32767).astype("<i2").tobytes())
        return buf.getvalue()
