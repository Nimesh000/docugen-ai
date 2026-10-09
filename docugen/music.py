"""Royalty-free background music: an ambient loop synthesised with numpy (no samples, no licences)."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

MOODS = {
    # bpm, chord progression (MIDI notes), arpeggio notes per bar, low pulse per beat (0 = none)
    "ambient": (72, [[48, 55, 59, 64, 67], [45, 52, 55, 60, 64], [41, 48, 52, 57, 64], [43, 50, 55, 59, 64]], 8, 0.0),
    "uplifting": (96, [[50, 57, 62, 66, 69], [45, 52, 57, 61, 64], [47, 54, 59, 62, 66], [43, 50, 55, 59, 62]], 8, 0.05),
    "tense": (64, [[45, 52, 57, 60, 64], [41, 48, 53, 57, 60], [38, 45, 50, 53, 57], [40, 47, 52, 56, 59]], 4, 0.08),
}


def _synth_loop(path: Path, mood: str = "ambient", bars: int = 16, sr: int = 44100) -> None:
    import numpy as np

    def hz(m: int) -> float:
        return 440 * 2 ** ((m - 69) / 12)

    bpm, prog, arps, pulse = MOODS.get(mood, MOODS["ambient"])
    seconds_per_bar = 60 / bpm * 4
    beat = seconds_per_bar / 4
    n = int(bars * seconds_per_bar * sr)
    out = np.zeros((n, 2))
    t_bar = np.arange(int(seconds_per_bar * sr)) / sr
    rng = np.random.default_rng(7)
    for b in range(bars):
        chord, i0 = prog[b % 4], int(b * seconds_per_bar * sr)
        env = np.clip(np.minimum(1, t_bar / 1.2) * np.minimum(1, (seconds_per_bar - t_bar) / 1.0 + 0.25), 0, 1)
        pad = sum(sum(np.sin(2 * np.pi * hz(m) * (1 + d) * t_bar + ph) for d, ph in
                      ((-0.003, 0), (0, 1.3), (0.003, 2.1))) + 0.25 * np.sin(2 * np.pi * 2 * hz(m) * t_bar)
                  for m in chord) * env * 0.05
        pan = np.array([0.9, 1.0]) if b % 2 else np.array([1.0, 0.9])
        end = min(n, i0 + len(t_bar))
        out[i0:end] += pad[: end - i0, None] * pan
        if pulse:  # soft low heartbeat on every beat
            for q in range(4):
                st = i0 + int(q * beat * sr)
                tt = np.arange(min(int(0.5 * sr), n - st)) / sr
                kick = np.sin(2 * np.pi * (hz(chord[0] - 12) + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9) * pulse
                out[st:st + len(tt)] += kick[:, None]
        for k in range(arps):  # soft plucked arpeggio
            m = chord[[1, 2, 3, 4, 3, 2, 3, 4][k % 8]] + 12
            st = i0 + int(k * seconds_per_bar / arps * sr)
            tt = np.arange(min(int(1.2 * sr), n - st)) / sr
            pl = np.sin(2 * np.pi * hz(m) * tt) * np.exp(-tt * 3.5) * 0.06 * (0.7 + 0.3 * rng.random())
            p = 0.3 + 0.4 * (k % 2)
            out[st:st + len(tt), 0] += pl * (1 - p)
            out[st:st + len(tt), 1] += pl * p
    d = int(0.33 * sr)
    for g, sh in ((0.35, d), (0.2, 2 * d), (0.1, 3 * d)):  # ping-pong echo
        out[sh:, 0] += g * out[:-sh, 1]
        out[sh:, 1] += g * out[:-sh, 0]
    xf = int(2 * sr)  # crossfade tail into head -> seamless loop
    tail = out[-xf:].copy()
    out = out[:-xf]
    w = np.linspace(0, 1, xf)[:, None]
    out[:xf] = out[:xf] * w + tail * (1 - w)
    out /= np.abs(out).max() * 1.12
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((out * 32767).astype("<i2").tobytes())



def make_music(path: Path, mood: str = "ambient") -> Path:
    """Create the loop as MP3 next to `path` (cached)."""
    if path.exists():
        return path
    wav = path.with_suffix(".wav")
    _synth_loop(wav, mood)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", "lowpass=f=9000",
                    "-c:a", "libmp3lame", "-b:a", "160k", str(path)], check=True)
    wav.unlink(missing_ok=True)
    return path
