"""FFmpeg building blocks: narration timing, Ken Burns stills, motion clips, final mix."""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config
from .ffmpeg import run

FPS = config.FPS
LONG = config.FORMATS["long"]
VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]
GROUP = 18                   # clips per intermediate file when a film has many shots (keeps FFmpeg memory flat)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not read duration of {path.name}: {out.stderr[:300]}")


def frames(seconds: float) -> float:
    return max(1, round(seconds * FPS)) / FPS


def pad_narration(src: Path, dst: Path, lead: float = 0.12, tail: float = 0.28) -> float:
    """Resample to 48 kHz mono, add a little air before/after. Returns the exact (frame-aligned) length."""
    total = frames(duration(src) + lead + tail)
    run(["-i", src, "-af", f"aresample=48000,adelay={int(lead * 1000)}:all=1,apad",
         "-t", f"{total:.3f}", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", dst])
    return total


def silence(dst: Path, seconds: float) -> None:
    run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", dst])


# Ken Burns moves: (zoom start, zoom end, x start, x end, y start, y end); x/y are the crop's position as a
# fraction of the free space (0 = left/top, 1 = right/bottom). Eased with a smooth S-curve.
_MOVES = [
    (1.00, 1.22, 0.5, 0.5, 0.5, 0.45),   # push in
    (1.22, 1.02, 0.5, 0.5, 0.45, 0.5),   # pull back
    (1.18, 1.18, 0.0, 1.0, 0.5, 0.5),    # pan left -> right
    (1.05, 1.25, 0.3, 0.7, 0.6, 0.4),    # diagonal push
    (1.18, 1.18, 1.0, 0.0, 0.5, 0.5),    # pan right -> left
    (1.20, 1.20, 0.5, 0.5, 1.0, 0.1),    # tilt up
    (1.25, 1.05, 0.7, 0.4, 0.3, 0.5),    # diagonal pull back
]


def ken_burns(image: Path, dst: Path, seconds: float, move: int, size=LONG["size"], kb=LONG["kb"]) -> None:
    """Animate a still with a clearly visible, eased zoom / pan (supersampled canvas -> sub-pixel smooth motion)."""
    W, H = size
    KB_W, KB_H = kb
    n = max(2, round(seconds * FPS))
    z0, z1, x0, x1, y0, y1 = (1.0, 1.05, 0.5, 0.5, 0.5, 0.5) if move < 0 else _MOVES[move % len(_MOVES)]
    e = f"(0.5-0.5*cos(PI*on/{n - 1}))"
    z = f"{z0}+({z1 - z0})*{e}"
    x = f"(iw-iw/zoom)*({x0}+({x1 - x0})*{e})"
    y = f"(ih-ih/zoom)*({y0}+({y1 - y0})*{e})"
    vf = (f"scale={KB_W}:{KB_H}:force_original_aspect_ratio=increase:flags=bicubic,crop={KB_W}:{KB_H},"
          f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS},setsar=1")
    run(["-i", image, "-vf", vf, "-frames:v", str(n), *VENC, "-an", dst])


def motion_clip(src: Path, dst: Path, seconds: float, size=LONG["size"]) -> None:
    """Fit a generated clip to its slot: gentle speed change (at most 1.35x slower), upscale, smooth 30 fps."""
    W, H = size
    clip = duration(src)
    stretch = min(1.35, max(0.85, seconds / clip))
    hold = max(0.0, seconds - clip * stretch)
    vf = (f"setpts={stretch:.4f}*PTS,scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={W}:{H},unsharp=5:5:0.6,framerate=fps={FPS},setsar=1,"
          f"tpad=stop_mode=clone:stop_duration={hold + 0.2:.3f}")
    run(["-i", src, "-vf", vf, "-frames:v", str(round(seconds * FPS)), *VENC, "-an", dst])


def xfade_chain(clips: list[tuple[Path, float]], xfade: float) -> tuple[list[str], str]:
    """Inputs + filter for joining clips with cross-dissolves. clips = [(path, display seconds)]; every clip except
    the last must be `xfade` seconds longer than its display time, so the result lasts exactly sum(display)."""
    inputs = [a for p, _ in clips for a in ("-i", str(p))]
    if len(clips) == 1:
        return inputs, "[0:v]null[vx]"
    fc, prev, offset = [], "[0:v]", 0.0
    for i in range(1, len(clips)):
        offset += clips[i - 1][1]
        out = f"[x{i}]" if i < len(clips) - 1 else "[vx]"
        fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={xfade}:offset={offset:.4f}{out}")
        prev = out
    return inputs, ";".join(fc)


def group_clips(work: Path, clips: list[tuple[Path, float]], xfade: float, size: int = GROUP,
                threshold: int = 24) -> list[tuple[Path, float]]:
    """Long films have 60+ shots. Join them in groups first so the final graph has few inputs. A group behaves
    exactly like one clip: its display time is the sum of its clips and it keeps the `xfade` tail for the next join."""
    if len(clips) <= threshold:
        return clips
    out = []
    for g, start in enumerate(range(0, len(clips), size)):
        part = clips[start:start + size]
        last_overall = start + size >= len(clips)
        inputs, join = xfade_chain(part, xfade)
        display = sum(d for _, d in part)
        length = display + (0 if last_overall else xfade)
        dst = work / f"group_{g:02d}.mp4"
        run([*inputs, "-filter_complex", join + ";[vx]setpts=PTS-STARTPTS[v]", "-map", "[v]", "-t", f"{length:.4f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", dst])
        out.append((dst, display))
    return out


def concat_list(paths: list[Path], dst: Path) -> Path:
    dst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    return dst


def final_mix(work: Path, clips: list[tuple[Path, float]], xfade: float, audio_list: Path, music: Path | None,
              total: float, cards: list[tuple[float, float]], subs: Path | None, out_video: Path, out_audio: Path,
              on_progress=None, log=None) -> None:
    """One encode: cross-dissolve join + captions + narration + ducked music + loudness normalisation."""
    clips = group_clips(work, clips, xfade)
    inputs, join = xfade_chain(clips, xfade)
    na = len(clips)                      # input index of the narration
    inputs += ["-f", "concat", "-safe", "0", "-i", str(audio_list)]
    fc = [join]
    vchain = "[vx]setpts=PTS-STARTPTS"
    if subs is not None:
        vchain += f",subtitles={subs.name}:fontsdir=fonts"
    fc.append(f"{vchain},format=yuv420p[v]")
    loud = "loudnorm=I=-15:TP=-1.5:LRA=11,aresample=48000"
    if music is not None:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        bed, card, ramp = 0.08, 0.30, 0.8
        win = "+".join(f"clip((t-{a:.2f}+{ramp})/{ramp},0,1)*clip(({b:.2f}+{ramp}-t)/{ramp},0,1)" for a, b in cards)
        vol = f"{bed}+({card}-{bed})*min(1,{win})" if win else str(bed)
        fc += [
            f"[{na}:a]asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=mono,asplit=2[nar][key]",
            "[nar]pan=stereo|c0=c0|c1=c0[nars]",
            "[key]pan=stereo|c0=c0|c1=c0[keys]",
            f"[{na + 1}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total:.3f},asetpts=PTS-STARTPTS,"
            f"volume='{vol}':eval=frame,afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, total - 2.5):.3f}:d=2.5[mus]",
            "[mus][keys]sidechaincompress=threshold=0.02:ratio=5:attack=30:release=400[duck]",
            f"[nars][duck]amix=inputs=2:duration=first:normalize=0,{loud},asplit=2[a1][a2]",
        ]
    else:
        fc.append(f"[{na}:a]asetpts=PTS-STARTPTS,pan=stereo|c0=c0|c1=c0,{loud},asplit=2[a1][a2]")
    run([*inputs, "-filter_complex", ";".join(fc),
         "-map", "[v]", "-map", "[a1]", "-t", f"{total:.3f}", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_video),
         "-map", "[a2]", "-t", f"{total:.3f}", "-c:a", "pcm_s16le", str(out_audio)],
        cwd=work, total_seconds=total, on_progress=on_progress, log=log)


def run_concat_audio(audio_list: Path, dst: Path) -> None:
    run(["-f", "concat", "-safe", "0", "-i", str(audio_list), "-c:a", "pcm_s16le", str(dst)])
