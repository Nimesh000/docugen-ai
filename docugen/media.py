"""FFmpeg building blocks: narration timing, Ken Burns stills, motion clips, final mix."""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config
from .ffmpeg import run

FPS = config.FPS
W, H = config.WIDTH, config.HEIGHT
VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]
FADE = 0.4


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not read duration of {path.name}: {out.stderr[:300]}")


def frames(seconds: float) -> float:
    return max(1, round(seconds * FPS)) / FPS


def pad_narration(src: Path, dst: Path, lead: float = 0.35, tail: float = 0.6) -> float:
    """Resample to 48 kHz mono, add a little air before/after. Returns the exact (frame-aligned) length."""
    total = frames(duration(src) + lead + tail)
    run(["-i", src, "-af", f"aresample=48000,adelay={int(lead * 1000)}:all=1,apad",
         "-t", f"{total:.3f}", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", dst])
    return total


def silence(dst: Path, seconds: float) -> None:
    run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", dst])


_MOVES = [
    ("1+0.12*on/{n}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),            # slow push in
    ("1.12-0.12*on/{n}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),         # slow pull out
    ("1.12", "(iw-iw/zoom)*on/{n}", "ih/2-(ih/zoom/2)"),                  # pan left -> right
    ("1.12", "(iw-iw/zoom)*(1-on/{n})", "ih/2-(ih/zoom/2)"),              # pan right -> left
]


def ken_burns(image: Path, dst: Path, seconds: float, move: int, fade: bool = True) -> None:
    """Animate a still with a slow zoom or pan. The image is upscaled first so the motion is smooth."""
    n = round(seconds * FPS)
    z, x, y = (s.format(n=n) for s in _MOVES[move % len(_MOVES)])
    vf = (f"scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,"
          f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS},setsar=1")
    if fade:
        vf += f",fade=t=in:st=0:d={FADE},fade=t=out:st={seconds - FADE:.3f}:d={FADE}"
    run(["-i", image, "-vf", vf, "-frames:v", str(n), *VENC, "-an", dst])


def motion_clip(src: Path, dst: Path, seconds: float) -> None:
    """Fit a ~5 s generated clip to the scene: ping-pong loop, gentle slow-motion, then hold."""
    clip = duration(src)
    loop = clip * 2
    stretch = min(1.5, max(1.0, seconds / loop))
    hold = max(0.0, seconds - loop * stretch)
    fc = (f"[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0,setpts={stretch:.4f}*PTS,"
          f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},fps={FPS},setsar=1,"
          f"tpad=stop_mode=clone:stop_duration={hold + 0.1:.3f},"
          f"fade=t=in:st=0:d={FADE},fade=t=out:st={seconds - FADE:.3f}:d={FADE}[v]")
    run(["-i", src, "-filter_complex", fc, "-map", "[v]", "-frames:v", str(round(seconds * FPS)), *VENC, "-an", dst])


def concat_list(paths: list[Path], dst: Path) -> Path:
    dst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    return dst


def final_mix(work: Path, video_list: Path, audio_list: Path, music: Path | None, total: float,
              cards: list[tuple[float, float]], subs: Path | None, out_video: Path, out_audio: Path,
              on_progress=None, log=None) -> None:
    """Subtitles + narration + ducked music + loudness normalisation in one encode."""
    inputs = ["-f", "concat", "-safe", "0", "-i", str(video_list), "-f", "concat", "-safe", "0", "-i", str(audio_list)]
    fc = []
    vchain = "[0:v]setpts=PTS-STARTPTS"
    if subs is not None:
        vchain += f",subtitles={subs.name}:fontsdir=fonts"
    fc.append(f"{vchain},format=yuv420p[v]")
    loud = "loudnorm=I=-15:TP=-1.5:LRA=11,aresample=48000"
    if music is not None:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        bed, card, ramp = 0.07, 0.30, 0.8
        win = "+".join(f"clip((t-{a:.2f}+{ramp})/{ramp},0,1)*clip(({b:.2f}+{ramp}-t)/{ramp},0,1)" for a, b in cards)
        vol = f"{bed}+({card}-{bed})*min(1,{win})" if win else str(bed)
        fc += [
            "[1:a]asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=mono,asplit=2[nar][key]",
            "[nar]pan=stereo|c0=c0|c1=c0[nars]",
            "[key]pan=stereo|c0=c0|c1=c0[keys]",
            f"[2:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total:.3f},asetpts=PTS-STARTPTS,"
            f"volume='{vol}':eval=frame,afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, total - 3):.3f}:d=3[mus]",
            "[mus][keys]sidechaincompress=threshold=0.02:ratio=5:attack=30:release=500[duck]",
            f"[nars][duck]amix=inputs=2:duration=first:normalize=0,{loud},asplit=2[a1][a2]",
        ]
    else:
        fc.append(f"[1:a]asetpts=PTS-STARTPTS,pan=stereo|c0=c0|c1=c0,{loud},asplit=2[a1][a2]")
    run([*inputs, "-filter_complex", ";".join(fc),
         "-map", "[v]", "-map", "[a1]", "-t", f"{total:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_video),
         "-map", "[a2]", "-t", f"{total:.3f}", "-c:a", "pcm_s16le", str(out_audio)],
        cwd=work, total_seconds=total, on_progress=on_progress, log=log)
