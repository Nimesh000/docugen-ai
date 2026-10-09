"""Title card, end card and thumbnail (Pillow)."""
from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [os.environ.get("DOCUGEN_FONT_DIR", "/root/fonts"), "/usr/share/fonts/opentype/inter",
             "/usr/share/fonts/truetype/dejavu", r"C:\Windows\Fonts"]
FONT_FILES = {
    "Bold": ["Inter-Bold.otf", "DejaVuSans-Bold.ttf", "segoeuib.ttf"],
    "Regular": ["Inter-Regular.otf", "DejaVuSans.ttf", "segoeui.ttf"],
}
ACCENT = (232, 176, 75)
BG = (10, 12, 16)


def font_path(weight: str) -> str | None:
    for d in FONT_DIRS:
        for name in FONT_FILES[weight]:
            p = Path(d) / name
            try:
                if p.is_file():
                    return str(p)
            except OSError:  # e.g. /root not readable for non-root users
                continue
    return None


def font_family() -> str:
    p = font_path("Bold") or ""
    return "Inter" if "Inter" in p else "DejaVu Sans" if "DejaVu" in p else "Segoe UI"


def _font(weight: str, size: int):
    p = font_path(weight)
    return ImageFont.truetype(p, size) if p else ImageFont.load_default(size=size)


def _wrap(d: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 3) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def _backdrop(bg_png: bytes | None, w: int, h: int, darken: float = 0.55) -> Image.Image:
    if bg_png:
        img = Image.open(io.BytesIO(bg_png)).convert("RGB")
        img = img.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(h // 90))
        return Image.blend(img, Image.new("RGB", (w, h), BG), darken)
    return Image.new("RGB", (w, h), BG)


def title_card(path: Path, w: int, h: int, title: str, logline: str, bg_png: bytes | None = None) -> None:
    img = _backdrop(bg_png, w, h, 0.6)
    d = ImageDraw.Draw(img)
    s = min(w, h) / 1080
    tf = _font("Bold", int(96 * s))
    lines = _wrap(d, title.upper(), tf, int(w * 0.84), 4 if h > w else 3)
    lh = int(112 * s)
    y = (h - lh * len(lines)) // 2 - int(40 * s)
    d.text((w // 2, y - int(60 * s)), "A  D O C U G E N  F I L M", font=_font("Regular", int(30 * s)),
           fill=ACCENT, anchor="mm")
    for line in lines:
        d.text((w // 2, y + lh // 2), line, font=tf, fill=(245, 245, 245), anchor="mm")
        y += lh
    d.rectangle((w // 2 - int(70 * s), y + int(20 * s), w // 2 + int(70 * s), y + int(26 * s)), fill=ACCENT)
    if logline:
        lf = _font("Regular", int(38 * s))
        for i, line in enumerate(_wrap(d, logline, lf, int(w * (0.8 if h > w else 0.62)), 3 if h > w else 2)):
            d.text((w // 2, y + int(80 * s) + i * int(52 * s)), line, font=lf, fill=(200, 200, 205), anchor="mm")
    img.save(path)


def end_card(path: Path, w: int, h: int, bg_png: bytes | None = None) -> None:
    img = _backdrop(bg_png, w, h, 0.75)
    d = ImageDraw.Draw(img)
    s = min(w, h) / 1080
    d.text((w // 2, h // 2 - int(30 * s)), "Thanks for watching", font=_font("Bold", int(84 * s)),
           fill=(245, 245, 245), anchor="mm")
    d.rectangle((w // 2 - int(70 * s), h // 2 + int(15 * s), w // 2 + int(70 * s), h // 2 + int(21 * s)), fill=ACCENT)
    sf = _font("Regular", int(34 * s))
    for i, line in enumerate(_wrap(d, "Written, voiced, illustrated and animated by open-source AI models", sf,
                                   int(w * 0.84), 2)):
        d.text((w // 2, h // 2 + int(70 * s) + i * int(46 * s)), line, font=sf, fill=(190, 190, 196), anchor="mm")
    img.save(path)


def thumbnail(bg_png: bytes, path: Path, title: str, size: tuple[int, int] = (1280, 720)) -> None:
    tw, th = size
    img = Image.open(io.BytesIO(bg_png)).convert("RGB")
    sc = max(tw / img.width, th / img.height)
    img = img.resize((round(img.width * sc), round(img.height * sc)), Image.LANCZOS)
    left, top = (img.width - tw) // 2, (img.height - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    band = int(th * 0.56)
    for i in range(band):
        sd.line((0, th - band + i, tw, th - band + i), fill=(0, 0, 0, int(210 * (i / band) ** 1.3)))
    img = Image.alpha_composite(img.convert("RGBA"), shade).convert("RGB")
    d = ImageDraw.Draw(img)
    fs = int(68 * min(tw, th) / 720)
    f = _font("Bold", fs)
    lines = _wrap(d, title.upper(), f, tw - 120, 3 if th > tw else 2)
    y = th - int(th * 0.1) - int(fs * 1.18) * len(lines)
    d.rectangle((60, y - 28, 170, y - 20), fill=ACCENT)
    for line in lines:
        d.text((60, y), line, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        y += int(fs * 1.18)
    img.save(path, quality=92)
