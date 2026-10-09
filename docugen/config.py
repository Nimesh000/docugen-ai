"""Static settings: models, formats, visual styles, voices, pacing, prices and safety limits."""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- models
LLM_MODEL = os.environ.get("DOCUGEN_LLM_MODEL", "openai/gpt-oss-120b")   # Groq (falls back automatically, see script.py)
IMAGE_MODEL = os.environ.get("DOCUGEN_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")            # 8-step photoreal DiT
VIDEO_MODEL = os.environ.get("DOCUGEN_VIDEO_MODEL", "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers")  # 4-step I2V
TTS_REPO = "hexgrad/Kokoro-82M"                                                              # 82M params, runs on a T4

# --------------------------------------------------------------------------- output formats
FPS = 30
IMAGE_STEPS = 9                        # Z-Image-Turbo: 9 steps = 8 DiT forwards
VIDEO_FRAMES, VIDEO_FPS = 81, 16       # ~5 s clip
VIDEO_STEPS = 4                        # Lightning distillation
MOTION_SECONDS = VIDEO_FRAMES / VIDEO_FPS

FORMATS = {
    "long": {
        "label": "Long video", "aspect": "16:9", "hint": "YouTube, presentations, 1 to 5 minutes",
        "size": (1920, 1080), "image": (1536, 864), "video": (960, 544), "kb": (3200, 1800),
        "lengths": {60: "1 min", 120: "2 min", 180: "3 min", 300: "5 min"}, "default_seconds": 60,
        "shot_seconds": 4.0, "title_card": 2.6, "end_card": 2.6,
        "scenes": (3, 20), "seconds_per_scene": 12, "max_heroes": 6, "captions": "clean",
        "framing": "wide 16:9 cinematic composition",
    },
    "reel": {
        "label": "Reel / Short", "aspect": "9:16", "hint": "Instagram Reels, YouTube Shorts, 15 to 60 seconds",
        "size": (1080, 1920), "image": (864, 1536), "video": (544, 960), "kb": (1800, 3200),
        "lengths": {15: "15 s", 30: "30 s", 45: "45 s", 60: "60 s"}, "default_seconds": 30,
        "shot_seconds": 2.4, "title_card": 0.0, "end_card": 1.4,
        "scenes": (3, 8), "seconds_per_scene": 6, "max_heroes": 4, "captions": "pop",
        "framing": "tall vertical 9:16 composition, main subject centred in the middle of the frame",
    },
}
DEFAULT_FORMAT = "long"
MAX_SHOTS_PER_SCENE = 8
XFADE = 0.3                            # cross-dissolve between shots

# --------------------------------------------------------------------------- narration pacing
WORDS_PER_SECOND = 2.4                 # Kokoro (bm_george) at SPEECH_SPEED with sentence gaps trimmed
SPEECH_SPEED = 1.15
MAX_SPEECH_SPEED = 1.35                # narration is re-voiced faster (up to this) if it overshoots the length

# --------------------------------------------------------------------------- choices shown in the UI
STYLES = {
    "cinematic": {
        "label": "Cinematic photo",
        "prompt": "cinematic documentary photograph, natural light, rich color grading, shallow depth of field, "
                  "sharp focus, highly detailed, realistic textures, professional photography",
    },
    "golden": {
        "label": "Golden hour",
        "prompt": "documentary photograph at golden hour, warm low sunlight, long soft shadows, glowing rim light, "
                  "rich warm tones, realistic, highly detailed",
    },
    "archival": {
        "label": "Archival film",
        "prompt": "vintage archival documentary photograph, 1970s kodachrome film, film grain, faded colors, "
                  "authentic historical photo, realistic",
    },
    "monochrome": {
        "label": "Black & white",
        "prompt": "black and white documentary photograph, high contrast, deep blacks, fine film grain, "
                  "classic photojournalism, realistic",
    },
    "noir": {
        "label": "Dark & mysterious",
        "prompt": "moody low-key documentary photograph, dramatic chiaroscuro lighting, deep shadows, volumetric haze, "
                  "cinematic, realistic",
    },
    "painterly": {
        "label": "Painted illustration",
        "prompt": "detailed digital painting, documentary illustration, painterly brush strokes, dramatic lighting, "
                  "rich textures, concept art",
    },
    "watercolor": {
        "label": "Watercolor",
        "prompt": "delicate watercolor illustration, soft washes of color, visible paper texture, loose ink outlines, "
                  "gentle light, hand painted",
    },
    "animated3d": {
        "label": "3D animation",
        "prompt": "stylised 3D animated film still, soft global illumination, expressive characters, vibrant colors, "
                  "clean shapes, highly detailed render",
    },
}
NEGATIVE_PROMPT = "text, watermark, logo, letters, blurry, low quality, deformed, cartoon"
CLEAN_SUFFIX = ("no visible text, no lettering, no numbers, no paper documents, no signage, no logos, "
                "no cameras or film equipment in frame")

VOICES = {  # Kokoro-82M voices
    "bm_george": "Deep British male (documentary)",
    "bm_lewis": "Calm British male",
    "am_michael": "American male",
    "am_onyx": "Deep American male",
    "bf_emma": "British female",
    "bf_isabella": "Soft British female",
    "af_heart": "Warm American female",
    "af_bella": "Bright American female",
}
DEFAULT_VOICE = "bm_george"

TONES = {
    "informative": ("Informative", "clear, curious and factual, like a good explainer"),
    "dramatic": ("Dramatic", "cinematic and suspenseful, with rising stakes and turning points"),
    "inspiring": ("Inspiring", "uplifting and human, focused on people and what they achieved"),
    "mysterious": ("Mysterious", "intriguing, posing questions and revealing answers step by step"),
    "playful": ("Fun & light", "light, witty and energetic, still accurate"),
}
AUDIENCES = {
    "general": ("Everyone", "a general audience"),
    "students": ("Students", "college students; explain terms briefly"),
    "kids": ("Kids", "children aged 8 to 12; simple words, short sentences, nothing scary"),
    "professionals": ("Professionals", "busy professionals; dense with useful facts and numbers"),
}
PACING = {"calm": ("Calm", 1.3), "balanced": ("Balanced", 1.0), "fast": ("Fast cuts", 0.75)}
CAPTIONS = {"pop": "Bold word-by-word", "clean": "Clean subtitles", "none": "No captions"}
MUSIC = {"ambient": "Ambient", "uplifting": "Uplifting", "tense": "Tense", "none": "No music"}
RENDER_MODES = {   # how many GPU containers may run at once: fewer = fewer model loads = cheaper
    "economy": {"label": "Economy", "hint": "Lowest cost, a little slower", "image_gpus": 2, "motion_gpus": 1},
    "fast": {"label": "Fast", "hint": "More GPUs in parallel, costs more", "image_gpus": 4, "motion_gpus": 4},
}
MAX_KEY_POINTS_CHARS = 400

# --------------------------------------------------------------------------- public-demo guard rails
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "10"))
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
MAX_TOPIC_CHARS = 200
ALL_LENGTHS = sorted({s for f in FORMATS.values() for s in f["lengths"]})
MIN_SCENES = min(f["scenes"][0] for f in FORMATS.values())
MAX_SCENES = max(f["scenes"][1] for f in FORMATS.values())
MAX_HEROES = max(f["max_heroes"] for f in FORMATS.values())


def fmt(name: str | None) -> dict:
    return FORMATS.get(name or DEFAULT_FORMAT, FORMATS[DEFAULT_FORMAT])


def default_scenes(seconds: int, format: str | None = None) -> int:
    f = fmt(format)
    lo, hi = f["scenes"]
    return max(lo, min(hi, round(seconds / f["seconds_per_scene"])))


def shot_seconds(format: str | None, pacing: str | None) -> float:
    return fmt(format)["shot_seconds"] * PACING.get(pacing or "balanced", PACING["balanced"])[1]


def narration_seconds(seconds: int, format: str | None = None) -> float:
    f = fmt(format)
    return max(8.0, seconds - f["title_card"] - f["end_card"])
