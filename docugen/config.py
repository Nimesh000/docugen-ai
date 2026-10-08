"""Static settings: models, visual styles, narrator voices, pacing and safety limits."""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- models
LLM_MODEL = os.environ.get("DOCUGEN_LLM_MODEL", "openai/gpt-oss-120b")   # Groq (falls back automatically, see script.py)
IMAGE_MODEL = os.environ.get("DOCUGEN_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")            # 8-step photoreal DiT
VIDEO_MODEL = os.environ.get("DOCUGEN_VIDEO_MODEL", "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers")  # 4-step I2V
TTS_REPO = "hexgrad/Kokoro-82M"                                                              # 82M, faster than real time on CPU

# --------------------------------------------------------------------------- output
WIDTH, HEIGHT, FPS = 1920, 1080, 30
IMAGE_W, IMAGE_H = 1536, 864           # 16:9, multiples of 16
IMAGE_STEPS = 9                        # Z-Image-Turbo: 9 steps = 8 DiT forwards
VIDEO_W, VIDEO_H = 960, 544            # Wan 2.2: multiples of 16
VIDEO_FRAMES, VIDEO_FPS = 81, 16       # ~5 s clip
VIDEO_STEPS = 4                        # Lightning distillation
MOTION_SECONDS = VIDEO_FRAMES / VIDEO_FPS

# --------------------------------------------------------------------------- pacing
WORDS_PER_SECOND = 2.6                 # Kokoro at SPEECH_SPEED reads ~155 words a minute
SPEECH_SPEED = 1.05
SHOT_SECONDS = 3.2                     # target length of one still shot -> a new picture every ~3 s
MAX_SHOTS_PER_SCENE = 5
XFADE = 0.3                            # cross-dissolve between shots
CARD_SECONDS = 2.6                     # title and end cards

# --------------------------------------------------------------------------- choices shown in the UI
STYLES = {
    "cinematic": {
        "label": "Cinematic photo",
        "prompt": "cinematic documentary photograph, shot on Arri Alexa, 35mm lens, natural light, rich color grading, "
                  "sharp focus, highly detailed, realistic textures, professional photography",
    },
    "archival": {
        "label": "Archival film",
        "prompt": "vintage archival documentary photograph, 1970s kodachrome film, film grain, faded colors, "
                  "authentic historical photo, realistic",
    },
    "painterly": {
        "label": "Painted illustration",
        "prompt": "detailed digital painting, documentary illustration, painterly brush strokes, dramatic lighting, "
                  "rich textures, concept art",
    },
    "noir": {
        "label": "Dark & mysterious",
        "prompt": "moody low-key documentary photograph, dramatic chiaroscuro lighting, deep shadows, volumetric haze, "
                  "cinematic, realistic",
    },
}
NEGATIVE_PROMPT = "text, watermark, logo, letters, blurry, low quality, deformed, cartoon"

VOICES = {  # Kokoro-82M voices
    "bm_george": "Deep British male (documentary)",
    "am_michael": "American male",
    "am_onyx": "Deep American male",
    "bf_emma": "British female",
    "af_heart": "Warm American female",
}
DEFAULT_VOICE = "bm_george"

LENGTHS = {30: "30 s", 60: "1 min", 90: "1.5 min", 120: "2 min"}
MIN_SCENES, MAX_SCENES = 3, 12
MAX_HEROES = 6

# --------------------------------------------------------------------------- public-demo guard rails
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "10"))
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
MAX_TOPIC_CHARS = 200


def default_scenes(seconds: int) -> int:
    return max(MIN_SCENES, min(MAX_SCENES, round(seconds / 10)))
