"""Static settings: visual styles, narrator voices, models and safety limits."""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- models
LLM_MODEL = os.environ.get("DOCUGEN_LLM_MODEL", "llama-3.3-70b-versatile")   # Groq
IMAGE_MODEL = os.environ.get("DOCUGEN_IMAGE_MODEL", "RunDiffusion/Juggernaut-XL-v9")
VIDEO_MODEL = os.environ.get("DOCUGEN_VIDEO_MODEL", "Lightricks/LTX-Video")
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
WHISPER_MODEL = os.environ.get("DOCUGEN_WHISPER_MODEL", "base")

# --------------------------------------------------------------------------- output
WIDTH, HEIGHT, FPS = 1920, 1080, 30
IMAGE_W, IMAGE_H = 1344, 768          # SDXL 16:9 bucket
VIDEO_W, VIDEO_H, VIDEO_FRAMES = 768, 448, 121   # LTX: multiples of 32, 8k+1 frames (~5 s @ 24 fps)
WORDS_PER_SECOND = 2.4                # calm documentary narration pace

# --------------------------------------------------------------------------- choices shown in the UI
STYLES = {
    "cinematic": {
        "label": "Cinematic photo",
        "prompt": "cinematic documentary photograph, 35mm film, natural light, shallow depth of field, "
                  "rich color grading, highly detailed, award-winning photography",
    },
    "archival": {
        "label": "Archival film",
        "prompt": "vintage archival documentary footage still, 1970s film grain, faded kodachrome colors, "
                  "authentic historical photograph, soft focus",
    },
    "painterly": {
        "label": "Painted illustration",
        "prompt": "detailed digital painting, documentary illustration, painterly brush strokes, "
                  "dramatic lighting, concept art, rich textures",
    },
    "noir": {
        "label": "Dark & mysterious",
        "prompt": "moody low-key documentary photograph, dramatic chiaroscuro lighting, deep shadows, "
                  "volumetric fog, cinematic, mysterious atmosphere",
    },
}
NEGATIVE_PROMPT = ("text, watermark, logo, caption, letters, signature, blurry, low quality, deformed, "
                   "disfigured, extra fingers, bad anatomy, cartoon, oversaturated, frame, border")

VOICES = {  # XTTS-v2 built-in studio speakers
    "Damien Black": "Deep male narrator",
    "Viktor Eka": "Calm male narrator",
    "Claribel Dervla": "Warm female narrator",
    "Ana Florence": "Clear female narrator",
}
DEFAULT_VOICE = "Damien Black"

LENGTHS = {60: "~1 minute", 120: "~2 minutes", 180: "~3 minutes"}

# --------------------------------------------------------------------------- public-demo guard rails
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "10"))
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
MAX_TOPIC_CHARS = 200
