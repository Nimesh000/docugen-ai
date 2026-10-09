"""End-to-end pipeline run with the fake GPU backend (exercises all FFmpeg editing for real)."""
import json
import subprocess

from docugen import config, pipeline
from tests.fakes import FakeBackend


def probe(path):
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
        capture_output=True, text=True).stdout)
    return {s["codec_type"]: float(s["duration"]) for s in data["streams"]}


def test_plan_scene_cuts_into_short_shots():
    shots = pipeline.plan_scene(10.0, hero=False)
    assert len(shots) == 3 and abs(sum(s["seconds"] for s in shots) - 10.0) < 1e-6
    hero = pipeline.plan_scene(11.0, hero=True)
    assert hero[0]["kind"] == "motion" and abs(hero[0]["seconds"] - config.MOTION_SECONDS) < 1e-6
    assert all(s["kind"] == "still" for s in hero[1:])
    short = pipeline.plan_scene(6.0, hero=True)       # too short for a still after the clip
    assert len(short) == 1 and short[0]["seconds"] == 6.0


def test_plan_parameters_clamps():
    assert pipeline.plan_parameters({"seconds": 60, "scenes": 99, "motion": 50}) == (60, config.MAX_SCENES,
                                                                                       config.MAX_HEROES)
    assert pipeline.plan_parameters({"seconds": 30, "scenes": 3, "motion": 5}) == (30, 3, 3)


def test_full_pipeline(tmp_path):
    params = {"topic": "UPI", "seconds": 30, "scenes": 4, "style": "cinematic", "voice": "bm_george", "motion": 2,
              "music": True, "subtitles": True}
    store = {}
    backend = FakeBackend()
    status = pipeline.Status(store, "job1", pipeline.new_state("job1", params))
    ok = pipeline.run(params, tmp_path, backend, status)
    state = store["job1"]
    assert ok, state["error"]
    assert backend.prewarmed[1] == 2
    out = tmp_path / "output"
    for name in ("documentary.mp4", "documentary_audio.wav", "narration.wav", "subtitles.srt", "thumbnail.jpg",
                 "script.md", "script.json"):
        assert (out / name).exists(), name
    assert len(state["scenes"]) == 4
    # one hero animated, one fell back to Ken Burns
    assert sum(1 for s in state["scenes"] if s["motion"]) == 1
    assert sum(len(s["shots"]) for s in state["scenes"]) > 4          # several shots per scene
    durs = probe(out / "documentary.mp4")
    assert abs(durs["video"] - durs["audio"]) < 0.15
    assert abs(durs["video"] - state["duration"]) < 0.2


def test_reel_pipeline_is_vertical_with_hook(tmp_path):
    params = {"topic": "UPI", "format": "reel", "seconds": 15, "scenes": 3, "motion": 1, "captions": "pop",
              "music": "uplifting", "tone": "dramatic"}
    store = {}
    backend = FakeBackend(fail_motion_index=None)
    status = pipeline.Status(store, "reel1", pipeline.new_state("reel1", params))
    ok = pipeline.run(params, tmp_path, backend, status)
    state = store["reel1"]
    assert ok, state["error"]
    assert backend.image_size == (864, 1536)
    assert all("director shot" in p for p in backend.prompts)            # Visual Director prompts were used
    assert "vertical" in backend.prompts[0]
    assert backend.brief["format"] == "reel" and backend.brief["tone"] == "dramatic"
    data = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                      "stream=width,height", "-of", "json", str(tmp_path / "output/documentary.mp4")],
                                     capture_output=True, text=True).stdout)["streams"][0]
    assert (data["width"], data["height"]) == (1080, 1920)
    ass = (tmp_path / "work/subs.ass").read_text()
    assert "Hook,," in ass and "\\c&H4BB0E8&" in ass                     # title hook + word highlight
    assert not (tmp_path / "work/title.png").exists()                     # reels start straight on the story
    assert state["fact_check"] == ["Corrected a launch year"]
    assert state["cost"]["usd"] > 0 and state["estimate"]["usd"] > 0
    assert state["director"]["bible"]["places"] == ["Mumbai"]


def test_group_clips_keeps_duration(tmp_path):
    from docugen import media

    clips = []
    for k in range(7):
        p = tmp_path / f"c{k}.mp4"
        dur = 1.0 + (0.3 if k < 6 else 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{k * 30:02x}4060:s=320x180:d={dur}",
                        "-r", "30", "-pix_fmt", "yuv420p", str(p)], check=True)
        clips.append((p, 1.0))
    grouped = media.group_clips(tmp_path, clips, 0.3, size=3, threshold=0)
    assert len(grouped) == 3 and [d for _, d in grouped] == [3.0, 3.0, 1.0]
    inputs, join = media.xfade_chain(grouped, 0.3)
    out = tmp_path / "joined.mp4"
    media.run([*inputs, "-filter_complex", join, "-map", "[vx]", "-pix_fmt", "yuv420p", out])
    assert abs(media.duration(out) - 7.0) < 0.1
