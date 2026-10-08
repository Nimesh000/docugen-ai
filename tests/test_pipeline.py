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
