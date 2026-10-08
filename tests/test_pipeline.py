"""End-to-end pipeline run with the fake GPU backend (exercises all FFmpeg editing for real)."""
import json
import subprocess

from docugen import pipeline
from tests.fakes import FakeBackend


def test_full_pipeline(tmp_path):
    params = {"topic": "UPI", "seconds": 60, "style": "cinematic", "voice": "Damien Black", "motion": 2,
              "music": True, "subtitles": True}
    store = {}
    status = pipeline.Status(store, "job1", pipeline.new_state("job1", params))
    ok = pipeline.run(params, tmp_path, FakeBackend(), status)
    state = store["job1"]
    assert ok, state["error"]
    assert state["status"] == "done"
    out = tmp_path / "output"
    for name in ("documentary.mp4", "documentary_audio.wav", "narration.wav", "subtitles.srt", "thumbnail.jpg",
                 "script.md", "script.json"):
        assert (out / name).exists(), name
    # one hero animated, one fell back to Ken Burns
    assert sum(1 for s in state["scenes"] if s["motion"]) == 1
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json",
         str(out / "documentary.mp4")], capture_output=True, text=True).stdout)
    durs = {s["codec_type"]: float(s["duration"]) for s in probe["streams"]}
    assert abs(durs["video"] - durs["audio"]) < 0.15
    assert abs(durs["video"] - state["duration"]) < 0.2
