import pytest

from docugen.script import normalize, parse_json, targets, to_markdown


def test_targets_follow_scene_choice():
    assert targets(60)[0] < targets(120)[0]
    assert targets(60, scenes=8)[0] == 8
    assert targets(60, scenes=99)[0] == 20
    assert targets(30, scenes=99, format="reel")[0] == 8
    assert targets(30, format="reel")[1] < targets(30)[1] + 10
    assert targets(120)[1] > targets(60)[1]


def test_normalize_repairs_heroes_and_cleans_text():
    raw = {"title": "UPI", "logline": "x", "scenes": [
        {"narration": "Narrator: **India** changed   how it pays forever and ever.", "visual": "a market"},
        {"narration": "too short", "visual": "x"},
        {"narration": "Second scene with enough words here.", "shots": ["a phone", "a hand"], "hero": True},
        {"narration": "Third scene with enough words here.", "visual": "a train"},
        {"narration": "Fourth scene with enough words here.", "visual": "a city"},
    ]}
    s = normalize(raw, "upi", heroes=2)
    assert len(s["scenes"]) == 4
    assert s["scenes"][0]["narration"].startswith("India changed how")
    assert sum(sc["hero"] for sc in s["scenes"]) == 2
    assert [sc["id"] for sc in s["scenes"]] == [0, 1, 2, 3]
    assert s["scenes"][1]["shots"] == ["a phone", "a hand"] and s["scenes"][0]["shots"] == ["a market"]
    assert "Scene 1" in to_markdown(s)


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        normalize({"scenes": [{"narration": "hi", "visual": ""}]}, "t", 1)
    with pytest.raises(ValueError):
        normalize({"nothing": 1}, "t", 1)


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 2}') == {"a": 2}
