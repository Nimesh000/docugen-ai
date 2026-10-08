import pytest

from docugen.script import normalize, parse_json, targets, to_markdown


def test_targets_scale_with_length():
    assert targets(60)[0] < targets(180)[0]
    assert targets(120)[1] == int(120 * 2.4)


def test_normalize_repairs_heroes_and_cleans_text():
    raw = {"title": "UPI", "logline": "x", "scenes": [
        {"narration": "Narrator: **India** changed   how it pays forever and ever.", "visual": "a market"},
        {"narration": "too short", "visual": "x"},
        {"narration": "Second scene with enough words here.", "visual": "a phone", "hero": True},
        {"narration": "Third scene with enough words here.", "visual": "a train"},
        {"narration": "Fourth scene with enough words here.", "visual": "a city"},
    ]}
    s = normalize(raw, "upi", heroes=2)
    assert len(s["scenes"]) == 4
    assert s["scenes"][0]["narration"].startswith("India changed how")
    assert sum(sc["hero"] for sc in s["scenes"]) == 2
    assert [sc["id"] for sc in s["scenes"]] == [0, 1, 2, 3]
    assert "Scene 1" in to_markdown(s)


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        normalize({"scenes": [{"narration": "hi", "visual": ""}]}, "t", 1)
    with pytest.raises(ValueError):
        normalize({"nothing": 1}, "t", 1)


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 2}') == {"a": 2}
