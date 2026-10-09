from docugen import config, cost, director, pipeline


SCRIPT = {"title": "UPI", "logline": "x", "scenes": [
    {"narration": "In 2016 India launched a new way to pay. " * 3, "shots": ["a market"], "visual": "a market",
     "motion": "walk", "hero": True, "id": 0},
    {"narration": "Street vendors adopted it within a few years across the country. " * 2, "shots": ["a vendor"],
     "visual": "a vendor", "motion": "", "hero": False, "id": 1},
]}


def test_lint_and_sanitize():
    good = ("Low-angle medium shot of a tea vendor in a white kurta pouring chai at a busy Mumbai street stall, "
            "India, 2019, steam rising, commuters blurred behind him, warm morning light, 35mm lens, shallow depth "
            "of field, cheerful mood")
    assert director.lint(good) == []
    bad = "A phone screen showing a payment and a poster with text"
    problems = director.lint(bad)
    assert any("render as text" in p for p in problems) and any("short" in p for p in problems)
    clean = director.sanitize(bad)
    assert not director.FORBIDDEN.search(clean) and "seen from behind" in clean


def test_plan_counts_follow_format():
    long = director.plan_counts(SCRIPT, "long", "balanced", pipeline.plan_scene)
    fast = director.plan_counts(SCRIPT, "reel", "fast", pipeline.plan_scene)
    assert all(c >= 2 for c in long) and sum(fast) > sum(long)


def test_direct_with_fake_llm(monkeypatch):
    calls = []

    def fake_chat(messages, log=print, **kw):
        calls.append(messages[0]["content"][:30])
        if "critic" in messages[0]["content"]:
            return {"fixes": [{"id": "1.0", "prompt": "Wide aerial shot of a crowded vegetable market in Pune, India, "
                                                      "2020, hundreds of shoppers between stalls, golden evening light, "
                                                      "drone view, deep focus, lively mood, palette of green and saffron"}]}
        return {"bible": {"era": "2016-2024", "places": ["India"]}, "scenes": [
            {"scene": 1, "shots": ["Wide establishing shot of a busy Mumbai market, India, 2016, shoppers and stalls, "
                                   "morning haze, 24mm lens, deep focus, hopeful mood, ochre and teal palette, crowds"] * 3,
             "motion": "shoppers walk past the stalls, slow push in"},
            {"scene": 2, "shots": ["a vendor with a sign"]}]}

    monkeypatch.setattr(director, "chat_json", fake_chat)
    counts = director.plan_counts(SCRIPT, "long", "balanced", pipeline.plan_scene)
    out = director.direct(SCRIPT, counts, {"topic": "UPI", "format": "long"}, log=lambda m: None)
    assert out["bible"]["places"] == ["India"]
    assert out["scenes"][0]["motion"].startswith("shoppers walk")
    assert out["scenes"][1]["shots"][0].startswith("Wide aerial shot")       # repaired by the critic
    assert out["stats"]["repaired"] == 1 and len(calls) == 2


def test_cost_estimate_and_bill():
    cheap = cost.estimate({"format": "reel", "seconds": 15, "scenes": 3, "motion": 0})
    heavy = cost.estimate({"format": "long", "seconds": 300, "scenes": 20, "motion": 6, "render": "fast"})
    assert 0 < cheap["usd"] < heavy["usd"]
    assert cost.estimate({"format": "long", "seconds": 60, "scenes": 5, "motion": 3, "render": "economy"})["usd"] < \
        cost.estimate({"format": "long", "seconds": 60, "scenes": 5, "motion": 3, "render": "fast"})["usd"]
    assert cost.gpu_key("NVIDIA L40S") == "L40S" and cost.gpu_key("NVIDIA A100-SXM4-80GB") == "A100-80"
    assert cost.gpu_key("Tesla T4") == "T4" and cost.gpu_key("NVIDIA H200") == "H200"
    b = cost.bill({"motion": {"gpu": "NVIDIA H200", "busy": 90, "load": 100, "containers": 1}}, 120)
    assert abs(b["breakdown"]["motion"] - (90 + 100 + cost.IDLE["motion"]) * 0.001261) < 1e-3


def test_default_scenes_per_format():
    assert config.default_scenes(15, "reel") == 3 and config.default_scenes(60, "reel") == 8
    assert config.default_scenes(300, "long") == 20 and config.default_scenes(60, "long") == 5
