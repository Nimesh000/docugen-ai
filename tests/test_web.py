from fastapi.testclient import TestClient

from docugen import config
from docugen.web.server import create_app


def make(tmp_path, monkeypatch, code="", limit=2):
    monkeypatch.setattr(config, "ACCESS_CODE", code)
    monkeypatch.setattr(config, "DAILY_LIMIT", limit)
    store, spawned = {}, []
    app = create_app(store, lambda jid, p: spawned.append((jid, p)), tmp_path)
    return TestClient(app), store, spawned


def test_create_job_and_daily_limit(tmp_path, monkeypatch):
    client, store, spawned = make(tmp_path, monkeypatch)
    body = {"topic": "The rise of India's UPI", "seconds": 60, "scenes": 6, "motion": 2}
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 200
    jid = r.json()["id"]
    assert spawned[0][0] == jid and spawned[0][1]["topic"] == "The rise of India's UPI"
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "queued"
    assert client.post("/api/jobs", json=body).status_code == 200
    assert client.post("/api/jobs", json=body).status_code == 429   # limit 2


def test_access_code_and_validation(tmp_path, monkeypatch):
    client, _, _ = make(tmp_path, monkeypatch, code="UPI2026")
    assert client.get("/api/config").json()["access_required"] is True
    assert client.post("/api/jobs", json={"topic": "x topic", "code": "nope"}).status_code == 403
    assert client.post("/api/jobs", json={"topic": "x topic", "code": "UPI2026", "seconds": 999}).status_code == 400
    assert client.post("/api/jobs", json={"topic": "x topic", "code": "UPI2026", "scenes": 3,
                                          "motion": 4}).status_code == 400
    assert client.post("/api/jobs", json={"topic": "x topic", "code": "UPI2026", "scenes": 20}).status_code == 422
    assert client.post("/api/jobs", json={"topic": "x topic", "code": "UPI2026"}).status_code == 200
    cfg = client.get("/api/config").json()
    assert cfg["scenes"]["max"] == 12 and cfg["default_voice"] in cfg["voices"]


def test_file_serving_is_sandboxed(tmp_path, monkeypatch):
    client, _, _ = make(tmp_path, monkeypatch)
    (tmp_path / "job1" / "output").mkdir(parents=True)
    (tmp_path / "job1" / "output" / "script.md").write_text("# hi")
    assert client.get("/api/jobs/job1/files/output/script.md").status_code == 200
    assert client.get("/api/jobs/job1/files/../secret.md").status_code in (400, 404)
    assert client.get("/api/jobs/job1/files/output/evil.py").status_code == 404
    assert client.get("/").status_code == 200
