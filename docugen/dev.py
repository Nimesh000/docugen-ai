"""Local preview of the web app with a fake (CPU) backend - no GPUs, no API keys.

    python -m docugen.dev        -> http://127.0.0.1:8000
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docugen import pipeline  # noqa: E402
from docugen.web.server import create_app, remember_recent  # noqa: E402
from tests.fakes import FakeBackend  # noqa: E402

JOBS = ROOT / "jobs_dev"
STORE: dict = {}


def spawn(job_id: str, params: dict) -> None:
    def work() -> None:
        status = pipeline.Status(STORE, job_id, STORE[job_id])
        if pipeline.run(params, JOBS / job_id, FakeBackend(), status):
            remember_recent(STORE, job_id)

    threading.Thread(target=work, daemon=True).start()


app = create_app(STORE, spawn, JOBS)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
