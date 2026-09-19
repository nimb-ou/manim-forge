"""Local Manim Forge server — prompt in, animation out.

Runs entirely on this machine except for the optional remote backend. The
pipeline is the one we built and measured:

    prompt -> generate -> lint -> render -> (repair with real API signatures) -> mp4

Every attempt is logged to ``data/sessions/``, including the failures and what
was done about them. That log is the flywheel: real prompts with real outcomes
are better training data than anything synthetic, because they carry genuine
intent rather than a topic list.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from forge.harness import RenderHarness, is_repairable
from forge.repair.loop import SYSTEM, extract_code, repair_prompt
from forge.repair.lint import lint

APP_DIR = Path(__file__).parent
MEDIA = Path("data/app_renders")
SESSIONS = Path("data/sessions")

app = FastAPI(title="Manim Forge")
_state: dict = {}


class Job(BaseModel):
    prompt: str
    backend: str = "gemini"      # "gemini" | "local"
    max_rounds: int = 2
    quality: str = "low"


def _generator(kind: str):
    if kind in _state:
        return _state[kind]
    if kind == "local":
        from forge.app.generator import LocalMLX
        g = LocalMLX()
    else:
        from forge.app.generator import RemoteGemini
        g = RemoteGemini()
    _state[kind] = g
    return g


def _harness() -> RenderHarness:
    if "harness" not in _state:
        _state["harness"] = RenderHarness(
            python_bin="./.venv/bin/python", cache_dir="data/frames",
            timeout=120, store_video=True)
    return _state["harness"]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "index.html").read_text()


@app.post("/generate")
def generate(job: Job) -> JSONResponse:
    gen = _generator(job.backend)
    harness = _harness()
    jid = uuid.uuid4().hex[:12]
    t0 = time.monotonic()
    attempts = []

    code = extract_code(gen.complete(SYSTEM, job.prompt, max_tokens=4500))
    code, rules = lint(code)
    result = harness.render(code, quality=job.quality, frames=6)
    attempts.append({"round": 0, "ok": result.ok,
                     "error": result.error_kind.value, "lint": rules})

    rounds = 0
    while not result.ok and rounds < job.max_rounds:
        if result.is_environment_failure or not is_repairable(result.error_kind):
            break
        rounds += 1
        code = extract_code(gen.complete(
            SYSTEM, repair_prompt(job.prompt, code, result), max_tokens=4500))
        code, more = lint(code)
        rules += more
        result = harness.render(code, quality=job.quality, frames=6)
        attempts.append({"round": rounds, "ok": result.ok,
                         "error": result.error_kind.value, "lint": more})

    video_url = None
    if result.ok and result.video_path:
        MEDIA.mkdir(parents=True, exist_ok=True)
        dest = MEDIA / f"{jid}.mp4"
        dest.write_bytes(Path(result.video_path).read_bytes())
        video_url = f"/video/{jid}.mp4"

    record = {
        "id": jid, "prompt": job.prompt, "backend": gen.name,
        "ok": result.ok, "code": code, "attempts": attempts,
        "rounds": rounds, "seconds": round(time.monotonic() - t0, 1),
        "video_s": result.duration_s, "error": result.feedback(8),
    }
    SESSIONS.mkdir(parents=True, exist_ok=True)
    with (SESSIONS / "log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    return JSONResponse({**record, "video_url": video_url})


@app.get("/video/{name}")
def video(name: str):
    p = MEDIA / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "backends": list(_state)}
