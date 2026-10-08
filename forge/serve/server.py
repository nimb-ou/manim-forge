"""Manim Forge, served locally: a request in, the plan and code streamed, a video out.

    ./.venv/bin/python -m forge.serve              # http://127.0.0.1:8765

One worker thread owns the model (this Mac has one GPU and 16 GB, and two
7B models at once swap), so jobs queue. Each job streams its stages -- plan
beats as the planner writes them, code per beat, what assembly repaired or
dropped, the render -- as server-sent events. Both adapters share one base
model through forge.app.pipeline.SwapHost.

Every job is written to data/sessions/ with its request, plan, code, and
outcome, including failures. Real requests with real outcomes are better
training data than anything synthetic.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
# The app retrieves from every hand-written scene, held-out topics included
# (forge/kit/library.py); evaluations do not.
os.environ.setdefault("FORGE_LIBRARY", "release")
SESSIONS = ROOT / "data" / "sessions"
HERE = Path(__file__).parent

PLANNER = ROOT / "adapters" / "mlx-planner3"
# The kit coder when it exists. Kit mode by prompt alone does not work -- the
# raw-Manim coder copies the prompt's example verbatim -- so the kit is on by
# default only with a coder trained on it.
# Kit v6 (teacher + self-training rows): 47% visual, new and on-subject on
# the short prompts with relevance, vs 43% for v5 and 44% for GRPO v2.
KIT_CODER = ROOT / "adapters" / "mlx-coder6-kit"
CODER = KIT_CODER if (KIT_CODER / "adapters.safetensors").exists() \
    else ROOT / "adapters" / "mlx-coder2"
KIT_DEFAULT = CODER == KIT_CODER
# One shot (forge/app/oneshot.py): the whole scene in one call, adapted from
# the nearest hand-written scenes. Used when its adapter exists and ENGINE
# says so; "twostage" is the planner + per-beat coder above.
# The one shot runs untuned on oneshot.BASE_MODEL (Qwen3.5-9B) with the kit
# reference in its prompt: 13 of 20 in-scope right by eye, against 0 for the
# two-stage pipeline and 6 for the one-shot fine-tune (docs/RESULTS.md).
ENGINE = "oneshot"


class JobIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    beats: int = Field(default=8, ge=2, le=24)
    quality: str = Field(default="medium", pattern="^(low|medium|high)$")
    kit: bool = KIT_DEFAULT
    engine: str = Field(default=ENGINE, pattern="^(twostage|oneshot)$")
    narrate: bool = True


@dataclass
class Job:
    id: str
    spec: JobIn
    created: float = field(default_factory=time.time)
    events: list = field(default_factory=list)
    done: bool = False
    video: str | None = None
    cond: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, e: dict) -> None:
        e = {"t": round(time.time() - self.created, 1), **e}
        with self.cond:
            self.events.append(e)
            if e.get("stage") == "done":
                self.done = True
                self.video = e.get("video")
            self.cond.notify_all()


class Worker(threading.Thread):
    """Runs jobs one at a time on the one model."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q: queue.Queue[Job] = queue.Queue()
        self.host = None
        self.oneshot = None                  # (model, tokenizer)
        self.jobs: dict[str, Job] = {}

    def submit(self, spec: JobIn) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], spec=spec)
        self.jobs[job.id] = job
        job.emit({"stage": "queued", "position": self.q.qsize()})
        self.q.put(job)
        return job

    def run(self) -> None:
        from forge.app.pipeline import Options, SwapHost, run
        while True:
            job = self.q.get()
            try:
                if job.spec.engine == "oneshot":
                    self._run_oneshot(job)
                    continue
                if self.host is None:
                    self.oneshot = None          # one 7B in memory at a time
                    job.emit({"stage": "loading", "note": "loading the model "
                              "(once per server start, ~20 s)"})
                    self.host = SwapHost(str(PLANNER), str(CODER))
                # Relevance (subject hint + resampling off-subject beats)
                # with the kit: 34% -> 43% on-subject pictures, short evals.
                opts = Options(max_beats=job.spec.beats,
                               quality=job.spec.quality, kit=job.spec.kit,
                               relevance=job.spec.kit,
                               narrate=job.spec.narrate and job.spec.kit)
                res = run(job.spec.prompt, self.host, job.emit, opts)
                self._log(job, res)
            except Exception as exc:                          # noqa: BLE001
                job.emit({"stage": "done", "ok": False,
                          "error": f"{type(exc).__name__}: {exc}",
                          "trace": traceback.format_exc()[-2000:]})

    def _run_oneshot(self, job: Job) -> None:
        from forge.app.oneshot import run_oneshot
        from forge.app.pipeline import Options, load
        if self.oneshot is None:
            self.host = None
            job.emit({"stage": "loading", "note": "loading the model "
                      "(once per server start, ~20 s)"})
            from forge.app.oneshot import BASE_MODEL
            self.oneshot = load(None, base=BASE_MODEL)
        model, tok = self.oneshot
        opts = Options(max_beats=job.spec.beats, quality=job.spec.quality,
                       kit=True, narrate=job.spec.narrate)
        # A second sample only when the first fails to render whole or shows
        # an arithmetic slip or a layout problem: both of Qwen3.5-9B's
        # in-scope failures in run 2 were render failures.
        res = run_oneshot(job.spec.prompt, model, tok, job.emit, opts,
                          api=True, samples=2, plan=True)
        self._log(job, res)

    def _log(self, job: Job, res) -> None:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        rec = {"id": job.id, "created": job.created,
               "request": job.spec.prompt, "options": job.spec.model_dump(),
               "ok": res.ok, "error": res.error, "notes": res.notes,
               "beats": [{"n": b.n, "seconds": b.seconds, "intent": b.intent,
                          "narration": b.narration} for b in res.beats],
               "bodies": res.bodies, "code": res.code, "video": res.video,
               "seconds": round(res.seconds, 1)}
        (SESSIONS / f"{time.strftime('%Y%m%d-%H%M%S')}-{job.id}.json") \
            .write_text(json.dumps(rec, indent=1))


worker = Worker()


@asynccontextmanager
async def _lifespan(_app):
    if not worker.is_alive():
        worker.start()
    yield


app = FastAPI(title="Manim Forge", lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text()


@app.post("/api/jobs")
def create(spec: JobIn) -> dict:
    job = worker.submit(spec)
    return {"id": job.id}


@app.get("/api/jobs/{jid}/events")
def events(jid: str) -> StreamingResponse:
    job = worker.jobs.get(jid)
    if job is None:
        raise HTTPException(404, "no such job")

    def stream():
        sent = 0
        while True:
            with job.cond:
                while sent >= len(job.events) and not job.done:
                    job.cond.wait(timeout=15)
                    if sent >= len(job.events) and not job.done:
                        break            # heartbeat below keeps the socket open
                new = job.events[sent:]
                sent = len(job.events)
                finished = job.done and sent >= len(job.events)
            for e in new:
                yield f"data: {json.dumps(e)}\n\n"
            if not new:
                yield ": keep-alive\n\n"
            if finished:
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/jobs/{jid}/video")
def video(jid: str) -> FileResponse:
    job = worker.jobs.get(jid)
    if job is None or not job.video or not Path(job.video).exists():
        raise HTTPException(404, "no video for this job")
    return FileResponse(job.video, media_type="video/mp4")


@app.get("/api/sessions")
def sessions(limit: int = 20) -> list[dict]:
    out = []
    for f in sorted(SESSIONS.glob("*.json"), reverse=True)[:limit]:
        try:
            r = json.loads(f.read_text())
        except Exception:                                     # noqa: BLE001
            continue
        out.append({"id": r["id"], "request": r["request"], "ok": r["ok"],
                    "beats": len(r.get("beats", [])),
                    "seconds": r.get("seconds")})
    return out


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_loaded": worker.host is not None or worker.oneshot is not None,
            "queued": worker.q.qsize(),
            "planner": PLANNER.name, "coder": CODER.name, "kit": KIT_DEFAULT,
            "engine": ENGINE, "model": __import__("forge.app.oneshot",
                                                   fromlist=["BASE_MODEL"]).BASE_MODEL}
