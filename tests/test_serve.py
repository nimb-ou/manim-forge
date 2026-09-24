"""The server's plumbing, with the model replaced by a fake pipeline."""
import json
import time

from fastapi.testclient import TestClient


def test_job_streams_events_and_logs_a_session(tmp_path, monkeypatch):
    import forge.serve.server as srv
    from forge.app import pipeline

    monkeypatch.setattr(srv, "SESSIONS", tmp_path)
    monkeypatch.setattr(pipeline, "SwapHost", lambda p, c: object())

    def fake_run(req, host, emit, opts):
        emit({"stage": "plan", "beat": {"n": 1, "seconds": 5, "intent": "a circle",
                                        "narration": ""}})
        emit({"stage": "code", "beat": 1, "intent": "a circle",
              "code": "c = Circle()", "note": ""})
        emit({"stage": "done", "ok": False, "error": "fake"})
        return pipeline.Result(req, [pipeline.Beat(1, 5, "a circle")],
                               ["c = Circle()"], error="fake")

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(srv.app) as client:
        jid = client.post("/api/jobs", json={"prompt": "circles"}).json()["id"]
        with client.stream("GET", f"/api/jobs/{jid}/events") as r:
            events = [json.loads(l[6:]) for l in r.iter_lines()
                      if l.startswith("data: ")]
        stages = [e["stage"] for e in events]
        assert stages[0] == "queued" and stages[-1] == "done"
        assert "plan" in stages and "code" in stages
        for _ in range(20):
            if list(tmp_path.glob("*.json")):
                break
            time.sleep(0.1)
        rec = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert rec["request"] == "circles" and rec["bodies"] == ["c = Circle()"]
