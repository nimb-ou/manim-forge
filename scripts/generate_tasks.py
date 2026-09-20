"""Generate the non-code training tasks — planning, narration, critique, and so on.

Runs alongside the code daemon on the same rotated free-tier quota. These
outputs are prose, so the render gate cannot judge them; quality control is the
teacher's own competence plus cheap structural checks (a plan must parse as
numbered beats, a paraphrase set must contain distinct lines). Anything that
fails those is dropped rather than kept on faith.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import signal
import threading
import time
from collections import deque
from concurrent.futures import (FIRST_COMPLETED, ThreadPoolExecutor,
                                wait)
from pathlib import Path

STOP = False


def _handle(s, f):
    global STOP
    STOP = True
    print("\n  stopping after in-flight tasks", flush=True)


# --- structural validators: prose cannot be render-gated, so check shape ------

def valid_plan(text: str) -> bool:
    lines = [l for l in text.splitlines() if re.match(r"^\s*\d+[.)]", l)]
    return len(lines) >= 2 and any("--" in l or "—" in l for l in lines)


def valid_paraphrase(text: str, want: int) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return len(set(lines)) >= max(2, want // 2)


def valid_prose(text: str, min_chars: int = 40) -> bool:
    return len(text.strip()) >= min_chars and "```" not in text


VALIDATORS = {
    "plan": lambda t, m: valid_plan(t),
    "paraphrase": lambda t, m: valid_paraphrase(t, m.get("n", 6)),
    "decompose": lambda t, m: len([l for l in t.splitlines() if ":" in l]) >= 2,
    "narrate": lambda t, m: valid_prose(t, 30),
    "explain": lambda t, m: valid_prose(t, 60),
    "critique": lambda t, m: valid_prose(t, 20) or "NO ISSUES" in t,
}


def main() -> None:
    signal.signal(signal.SIGINT, _handle)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/tasks.jsonl")
    ap.add_argument("--kinds", default="plan,paraphrase,decompose,narrate,explain,critique")
    ap.add_argument("--per-kind", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=20.0)
    a = ap.parse_args()

    from forge.synth.teacher import Teacher, GEMINI_ROTATION, PROVIDERS
    from forge.synth.tasks import SPECS, PROSE_SYSTEM
    from forge.synth.daemon import ModelPool
    from forge.synth.topics import ALL

    verified = [json.loads(l) for p in
                ("data/verified/train.jsonl",) if Path(p).exists()
                for l in Path(p).open()]
    code_samples = []
    seen = set()
    for r in verified:
        meta = r.get("meta", {})
        if meta.get("id") in seen:
            continue
        seen.add(meta.get("id"))
        body = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        code = body.replace("```python", "").replace("```", "").strip()
        if 200 < len(code) < 4000:
            code_samples.append(code)
    random.Random(3).shuffle(code_samples)

    jobs = []
    kinds = [k.strip() for k in a.kinds.split(",")]
    for kind in kinds:
        spec = SPECS[kind]
        for i in range(a.per_kind):
            if kind in ("plan", "decompose"):
                topic, domain = ALL[i % len(ALL)]
                n = 4 if kind == "decompose" else 0
                prompt = spec.template.format(request=topic, n=n) if n else \
                         spec.template.format(request=topic)
                meta = {"topic": topic, "domain": domain, "n": n}
            elif kind == "paraphrase":
                topic, domain = ALL[i % len(ALL)]
                prompt = spec.template.format(topic=topic, n=6)
                meta = {"topic": topic, "domain": domain, "n": 6}
            else:
                if not code_samples:
                    break
                code = code_samples[i % len(code_samples)]
                prompt = spec.template.format(code=code, beat=code[:600])
                meta = {"code_chars": len(code)}
            jobs.append({"kind": kind, "prompt": prompt, "meta": meta,
                         "key": f"{kind}::{i}"})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open():
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    jobs = [j for j in jobs if j["key"] not in done]
    random.Random(5).shuffle(jobs)
    print(f"{len(jobs)} prose tasks | {len(code_samples)} verified scenes to draw on")
    print(f"kinds: {kinds}\n", flush=True)

    pool = ModelPool(list(GEMINI_ROTATION))
    lock = threading.Lock()
    sink = out.open("a")
    t0 = time.monotonic()
    counts = {"done": 0, "kept": 0, "retried": 0, "dropped": 0}

    def run(job):
        """Attempt one prose task. Returns True if the job is finished with.

        Returning False means *retry*, not *fail*. A 429 says the model is
        busy, which is a fact about the minute, not about the job -- and a
        daemon that treats the two the same eats its whole queue the first
        time the quota runs out. That is exactly what happened here: 5,789
        tasks consumed, 0 rows written, and a log that called it "API errors"
        rather than "queue destroyed".
        """
        if STOP or (time.monotonic() - t0) / 3600 > a.max_hours:
            return True                     # out of time: genuinely done
        model = pool.next_model()
        if model is None:
            time.sleep(20)
            return False                    # everything cooling down; keep the job
        t = Teacher(provider="gemini", model=model)
        try:
            import forge.synth.teacher as T
            original = T.SYSTEM
            T.SYSTEM = PROSE_SYSTEM          # code system prompt corrupts prose
            try:
                text = t._gemini_rest(job["prompt"], 3000)
            finally:
                T.SYSTEM = original
        except Exception as e:
            msg = str(e)
            if "429" in msg or "503" in msg or "500" in msg:
                pool.retire(model)
                with lock:
                    counts["retried"] += 1
                return False                 # transient: put it back
            with lock:
                counts["dropped"] += 1       # a real error; record and move on
            return True
        ok = VALIDATORS[job["kind"]](text, job["meta"])
        with lock:
            sink.write(json.dumps({**job, "output": text.strip(),
                                   "valid": ok, "model": model}) + "\n")
            sink.flush()
            counts["done"] += 1
            counts["kept"] += int(ok)
            if counts["done"] % 10 == 0:
                el = max((time.monotonic() - t0) / 60, .01)
                print(f"  [{counts['done']:>5}] kept {counts['kept']:>5} "
                      f"({counts['kept'] / counts['done']:5.1%})  "
                      f"{counts['done'] / el:4.1f}/min  "
                      f"retried {counts['retried']}  dropped {counts['dropped']}",
                      flush=True)
        return True

    try:
        # A real queue, not a fixed list of futures: a job that comes back
        # False goes to the end and is tried again later, so a rate limit
        # costs time rather than data.
        pending = deque(jobs)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            inflight = {}
            while (pending or inflight) and not STOP:
                while pending and len(inflight) < a.workers:
                    j = pending.popleft()
                    inflight[ex.submit(run, j)] = j
                if not inflight:
                    break
                done_fs, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in done_fs:
                    j = inflight.pop(f)
                    try:
                        finished = f.result()
                    except Exception:
                        finished = False
                    if not finished and not STOP:
                        pending.append(j)
    except KeyboardInterrupt:
        pass

    print(f"\n{counts['kept']}/{counts['done']} valid -> {out}"
      f"   ({counts['retried']} retried, {counts['dropped']} dropped)")


if __name__ == "__main__":
    main()
