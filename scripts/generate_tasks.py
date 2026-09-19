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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    counts = {"done": 0, "kept": 0}

    def run(job):
        if STOP or (time.monotonic() - t0) / 3600 > a.max_hours:
            return
        model = pool.next_model()
        while model is None:
            if STOP: return
            time.sleep(20); model = pool.next_model()
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
            if "429" in str(e) or "503" in str(e):
                pool.retire(model)
            return
        ok = VALIDATORS[job["kind"]](text, job["meta"])
        with lock:
            sink.write(json.dumps({**job, "output": text.strip(),
                                   "valid": ok, "model": model}) + "\n")
            sink.flush()
            counts["done"] += 1; counts["kept"] += int(ok)
            if counts["done"] % 10 == 0:
                el = max((time.monotonic()-t0)/60, .01)
                print(f"  [{counts['done']:>5}] kept {counts['kept']:>5} "
                      f"({counts['kept']/counts['done']:5.1%})  {counts['done']/el:4.1f}/min",
                      flush=True)

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            fs = [ex.submit(run, j) for j in jobs]
            for _ in as_completed(fs):
                if STOP:
                    for f in fs: f.cancel()
                    break
    finally:
        sink.close()
    print(f"\n{counts['kept']}/{counts['done']} valid -> {out}")


if __name__ == "__main__":
    main()
