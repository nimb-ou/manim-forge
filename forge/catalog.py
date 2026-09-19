"""A catalogue of every dataset the project produces.

`data/` grows a new jsonl every time a pipeline is added, and undocumented
jsonl files are how a corpus becomes unusable six weeks later — nobody
remembers which file holds what, which schema it uses, or whether the failures
in it were kept on purpose.

So every dataset is declared here with its schema, its purpose, and crucially
**why its failures are worth keeping**. Failures are not waste in this project:
a broken generation with its traceback and its fix is the rarest training data
the pipeline produces.

Running this prints live counts, so the catalogue cannot drift from the data
the way a README would.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    path: str
    purpose: str
    schema: str
    keeps_failures: str = ""
    licence: str = "CC BY-NC-SA 4.0"
    archive: bool = True          # include in the off-machine backup

    def stats(self) -> dict:
        p = Path(self.path)
        if not p.exists():
            return {"exists": False}
        if p.is_dir():
            files = list(p.rglob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())
            return {"exists": True, "files": sum(1 for f in files if f.is_file()),
                    "mb": round(size / 1e6, 1)}
        rows = ok = 0
        try:
            with p.open() as f:
                for line in f:
                    rows += 1
                    try:
                        if json.loads(line).get("ok"):
                            ok += 1
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return {"exists": True, "rows": rows, "verified": ok,
                "mb": round(p.stat().st_size / 1e6, 1)}


CATALOG: list[Dataset] = [
    Dataset(
        "data/normalized/corpus.jsonl",
        "Five public sources normalised to one schema, deduplicated by AST structure.",
        "id, source, license, prompt, code, flavor, scene_class, n_play_calls, dedupe_key, tags",
        "Rows that fail later gating stay here — the gate's verdict is recorded separately, "
        "so a row rejected under one Manim version can be re-gated under the next.",
    ),
    Dataset(
        "data/verified/gate.jsonl",
        "The render gate's verdict on every corpus row.",
        "id, ok, error_kind, is_env_failure, n_frames, duration_s, elapsed_s, stderr_tail, lint, code",
        "Failures are the point: error_kind and stderr_tail per row are what the repair "
        "objective trains on, and environment failures are marked so they can be retried "
        "rather than mistaken for bad data.",
    ),
    Dataset(
        "data/verified/train.jsonl",
        "Chat-formatted training examples, animated rows weighted.",
        "messages[system,user,assistant], meta{id,source,license,n_play_calls,animated}",
    ),
    Dataset(
        "data/verified/example_index.jsonl",
        "Embedded verified scenes for few-shot retrieval (nomic-embed-text, 768-dim).",
        "prompt, code, source, n_play_calls, vec[768]",
    ),
    Dataset(
        "data/synthetic/stream.jsonl",
        "Continuous teacher generation — topics and 3b1b narration interleaved.",
        "task_key, kind, prompt, code, ok, error_kind, repair_rounds, history, attempts[], teacher_model, domain",
        "attempts[] holds every failed draft with its traceback. Paired with the final "
        "code these are (broken -> error -> fixed) triples, the rarest data here.",
    ),
    Dataset(
        "data/synthetic/tasks.jsonl",
        "Non-code skills: planning, narration, paraphrase, explanation, critique, decomposition.",
        "kind, prompt, output, valid, model, meta",
        "Invalid outputs are kept with valid=false — they show what the teacher gets wrong "
        "on each task type, which is worth knowing before we train on it.",
    ),
    Dataset(
        "data/style/narration.jsonl",
        "3Blue1Brown narration, segmented into beat-sized spans. 151 videos.",
        "video_id, title, index, start, end, text",
        licence="CC BY-NC-SA 4.0 (transcripts of 3b1b videos)",
    ),
    Dataset(
        "data/gold/gold.jsonl",
        "Hand-authored reference scenes — the style anchor, weighted 6x at training.",
        "prompt, code, meta{id,source,scene,tags,n_play_calls}",
    ),
    Dataset(
        "data/frames",
        "Sampled frames from every successful render, for visual comparison.",
        "<hash>/frames/fNN.jpg plus result.json",
    ),
    Dataset(
        "data/bench",
        "Every benchmark run, with per-prompt outcomes.",
        "summary{} + trials[] per run",
        "Failed trials retain their generated code and error trajectory — that is how "
        "we found repair was cycling rather than converging.",
    ),
    Dataset(
        "data/sessions/log.jsonl",
        "Every request made through the local platform, with outcome.",
        "id, prompt, backend, ok, code, attempts[], rounds, seconds",
        "Real prompts with real outcomes are better training data than anything "
        "synthetic, because they carry genuine intent.",
        archive=True,
    ),
    Dataset(
        "data/renders",
        "Rendered mp4s for gold scenes and platform output.",
        "videos/<scene>/<quality>/*.mp4 plus per-beat sections",
        archive=False,   # regenerable from code, and large
    ),
]


def report() -> None:
    print(f"{'dataset':<38} {'rows':>8} {'verified':>9} {'MB':>8}  archive")
    print("-" * 82)
    total_mb = arch_mb = 0.0
    for d in CATALOG:
        s = d.stats()
        if not s.get("exists"):
            print(f"{d.path:<38} {'—':>8} {'—':>9} {'—':>8}  {'yes' if d.archive else 'no'}")
            continue
        rows = s.get("rows", s.get("files", 0))
        ver = s.get("verified", "")
        mb = s.get("mb", 0.0)
        total_mb += mb
        if d.archive:
            arch_mb += mb
        print(f"{d.path:<38} {rows:>8} {str(ver):>9} {mb:>8.1f}  {'yes' if d.archive else 'no'}")
    print("-" * 82)
    print(f"{'TOTAL':<38} {'':>8} {'':>9} {total_mb:>8.1f}")
    print(f"{'to archive':<38} {'':>8} {'':>9} {arch_mb:>8.1f}")


if __name__ == "__main__":
    report()
