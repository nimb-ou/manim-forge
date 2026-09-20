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
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Not every catalogued file is JSONL — a plain .json file's
                    # lines parse as fragments or bare strings, and asking them
                    # for .get crashed the whole report.
                    if isinstance(rec, dict) and rec.get("ok"):
                        ok += 1
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
        "data/synthetic/generated.jsonl",
        "First-generation topic synthesis, before the continuous daemon.",
        "id, source, prompt, code, ok, error_kind, topic, domain, length, teacher_model",
        "Failures carry error_kind, which is how the 61%-to-94% yield jump was diagnosed.",
    ),
    Dataset(
        "data/synthetic/from_narration.jsonl",
        "Scenes generated from 3b1b narration passages, before the daemon merged the two.",
        "id, prompt (the passage), code, ok, video_id, title, seg_index, teacher_model",
    ),
    Dataset(
        "data/verified/regate.jsonl",
        "Second pass over previously-failed rows after the lint improved.",
        "id, retried, ok, error_kind, lint, code",
        "Records which rows were retried and which recovered — without it the same "
        "rows would be re-attempted on every future lint change.",
    ),
    Dataset(
        "data/train",
        "Final training splits, grouped by prompt so paraphrases cannot straddle them.",
        "train.jsonl / valid.jsonl — messages[system,user,assistant]",
    ),
    Dataset(
        "data/verified/bench_prompts.json",
        "The held-out benchmark prompts, frozen so scores stay comparable across machines.",
        "prompts[]",
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
        "data/verified/gate_smoke.jsonl",
        "Thirty-row smoke test used to validate the gate before committing hours to it.",
        "same shape as gate.jsonl",
        "Kept rather than deleted, but not archived: it is a throwaway from one "
        "afternoon and carries nothing the full gate run does not.",
        archive=False,
    ),
    Dataset(
        "data/renders",
        "Rendered mp4s for gold scenes and platform output.",
        "videos/<scene>/<quality>/*.mp4 plus per-beat sections",
        archive=False,   # regenerable from code, and large
    ),
    Dataset(
        "data/showcase/rendered.jsonl",
        "Presentation-quality render ledger: one verdict per gold scene at 1080p60.",
        "scene, module, quality, ok, seconds, declared_s, error, interrupted?",
        "Declared-versus-actual runtime is the only place the beat timings are "
        "checked against a real render rather than against the word count. "
        "`interrupted` marks an attempt killed by a signal -- not a verdict on "
        "the scene.",
        archive=True,
    ),
    Dataset(
        "data/showcase/videos",
        "The reference renders themselves: every gold scene at 1080p60, with sections.",
        "videos/<scene>/1080p60/*.mp4 plus sections/<Scene>.json per beat",
        "Regenerable in principle, but each one costs minutes of CPU and these "
        "are the artefacts the style is actually judged against -- the gallery, "
        "and the reference a human compares a generated scene to. Uncatalogued "
        "until now, which meant 73 MB of finished product was outside every "
        "backup.",
        archive=True,
    ),
    Dataset(
        "data/checks",
        "Per-scene render check output from verify_gold and the smoke harness.",
        "<Scene>/ render artefacts and verdicts",
        archive=False,   # regenerable, and superseded by the showcase ledger
    ),
]


def orphans() -> list[str]:
    """Data files on disk that no Dataset entry covers.

    The catalogue only prevents drift if it notices what it is missing. Five
    datasets went uncatalogued — and therefore unarchived and unuploaded —
    simply because adding a pipeline and adding its entry are separate acts and
    the second is easy to forget. This makes forgetting visible.
    """
    declared = {Path(d.path) for d in CATALOG}
    found = set()
    for pattern in ("*.jsonl", "*.json"):
        found |= {p for p in Path("data").rglob(pattern)}
    missing = []
    for f in sorted(found):
        if any(f == d or d in f.parents for d in declared):
            continue
        missing.append(str(f))
    return missing


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
    missing = orphans()
    if missing:
        print(f"\nUNCATALOGUED — not archived, not uploaded ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    else:
        print("\nno uncatalogued data files")


if __name__ == "__main__":
    report()
