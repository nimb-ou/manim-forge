"""Score the hard eval: a whole explainer from a single line.

Expected to be poor at first. The point is a measurement that keeps moving as
the model improves at the job the tool is actually for, rather than one that
saturates on single-scene prompts.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--backend", default="local", choices=["local", "gemini"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--tag", default="hard")
    a = ap.parse_args()

    from forge.evaluate.hard_eval import build_tasks, concept_coverage, HardScore
    from forge.harness import RenderHarness
    from forge.repair.loop import RepairLoop, SYSTEM

    tasks = build_tasks()[: a.n]
    print(f"{len(tasks)} hard tasks | backend={a.backend} retrieval={a.retrieval}\n", flush=True)

    harness = RenderHarness(python_bin="./.venv/bin/python",
                            cache_dir="data/frames", timeout=180)

    index = None
    if a.retrieval:
        from forge.retrieve.examples import ExampleIndex
        index = ExampleIndex.load(Path("data/verified/example_index.jsonl"))

    if a.backend == "local":
        from mlx_lm import load
        model, tok = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                          **({"adapter_path": a.adapter} if a.adapter else {}))
        loop = RepairLoop(model, tok, harness, max_rounds=a.rounds,
                          max_tokens=3000, index=index)
        run = lambda p: loop.run(p)
    else:
        from forge.app.generator import RemoteGemini
        from forge.repair.loop import extract_code
        from forge.repair.lint import lint
        gen = RemoteGemini()

        def run(prompt):
            code = extract_code(gen.complete(SYSTEM, prompt, max_tokens=12000))
            code, _ = lint(code)
            r = harness.render(code, quality="low", frames=6)
            class R:
                ok, rounds_used, final_error = r.ok, 0, r.error_kind.value
            R.code = code
            R.result = r
            return R

    scores = []
    for i, t in enumerate(tasks, 1):
        t0 = time.monotonic()
        res = run(t.prompt)
        rendered = harness.render(res.code, quality="low", frames=6)
        cov, hits = concept_coverage(res.code, t.terms)

        s = HardScore(
            video_id=t.video_id, title=t.title, prompt=t.prompt,
            ok=rendered.ok, error_kind=rendered.error_kind.value,
            seconds=rendered.duration_s or 0.0, real_seconds=t.real_seconds,
            n_beats=res.code.count("@beat") or res.code.count("next_section"),
            n_play_calls=res.code.count("self.play("),
            coverage=round(cov, 3), matched_terms=hits,
        )
        scores.append(s)
        flag = "PASS" if s.ok else f"FAIL {s.error_kind}"
        print(f"  [{i:>2}/{len(tasks)}] {flag:<18} {s.seconds:5.1f}s "
              f"({s.length_ratio:5.1%} of real)  plays={s.n_play_calls:<3} "
              f"cov={s.coverage:5.1%}  {time.monotonic()-t0:5.1f}s | {t.prompt[:34]}",
              flush=True)

    out = Path(f"data/bench/{a.tag}_n{a.n}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(s) for s in scores], indent=2))

    ok = [s for s in scores if s.ok]
    print(f"\n{'='*64}\n  HARD EVAL — whole explainer from one line\n{'='*64}")
    print(f"  render success   {len(ok)}/{len(scores)} = {len(ok)/max(len(scores),1):.0%}")
    if ok:
        print(f"  mean duration    {sum(s.seconds for s in ok)/len(ok):.1f}s "
              f"(real videos average {sum(s.real_seconds for s in ok)/len(ok)/60:.0f} min)")
        print(f"  mean length ratio{sum(s.length_ratio for s in ok)/len(ok):>7.2%}")
        print(f"  mean play calls  {sum(s.n_play_calls for s in ok)/len(ok):.1f}")
        print(f"  concept coverage {sum(s.coverage for s in ok)/len(ok):.1%}")
    print(f"  -> {out}\n{'='*64}")


if __name__ == "__main__":
    main()
