#!/usr/bin/env python3
"""Run the two-stage pipeline end to end: plan, implement, assemble, render.

    ./.venv/bin/python scripts/run_twostage.py --n 8            # untuned
    ./.venv/bin/python scripts/run_twostage.py --n 8 \
        --planner adapters/mlx-planner --coder adapters/mlx-coder

Run it **untuned first**. The split's own machinery -- windowed planning,
per-beat implementation, assembly -- can fail on its own, and if it does then
no adapter will rescue it. Measuring the pipeline before the adapters exist
separates "the split does not work" from "the adapters are not good yet",
and those call for completely different responses.

Both adapters are LoRA over one base, so they are loaded as two models
sharing it only in spirit -- mlx-lm has no adapter hot-swap, and holding two
7B 4-bit models is 8 GB on a 16 GB machine. So they are loaded one at a
time: plan every beat first, then swap and implement them. That costs a load
per prompt and buys the ability to run at all.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
import time
from pathlib import Path

from forge.app.twostage import (assemble, extract_code, names_in_scope,
                                parse_plan)
from forge.harness import RenderHarness

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

PLAN_SYSTEM = (
    "You plan 3Blue1Brown-style mathematical animations, a few beats at a "
    "time. You are given the request and the beats already planned. Continue "
    "the arc: write the next beats and nothing else, in the form\n"
    "  N. [seconds] intent -- narration\n"
    "keeping the numbering running. Build a real explanatory arc rather than "
    "a list of topics. When the explanation is complete, write END on its "
    "own line after the last beat."
)
CODE_SYSTEM = (
    "You write one beat of a 3Blue1Brown-style Manim scene. You are given the "
    "whole request, the beats already on screen, and the helpers the scene "
    "defines. Write only the code for the beat you are asked for, as "
    "statements at method-body level -- no class, no def, no imports. Reuse "
    "the names already in scope rather than rebuilding what they refer to, "
    "and do not use a name that is not listed."
)


def load(adapter: str | None):
    from mlx_lm import load as mlx_load
    return mlx_load(MODEL, **({"adapter_path": adapter} if adapter else {}))


def ask(model, tok, system: str, user: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    chat = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt=chat, max_tokens=max_tokens,
                    sampler=make_sampler(temp=0.0), verbose=False)


def plan(model, tok, request: str, stride: int, max_beats: int) -> list:
    """Feed the planner its own output until it says END or hits the cap."""
    beats: list = []
    for _ in range(max_beats // stride + 1):
        shown = "\n".join(f"{b.n}. [{b.seconds:g}s] {b.intent}"
                          for b in beats[-12:]) or \
            "  (nothing yet — open the explanation)"
        reply = ask(model, tok, PLAN_SYSTEM,
                    f"REQUEST\n{request}\n\nBEATS SO FAR\n{shown}\n\n"
                    f"Write the next {stride} beat(s), numbered from "
                    f"{len(beats) + 1}.", max_tokens=700)
        new, ended = parse_plan(reply, limit=stride)
        new = [b for b in new if b.n > len(beats)]
        if not new:
            break
        beats.extend(new)
        if ended or len(beats) >= max_beats:
            break
    return beats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--planner", default=None)
    ap.add_argument("--coder", default=None)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--beat-tokens", type=int, default=900,
                    help="600 truncated half the untuned run's beats")
    ap.add_argument("--max-beats", type=int, default=24)
    ap.add_argument("--tag", default="twostage")
    ap.add_argument("--plan-only", action="store_true",
                    help="stop after planning. The length question -- does "
                         "the planner produce a 16-minute arc -- is answerable "
                         "without the coder, and at 40 beats x 81 tasks the "
                         "implementation is ~9 hours of CPU for beats an "
                         "untuned coder will write badly anyway.")
    ap.add_argument("--hard", action="store_true",
                    help="run the 81 real 3Blue1Brown titles instead of the "
                         "single-scene prompts, and score coverage and length "
                         "ratio the way run_hard_eval does -- so the split's "
                         "number sits next to Phase 1's on the same tasks")
    a = ap.parse_args()

    hard_tasks = None
    if a.hard:
        from forge.evaluate.hard_eval import build_tasks
        hard_tasks = build_tasks()[: a.n]
        requests = [t.prompt for t in hard_tasks]
    else:
        raw = json.loads((ROOT / "data" / "verified" / "bench_prompts.json")
                         .read_text())
        tasks = raw["prompts"] if isinstance(raw, dict) else raw
        requests = [t["prompt"] if isinstance(t, dict) else t
                    for t in tasks][: a.n]

    print(f"planning {len(requests)} requests "
          f"(adapter: {a.planner or 'none'})", flush=True)
    pm, ptok = load(a.planner)
    plans = []
    for i, r in enumerate(requests, 1):
        t0 = time.time()
        bs = plan(pm, ptok, r, a.stride, a.max_beats)
        plans.append(bs)
        print(f"  [{i}/{len(requests)}] {len(bs)} beats "
              f"({time.time() - t0:.0f}s)", flush=True)
    del pm, ptok

    if a.plan_only:
        counts = [len(b) for b in plans]
        secs = [sum(x.seconds or 0 for x in b) for b in plans]
        real = ([t.real_seconds for t in hard_tasks] if hard_tasks
                else [0] * len(plans))
        ratios = [s / r for s, r in zip(secs, real) if r]
        out = ROOT / "data" / "bench" / f"{a.tag}_plans_n{len(plans)}.json"
        out.write_text(json.dumps(
            {"meta": {"planner": a.planner, "n": len(plans),
                      "plan_only": True},
             "trials": [{"index": i, "request": r, "beats": len(b),
                         "declared_seconds": s,
                         "real_seconds": (hard_tasks[i].real_seconds
                                          if hard_tasks else None),
                         "plan": [{"n": x.n, "seconds": x.seconds,
                                   "intent": x.intent,
                                   "narration": x.narration} for x in b]}
                        for i, (r, b, s) in enumerate(zip(requests, plans, secs))]},
            indent=2))
        print(f"\nbeats/arc  {sum(counts)/len(counts):.1f}   "
              f"(real 3Blue1Brown arcs average 39.7)")
        print(f"declared   {sum(secs)/len(secs)/60:.1f} min per arc")
        if ratios:
            print(f"length ratio {sum(ratios)/len(ratios):.1%}   "
                  f"(Phase 1: control 1.76%, run 17 3.47% -- but those are "
                  f"rendered seconds, and this is what the plan asks for)")
        print(f"-> {out}")
        return 0

    print(f"\nimplementing (adapter: {a.coder or 'none'})", flush=True)
    cm, ctok = load(a.coder)
    h = RenderHarness(python_bin="./.venv/bin/python", cache_dir="data/frames",
                      timeout=180)
    rows = []
    for i, (req, beats) in enumerate(zip(requests, plans), 1):
        bodies, dropped = [], []
        for j, b in enumerate(beats):
            prior = "\n".join(f"  {k + 1}. {beats[k].intent}"
                              for k in range(j)) or \
                "  (nothing yet — this is the opening beat)"
            scope = names_in_scope(bodies)
            body, why = "", ""
            # Two tries, and each one's output has to parse on its own.
            #
            # The untuned run lost three of six scenes to "'(' was never
            # closed" -- one beat truncated mid-expression and the whole
            # assembled scene stopped parsing, so five good beats were thrown
            # away by the sixth. A beat that does not parse alone cannot help
            # the scene, and dropping it costs one beat instead of all of
            # them.
            for attempt in range(2):
                reply = ask(cm, ctok, CODE_SYSTEM,
                        f"REQUEST\n{req}\n\nALREADY ON SCREEN\n{prior}\n\n"
                        f"NAMES IN SCOPE\n  "
                        + (", ".join(scope) if scope else "(none yet)")
                        + "\n\n"
                        f"WRITE THIS BEAT — step {j + 1} of {len(beats)}\n"
                        f"  intent: {b.intent}", max_tokens=a.beat_tokens)
                cand = extract_code(reply)
                try:
                    ast.parse(textwrap.dedent(cand))
                except SyntaxError as exc:
                    why = f"beat {j + 1} did not parse: {exc.msg}"
                    continue
                body, why = cand, ""
                break
            if why:
                dropped.append(why)
            bodies.append(body)
        asm = assemble(beats, bodies)

        # One targeted repair of the split's own failure. When a beat uses a
        # name no beat defines, the beat is asked again with the offending
        # names quoted back -- which is information the first prompt could
        # not contain, because it did not know what the model would invent.
        missing = [p for p in asm.problems if p.startswith("beats use names")]
        if missing and beats:
            names = missing[0].split(":", 1)[1].strip()
            scope = names_in_scope(bodies)
            for j, b in enumerate(beats):
                if not any(re.search(rf"\b{re.escape(n.strip())}\b", bodies[j])
                           for n in names.split(",") if n.strip()):
                    continue
                reply = ask(cm, ctok, CODE_SYSTEM,
                            f"REQUEST\n{req}\n\nNAMES IN SCOPE\n  "
                            + (", ".join(scope) if scope else "(none yet)")
                            + f"\n\nYour previous attempt at this beat used "
                              f"{names}, which nothing defines. Rewrite it "
                              f"using only the names in scope, or build what "
                              f"you need first.\n\n"
                              f"WRITE THIS BEAT — step {j + 1} of {len(beats)}"
                              f"\n  intent: {b.intent}", max_tokens=a.beat_tokens)
                cand = extract_code(reply)
                try:
                    ast.parse(textwrap.dedent(cand))
                except SyntaxError:
                    continue
                bodies[j] = cand
            asm = assemble(beats, bodies)
        asm.problems.extend(dropped)
        res = h.render(asm.code, quality="low", frames=4) if asm.ok else None
        row = {"index": i - 1, "request": req, "beats": len(beats),
               "assembled": bool(asm.bodies) and not [p for p in asm.problems
                                                     if "did not parse" not in p],
               "problems": asm.problems, "dropped": len(dropped),
               "ok": bool(res and res.ok),
               "error": (res.error_kind.value if res else "assembly"),
               "code": asm.code}
        if hard_tasks is not None:
            # Same fields and the same names run_hard_eval writes, so
            # paired_eval and compare_eval read this artefact without
            # knowing which pipeline produced it.
            from forge.evaluate.hard_eval import concept_coverage
            cov, hits = concept_coverage(asm.code, hard_tasks[i - 1].terms)
            row.update(coverage=round(cov, 3), matched_terms=hits,
                       real_seconds=hard_tasks[i - 1].real_seconds,
                       seconds=(res.duration_s or 0.0) if res else 0.0,
                       n_play_calls=asm.code.count("self.play("),
                       n_beats=len(beats), title=hard_tasks[i - 1].prompt[:60])
        if res is not None and not res.ok:
            row["stderr"] = (res.stderr or "")[-600:]
        rows.append(row)
        # The error kind, not just the boolean. A run that assembles and does
        # not render prints "rendered=False []" -- an empty problems list
        # because assembly was fine -- which says nothing about why, and the
        # answer was already in the row.
        print(f"  [{i}/{len(requests)}] {len(beats)} beats | "
              f"assembled={row['assembled']} rendered={row['ok']} "
              f"{row['error']} {asm.problems[:1]}", flush=True)

    out = ROOT / "data" / "bench" / f"{a.tag}_n{len(rows)}.json"
    out.write_text(json.dumps({"meta": {"planner": a.planner,
                                        "coder": a.coder, "n": len(rows)},
                               "trials": rows}, indent=2))
    asmok = sum(r["assembled"] for r in rows)
    ok = sum(r["ok"] for r in rows)
    print(f"\nassembled {asmok}/{len(rows)}   rendered {ok}/{len(rows)}   "
          f"mean beats {sum(r['beats'] for r in rows)/len(rows):.1f}")
    if hard_tasks is not None:
        done = [r for r in rows if r["ok"]]
        print(f"  coverage, all trials {sum(r['coverage'] for r in rows)/len(rows):.1%}"
              f"   (Phase 1: control 8.4%, run 17 18.8%)")
        if done:
            print(f"  coverage, rendered   "
                  f"{sum(r['coverage'] for r in done)/len(done):.1%}")
            print(f"  length ratio         "
                  f"{sum(r['seconds']/r['real_seconds'] for r in done)/len(done):.2%}"
                  f"   (Phase 1: control 1.76%, run 17 3.47%)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
