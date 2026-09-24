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

from forge.app.twostage import (assemble, beat_prompt, extract_code,
                                failing_beat, intent_key, missing_names,
                                names_in_scope, parsing_prefix,
                                prelude_prompt, prune_statements,
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
    "statements at method-body level -- no class, no def, no imports. "
    "NAMES IN SCOPE lists what earlier beats already built: reuse those "
    "rather than rebuilding them. Anything else you use you must "
    "CONSTRUCT in this beat before you animate it -- `self.play(Create(dot))` "
    "is wrong unless a line above it makes `dot`."
)


def load(adapter: str | None):
    from mlx_lm import load as mlx_load
    return mlx_load(MODEL, **({"adapter_path": adapter} if adapter else {}))


def ask(model, tok, system: str, user: str, max_tokens: int,
        temp: float = 0.0, rep_penalty: float = 0.0) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    chat = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=False)
    procs = (make_logits_processors(repetition_penalty=rep_penalty,
                                    repetition_context_size=256)
             if rep_penalty else None)
    return generate(model, tok, prompt=chat, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temp, top_p=0.95 if temp else 0.0),
                    logits_processors=procs, verbose=False)


#: Planner decoding, set from --plan-temp / --plan-rep-penalty. Sampled by
#: default: on eight hard titles planner v2 wrote 3.8 beats before its first
#: repeated intent under greedy decoding and 22.6 at temperature 0.5 with a
#: 1.1 repetition penalty -- real arcs, not paraphrased loops. The coder
#: stays greedy.
PLAN_DECODE = {"temp": 0.5, "rep_penalty": 1.1}


def plan(model, tok, request: str, stride: int, max_beats: int) -> list:
    """Feed the planner its own output until it says END or hits the cap."""
    beats: list = []
    for _ in range(max_beats // stride + 1):
        # A sampled beat can come back without a duration; "[?]" keeps the
        # history line well-formed instead of crashing the run at beat 20.
        shown = "\n".join(f"{b.n}. [{f'{b.seconds:g}s' if b.seconds else '?'}] "
                           f"{b.intent}"
                          for b in beats[-12:]) or \
            "  (nothing yet — open the explanation)"
        reply = ask(model, tok, PLAN_SYSTEM,
                    f"REQUEST\n{request}\n\nBEATS SO FAR\n{shown}\n\n"
                    f"Write the next {stride} beat(s), numbered from "
                    f"{len(beats) + 1}.", max_tokens=700, **PLAN_DECODE)
        new, ended = parse_plan(reply, limit=stride)
        new = [b for b in new if b.n > len(beats)]
        # A repeated intent ends the arc. Planner v2 wrote 38-51 repeats in
        # every 48-beat plan and never wrote END: greedy decoding fed its own
        # output falls into a loop, and half its synthetic training arcs
        # padded that way. The training windows now cut synthetic arcs at
        # their first repeat, and this is the same rule at inference.
        seen = {intent_key(b.intent) for b in beats}
        for k, b in enumerate(new):
            key = intent_key(b.intent)
            if key in seen:
                new, ended = new[:k], True
                break
            seen.add(key)
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
    ap.add_argument("--plan-temp", type=float, default=0.5)
    ap.add_argument("--plan-rep-penalty", type=float, default=1.1)
    ap.add_argument("--salvage", action="store_true",
                    help="after repair, drop beats that still use undefined "
                         "names and render the rest, if half survive")
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
    PLAN_DECODE.update(temp=a.plan_temp, rep_penalty=a.plan_rep_penalty)

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
                            beat_prompt(req, beats, j, bodies),
                            max_tokens=a.beat_tokens)
                cand = extract_code(reply)
                try:
                    ast.parse(textwrap.dedent(cand))
                except SyntaxError as exc:
                    why = f"beat {j + 1} did not parse: {exc.msg}"
                    continue
                body, why = cand, ""
                break
            if why:
                # A truncated beat keeps what it wrote before the cut, if
                # that still animates something.
                head = parsing_prefix(cand)
                if "self.play(" in head:
                    body = head
                    why += " (kept the parsing prefix)"
                dropped.append(why)
            bodies.append(body)
        asm = assemble(beats, bodies)

        # One targeted repair of the split's own failure. When a beat uses a
        # name no beat defines, the beat is asked again with the offending
        # names quoted back -- which is information the first prompt could
        # not contain, because it did not know what the model would invent.
        missing = [p for p in asm.problems if p.startswith("beats use names")]
        if missing and beats:
            # Logged, because "the repair exists" and "the repair runs" are
            # different claims and I had been assuming the second from the
            # first.
            print(f"      repairing: {missing[0][:80]}", flush=True)
            names = [n.strip() for n in
                     missing[0].split(":", 1)[1].split(",") if n.strip()]
            scope = names_in_scope(bodies)
            # Repair the *first* beat that uses each missing name, not every
            # beat that uses it. The old loop rewrote beats 2, 3, 4 and 5 --
            # all users -- and left the scene just as broken, because none of
            # them had been made the definer. Whoever mentions an object
            # first is the one that has to build it.
            owners: dict[int, list[str]] = {}
            for n in names:
                for j in range(len(beats)):
                    if re.search(rf"\b{re.escape(n)}\b", bodies[j]):
                        owners.setdefault(j, []).append(n)
                        break
            for j, b in enumerate(beats):
                if j not in owners:
                    continue
                mine = ", ".join(owners[j])
                reply = ask(cm, ctok, CODE_SYSTEM,
                            f"REQUEST\n{req}\n\nNAMES IN SCOPE\n  "
                            + (", ".join(scope) if scope else "(none yet)")
                            + f"\n\nYour previous attempt used {mine}, which "
                              f"nothing in the scene creates. This beat is "
                              f"the first to mention {mine}, so CONSTRUCT "
                              f"{mine} here -- an assignment line for each, "
                              f"before any self.play that uses it -- and then "
                              f"animate as the intent asks.\n\n"
                              f"WRITE THIS BEAT — step {j + 1} of {len(beats)}"
                              f"\n  intent: {b.intent}", max_tokens=a.beat_tokens)
                cand = extract_code(reply)
                try:
                    ast.parse(textwrap.dedent(cand))
                except SyntaxError:
                    continue
                # Keep a rewrite only if the scene is left with fewer
                # missing names. Asked to fix `self.mobjects`, the coder
                # rewrote the beat around `dot, square, triangle` -- three
                # new unknowns for one -- and the worse version was kept.
                def n_missing(bs):
                    probs = [q for q in assemble(beats, bs).problems
                             if "names no beat" in q]
                    return len(probs[0].split(":", 1)[1].split(",")) \
                        if probs else 0
                trial = list(bodies)
                trial[j] = cand
                if n_missing(trial) >= n_missing(bodies):
                    print(f"      beat {j + 1} rewrite rejected "
                          f"(no fewer missing names)", flush=True)
                    continue
                bodies[j] = cand
                print(f"      beat {j + 1} rewritten", flush=True)
            asm = assemble(beats, bodies)
            print(f"      after repair: "
                  f"{asm.problems[0][:70] if asm.problems else 'clean'}",
                  flush=True)
        # Set-up repair: build what every beat assumes, once, up front.
        missing_now = missing_names(beats, bodies)
        if missing_now and any(b.strip() for b in bodies):
            reply = ask(cm, ctok, CODE_SYSTEM,
                        prelude_prompt(req, bodies, missing_now),
                        max_tokens=a.beat_tokens)
            pre = parsing_prefix(extract_code(reply))
            with (ROOT / "data" / "logs" / "setup_replies.jsonl").open("a") as f:
                f.write(json.dumps({"request": req[:80], "missing": missing_now,
                                    "reply": reply[:3000]}) + "\n")
            try:
                if not pre.strip():
                    raise SyntaxError("empty")
                ast.parse(textwrap.dedent(pre))
                first = next(k for k, b in enumerate(bodies) if b.strip())
                trial = list(bodies)
                trial[first] = textwrap.dedent(pre).strip() + "\n" + \
                    textwrap.dedent(trial[first])
                after = missing_names(beats, trial)
                if len(after) < len(missing_now):
                    bodies = trial
                    asm = assemble(beats, bodies)
                    print(f"      set-up built {len(missing_now) - len(after)}"
                          f" of {len(missing_now)} missing names", flush=True)
                else:
                    print("      set-up rejected (no fewer missing names)",
                          flush=True)
            except SyntaxError:
                print("      set-up did not parse", flush=True)
        salvaged = 0
        if a.salvage:
            # A partial scene over no scene, reported as partial. The beat
            # that should have built `circle` was dropped for not parsing,
            # and every later beat that animates `circle` then fails the
            # whole scene. Blanking the users -- repeatedly, because a
            # blanked beat can have been the definer of something else --
            # keeps the beats that stand on their own. Counted, so a salvaged
            # render is never mistaken for a whole one.
            live = sum(1 for b in bodies if b.strip())
            # Statements first: remove only the lines that read a missing
            # name, and what those lines fed, keeping the rest of each beat.
            # Kept if the scene is then clean and half the beats still play.
            pruned_stmts = 0
            miss_now = missing_names(beats, bodies)
            if miss_now:
                trial, pruned_stmts = prune_statements(bodies, miss_now)
                t_asm = assemble(beats, trial)
                playing = sum(1 for b in trial if "self.play(" in b)
                if t_asm.ok and playing * 2 >= live:
                    bodies, asm = trial, t_asm
                    print(f"      pruned {pruned_stmts} statements using "
                          f"{', '.join(miss_now[:4])}", flush=True)
                else:
                    pruned_stmts = 0
            for _ in range(len(bodies)):
                miss = [p for p in asm.problems if "names no beat" in p]
                if not miss:
                    break
                names = [n.strip() for n in
                         miss[0].split(":", 1)[1].split(",") if n.strip()]
                hit = False
                for j, body in enumerate(bodies):
                    if body.strip() and any(
                            re.search(rf"\b{re.escape(n)}\b", body)
                            for n in names):
                        bodies[j] = ""
                        salvaged += 1
                        hit = True
                if not hit:
                    break
                asm = assemble(beats, bodies)
            if salvaged and salvaged * 2 > live:
                asm.problems.append(f"salvage would drop {salvaged} of {live} "
                                    f"beats; not rendered")
            elif salvaged:
                print(f"      salvaged: {salvaged} of {live} beats dropped",
                      flush=True)
        # Render on the assembly's own verdict, then record the dropped
        # beats. Extending first made every scene with one unparsable beat
        # unrenderable -- `ok` is "no problems" -- so the drop-one-beat rule
        # above cost the whole scene after all, and was reported as an
        # assembly failure.
        renderable = asm.ok
        asm.problems.extend(dropped)
        res = h.render(asm.code, quality="low", frames=4) if renderable \
            else None
        if a.salvage and res is not None and not res.ok:
            # Runtime salvage: one beat calling `self.camera.frame` kills a
            # 24-beat scene. Drop the beat the traceback points at and
            # render again, three times at most, under the same half rule.
            live = sum(1 for b in bodies if b.strip())
            for _ in range(3):
                k = failing_beat(asm.code, res.stderr or "")
                if k is None or not bodies[k - 1].strip():
                    break
                bodies[k - 1] = ""
                salvaged += 1
                if salvaged * 2 > live:
                    break
                trial = assemble(beats, bodies)
                if not trial.ok:
                    break
                asm = trial
                print(f"      runtime salvage: beat {k} dropped, re-rendering",
                      flush=True)
                res = h.render(asm.code, quality="low", frames=4)
                if res.ok:
                    break
        row = {"index": i - 1, "request": req, "beats": len(beats),
               "assembled": renderable,
               "problems": asm.problems, "dropped": len(dropped),
               "salvaged": salvaged,
               "pruned_statements": pruned_stmts if a.salvage else 0,
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
