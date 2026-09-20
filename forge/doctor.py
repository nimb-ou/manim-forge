"""Re-derive every number that describes this project, and diff it.

docs/STATE.md drifted to claiming 10 gold scenes when there were 44, 89% when
the measured figure was 93%, and a corpus of ~2,300 against an honest 1,759.
The file's own first paragraph says that if it and the repo disagree the repo
is right. Nobody ran the comparison, because running it was manual.

So this is the comparison, as one command:

    python -m forge.doctor            # report, exit 1 if anything is wrong
    python -m forge.doctor --write    # update the block in STATE.md

It checks two different kinds of thing:

**Invariants** -- facts that must hold, like "every gold scene module appears
in the exported dataset". These are bugs when they fail. Three of the four
defects that silently destroyed work in this project were invariant
violations that nothing was watching.

**Figures** -- counts that change as work proceeds. These are not bugs; they
are only wrong when a document claims a stale one, which is what the marked
block in STATE.md is for.
"""

from __future__ import annotations

import argparse
import ast
import json
import pkgutil
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- doctor:begin -->", "<!-- doctor:end -->"


# --- measurement -----------------------------------------------------------

def _lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def gold_modules() -> set[str]:
    import forge.gold
    return {m.name for m in pkgutil.iter_modules(forge.gold.__path__)
            if m.name != "curriculum"}


def gold_export_list() -> list[dict]:
    src = (ROOT / "scripts" / "export_gold.py").read_text()
    ns: dict = {}
    exec(src[src.index("GOLD = ["):src.index("out = Path(")], ns)
    return ns["GOLD"]


@dataclass
class Figures:
    corpus: int = 0
    gate_passed: int = 0
    lint_recovered: int = 0
    gated_unique: int = 0
    synthetic_verified: int = 0
    gold_authored: int = 0
    gold_total: int = 0
    gold_beats: int = 0
    showcase_done: int = 0
    train: int = 0
    valid: int = 0
    narration: int = 0


def measure() -> Figures:
    from forge.gold.curriculum import CURRICULUM
    from forge.train.prepare import load_gated, load_synthetic

    d = ROOT / "data"
    gate = _rows(d / "verified" / "gate.jsonl")
    showcase = _rows(d / "showcase" / "rendered.jsonl")
    passed = {r["id"] for r in gate if r.get("ok")}
    rescued = {r["id"] for r in _rows(d / "verified" / "regate.jsonl")
               if r.get("ok") and r.get("code")} - passed
    return Figures(
        corpus=_lines(d / "normalized" / "corpus.jsonl"),
        gate_passed=len(passed),
        lint_recovered=len(rescued),
        gated_unique=len(load_gated(d / "verified" / "train.jsonl")),
        synthetic_verified=len(load_synthetic(
            d / "synthetic" / "generated.jsonl",
            d / "synthetic" / "stream.jsonl",
            d / "synthetic" / "from_narration.jsonl")),
        gold_authored=sum(1 for c in CURRICULUM if c.done),
        gold_total=len(CURRICULUM),
        gold_beats=sum(r.get("meta", {}).get("n_beats", 0)
                       for r in _rows(d / "gold" / "gold.jsonl")),
        showcase_done=len({r["scene"] for r in showcase if r.get("ok")}),
        train=_lines(d / "train" / "train.jsonl"),
        valid=_lines(d / "train" / "valid.jsonl"),
        narration=_lines(d / "style" / "narration.jsonl"),
    )


def render_block(f: Figures) -> str:
    verified = f.gate_passed + f.lint_recovered
    pct = (100 * verified / f.corpus) if f.corpus else 0
    return "\n".join([
        BEGIN,
        "",
        "| metric | value |",
        "|---|---|",
        f"| Scraped corpus | {f.corpus:,} deduplicated rows |",
        f"| — verified by the gate | **{verified:,}** ({pct:.1f}%) |",
        f"| — of those, rescued by lint | {f.lint_recovered} (no API cost) |",
        f"| Synthetic, verified | **{f.synthetic_verified:,}** unique |",
        f"| Gold scenes, authored | **{f.gold_authored} of {f.gold_total}** "
        f"· {f.gold_beats} beats |",
        f"| Gold rendered at 1080p60 | **{f.showcase_done} of {f.gold_authored}** |",
        f"| Training mix | **{f.train:,}** train / {f.valid} valid |",
        f"| 3b1b narration segments | {f.narration:,} |",
        "",
        f"*Re-derived by `python -m forge.doctor`. Do not edit by hand.*",
        END,
    ])


# --- invariants ------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    #: A warning describes a fact about data already on disk that cannot be
    #: changed and does not need to be -- it does not fail the run. An
    #: invariant describes something that must be true of what the code
    #: produces *now*. Conflating the two gives a doctor that is never clean,
    #: which is a doctor nobody runs.
    warn_only: bool = False
    #: True when the check reads data/, which is gitignored and lives on
    #: Hugging Face. A CI runner has no corpus, and a check that cannot see
    #: its input has not failed -- it has not run. Saying otherwise turns a
    #: green build red for the wrong reason, and people stop reading it.
    needs_data: bool = False
    skipped: bool = False


def invariants() -> list[Check]:
    from forge.gold.curriculum import CURRICULUM
    out: list[Check] = []
    d = ROOT / "data"
    have_data = (d / "train" / "train.jsonl").exists()

    mods = gold_modules()
    listed = {Path(g["module"]).stem for g in gold_export_list()}
    exported = {r["meta"]["id"].removeprefix("gold:")
                for r in _rows(d / "gold" / "gold.jsonl")}

    miss = sorted(mods - listed)
    out.append(Check("every authored scene is in the GOLD list", not miss,
                     f"{len(miss)} missing: {miss[:6]}" if miss else "",
                     "scripts/register_gold.py"))

    miss = sorted(listed - exported)
    out.append(Check("every listed scene reached gold.jsonl", not miss,
                     f"{len(miss)} missing: {miss[:6]}" if miss else "",
                     "./.venv/bin/python scripts/export_gold.py",
                     needs_data=True,
                     skipped=not (d / "gold" / "gold.jsonl").exists()))

    known = {c.key for c in CURRICULUM}
    unmarked = sorted((mods & known) - {c.key for c in CURRICULUM if c.done})
    out.append(Check("every authored scene is marked done", not unmarked,
                     f"{len(unmarked)}: {unmarked[:6]}" if unmarked else "",
                     "scripts/register_gold.py"))

    ids = [r["meta"]["id"] for r in _rows(d / "gold" / "gold.jsonl")]
    out.append(Check("gold ids are unique and non-positional",
                     len(set(ids)) == len(ids)
                     and not any(i.removeprefix("gold:").isdigit() for i in ids),
                     "positional or duplicated ids"))

    train = _rows(d / "train" / "train.jsonl")
    out.append(Check("training rows carry their metadata",
                     bool(train) and all("meta" in r for r in train),
                     "meta missing", "scripts/prepare_training.py",
                     needs_data=True, skipped=not have_data))

    def plays(r):
        c = r["messages"][-1]["content"]
        return c.count("self.play(")
    static = sum(1 for r in train if plays(r) == 0)
    out.append(Check("no training row is a still image", static == 0,
                     f"{static} rows never call self.play()",
                     "scripts/prepare_training.py",
                     needs_data=True, skipped=not have_data))

    gold_in_train = sum(1 for r in train if r.get("meta", {}).get("tier") == "gold")
    out.append(Check("gold scenes reached the training mix", gold_in_train > 0,
                     f"{gold_in_train} gold rows", "scripts/prepare_training.py",
                     needs_data=True, skipped=not have_data))

    # Rows the lint rescued must reach the verified export. They did not,
    # for the whole life of the project: regate.py wrote its recoveries to a
    # file no consumer ever opened.
    rec = {r["id"] for r in _rows(d / "verified" / "regate.jsonl")
           if r.get("ok") and r.get("code")}
    if rec:
        exported = {r.get("meta", {}).get("id")
                    for r in _rows(d / "verified" / "train.jsonl")}
        lost = sorted(rec - exported)
        out.append(Check("lint recoveries reached the verified export", not lost,
                         f"{len(lost)} recovered row(s) never exported: {lost[:4]}",
                         "./.venv/bin/python scripts/export_dataset.py",
                         needs_data=True, skipped=not have_data))

    from forge.catalog import orphans
    orph = orphans()
    out.append(Check("every data file is catalogued", not orph,
                     f"{len(orph)} uncatalogued: {orph[:3]}", "forge/catalog.py",
                     needs_data=True, skipped=not d.exists()))

    # The invariant that matters is about rows minted now, not rows already
    # on disk. generate_forever.py passed index=0 for every row it ever
    # wrote; the historical files still carry those collisions and always
    # will. What must hold is that a new row gets an id that identifies it.
    from forge.ingest.schema import CorpusRow
    a, b = (CorpusRow.build(source="check", license="x", prompt="p", index=None,
                            code="from manim import *\n"
                                 "class A(Scene):\n"
                                 "    def construct(self):\n"
                                 f"{body}").id
            for body in ("        self.play(Create(Dot()))\n",
                         "        self.play(Create(Dot()))\n"
                         "        self.wait()\n"))
    out.append(Check("newly minted ids identify the row", a != b,
                     "CorpusRow.build gives two different scenes one id",
                     "forge/ingest/schema.py"))

    for name in ("stream", "generated", "from_narration"):
        rows = [r for r in _rows(d / "synthetic" / f"{name}.jsonl") if r.get("ok")]
        if not rows:
            continue
        n_id = len({r.get("id") for r in rows})
        n_key = len({r.get("dedupe_key") for r in rows})
        out.append(Check(
            f"historical ids in {name}.jsonl", n_id >= n_key,
            f"{len(rows)} rows share {n_id} ids between {n_key} distinct "
            f"bodies — written before the index=0 fix; every loader keys on "
            f"dedupe_key, so nothing is lost", warn_only=True))
    return out


def running() -> list[str]:
    try:
        ps = subprocess.run(["ps", "-Ao", "pid,command"], capture_output=True,
                            text=True, timeout=10).stdout
    except Exception:
        return []
    pat = re.compile(r"(manim |forge_run|generate_forever|generate_tasks|"
                     r"regate\.py|render_showcase)")
    return [l.strip()[:90] for l in ps.splitlines()
            if pat.search(l) and "grep" not in l and "doctor" not in l]


def keys_present() -> list[tuple[str, bool]]:
    """Which provider keys exist. Names only — never the values."""
    import os
    from forge.synth.teacher import PROVIDERS, _load_dotenv
    _load_dotenv()
    seen = {}
    for name, cfg in PROVIDERS.items():
        env = cfg.get("key_env")
        if env:
            seen[name] = bool(os.environ.get(env))
    seen["gemini"] = bool(os.environ.get("GEMINI_API_KEY")
                          or os.environ.get("GOOGLE_API_KEY"))
    # Hugging Face and Kaggle both cache a login outside the environment, so
    # checking env vars alone reports a working credential as missing --
    # which this did, for a token that had been in use for a day.
    seen["huggingface"] = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        or (Path.home() / ".cache/huggingface/token").exists()
        or (Path.home() / ".huggingface/token").exists())
    seen["kaggle"] = bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (Path.home() / ".kaggle/access_token").exists()
        or (Path.home() / ".kaggle/kaggle.json").exists())
    return sorted(seen.items())


# --- report ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="update the doctor block in docs/STATE.md")
    a = ap.parse_args()

    f = measure()
    checks = invariants()
    bad = [c for c in checks if not c.ok and not c.warn_only and not c.skipped]
    warns = [c for c in checks if not c.ok and c.warn_only and not c.skipped]
    skipped = [c for c in checks if c.skipped]

    print("=" * 66)
    print("  INVARIANTS")
    print("=" * 66)
    for c in checks:
        mark = ("·· " if c.skipped else
                "ok " if c.ok else "-- " if c.warn_only else "XX ")
        print(f"  {mark} {c.name}"
              + ("   (no data/ — not run)" if c.skipped else ""))
        if not c.ok and not c.skipped:
            print(f"        {c.detail}")
            if c.fix:
                print(f"        fix: {c.fix}")

    print()
    print("=" * 66)
    print("  FIGURES")
    print("=" * 66)
    for line in render_block(f).splitlines():
        if line.startswith("|") and "---" not in line:
            print("  " + line)

    state = ROOT / "docs" / "STATE.md"
    drift = False
    # With no data/ every figure is zero, so the block would always look
    # stale. A runner that cannot see the corpus has nothing to say about
    # whether the corpus numbers are current.
    have_data = (ROOT / "data" / "train" / "train.jsonl").exists()
    if state.exists() and have_data:
        s = state.read_text()
        block = render_block(f)
        if BEGIN in s and END in s:
            current = s[s.index(BEGIN):s.index(END) + len(END)]
            drift = current.strip() != block.strip()
            if drift and a.write:
                state.write_text(s.replace(current, block))
                print("\n  STATE.md updated")
                drift = False
        elif a.write:
            state.write_text(s.rstrip() + "\n\n" + block + "\n")
            print("\n  doctor block added to STATE.md")
        else:
            drift = True

    print()
    print("=" * 66)
    print("  ENVIRONMENT")
    print("=" * 66)
    for name, have in keys_present():
        print(f"  {'key ' if have else '--  '} {name}")
    live = running()
    print(f"\n  {len(live)} forge/manim process(es) running")
    for l in live[:6]:
        print(f"      {l}")

    print()
    if skipped:
        print(f"  {len(skipped)} check(s) need data/ and did not run")
    if warns:
        print(f"  {len(warns)} note(s) about data already on disk — not faults")
    if bad:
        print(f"  {len(bad)} invariant(s) broken")
    if drift:
        print("  docs/STATE.md is stale — run with --write")
    if not have_data:
        print("  no data/ — figures not checked "
              "(restore with scripts/backup_to_hf.py --restore)")
    if not bad and not drift:
        print("  clean")
    sys.exit(1 if (bad or drift) else 0)


if __name__ == "__main__":
    main()
