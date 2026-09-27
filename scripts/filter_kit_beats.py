"""Keep the kit beats whose picture is about their beat, and is new.

The first full teacher scene I looked at was kit-shaped slideware: every
beat of an arc about Escher's Print Gallery drew the same plane with two
vectors under a different title. It renders and it "draws something", so
the visual-beat filter passed it; trained on, it teaches "draw vectors for
everything". Two filters, both deterministic (the LLM judges tried said YES
to "Escher's lithograph" drawn as two vectors):

  * relevance -- each kit family (linear algebra, calculus, waves, complex
    numbers, chance, numbers, geometry, networks and algorithms) has words
    that its pictures are about; a beat is kept only if a family it draws
    with matches the beat's own intent, narration or request. A beat
    drawing with raw Manim only is kept (this cannot judge it);
  * novelty -- a beat whose kit calls, with their literal arguments, repeat
    an earlier beat's in the same scene is the same picture again.

    ./.venv/bin/python scripts/filter_kit_beats.py
      -> data/kit/kit_beats_clean.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "kit" / "kit_beats.jsonl"
OUT = ROOT / "data" / "kit" / "kit_beats_clean.jsonl"

from forge.kit.families import BLOCK_FAMILY, FAMILIES, kit_calls  # noqa: E402,F401



def beat_text(row: dict) -> str:
    user = row["messages"][1]["content"]
    got = re.findall(r"^\s*(?:intent|narration):\s*(.*)$", user, re.M)
    return (" ".join(got) + " " + row.get("request", "")).lower()


def body(row: dict) -> str:
    text = row["messages"][2]["content"]
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def main() -> int:
    # Teacher rows, the kit coder's own (self_kit_beats.py) and the
    # hand-written ones (claude_kit_scenes.py) alike.
    rows = []
    for src in (SRC, ROOT / "data" / "kit" / "self_beats.jsonl",
                ROOT / "data" / "kit" / "claude_beats.jsonl"):
        if src.exists():
            rows += [json.loads(l) for l in src.read_text().splitlines()
                     if l.strip()]
    # The visual critic's verdicts (critic_kit_scenes.py), where it has run:
    # a beat whose frame a vision model judged not to show its idea is out.
    critic = ROOT / "data" / "kit" / "critic.jsonl"
    no = set()
    if critic.exists():
        for l in critic.read_text().splitlines():
            if l.strip():
                c = json.loads(l)
                no |= {(c["scene"], int(k)) for k, v in c["verdicts"].items()
                       if v == "NO"}
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["meta"]["scene"]].append(r)
    why = Counter()
    kept = []
    for scene, rs in by_scene.items():
        rs.sort(key=lambda r: r["meta"]["index"])
        seen: set[tuple] = set()
        for r in rs:
            if (scene, r["meta"]["index"]) in no:
                why["dropped: the visual critic said no"] += 1
                continue
            calls = kit_calls(body(r))
            if not calls:
                kept.append(r)
                why["kept (raw Manim, not judged)"] += 1
                continue
            sig = tuple(sorted(calls))
            if sig in seen:
                why["dropped: repeats an earlier picture"] += 1
                continue
            seen.add(sig)
            text = beat_text(r)
            fams = {BLOCK_FAMILY[b] for b, _ in calls}
            if not any(re.search(FAMILIES[f][1], text) for f in fams):
                why["dropped: picture not about the beat"] += 1
                continue
            kept.append(r)
            why["kept"] += 1
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    print(f"{len(kept)} of {len(rows)} rows kept -> {OUT}")
    for k, v in why.most_common():
        print(f"  {k:40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
