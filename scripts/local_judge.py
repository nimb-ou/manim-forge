"""The local vision judge (forge/evaluate/local_vision.py) on scorecard runs.

    --calibrate TAG...   judge every beat Gemini judged in those runs and
                         print agreement, kappa and the confusion table
    TAG...               judge the runs; writes judge_local.json beside
                         judge.json and prints the share per tag

    ./.venv/bin/python scripts/local_judge.py --calibrate ab_held_names ab_short_names
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.evaluate.local_vision import LocalJudge  # noqa: E402
from judge_sheets import tiles  # noqa: E402


def judge_tag(j: LocalJudge, tag: str) -> dict:
    """{scene key: {"title", "intents", "verdicts"}} for one run."""
    d = ROOT / "data" / "scorecard" / tag
    rows = json.loads((d / "scorecard.json").read_text())["rows"]
    out, tmp = {}, Path(tempfile.mkdtemp())
    for i, r in enumerate(rows, 1):
        k = f"{i:02d}"
        py = next(iter(sorted(d.glob(f"{k}-*.py"))), None)
        sheet = next(iter(sorted(d.glob(f"{k}-*.jpg"))), None)
        if not r["ok"] or py is None or sheet is None:
            continue
        import re
        intents = re.findall(r"^\s*# beat \d+:\s*(.*)$", py.read_text(), re.M)
        imgs = tiles(sheet, len(intents))
        if not intents or len(imgs) != len(intents):
            continue
        v, ps = {}, {}
        for n, (b, t) in enumerate(zip(imgs, intents), 1):
            f = tmp / f"{k}_{n}.jpg"
            f.write_bytes(b)
            pv = j.p_yes(r["title"], t, f)
            v[str(n)] = "YES" if pv >= 0.5 else "NO"
            ps[str(n)] = round(pv, 4)
        print(f"    {tag} {k}: {sum(x == 'YES' for x in v.values())}/{len(v)}", flush=True)
        out[k] = {"title": r["title"], "intents": intents, "verdicts": v, "p": ps}
    (d / "judge_local.json").write_text(json.dumps(out, indent=1))
    planned = sum(r["beats"] for r in rows)
    yes = sum(sum(x == "YES" for x in s["verdicts"].values()) for s in out.values())
    print(f"{tag:24s} local judge share {yes / planned:.0%}  ({yes}/{planned})", flush=True)
    return out


def main() -> int:
    args = sys.argv[1:]
    cal = "--calibrate" in args
    tags = [a for a in args if not a.startswith("--")]
    j = LocalJudge()
    tab = {("YES", "YES"): 0, ("YES", "NO"): 0, ("NO", "YES"): 0, ("NO", "NO"): 0}
    for tag in tags:
        mine = judge_tag(j, tag)
        if not cal:
            continue
        gem = json.loads((ROOT / "data" / "scorecard" / tag / "judge.json").read_text())
        for k, s in gem.items():
            for n, g in s["verdicts"].items():
                m = mine.get(k, {}).get("verdicts", {}).get(n)
                if m:
                    tab[(g, m)] += 1
    if cal:
        n = sum(tab.values())
        agree = (tab[("YES", "YES")] + tab[("NO", "NO")]) / n
        py = (tab[("YES", "YES")] + tab[("YES", "NO")]) / n       # Gemini YES rate
        qy = (tab[("YES", "YES")] + tab[("NO", "YES")]) / n       # local YES rate
        pe = py * qy + (1 - py) * (1 - qy)
        print(f"\n{n} beats  agreement {agree:.1%}  kappa {(agree - pe) / (1 - pe):.2f}")
        print(f"  gemini YES: local YES {tab[('YES', 'YES')]}  local NO {tab[('YES', 'NO')]}")
        print(f"  gemini NO : local YES {tab[('NO', 'YES')]}  local NO {tab[('NO', 'NO')]}")
        print(f"  YES rate gemini {py:.0%}  local {qy:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
