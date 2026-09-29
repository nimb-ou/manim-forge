"""Pick the local critic's P(YES) threshold against Gemini's verdicts.

Reads every scorecard run with both judge.json (Gemini) and judge_local.json
(local_judge.py, with "p"), and prints agreement, kappa, and precision /
recall of the local YES against Gemini's YES at each threshold. For
training data precision matters most: a false YES lets a bad beat in.

    ./.venv/bin/python scripts/local_threshold.py
"""
import glob
import json
import os

pairs = []
for f in glob.glob("data/scorecard/*/judge_local.json"):
    g = json.load(open(os.path.join(os.path.dirname(f), "judge.json")))
    m = json.load(open(f))
    for k, s in g.items():
        for n, v in s["verdicts"].items():
            p = m.get(k, {}).get("p", {}).get(n)
            if p is not None:
                pairs.append((v == "YES", p))
print(f"{len(pairs)} beats, Gemini YES {sum(g for g, _ in pairs) / len(pairs):.0%}")
for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    tp = sum(g and p >= t for g, p in pairs); fp = sum(not g and p >= t for g, p in pairs)
    fn = sum(g and p < t for g, p in pairs); tn = len(pairs) - tp - fp - fn
    n = len(pairs); a = (tp + tn) / n
    py, qy = (tp + fn) / n, (tp + fp) / n
    pe = py * qy + (1 - py) * (1 - qy)
    print(f"  t={t:.2f}  agree {a:.1%}  kappa {(a - pe) / (1 - pe):.2f}  "
          f"precision {tp / max(tp + fp, 1):.0%}  recall {tp / max(tp + fn, 1):.0%}  local YES {qy:.0%}")
