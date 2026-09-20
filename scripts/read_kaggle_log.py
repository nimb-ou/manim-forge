"""Print a Kaggle kernel log, which is JSON rather than text.

`kaggle kernels output` writes <slug>.log, and that file is a JSON array of
{stream_name, time, data} objects. `tail` on it shows one enormous line, so
the first failed run's traceback -- a SyntaxError on line 19 -- was in the
artefact and unreadable in the workflow output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    where = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    logs = sorted(where.glob("*.log"))
    if not logs:
        print(f"no .log in {where}")
        return
    for f in logs:
        print(f"===== {f.name} =====")
        try:
            entries = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f.read_text()[-4000:])
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            text = str(e.get("data", "")).rstrip()
            if text:
                print(f"[{e.get('stream_name', '?')}] {text}")


if __name__ == "__main__":
    main()
