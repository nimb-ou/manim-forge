"""Register finished gold scenes: export list, curriculum, retime, audit.

Four bookkeeping steps that must happen together and were being done by hand,
which is how a scene ends up rendered, committed, and invisible to training
because nobody added it to GOLD.

    ./.venv/bin/python scripts/register_gold.py \
        --scene forge/gold/foo.py:FooScene:domain,tags \
        --prompt "how a person would actually ask for it"
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path


def wrap(prompt: str, width: int = 62) -> str:
    words, lines, cur = prompt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    body = f'        "prompt": "{lines[0]} "\n'
    for ln in lines[1:-1]:
        body += f'                  "{ln} "\n'
    return body + f'                  "{lines[-1]}",\n'


def add_to_export(module: str, scene: str, prompt: str, tags: list[str]) -> bool:
    p = Path("scripts/export_gold.py")
    s = io.open(p, encoding="utf-8").read()
    if f'"{module}"' in s:
        return False
    entry = ('    {\n'
             f'        "module": "{module}",\n'
             f'        "scene": "{scene}",\n' + wrap(prompt) +
             f'        "tags": {tags!r},\n    }},\n')
    io.open(p, "w", encoding="utf-8").write(
        s.replace("GOLD = [\n", "GOLD = [\n" + entry, 1))
    return True


def mark_done(key: str) -> bool:
    p = Path("forge/gold/curriculum.py")
    s = io.open(p, encoding="utf-8").read()
    m = re.search(rf'(Scene\("{key}",.*?)(\),\n)', s, re.S)
    if not m:
        print(f"   ! no curriculum entry for {key}", file=sys.stderr)
        return False
    if "done=True" in m.group(1):
        return False
    io.open(p, "w", encoding="utf-8").write(
        s[:m.end(1)] + ",\n          done=True" + s[m.end(1):])
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", required=True,
                    help="path.py:SceneClass:tag,tag")
    ap.add_argument("--prompt", action="append", required=True,
                    help="one per --scene, in the same order")
    a = ap.parse_args()
    if len(a.scene) != len(a.prompt):
        sys.exit("need one --prompt per --scene")

    for spec, prompt in zip(a.scene, a.prompt):
        module, scene, tags = spec.split(":")
        key = Path(module).stem
        added = add_to_export(module, scene, prompt, tags.split(","))
        marked = mark_done(key)
        print(f"  {scene:<22} export={'+' if added else '='} "
              f"curriculum={'+' if marked else '='}")

    py = "./.venv/bin/python"
    print()
    subprocess.run([py, "scripts/retime_beats.py", "--apply"])
    print()
    # Writing the scene into the GOLD list is not the same as writing it into
    # the dataset. This step was missing, so data/gold/gold.jsonl sat at 12
    # rows while 44 scenes existed on disk -- 32 of them finished, committed,
    # audited, and invisible to training, which is the exact failure the
    # module docstring says this script exists to prevent.
    subprocess.run([py, "scripts/export_gold.py"], check=True)
    print()
    subprocess.run([py, "scripts/audit_narration.py", "--strict"], check=True)
    subprocess.run([py, "-c",
                    "from forge.gold import curriculum as c;"
                    "s=c.summary();print(f\"\\n{s['done']}/{s['total']} done, \""
                    "f\"{s['remaining']} left\");"
                    "print('next:', [x.key for x in c.next_up(6)])"])


if __name__ == "__main__":
    main()
