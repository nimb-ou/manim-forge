"""What is wrong with a rendered draft, in sentences the model can act on.

Best of 2 replaces a flawed scene with another one drawn blind; the second
is often flawed the same way. Graded by eye, the commonest fault on the
world set (2026-10-09) was a beat whose intent promised a motion -- "the
grid becomes a parallelogram", "the agent overshoots and bounces back" --
and whose code only changed the caption: 11 of 30 scenes had one. The
model can fix that when told which beat and what is missing. So a draft is
checked here, the problems are written out, and the one-shot generator
hands them back with the draft for a rewrite (forge/app/oneshot.py).

Three sources, all cheap:
  * the code itself: beats that draw nothing (static_problems);
  * the arithmetic on screen (forge/app/checks.py);
  * what the kit saw while rendering -- points off their axes, curves
    leaving their axes, text on text, empty pictures (``ISSUE`` lines from
    forge/kit/kit.py, collected into Result.issues by pipeline.finish).
"""
from __future__ import annotations

import re

# Calls that put no picture on screen, or are plain Python.
_TEXT_ONLY = re.compile(
    r"^(?:stage\.(?:caption|title|pause|mark|label|wait)|self\.wait|range|len|int|float|"
    r"round|str|abs|min|max|sum|print|list|tuple|enumerate|zip|sorted|"
    r"np\.\w+|math\.\w+|f)$")
_MOTION = re.compile(
    r"\b(appl(?:y|ies|ied)|transform\w*|mov(?:e|es|ing)|slid\w*|rotat\w*|grow\w*|draw\w*|"
    r"shad\w*|fill\w*|split\w*|roll\w*|drop\w*|fall\w*|bounc\w*|step\w*|walk\w*|"
    r"shift\w*|stretch\w*|flip\w*|swap\w*|overshoot\w*|creep\w*|sweep\w*|trac\w*|"
    r"plot\w*|mark\w*|highlight\w*|arrow\w*|tilt\w*|shear\w*|distort\w*|becom\w*)\b", re.I)


def _calls(body: str) -> list[str]:
    code = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
    code = re.sub(r"""(["'])(?:\\.|(?!\1).)*\1""", '""', code)      # no calls inside strings
    return re.findall(r"([A-Za-z_][\w.]*)\s*\(", code)


def static_problems(beats, bodies) -> list[tuple[int, str]]:
    """Beats whose code puts no picture on screen. A beat that only writes
    an equation is fine (the rule, the payoff) unless its intent is a
    motion."""
    out = []
    for n, (b, body) in enumerate(zip(beats, bodies), 1):
        intent = (b.intent or "").strip()
        if not body.strip():
            out.append((n, "its code failed and was dropped; write it again, simply, "
                           "with kit blocks"))
            continue
        calls = _calls(body)
        drawing = [c for c in calls if not _TEXT_ONLY.match(c) and c != "stage.equation"]
        if drawing:
            continue
        if "stage.equation" not in calls:
            out.append((n, f"it only changes the text, so nothing on screen shows "
                           f"\"{intent}\"; draw or animate that with the kit"))
        elif _MOTION.search(intent):
            out.append((n, f"it only writes an equation, but the beat is \"{intent}\"; "
                           f"animate that with the kit as well"))
    return out


def problems(res) -> list[str]:
    """Every problem found in a rendered (or failed) draft, one sentence
    each, beat by beat, without repeats."""
    from forge.app.checks import arithmetic_errors
    out: list[str] = []
    if not res.ok:
        why = getattr(res, "failure", "") or res.error or "error"
        out.append(f"the scene did not render: {why}; fix or drop that statement, and "
                   "use only kit blocks and names from the examples")
    found = static_problems(res.beats, res.bodies) + list(getattr(res, "issues", []))
    for n, what in sorted(found, key=lambda x: x[0]):
        line = f"beat {n}: {what}" if n else what
        if line not in out:
            out.append(line)
    for slip in arithmetic_errors("\n".join(res.bodies)):
        out.append(f"on screen, \"{slip}\" is false; work the numbers out again")
    return out[:12]


def revise_prompt(user: str, draft: str, found: list[str]) -> str:
    """The original request with the draft and its problems, asking for the
    whole scene again."""
    return (f"{user}\n\nYOUR DRAFT\n{draft.strip()}\n\n"
            "PROBLEMS FOUND WHEN THE DRAFT WAS RENDERED\n"
            + "\n".join(f"- {p}" for p in found)
            + "\n\nWrite the whole scene again in the same format, fixing every "
              "problem above. Keep what already works, and keep to 3-6 beats.")
