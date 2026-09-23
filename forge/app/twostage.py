"""Plan then implement: the two-stage generator.

One model doing both is what run 17 measured. It plans well now -- beats 0 ->
4.54, coverage 8.4% -> 18.8%, length 1.76% -> 3.47% -- and then has to
implement all of that in a single pass, where its failures run 80% longer
than its passes and it hand-builds a `Polyhedron` where `Dodecahedron()`
exists. Both attempts to fix that from outside the weights did nothing: a
pre-render API check catches 1 of 23, and a worked example in the repair
prompt moved 77% to 77%.

So split the job. The planner writes the arc, in windows, feeding its own
output back until it writes END -- ambition expressed in text cannot fail to
render. The coder writes one beat at a time against a stated intent and the
objects already on screen. Each coder call is a few dozen lines with its
setup handed to it, rather than a whole scene invented from one sentence.

**Assembly is the new failure mode**, and it is the honest risk of the
split: a single model cannot fail this way because it never assembles
anything. Two beats can name the same variable, or a later beat can use one
an earlier beat never made. So the assembled scene is parsed before it is
rendered, and undefined names are reported as an assembly failure rather
than left to surface as a confusing render error.
"""
from __future__ import annotations

import ast
import re
import textwrap
import textwrap
from dataclasses import dataclass, field

# The bracketed duration is optional. The untuned model answers
#
#     1. 0.000 intent -- Start with a blank screen.
#
# -- no brackets, and the word "intent" copied straight out of the format
# spec. Requiring brackets scored that as zero beats, which reads as "the
# planner produced nothing" when what happened is "the planner produced an
# arc in a different format". Those are opposite findings and the measurement
# has to tell them apart; teaching the format is the adapter's job, and the
# parser should not do it by fiat.
BEAT = re.compile(
    r"^\s*(\d+)[.)]\s*"
    r"(?:\[([^\]]*)\]|(\d+(?:\.\d+)?)\s*s?\b)?\s*"
    r"(.*?)(?:\s+--\s+(.*))?$")
PREAMBLE = "from manim import *\nimport numpy as np\nimport math\n"


@dataclass
class Beat:
    n: int
    seconds: float | None
    intent: str
    narration: str = ""


@dataclass
class Assembly:
    code: str
    beats: list[Beat] = field(default_factory=list)
    bodies: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.bodies)


def parse_plan(text: str, limit: int = 40) -> tuple[list[Beat], bool]:
    """Beats out of a planner window, and whether it declared the arc over.

    `ended` means the planner wrote END. It does **not** mean the limit was
    reached, and conflating the two capped every plan in the hard eval at six
    beats: the driver asks for a window of six, the parser hit its limit and
    said "ended", and the driver believed it and never asked for a seventh
    beat. Real arcs run about 40. The measurement would have shown the
    planner producing short explanations when the planner was never asked.
    """
    beats, ended = [], False
    for line in text.splitlines():
        if line.strip() == "END":
            ended = True
            continue
        m = BEAT.match(line)
        if not m:
            continue
        secs = None
        raw = (m.group(2) or m.group(3) or "").strip().rstrip("s")
        try:
            secs = float(raw)
        except ValueError:
            pass
        intent = m.group(4).strip()
        if not intent:
            continue
        beats.append(Beat(int(m.group(1)), secs, intent,
                          (m.group(5) or "").strip()))
        if len(beats) >= limit:
            break
    return beats, ended


# Chat templates end a turn with a special token, and mlx-lm hands it back
# in the generated string. Left in place it is the last thing on the line:
#
#     circle.become(square)<|im_end|>
#
# which does not parse. That was reported as "beat did not parse: invalid
# syntax" on nearly every beat of the untuned run, and read as the coder
# being unable to write a statement -- when the statement was correct and the
# harness was breaking it. Anything from the first such token is dropped.
STOP = re.compile(r"<\|(?:im_end|endoftext|eot_id|end_of_text)\|>")


def extract_code(text: str) -> str:
    """The largest fenced block, or the whole thing if it is unfenced."""
    text = STOP.split(text, 1)[0]
    if "```" not in text:
        return text.strip()
    body = max(text.split("```"), key=len)
    if body.startswith("python"):
        body = body[len("python"):]
    return body.strip("\n")


def defined_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
    return out


def names_in_scope(bodies: list[str]) -> list[str]:
    """Variables earlier beats left behind, for the next beat's prompt.

    The untuned run failed with "beats use names no beat defines: Diagram,
    radius, show_smaller_angle" -- the coder invented objects because nothing
    told it what already existed. Listing intents is not enough: "a circle
    appears" does not say the circle is called `c`.

    Skipped names that are almost certainly not reusable state: loop and
    comprehension targets, which are dead outside their block.
    """
    out: list[str] = []
    for src in bodies:
        try:
            tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            continue
        transient: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                transient |= {t.id for t in ast.walk(node.target)
                              if isinstance(t, ast.Name)}
            elif isinstance(node, ast.comprehension):
                transient |= {t.id for t in ast.walk(node.target)
                              if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) \
                    and node.id not in transient and node.id not in out:
                out.append(node.id)
            # `self.x = ...` as well as `x = ...`. The gold scenes carry
            # state between beats on self, because that is how @beat methods
            # talk to each other, while the decomposed corpus rows use plain
            # locals. Unifying the output *format* left the two state
            # conventions in place, and gold is weighted six times -- so the
            # coder learned the self idiom and then read `self.axes` in beat
            # one, having been told the scene had no names in it at all.
            elif isinstance(node, ast.Attribute) \
                    and isinstance(node.ctx, ast.Store) \
                    and isinstance(node.value, ast.Name) \
                    and node.value.id == "self":
                name = f"self.{node.attr}"
                if name not in out:
                    out.append(name)
    return out


def assemble(beats: list[Beat], bodies: list[str],
             scene: str = "ForgeScene") -> Assembly:
    """Concatenate beat bodies into one renderable scene, and check it.

    Names are checked against Manim's exports plus the builtins plus whatever
    the scene itself binds. Anything left over is a beat referring to
    something no beat created -- the failure the split introduces, caught
    here rather than in a render log.
    """
    body = "\n\n".join(
        f"        # beat {b.n}: {b.intent}\n"
        + textwrap.indent(textwrap.dedent(src).strip("\n"), " " * 8)
        for b, src in zip(beats, bodies) if src.strip())
    # A trailing hold, unless the last beat already ends on one. A scene
    # whose final beat is construction renders zero frames and the harness
    # calls it `empty_render` -- which is true and unhelpful, because the
    # scene is fine and nobody told it to stay on screen. A single model
    # rarely hits this because it writes the ending itself; a concatenation
    # of beats has no ending unless one is added.
    tail = "" if re.search(r"self\.wait\([^)]*\)\s*$", body) else \
        "\n        self.wait(1)"
    code = (f"{PREAMBLE}\n\nclass {scene}(Scene):\n"
            f"    def construct(self):\n{body}{tail}\n")
    out = Assembly(code=code, beats=list(beats), bodies=list(bodies))
    if not any(src.strip() for src in bodies):
        # Reported as itself. An empty construct fails to parse, and
        # "expected an indented block" is a confusing way to say that the
        # planner returned nothing.
        out.problems.append("no beats to assemble")
        return out

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        out.problems.append(f"assembled scene does not parse: {exc}")
        return out

    import builtins
    known = defined_names(tree) | set(dir(builtins)) | {"self"}
    try:
        import manim
        known |= {n for n in dir(manim) if not n.startswith("_")}
    except Exception:                                         # noqa: BLE001
        out.problems.append("manim not importable; names unchecked")
        return out

    # Attributes read off self that nothing assigns. Manim's Scene supplies
    # some -- camera, renderer, mobjects -- so only names no beat sets and
    # Scene does not define are reported.
    try:
        import manim as _m
        scene_attrs = set(dir(_m.Scene))
    except Exception:                                         # noqa: BLE001
        scene_attrs = set()
    # Instance attributes Scene.__init__ sets, which dir(Scene) cannot see.
    # `FadeOut(*self.mobjects)` is the idiomatic clear-the-screen, and
    # flagging it sent three of eight scenes to a repair that made them worse.
    scene_attrs |= {"mobjects", "camera", "renderer", "foreground_mobjects",
                    "time", "moving_mobjects", "static_mobjects"}
    set_attrs = {n.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute)
                 and isinstance(n.ctx, ast.Store)
                 and isinstance(n.value, ast.Name) and n.value.id == "self"}
    missing = sorted({
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        and n.id not in known}
        | {f"self.{n.attr}" for n in ast.walk(tree)
           if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
           and isinstance(n.value, ast.Name) and n.value.id == "self"
           and n.attr not in set_attrs and n.attr not in scene_attrs})
    if missing:
        out.problems.append(
            "beats use names no beat defines: " + ", ".join(missing[:8]))
    return out
