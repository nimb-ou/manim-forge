"""The other skills the system needs, and prompts that teach them.

Everything generated so far is (request -> code). That is one skill, and the
architecture needs several. A planner that cannot decompose, or a coder that
cannot read its own traceback, fails in ways more code examples will not fix.

Each task type here is a different mapping, and all of them are cheap on a free
tier we are not close to exhausting:

``PLAN``       request -> ordered beats. This is the planner's entire job and
               we have almost no direct training data for it.
``NARRATE``    a visual beat -> what a narrator says over it. Teaches the model
               to explain rather than to label.
``PARAPHRASE`` one topic -> many ways a person might ask for it. Prompts in the
               corpus are uniformly phrased, which teaches brittleness to
               wording rather than robustness.
``EXPLAIN``    code -> what it does on screen. The reverse direction; a model
               that can read Manim writes better Manim.
``CRITIQUE``   code -> what is visually wrong with it. Feeds the layout gate
               and teaches the model what "bad" looks like.
``DECOMPOSE``  a large request -> several scene-sized requests. Required for
               the 60-second explainers, and nothing else produces it.
"""

from __future__ import annotations

from dataclasses import dataclass

PLAN = """You are planning a 3Blue1Brown-style animation.

Given a request, produce ONLY a numbered list of beats. A beat is one
communicative step lasting 2-5 seconds: something appears, transforms, or is
annotated. Each line is:

  N. [seconds] intent -- narration

`intent` is what happens on screen. `narration` is what a narrator says over it.
Aim for a real explanatory arc: establish, develop, complicate, resolve. Do not
write any code.

REQUEST: {request}"""

NARRATE = """Here is one beat of a mathematical animation:

{beat}

Write one or two sentences a narrator would say over it — spoken English, as if
explaining to a person, not a caption. Do not describe the visuals literally;
say what they mean. Reply with only the narration."""

PARAPHRASE = """Someone wants an animation explaining: {topic}

Write {n} different ways a real person might type that request. Vary the
register: some terse, some rambling, some naming the concept, some describing
only what they want to see, some with a typo or an informal phrasing. One per
line, no numbering, no commentary."""

EXPLAIN = """Here is Manim code:

```python
{code}
```

Describe what appears on screen, in order, as a viewer would experience it.
Two to four sentences. Do not describe the code; describe the animation."""

CRITIQUE = """Here is Manim code for a mathematical animation:

```python
{code}
```

You are a harsh reviewer. List what would look WRONG when this renders —
elements overlapping, text off-screen or too small, things crowded, objects
left on screen from an earlier moment, motion too fast to follow, a claim
stated but never computed. Be specific and cite the line. If it is genuinely
fine, say exactly: NO ISSUES."""

DECOMPOSE = """Someone asks for: {request}

That is too large for one animation. Break it into {n} separate scenes, each
2-5 beats and self-contained, that together build the explanation in order.
Give each scene one line: a title, then a colon, then what it covers. No code."""


@dataclass
class TaskSpec:
    kind: str
    template: str
    #: Whether the output is code (and so must pass the render gate) or prose
    #: (which cannot be gated mechanically and is kept on provenance alone).
    produces_code: bool = False


SPECS = {
    "plan": TaskSpec("plan", PLAN),
    "narrate": TaskSpec("narrate", NARRATE),
    "paraphrase": TaskSpec("paraphrase", PARAPHRASE),
    "explain": TaskSpec("explain", EXPLAIN),
    "critique": TaskSpec("critique", CRITIQUE),
    "decompose": TaskSpec("decompose", DECOMPOSE),
}

#: A neutral system prompt for the prose tasks. The code system prompt insists
#: on outputting a Python block, which corrupts every one of these.
PROSE_SYSTEM = (
    "You are an expert at explaining mathematics visually, in the style of "
    "3Blue1Brown. You answer exactly what is asked, in plain text, with no "
    "preamble and no markdown code fences."
)
