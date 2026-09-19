"""Beats — the unit of work for Manim Forge.

A **beat** is one communicative step: something appears, transforms, or gets
annotated. Two to five seconds. A 3-second animation is one beat; a 60-second
explainer is fifteen composed beats. Same system, one parameter.

Beats are built on Manim's own *sections* rather than on a bespoke mechanism.
``next_section()`` plus ``--save_sections`` makes the renderer emit one mp4 per
beat, which buys per-beat video, timing and caching without inventing anything.

Why this is the unit:

* **Failure is local.** One bad beat is a three-second hole, not a dead video.
* **Generation is tractable.** Asking a 7B model for fifteen lines succeeds far
  more often than asking for three hundred.
* **Editing is possible.** Regenerate beat 7 and leave the rest alone — which
  is what makes a timeline editor feasible later.
* **Data multiplies.** A fifteen-beat scene is fifteen training examples, not
  one. That matters when the public corpus is as thin as this one is.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from itertools import count

from manim import Scene, Mobject

_ORDER = count()


@dataclass
class BeatSpec:
    """What a beat declares about itself, independent of Manim."""
    name: str
    intent: str
    seconds: float | None
    order: int
    source: str = ""

    def as_instruction(self) -> str:
        """The line a planner emits and a coder consumes."""
        dur = f" (~{self.seconds:g}s)" if self.seconds else ""
        return f"{self.intent}{dur}"


def beat(intent: str, seconds: float | None = None):
    """Mark a method as one beat.

    ``intent`` is the natural-language description of what this step
    communicates — it is both documentation and the training target for the
    planner model, so write it the way a person would ask for it::

        @beat("Slide the rook to the corner and re-count its moves", seconds=5)
        def corner(self):
            ...
    """
    def decorate(fn):
        fn._forge_beat = BeatSpec(
            name=fn.__name__,
            intent=intent,
            seconds=seconds,
            order=next(_ORDER),
        )
        return fn
    return decorate


def beats_of(cls) -> list:
    """Every beat method on a scene class, in definition order.

    Ordered by the decorator's global counter rather than by ``dir()``, which
    sorts alphabetically and would silently scramble the narrative.
    """
    found = []
    for _, fn in inspect.getmembers(cls, predicate=callable):
        spec = getattr(fn, "_forge_beat", None)
        if spec is not None:
            found.append(fn)
    return sorted(found, key=lambda f: f._forge_beat.order)


def storyboard(cls) -> list[BeatSpec]:
    """The beat specs alone — the planner's output format, and what we show
    the user as an editable outline before committing to a render."""
    return [f._forge_beat for f in beats_of(cls)]


@dataclass
class StageItem:
    name: str
    kind: str
    position: tuple[float, float]

    def __str__(self) -> str:
        x, y = self.position
        return f"{self.name}: {self.kind} at ({x:+.1f}, {y:+.1f})"


class ForgeScene(Scene):
    """Base class for beat-structured scenes.

    Subclasses define ``@beat`` methods instead of one long ``construct``.
    Mobjects assigned to ``self`` persist across beats and form the *stage* —
    the shared state beat N+1 inherits from beat N.

    That stage is not just bookkeeping. When a model writes beat N+1 it needs
    to know what is already on screen, and passing a compact manifest costs a
    fraction of the tokens that re-sending every previous beat's source would.
    """

    def construct(self) -> None:
        for fn in beats_of(type(self)):
            spec = fn._forge_beat
            self.next_section(spec.name)
            fn(self)

    # -- stage state ---------------------------------------------------------

    def stage(self) -> list[StageItem]:
        """Named mobjects currently held on the scene.

        Reads ``self.__dict__`` rather than ``self.mobjects`` on purpose: we
        want the *names* the code uses, since those are the handles the next
        beat will reference.
        """
        items = []
        for name, value in vars(self).items():
            if name.startswith("_") or not isinstance(value, Mobject):
                continue
            try:
                c = value.get_center()
                pos = (float(c[0]), float(c[1]))
            except Exception:
                pos = (0.0, 0.0)
            items.append(StageItem(name, type(value).__name__, pos))
        return sorted(items, key=lambda i: i.name)

    def stage_manifest(self) -> str:
        """The stage as a prompt fragment for the coder model."""
        items = self.stage()
        if not items:
            return "(stage is empty)"
        return "\n".join(f"  - {i}" for i in items)
