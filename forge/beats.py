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
    #: What a narrator would say over this beat. Optional, and worth having
    #: even with no audio: writing it forces the planner to reason about the
    #: *explanation* rather than only the picture, which is the difference
    #: between an explainer and a diagram. It is also exactly what a
    #: text-to-speech track needs later, timed against ``seconds``.
    narration: str = ""

    def as_instruction(self) -> str:
        """The line a planner emits and a coder consumes."""
        dur = f" (~{self.seconds:g}s)" if self.seconds else ""
        return f"{self.intent}{dur}"


def beat(intent: str, seconds: float | None = None, narration: str = ""):
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
            narration=narration,
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


def script_of(cls) -> str:
    """The narration track, in order — captions today, a voiceover later."""
    lines = []
    for spec in storyboard(cls):
        if spec.narration:
            lines.append(spec.narration.strip())
    return "\n\n".join(lines)


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

    #: Words a narrator says per second. 3blue1brown sits near here; faster
    #: than this and a listener stops following the picture to keep up.
    NARRATION_WPS = 2.6

    #: Held frame after a beat's last animation, when the beat finished early.
    #:
    #: Not silence -- the narration is still running over it, and holding a
    #: diagram while explaining it is what an explainer does. The cap exists
    #: so a mis-declared beat cannot stall the video outright, and a beat that
    #: hits it is under-animated rather than over-declared: it has more to say
    #: than to show, and wants another visual step, not a longer pause.
    MAX_PAD_S = 12.0

    def construct(self) -> None:
        for fn in beats_of(type(self)):
            spec = fn._forge_beat
            self.next_section(spec.name)
            started = self.renderer.time
            fn(self)
            self._hold_for(spec, self.renderer.time - started)

    def _hold_for(self, spec: "BeatSpec", elapsed: float) -> None:
        """Pad a beat out to the duration it declared.

        ``seconds=`` used to be documentation: a number in the decorator that
        nothing enforced, so the animation ran at whatever speed its run_times
        happened to sum to. That is how every scene in the corpus ended up
        with narration too long to say over it -- the script was written for
        the declared length and the picture ran short.

        The beat now waits out the difference, so the declared length is the
        real one and the narration has room. A beat that overruns its
        declaration is left alone; shortening it would cut an animation.
        """
        target = spec.seconds
        if not target:
            return
        # Whichever is longer: what the beat declared, or what its own
        # narration needs at a speakable pace.
        words = len((spec.narration or "").split())
        if words:
            target = max(target, words / self.NARRATION_WPS)
        pad = min(target - elapsed, self.MAX_PAD_S)
        if pad > 0.05:
            self.wait(pad)

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
