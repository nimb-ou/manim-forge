"""The held-out topics as one pattern, for every training-data builder.

The headline metric is the judged share on 20 held-out prompts
(heldout_prompts.json) whose topics nothing should have been trained on. An
audit on 2026-10-01 found them in the data anyway: 3Blue1Brown narration arcs
on the central limit theorem and the chain rule, synthetic arcs on Monty Hall
and the birthday paradox, gold scenes (ChainRule, MontyHall, IrrationalSqrt2),
and a few Claude kit scenes (vectors tip to tail, a bell curve from samples).
Each builder now drops a scene or arc whose natural-language text touches one
of these topics. The pattern errs wide: losing a neighbouring scene costs a
few rows, letting a held-out one through costs the metric.
"""
from __future__ import annotations

import re

HELDOUT = re.compile(
    r"harmonic series|chain rule"
    r"|matri\w* multiplication|composition of (linear )?(transformations|maps)"
    r"|one transformation after another"
    r"|monty hall|compound(ed)? interest|continuously compound|euler'?s number|the number e\b"
    r"|logarithm|\blog scale"
    r"|birthday (paradox|problem)"
    r"|standard deviation|least squares|line of best fit|linear regression"
    r"|merge ?sort|pascal'?s triangle|roots? of unity|mean value theorem"
    r"|half.life|exponential decay|radioactive decay"
    r"|tip.to.tail|adding (two )?vectors|vector addition"
    r"|root (of )?(2|two)\b|√2|sqrt\(?2|irrational"
    r"|fibonacci|golden ratio"
    r"|fourier transform"
    r"|determinant (is |of |equal to )?(zero|0)\b|zero determinant"
    r"|central limit|bell curve|normal distribution|gaussian"
    r"|sums? of (two |several |many |\d+ )?dice|dice sums?",
    re.I)


def touches_heldout(text: str) -> bool:
    return bool(HELDOUT.search(text or ""))


def request_of(row: dict) -> str:
    """What a training row is about: its REQUEST block and request fields."""
    user = next((m["content"] for m in row.get("messages", [])
                 if m.get("role") == "user"), "")
    m = re.search(r"REQUEST\n(.*?)(?:\n\n|$)", user, re.S)
    meta = row.get("meta") or {}
    return " ".join(str(x) for x in (m.group(1) if m else "", row.get("request") or "",
                                     meta.get("request") or ""))


def row_touches_heldout(row: dict) -> bool:
    """A row whose scene or arc is about a held-out topic. Only the request
    is read: long narration histories mention "logarithm" in passing, and
    matching those would drop half the 3Blue1Brown arcs for no reason."""
    return touches_heldout(request_of(row))


def beat_intent_of(row: dict) -> str:
    """The intent of the one beat a kit coder row writes."""
    user = next((m["content"] for m in row.get("messages", [])
                 if m.get("role") == "user"), "")
    m = re.search(r"WRITE THIS BEAT.*?\n\s*intent:\s*(.*)", user, re.S)
    return m.group(1).split("\n")[0] if m else ""


def kit_row_touches_heldout(row: dict) -> bool:
    """A kit coder row is one beat, so its own intent counts as well as the
    scene's request: a beat drawing two vectors "tip to tail" inside a
    linear-combination scene teaches exactly the held-out skill. (The
    planner learns whole arcs, so for it the request is the topic.)"""
    return row_touches_heldout(row) or touches_heldout(beat_intent_of(row))
