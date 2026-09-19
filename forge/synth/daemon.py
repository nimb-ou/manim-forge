"""Continuous generation — run until the free tier is genuinely exhausted.

Rate limits on this API are *per model*, so the usable daily budget is the sum
across every model that answers, not the quota of any one. A single-model run
stops at its own 429 while five other quotas sit untouched.

This keeps a live view of which models still answer, rotates through them,
retires one on a 429 and brings it back after a cooldown. When every model is
in cooldown it sleeps rather than exits, because a quota that resets is a
quota worth waiting for on a job that has all night.

Work is drawn from a queue of tasks — topics, narration segments, variations —
and every result is render-gated before it counts, so "generated" and
"verified" never get confused.
"""

from __future__ import annotations

import json
import random
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelPool:
    """Which models still answer, and when a retired one may be retried.

    Thread-safe, because workers run concurrently: generation waits on the
    network and rendering waits on a subprocess, so a sequential loop leaves
    both the API quota and the CPU mostly idle.

    ``next_model`` round-robins rather than always returning the strongest
    available one. Hammering a single model drives it straight into its 429
    while the others sit unused — which is the exact failure this pool exists
    to avoid.
    """
    models: list[str]
    cooldown_s: float = 600.0
    _retired: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cursor: int = 0

    def available(self) -> list[str]:
        now = time.monotonic()
        return [m for m in self.models
                if m not in self._retired or self._retired[m] <= now]

    def retire(self, model: str) -> None:
        with self._lock:
            self._retired[model] = time.monotonic() + self.cooldown_s

    def next_model(self) -> str | None:
        with self._lock:
            free = self.available()
            if not free:
                return None
            self._cursor = (self._cursor + 1) % len(free)
            return free[self._cursor]

    def seconds_until_any(self) -> float:
        if self.available():
            return 0.0
        return max(0.0, min(self._retired.values()) - time.monotonic())

    def status(self) -> str:
        now = time.monotonic()
        parts = []
        for m in self.models:
            wait = self._retired.get(m, 0) - now
            parts.append(f"{m.replace('gemini-', '')}:{'ok' if wait <= 0 else f'{wait/60:.0f}m'}")
        return " ".join(parts)


@dataclass
class Task:
    """One unit of generation work."""
    kind: str           # "topic" | "narration"
    key: str            # stable identity, for resume
    request: str        # the prompt text sent to the teacher
    n_beats: int
    length_hint: str
    meta: dict = field(default_factory=dict)


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.open():
        try:
            done.add(json.loads(line)["task_key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def build_queue(topics_per_length: bool = True,
                narration_path: Path = Path("data/style/narration.jsonl"),
                min_chars: int = 260, max_chars: int = 1600,
                variations: int = 1) -> list[Task]:
    """Every piece of work available, topics and narration together.

    Interleaved rather than run in sequence: a corpus that is all topics for a
    day and all narration the next is skewed by whatever the quota happened to
    allow, and the training mix should not be an accident of scheduling.
    """
    from forge.synth.topics import ALL, LENGTHS

    tasks: list[Task] = []

    for topic, domain in ALL:
        for (length, n_beats, hint) in LENGTHS:
            for v in range(variations):
                tasks.append(Task(
                    kind="topic",
                    key=f"topic::{topic}::{length}::{v}",
                    request=topic, n_beats=n_beats, length_hint=hint,
                    meta={"domain": domain, "length": length, "variation": v},
                ))

    if narration_path.exists():
        from forge.synth.teacher import NARRATION_INSTRUCTION as INSTRUCTION
        for line in narration_path.open():
            s = json.loads(line)
            if not (min_chars <= len(s["text"]) <= max_chars):
                continue
            tasks.append(Task(
                kind="narration",
                key=f"narration::{s['video_id']}::{s['index']}",
                request=INSTRUCTION + s["text"],
                n_beats=6, length_hint="matching the passage",
                meta={"video_id": s["video_id"], "title": s["title"],
                      "seg_index": s["index"], "domain": "3b1b"},
            ))

    random.Random(11).shuffle(tasks)
    return tasks
