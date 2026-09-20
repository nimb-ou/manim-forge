"""The gold scenes are the most expensive artefact in the project.

Several hours each, hand-authored, and they are the only thing in the corpus
that teaches style rather than syntax. There are four places a scene has to
be recorded, and for 32 scenes exactly one of them was skipped -- so they were
finished, committed, retimed, audited, and absent from every training run.

These tests compare those four places against each other. See
docs/POSTMORTEM.md A1.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD_JSONL = ROOT / "data" / "gold" / "gold.jsonl"


def _modules() -> set[str]:
    """Gold scene modules on disk."""
    import forge.gold
    return {m.name for m in pkgutil.iter_modules(forge.gold.__path__)
            if m.name != "curriculum"}


def _export_list() -> list[dict]:
    """The GOLD list that scripts/export_gold.py builds the dataset from."""
    src = (ROOT / "scripts" / "export_gold.py").read_text()
    ns: dict = {}
    # Execute only the list literal, not the export it performs on import.
    exec(src[src.index("GOLD = ["):src.index("out = Path(")], ns)
    return ns["GOLD"]


def _exported_rows() -> list[dict]:
    if not GOLD_JSONL.exists():
        pytest.skip("gold.jsonl not built; run scripts/export_gold.py")
    return [json.loads(l) for l in GOLD_JSONL.open()]


def test_every_authored_scene_is_in_the_export_list():
    listed = {Path(g["module"]).stem for g in _export_list()}
    missing = _modules() - listed
    assert not missing, (
        f"{len(missing)} scene module(s) exist but are not in GOLD: "
        f"{sorted(missing)} -- run scripts/register_gold.py")


def test_every_listed_scene_reached_the_dataset():
    """The step that was missing. Being in the list is not being in the data."""
    listed = {Path(g["module"]).stem for g in _export_list()}
    exported = {r["meta"]["id"].removeprefix("gold:") for r in _exported_rows()}
    missing = listed - exported
    assert not missing, (
        f"{len(missing)} scene(s) in GOLD never reached gold.jsonl: "
        f"{sorted(missing)} -- run scripts/export_gold.py")


def test_every_authored_scene_is_marked_done_in_the_curriculum():
    from forge.gold.curriculum import CURRICULUM
    done = {c.key for c in CURRICULUM if c.done}
    known = {c.key for c in CURRICULUM}
    # A scene may exist without a curriculum entry; one that has an entry and
    # is written must be marked.
    unmarked = (_modules() & known) - done
    assert not unmarked, f"written but not marked done: {sorted(unmarked)}"


def test_gold_ids_are_stable_under_reordering():
    """They were list positions, and register_gold.py prepends.

    Every existing scene was renamed each time a new one was added, so the
    id identified nothing across two exports.
    """
    ids = [r["meta"]["id"] for r in _exported_rows()]
    assert len(set(ids)) == len(ids), "duplicate gold ids"
    assert not any(i.removeprefix("gold:").isdigit() for i in ids), \
        "gold ids are positional; they must be keyed on the module"


def test_every_gold_row_carries_code_that_animates():
    for r in _exported_rows():
        assert r["meta"]["n_play_calls"] > 0, f"{r['meta']['id']} never animates"
        assert r["meta"]["n_beats"] > 0, f"{r['meta']['id']} has no beats"
        assert r["prompt"].strip(), f"{r['meta']['id']} has no prompt"


def test_every_gold_scene_imports_and_declares_beats():
    """A scene that cannot be imported cannot be rendered or trained on."""
    from forge.beats import beats_of
    broken = []
    for name in sorted(_modules()):
        try:
            mod = importlib.import_module(f"forge.gold.{name}")
        except Exception as exc:                      # noqa: BLE001
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        scenes = [v for v in vars(mod).values()
                  if inspect.isclass(v) and v.__module__ == mod.__name__
                  and hasattr(v, "construct")]
        if not scenes:
            broken.append(f"{name}: no Scene class")
        elif not any(beats_of(s) for s in scenes):
            broken.append(f"{name}: no beats")
    assert not broken, "\n".join(broken)
