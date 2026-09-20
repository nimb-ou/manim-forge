"""The defects that silently threw finished work away.

Every one of these sat between a step that worked and a step that consumed
its output, and was invisible because nothing compared the two ends. So these
tests compare the two ends.

See docs/POSTMORTEM.md, A1-A3 and B1-B2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.ingest.schema import CorpusRow
from forge.train.prepare import build, load_synthetic

# Fixtures must differ in AST *structure*, not only in the names or the
# constants they use. forge.ingest.schema.normalized_hash walks the tree and
# keeps node types only -- so `Create(Dot())` and `Create(Circle())` hash the
# same, and so do `range(1)` and `range(2)`. That is deliberate: the hash
# exists to collapse the same scene reappearing across four public datasets
# with renamed variables and reflowed whitespace.
#
# Measured on real data it causes no false collapse at all -- 667 verified
# synthetic scenes and 44 gold scenes give 667 and 44 distinct hashes,
# identical to hashing the exact text. Real scenes are long enough to differ
# structurally. One-statement fixtures are not, so `scene()` varies the
# number of statements.

def scene(name: str, shape: str = "Dot", plays: int = 1) -> str:
    body = "\n".join(f"        self.play(Create({shape}()))"
                      for _ in range(plays))
    return (f"from manim import *\n\n"
            f"class {name}(Scene):\n"
            f"    def construct(self):\n{body}\n"
            f"        self.wait()\n")


STILL = """from manim import *

class Still(Scene):
    def construct(self):
        self.add(Text("no animation here"))
        self.wait()
"""


def _row(code: str, rid: str, **extra) -> dict:
    built = CorpusRow.build(source="t", license="x", prompt="draw a thing",
                            code=code, index=None)
    return {"id": rid, "ok": True, "code": code, "prompt": "draw a thing",
            "dedupe_key": built.dedupe_key,
            "n_play_calls": built.n_play_calls, **extra}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- A2: the id ------------------------------------------------------------

def test_generated_rows_get_distinct_ids_without_a_counter():
    """generate_forever.py passed index=0 for every row it ever wrote.

    399 distinct scenes in stream.jsonl carry 10 ids between them; 369 are
    called stream-narration:000000.
    """
    a = CorpusRow.build(source="s", license="x", prompt="p",
                        code=scene("A", "Dot", 1), index=None)
    b = CorpusRow.build(source="s", license="x", prompt="p",
                        code=scene("A", "Circle", 2), index=None)
    assert a.id != b.id
    assert a.id.startswith("s:") and not a.id.split(":")[-1].isdigit()


def test_an_explicit_index_still_numbers_the_row():
    """Ingest sources do have a position in a file, and should keep using it."""
    assert CorpusRow.build(source="s", license="x", prompt="p",
                           code="x", index=7).id == "s:000007"


def test_structurally_identical_scenes_share_an_id():
    """The id is the dedupe key, so a rename must not create a new row.

    This is the property the hash exists for: the same scene reappears
    across four public datasets with different variable names.
    """
    a = CorpusRow.build(source="s", license="x", prompt="p",
                        code=scene("Alpha", "Dot", 1), index=None)
    b = CorpusRow.build(source="s", license="x", prompt="p",
                        code=scene("Beta", "Dot", 1), index=None)
    assert a.id == b.id


# --- A2/A3: the loader -----------------------------------------------------

def test_loader_keeps_distinct_scenes_that_share_a_bad_id(tmp_path):
    """The historical files still carry colliding ids; they must survive."""
    rows = [_row(scene("A", "Dot", 1), "stream:000000"),
            _row(scene("A", "Circle", 2), "stream:000000"),
            _row(scene("A", "Square", 3), "stream:000000")]
    got = load_synthetic(_write(tmp_path / "s.jsonl", rows))
    assert len(got) == 3


def test_loader_still_collapses_genuine_duplicates(tmp_path):
    same = scene("A", "Dot", 1)
    rows = [_row(same, "a"), _row(same, "b")]
    assert len(load_synthetic(_write(tmp_path / "s.jsonl", rows))) == 1


def test_loader_reads_every_file_it_is_given(tmp_path):
    """Only one of three synthetic files was in the default mix."""
    one = _write(tmp_path / "one.jsonl", [_row(scene("A", "Dot", 1), "1")])
    two = _write(tmp_path / "two.jsonl", [_row(scene("A", "Circle", 2), "2")])
    assert len(load_synthetic(one, two)) == 2
    assert load_synthetic(tmp_path / "absent.jsonl") == []


def test_loader_skips_rows_the_gate_rejected(tmp_path):
    rows = [_row(scene("A", "Dot", 1), "1"),
            {**_row(scene("A", "Circle", 2), "2"), "ok": False}]
    assert len(load_synthetic(_write(tmp_path / "s.jsonl", rows))) == 1


# --- B1/B2: the mix --------------------------------------------------------

@pytest.fixture
def mix(tmp_path):
    """A tiny corpus: two animated rows, one still, one gold scene."""
    gated = tmp_path / "gated.jsonl"
    gated.write_text("".join(json.dumps({
        "messages": [{"role": "system", "content": "s"},
                     {"role": "user", "content": p},
                     {"role": "assistant", "content": f"```python\n{c}\n```"}],
        "meta": {"id": i, "n_play_calls": n},
    }) + "\n" for i, p, c, n in [
        ("g1", "animate one", scene("A", "Dot", 1), 1),
        ("g2", "animate two", scene("B", "Circle", 4), 1),
        ("g3", "a still image", STILL, 0),
    ]))
    synth = _write(tmp_path / "syn.jsonl",
                   [_row(scene("C", "Square", 5), "s1")])
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({
        "prompt": "the gold one", "code": scene("G", "Line", 6),
        "meta": {"id": "gold:x", "source": "gold", "n_play_calls": 1}}) + "\n")
    return tmp_path, gated, synth, gold


def test_rows_with_no_animation_are_dropped_and_counted(mix):
    """`reps = max(1, reps // 2)` on a gated row is max(1, 0) == 1.

    It removed nothing. 243 stills went into training at full weight, each
    paired with a system prompt reading "Always animate with self.play(...)".
    """
    out, gated, synth, gold = mix
    stats = build(out / "d", gated, [synth], gold, val_frac=0.01)
    assert stats["static_dropped"] == 1
    assert stats["static_kept"] == 0

    written = [json.loads(l) for l in (out / "d" / "train.jsonl").open()]
    assert written, "nothing was written"
    assert all("self.play(" in r["messages"][-1]["content"] for r in written)


def test_keeping_stills_is_possible_but_must_be_asked_for(mix):
    out, gated, synth, gold = mix
    stats = build(out / "d", gated, [synth], gold, val_frac=0.01, drop_static=False)
    assert stats["static_kept"] == 1
    assert stats["static_dropped"] == 0


def test_the_written_file_says_what_it_is_made_of(mix):
    """meta was stripped on write, so the mix could not be audited at all.

    Asking "how much of this is gold" meant re-deriving it from the inputs
    and trusting the derivation. During the audit it did not match: the
    answer was 3.6%, not the ~15% the weighting implied.
    """
    out, gated, synth, gold = mix
    stats = build(out / "d", gated, [synth], gold, val_frac=0.01)
    rows = [json.loads(l) for l in (out / "d" / "train.jsonl").open()]
    assert all("meta" in r and "tier" in r["meta"] for r in rows)

    from collections import Counter
    assert Counter(r["meta"]["tier"] for r in rows) == stats["train_by_tier"]


def test_gold_is_weighted_up(mix):
    out, gated, synth, gold = mix
    build(out / "d", gated, [synth], gold, gold_weight=6, synth_weight=2,
          val_frac=0.01)
    rows = [json.loads(l) for l in (out / "d" / "train.jsonl").open()]
    assert sum(1 for r in rows if r["meta"]["tier"] == "gold") == 6


def test_no_prompt_appears_on_both_sides_of_the_split(mix):
    out, gated, synth, gold = mix
    build(out / "d", gated, [synth], gold, val_frac=0.4)
    def prompts(name):
        return {json.loads(l)["messages"][1]["content"]
                for l in (out / "d" / f"{name}.jsonl").open()}
    assert not (prompts("train") & prompts("valid"))
