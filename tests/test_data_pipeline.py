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


# --- the recovered rows have to arrive too ---------------------------------

def test_lint_recoveries_supersede_the_failure_they_came_from(tmp_path):
    """regate.py wrote its recoveries to a file nothing ever read.

    The same defect as the gold export, the colliding synthetic ids and the
    unread synthetic files: a step that produced, and a step that never
    collected. Every hour of CPU regate consumed produced nothing.
    """
    from forge.gate.export import build as gate_build

    code = scene("A", "Dot", 2)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({
        "id": "r1", "code": "class A(Scene):\n    pass\n",
        "prompt": "a prompt long enough to survive the filter",
        "source": "t", "license": "x", "n_play_calls": 2, "flavor": "ce"}) + "\n")
    gate = _write(tmp_path / "gate.jsonl", [{"id": "r1", "ok": False}])
    regate = _write(tmp_path / "regate.jsonl",
                    [{"id": "r1", "ok": True, "code": code, "retried": True}])

    out = tmp_path / "verified.jsonl"
    stats = gate_build(corpus, gate, out, regate=regate)
    assert stats["recovered_by_lint"] == 1
    assert stats["unique_verified"] == 1

    rows = [json.loads(l) for l in out.open()]
    assert rows, "the recovered row never reached the export"
    # The repaired source is the training target, not the broken original.
    assert "self.play(" in rows[0]["messages"][-1]["content"]


def test_a_row_that_already_passes_is_not_superseded(tmp_path):
    from forge.gate.export import build as gate_build
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({
        "id": "r1", "code": scene("A", "Dot", 2),
        "prompt": "a prompt long enough to survive the filter",
        "source": "t", "license": "x", "n_play_calls": 2, "flavor": "ce"}) + "\n")
    gate = _write(tmp_path / "gate.jsonl", [{"id": "r1", "ok": True}])
    regate = _write(tmp_path / "regate.jsonl",
                    [{"id": "r1", "ok": True, "code": "WRONG", "retried": True}])
    stats = gate_build(corpus, gate, tmp_path / "v.jsonl", regate=regate)
    assert stats["recovered_by_lint"] == 0


def test_restore_does_not_require_the_package_to_be_installed():
    """A fresh machine restores *before* it can import anything.

    The first training run failed here: scripts/backup_to_hf.py imported
    forge.catalog at the top of main(), and a CI runner that has checked out
    the repo but not installed it gets ModuleNotFoundError before it can
    fetch the data the project needs. Restoring is the bootstrap step; it
    cannot depend on the thing it is bootstrapping.
    """
    import ast
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "scripts" / "backup_to_hf.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")

    # Find where the --restore branch returns, and where forge is imported.
    restore_return = next(
        (n.lineno for n in ast.walk(fn)
         if isinstance(n, ast.Return) and n.lineno > 0
         and "restore" in ast.get_source_segment(src, fn).split("\n")[
             max(0, n.lineno - fn.lineno - 1)]),
        None)
    forge_import = next(
        (n.lineno for n in ast.walk(fn)
         if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("forge")),
        None)

    assert forge_import is not None, "expected a forge import in main()"
    assert restore_return is not None, "expected the --restore branch to return"
    assert restore_return < forge_import, (
        f"forge is imported at line {forge_import}, before --restore returns at "
        f"line {restore_return} — a runner with no package installed cannot "
        f"restore the data it needs to install anything")


# --- PEFT -> MLX adapter conversion ----------------------------------------

def test_peft_to_mlx_produces_the_same_delta(tmp_path):
    """The two libraries disagree about shape, scaling and key names.

    PEFT:  A (r, in)   B (out, r)   delta = (alpha/r) · B @ A
    MLX:   lora_a (in, r)  lora_b (r, out)  delta = scale · lora_bᵀ @ lora_aᵀ

    Get a transpose wrong and the weights still load, still generate, and
    quietly compute something else. So this asserts on the delta, which is
    the only thing that affects the model's output.
    """
    import numpy as np
    mx = pytest.importorskip("mlx.core")
    from scripts.peft_to_mlx import convert

    r, alpha, d_in, d_out = 8, 16, 64, 32
    rng = np.random.default_rng(0)
    A = rng.normal(size=(r, d_in)).astype(np.float32)
    B = rng.normal(size=(d_out, r)).astype(np.float32)

    peft = tmp_path / "peft"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(json.dumps({
        "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
        "target_modules": ["q_proj"], "peft_type": "LORA"}))
    stem = "base_model.model.model.layers.3.self_attn.q_proj"
    mx.save_safetensors(str(peft / "adapter_model.safetensors"),
                        {f"{stem}.lora_A.weight": mx.array(A),
                         f"{stem}.lora_B.weight": mx.array(B)})

    stats = convert(peft, tmp_path / "mlx")
    assert stats["rank"] == r
    assert stats["scale"] == alpha / r
    assert stats["layers"] == 4          # layer index 3 -> 4 layers deep

    got = mx.load(str(tmp_path / "mlx" / "adapters.safetensors"))
    key_a = "model.layers.3.self_attn.q_proj.lora_a"
    key_b = "model.layers.3.self_attn.q_proj.lora_b"
    assert set(got) == {key_a, key_b}, f"unexpected keys: {sorted(got)}"
    assert got[key_a].shape == (d_in, r)
    assert got[key_b].shape == (r, d_out)

    peft_delta = (alpha / r) * (B @ A)                       # (out, in)
    mlx_delta = stats["scale"] * (np.array(got[key_b]).T @ np.array(got[key_a]).T)

    # Relative, not absolute. The adapter is stored in float16 -- which is
    # what MLX's own trainer writes and what the model runs in -- so on a
    # delta of magnitude ~32 the absolute error is ~0.016 and means nothing.
    # A transposition error, by contrast, is order-1 relative.
    rel = np.abs(peft_delta - mlx_delta).max() / np.abs(peft_delta).max()
    assert rel < 1e-3, f"deltas differ by {rel:.4%} — check the transposes"


def test_peft_to_mlx_refuses_a_file_that_is_not_an_adapter(tmp_path):
    from scripts.peft_to_mlx import convert
    mx = pytest.importorskip("mlx.core")
    peft = tmp_path / "peft"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}))
    mx.save_safetensors(str(peft / "adapter_model.safetensors"),
                        {"something.else.weight": mx.zeros((4, 4))})
    with pytest.raises(SystemExit, match="no LoRA tensors"):
        convert(peft, tmp_path / "mlx")
