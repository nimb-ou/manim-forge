import json
from pathlib import Path

from forge.evaluate.heldout_guard import kit_row_touches_heldout, row_touches_heldout, touches_heldout

ROOT = Path(__file__).resolve().parents[1]


def test_every_heldout_prompt_is_caught():
    d = json.loads((ROOT / "forge" / "evaluate" / "heldout_prompts.json").read_text())
    for p in d["prompts"]:
        assert touches_heldout(p["prompt"]), p["prompt"]


def test_neighbouring_topics_pass():
    for t in ("prove the Pythagorean theorem with squares on the sides",
              "how do you add 1/3 and 1/4?",
              "what does a 10% hill sign mean?"):
        assert not touches_heldout(t), t


def test_rows_are_judged_by_their_request_only():
    def row(request, beat):
        return {"messages": [{"role": "system", "content": "x"},
                             {"role": "user", "content": f"REQUEST\n{request}\n\nBEATS SO FAR\n  {beat}"}]}
    assert row_touches_heldout(row("the Monty Hall problem", "three doors"))
    # A passing mention in the history is not the topic.
    assert not row_touches_heldout(row("how slide rules multiply", "a logarithm scale"))


def test_a_kit_beat_is_judged_by_its_own_intent_too():
    def row(intent):
        return {"messages": [{"role": "system", "content": "x"},
                             {"role": "user", "content": "REQUEST\nlinear combinations of two vectors\n\n"
                              f"WRITE THIS BEAT — step 3 of 4\n  intent: {intent}\n  narration: n"}]}
    assert kit_row_touches_heldout(row("u + v drawn tip to tail"))
    assert not kit_row_touches_heldout(row("the grid with u and v"))
