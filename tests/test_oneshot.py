"""The one-shot generator's library, prompt and parser."""
from forge.app.oneshot import parse_scene, user_prompt
from forge.evaluate.heldout_guard import touches_heldout
from forge.kit.library import as_body, scenes, similar


def test_library_has_no_heldout_topic():
    lib = scenes()
    assert len(lib) > 500
    for s in lib:
        assert not touches_heldout(
            s["request"] + " " + " ".join(b["intent"] for b in s["beats"]))


def test_similar_excludes_the_request_itself():
    s = scenes()[0]
    got = similar(s["request"], 3, exclude={s["request"]})
    assert got and all(x["request"] != s["request"] for _, x in got)
    assert similar(s["request"], 1)[0][1]["request"] == s["request"]


def test_parse_round_trips_a_library_scene():
    s = scenes()[3]
    beats, bodies = parse_scene(as_body(s))
    assert [b.intent for b in beats] == [b["intent"] for b in s["beats"]]
    assert [b.narration for b in beats] == [b.get("narration", "") for b in s["beats"]]
    assert [x.strip() for x in bodies] == [b["code"].strip() for b in s["beats"]]


def test_parse_handles_a_fenced_reply_and_lead_lines():
    reply = ("Here you go:\n```python\nstage.title(\"T\")\n# beat 1: a dot\n"
             "# say: one dot\nd = Dot()\nself.play(Create(d))\n"
             "# beat 2: it moves\nself.play(d.animate.shift(RIGHT))\n```")
    beats, bodies = parse_scene(reply)
    assert [b.intent for b in beats] == ["a dot", "it moves"]
    assert beats[0].narration == "one dot"
    assert bodies[0].startswith('stage.title("T")') and "Create(d)" in bodies[0]
    assert bodies[1] == "self.play(d.animate.shift(RIGHT))"


def test_prompt_shows_examples_and_the_request():
    p = user_prompt("what is 15% of 240?", k=2)
    assert p.count("REQUEST:") == 3 and p.rstrip().endswith("what is 15% of 240?")
    assert "# beat 1:" in p


def test_score_prefers_whole_clean_renders():
    from forge.app.oneshot import score
    from forge.app.pipeline import Result
    from forge.app.twostage import Beat
    beats = [Beat(i, None, "x") for i in range(1, 5)]
    clean = Result("r", beats, ["a"] * 4, ok=True, layout=[0, 0, 0, 0])
    messy = Result("r", beats, ["a"] * 4, ok=True, layout=[0, 2, 0, 1])
    short = Result("r", beats, ["a", "", "a", "a"], ok=True, layout=[0, 0, 0])
    failed = Result("r", beats, ["a"] * 4, ok=False)
    assert score(clean) > score(messy) > score(short) > score(failed)
