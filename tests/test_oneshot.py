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
    slip = Result("r", beats, ['stage.equation("2 + 2 = 5")'] + ["a"] * 3, ok=True,
                  layout=[0, 0, 0, 0])
    assert score(clean) > score(messy) > score(short) > score(failed)
    assert score(clean) > score(slip) and score(messy) > score(slip)


def test_the_panel_ends_on_the_winning_sample(monkeypatch):
    from forge.app import oneshot, pipeline
    replies = iter(["# beat 1: first try\nstage.title('a')",
                    "# beat 1: second try\nstage.title('b')"])
    monkeypatch.setattr(pipeline, "ask", lambda *a, **k: next(replies))

    def fake_finish(request, beats, bodies, *a, **k):
        first = beats[0].intent == "first try"
        return pipeline.Result(request, beats, bodies, ok=first,
                               layout=[1] if first else [])
    monkeypatch.setattr(pipeline, "finish", fake_finish)
    events = []
    res = oneshot.run_oneshot("r", None, None, events.append, samples=2)
    assert res.beats[0].intent == "first try"
    last_plan = [e["beat"]["intent"] for e in events
                 if e.get("stage") == "plan" and "beat" in e][-1]
    assert last_plan == "first try"
    assert any(e.get("reset") for e in events)


def test_critique_finds_beats_that_draw_nothing():
    from forge.app.critique import static_problems
    from forge.app.twostage import Beat
    beats = [Beat(1, None, "a plane with a vector"),
             Beat(2, None, "the matrix is applied and the grid becomes a parallelogram"),
             Beat(3, None, "the rule"),
             Beat(4, None, "apply the shear to the grid")]
    bodies = ["p = draw_plane(stage)\nv = draw_vector(stage, p, (1, 2))",
              'stage.caption("The grid becomes a parallelogram")\nstage.pause(1)',
              r'stage.equation(r"A v = (2, 3)")',
              r'stage.equation(r"A = [[1, 1], [0, 1]]")  # apply_matrix(stage, p, A)']
    found = dict(static_problems(beats, bodies))
    assert set(found) == {2, 4}
    assert "only changes the text" in found[2] and "equation" in found[4]


def test_problems_list_kit_issues_and_slips_by_beat():
    from forge.app.critique import problems
    from forge.app.pipeline import Result
    from forge.app.twostage import Beat
    beats = [Beat(1, None, "axes"), Beat(2, None, "sum")]
    res = Result("r", beats, ["ax = draw_axes(stage)", 'stage.equation("2 + 2 = 5")'],
                 ok=True, issues=[(1, "the point (3, 4) is outside its axes")])
    found = problems(res)
    assert found[0] == "beat 1: the point (3, 4) is outside its axes"
    assert any("2 + 2 = 5" in f for f in found)


def test_a_flawed_draft_goes_back_with_its_problems(monkeypatch):
    from forge.app import oneshot, pipeline
    prompts = []
    replies = iter(["# beat 1: grid\np = draw_plane(stage)\n# beat 2: the grid shears\n"
                    "stage.caption('sheared')",
                    "# beat 1: grid\np = draw_plane(stage)\n# beat 2: the grid shears\n"
                    "apply_matrix(stage, p, [[1, 1], [0, 1]])"])

    def fake_ask(model, tok, system, user, **k):
        prompts.append((user, k.get("temp")))
        return next(replies)
    monkeypatch.setattr(pipeline, "ask", fake_ask)
    monkeypatch.setattr(pipeline, "finish", lambda request, beats, bodies, *a, **k:
                        pipeline.Result(request, beats, bodies, ok=True, layout=[0, 0]))
    res = oneshot.run_oneshot("shear", None, None, samples=2, revise=1)
    assert "PROBLEMS FOUND" in prompts[1][0] and "beat 2:" in prompts[1][0]
    assert "stage.caption('sheared')" in prompts[1][0] and prompts[1][1] == 0.0
    assert "apply_matrix" in res.bodies[1]
