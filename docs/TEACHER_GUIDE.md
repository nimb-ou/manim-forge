# Writing teacher scenes for the kit coder

You are writing training data for a 7B model that turns a request into a
3Blue1Brown-style animation, one *beat* at a time, using the Forge kit
(`forge/kit/kit.py`). Every scene you write becomes training rows: for each
beat, the model sees the request, the earlier beats' code and this beat's
intent + narration, and must produce this beat's code. It learns exactly
what you write — so write what a great animator would write with this kit.

## Format

A batch is a JSON list of scenes (write it with Python's `json.dump`, not by
hand, to get escaping right):

```json
[{"request": "why the angles of a quadrilateral add up to 360 degrees",
  "beats": [
    {"intent": "A quadrilateral split by one diagonal into two triangles",
     "narration": "Draw any four-sided shape and cut it along a diagonal.",
     "code": "stage.title(\"Four corners\")\nq = draw_polygon(stage, [(-2.5, -1.5), (2.5, -1.2), (1.8, 1.6), (-1.6, 1.2)])\nd = draw_line(stage, (-2.5, -1.5), (1.8, 1.6), dashed=True)\nstage.caption(\"One diagonal makes two triangles\")"},
    ...]}]
```

- 3 to 6 beats per scene. 20-60 seconds of animation in total.
- `request`: how a person would ask — short, plain English, one idea.
- `intent`: one line that **names the picture** of the beat ("A ball stepping
  down the parabola", not "Understanding gradient descent"). The planner is
  trained on these too; intents that describe pictures are what we want it to
  learn.
- `narration`: one or two spoken sentences.
- `code`: the beat's statements only — no class, no def, no imports.
  `stage = Stage(self)` already exists; `np`, `PI`, `TAU`, colours and every
  Manim name are in scope. Names you create in one beat are usable in later
  beats.

## The kit

Read `KIT_API` near the end of `forge/kit/kit.py` (the full list with
arguments) and the worked scenes in `forge/kit/exemplars.py` (good style).
Blocks take `stage` first, draw *and animate*, and return what they made.

## What makes a good beat

1. **Every beat shows its idea as a picture.** A beat that only sets a title
   or caption, or only draws empty axes, is a bad beat. Use the kit's
   motion blocks (`slide_tangent`, `riemann_refine`, `gradient_descent`,
   `apply_matrix`, `multiply_complex`, `grow_histogram`, `swap_bars`, ...)
   where the idea moves.
2. **Text only through the stage**: `stage.title("...")` (a few words),
   `stage.caption("...")` (at most ~10 words), `stage.label(obj, "...")`,
   `stage.equation("...", where="right")` (LaTeX; it moves the picture aside).
   No `Text(...)`/`MathTex(...)` piles.
3. **One picture at a time.** `stage.clear()` (or `stage.clear(keep=[ax, g])`)
   when the scene moves to a new picture. Don't stack diagrams.
4. **Sensible ranges.** Axes whose `y_range` contains the curve; points on
   the plane within about ±4 × ±3.
5. **Correct mathematics.** Numbers, labels and equations must be right.
6. **Prefer kit blocks**; raw Manim is fine where no block fits (a `Dot`,
   `Line`, `Polygon`, `VGroup`, `.animate`) — keep it simple and on screen.
7. Vary the style across scenes (different blocks, layouts, numbers).

## Kit behaviour worth knowing (fixed 2026-09-28)

- `stage.equation(..., where="right")` moves *everything* on screen (bars,
  networks, raw shapes too) into the other half first; no need to draw on
  the left in advance. `stage.clear(keep=[ax])` still keeps `ax` after it.
- Vector labels (`draw_vector(label=...)`, `draw_basis`) ride the arrow tip
  through `apply_matrix`; `draw_array`'s numbers follow `swap_bars`.
- Axis labels belong to the axes; integer ticks show without ".0".
- A new plane/axes, or a new self-contained picture (dice grid, bars,
  network, Bayes square, ...), clears the previous picture automatically.
- `draw_polygon(stage, pts, where=None)` keeps the points' own coordinates.
- A plotted graph is callable: `g = plot_graph(...)`, then `g(1.5)`.

## Check your work

```bash
.venv/bin/python scripts/check_scenes.py data/kit/claude_scenes/<your batch>.json
```

prints `OK k` or `FAIL k: <error> at <line>` per scene, and writes a
contact sheet of each beat's last frame to `<batch>_sheets/<k>.jpg`. Fix
every FAIL (re-run with a scene list: `... <batch>.json 3,7`). Then **look**
at the sheets (the Read tool shows images) — at least half of them — and fix
any beat whose frame is empty, cluttered, off-screen or shows the wrong
thing. A batch is done when every scene renders and the sheets you looked at
show a clear picture in every beat.

## Do not

- Write about these topics (they are the held-out evaluation):
  harmonic series; chain rule; matrix multiplication as composition; Monty
  Hall; compound interest and e; logarithms; birthday paradox; standard
  deviation; least squares; merge sort; Pascal's triangle; roots of unity;
  mean value theorem; exponential decay / half-life; adding vectors tip to
  tail; √2 irrational; Fibonacci / golden ratio; Fourier transform of a
  chord; determinant zero squashing the plane; central limit theorem.
- Edit anything outside `data/kit/claude_scenes/`. (Finished batches are
  moved to `forge/kit/teacher/`, which is tracked; `data/` is not.)
