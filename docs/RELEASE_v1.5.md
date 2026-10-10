# Manim Forge v1.5 — 2026-10-10

After trying v1.0, Nimit's verdict was that it needed to be more intelligent,
better at visuals and code, and better at deciding what to explain and how.
v1.5 is the engine changes that survived a blind test on requests nobody had
looked at before.

## What changed

- **A lesson plan first.** Before the beats, the model writes `# plan:` lines:
  the one idea, the picture that makes it obvious, every number worked out,
  the arc. (Dev set: +2 good of 30.)
- **Critique and rewrite.** The kit now says what it repaired or saw, in
  words: a point off its axes, a curve that leaves its axes, a vector off the
  screen, a gradient descent that lands in one step, text on text, a frame
  with no picture. A critic adds beats whose code only changes the caption
  ("the grid becomes a parallelogram" over a still grid), arithmetic slips,
  and the exact error and line of a failed render. The draft goes back to the
  model with that list for one rewrite (`forge/app/critique.py`).
- **A kinder kit.**
  - Curves with a hole are plotted around it.
  - Labels take the first free side instead of landing on other text.
  - LaTeX whose backslashes Python swallowed ("imes", "egin{bmatrix}") is restored and set as maths.
  - `draw_vector` takes a from–to pair.
  - A misused plane slot no longer wipes the scene.
  - Gradient descent shows its steps, tangent and trail.
- **Six more checked scenes** in the library (learning rate, neuron,
  projectile, momentum, amplitude and frequency, dividing by zero): 966 now.

## How good it is

Graded blind, three ways, on 20 fresh requests (a computer adding binary,
the moon's phases, a probability tree, normalising a vector, …):

| | good | partial | bad |
|---|---|---|---|
| v1.0 | 7 | 11 | 2 |
| **v1.5** | **10** | 7 | 3 |

Better on 8 requests, worse on 6. Still wrong on: a rainbow (no colours), the
mean of a list (a slipped sum), anything needing a mechanical diagram.

## What was tried and is not in it

- **A fine-tune on its own best scenes** (64 graded good by eye, a LoRA on
  Qwen3.5-9B): 18 vs 15 good on the dev set, 8 vs 10 on the fresh set. No
  difference; published at
  [nimitttt/manim-forge-v1.5-lora](https://huggingface.co/nimitttt/manim-forge-v1.5-lora),
  `FORGE_ADAPTER=hub` to try it.
- **The model checking its own numbers**: it flagged good scenes as often as
  bad ones and "corrected" right maths into wrong. Off.
- **Thinking mode**: spent its whole budget thinking and wrote no scene.

## Cost

About 2.5 minutes a scene when the first draft needs its rewrite, ~50 s when
it does not. Same Mac, same model as v1.0.

CC BY-NC-SA 4.0.
