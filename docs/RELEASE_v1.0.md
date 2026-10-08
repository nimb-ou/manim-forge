# Manim Forge v1.0 — 2026-10-08

Type a sentence, get a short, narrated, 3Blue1Brown-style animation, made
entirely on a 16 GB Apple Silicon Mac. No cloud, no API key.

## What is in it

- **One-shot generation** (`forge/app/oneshot.py`): one call writes the whole
  scene — beats, narration, code — so later beats build on earlier ones.
- **Retrieval over 942 hand-written, checked scenes** (`forge/kit/library.py`):
  the two nearest are shown to the model to adapt.
- **Qwen3.5-9B, untuned**, 4-bit under MLX, with the Forge kit's reference in
  its prompt.
- **The Forge kit** (`forge/kit/kit.py`): 64 animation blocks a small model
  can call reliably.
- **Checks**: on-screen arithmetic, text collisions, off-frame objects, empty
  pictures, render failures — a failing sample is replaced (best of 2).
- **Narration**: Kokoro-82M speaks each beat; the beat waits for its line.
- **A local web app** (`python -m forge.serve`) that streams the beats and
  code as they are written and plays the video.

## How good it is

Graded by eye on 20 prompts per set, as shipped (`docs/RESULTS.md`):

| | good | partial | bad |
|---|---|---|---|
| in-scope: school maths and everyday quantities | 12 | 6 | 2 |
| held-out: 20 classic topics with no library scene | 9 | 4 | 7 |

About 50 s a scene; about 3 minutes when a second sample is needed. The
gallery (`docs/GALLERY.md`) has ten scenes and the four that did not make it.

## Known limits

- It is sometimes confidently wrong: a wrong number or the wrong question
  answered, in roughly one scene in ten in-scope and more on harder topics.
- Pictures are simple; long arcs (beyond ~6 beats) are not attempted.
- English only; ManimCE 0.21; Apple Silicon only (MLX).

## How it got here

Three weeks, ~500 commits, 16 fine-tuned adapters that v1.0 does not use:
retrieval, a stronger base model and checks beat every one of them. The
whole story, mistakes included: `docs/HISTORY.md`.

CC BY-NC-SA 4.0.
