# Serving Manim Forge

How a prompt typed into a web page becomes a video, where each piece of that
runs, and what it costs. Written against measured numbers from this repo
rather than estimates.

## The numbers that decide the design

End-to-end latency, from the 100-prompt benchmark (generate → repair → render,
480p15, on an M-series Mac):

| | seconds |
|---|---|
| median | **13.4** |
| p75 | 17.9 |
| p90 | 45.7 |
| p95 | 66.4 |
| p99 | 107.1 |

Split by outcome: **11.8s** median when the first generation renders, **48.2s**
when the repair loop has to run. 83% of requests never enter repair.

Two consequences, and they pull in opposite directions:

- A median of 13 seconds is short enough that the page can simply *wait*, with
  something interesting on screen. No email-me-when-done flow is needed.
- A p99 of 107 seconds is far too long to hold an HTTP request open. And one
  gold scene — the 3D sphere with 600 sampled points — takes **1710 seconds**
  at 480p15, which is a different category of thing entirely.

So: a job queue with streamed progress, not a request/response endpoint. The
page feels synchronous; the backend is not.

## Shape

```
  browser ──POST /generate──▶  API  ──enqueue──▶  queue
     ▲                          │                   │
     │                          │            ┌──────┴──────┐
     │                          │            ▼             ▼
     └───────SSE progress───────┘        inference      render
                                          (GPU)        (CPU, many)
                                             │             │
                                             └──▶ object store ──▶ CDN
```

Four services, because they have four different cost shapes:

1. **Static frontend.** No server. Free anywhere.
2. **API + queue.** Always reachable, near-zero compute, holds the SSE stream.
3. **Inference.** Expensive per second, GPU, scales to zero, painful cold start.
4. **Render.** Cheap per second, CPU-only, embarrassingly parallel, needs a
   3 GB image with LaTeX in it.

Fusing 3 and 4 is the obvious mistake: it would hold a GPU idle for the whole
render, which is most of the wall clock on a long scene and needs no GPU at all.

## Where each piece runs

### Frontend — Cloudflare Pages or an HF Space

Static HTML/JS. The existing `forge/app/index.html` is already close. Free,
global, no cold start.

### API and queue — Modal web endpoints

A Modal `@web_endpoint` accepts the prompt, writes a job row, and returns a job
id. A second endpoint streams progress over SSE. Modal bills per second of
actual handler execution, so an idle API costs nothing, and the SSE stream is
cheap because it is mostly waiting.

The alternative — a small always-on VPS — is about $5/month and removes all
cold-start concerns on the API path. Worth it once there is traffic; not worth
it before.

### Inference — Modal, L4, scale-to-zero

[Modal's published rates](https://modal.com/pricing): **L4 $0.000222/s**
($0.80/hr), T4 $0.000164/s, A10 $0.000306/s, A100-40GB $0.000583/s. CPU is
$0.0000131 per physical core per second. The Starter plan includes **$30/month
of free compute** that does not roll over.

A 7B model quantised to 4-bit fits an L4 (24 GB) comfortably and serves at
roughly 40–60 tok/s under vLLM. A typical scene is ~1500 tokens of code, so
**~30s of GPU per generation ≈ $0.0067**.

**Cold start is the real problem, not the rate.** A fresh GPU container takes
30–60 seconds to come up. At low traffic that doubles the median latency and
it is the single worst thing about the user experience. Three mitigations, in
order of preference:

- `min_containers=1` during waking hours only. One L4 held warm for 12 hours a
  day is $9.60 — a third of the free credit, and it removes cold start for
  every request in that window.
- Start the *planner* model first and stream the beat plan to the user while
  the coder container warms. The plan is worth reading, so the wait stops
  being dead time.
- Fall back to the Gemini API when no container is warm and the queue is
  short. Already implemented — `forge/app/generator.py` has `RemoteGemini` and
  `Escalating`.

### Render — Modal, CPU, fan-out per beat

This is where the architecture earns its keep. Beats are already separate
sections (`next_section()` + `--save_sections`), so a 15-beat scene is **15
independent render jobs**. Fan them out across containers, collect, concatenate
with ffmpeg.

Cost: a 10-second render on one physical core is $0.00013. Even a
two-minute render is $0.0016. Render compute is, in practice, free.

What is not free is the image. Manim + a TeX distribution is ~3 GB. Modal
caches images per-revision, so this is a one-off per deploy, but it argues
strongly for pinning the image and never rebuilding it casually.

### Storage and delivery — R2 or S3

Videos are a few MB each. Cloudflare R2 has no egress charge, which matters for
a site whose whole output is video. Serve behind the CDN, expire anonymous
renders after a week, keep signed-in users' work.

### Cost per video, end to end

| | |
|---|---|
| inference (~30s L4) | $0.0067 |
| render (~15s CPU) | $0.0002 |
| storage + egress | ~$0.0001 |
| **total** | **≈ $0.007** |

$30 of free monthly credit is therefore **roughly 4,000 videos a month**,
before any warm-container spend. That is a real free tier for an educational
site, and it is the reason this can stay free as the user intends.

## What the user sees

The beat structure is not only a training-data decision; it is the entire
reason the interface can feel responsive.

**1. Prompt.** One box, and a gallery of worked examples underneath — the gold
scenes, which are the best work in the repo and double as the style promise.

**2. The plan, streamed.** The planner returns beats before any code exists:

```
  ▸ Draw the axes and mark the origin            8s
  ▸ Place a point and drop guide lines          11s
  ▸ Swap the coordinates, land somewhere else   11s
  ▸ A rule picks out a shape                    13s
```

This is the most important screen in the product. It arrives in a second or
two, it is legible, and it converts the wait into reading. It is also the
natural place to intervene — reorder, delete, or rewrite a beat *before*
spending compute on it.

**3. Beats render and appear as they finish.** Because they are independent
jobs, beat 1 can be playing while beat 7 is still rendering. A progress strip
along the bottom fills in left to right.

**4. The video, and the timeline.** Beats become the editing unit — the
Canva-like part. Per beat: regenerate, retime, edit narration, swap colours.
Regenerating one beat costs one beat's compute, not the whole scene.

**5. Download.** MP4 at three qualities, GIF, the per-beat clips, the narration
script as text, and the Python source. Giving away the source is the right
default for an educational tool and costs nothing.

## Failure, which is not an edge case

15% of hard-eval prompts do not render at all. The site has to be honest about
that rather than spinning.

The render gate already in this repo is the production error path, not just a
corpus filter. When a beat fails, the classifier in `forge/harness/errors.py`
says what kind of failure it is, and the response differs:

- `api_misuse`, `name`, `syntax` — repair loop, up to 4 rounds. Invisible to
  the user; it is why median latency is 13s and not 48s.
- `latex_missing`, `memory`, `ffmpeg` — environment. Never the user's fault,
  never the model's. Retry on a different container, alert.
- Still failing after repair — show the beats that *did* render, say plainly
  that one did not, and offer to retry that beat alone. A partial scene is far
  better than a spinner that ends in nothing.

## Order of work

1. Ship the renderer as a Modal function, fanned out per beat, driven by the
   existing harness. Render compute is the cheap, well-understood half, and
   getting it working proves the queue and storage paths.
2. Point inference at the Gemini API first. The site works end to end before
   the fine-tune is ready, and the comparison is then measured rather than
   assumed.
3. Swap in the fine-tuned model behind the same interface, and keep Gemini as
   the escalation path. `forge/app/generator.py` is already built this way.
4. Add the beat timeline editor last. It is the most product work and the
   least infrastructure risk, and it needs the rest to be solid first.

## Sources

- Modal pricing — https://modal.com/pricing
- Hugging Face Spaces hardware and ZeroGPU — https://huggingface.co/docs/hub/en/spaces-overview
