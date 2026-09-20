# Setup — the things only Nimit can do

Everything here needs a human: an account, a token, a payment, or a decision.
Nothing else in the project does. Ordered by when it blocks something, so you
can stop at any line and I still have work.

Total cost of the whole list: **one domain, about $10 a year.** Everything
else is a free tier, and §7 explains what happens when one is exhausted.

**Never paste a key into the chat.** Put it in `.env` in the project root
(gitignored) as `NAME=value`, or into GitHub Secrets. I read them from the
environment and verify they work without ever printing the value. `python -m
forge.doctor` shows which keys are present by name only.

---

## Block A — unblocks Phase 1 (training). Do these first.

### A1. Make the GitHub repo public · free · 1 minute

```bash
gh repo edit nimb-ou/manim-forge --visibility public --accept-visibility-change-consequences
```

**Why it matters more than it sounds.** GitHub Actions is unlimited and free
on public repositories, and — the part that decides it — **scheduled
workflows do not run at all on a free private repo.** No cron means no
nightly `doctor`, no automated training run, no unattended anything. The
alternatives are $4/month for Pro or publishing the code, and for a project
whose stated end is free and educational that is not a close call.

**What becomes public:** the code, the 44 gold scenes, the docs, the
postmortem. **What does not:** `data/` is gitignored and stays on Hugging
Face — it contains rows from five scraped datasets whose licences I have not
finished reviewing, and they should not be republished by accident.

Worth reading `docs/POSTMORTEM.md` before you push the button. It is honest
about eleven defects and I would not change a word of it, but you should know
it is there.

### A2. Kaggle account + API token · free · 5 minutes

1. Sign up at [kaggle.com](https://kaggle.com) and **verify your phone
   number** — GPU access is gated behind that, and it is easy to miss.
2. Account → Settings → API → **Create New Token** → downloads `kaggle.json`.
3. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

**What it buys:** 30 GPU-hours a week, free, in 9-hour sessions, on a dual
T4 (2×16 GB) or a P100. Our training mix is about 2.7M tokens an epoch — with
Unsloth and QLoRA that is minutes, not hours, so the quota is generous rather
than tight. Kaggle also gives root, which means LaTeX and ffmpeg install and
**the render gate runs in the same place as training.** That is not a
convenience: GRPO's reward *is* a render, so Phase 3 requires it.

### A3. Hugging Face write token · free · 3 minutes

You have an account (`nimitttt`) but there is no token in `.env` — the doctor
reports `-- huggingface`.

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) →
   New token → **Write**.
2. Add to `.env`: `HF_TOKEN=hf_...`

**What it buys:** 100 GB of private storage free, effectively unlimited
public. The corpus is 460 MB. This is where the dataset lives so Kaggle can
pull it, where the trained adapters go, and where the backup runs to.

---

## Block B — unblocks Phase 4 (serving). Needed in ~3 weeks, not today.

### B1. Modal account · free · 5 minutes

1. Sign up at [modal.com](https://modal.com) with GitHub. No card required.
2. `pip install modal && modal token new`

**What it buys:** $30 of compute credit every month, per-second billing,
scale-to-zero. Measured against our own latency that is roughly **4,000–6,000
videos a month at zero marginal cost**: inference on an L4 costs $0.003 a
video, the render another $0.0016.

**One thing to set the day you sign up:** a hard spend cap at $30. An
autonomous system with a payment method attached and no ceiling is the single
failure mode in this plan that can cost real money.

### B2. Cloudflare account · free · 5 minutes

Sign up at [cloudflare.com](https://cloudflare.com). Three things at once:

- **R2** — object storage with **zero egress fees**, 10 GB free. For a video
  product that is the entire storage decision; the same bytes leaving S3
  would dominate the bill.
- **Pages** — the frontend. Static assets are unmetered.
- **Registrar** — at-cost domains, see B3.

### B3. A domain · ~$10/year · needs a decision from you

Cloudflare Registrar charges the wholesale registry price plus the ICANN fee
and adds nothing: about **$8.50/yr for `.org`**, **$10.46 for `.com`**. No
first-year discount that triples on renewal.

**This is the one thing I cannot decide for you.** Some directions, given
that it is free, educational, and non-commercial:

| | |
|---|---|
| `.org` | reads non-commercial, which is what this is, and is the cheapest |
| `.dev` | reads like a tool for people who write Manim |
| `.com` | the default, and the one people mistype least |

Whatever you pick, tell me and I will wire DNS, Pages and R2 to it. Nothing
before Phase 4 depends on the name, so there is no hurry — but names go, so
if one you like is free it is $10 to hold it.

---

## Block C — optional, any time. Each one makes something faster.

### C1. Groq API key · free · 2 minutes

[console.groq.com](https://console.groq.com) → API Keys. Add to `.env` as
`GROQ_API_KEY=...`.

Roughly 1,000 requests a day on Llama-3.3-70B, and Groq is genuinely fast.
More teacher capacity is worth having *after* Phase 2 defines what a good row
is — before that it would only generate more of what the plan says not to
generate.

**A warning about this whole category.** Between June and September 2026,
Cerebras, GitHub Models, Together and SambaNova all dropped or gated their
free tiers, and Groq cut its model list. The provider rotation in
`forge/synth/teacher.py` is built so that any one of them disappearing costs
us nothing. Do not build a plan around a free tier; treat each one as a
windfall.

### C2. GitHub Secrets · free · 5 minutes · needed for automated training

Once A1 and A2 are done, for Actions to push training runs on its own:

Repo → Settings → Secrets and variables → Actions → New repository secret:

| name | value |
|---|---|
| `KAGGLE_USERNAME` | from `kaggle.json` |
| `KAGGLE_KEY` | from `kaggle.json` |
| `HF_TOKEN` | the same token as A3 |

### C3. Gemini key — already working

In `.env`, verified. Measured reality: about **900–1,000 calls a day in
total**, not the documented 1,500 per model. Rotation across models reaches
that ceiling; it does not raise it.

---

## What I do with each

| you do | I can then | phase |
|---|---|---|
| A1 repo public | nightly doctor, CI, scheduled training | 1 |
| A2 Kaggle | run the fine-tune and measure it | 1 |
| A3 HF token | publish the dataset, store adapters, back up | 1 |
| B1 Modal | deploy inference and render | 4 |
| B2 Cloudflare | store and serve video, host the site | 4 |
| B3 domain | put it on a real address | 4 |
| C1 Groq | generate faster, once there is a measure worth generating against | 2 |
| C2 secrets | training without you starting it | 5 |

---

## When a free tier runs out

Stated up front, so it is never a surprise mid-run.

| | what happens | what I do |
|---|---|---|
| Kaggle 30 h/week | training blocks | fall back to Modal credit (~50 T4-hours), or wait for the weekly reset |
| Modal $30/mo | serving stops at the cap | the cap is the point — it fails closed, not expensively |
| R2 10 GB | ~1,400 videos at 7 MB | prune oldest, or $0.015/GB/month |
| Gemini ~950/day | generation parks | CPU work expands; resets at Pacific midnight |
| HF 100 GB private | nothing soon | 460 MB used |
| Actions on public | no practical limit | — |

Nothing here fails into a bill. Every ceiling stops work instead of
continuing it at cost, which is deliberate.
