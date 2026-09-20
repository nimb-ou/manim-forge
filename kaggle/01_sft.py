# ── Manim Forge · SFT on Kaggle ────────────────────────────────────────────
# Pushed by `kaggle kernels push -p kaggle/` (see kernel-metadata.json), or
# pasted into a notebook: Accelerator GPU T4 x2, Internet ON, with the
# "manim-forge-data" dataset attached.
#
# NOT UNSLOTH, deliberately. Unsloth is roughly 2x faster at ~60% of the
# memory and it is the right answer eventually -- but this run exists to
# measure whether a render-gated corpus is worth anything, and swapping the
# trainer in the same run would put two variables in one experiment. The
# job is about four hours on a T4 against a nine-hour session limit, so
# speed is not the constraint. Switch after the first number is on record.
#
# Kaggle gives 30 GPU-hours a week free and runs notebooks as root, so LaTeX
# and ffmpeg install cleanly. That second part matters more than the GPU:
# GRPO's reward *is* a render, so training and verification must live in the
# same machine. This notebook does SFT; the GRPO one reuses its output.

# ── 1. environment ─────────────────────────────────────────────────────────
# subprocess, not `!pip`. This is pushed as kernel_type "script", which is
# plain Python -- IPython's ! magic is a SyntaxError there, and the whole
# first run died on line 19 before importing anything.
import json, os, subprocess, sys, tarfile, textwrap
from pathlib import Path

# Pinned, not floated. trl renamed max_seq_length -> max_length and dropped
# warmup_ratio between the version this was written against and the one pip
# installs today; an unpinned `trl>=0.12` therefore means the training
# config is a different config every run, which is the opposite of one
# variable per experiment. These are the versions verified locally.
# Pinned to the versions the configs below were actually constructed
# against, locally, before this was pushed. trl renamed max_seq_length ->
# max_length and dropped warmup_ratio between the version this was first
# written for and the one pip installs today, so an unpinned `trl>=0.12`
# means a different training config every run -- the opposite of one
# variable per experiment.
#
# torch is not pinned: Kaggle ships a build matched to its CUDA driver and
# replacing it is how a working GPU becomes a CPU. bitsandbytes is not
# pinned either, and that is the one thing here I could not verify -- it has
# no macOS build, so it is the only dependency going in untested.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.17.0", "trl==1.13.0", "peft==0.21.0",
                "accelerate==1.15.0", "datasets==5.0.1",
                "bitsandbytes>=0.48", "huggingface_hub"], check=True)

WORK = Path("/kaggle/working")


def find_data() -> Path:
    """Locate the mounted dataset instead of assuming where Kaggle puts it.

    It is not /kaggle/input/<slug>. The real path is

        /kaggle/input/datasets/<owner>/<slug>/

    which three consecutive runs died on -- each time reporting that the
    directory "does not exist at all", which was true of the path being
    looked at and false of the data, and which I misdiagnosed twice as a
    version-processing race and built a whole readiness system around.

    So: search for a file the kernel actually needs, and let the answer come
    from the filesystem rather than from a constant.
    """
    root = Path("/kaggle/input")
    hits = sorted(root.rglob("train.jsonl")) if root.exists() else []
    if len(hits) == 1:
        return hits[0].parent
    if len(hits) > 1:
        raise SystemExit(f"train.jsonl found in several places: {hits}")

    print("Could not find train.jsonl anywhere under /kaggle/input.",
          file=sys.stderr)
    if root.exists():
        print("What is mounted:", file=sys.stderr)
        for q in sorted(root.rglob("*"))[:40]:
            print(f"  {q}", file=sys.stderr)
    else:
        print("  /kaggle/input does not exist — no dataset is attached at all.",
              file=sys.stderr)
    raise SystemExit("no dataset; check dataset_sources in kernel-metadata.json")


DATA = find_data()
print(f"dataset mounted at {DATA}")

os.chdir(WORK)

# The same forge package that runs locally, so a number measured here means
# the same thing as a number measured on the laptop.
#
# **Kaggle unpacks archives on upload.** forge.tar.gz was shipped as a
# tarball and arrives already extracted, as forge/forge/, so opening it
# raises FileNotFoundError and takes the run with it -- which is how run 3
# died, one line before it would have loaded any data.
#
# SFT does not import forge at all; only the GRPO notebook does, where the
# reward is a render. So this makes it importable when present and says so
# when it is not, rather than being a hard dependency of a step that never
# uses it.
def _add_forge_to_path() -> str:
    for candidate in (DATA / "forge", DATA, WORK):
        if (candidate / "forge" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return f"forge importable from {candidate}"
    tar = DATA / "forge.tar.gz"
    if tar.exists():                      # if Kaggle ever stops unpacking
        with tarfile.open(tar) as t:
            t.extractall(WORK)
        sys.path.insert(0, str(WORK))
        return f"forge extracted from {tar}"
    return ("forge package not found — fine for SFT, which does not use it; "
            "GRPO will need it")

print(_add_forge_to_path())

# ── 2. data ────────────────────────────────────────────────────────────────
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train": str(DATA / "train.jsonl"),
    "valid": str(DATA / "valid.jsonl"),
})
print(ds)
print("\nexample:\n", textwrap.shorten(ds["train"][0]["messages"][1]["content"], 200))

# ── 3. model ───────────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 4-bit so a 7B fits a single T4 with room for activations.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE)
# `dtype`, not `torch_dtype`: transformers 5 still honours the old name but
# warns, and the warning was sitting in run 7's log while I read past it.
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto", dtype=torch.float16)
model.config.use_cache = False

# The step this recipe was missing. Run 8 reached report_memory("before train")
trainer.train(), completed
# a forward and a backward pass, and died in gradient clipping:
#
#   NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
#                        not implemented for 'BFloat16'
#
# Qwen2.5's config declares bfloat16, fp16=True turns on a GradScaler, and
# torch's AMP unscale has no bf16 CUDA kernel. prepare_model_for_kbit_training
# is the standard QLoRA preparation and it resolves exactly this: it upcasts
# every fp16/bf16 parameter to fp32, casts the layer norms, and makes the
# output embedding require grad. Skipping it is what left bf16 gradients for
# the scaler to choke on.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def report_memory(tag: str) -> None:
    """Where the 15 GB went, before anything has a chance to run out of it.

    Upcasting to fp32 is not free: Qwen's embedding table is 152k x 3584, so
    it costs ~2.2 GB there that it did not cost in fp16, and the un-chunked
    cross-entropy this recipe is forced into wants another ~620 MB of logits
    at seq=2048. If this OOMs, the answer is max_length=1024 -- and this
    print is what makes that a decision rather than a guess.
    """
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[mem] {tag}: {used:.2f} / {total:.1f} GB allocated", flush=True)


report_memory("after kbit prep")
by_dtype = {}
for _n, _p in model.named_parameters():
    by_dtype[str(_p.dtype)] = by_dtype.get(str(_p.dtype), 0) + _p.numel()
print("[mem] parameters by dtype: "
      + ", ".join(f"{k.replace('torch.','')} {v/1e6:.0f}M"
                  for k, v in sorted(by_dtype.items())), flush=True)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    # Attention *and* MLP projections. Attention-only adapters underfit on a
    # task this syntactically specific — the model has to learn Manim's
    # vocabulary, not just where to attend.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ── 4. train ───────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer

# 3% of total steps, computed rather than declared: trl 1.13 has no
# warmup_ratio, only warmup_steps.
EPOCHS, BATCH, ACCUM = 3, 1, 8
steps_per_epoch = max(1, len(ds["train"]) // (BATCH * ACCUM))
total_steps = steps_per_epoch * EPOCHS

cfg = SFTConfig(
    output_dir=str(WORK / "sft-out"),
    num_train_epochs=EPOCHS,          # 3,010 examples, ~900 tokens each
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=1e-4,               # LoRA tolerates far more than full FT
    lr_scheduler_type="cosine",
    warmup_steps=max(10, int(0.03 * total_steps)),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=60,
    save_steps=120,
    save_total_limit=3,               # Kaggle output quota is finite
    bf16=False, fp16=True,
    max_length=2048,                  # was max_seq_length before trl 1.x

    # trl 1.13 defaults to loss_type="chunked_nll", which patches the LM head
    # for a chunked cross-entropy. That patch does
    #     inspect.signature(original_forward.__func__)
    # and on a bitsandbytes-quantised model `forward` is a functools.partial
    # with no __func__, so constructing the trainer raises AttributeError.
    # Run 7 died there, after successfully loading all 339 weight tensors and
    # tokenising the whole corpus.
    #
    # "nll" is plain negative log-likelihood and skips the patch. The cost is
    # real: chunked CE exists to keep the logits tensor small, and Qwen's
    # vocabulary is 152k, so one un-chunked forward at seq=2048 materialises
    # ~620 MB of logits plus the same again for gradients. That should fit
    # beside a 5 GB 4-bit model on a 15 GB T4, but it is the most likely
    # place for this to OOM -- and if it does, the fix is max_length=1024,
    # not a different loss.
    loss_type="nll",
    gradient_checkpointing=True,
    report_to="none",
    # Loss on the completion only. Training the model to predict prompts it
    # will always be handed wastes capacity and dilutes the signal.
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model, args=cfg, peft_config=peft_cfg,
    train_dataset=ds["train"], eval_dataset=ds["valid"], processing_class=tok,
)
report_memory("before train")
trainer.train()
trainer.save_model(str(WORK / "adapter"))
tok.save_pretrained(str(WORK / "adapter"))
print("adapter saved to /kaggle/working/adapter")

# Record what produced this. Every eval number has to be traceable to the mix
# and the config that made it, or "one variable per experiment" is a slogan
# rather than a property -- and it has already been broken three times.
import hashlib
mix = hashlib.sha256(
    (DATA / "train.jsonl").read_bytes()).hexdigest()[:16]
(WORK / "adapter" / "run.json").write_text(json.dumps({
    "base": BASE, "mix_sha256": mix,
    "train_rows": len(ds["train"]), "valid_rows": len(ds["valid"]),
    "epochs": cfg.num_train_epochs, "lr": cfg.learning_rate,
    "lora_r": peft_cfg.r, "lora_alpha": peft_cfg.lora_alpha,
    "target_modules": peft_cfg.target_modules,
    "max_seq_length": cfg.max_seq_length,
}, indent=2))
print(json.dumps(json.loads((WORK / "adapter" / "run.json").read_text()), indent=2))

# ── 4b. the adapter leaves via /kaggle/working ─────────────────────────────
# Not pushed to Hugging Face from here. A Kaggle kernel has no access to the
# runner's environment: HF_TOKEN would have to be attached as a Kaggle
# Secret through the web UI, which is a manual step, and the `or ""` fallback
# above would have skipped the upload silently when it was missing -- the
# adapter would sit in /kaggle/working and the loop would look like it
# worked.
#
# The workflow already runs `kaggle kernels output`, and it already holds
# HF_TOKEN. So it does the push. One secret, in one place.
print(f"adapter is in {WORK / 'adapter'} — the workflow collects it")
for f in sorted((WORK / "adapter").iterdir()):
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")

# ── 5. sanity check: generate one scene ────────────────────────────────────
from transformers import pipeline
gen = pipeline("text-generation", model=trainer.model, tokenizer=tok)
msg = [
    {"role": "system", "content": ds["valid"][0]["messages"][0]["content"]},
    {"role": "user", "content": "draw a blue circle and transform it into a red square"},
]
out = gen(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True),
          max_new_tokens=400, do_sample=False)[0]["generated_text"]
print(out[-900:])

# ── 6. what happens next, and what would make this a failure ───────────────
#
#   ./.venv/bin/python scripts/run_repair_benchmark.py --n 100 --retrieval \
#       --adapter adapters/kaggle-sft --tag tuned100
#   ./.venv/bin/python scripts/run_hard_eval.py --n 81 --backend local \
#       --retrieval --adapter adapters/kaggle-sft --tag tuned81
#
# Baselines to beat, both untuned, both already in docs/RESULTS.md:
#
#   benchmark   93% render at repair rounds=4
#   hard eval   85% render | 16.3s mean | 1.76% length ratio | 8.8% coverage
#
# The render rate is the *least* interesting of those. It will probably rise
# a few points and that proves little -- the untuned model already renders.
# The question this run exists to answer is whether **length ratio and
# concept coverage move at all**, because the corpus analysis in
# docs/PLAN.md §2 predicts they will not: ValueTracker appears in 1.3% of
# training rows against 9.1% of the gold scenes.
#
# If they do not move, this run was not wasted -- it is the evidence that
# Phase 2 (a gate that judges animation, not execution) is the whole
# project. Record the numbers either way.
