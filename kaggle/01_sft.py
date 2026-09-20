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

# Kaggle gives two T4s, and `device_map="auto"` splits a model across both.
# That is a different accelerate path -- dispatch hooks, a forward that is a
# functools.partial rather than a bound method, and gradients that do not
# keep the dtype of their parameters. Five runs of this project died in
# those two symptoms (trl could not patch the LM head; the GradScaler met
# bf16 gradients while every trainable parameter was fp32).
#
# One GPU, chosen before torch is imported, because CUDA_VISIBLE_DEVICES is
# read at initialisation.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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

# Print what actually got installed. `grad_dtype` -- the attribute that lets
# an fp32 parameter carry a bf16 gradient, and the mechanism behind five
# failed runs -- is a torch 2.14 feature, and Kaggle's torch is whatever
# Kaggle ships. Better to see it than to assume it.
import transformers as _tf, trl as _trl, peft as _peft, accelerate as _acc
print(f"[env] torch {torch.__version__} | transformers {_tf.__version__} | "
      f"trl {_trl.__version__} | peft {_peft.__version__} | "
      f"accelerate {_acc.__version__}", flush=True)
print(f"[env] grad_dtype attribute present: "
      f"{hasattr(torch.empty(1), 'grad_dtype')}", flush=True)
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    print(f"[env] {torch.cuda.device_count()}x {_p.name}, "
          f"{_p.total_memory/1e9:.1f} GB, bf16 supported: "
          f"{torch.cuda.is_bf16_supported()}", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (LoraConfig, get_peft_model,
                  prepare_model_for_kbit_training)

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"

# This block follows a published Kaggle T4 QLoRA recipe pinned to these
# exact library versions, rather than the variant I arrived at by iterating
# against the GPU. Twelve runs of one-change-at-a-time is not a method.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,    # a T4 is sm_75: no bf16
    bnb_4bit_use_double_quant=True,
    # Storage dtype of the packed weights. Left unset it follows the model
    # config, which for Qwen2.5 says bfloat16 -- and that propagates into
    # gradients no amount of parameter casting reaches.
    bnb_4bit_quant_storage=torch.float16,
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb,
    dtype=torch.float16,          # `torch_dtype` still works but warns
    attn_implementation="sdpa",
    device_map={"": 0},           # pinned, not "auto" -- see the note above
)
model.config.use_cache = False

# Standard QLoRA preparation: upcasts fp16/bf16 params to fp32, casts the
# norms, makes the output embedding require grad. use_reentrant=False is the
# supported checkpointing path.
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})

# Belt and braces on top of that: every norm explicitly fp32. fp16 training
# overflows in normalisation before it overflows anywhere else.
import torch.nn as nn
_n_norm = 0
for _m in model.modules():
    if (isinstance(_m, (nn.LayerNorm,))
            or _m.__class__.__name__.endswith(("RMSNorm", "LayerNorm"))):
        _m.float()
        _n_norm += 1
print(f"[mem] norms forced to fp32: {_n_norm}", flush=True)


def report_memory(tag: str) -> None:
    """Where the 15 GB goes, printed rather than assumed.

    Reading this after run 10 is what ruled out both the memory ceiling and
    the parameter dtypes, and I still spent two more runs casting parameters.
    """
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[mem] {tag}: {used:.2f} / {total:.1f} GB allocated", flush=True)


report_memory("after kbit prep")
_by_dtype = {}
for _n, _q in model.named_parameters():
    _by_dtype[str(_q.dtype)] = _by_dtype.get(str(_q.dtype), 0) + _q.numel()
print("[mem] parameters by dtype: "
      + ", ".join(f"{k.replace('torch.','')} {v/1e6:.0f}M"
                  for k, v in sorted(_by_dtype.items())), flush=True)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    # Attention *and* MLP projections. Attention-only adapters underfit on a
    # task this syntactically specific — the model has to learn Manim's
    # vocabulary, not just where to attend.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Attach the adapters here rather than handing peft_config to SFTTrainer.
#
# Run 10 failed the same way run 8 did -- bf16 gradients in the AMP unscale
# -- *despite* prepare_model_for_kbit_training. The dtype report explained
# why it was not a contradiction:
#
#     [mem] parameters by dtype: float32 1090M, uint8 3263M
#
# No bf16 parameter existed at that point. The LoRA adapters did not exist
# either: SFTTrainer creates them from peft_config, after this report, and
# PEFT picks their dtype from the base model's config -- which for Qwen2.5
# says bfloat16. The only parameters carrying gradients were therefore the
# only ones the report could not see.
#
# So: create them here, and cast every trainable parameter to fp32. The
# GradScaler only ever touches parameters with gradients, and fp32 LoRA
# weights over a 4-bit base is the standard QLoRA arrangement anyway.
model = get_peft_model(model, peft_cfg)
_recast = [n for n, q in model.named_parameters()
           if q.requires_grad and q.dtype is not torch.float32]
for _n, _q in model.named_parameters():
    if _q.requires_grad and _q.dtype is not torch.float32:
        _q.data = _q.data.to(torch.float32)
print(f"[mem] recast {len(_recast)} trainable params to fp32", flush=True)

_train_dtypes = {}
for _n, _q in model.named_parameters():
    if _q.requires_grad:
        _train_dtypes[str(_q.dtype)] = _train_dtypes.get(str(_q.dtype), 0) + _q.numel()
print("[mem] trainable by dtype: "
      + ", ".join(f"{k.replace('torch.','')} {v/1e6:.1f}M"
                  for k, v in sorted(_train_dtypes.items())), flush=True)
assert set(_train_dtypes) == {"torch.float32"}, (
    f"a trainable parameter is not fp32: {_train_dtypes} — "
    f"the GradScaler has no bf16 CUDA kernel and training will die at the "
    f"first gradient clip")
model.print_trainable_parameters()


# ── the bf16 gradient, cornered ────────────────────────────────────────────
# Five runs died in the GradScaler with
#   NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda
#                        not implemented for 'BFloat16'
# and every explanation I had is ruled out by the reports above: no
# parameter is bf16, all 40.4M trainable ones are fp32, the model is on one
# pinned GPU, quant_storage is fp16. I also confirmed locally that PyTorch
# normalises a gradient to its parameter's dtype before storing it, so a
# backward hook would never even fire.
#
# So this stops theorising and looks. on_pre_optimizer_step runs immediately
# before accelerate unscales, which is the exact frame that raises. It
# prints what is actually in the optimizer -- and casts anything that is not
# fp32, which is also the fix if the report turns out to be boring.
class GradDtypeGuard(TrainerCallback):
    """Report and normalise gradient dtypes at the point the scaler reads them."""

    def __init__(self):
        self.reported = False

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        opt = optimizer
        for _ in range(4):                     # unwrap AcceleratedOptimizer
            inner = getattr(opt, "optimizer", None)
            if inner is None:
                break
            opt = inner
        groups = getattr(opt, "param_groups", None)
        if not groups:
            if not self.reported:
                self.reported = True
                print(f"[grad] no param_groups on {type(optimizer).__name__}",
                      flush=True)
            return

        seen, fixed = {}, 0
        for g in groups:
            for prm in g["params"]:
                if prm.grad is None:
                    continue
                key = f"param={str(prm.dtype).replace('torch.','')} " \
                      f"grad={str(prm.grad.dtype).replace('torch.','')}"
                seen[key] = seen.get(key, 0) + 1
                if prm.grad.dtype is not torch.float32:
                    # torch 2.14 gives tensors a `grad_dtype`, and assigning
                    # a gradient that disagrees with it raises. So clear it
                    # first -- that attribute is how an fp32 parameter comes
                    # to hold a bf16 gradient at all, which is the thing five
                    # runs died on and which no amount of casting parameters
                    # could have reached.
                    # `None` means "allow any dtype", which is what the
                    # error message itself recommends. Setting it to
                    # torch.float32 does *not* permit the reassignment --
                    # tested, and my first attempt at this fix was exactly
                    # that and failed.
                    try:
                        prm.grad_dtype = None
                    except (AttributeError, RuntimeError):
                        pass
                    prm.grad = prm.grad.float()
                    fixed += 1
        if not self.reported:
            self.reported = True
            print(f"[grad] at unscale: " + ", ".join(f"{k} x{v}"
                                                     for k, v in seen.items())
                  + (f"  -- cast {fixed} to fp32" if fixed else ""), flush=True)



# ── 4. train ───────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

# 3% of total steps, computed rather than declared: trl 1.13 has no
# warmup_ratio, only warmup_steps.
# A smoke run exercises every step -- data, model, LoRA, the optimiser step
# that five runs died on, saving, run.json, and the artefact the workflow
# collects -- in about eight minutes instead of four hours. Nothing about
# the resulting adapter is meaningful; the point is that the path works.
# Set by the workflow rewriting this exact line before `kaggle kernels
# push`. It cannot be an environment variable: a Kaggle kernel sees none of
# the runner's environment, which is the same fact that moved the adapter
# upload out of this file and which I nearly forgot again here.
SMOKE = False  # workflow-managed

EPOCHS, BATCH, ACCUM = 3, 1, 8
if SMOKE:
    ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))
    ds["valid"] = ds["valid"].select(range(min(8, len(ds["valid"]))))
    print(f"[smoke] {len(ds['train'])} train / {len(ds['valid'])} valid rows, "
          f"20 steps", flush=True)

steps_per_epoch = max(1, len(ds["train"]) // (BATCH * ACCUM))
total_steps = steps_per_epoch * EPOCHS

cfg = SFTConfig(
    output_dir=str(WORK / "sft-out"),
    num_train_epochs=EPOCHS,
    max_steps=20 if SMOKE else -1,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=1e-4,               # LoRA tolerates far more than full FT
    lr_scheduler_type="cosine",
    warmup_steps=max(10, int(0.03 * total_steps)),
    logging_steps=1 if SMOKE else 10,
    eval_strategy="steps",
    eval_steps=10 if SMOKE else 60,
    save_steps=10 if SMOKE else 120,
    save_total_limit=2,               # Kaggle's output quota is finite
    max_length=2048,                  # was max_seq_length before trl 1.x
    packing=False,

    # fp16 with the GradScaler, as the reference recipe does. My previous
    # run turned AMP off to escape a bf16 unscale error; the real cause was
    # device_map="auto" and an unset quant_storage, both fixed above. On a
    # T4 fp16 is most of the speed, so escaping the error by giving it up
    # was the wrong trade.
    fp16=True, bf16=False,
    max_grad_norm=1.0,

    # Paged 8-bit Adam: optimiser state for 40M LoRA params in fp32 would be
    # ~320 MB of moments, and paging survives a fragmentation spike.
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_num_proc=2,
    report_to="none",
    seed=17,

    # trl 1.13 defaults to chunked_nll, which patches the LM head assuming
    # `forward` is a bound method. Run 7 proved that fails on this model.
    # Plain nll costs ~620 MB of logits at seq=2048 over a 152k vocabulary,
    # and the memory report shows 2.55 of 15.6 GB in use, so it is affordable.
    loss_type="nll",

    # Loss on the completion only. Training the model to predict prompts it
    # will always be handed wastes capacity and dilutes the signal.
    completion_only_loss=True,
)


class StabilityGuard(TrainerCallback):
    """Stop if the loss goes non-finite or runs away.

    fp16 training can diverge quietly and spend the remaining three hours
    producing an adapter that is worse than no adapter. Better to fail at
    step 40 than to finish and measure noise.
    """

    def __init__(self, factor: float = 3.0):
        self.baseline = None
        self.factor = factor

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is None:
            return
        if self.baseline is None:
            self.baseline = loss
        if loss != loss or loss in (float("inf"), float("-inf")) \
                or loss > self.factor * self.baseline:
            print(f"\nSTABILITY GUARD: loss={loss} at step {state.global_step} "
                  f"(baseline {self.baseline:.4f}). Stopping.", flush=True)
            control.should_training_stop = True


# No peft_config: the model is already a PeftModel, with its adapters
# created and cast above where their dtype can be checked.
trainer = SFTTrainer(
    model=model, args=cfg,
    train_dataset=ds["train"], eval_dataset=ds["valid"], processing_class=tok,
    callbacks=[GradDtypeGuard(), StabilityGuard()],
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
# Wrapped, deliberately. This runs *after* the adapter is saved, and a
# non-zero exit here makes the kernel report ERROR -- at which point the
# workflow's poller fails the run and never publishes. A nicety at the end
# of a four-hour job must not be able to throw the job away.
try:
    from transformers import pipeline
    gen = pipeline("text-generation", model=trainer.model, tokenizer=tok)
    msg = [
        {"role": "system", "content": ds["valid"][0]["messages"][0]["content"]},
        {"role": "user",
         "content": "draw a blue circle and transform it into a red square"},
    ]
    out = gen(tok.apply_chat_template(msg, tokenize=False,
                                      add_generation_prompt=True),
              max_new_tokens=400, do_sample=False)[0]["generated_text"]
    print(out[-900:])
except Exception as exc:                                    # noqa: BLE001
    print(f"sanity generation failed ({type(exc).__name__}: {exc}) — "
          f"the adapter is saved and unaffected", flush=True)

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
