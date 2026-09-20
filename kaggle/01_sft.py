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

# Run 15 died in backward at step 3, trying to allocate 1.16 GiB with 208 MB
# free and **2.39 GB reserved but unallocated** -- the card was not full, it
# was fragmented. 1.16 GiB is not a mystery either: 2048 tokens x 151936
# vocab x 4 bytes is the logits tensor, to the byte, which is the one
# allocation in this model that is both huge and short-lived, and so the one
# that a fragmented pool cannot find room for.
#
# expandable_segments lets the allocator grow a segment instead of hunting
# for a contiguous block of exactly the right size. torch's own error
# message recommends it for this signature. Like CUDA_VISIBLE_DEVICES it is
# read when the allocator initialises, so it has to be set before the import.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainerCallback)
# Imported here, not in section 4. GradSafeSFTTrainer subclasses SFTTrainer,
# and a base class is evaluated when the class statement runs -- the exact
# mistake that killed the previous smoke run with TrainerCallback.
from trl import SFTConfig, SFTTrainer
from peft import (LoraConfig, get_peft_model,
                  prepare_model_for_kbit_training)

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"


def train_once(use_fp16: bool, smoke: bool, tag: str,
               max_len: int = 2048) -> dict:
    """Build the model and train it once, reporting whether it stayed finite.

    This is a function rather than a straight-line script so the kernel can
    make its own decision and act on it without me. Kaggle gives a GPU
    session twelve hours; a 20-step smoke costs about fifteen minutes. So the
    kernel smoke-tests the exact path it is about to spend four hours on,
    and if the numerics come out NaN it rebuilds from scratch with AMP off
    and tries again -- which is the one configuration that cannot hit either
    of the dtype failures, at the cost of T4 throughput.

    Every attempt rebuilds the model: training mutates it, and a retry that
    reuses a NaN-poisoned model measures nothing.
    """

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
    def _normalise_grads(model) -> dict:
        """Make every gradient match fp32 before anything unscales it.

        torch gives tensors a `grad_dtype` (present on Kaggle's 2.10, confirmed
        by the [env] line), and when it is set an fp32 parameter legitimately
        carries a bf16 gradient. torch's AMP unscale has no bf16 CUDA kernel, so
        five runs died there -- and casting *parameters*, which I did three
        times, could never reach it.
        """
        seen, fixed = {}, 0
        for prm in model.parameters():
            if prm.grad is None:
                continue
            key = (f"param={str(prm.dtype).replace('torch.', '')} "
                   f"grad={str(prm.grad.dtype).replace('torch.', '')}")
            seen[key] = seen.get(key, 0) + 1
            if prm.grad.dtype is not torch.float32:
                # None means "allow any dtype" -- setting it to torch.float32
                # does *not* permit the reassignment. Tested; my first attempt
                # at this fix was exactly that and raised.
                try:
                    prm.grad_dtype = None
                except (AttributeError, RuntimeError):
                    pass
                prm.grad = prm.grad.float()
                fixed += 1
        return {"seen": seen, "fixed": fixed}


    class GradSafeSFTTrainer(SFTTrainer):
        """SFTTrainer that normalises gradient dtypes in the frame that fails.

        transformers calls, in this order:

            if self.args.max_grad_norm > 0:
                grad_norm = self._clip_grad_norm(model)          <- raises
            ...
            self.callback_handler.on_pre_optimizer_step(...)     <- too late

        My first attempt put the fix in that callback, which is one frame after
        the exception. The [grad] line simply never printed, which is how I
        know: an instrument that stays silent is telling you it was not reached.
        """

        _grad_reported = False

        def _clip_grad_norm(self, model):
            info = _normalise_grads(model)
            if not GradSafeSFTTrainer._grad_reported:
                GradSafeSFTTrainer._grad_reported = True
                print("[grad] at clip: "
                      + ", ".join(f"{k} x{v}" for k, v in info["seen"].items())
                      + (f"  -- cast {info['fixed']} to fp32" if info["fixed"]
                         else "  -- nothing to cast"), flush=True)
            return super()._clip_grad_norm(model)


    # ── 4. train ───────────────────────────────────────────────────────────────

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

    EPOCHS, BATCH, ACCUM = 3, 1, 8

    # Local names, not `ds[...] = ds[...].select(...)`. The smoke attempt used
    # to assign back into the module-level DatasetDict, which meant the full
    # run that follows it in the same kernel would have trained on the 64
    # rows the smoke left behind -- and finished, and saved, and reported
    # success. A four-hour run against 1.5% of the corpus is worse than a
    # crash, because nothing about the log would say so.
    train_ds, valid_ds = ds["train"], ds["valid"]
    if smoke:
        train_ds = train_ds.select(range(min(64, len(train_ds))))
        valid_ds = valid_ds.select(range(min(8, len(valid_ds))))
        print(f"[smoke] {len(train_ds)} train / {len(valid_ds)} valid rows, "
              f"20 steps", flush=True)
    print(f"[data] training on {len(train_ds)} rows, "
          f"evaluating on {len(valid_ds)}", flush=True)

    steps_per_epoch = max(1, len(train_ds) // (BATCH * ACCUM))
    total_steps = steps_per_epoch * EPOCHS

    cfg = SFTConfig(
        output_dir=str(WORK / f"sft-out-{tag}"),
        num_train_epochs=EPOCHS,
        max_steps=20 if smoke else -1,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=ACCUM,
        learning_rate=1e-4,               # LoRA tolerates far more than full FT
        lr_scheduler_type="cosine",
        warmup_steps=max(10, int(0.03 * total_steps)),
        logging_steps=1 if smoke else 10,
        eval_strategy="steps",
        eval_steps=10 if smoke else 60,
        save_steps=10 if smoke else 120,
        save_total_limit=2,               # Kaggle's output quota is finite
        # Not lowered by default, and that is a measurement rather than a
        # preference: 2048 already truncates 9.1% of the corpus and 1536
        # truncates 18.4%. Truncation here does not trim padding, it cuts
        # the *end off a scene* -- so the cheap memory saving teaches the
        # model to emit code that stops mid-construct, on a task whose gate
        # is whether the code renders. It is the last lever to pull, not the
        # first, which is why it is a retry below rather than a constant.
        max_length=max_len,               # was max_seq_length before trl 1.x
        packing=False,

        # fp16 with the GradScaler, as the reference recipe does. My previous
        # run turned AMP off to escape a bf16 unscale error; the real cause was
        # device_map="auto" and an unset quant_storage, both fixed above. On a
        # T4 fp16 is most of the speed, so escaping the error by giving it up
        # was the wrong trade.
        fp16=use_fp16, bf16=False,
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
            self.failed = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            # transformers 5.x formats logged scalars as *strings*:
            #   {'loss': '0', 'grad_norm': 'nan', 'entropy': 'nan', ...}
            # so `loss != loss` is False for a NaN and `loss > 3 * baseline`
            # compares str to float. This guard was a silent no-op while eight
            # consecutive NaN steps went past it. Coerce, and treat an
            # uncoercible value as a reason to stop rather than to continue.
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                print(f"\nSTABILITY GUARD: uninterpretable loss {loss!r} at step "
                      f"{state.global_step}. Stopping.", flush=True)
                control.should_training_stop = True
                self.failed = True
                return
            if self.baseline is None:
                self.baseline = loss
            if loss != loss or loss in (float("inf"), float("-inf")) \
                    or loss > self.factor * self.baseline:
                print(f"\nSTABILITY GUARD: loss={loss} at step {state.global_step} "
                      f"(baseline {self.baseline:.4f}). Stopping.", flush=True)
                control.should_training_stop = True
                self.failed = True


    guard = StabilityGuard()

    # No peft_config: the model is already a PeftModel, with its adapters
    # created and cast above where their dtype can be checked.
    trainer = GradSafeSFTTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=valid_ds, processing_class=tok,
        callbacks=[guard],
    )
    # The assert above passed -- all 40.4M trainable parameters were fp32 -- and
    # yet the run reported, from inside the training loop:
    #
    #     [grad] at clip: param=bfloat16 grad=bfloat16 x392  -- cast 392 to fp32
    #
    # 392 is 28 layers x 7 target modules x {A, B}: every LoRA parameter. So
    # something between get_peft_model and the first optimiser step casts them,
    # and bfloat16 appears nowhere in this file -- it is the dtype in Qwen2.5's
    # own config, which some layer of Trainer/accelerate/peft applies on the way
    # in. On a T4 (sm_75, no bf16) with fp16 AMP that is incoherent, and the run
    # went NaN at step 3 and stayed there.
    #
    # I am not going to spend another run identifying which library does it.
    # Checking once before construction was the mistake; check at every boundary
    # and fix at the last one that still has a fix available.
    def _force_fp32_trainables(model, where: str) -> None:
        bad = {}
        for _n, _q in model.named_parameters():
            if _q.requires_grad and _q.dtype is not torch.float32:
                bad[str(_q.dtype)] = bad.get(str(_q.dtype), 0) + 1
                _q.data = _q.data.to(torch.float32)
        print(f"[dtype] {where}: recast {sum(bad.values())} trainable params"
              + (f" from {bad}" if bad else " (none needed)"), flush=True)


    _force_fp32_trainables(trainer.model, "after trainer construction")


    class Fp32Trainables(TrainerCallback):
        """Re-assert fp32 trainables once the accelerator has prepared the model.

        on_train_begin fires after Trainer.train() has called
        accelerator.prepare, which is the last frame before the first forward
        and therefore the last place a cast can still land ahead of the
        optimiser reading these tensors.
        """

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if model is not None:
                _force_fp32_trainables(model, "on_train_begin")


    trainer.add_callback(Fp32Trainables())
    report_memory("before train")
    trainer.train()

    return {
        "ok": not guard.failed,
        "trainer": trainer, "cfg": cfg, "peft_cfg": peft_cfg,
        "guard": guard, "tok": tok,
        "train_rows": len(train_ds), "valid_rows": len(valid_ds),
    }

# ── 4c. the kernel decides, and the decision is on the record ──────────────
# Thirteen runs of this project ended with me reading a log and pushing a
# slightly different kernel. That loop is the bottleneck, not the GPU, and
# it stops here: the plan below is executed by the kernel itself.
#
#   1. smoke with AMP on   -- 20 steps, ~15 min, exercises every frame that
#                             has ever failed: data, 4-bit load, LoRA, the
#                             clip that five runs died in, eval, saving.
#   2. if that stayed finite, train for real with AMP on.
#   3. if it did not, smoke again with AMP off and, if *that* is finite,
#      train for real without AMP. fp16=False cannot hit the unscale kernel
#      or the bf16/fp16 mismatch at all; it costs T4 throughput, which is a
#      price worth paying for an adapter that exists.
#
# If step 3's smoke also goes NaN the fault is not AMP, and a four-hour run
# would only produce an expensive version of the same information -- so the
# kernel stops and says so.
import gc

RESULT = None
ATTEMPTS = []

# Two independent things kill an attempt here, and they want opposite
# remedies, so the next attempt is chosen from *why* the last one failed
# rather than read off a fixed list.
#
#   non-finite -> turn AMP off. Costs T4 throughput and nothing else.
#   oom        -> shorten the sequence. Costs corpus, permanently, in the
#                 adapter, so it is the lever of last resort.
#
# Marching down a fixed list would answer an OOM by turning AMP off, which
# makes activations fp32 and therefore uses *more* memory -- twenty minutes
# spent proving something already known.
def next_attempt(use_fp16: bool, max_len: int, why: str):
    if why == "oom":
        # 2048 already truncates 9.1% of the corpus and 1536 truncates 18.4%.
        # Below that the cure is worse than the disease: at 1024 it is 56%,
        # and an adapter trained on half-scenes is not worth a GPU session.
        return (use_fp16, 1536) if max_len > 1536 else None
    if use_fp16:
        return (False, max_len)
    return None


_attempt, _seen = (True, 2048), set()

while _attempt is not None and _attempt not in _seen:
    _seen.add(_attempt)
    _use_fp16, _max_len = _attempt
    _label = f"{'fp16' if _use_fp16 else 'fp32'}-{_max_len}"
    print(f"\n{'=' * 74}\n[plan] smoke {_label} "
          f"(AMP {'on' if _use_fp16 else 'off'}, max_len {_max_len})"
          f"\n{'=' * 74}", flush=True)
    try:
        _smoke = train_once(use_fp16=_use_fp16, smoke=True,
                            tag=f"smoke-{_label}", max_len=_max_len)
        _ok, _why = _smoke["ok"], None if _smoke["ok"] else "non-finite"
    except torch.OutOfMemoryError as _exc:
        # Caught rather than allowed to end the kernel, because an OOM is
        # information about *this* combination and not about the next one,
        # and because the whole point of the plan is that a failure costs
        # fifteen minutes instead of a day.
        _smoke, _ok, _why = {"ok": False}, False, "oom"
        print(f"[plan] {_label} smoke ran out of memory: "
              f"{str(_exc).splitlines()[0]}", flush=True)
    ATTEMPTS.append({"phase": "smoke", "amp": _use_fp16, "max_len": _max_len,
                     "ok": _ok, "why": _why})

    # The next attempt loads a second 7B model into a 15 GB card, so the
    # first one has to be genuinely gone -- not merely out of scope. Trainer,
    # model and callbacks reference each other, so a plain del leaves a cycle
    # that only gc.collect() breaks, and empty_cache() before that frees
    # nothing. The number is printed because "I freed it" is a claim and
    # 0.1 GB is evidence.
    del _smoke
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[mem] after releasing the smoke model: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB still allocated",
          flush=True)
    if not _ok:
        _attempt = next_attempt(_use_fp16, _max_len, _why)
        print(f"[plan] smoke {_label} failed ({_why}); not spending four "
              f"hours on it. Next: {_attempt or 'nothing left to try'}",
              flush=True)
        continue

    print(f"\n{'=' * 74}\n[plan] full run, {_label}\n{'=' * 74}", flush=True)
    try:
        RESULT = train_once(use_fp16=_use_fp16, smoke=False,
                            tag=f"full-{_label}", max_len=_max_len)
        _full_ok, _full_why = RESULT["ok"], None if RESULT["ok"] else "non-finite"
    except torch.OutOfMemoryError as _exc:
        # A smoke that fit and a full run that did not is a real case: the
        # smoke sees 64 rows and the longest scenes are in the other 2946.
        RESULT, _full_ok, _full_why = None, False, "oom"
        print(f"[plan] {_label} full run ran out of memory: "
              f"{str(_exc).splitlines()[0]}", flush=True)
    ATTEMPTS.append({"phase": "full", "amp": _use_fp16, "max_len": _max_len,
                     "ok": _full_ok, "why": _full_why})
    if _full_ok:
        break
    # A smoke that held for 20 steps and a full run that did not is a real
    # case rather than a contradiction: divergence arrives with the learning
    # rate and the smoke barely leaves warmup, and the longest scenes in the
    # corpus are in the 2946 rows the smoke never sees.
    _attempt = next_attempt(_use_fp16, _max_len, _full_why)
    print(f"[plan] the full run failed ({_full_why}) after a clean smoke. "
          f"Next: {_attempt or 'nothing left to try'}", flush=True)
    RESULT = None
    gc.collect()
    torch.cuda.empty_cache()

print("\n[plan] attempts: " + json.dumps(ATTEMPTS), flush=True)
if RESULT is None:
    raise SystemExit(
        "every combination in the plan failed; see the 'why' field above. "
        "All non-finite means AMP is not the cause and the [dtype] lines are "
        "the next thing to read. All oom at 1536 means the logits tensor is "
        "not the binding constraint and the next lever is the optimiser or "
        "chunked loss, not another sequence length.")

# The tokenizer comes back out with the rest of it. It used to be a module
# global; wrapping the build in a function made it a local, and the three
# places below that save and use it would each have raised NameError --
# after the four-hour run had already succeeded. scripts/check_kernel.py
# found that, which is the entire reason that file exists.
trainer = RESULT["trainer"]
cfg, peft_cfg, tok = RESULT["cfg"], RESULT["peft_cfg"], RESULT["tok"]

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
    "train_rows": RESULT["train_rows"],
    "valid_rows": RESULT["valid_rows"],
    "epochs": cfg.num_train_epochs, "lr": cfg.learning_rate,
    "lora_r": peft_cfg.r, "lora_alpha": peft_cfg.lora_alpha,
    "target_modules": peft_cfg.target_modules,
    "max_length": cfg.max_length,
    "amp_fp16": cfg.fp16,
    "attempts": ATTEMPTS,
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
