"""Score a one-shot adapter on Kaggle: generate on the T4, render on the CPUs.

Written 2026-10-07, when the Mac running the evaluations turned out to be on
battery and fifty times slower than usual. The scoring is the Mac's own
scripts/scorecard.py, unchanged: only the two functions that touch MLX --
forge.app.pipeline.load and .ask -- are replaced by transformers + PEFT
versions, and `mlx.core` is stubbed for the per-prompt seed. Retrieval,
parsing, assembly, salvage, rendering and contact sheets are the same code,
so the output directories judge on the Mac like any other run.

Inputs:  dataset nimbou/manim-forge-eval-code (forge_eval.tar.gz: forge/ and
         the scorecard scripts), and the SFT kernel's output (adapter/).
Output:  /kaggle/working/scorecard/<tag>/ for each run in SPECS.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

t0 = time.time()
# (prompt-set flag, tag, samples, base model, use the trained adapter, kit API in prompt)
# Run 2 (2026-10-07): v1 scored worse in-scope than the untuned base did on
# the Mac (5 vs 12 of 16 good by eye). A separates the engine (transformers
# nf4 here, MLX there) from the training; B asks whether a 2026 base model,
# untuned, does better still.
SPECS = [
    ("--inscope", "is_base_k", 1, "Qwen/Qwen2.5-Coder-7B-Instruct", False, True),
    ("--inscope", "is_q35_9b", 1, "Qwen/Qwen3.5-9B", False, True),
    ("--heldout", "held_q35_9b", 1, "Qwen/Qwen3.5-9B", False, True),
]


sys.stdout.reconfigure(line_buffering=True)


def sh(cmd, t=3600):
    # Streamed, not captured: version 1 and 2 of this kernel died during set-up
    # with an empty log, because everything was captured and nothing printed.
    print("$", cmd[:140], flush=True)
    r = subprocess.run(cmd, shell=True, text=True, timeout=t)
    print("  exit", r.returncode, round(time.time() - t0), "s", flush=True)
    return r


sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
   "libcairo2-dev libpango1.0-dev ffmpeg texlive-latex-base texlive-latex-extra "
   "texlive-fonts-recommended dvisvgm 2>&1 | tail -3")
sh(f"{sys.executable} -m pip install -q manim==0.21.0 transformers==5.17.0 "
   "peft==0.21.0 accelerate==1.15.0 'bitsandbytes>=0.48' 2>&1 | tail -5")
sh("nvidia-smi --query-gpu=name,memory.used --format=csv; free -g; df -h /kaggle/working")
print("setup", round(time.time() - t0), "s", flush=True)

adapter = Path(glob.glob("/kaggle/input/**/adapter/adapter_config.json",
                         recursive=True)[0]).parent
ROOT = Path("/kaggle/working/mf")
ROOT.mkdir(parents=True, exist_ok=True)
# Kaggle sometimes unpacks an uploaded archive and sometimes does not.
unpacked = glob.glob("/kaggle/input/**/scripts/scorecard.py", recursive=True)
if unpacked:
    shutil.copytree(Path(unpacked[0]).parents[1], ROOT, dirs_exist_ok=True)
    print("code (unpacked)", unpacked[0], flush=True)
else:
    tar = glob.glob("/kaggle/input/**/forge_eval.tar.gz", recursive=True)[0]
    with tarfile.open(tar) as t:
        t.extractall(ROOT)
    print("code", tar, flush=True)
print("adapter", adapter, flush=True)
# finish() renders with <root>/.venv/bin/python.
(ROOT / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
os.symlink(sys.executable, ROOT / ".venv" / "bin" / "python")
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

mlx = types.ModuleType("mlx")
core = types.ModuleType("mlx.core")
core.random = types.SimpleNamespace(seed=lambda s: torch.manual_seed(s))
core.clear_cache = lambda: None
mlx.core = core
sys.modules["mlx"], sys.modules["mlx.core"] = mlx, core

CUR = {}


def use(base: str, trained: bool):
    """Load a base (4-bit nf4) and optionally the one-shot adapter, freeing
    the previous model first: one 9B and one 7B do not fit a T4 together."""
    key = (base, trained)
    if CUR.get("key") == key:
        return
    CUR.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base, device_map={"": 0}, dtype=torch.float16, quantization_config=q)
    except (ValueError, KeyError) as exc:      # a vision-language checkpoint
        print("CausalLM failed, trying ImageTextToText:", exc, flush=True)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            base, device_map={"": 0}, dtype=torch.float16, quantization_config=q)
    if trained:
        model = PeftModel.from_pretrained(model, str(adapter))
    CUR.update(key=key, model=model.eval(), tok=tok)
    print("loaded", base, "adapter" if trained else "untuned",
          round(time.time() - t0), "s", flush=True)


def hf_load(_adapter=None):
    return CUR["model"], CUR["tok"]


def hf_ask(model, tok, system, user, max_tokens, temp=0.0, rep_penalty=0.0):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:     # Qwen3.x thinks unless told not to; the answer is the scene itself
        chat = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(chat, return_tensors="pt").to("cuda")
    kw = dict(max_new_tokens=max_tokens, do_sample=temp > 0,
              pad_token_id=tok.eos_token_id)
    if temp > 0:
        kw.update(temperature=temp, top_p=0.95)
    if rep_penalty:
        kw["repetition_penalty"] = rep_penalty
    with torch.no_grad():
        out = model.generate(**ids, **kw)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


import forge.app.pipeline as P  # noqa: E402
P.load, P.ask = hf_load, hf_ask
import scorecard  # noqa: E402

OUT = Path("/kaggle/working/scorecard")
OUT.mkdir(exist_ok=True)
for flag, tag, n, base, trained, api in SPECS:
    t1 = time.time()
    try:
        use(base, trained)
    except Exception as exc:                                  # noqa: BLE001
        print(f"{tag} could not load {base}: {type(exc).__name__}: {exc}", flush=True)
        continue
    sys.argv = ["scorecard.py", flag, "--n", "20", "--oneshot", "--coder",
                str(adapter) if trained else "none", "--samples", str(n), "--tag", tag] \
        + (["--api"] if api else [])
    try:
        scorecard.main()
    except Exception as exc:                                  # noqa: BLE001
        print(f"{tag} FAILED: {type(exc).__name__}: {exc}", flush=True)
    src = ROOT / "data" / "scorecard" / tag
    if src.exists():
        shutil.copytree(src, OUT / tag, dirs_exist_ok=True)
    print(f"{tag} done in {round(time.time() - t1)} s (total {round(time.time() - t0)} s)",
          flush=True)
shutil.rmtree(ROOT, ignore_errors=True)
# One archive: `kaggle kernels output` fetched ~20 files an hour on
# 2026-10-07, so the scorecard dirs also leave as a single tar.
with tarfile.open("/kaggle/working/scorecards.tar.gz", "w:gz") as t:
    for d in sorted(OUT.iterdir()):
        t.add(d, arcname=d.name)
print("all done", round(time.time() - t0), "s", flush=True)
