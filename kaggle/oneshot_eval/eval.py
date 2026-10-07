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
import shutil
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

t0 = time.time()
SPECS = [  # (prompt set flag, tag, samples)
    ("--inscope", "is_os1", 1),
    ("--heldout", "held_os1", 1),
    ("--short", "short_os1", 1),
    ("--inscope", "is_os1_n3", 3),
]
BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"


def sh(cmd, t=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    if r.returncode:
        print("FAILED:", cmd[:120], r.stderr[-800:], flush=True)
    return r


sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
   "libcairo2-dev libpango1.0-dev ffmpeg texlive-latex-base texlive-latex-extra "
   "texlive-fonts-recommended dvisvgm > /dev/null")
sh(f"{sys.executable} -m pip install -q manim==0.21.0 transformers==5.17.0 "
   "peft==0.21.0 accelerate==1.15.0 'bitsandbytes>=0.48'")
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

TOK = AutoTokenizer.from_pretrained(BASE)
MODEL = AutoModelForCausalLM.from_pretrained(
    BASE, device_map={"": 0}, dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
MODEL = PeftModel.from_pretrained(MODEL, str(adapter)).eval()
print("model loaded", round(time.time() - t0), "s", flush=True)


def hf_load(_adapter=None):
    return MODEL, TOK


def hf_ask(model, tok, system, user, max_tokens, temp=0.0, rep_penalty=0.0):
    chat = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(chat, return_tensors="pt").to("cuda")
    kw = dict(max_new_tokens=max_tokens, do_sample=temp > 0,
              pad_token_id=tok.eos_token_id)
    if temp > 0:
        kw.update(temperature=temp, top_p=0.95)
    if rep_penalty:
        kw["repetition_penalty"] = rep_penalty
    with torch.no_grad():
        out = model.generate(**ids, **kw)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


import forge.app.pipeline as P  # noqa: E402
P.load, P.ask = hf_load, hf_ask
import scorecard  # noqa: E402

OUT = Path("/kaggle/working/scorecard")
OUT.mkdir(exist_ok=True)
for flag, tag, n in SPECS:
    t1 = time.time()
    sys.argv = ["scorecard.py", flag, "--n", "20", "--oneshot", "--coder", str(adapter),
                "--samples", str(n), "--tag", tag]
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
print("all done", round(time.time() - t0), "s", flush=True)
