# GRPO on the coder: reward = the assembled scene renders.
#
# Pushed by scripts/push_kernel.py-style `kaggle kernels push -p kaggle/grpo`.
# Dataset nimbou/manim-forge-grpo carries prompts.jsonl (beats on top of the
# coder's own earlier beats, from real two-stage runs), twostage.py (the same
# assemble() inference uses) and coder/ (the SFT adapter to start from).
#
# MF_SMOKE=1 runs three steps to prove the path before spending the session.
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

T0 = time.time()
SMOKE = os.environ.get("MF_SMOKE") == "1"
BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
WORK = Path("/kaggle/working")
HOURS = 8.0


def sh(cmd, t=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=t)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    return r.returncode


# -- renderer (probed: 66 s apt, 42 s pip, 2.6 s a render) -------------------
assert sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get "
          "install -y -qq libcairo2-dev libpango1.0-dev ffmpeg "
          "texlive-latex-base texlive-latex-extra texlive-fonts-recommended "
          "dvisvgm > /dev/null") == 0
# The SFT kernel's pins, which it verified: an unpinned trl moved GRPO's
# config (max_prompt_length is gone in 1.13), and bitsandbytes is not in
# Kaggle's image -- the first push died at model load without it.
assert sh(f"{sys.executable} -m pip install -q manim==0.21.0 "
          "transformers==5.17.0 trl==1.13.0 peft==0.21.0 accelerate==1.15.0 "
          "datasets==5.0.1 'bitsandbytes>=0.48' huggingface_hub") == 0
print(f"[setup] {time.time() - T0:.0f}s", flush=True)

DATA = next(Path("/kaggle/input").glob("*/prompts.jsonl")).parent
sys.path.insert(0, str(DATA))
# The dataset carries forge/app/twostage.py and forge/kit/kit.py in the
# repo's layout, so assemble(kit=True) finds the kit where it looks for it.
from forge.app.twostage import Beat, assemble, extract_code  # noqa: E402

rows = [json.loads(l) for l in (DATA / "prompts.jsonl").open() if l.strip()]
print(f"[data] {len(rows)} prompts from {DATA}", flush=True)


def render_ok(code: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "scene.py"
        f.write_text(code)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "manim", "-ql", "--disable_caching",
                 "--media_dir", d, str(f), "ForgeScene"],
                capture_output=True, text=True, timeout=90, cwd=d)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0 and any(Path(d).rglob("ForgeScene.mp4"))


import ast as _ast  # noqa: E402

_TEXT = {"Text", "MathTex", "Tex", "Title", "MarkupText", "Paragraph",
         "BulletedList", "Code", "Integer", "DecimalNumber", "Variable"}
_MOTION = {"Transform", "ReplacementTransform", "TransformMatchingTex",
           "TransformMatchingShapes", "Rotate", "Rotating", "MoveAlongPath",
           "ApplyMatrix", "ApplyPointwiseFunction", "Homotopy",
           "ValueTracker", "always_redraw", "MoveToTarget"}
_MOBJECT = None
# Kit blocks draw and animate inside the kit, so a kit beat never contains
# "self.play(" or a Mobject constructor; these count instead.
from forge.kit.kit import KIT_BLOCKS as _KIT_DRAWS  # noqa: E402
from forge.kit.kit import KIT_MOVES as _KIT_MOVES  # noqa: E402


def _shape_and_motion(code: str) -> tuple[bool, bool]:
    """Does the beat build a non-text mobject, and does anything move?

    The first hard-eval renders were illustrated slides -- one scene had 27
    text objects and no geometry -- and a reward that is only "it renders"
    would teach more of them. The animation gate scores whole scenes, and
    on a single beat it gives `self.wait(1)` more than a real vector beat,
    so this is a small beat-level bonus instead.
    """
    global _MOBJECT
    import manim
    if _MOBJECT is None:
        _MOBJECT = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
                    and issubclass(getattr(manim, n), manim.Mobject)}
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False, False
    calls = {n.func.id for n in _ast.walk(tree)
             if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
    shape = bool((calls & _MOBJECT) - _TEXT) or ".plot(" in code \
        or bool(calls & _KIT_DRAWS)
    motion = motion_calls(calls, code)
    return shape, motion


def motion_calls(calls, code):
    return bool(calls & _MOTION) or ".animate" in code or bool(calls & _KIT_MOVES)


def score(completion: str, row: dict) -> float:
    """Renders: 0.6, +0.2 for a non-text mobject, +0.2 if something moves.
    0.2 assembles but fails at runtime; 0.1 names nothing defines; 0.0 does
    not parse. A beat with no self.play is capped at 0.3 -- `self.wait(1)`
    always renders and teaches nothing."""
    code = extract_code(completion)
    if not code.strip():
        return 0.0
    beats = [Beat(n=k + 1, seconds=None, intent=t)
             for k, t in enumerate(row["intents"])]
    kit = bool(row.get("kit"))
    asm = assemble(beats, list(row["prefix"]) + [code], kit=kit)
    if asm.problems:
        return 0.1 if any("names no beat" in p for p in asm.problems) else 0.0
    if not render_ok(asm.code):
        return 0.2
    shape, motion = _shape_and_motion(code)
    if "self.play(" not in code and not (kit and (shape or motion)):
        return 0.3
    return 0.6 + 0.2 * shape + 0.2 * motion


POOL = ThreadPoolExecutor(max_workers=4)
BY_ID = {r["id"]: r for r in rows}


def render_reward(prompts, completions, id, **kw):          # noqa: A002
    texts = [c[-1]["content"] if isinstance(c, list) else c
             for c in completions]
    return list(POOL.map(lambda p: score(p[0], BY_ID[p[1]]),
                         zip(texts, id)))


# -- model ---------------------------------------------------------------------
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_storage=torch.float16)
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, dtype=torch.float16,
    attn_implementation="sdpa", device_map={"": 0})
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
model = PeftModel.from_pretrained(model, str(DATA / "coder"),
                                  is_trainable=True)


def fp32_trainables(m, where):
    n = 0
    for _, q in m.named_parameters():
        if q.requires_grad and q.dtype is not torch.float32:
            q.data = q.data.to(torch.float32)
            n += 1
    print(f"[dtype] {where}: recast {n}", flush=True)


fp32_trainables(model, "after load")

ds = Dataset.from_list([{"prompt": r["prompt"], "id": r["id"]} for r in rows])
ds = ds.shuffle(seed=17)


class Budget(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if model is not None:
            fp32_trainables(model, "on_train_begin")

    def on_step_end(self, args, state, control, **kw):
        if time.time() - T0 > HOURS * 3600:
            print("[budget] stopping", flush=True)
            control.should_training_stop = True
            control.should_save = True


cfg = GRPOConfig(
    output_dir=str(WORK / "grpo-out"),
    num_generations=4, per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    # Completions ran a mean of 74 tokens and at most 118 over 87 steps; a
    # 512 cap reserved memory for nothing and the run died of OOM at step 87.
    max_completion_length=256,
    temperature=0.8, learning_rate=5e-6, beta=0.04,
    max_steps=3 if SMOKE else 400, logging_steps=1,
    save_steps=25, save_total_limit=2, fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to=[], remove_unused_columns=False,
)
trainer = GRPOTrainer(model=model, args=cfg, reward_funcs=[render_reward],
                      train_dataset=ds, processing_class=tok,
                      callbacks=[Budget()])
fp32_trainables(trainer.model, "after trainer construction")
trainer.train()
trainer.save_model(str(WORK / "grpo-adapter"))
log = [e for e in trainer.state.log_history if "reward" in e]
(WORK / "rewards.json").write_text(json.dumps(log, indent=1))
print(f"[done] {len(log)} logged steps, first reward "
      f"{log[0]['reward'] if log else '-'}, last "
      f"{log[-1]['reward'] if log else '-'}; {time.time() - T0:.0f}s",
      flush=True)
