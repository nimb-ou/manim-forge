# ── Manim Forge · SFT on Kaggle ────────────────────────────────────────────
# Paste into a Kaggle notebook. Settings -> Accelerator: GPU T4 x2,
# Internet: ON. Attach the "manim-forge-data" dataset.
#
# Kaggle gives 30 GPU-hours a week free and runs notebooks as root, so LaTeX
# and ffmpeg install cleanly. That second part matters more than the GPU:
# GRPO's reward *is* a render, so training and verification must live in the
# same machine. This notebook does SFT; the GRPO one reuses its output.

# ── 1. environment ─────────────────────────────────────────────────────────
!pip -q install -U "transformers>=4.45" "trl>=0.12" "peft>=0.13" \
                   "bitsandbytes>=0.44" "accelerate>=1.0" datasets

import json, os, tarfile, textwrap
from pathlib import Path

DATA = Path("/kaggle/input/manim-forge-data")
WORK = Path("/kaggle/working")

# The same forge package that runs locally, so a number measured here means
# the same thing as a number measured on the laptop.
with tarfile.open(DATA / "forge.tar.gz") as t:
    t.extractall(WORK)
os.chdir(WORK)

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
from peft import LoraConfig

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 4-bit so a 7B fits a single T4 with room for activations.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
model.config.use_cache = False

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

cfg = SFTConfig(
    output_dir=str(WORK / "sft-out"),
    num_train_epochs=2,               # more overfits ~1.8k examples
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,               # LoRA tolerates far more than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=60,
    save_steps=120,
    save_total_limit=3,               # Kaggle output quota is finite
    bf16=False, fp16=True,
    max_seq_length=2048,
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
trainer.train()
trainer.save_model(str(WORK / "adapter"))
print("adapter saved to /kaggle/working/adapter")

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

# Download /kaggle/working/adapter, then locally:
#   ./.venv/bin/python scripts/run_repair_benchmark.py --n 100 --retrieval \
#       --adapter adapters/kaggle-sft --tag tuned100
# It has to beat the untuned baseline or it has not earned its place.
