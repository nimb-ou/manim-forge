"""Round-2 self-training on a Kaggle GPU: the new kit coder writes many arcs
at once (batched, one beat of every arc per call), the session's CPUs
render and salvage them, and each kept scene's beat-end frames are saved
for the Mac's vision critic. The logic is forge/app/selfgen.py, the same
module a local run uses; this file only loads the model.

Inputs  dataset nimbou/manim-forge-selfgen: forge/ and arcs.jsonl
        kernel  nimbou/manim-forge-kit-sft: the kit coder adapter (adapter/)
Output  /kaggle/working/selfgen/scenes.jsonl, frames/*.jpg
"""
import json, os, subprocess, sys, time
from pathlib import Path



def main():
    global tok, model
    T0 = time.time()
    HOURS = 7.0          # stop writing after this; renders finish before the 9 h cap
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    subprocess.run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                   "libcairo2-dev libpango1.0-dev ffmpeg texlive-latex-base texlive-latex-extra "
                   "texlive-fonts-recommended dvisvgm > /dev/null", shell=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "manim==0.21.0",
                    "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
                    "bitsandbytes>=0.48"], check=True)
    print("setup", round(time.time() - T0), "s", flush=True)

    root = Path("/kaggle/input")
    arcs_f = next(root.rglob("arcs.jsonl"))
    sys.path.insert(0, str(next(p for p in arcs_f.parent.rglob("forge") if (p / "app").is_dir()).parent))
    # The SFT output also holds checkpoint folders with their own adapter_config;
    # the final adapter is the one in a folder named "adapter".
    adapter = next(p.parent for p in sorted(root.rglob("adapter_config.json"))
                   if p.parent.name == "adapter")
    print("arcs", arcs_f, "adapter", adapter, flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, dtype=torch.float16,
                                                 attn_implementation="sdpa", device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    print("model loaded", round(time.time() - T0), "s", flush=True)


    class HFGenerator:
        """Batched generation: greedy prompts and sampled prompts in separate calls."""

        def __init__(self, bs: int = 24):
            self.bs = bs

        def _run(self, chats, temp, max_tokens):
            out = []
            for s in range(0, len(chats), self.bs):
                enc = tok(chats[s: s + self.bs], return_tensors="pt", padding=True).to(0)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=max_tokens, do_sample=temp > 0,
                                       temperature=temp if temp > 0 else None,
                                       top_p=0.95 if temp > 0 else None,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
                out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return out

        def batch(self, system, users, temps, max_tokens):
            chats = [tok.apply_chat_template([{"role": "system", "content": system},
                                              {"role": "user", "content": u}],
                                             add_generation_prompt=True, tokenize=False)
                     for u in users]
            res = [None] * len(users)
            for t in sorted(set(temps)):
                idx = [i for i, x in enumerate(temps) if x == t]
                for i, o in zip(idx, self._run([chats[i] for i in idx], t, max_tokens)):
                    res[i] = o
            return res


    from forge.app.pipeline import CODE_SYSTEM_KIT_TRAINED
    from forge.app.selfgen import run
    from forge.app.twostage import Beat

    arcs = []
    for l in arcs_f.open():
        r = json.loads(l)
        arcs.append((r["id"], r["request"], [Beat(*b) for b in r["beats"]]))
    print(len(arcs), "arcs", flush=True)
    run(HFGenerator(), arcs, Path("/kaggle/working/selfgen"), CODE_SYSTEM_KIT_TRAINED,
        sys.executable, "kit-v8", chunk=48, workers=max(2, (os.cpu_count() or 4)),
        deadline=T0 + HOURS * 3600)
    print("done", round(time.time() - T0), "s", flush=True)


if __name__ == "__main__":
    main()
