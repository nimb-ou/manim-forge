"""Kaggle CPU shard of the vision critic: render assembled kit scenes and cut
the frame at the end of every beat. No GPU quota; the Mac judges the frames.

Input  /kaggle/input/manim-forge-critic/scenes.jsonl  {i, scene, cls, code, live}
Output /kaggle/working/frames/<i>_<k>.jpg and manifest.jsonl {i, scene, live, ok}
SHARD / NSHARDS are written into this file by scripts/kaggle_critic.py.
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

SHARD, NSHARDS = 0, 1
t0 = time.time()
def sh(cmd, t=3600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)

sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
   "libcairo2-dev libpango1.0-dev ffmpeg texlive-latex-base texlive-latex-extra "
   "texlive-fonts-recommended dvisvgm > /dev/null")
sh(f"{sys.executable} -m pip install -q manim==0.21.0")
print("setup", round(time.time() - t0), "s", flush=True)

src = next(p for p in ("/kaggle/input/manim-forge-critic/scenes.jsonl",
                       "/kaggle/input/datasets/nimbou/manim-forge-critic/scenes.jsonl")
           if os.path.exists(p))
rows = [json.loads(l) for l in open(src)]
rows = [r for r in rows if r["i"] % NSHARDS == SHARD]
os.makedirs("/kaggle/working/frames", exist_ok=True)
man = open("/kaggle/working/manifest.jsonl", "w")


def one(r):
    d = f"/tmp/r{r['i']}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/scene.py", "w").write(r["code"])
    try:
        p = subprocess.run([sys.executable, "-m", "manim", "-ql", "--disable_caching",
                            "--media_dir", f"{d}/media", f"{d}/scene.py", r["cls"]],
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"i": r["i"], "scene": r["scene"], "ok": False}
    m = re.search(r"BEAT_ENDS \[([^\]]*)\]", p.stdout or "")
    ends = [float(x) for x in m.group(1).split(",")] if m and m.group(1) else []
    vids = [os.path.join(a, f) for a, _, fs in os.walk(f"{d}/media") for f in fs
            if f.endswith(".mp4") and "partial" not in a]
    if p.returncode or not vids or len(ends) != len(r["live"]):
        return {"i": r["i"], "scene": r["scene"], "ok": False}
    for k, t in zip(r["live"], ends):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(t - 0.25, 0):.2f}",
                        "-i", vids[0], "-frames:v", "1", "-vf", "scale=640:-1",
                        f"/kaggle/working/frames/{r['i']}_{k}.jpg"])
    sh(f"rm -rf {d}")
    ok = all(os.path.exists(f"/kaggle/working/frames/{r['i']}_{k}.jpg") for k in r["live"])
    return {"i": r["i"], "scene": r["scene"], "live": r["live"], "ok": ok}


with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
    for n, res in enumerate(pool.map(one, rows), 1):
        man.write(json.dumps(res) + "\n"); man.flush()
        if n % 20 == 0:
            print(n, "of", len(rows), round(time.time() - t0), "s", flush=True)
print("done", len(rows), round(time.time() - t0), "s", flush=True)
