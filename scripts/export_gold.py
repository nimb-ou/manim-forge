"""Collect hand-authored gold scenes into training rows.

Gold scenes are the style anchor: few, deliberate, and weighted heavily at
training time. Each carries a natural-language prompt written the way a person
would actually ask for it, because that prompt is what the model learns to map
from.
"""
import json
from pathlib import Path

GOLD = [
    {
        "module": "forge/gold/rook_corners.py",
        "scene": "RookCorners",
        "prompt": "build a chess grid, add rooks on it and animate the rook's "
                  "possible movements, then calculate the probability that a "
                  "randomly placed rook is on a corner square",
        "tags": ["discrete", "probability", "grid"],
    },
    {
        "module": "forge/gold/gradient_descent.py",
        "scene": "LearningRate",
        "prompt": "explain how the learning rate affects gradient descent — "
                  "show a ball rolling down a loss curve with a good rate, "
                  "then one that is too large and overshoots, then one that is "
                  "too small and never arrives",
        "tags": ["machine-learning", "optimisation", "plot"],
    },
    {
        "module": "forge/gold/sphere_cube.py",
        "scene": "SphereInCube",
        "prompt": "show a sphere in 3D built out of dots, fit the largest cube "
                  "that can fit inside it, and calculate how much volume is "
                  "left over between the cube and the sphere",
        "tags": ["shapes-3d", "geometry", "volume"],
    },
    {
        "module": "forge/gold/riemann.py",
        "scene": "RiemannConvergence",
        "prompt": "show how Riemann sums converge to the area under a curve — "
                  "start with a couple of rectangles, keep doubling them, and "
                  "show the total approaching the exact integral, then show "
                  "that sampling the midpoint converges much faster",
        "tags": ["calculus", "convergence", "plot"],
    },
    {
        "module": "forge/gold/spring.py",
        "scene": "SpringTrace",
        "prompt": "show a mass on a spring oscillating, trace its position "
                  "over time to reveal a sine wave, show the energy moving "
                  "between kinetic and potential while the total stays "
                  "constant, and show how mass and stiffness change the period",
        "tags": ["physics", "oscillation", "energy"],
    },
    {
        "module": "forge/gold/dot_product.py",
        "scene": "DotProduct",
        "prompt": "explain the dot product as a projection — show two vectors, "
                  "drop a perpendicular so one casts a shadow on the other, and "
                  "show that the shadow times the length gives the dot product, "
                  "then swing past ninety degrees so the sign goes negative",
        "tags": ["vectors", "geometry", "projection"],
    },
]

out = Path("data/gold/gold.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for i, g in enumerate(GOLD):
        code = Path(g["module"]).read_text()
        f.write(json.dumps({
            "prompt": g["prompt"],
            "code": code,
            "meta": {"id": f"gold:{i:03d}", "source": "gold",
                     "scene": g["scene"], "tags": g["tags"],
                     "n_play_calls": code.count("self.play(")},
        }) + "\n")
print(f"{len(GOLD)} gold scenes -> {out}")
for g in GOLD:
    c = Path(g['module']).read_text()
    print(f"  {g['scene']:<16} {c.count('@beat')} beats, {c.count('self.play(')} play calls")
