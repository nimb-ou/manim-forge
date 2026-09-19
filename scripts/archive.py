"""Package everything worth keeping into a single dated archive.

Datasets are declared in forge.catalog with an `archive` flag, so what gets
backed up is decided once, where the dataset is documented, rather than in a
shell glob that drifts.

Renders are excluded — they are large and fully regenerable from the code and
the seeds. Everything else is kept, failures included: a broken generation with
its traceback and its fix is the rarest training data the project produces, and
losing it would cost far more than the disk it occupies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="archive")
    ap.add_argument("--include-renders", action="store_true")
    a = ap.parse_args()

    from forge.catalog import CATALOG

    stamp = time.strftime("%Y%m%d-%H%M")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tarball = out / f"manim-forge-data-{stamp}.tar.gz"

    included, missing = [], []
    with tarfile.open(tarball, "w:gz") as tar:
        for d in CATALOG:
            if not d.archive and not a.include_renders:
                continue
            p = Path(d.path)
            if not p.exists():
                missing.append(d.path)
                continue
            tar.add(p, arcname=p.as_posix(),
                    filter=lambda t: None if "__pycache__" in t.name else t)
            included.append(d.path)
        # The code that produced it all, so the archive is self-describing.
        for extra in ("forge", "scripts", "kaggle", "README.md", "LICENSE",
                      "pyproject.toml", "requirements.txt"):
            if Path(extra).exists():
                tar.add(extra, arcname=extra,
                        filter=lambda t: None if "__pycache__" in t.name else t)

    size_mb = tarball.stat().st_size / 1e6
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()[:16]
    manifest = {
        "created": stamp, "file": tarball.name,
        "size_mb": round(size_mb, 1), "sha256_16": digest,
        "datasets": included, "missing": missing,
        "renders_included": bool(a.include_renders),
    }
    (out / f"manifest-{stamp}.json").write_text(json.dumps(manifest, indent=2))

    print(f"archive : {tarball}")
    print(f"size    : {size_mb:.1f} MB")
    print(f"sha256  : {digest}")
    print(f"datasets: {len(included)} included, {len(missing)} missing")
    if missing:
        print(f"          missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
