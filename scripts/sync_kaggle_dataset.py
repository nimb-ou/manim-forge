"""Put the training mix on Kaggle, and do not lie about when it is usable.

Two failures live here, one behind the other.

**The race.** `kaggle datasets version` returns as soon as the bytes are
accepted; the version then processes asynchronously. Run 5 pushed a kernel
four seconds later and the dataset did not mount *at all* -- not partially,
absent. The wait I added to prevent exactly this returned in one second,
because `dataset_status` and the file listing were still describing the
*previous* version. Asking "is the dataset ready" is the wrong question when
a new version is the thing you are waiting for; the right one is "has the
version number advanced, and is *that* version ready".

**The churn.** None of it needs to happen when the data has not changed. The
mix only moves when the corpus or the gold scenes do, so this fingerprints
the files and skips the upload entirely otherwise -- which removes the race
from most runs rather than racing more carefully.

    python scripts/sync_kaggle_dataset.py kaggle/manim-forge-data \\
        --ref nimbou/manim-forge-data --expect train.jsonl valid.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

FINGERPRINT = ".fingerprint.json"


def fingerprint(folder: Path) -> str:
    """Hash of every file's name and content, excluding the marker itself."""
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.name == FINGERPRINT:
            continue
        h.update(p.relative_to(folder).as_posix().encode())
        h.update(str(p.stat().st_size).encode())
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def remote_fingerprint(api, ref: str) -> str | None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        try:
            api.dataset_download_file(ref, FINGERPRINT, path=tmp)
        except Exception:
            return None
        for f in Path(tmp).rglob(FINGERPRINT + "*"):
            try:
                return json.loads(f.read_text())["fingerprint"]
            except Exception:
                return None
    return None


def version_of(api, ref: str) -> int:
    for d in api.dataset_list(mine=True, search=ref.split("/")[-1]):
        if d.ref == ref:
            return int(getattr(d, "current_version_number", 0) or 0)
    return 0


def files_of(api, ref: str) -> set[str]:
    names, token = set(), None
    for _ in range(40):
        r = api.dataset_list_files(ref, page_token=token, page_size=200)
        names |= {f.name for f in r.files}
        token = (getattr(r, "nextPageToken", None)
                 or getattr(r, "next_page_token", None))
        if not token:
            break
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--expect", nargs="*", default=[])
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    local = fingerprint(a.folder)
    (a.folder / FINGERPRINT).write_text(
        json.dumps({"fingerprint": local, "at": time.strftime("%FT%T")}) + "\n")
    remote = remote_fingerprint(api, a.ref)
    print(f"local  fingerprint: {local}")
    print(f"remote fingerprint: {remote or '(none)'}")

    before = version_of(api, a.ref)
    if remote == local and not a.force:
        missing = [f for f in a.expect if f not in files_of(api, a.ref)]
        if not missing:
            print(f"unchanged — version {before} already has what the kernel "
                  f"needs; not uploading")
            return
        print(f"fingerprint matches but {missing} are not listed; re-uploading")

    print(f"uploading (current version {before}) …", flush=True)
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(a.folder), "-r", "zip",
         "-m", f"mix {local} {time.strftime('%F %T')}"],
        capture_output=True, text=True)
    print((r.stdout or "").strip()[-500:])
    if r.returncode != 0:
        sys.exit((r.stderr or "").strip()[-800:] or "upload failed")

    deadline, last = time.time() + a.timeout, None
    while time.time() < deadline:
        time.sleep(a.interval)
        now = version_of(api, a.ref)
        try:
            status = str(api.dataset_status(a.ref)).strip().lower()
        except Exception as exc:
            status = f"unreadable ({type(exc).__name__})"
        listed = files_of(api, a.ref) if status == "ready" else set()
        missing = [f for f in a.expect if f not in listed]
        # The only check that cannot be stale: download a file from the
        # dataset and see whether the *new* content is what comes back.
        # Version numbers advance the moment Kaggle accepts the bytes, and
        # both dataset_status and the file listing kept describing the
        # previous version while the new one processed -- which is how a
        # one-second "wait" declared success on a dataset that then failed
        # to mount at all.
        served = remote_fingerprint(api, a.ref) if status == "ready" else None
        state = (f"v{now} {status} missing={missing or 'none'} "
                 f"served={served or '-'}")
        if state != last:
            print(f"  {state}", flush=True)
            last = state
        if status in ("failed", "deleted"):
            sys.exit(f"dataset version is {status}")
        if now > before and status == "ready" and not missing and served == local:
            print(f"  version {now} is serving {local} — mountable")
            return

    sys.exit(f"dataset not mountable after {a.timeout}s (last: {last})")


if __name__ == "__main__":
    main()
