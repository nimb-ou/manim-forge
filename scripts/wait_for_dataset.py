"""Wait until a Kaggle dataset version is actually mounted-ready.

Run 4 died here. The workflow uploaded a new dataset version and pushed the
kernel five seconds later; Kaggle was still processing, so the kernel mounted
an incomplete dataset and `load_dataset` could not find train.jsonl -- a file
the API's own listing showed as present.

A version moves through NOT_YET_PERSISTED -> BLOBS_RECEIVED ->
BLOBS_DECOMPRESSED -> BLOBS_COPIED_TO_SDS -> INDIVIDUAL_BLOBS_COMPRESSED ->
READY, and only the last of those is safe to attach to a kernel.

Status alone is not enough, though. It says the *version* is processed, not
that the version contains what you uploaded -- so this also checks that the
files the kernel needs are listed. Both, because the failure this is
preventing looked exactly like success from every angle that was checked.

    python scripts/wait_for_dataset.py nimbou/manim-forge-data \\
        --expect train.jsonl valid.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time

TERMINAL_BAD = {"failed", "deleted"}


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
    ap.add_argument("ref")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="files the kernel will read; all must be listed")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--interval", type=int, default=15)
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    deadline = time.time() + a.timeout
    last = None
    while time.time() < deadline:
        try:
            status = str(api.dataset_status(a.ref)).strip().lower()
        except Exception as exc:                       # transient API error
            status = f"unreadable ({type(exc).__name__})"
        if status != last:
            print(f"  [{int(time.time() - (deadline - a.timeout)):>4}s] {status}",
                  flush=True)
            last = status

        if status in TERMINAL_BAD:
            sys.exit(f"dataset version is {status}")

        if status == "ready":
            listed = files_of(api, a.ref)
            missing = [f for f in a.expect if f not in listed]
            if not missing:
                print(f"  ready — {len(listed)} files listed")
                return
            # READY but incomplete: keep waiting rather than push a kernel
            # at a dataset that cannot serve it.
            print(f"  ready but missing {missing}; still waiting", flush=True)
            last = None

        time.sleep(a.interval)

    sys.exit(f"dataset not ready after {a.timeout}s (last status: {last})")


if __name__ == "__main__":
    main()
