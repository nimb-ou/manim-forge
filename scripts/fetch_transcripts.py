"""Fetch 3Blue1Brown transcripts and segment them into narration beats.

Subtitles only — no video is ever downloaded. Resumable: a video whose .vtt is
already on disk is skipped, so this can be stopped and restarted freely.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from forge.style.transcripts import parse_vtt, segment

CHANNEL = "https://www.youtube.com/@3blue1brown/videos"


def list_videos(limit: int, ytdlp: str = "./.venv/bin/yt-dlp") -> list[dict]:
    """Video ids and titles, without touching the media streams."""
    r = subprocess.run(
        [ytdlp, "--flat-playlist", "--dump-json", "--playlist-end", str(limit), CHANNEL],
        capture_output=True, text=True, timeout=600,
    )
    out = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("id"):
            out.append({"id": d["id"], "title": d.get("title", "")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--subs-dir", default="data/style/subs")
    ap.add_argument("--out", default="data/style/narration.jsonl")
    a = ap.parse_args()

    subs = Path(a.subs_dir)
    subs.mkdir(parents=True, exist_ok=True)

    print(f"listing up to {a.limit} videos from the channel ...", flush=True)
    videos = list_videos(a.limit)
    print(f"found {len(videos)}\n", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["video_id"])
            except (json.JSONDecodeError, KeyError):
                pass

    total_segs = 0
    with out.open("a") as sink:
        for i, v in enumerate(videos, 1):
            if v["id"] in done:
                print(f"  [{i:>3}/{len(videos)}] skip (have) {v['title'][:52]}", flush=True)
                continue
            existing = list(subs.glob(f"{v['id']}*.vtt"))
            if not existing:
                subprocess.run(
                    ["./.venv/bin/yt-dlp", "--skip-download", "--write-sub",
                     "--write-auto-sub", "--sub-lang", "en", "--sub-format", "vtt",
                     "--output", str(subs / "%(id)s.%(ext)s"),
                     f"https://www.youtube.com/watch?v={v['id']}"],
                    capture_output=True, timeout=300,
                )
                existing = list(subs.glob(f"{v['id']}*.vtt"))
            if not existing:
                print(f"  [{i:>3}/{len(videos)}] NO SUBS {v['title'][:52]}", flush=True)
                continue

            segs = segment(parse_vtt(existing[0]))
            for j, s in enumerate(segs):
                sink.write(json.dumps({
                    "video_id": v["id"], "title": v["title"], "index": j,
                    "start": s["start"], "end": s["end"], "text": s["text"],
                }) + "\n")
            sink.flush()
            total_segs += len(segs)
            print(f"  [{i:>3}/{len(videos)}] {len(segs):>3} segments  {v['title'][:52]}", flush=True)
            time.sleep(0.6)

    print(f"\n{'='*58}\n  {total_segs} new narration segments -> {out}\n{'='*58}")


if __name__ == "__main__":
    main()
