"""Narration: each beat's line spoken, and the beat held until it is said.

A 3Blue1Brown video is a voice over a picture; the scenes here have always
carried a narration line per beat and rendered silent. Kokoro-82M (via
mlx-audio, local, ~1 s per line on this Mac) speaks each line; the kit's
``stage.mark(min_seconds=...)`` pads a beat that would end before its line
does; and ffmpeg lays each clip at its beat's start.

Everything here is optional: without mlx-audio (or misaki) the video simply
stays silent, and the pipeline says so in its notes.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

VOICE_MODEL = "mlx-community/Kokoro-82M-bf16"
VOICE = "af_heart"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
GAP = 0.35            # seconds of quiet after a line before the beat may end


@lru_cache(maxsize=1)
def _model():
    from mlx_audio.tts.utils import load_model
    return load_model(VOICE_MODEL)


def speak(lines: list[str], out_dir: Path, voice: str = VOICE,
          speed: float = 1.0) -> list[tuple[Path | None, float]]:
    """One wav per non-empty line, as (path, seconds); (None, 0) for blanks."""
    import numpy as np
    from mlx_audio.audio_io import write as audio_write
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _model()
    out: list[tuple[Path | None, float]] = []
    for i, line in enumerate(lines):
        line = " ".join((line or "").split())
        if not line:
            out.append((None, 0.0))
            continue
        chunks = [np.array(r.audio) for r in
                  model.generate(text=line, voice=voice, speed=speed, lang_code="a")]
        if not chunks:
            out.append((None, 0.0))
            continue
        audio = np.concatenate(chunks)
        path = out_dir / f"beat{i + 1:02d}.wav"
        audio_write(str(path), audio, model.sample_rate, format="wav")
        out.append((path, len(audio) / model.sample_rate))
    return out


def mux(video: str, clips: list[tuple[Path, float]], out: Path) -> bool:
    """Lay each (wav, start seconds) over the video. True on success."""
    clips = [(p, t) for p, t in clips if p is not None]
    if not clips:
        return False
    inputs = sum([["-i", str(p)] for p, _ in clips], [])
    delays = ";".join(f"[{k + 1}:a]adelay={int(t * 1000)}:all=1[a{k}]"
                      for k, (_, t) in enumerate(clips))
    mix = "".join(f"[a{k}]" for k in range(len(clips)))
    # Speech loudness for a video (-16 LUFS); Kokoro's raw level is ~-26 dB.
    graph = (f"{delays};{mix}amix=inputs={len(clips)}:normalize=0[m];"
             f"[m]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", video, *inputs,
                        "-filter_complex", graph, "-map", "0:v", "-map", "[aout]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(out)],
                       capture_output=True, text=True)
    return r.returncode == 0 and out.exists()
