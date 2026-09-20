#!/usr/bin/env bash
# Fast check on one scene: does it render at all, and what does it look like?
#
#   ./scripts/verify_scene.sh forge/gold/coordinate_plane.py CoordinatePlane
#
# 480p15 and no caching, so it is a correctness check rather than a preview.
# Pulls four evenly-spaced stills out at the end, which is where composition
# errors show up -- a label over a figure renders perfectly happily.
#
# Deliberately a separate file from queue.sh: editing a shell script while
# bash is executing it can corrupt the running job.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/Library/TeX/texbin:$PATH"
export PYTHONUNBUFFERED=1

FILE="${1:?usage: verify_scene.sh <file.py> <SceneName>}"
NAME="${2:?usage: verify_scene.sh <file.py> <SceneName>}"
OUT="data/checks/$NAME"
mkdir -p "$OUT"

echo "checking $NAME ..."
S=$(date +%s)
./.venv/bin/manim render -ql --disable_caching --media_dir "$OUT" \
    "$FILE" "$NAME" < /dev/null 2>&1 | tail -20
RC=${PIPESTATUS[0]}
D=$(( $(date +%s) - S ))

V=$(find "$OUT" -name "$NAME.mp4" -not -path "*partial*" | head -1)
if [[ $RC -ne 0 || -z "$V" ]]; then
  echo "FAILED rc=$RC after ${D}s"; exit 1
fi

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$V")
printf "rendered in %ds   runtime %.1fs   %s\n" "$D" "$DUR" "$V"

# four stills, skipping the very start and end where things are still fading
for i in 1 2 3 4; do
  T=$(./.venv/bin/python -c "print(f'{$DUR * (0.18 + 0.22*($i-1)):.2f}')")
  ffmpeg -y -loglevel error -ss "$T" -i "$V" -vframes 1 -vf scale=620:-1 \
         "$OUT/still_$i.png"
done
ffmpeg -y -loglevel error \
  -i "$OUT/still_1.png" -i "$OUT/still_2.png" \
  -i "$OUT/still_3.png" -i "$OUT/still_4.png" \
  -filter_complex "[0:v][1:v]hstack[a];[2:v][3:v]hstack[b];[a][b]vstack" \
  "$OUT/contact.png"
echo "contact sheet: $OUT/contact.png"
