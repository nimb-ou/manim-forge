#!/bin/bash
# Double-click to start Manim Forge; it opens in your web browser.
# Close this window to stop it.
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
PORT=8765
URL="http://127.0.0.1:$PORT"
if [ ! -x .venv/bin/python ]; then
  echo "Manim Forge is not installed yet. In Terminal, in this folder, run:  ./install.sh"
  read -r -p "Press Return to close."; exit 1
fi
if curl -fs "$URL/api/health" >/dev/null 2>&1; then
  echo "Manim Forge is already running."; open "$URL"; exit 0
fi
echo "Starting Manim Forge…"
.venv/bin/python -m forge.serve --port "$PORT" >"$TMPDIR/manim-forge.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  curl -fs "$URL/api/health" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "It stopped while starting. The log is in $TMPDIR/manim-forge.log"; read -r -p "Press Return to close."; exit 1; }
  sleep 1
done
open "$URL"
echo "Manim Forge is running at $URL — keep this window open while you use it."
echo "Close this window (or press Ctrl-C) to stop it."
wait $SERVER
