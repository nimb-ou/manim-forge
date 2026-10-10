#!/bin/bash
# Manim Forge installer for an Apple Silicon Mac. Run it from this folder:
#     ./install.sh
# It installs the drawing tools (Homebrew, LaTeX, ffmpeg), a private Python
# environment in .venv, downloads the AI model (~6 GB) and checks it all.
# It is safe to run again; it only adds what is missing.
set -e
cd "$(dirname "$0")"
say() { printf "\n\033[1m%s\033[0m\n" "$*"; }

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "Manim Forge needs a Mac with Apple Silicon (M1 or later)."; exit 1
fi
mem=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$mem" -ge 15 ] || echo "Note: this Mac has ${mem} GB of memory; 16 GB is recommended and it may be slow."
free=$(df -g . | tail -1 | awk '{print $4}')
[ "$free" -ge 12 ] || { echo "About 12 GB of free disk space is needed (${free} GB free)."; exit 1; }

if ! command -v brew >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/brew ]; then
  say "Homebrew (the Mac's package manager) is needed. Installing it — it will ask for your password."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1

say "Drawing tools"
brew install cairo pango pkg-config ffmpeg python@3.12
if [ ! -x /Library/TeX/texbin/latex ]; then
  say "LaTeX, for maths on screen (it will ask for your password)"
  brew install --cask basictex
fi
export PATH="/Library/TeX/texbin:$PATH"

say "Python environment"
PY="$(brew --prefix python@3.12)/bin/python3.12"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements-app.txt

say "Downloading the AI model and checking everything"
.venv/bin/python -m forge.serve.setup
chmod +x "Start Manim Forge.command"
say "Done. Double-click “Start Manim Forge” in this folder whenever you want to use it."
