#!/usr/bin/env bash
# Install a Kaggle API token from wherever it actually is.
#
# Kaggle's "Create New Token" downloads a kaggle.json rather than copying
# anything, but browsers put it in different places and some people copy the
# JSON by hand instead. This looks in all of them, validates what it finds,
# and never prints the key.
set -uo pipefail

DEST="$HOME/.kaggle/kaggle.json"

install_from() {
    python3 - "$1" <<'PY'
import json, os, sys, pathlib
raw = pathlib.Path(sys.argv[1]).read_text()
try:
    d = json.loads(raw)
except Exception:
    sys.exit(2)
if not (isinstance(d, dict) and d.get("username") and d.get("key")):
    sys.exit(2)
dest = pathlib.Path.home() / ".kaggle" / "kaggle.json"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(d))
dest.chmod(0o600)
print(d["username"])
PY
}

# an explicit path wins over any search
if [ "$#" -ge 1 ] && [ -f "$1" ]; then
    if user=$(install_from "$1"); then
        echo "installed from $1"
        echo "Kaggle account: $user"
        exit 0
    fi
    echo "$1 is not a valid Kaggle token (needs username and key)"
    exit 1
fi

if [ -f "$DEST" ]; then
    user=$(python3 -c "import json,pathlib;print(json.loads(pathlib.Path('$DEST').read_text())['username'])" 2>/dev/null)
    if [ -n "${user:-}" ]; then
        chmod 600 "$DEST"
        echo "already installed — Kaggle account: $user"
        exit 0
    fi
    echo "$DEST exists but is not a valid token; replacing"
fi

# 1. anywhere a browser might have put it
found=$(find "$HOME/Downloads" "$HOME/Desktop" "$HOME/Documents" "$HOME" \
        -maxdepth 2 -name 'kaggle*.json' 2>/dev/null | head -1)
if [ -n "$found" ]; then
    if user=$(install_from "$found"); then
        echo "installed from $found"
        echo "Kaggle account: $user"
        exit 0
    fi
fi

# 2. the clipboard, in case the JSON was copied by hand
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
pbpaste > "$tmp" 2>/dev/null || true
if user=$(install_from "$tmp"); then
    echo "installed from the clipboard"
    echo "Kaggle account: $user"
    exit 0
fi

# 3. type it in. Kaggle's newer settings page shows the token as a string to
#    copy rather than downloading a file, and a key pasted into a chat window
#    has to be treated as burnt -- so read it here, with echo off, and write
#    the file locally.
if [ "${1:-}" = "--manual" ] || [ -t 0 ]; then
    echo
    echo "No kaggle.json found. Entering it directly instead."
    echo "Nothing you type here is echoed or logged."
    echo
    printf '  Kaggle username: '
    read -r KU
    printf '  Kaggle key (hidden): '
    read -rs KK
    echo
    if [ -n "$KU" ] && [ -n "$KK" ]; then
        printf '{"username":"%s","key":"%s"}' "$KU" "$KK" > "$tmp"
        if user=$(install_from "$tmp"); then
            echo
            echo "installed to $DEST"
            echo "Kaggle account: $user"
            exit 0
        fi
        echo "  that did not parse as a valid token"
    fi
fi

cat <<'MSG'
No token found.

The download is the reliable path — "Create New Token" saves a file, it does
not copy anything to the clipboard:

  1. open  https://www.kaggle.com/settings
  2. scroll to  API
  3. click  Create New Token      (a kaggle.json downloads)
  4. re-run this script

If your browser asked where to save it and you chose somewhere unusual, drag
the file into the terminal after the script name:

  ./scripts/kaggle_token.sh /path/to/kaggle.json
MSG
exit 1
