#!/usr/bin/env bash
# Install the worker pool as a login agent, so it does not depend on anyone
# remembering to start it.
#
# That dependency is the whole reason this exists. An entire training run
# went by with twenty-five showcase renders outstanding and ten cores idle,
# because starting them was a thing I had to remember rather than a thing the
# machine did.
#
#   ./scripts/forge_service.sh install     # run at login, restart on crash
#   ./scripts/forge_service.sh status
#   ./scripts/forge_service.sh stop        # this boot only
#   ./scripts/forge_service.sh uninstall   # remove entirely
#
# User-level LaunchAgent: no password, no system files, one command to undo.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.manimforge.pool"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

write_plist() {
    mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/logs"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/.venv/bin/python</string>
    <string>-u</string>
    <string>$ROOT/scripts/forge_run.py</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/Library/TeX/texbin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>PYTHONPATH</key><string>$ROOT</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <!-- Restart if it exits non-zero. A clean exit means the pool decided
       there was nothing left to do, and relaunching that on a timer is the
       busy-loop this project already built once. -->
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>$ROOT/data/logs/pool.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/pool.log</string>
  <!-- Renders are not urgent; leave the machine responsive for its owner. -->
  <key>ProcessType</key><string>Background</string>
  <key>Nice</key><integer>5</integer>
</dict>
</plist>
EOF
}

case "${1:-status}" in
  install)
    write_plist
    launchctl bootout "gui/$UID/$LABEL" 2>/dev/null
    launchctl bootstrap "gui/$UID" "$PLIST" && echo "installed: $PLIST"
    sleep 2
    launchctl print "gui/$UID/$LABEL" 2>/dev/null | grep -E "state|pid" | head -2
    echo
    echo "starts at every login, restarts on crash, logs to data/logs/pool.log"
    echo "remove with: ./scripts/forge_service.sh uninstall"
    ;;
  stop)
    launchctl bootout "gui/$UID/$LABEL" 2>/dev/null && echo "stopped until next login" \
      || echo "not loaded"
    ;;
  uninstall)
    launchctl bootout "gui/$UID/$LABEL" 2>/dev/null
    rm -f "$PLIST" && echo "removed"
    ;;
  status)
    if launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1; then
        launchctl print "gui/$UID/$LABEL" | grep -E "^\s+(state|pid) " | sed 's/^/  /'
    else
        echo "  not installed — ./scripts/forge_service.sh install"
    fi
    echo
    "$ROOT/.venv/bin/python" "$ROOT/scripts/forge_run.py" --status
    ;;
  *) echo "usage: $0 {install|status|stop|uninstall}"; exit 1 ;;
esac
