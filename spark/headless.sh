#!/usr/bin/env bash
# Keep the Spark on the network with nobody at it. Unlike the rest of spark/, this needs root:
#
#   sudo spark/headless.sh install     # or uninstall
#   spark/headless.sh status
#
# The Spark is on Wi-Fi, and NetworkManager can leave the Wi-Fi off until someone types its
# password at the login screen; spark/wifiwatch.sh says when. After a restart on 2026-10-16
# that kept the Spark off the network for 16 hours. install runs frank-wifiwatch as root
# every minute, which connects the Wi-Fi again with the saved password when that happens.
# A cable to the router would avoid the question, since a wired connection needs no password.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WATCH=/usr/local/sbin/frank-wifiwatch
UNIT=/etc/systemd/system/frank-wifiwatch.service
TIMER=/etc/systemd/system/frank-wifiwatch.timer

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need_root() { [ "$(id -u)" = 0 ] || die "run it with sudo"; }

case "${1:-status}" in
    install)
        need_root
        command -v nmcli >/dev/null || die "no nmcli; this is for NetworkManager"
        # A copy owned by root: the service must not run a file the user account can change.
        install -m 0755 -o root -g root "$ROOT/spark/wifiwatch.sh" "$WATCH"
        cat > "$UNIT" <<EOF
[Unit]
Description=Connect the Wi-Fi again when it is left waiting for a password (frank)
After=NetworkManager.service

[Service]
Type=oneshot
ExecStart=$WATCH
EOF
        cat > "$TIMER" <<'EOF'
[Unit]
Description=Check the Wi-Fi every minute (frank)

[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
AccuracySec=5s

[Install]
WantedBy=timers.target
EOF
        systemctl daemon-reload
        systemctl enable --now frank-wifiwatch.timer
        "$0" status
        ;;
    uninstall)
        need_root
        systemctl disable --now frank-wifiwatch.timer 2>/dev/null || true
        rm -f "$WATCH" "$UNIT" "$TIMER" /run/frank-wifiwatch
        systemctl daemon-reload
        echo "removed"
        ;;
    status)
        printf '%-20s %s\n' frank-wifiwatch "$(systemctl is-active frank-wifiwatch.timer 2>/dev/null || true)"
        nmcli -t -f DEVICE,TYPE,STATE device 2>/dev/null | awk -F: '$2 == "wifi" || $2 == "ethernet" { printf "%-20s %s\n", $1, $3 }'
        journalctl -u frank-wifiwatch.service -n 4 --no-pager -o short-iso 2>/dev/null | grep 'frank-wifiwatch:' || true
        ;;
    *) die "usage: headless.sh install|uninstall|status" ;;
esac
