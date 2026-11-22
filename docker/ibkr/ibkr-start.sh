#!/bin/bash
# One container (docker/ibkr/Dockerfile.ibkr): IB Gateway, logged in by IBC (the image's own run.sh), and
# the broker bridge (tools/ibkr/bridge.py) beside it. Each is started again when it stops: IB Gateway when
# IBC exits (its nightly restart, a failed login, or the bridge's watchdog stopping a hung gateway with
# GATEWAY_RESTART=local), the bridge when it exits. docker stop ends both.
set -u

# Started as root only to read the certificate (DSM's files are root's): copied for the bridge to
# /run/tls (owner 1000), again whenever they change (checked hourly), then everything else runs as
# the image's user (1000).
if [ "$(id -u)" = 0 ]; then
  if [ -n "${TLS_SRC_CERT:-}" ] && [ -n "${TLS_SRC_KEY:-}" ]; then
    copy_tls() {
      mkdir -p /run/tls
      cp "$TLS_SRC_CERT" /run/tls/cert.pem.new && cp "$TLS_SRC_KEY" /run/tls/key.pem.new || return 1
      chown 1000:1000 /run/tls /run/tls/*.new && chmod 600 /run/tls/key.pem.new
      mv /run/tls/key.pem.new /run/tls/key.pem && mv /run/tls/cert.pem.new /run/tls/cert.pem
    }
    copy_tls || { echo "ibkr-start: cannot read the certificate $TLS_SRC_CERT"; exit 1; }
    (
      while sleep 3600; do
        cmp -s "$TLS_SRC_CERT" /run/tls/cert.pem && cmp -s "$TLS_SRC_KEY" /run/tls/key.pem || { copy_tls && echo "ibkr-start: renewed certificate copied"; }
      done
    ) &
    export TLS_CERT=/run/tls/cert.pem TLS_KEY=/run/tls/key.pem
  fi
  chown -R 1000:1000 /data/history 2>/dev/null || true
  exec setpriv --reuid=1000 --regid=1000 --init-groups env HOME=/home/ibgateway USER=ibgateway "$0" "$@"
fi

stop_all() {
  echo "ibkr-start: stopping"
  kill "$GATEWAY_LOOP" "$BRIDGE_LOOP" 2>/dev/null
  pkill -TERM -f ibcalpha.ibc 2>/dev/null
  pkill -TERM -f bridge.py 2>/dev/null
  wait
  exit 0
}
trap stop_all TERM INT

(
  while true; do
    "${SCRIPT_PATH:-/home/ibgateway/scripts}/run.sh"
    echo "ibkr-start: IB Gateway stopped (exit $?); starting it again in 10 seconds"
    # run.sh starts its display, VNC and port forwarding afresh: the old ones go first
    pkill x11vnc 2>/dev/null; pkill Xvfb 2>/dev/null; pkill -f run_socat.sh 2>/dev/null; pkill socat 2>/dev/null
    sleep 10
  done
) &
GATEWAY_LOOP=$!

# The bridge reaches IB Gateway inside the container: its own API port (paper 4002, live 4001).
if [ -z "${IB_PORT:-}" ]; then
  if [ "${TRADING_MODE:-paper}" = live ]; then IB_PORT=4001; else IB_PORT=4002; fi
fi
export IB_PORT

(
  cd /app || exit 1
  while true; do
    /opt/bridge/bin/python tools/ibkr/bridge.py
    echo "ibkr-start: the bridge stopped (exit $?); starting it again in 5 seconds"
    sleep 5
  done
) &
BRIDGE_LOOP=$!

wait
