#!/bin/bash
# One container (docker/ibkr/Dockerfile.ibkr): IB Gateway, logged in by IBC (the image's own run.sh), and the Java
# bridge (HTTPS for the page; IBKR's Java API; IBC's command server; the automatic day). Each is started again when
# it stops: IB Gateway when IBC exits (its nightly restart, a failed login, or a restart IBC did not do itself), the
# bridge when it exits.
# docker stop ends them all.
set -u

# Started as root only to read the certificate (DSM's files are root's): made into the Java bridge's PKCS#12
# file in /run/tls (owner 1000), again whenever the files change (checked hourly); then everything else runs as
# the image's user (1000).
if [ "$(id -u)" = 0 ]; then
  if [ -n "${TLS_SRC_CERT:-}" ] && [ -n "${TLS_SRC_KEY:-}" ]; then
    make_tls() {
      mkdir -p /run/tls
      openssl pkcs12 -export -in "$TLS_SRC_CERT" -inkey "$TLS_SRC_KEY" -name bridge -passout pass:bridge -out /run/tls/bridge.p12.new || return 1
      cp "$TLS_SRC_CERT" /run/tls/source-cert.pem && cp "$TLS_SRC_KEY" /run/tls/source-key.pem
      chown -R 1000:1000 /run/tls && chmod 600 /run/tls/*
      mv /run/tls/bridge.p12.new /run/tls/bridge.p12
    }
    make_tls || { echo "ibkr-start: cannot read the certificate $TLS_SRC_CERT"; exit 1; }
    (
      while sleep 3600; do
        cmp -s "$TLS_SRC_CERT" /run/tls/source-cert.pem && cmp -s "$TLS_SRC_KEY" /run/tls/source-key.pem || { make_tls && echo "ibkr-start: renewed certificate loaded"; }
      done
    ) &
    export TLS_P12=/run/tls/bridge.p12 TLS_P12_PASSWORD=bridge
  fi
  chown -R 1000:1000 /data/history 2>/dev/null || true
  exec setpriv --reuid=1000 --regid=1000 --init-groups env HOME=/home/ibgateway USER=ibgateway "$0" "$@"
fi

stop_all() {
  echo "ibkr-start: stopping"
  kill "$GATEWAY_LOOP" "$JAVA_LOOP" 2>/dev/null
  pkill -TERM -f ibcalpha.ibc 2>/dev/null
  pkill -TERM -f bridge.jar 2>/dev/null
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

# The bridge: the page's HTTPS on 8765 (without a certificate: plain HTTP on 127.0.0.1 only).
(
  cd /app || exit 1
  while true; do
    PORT=8765 BIND=0.0.0.0 /app/java -Xmx384m -jar /app/bridge.jar
    echo "ibkr-start: the bridge stopped (exit $?); starting it again in 5 seconds"
    sleep 5
  done
) &
JAVA_LOOP=$!

wait
