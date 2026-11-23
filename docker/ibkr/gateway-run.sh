#!/bin/bash
# IB Gateway under IBC on a virtual display, for the slim image (docker/ibkr/Dockerfile.ibkr): what the base image's
# run.sh did that this container uses, nothing else (no VNC, no socat or ssh forwarding: the bridge talks to the
# gateway's API on 127.0.0.1). ibkr-start.sh runs it as SCRIPT_PATH/run.sh and starts it again when it stops.
set -Eeuo pipefail

: "${TWS_PATH:=/home/ibgateway/Jts}" "${IBC_PATH:=/home/ibgateway/ibc}" "${IBC_INI:=/home/ibgateway/ibc/config.ini}"
: "${IBC_INI_TMPL:=/home/ibgateway/ibc/config.ini.tmpl}" "${TWS_INI:=jts.ini}" "${TWS_INI_TMPL:=jts.ini.tmpl}"
: "${TRADING_MODE:=paper}" "${TIME_ZONE:=${TZ:-Etc/UTC}}" "${TWOFA_TIMEOUT_ACTION:=restart}"
# The gateway's version: its folder (the image keeps the one the base image installed).
if [ -z "${TWS_MAJOR_VRSN:-}" ]; then TWS_MAJOR_VRSN=$(ls "$TWS_PATH/ibgateway" | head -1); fi
export TWS_MAJOR_VRSN IB_GATEWAY_VERSION="${IB_GATEWAY_VERSION:-$TWS_MAJOR_VRSN}"
case "$TRADING_MODE" in paper | live) ;; *) echo ".> invalid TRADING_MODE: $TRADING_MODE"; exit 1 ;; esac
export TRADING_MODE TIME_ZONE

# The password from a file (docker secrets) when TWS_PASSWORD_FILE is set.
if [ -z "${TWS_PASSWORD:-}" ] && [ -n "${TWS_PASSWORD_FILE:-}" ]; then TWS_PASSWORD=$(<"$TWS_PASSWORD_FILE"); fi
export TWS_PASSWORD="${TWS_PASSWORD:-}"
# IBC's settings from the environment (every ${VAR} of the template; an unset one is empty).
for v in ALLOW_BLIND_TRADING AUTO_LOGOFF_TIME AUTO_RESTART_TIME BYPASS_WARNING EXISTING_SESSION_DETECTED_ACTION READ_ONLY_API \
  RELOGIN_AFTER_TWOFA_TIMEOUT SAVE_TWS_SETTINGS TWOFA_DEVICE TWOFA_EXIT_INTERVAL TWS_ACCEPT_INCOMING TWS_COLD_RESTART \
  TWS_MASTER_CLIENT_ID TWS_USERID; do export "$v=${!v:-}"; done
umask 077
envsubst <"$IBC_INI_TMPL" >"$IBC_INI"
unset TWS_PASSWORD
umask 022
settings="${TWS_SETTINGS_PATH:-$TWS_PATH}"
mkdir -p "$settings"
[ -f "$settings/$TWS_INI" ] || envsubst <"$TWS_PATH/$TWS_INI_TMPL" >"$settings/$TWS_INI"

# The display: no GLX (the gateway's JavaFX draws in software, -Dprism.order=sw), so no OpenGL stack is loaded.
export DISPLAY=:1
rm -f /tmp/.X1-lock
mkdir -p /tmp/.X11-unix
Xvfb "$DISPLAY" -ac -screen 0 1024x768x16 -extension GLX -nolisten tcp &
xvfb=$!
for _ in $(seq 1 50); do [ -S /tmp/.X11-unix/X1 ] && break; sleep 0.2; done
[ -S /tmp/.X11-unix/X1 ] || { echo ".> Xvfb did not start"; exit 1; }

if [ -n "${JAVA_HEAP_SIZE:-}" ]; then
  sed -i "s/^-Xmx[0-9]*m$/-Xmx${JAVA_HEAP_SIZE}m/" "$TWS_PATH"/ibgateway/*/ibgateway.vmoptions
fi

echo ".> Starting IBC ($TRADING_MODE)"
"$IBC_PATH/scripts/ibcstart.sh" "$TWS_MAJOR_VRSN" -g "--tws-path=$TWS_PATH" "--ibc-path=$IBC_PATH" "--ibc-ini=$IBC_INI" \
  "--on2fatimeout=$TWOFA_TIMEOUT_ACTION" "--tws-settings-path=${TWS_SETTINGS_PATH:-}" &
ibc=$!
trap 'kill -TERM "$ibc" 2>/dev/null; wait "$ibc"; kill "$xvfb" 2>/dev/null; exit 0' TERM INT
set +e
wait "$ibc"
code=$?
kill "$xvfb" 2>/dev/null
exit "$code"
