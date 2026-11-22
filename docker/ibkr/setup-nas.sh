#!/bin/sh
# Asks the questions for the broker bridge on a NAS or any Linux box with Docker (the IBKR login, the
# store's token, the amount a trade, the automatic markets, HTTPS), keeps the answers in <dir>/.env and
# starts the one container (IB Gateway, IBC and the Java bridge: docker/ibkr/ibkr-nas.sh), so the page
# (https://svr-net.github.io/trading/) can use it from your other devices.
#
#   sudo sh deploy-nas.sh     (clones the repository, then runs this)
#
# HTTPS: lan (a certificate made for the NAS's address; trust it once on each device) or synology (DSM's
# own certificate for your name, renewed by DSM in place).
#
# Re-running keeps the answers in <dir>/.env (delete it to answer again). Every answer can come from the
# environment instead (IBKR_DIR, TWS_USERID, TWS_PASSWORD, TRADING_MODE, EOD_TOKEN, CAPITAL, TZ,
# BRIDGE_KEY, HTTPS_MODE=lan|synology, LAN_HOST, HTTPS_PORT, BRIDGE_DOMAIN, CERT_DIR, REPO_REF), e.g. for
# a script without questions. REPO_DIR: the clone deploy-nas.sh keeps; SKIP_START=1 writes the answers
# without starting anything.
set -eu

REPO_REF=${REPO_REF:-main}
RAW="https://raw.githubusercontent.com/svr-net/trading/$REPO_REF"
DIR=${IBKR_DIR:-$(pwd)/ibkr-bridge}

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
ask() {  # ask VAR "question" "default": keeps an existing value
  eval "cur=\${$1:-}"
  if [ -n "$cur" ]; then return; fi
  printf '%s%s: ' "$2" "${3:+ [$3]}"
  read -r ans || true
  eval "$1=\${ans:-\$3}"
}
ask_secret() {  # ask_secret VAR "question": not echoed
  eval "cur=\${$1:-}"
  if [ -n "$cur" ]; then return; fi
  printf '%s: ' "$2"
  stty -echo 2>/dev/null || true
  read -r ans || true
  stty echo 2>/dev/null || true
  printf '\n'
  eval "$1=\$ans"
}
fetch() {  # fetch URL FILE
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"; else wget -qO "$2" "$1"; fi
}

# --- Docker ------------------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "Docker is not installed (Synology: Container Manager; QNAP: Container Station; Unraid/TrueNAS: built in)."
docker info >/dev/null 2>&1 || die "Docker is not running, or this user may not use it (try: sudo sh setup-nas.sh)."
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || die "curl or wget is needed."
case "$(uname -m)" in x86_64|amd64|aarch64|arm64) ;; *) die "this processor ($(uname -m)) is not supported by the IB Gateway image (x86-64 or 64-bit ARM needed)." ;; esac

mkdir -p "$DIR" "$DIR/history"
cd "$DIR"
say "Setting up in $DIR"

# --- The answers (kept in .env) -------------------------------------------------------------------
if [ -f .env ]; then set -a; . ./.env; set +a; say "Using the answers in $DIR/.env"; fi
say ""
say "Interactive Brokers (the paper account's login: the free trial's username and password)"
ask TWS_USERID "  username"
ask_secret TWS_PASSWORD "  password (not shown)"
ask TRADING_MODE "  paper or live" "paper"
say "GitHub: the EOD_TOKEN (the daily job's token: repository svr-net/eod, Contents read and write)"
[ -n "${EOD_TOKEN:-}" ] || EOD_TOKEN=${EOD_READ_TOKEN:-}
ask_secret EOD_TOKEN "  token (not shown)"
ask CAPITAL "Amount a trade (account currency)" "5000"
ask AUTO_MARKETS "Place and close automatically each weekday (UK, EU, US, APAC, INDEX, comma separated, or none)" "UK"
# INDEX (index ETFs: since 13 Nov 2026 UCITS ETFs on Xetra, on Frankfurt's clock, as IBKR lets an EU account buy no
# US-listed ETF) came beside the US: a setup that trades the US and was answered before INDEX existed gets it too,
# saved in .env below (NO_AUTO_INDEX=1 keeps it out).
case ",$(echo "$AUTO_MARKETS" | tr -d ' ' | tr a-z A-Z)," in
  *,INDEX,*) ;;
  *,US,*) if [ "${NO_AUTO_INDEX:-}" != 1 ]; then AUTO_MARKETS="$AUTO_MARKETS,INDEX"; say "AUTO_MARKETS: INDEX added beside the US ($AUTO_MARKETS; NO_AUTO_INDEX=1 to keep it out)"; fi ;;
esac
true
ask TZ "Time zone" "Europe/London"
if [ -z "${BRIDGE_KEY:-}" ]; then
  if command -v openssl >/dev/null 2>&1; then BRIDGE_KEY=$(openssl rand -hex 24)
  else BRIDGE_KEY=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n'); fi
fi
say ""
say "HTTPS for the bridge:"
say "  lan    : private, on your network only (trust its certificate once on each device)"
say "  synology : your DSM certificate (e.g. name.synology.me); the router forwards one port to this NAS"
ask HTTPS_MODE "  lan or synology" "lan"
case "$HTTPS_MODE" in
  lan)
    guess=$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1)
    [ -n "$guess" ] || guess=$(hostname -I 2>/dev/null | awk '{print $1}')
    ask LAN_HOST "  the NAS's address on your network" "$guess"
    ask HTTPS_PORT "  HTTPS port (not 80 or 443, nor one the NAS already uses)" "8765"
    HTTP_PORT=""; BRIDGE_DOMAIN=""
    SITE="https://$LAN_HOST:$HTTPS_PORT"
    ;;
  synology)
    ask BRIDGE_DOMAIN "  the certificate's name (e.g. name.synology.me)"
    ask HTTPS_PORT "  HTTPS port (router: the same outside port to this NAS; not 80 or 443)" "8765"
    if [ -z "${CERT_DIR:-}" ]; then  # the DSM certificate holding the name (Control Panel, Security, Certificate)
      for d in /usr/syno/etc/certificate/_archive/*/; do
        [ -f "${d}fullchain.pem" ] && [ -f "${d}privkey.pem" ] || continue
        if openssl x509 -in "${d}cert.pem" -noout -text 2>/dev/null | grep -q "DNS:$BRIDGE_DOMAIN"; then CERT_DIR=${d%/}; break; fi
      done
    fi
    [ -n "${CERT_DIR:-}" ] && [ -f "$CERT_DIR/fullchain.pem" ] || die "no DSM certificate for $BRIDGE_DOMAIN found (set CERT_DIR to its folder in /usr/syno/etc/certificate/_archive)."
    say "  certificate: $CERT_DIR"
    HTTP_PORT=""; LAN_HOST=""
    SITE="https://$BRIDGE_DOMAIN:$HTTPS_PORT"
    ;;
  *) die "HTTPS_MODE must be lan or synology (public, with Caddy, is gone with the four containers)" ;;
esac
[ -n "$TWS_USERID" ] && [ -n "$TWS_PASSWORD" ] && [ -n "$EOD_TOKEN" ] || die "the username, password and token are needed."

umask 077
cat > .env <<EOF
TWS_USERID=$TWS_USERID
TWS_PASSWORD=$TWS_PASSWORD
TRADING_MODE=$TRADING_MODE
EOD_TOKEN=$EOD_TOKEN
CAPITAL=$CAPITAL
AUTO_MARKETS=$AUTO_MARKETS
TZ=$TZ
BRIDGE_KEY=$BRIDGE_KEY
HTTPS_MODE=$HTTPS_MODE
LAN_HOST=$LAN_HOST
HTTPS_PORT=$HTTPS_PORT
HTTP_PORT=$HTTP_PORT
BRIDGE_DOMAIN=$BRIDGE_DOMAIN
CERT_DIR=${CERT_DIR:-}
EOF
umask 022

# --- The one container -------------------------------------------------------------------------
if [ "${SKIP_START:-}" = 1 ]; then say "Answers written in $DIR/.env (SKIP_START=1: nothing started)."; exit 0; fi
script=""
[ -n "${REPO_DIR:-}" ] && [ -f "$REPO_DIR/docker/ibkr/ibkr-nas.sh" ] && script="$REPO_DIR/docker/ibkr/ibkr-nas.sh"
if [ -z "$script" ]; then
  script="$DIR/ibkr-nas.sh"
  fetch "$RAW/docker/ibkr/ibkr-nas.sh" "$script" || die "could not download ibkr-nas.sh"
fi
say ""
IBKR_DIR="$DIR" sh "$script"
say ""
say "Your key:      $BRIDGE_KEY   (also in $DIR/.env)"
say "Then open (Chrome or Edge):  https://svr-net.github.io/trading/?bridge=$SITE"
say "In 'Your broker' enter the key and Save; 'Check (dry run)' lists the book's orders."
