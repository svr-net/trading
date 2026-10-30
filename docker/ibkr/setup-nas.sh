#!/bin/sh
# Sets up IB Gateway (paper by default) and the page's bridge on a NAS or any Linux box with Docker,
# behind HTTPS so the page (https://svr-net.github.io/trading/) can use it from your other devices.
#
#   curl -fsSL https://raw.githubusercontent.com/svr-net/trading/main/docker/ibkr/setup-nas.sh -o setup-nas.sh
#   sh setup-nas.sh
#
# Two ways to HTTPS:
#   A (lan)    private: Caddy's own certificate authority for the NAS's address on your network;
#              nothing is exposed to the internet; install Caddy's root certificate once on each
#              device you open the page on (the script saves it and prints how).
#   B (public) a domain or DuckDNS name with Let's Encrypt; the router forwards two ports to the NAS.
#
# Re-running keeps the answers in <dir>/.env (delete it to answer again) and updates the files.
# Every answer can come from the environment instead (IBKR_DIR, TWS_USERID, TWS_PASSWORD,
# TRADING_MODE, EOD_READ_TOKEN, CAPITAL, TZ, BRIDGE_KEY, HTTPS_MODE=lan|public, LAN_HOST,
# HTTPS_PORT, BRIDGE_DOMAIN, HTTP_PORT, REPO_REF), e.g. for a script without questions.
# REPO_DIR: a clone of the repository to build from (deploy-nas.sh) instead of downloading the files;
# SKIP_START=1 writes the files without starting anything.
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

# --- Docker, compose, downloads ------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "Docker is not installed (Synology: Container Manager; QNAP: Container Station; Unraid/TrueNAS: built in)."
docker info >/dev/null 2>&1 || die "Docker is not running, or this user may not use it (try: sudo sh setup-nas.sh)."
if docker compose version >/dev/null 2>&1; then COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then COMPOSE="docker-compose"
else die "Docker Compose is missing (docker compose or docker-compose)."; fi
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || die "curl or wget is needed."
case "$(uname -m)" in x86_64|amd64|aarch64|arm64) ;; *) die "this processor ($(uname -m)) is not supported by the IB Gateway image (x86-64 or 64-bit ARM needed)." ;; esac

mkdir -p "$DIR"
cd "$DIR"
say "Setting up in $DIR"
if [ -n "${REPO_DIR:-}" ]; then
  # Built from the clone: its docker/ibkr/Dockerfile.bridge and tools/ibkr.
  [ -f "$REPO_DIR/docker/ibkr/Dockerfile.bridge" ] || die "no repository clone at $REPO_DIR"
  BUILD_CONTEXT=$REPO_DIR; BUILD_FILE=docker/ibkr/Dockerfile.bridge
else
  BUILD_CONTEXT=.; BUILD_FILE=Dockerfile.bridge
  mkdir -p tools/ibkr
  for f in tools/ibkr/bridge.py tools/ibkr/submit.py; do fetch "$RAW/$f" "$f"; done
  cat > Dockerfile.bridge <<'EOF'
FROM python:3.12-slim
RUN pip install --no-cache-dir ib_async
WORKDIR /app
COPY tools/ibkr/ tools/ibkr/
EXPOSE 8765
CMD ["python", "tools/ibkr/bridge.py"]
EOF
fi

# --- The answers (kept in .env) -------------------------------------------------------------------
if [ -f .env ]; then set -a; . ./.env; set +a; say "Using the answers in $DIR/.env"; fi
say ""
say "Interactive Brokers (the paper account's login: the free trial's username and password)"
ask TWS_USERID "  username"
ask_secret TWS_PASSWORD "  password (not shown)"
ask TRADING_MODE "  paper or live" "paper"
say "GitHub: a fine-grained token, repository svr-net/eod only, Contents read-only"
ask_secret EOD_READ_TOKEN "  token (not shown)"
ask CAPITAL "Amount a trade (account currency)" "5000"
ask TZ "Time zone" "Europe/London"
if [ -z "${BRIDGE_KEY:-}" ]; then
  if command -v openssl >/dev/null 2>&1; then BRIDGE_KEY=$(openssl rand -hex 24)
  else BRIDGE_KEY=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n'); fi
fi
say ""
say "HTTPS for the bridge:"
say "  lan    : private, on your network only (install a root certificate once on each device)"
say "  public : a domain or DuckDNS name; the router forwards two ports to this NAS"
ask HTTPS_MODE "  lan or public" "lan"
case "$HTTPS_MODE" in
  lan)
    guess=$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1)
    [ -n "$guess" ] || guess=$(hostname -I 2>/dev/null | awk '{print $1}')
    ask LAN_HOST "  the NAS's address on your network" "$guess"
    ask HTTPS_PORT "  HTTPS port (not one the NAS already uses)" "8765"
    HTTP_PORT=""; BRIDGE_DOMAIN=""
    SITE="https://$LAN_HOST:$HTTPS_PORT"
    ;;
  public)
    ask BRIDGE_DOMAIN "  domain (its DNS must point to your home's address)"
    ask HTTP_PORT "  NAS port for HTTP (router: outside 80 to this)" "8080"
    ask HTTPS_PORT "  NAS port for HTTPS (router: outside 443 to this)" "8443"
    LAN_HOST=""
    SITE="https://$BRIDGE_DOMAIN"
    ;;
  *) die "HTTPS_MODE must be lan or public" ;;
esac
[ -n "$TWS_USERID" ] && [ -n "$TWS_PASSWORD" ] && [ -n "$EOD_READ_TOKEN" ] || die "the username, password and token are needed."

umask 077
cat > .env <<EOF
TWS_USERID=$TWS_USERID
TWS_PASSWORD=$TWS_PASSWORD
TRADING_MODE=$TRADING_MODE
EOD_READ_TOKEN=$EOD_READ_TOKEN
CAPITAL=$CAPITAL
TZ=$TZ
BRIDGE_KEY=$BRIDGE_KEY
HTTPS_MODE=$HTTPS_MODE
LAN_HOST=$LAN_HOST
HTTPS_PORT=$HTTPS_PORT
HTTP_PORT=$HTTP_PORT
BRIDGE_DOMAIN=$BRIDGE_DOMAIN
EOF
umask 022

# --- Caddy and the services ----------------------------------------------------------------------
if [ "$HTTPS_MODE" = lan ]; then
  cat > Caddyfile <<EOF
{
	skip_install_trust
	default_sni $LAN_HOST
}
https://$LAN_HOST:$HTTPS_PORT {
	tls internal
	reverse_proxy bridge:8765
}
EOF
  CADDY_PORTS="      - \"$HTTPS_PORT:$HTTPS_PORT\""
else
  cat > Caddyfile <<EOF
$BRIDGE_DOMAIN {
	reverse_proxy bridge:8765
	header Strict-Transport-Security "max-age=31536000"
}
EOF
  CADDY_PORTS="      - \"$HTTP_PORT:80\"
      - \"$HTTPS_PORT:443\""
fi
cat > docker-compose.yml <<EOF
# Written by setup-nas.sh (run it again to change; the answers are in .env).
services:
  ib-gateway:
    image: ghcr.io/gnzsnz/ib-gateway:stable
    restart: always
    environment:
      TWS_USERID: \${TWS_USERID}
      TWS_PASSWORD: \${TWS_PASSWORD}
      TRADING_MODE: \${TRADING_MODE}
      READ_ONLY_API: "no"
      TWOFA_TIMEOUT_ACTION: restart
      AUTO_RESTART_TIME: "11:59 PM"
      TZ: \${TZ}
  bridge:
    build:
      context: $BUILD_CONTEXT
      dockerfile: $BUILD_FILE
    restart: always
    depends_on: [ib-gateway]
    environment:
      BRIDGE_KEY: \${BRIDGE_KEY}
      EOD_READ_TOKEN: \${EOD_READ_TOKEN}
      CAPITAL: \${CAPITAL}
      IB_HOST: ib-gateway
      IB_PORT: "${IB_API_PORT:-$([ "$TRADING_MODE" = live ] && echo 4003 || echo 4004)}"
      BIND: 0.0.0.0
      ALLOWED_ORIGIN: https://svr-net.github.io
  caddy:
    image: caddy:2.8
    restart: always
    depends_on: [bridge]
    ports:
$CADDY_PORTS
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
      - caddy-config:/config
volumes:
  caddy-data:
  caddy-config:
EOF

if [ "${SKIP_START:-}" = 1 ]; then say "Files written in $DIR (SKIP_START=1: nothing started)."; exit 0; fi
say ""
say "Starting (the first time downloads the images: a few minutes)..."
$COMPOSE up -d --build

# --- What to do next -----------------------------------------------------------------------------
PAGE="https://svr-net.github.io/trading/?bridge=$SITE"
if [ "$HTTPS_MODE" = lan ]; then
  i=0
  until $COMPOSE exec -T caddy test -f /data/caddy/pki/authorities/local/root.crt 2>/dev/null || [ $i -ge 30 ]; do
    i=$((i + 1)); sleep 2
    command -v curl >/dev/null 2>&1 && curl -sk -o /dev/null "$SITE/status" || true
  done
  $COMPOSE exec -T caddy cat /data/caddy/pki/authorities/local/root.crt > caddy-root.crt 2>/dev/null || true
fi
say ""
say "Done. IB Gateway logs in within a minute or two: $COMPOSE -f $DIR/docker-compose.yml logs -f ib-gateway"
say ""
say "Your bridge:   $SITE"
say "Your key:      $BRIDGE_KEY   (also in $DIR/.env)"
if [ "$HTTPS_MODE" = lan ]; then
  say ""
  say "Trust the bridge's certificate once on each device you open the page on:"
  say "  the file: $DIR/caddy-root.crt (copy it to the device)"
  say "  Windows: double-click it, Install Certificate, Local Machine, Trusted Root Certification Authorities"
  say "  macOS:   open it in Keychain Access (System), then Get Info, Trust, Always Trust"
  say "  iPhone/iPad: open it, install the profile; then Settings, General, About, Certificate Trust Settings, turn it on"
  say "  Android: Settings, Security, Encryption & credentials, Install a certificate, CA certificate"
  say "  then restart the browser."
else
  say ""
  say "Router: forward outside port 80 to this NAS port $HTTP_PORT, and 443 to $HTTPS_PORT;"
  say "the certificate is obtained once the domain points here: $COMPOSE -f $DIR/docker-compose.yml logs -f caddy"
fi
say ""
say "Then open (Chrome or Edge):  $PAGE"
say "In 'Your broker' enter the key and Save; 'Check (dry run)' lists the next open's orders."
