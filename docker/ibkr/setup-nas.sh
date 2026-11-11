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
# TRADING_MODE, EOD_TOKEN, CAPITAL, TZ, BRIDGE_KEY, HTTPS_MODE=lan|public, LAN_HOST,
# HTTPS_PORT, BRIDGE_DOMAIN, HTTP_PORT, REPO_REF), e.g. for a script without questions.
# REPO_DIR: the clone deploy-nas.sh keeps (only this script comes from it: the images are GitHub's);
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

mkdir -p "$DIR" "$DIR/history"   # history: the bind mount must exist before the bridge starts
cd "$DIR"
say "Setting up in $DIR"
# The images are built by GitHub (.github/workflows/bridge-image.yml), never here: the bridge's
# (ghcr.io/svr-net/trading-bridge, stamped with its commit) and, for the Synology setup, Caddy with
# openssl (ghcr.io/svr-net/trading-caddy). Each is pulled from GitHub's registry (at most 3 minutes),
# else downloaded as one file from the repository's "nas-images" release over IPv4 (a NAS whose
# Docker cannot reach the registry, e.g. over IPv6) and loaded. Neither works: stop, build nothing.
ARCH=$(uname -m); case "$ARCH" in x86_64|amd64) ARCH=amd64 ;; *) ARCH=arm64 ;; esac
pull_image() {
  if command -v timeout >/dev/null 2>&1; then timeout 180 docker pull -q "$1" >/dev/null 2>&1
  else docker pull -q "$1" >/dev/null 2>&1 & p=$!; i=0
    while kill -0 $p 2>/dev/null && [ $i -lt 180 ]; do sleep 1; i=$((i + 1)); done
    if kill -0 $p 2>/dev/null; then kill $p 2>/dev/null; return 1; fi
    wait $p
  fi
}
get_image() {  # get_image NAME (trading-bridge or trading-caddy): ghcr.io/svr-net/NAME:latest here
  img=ghcr.io/svr-net/$1:latest
  say "Getting $img (at most 3 minutes from the registry)..."
  if pull_image "$img"; then return; fi
  url=https://github.com/svr-net/trading/releases/download/nas-images/$1-$ARCH.tar.gz
  say "  the registry did not answer: downloading $url over IPv4"
  if command -v curl >/dev/null 2>&1; then
    curl -4 -fsSL --retry 3 --connect-timeout 20 --max-time 900 "$url" -o "/tmp/$1.tar.gz" || die "could not download $url (nothing is built on the NAS)"
  else
    wget -4 -q -T 60 -O "/tmp/$1.tar.gz" "$url" || die "could not download $url (nothing is built on the NAS)"
  fi
  gunzip -c "/tmp/$1.tar.gz" | docker load -q >/dev/null || die "could not load $1 (nothing is built on the NAS)"
  rm -f "/tmp/$1.tar.gz"
  docker image inspect "$img" >/dev/null 2>&1 || die "$img is not there after loading"
}
get_image trading-bridge
BRIDGE_SRC="    image: ghcr.io/svr-net/trading-bridge:latest"

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
true
ask TZ "Time zone" "Europe/London"
if [ -z "${BRIDGE_KEY:-}" ]; then
  if command -v openssl >/dev/null 2>&1; then BRIDGE_KEY=$(openssl rand -hex 24)
  else BRIDGE_KEY=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n'); fi
fi
say ""
say "HTTPS for the bridge:"
say "  lan    : private, on your network only (install a root certificate once on each device)"
say "  public : a domain or DuckDNS name; the router forwards two ports to this NAS"
say "  synology : your DSM certificate (e.g. name.synology.me); the router forwards one port to this NAS"
ask HTTPS_MODE "  lan, public or synology" "lan"
case "$HTTPS_MODE" in
  lan)
    guess=$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1)
    [ -n "$guess" ] || guess=$(hostname -I 2>/dev/null | awk '{print $1}')
    ask LAN_HOST "  the NAS's address on your network" "$guess"
    ask HTTPS_PORT "  HTTPS port (not 80 or 443, nor one the NAS already uses)" "8765"
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
  *) die "HTTPS_MODE must be lan, public or synology" ;;
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

# --- Caddy and the services ----------------------------------------------------------------------
if [ "$HTTPS_MODE" = lan ]; then
  cat > Caddyfile <<EOF
{
	skip_install_trust
	default_sni $LAN_HOST
	auto_https disable_redirects
	http_port 8080
	https_port $HTTPS_PORT
}
https://$LAN_HOST:$HTTPS_PORT {
	tls internal
	reverse_proxy bridge:8765
}
EOF
  CADDY_PORTS="      - \"$HTTPS_PORT:$HTTPS_PORT\""
elif [ "$HTTPS_MODE" = synology ]; then
  # DSM's own certificate (renewed by DSM in place; the folder is mounted, so a restart picks it up).
  cat > Caddyfile <<EOF
{
	default_sni $BRIDGE_DOMAIN
	auto_https disable_redirects
	http_port 8080
	https_port $HTTPS_PORT
}
https://$BRIDGE_DOMAIN:$HTTPS_PORT {
	tls /certs/fullchain.pem /certs/privkey.pem
	reverse_proxy bridge:8765
	header Strict-Transport-Security "max-age=31536000"
}
EOF
  CADDY_PORTS="      - \"$HTTPS_PORT:$HTTPS_PORT\""
  # Caddy reads a certificate when it starts. The wrapper reloads it only when the one in use fails:
  # it sleeps until that certificate's end date, then loads DSM's renewed files (DSM renews a month
  # ahead); while the files hold no valid certificate yet, it retries hourly. A failed reload keeps
  # the running certificate. The image is Caddy's with openssl added (to read the end date).
  cat > caddy-watch.sh <<'EOF'
#!/bin/sh
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid; exit 0' TERM INT
ends() {  # the end of the certificate in the files, in seconds since 1970 (0 if unreadable)
  e=$(openssl x509 -in /certs/fullchain.pem -noout -dateopt iso_8601 -enddate 2>/dev/null | cut -d= -f2)
  date -u -d "${e%Z}" +%s 2>/dev/null || echo 0
}
end=$(ends)
echo "certificate valid until $(date -u -d "@$end" 2>/dev/null)"
while kill -0 $pid 2>/dev/null; do
  now=$(date +%s)
  if [ "$end" -gt "$now" ]; then sleep $((end - now)) & wait $!; continue; fi
  new=$(ends)
  if [ "$new" -gt "$now" ] && caddy reload --force --config /etc/caddy/Caddyfile --adapter caddyfile; then
    end=$new; echo "certificate expired: renewed one loaded"
  else
    echo "certificate expired and no renewed one loaded yet: retrying in an hour"; sleep 3600 & wait $!
  fi
done
exit 1  # Caddy stopped: Docker restarts the container
EOF
  CADDY_CERTS="      - $CERT_DIR:/certs:ro
      - ./caddy-watch.sh:/caddy-watch.sh:ro
    command: [\"sh\", \"/caddy-watch.sh\"]"
  get_image trading-caddy
  CADDY_IMAGE="    image: ghcr.io/svr-net/trading-caddy:latest"
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
      # IBC: the API precautions ticked, so IB Gateway accepts the bridge's orders routed directly to
      # the exchange (into its opening and closing auctions) instead of rejecting them (10311). The
      # account owner's choice (2 Nov 2026); the bridge keeps its own checks on every order.
      BYPASS_WARNING: "yes"
      TWOFA_TIMEOUT_ACTION: restart
      AUTO_RESTART_TIME: "11:59 PM"
      TZ: \${TZ}
  bridge:
$BRIDGE_SRC
    restart: always
    depends_on: [ib-gateway]
    environment:
      BRIDGE_KEY: \${BRIDGE_KEY}
      EOD_TOKEN: \${EOD_TOKEN}
      CAPITAL: \${CAPITAL}
      AUTO_MARKETS: \${AUTO_MARKETS:-}
      HISTORY_DIR: /data/history
      IB_HOST: ib-gateway
      IB_PORT: "${IB_API_PORT:-$([ "$TRADING_MODE" = live ] && echo 4003 || echo 4004)}"
      BIND: 0.0.0.0
      ALLOWED_ORIGIN: https://svr-net.github.io
    volumes:
      - ./history:/data/history   # the orders placed and the fills, kept across updates
  restarter:
    # Restarts IB Gateway when the bridge finds its API hung (tools/ibkr/restarter.py). The only
    # container with Docker's socket: no port, no secrets, and it does nothing but that one restart.
$BRIDGE_SRC
    restart: always
    command: ["python", "tools/ibkr/restarter.py"]
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
  caddy:
${CADDY_IMAGE:-    image: caddy:2.8}
    restart: always
    depends_on: [bridge]
    ports:
$CADDY_PORTS
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
      - caddy-config:/config
${CADDY_CERTS:-}
volumes:
  caddy-data:
  caddy-config:
EOF

if [ "${SKIP_START:-}" = 1 ]; then say "Files written in $DIR (SKIP_START=1: nothing started)."; exit 0; fi
say ""
say "Starting (the first time downloads the images: a few minutes)..."
$COMPOSE up -d

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
if [ "$HTTPS_MODE" = synology ]; then
  say ""
  say "DSM renews the certificate in place; the bridge's HTTPS loads the renewed one by itself when the current one expires."
fi
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
