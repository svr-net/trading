#!/bin/sh
# One container on the NAS: IB Gateway (logged in by IBC) and the broker bridge together
# (ghcr.io/svr-net/trading-ibkr, docker/ibkr/Dockerfile.ibkr), in place of the four containers the
# setup had before (ib-gateway, bridge, restarter, caddy). Run it again to update to the newest image.
#
#   sudo sh /volume1/DEV/trading/docker/ibkr/ibkr-nas.sh
#
# It reads the answers setup-nas.sh saved in docker/ibkr/nas/.env (IBKR_ENV to use another file):
# the IBKR login, TRADING_MODE, EOD_TOKEN, CAPITAL, AUTO_MARKETS, TZ, BRIDGE_KEY, HTTPS_MODE,
# HTTPS_PORT and, for HTTPS_MODE=synology, CERT_DIR (DSM's certificate folder, read-only). The orders'
# history stays in docker/ibkr/nas/history. Nothing is built here: the image comes from GitHub's
# registry, else from the repository's nas-images release over IPv4.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
DIR=${IBKR_DIR:-$HERE/nas}
ENV=${IBKR_ENV:-$DIR/.env}
NAME=trading-ibkr
IMG=ghcr.io/svr-net/$NAME:latest

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || die "Docker is not installed (Synology: Package Center, Container Manager)."
docker info >/dev/null 2>&1 || die "Docker is not running, or this user may not use it (run with sudo)."
[ -f "$ENV" ] || die "no settings at $ENV: run docker/ibkr/deploy-nas.sh once first (it asks the questions)"
set -a; . "$ENV"; set +a
[ -n "${TWS_USERID:-}" ] && [ -n "${TWS_PASSWORD:-}" ] && [ -n "${BRIDGE_KEY:-}" ] || die "$ENV lacks TWS_USERID, TWS_PASSWORD or BRIDGE_KEY"
PORT=${HTTPS_PORT:-8765}
mkdir -p "$DIR/history"

# --- The image ---------------------------------------------------------------------------------------
ARCH=$(uname -m); case "$ARCH" in x86_64|amd64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;; *) die "processor $(uname -m) not supported" ;; esac
say "Getting $IMG (at most 3 minutes from the registry)..."
if ! timeout 180 docker pull -q "$IMG" >/dev/null 2>&1; then
  url=https://github.com/svr-net/trading/releases/download/nas-images/$NAME-$ARCH.tar.gz
  say "  the registry did not answer: downloading $url over IPv4"
  curl -4 -fsSL --retry 3 --connect-timeout 20 --max-time 1800 "$url" -o "/tmp/$NAME.tar.gz" || die "could not download $url"
  gunzip -c "/tmp/$NAME.tar.gz" | docker load -q >/dev/null || die "could not load the image"
  rm -f "/tmp/$NAME.tar.gz"
fi
docker image inspect "$IMG" >/dev/null 2>&1 || die "$IMG is not there"

# --- HTTPS: DSM's certificate, else one made here for the NAS (trust it once on each device) ------------
TLS=""
case "${HTTPS_MODE:-lan}" in
  synology)
    [ -n "${CERT_DIR:-}" ] && [ -f "$CERT_DIR/fullchain.pem" ] && [ -f "$CERT_DIR/privkey.pem" ] || die "HTTPS_MODE=synology but no fullchain.pem and privkey.pem in CERT_DIR ($CERT_DIR)"
    TLS="-v $CERT_DIR:/certs:ro -e TLS_SRC_CERT=/certs/fullchain.pem -e TLS_SRC_KEY=/certs/privkey.pem" ;;
  *)
    if [ ! -f "$DIR/tls/cert.pem" ]; then
      mkdir -p "$DIR/tls"
      host=${LAN_HOST:-${BRIDGE_DOMAIN:-localhost}}
      openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout "$DIR/tls/key.pem" -out "$DIR/tls/cert.pem" \
        -subj "/CN=$host" -addext "subjectAltName=DNS:$host,IP:${LAN_HOST:-127.0.0.1}" 2>/dev/null \
        || openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout "$DIR/tls/key.pem" -out "$DIR/tls/cert.pem" -subj "/CN=$host" \
        || die "could not make a certificate (openssl)"
      chmod 600 "$DIR/tls/key.pem"
      say "Made a certificate for $host: $DIR/tls/cert.pem (trust it once on each device you open the page on)"
    fi
    TLS="-v $DIR/tls:/certs:ro -e TLS_SRC_CERT=/certs/cert.pem -e TLS_SRC_KEY=/certs/key.pem" ;;
esac

# --- The old containers go, then the one starts ---------------------------------------------------------
if [ -f "$DIR/docker-compose.yml" ]; then
  say "Removing the old containers (ib-gateway, bridge, restarter, caddy)..."
  (cd "$DIR" && { docker compose down 2>/dev/null || docker-compose down 2>/dev/null || true; })
  mv "$DIR/docker-compose.yml" "$DIR/docker-compose.yml.old"
fi
docker rm -f tws "$NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart always \
  -e TWS_USERID="$TWS_USERID" -e TWS_PASSWORD="$TWS_PASSWORD" -e TRADING_MODE="${TRADING_MODE:-paper}" \
  -e BRIDGE_KEY="$BRIDGE_KEY" -e EOD_TOKEN="${EOD_TOKEN:-}" -e CAPITAL="${CAPITAL:-5000}" \
  -e AUTO_MARKETS="${AUTO_MARKETS:-}" -e TZ="${TZ:-Europe/London}" \
  -e VNC_SERVER_PASSWORD="${VNC_SERVER_PASSWORD:-}" \
  $TLS \
  -v "$DIR/history:/data/history" \
  -p "$PORT:8765" -p 127.0.0.1:4002:4004 -p 127.0.0.1:5900:5900 \
  "$IMG" >/dev/null

# --- After a NAS restart (Synology runs /usr/local/etc/rc.d/*.sh at boot) ---------------------------------
if [ -d /usr/local/etc/rc.d ] && [ -w /usr/local/etc/rc.d ]; then
  rm -f /usr/local/etc/rc.d/trading-bridge.sh
  cat > /usr/local/etc/rc.d/trading-ibkr.sh <<EOF
#!/bin/sh
# Written by ibkr-nas.sh: starts the IBKR container once Docker is up after the NAS restarts.
[ "\${1:-start}" = start ] || exit 0
(
  export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH
  i=0; until docker info >/dev/null 2>&1 || [ \$i -ge 180 ]; do i=\$((i + 1)); sleep 5; done
  j=0; until docker start $NAME >/dev/null 2>&1 || [ \$j -ge 20 ]; do j=\$((j + 1)); sleep 30; done
  echo "\$(date -u '+%F %T') UTC: started after boot"
) >> "$DIR/boot.log" 2>&1 &
EOF
  chmod 755 /usr/local/etc/rc.d/trading-ibkr.sh
fi

say ""
say "Started one container: $NAME (IB Gateway and the bridge)."
say "IB Gateway logs in within about 2 minutes:  sudo docker logs -f $NAME"
say "The bridge: https://${BRIDGE_DOMAIN:-${LAN_HOST:-this NAS}}:$PORT  (the same address and key as before)"
say "Update later: run this script again. Remove: sudo docker rm -f $NAME"
