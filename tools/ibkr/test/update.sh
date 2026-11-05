#!/bin/sh
# The restarter's /update on a stand-in NAS (CI, with Docker): a bridge running an image built here
# (version "old") and the restarter (this commit's code); /update must replace the bridge with
# GitHub's newest image, keep its settings (environment, volume, network name), report "up to date"
# the second time, and leave IB Gateway alone (a stand-in container that must keep its id).
set -eu
cd "$(dirname "$0")/../../.."
dir=$(mktemp -d)
docker build -q -f docker/ibkr/Dockerfile.bridge --build-arg VERSION=old -t trading-bridge-old . >/dev/null
docker build -q -f docker/ibkr/Dockerfile.bridge --build-arg VERSION=test -t trading-restarter-test . >/dev/null
cat > "$dir/docker-compose.yml" <<YML
services:
  ib-gateway:
    image: trading-restarter-test
    command: ["sleep", "infinity"]
  bridge:
    image: trading-bridge-old
    environment: { BRIDGE_KEY: standin, AUTO_MARKETS: "", HISTORY_DIR: /data/history, IB_HOST: ib-gateway, BIND: 0.0.0.0 }
    volumes: [ "./history:/data/history" ]
  restarter:
    image: trading-restarter-test
    command: ["python", "tools/ibkr/restarter.py"]
    volumes: [ "/var/run/docker.sock:/var/run/docker.sock" ]
YML
c="docker compose -p standin -f $dir/docker-compose.yml"
$c up -d
trap '$c down -v >/dev/null 2>&1 || true; docker rm -f standin-bridge-1-old >/dev/null 2>&1 || true' EXIT
sleep 5
gw=$($c ps -q ib-gateway)
post() { $c exec -T restarter python -c "import urllib.request as u; print(u.urlopen(u.Request('http://localhost:8766$1', data=b'', method='POST'), timeout=900).read().decode())"; }
out=$(post /update); echo "first: $out"
echo "$out" | grep -q '"updated from' || { echo 'FAIL: not updated'; exit 1; }
b=$($c ps -q bridge)
[ -n "$b" ] || { echo 'FAIL: no bridge container after the update'; exit 1; }
[ "$(docker inspect -f '{{.Image}}' "$b")" = "$(docker image inspect -f '{{.Id}}' ghcr.io/svr-net/trading-bridge:latest)" ] || { echo 'FAIL: not the newest image'; exit 1; }
env=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$b")
echo "$env" | grep -qx 'BRIDGE_KEY=standin' || { echo 'FAIL: the environment was lost'; exit 1; }
echo "$env" | grep -qx 'BRIDGE_VERSION=old' && { echo 'FAIL: the old image version stamp kept'; exit 1; }
docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' "$b" | grep -q /data/history || { echo 'FAIL: the volume was lost'; exit 1; }
sleep 5
$c exec -T restarter python -c "import urllib.request as u; print(u.urlopen('http://bridge:8765/', timeout=20).status)" >/dev/null 2>&1 \
  || $c exec -T restarter python -c "import socket; socket.create_connection(('bridge', 8765), 20)" || { echo 'FAIL: the bridge is not reachable by its name'; exit 1; }
[ "$($c ps -q ib-gateway)" = "$gw" ] || { echo 'FAIL: IB Gateway was touched'; exit 1; }
docker inspect standin-bridge-1-old >/dev/null 2>&1 && { echo 'FAIL: the old bridge was left behind'; exit 1; }
out=$(post /update); echo "second: $out"
echo "$out" | grep -q 'up to date' || { echo 'FAIL: not up to date the second time'; exit 1; }
echo 'update: ok'
