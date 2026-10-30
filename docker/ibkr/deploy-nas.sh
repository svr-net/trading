#!/bin/sh
# Deploys the broker bridge on a NAS (e.g. Synology): clones svr-net/trading into a folder (default
# /volume1/DEV/trading), or updates the clone, then builds and starts IB Gateway, the bridge and Caddy
# from it (docker/ibkr/setup-nas.sh), so the page (https://svr-net.github.io/trading/) can use it.
#
#   curl -fsSL https://raw.githubusercontent.com/svr-net/trading/main/docker/ibkr/deploy-nas.sh -o deploy-nas.sh
#   sudo sh deploy-nas.sh                      # first time: asks the questions; later: updates and restarts
#
# TARGET (default /volume1/DEV/trading), BRANCH (default main). Your answers stay in
# $TARGET/docker/ibkr/nas/.env (git-ignored, so updates never touch them); the other answers can come
# from the environment as in setup-nas.sh. Without git on the NAS, git runs in a small container.
set -eu

TARGET=${TARGET:-/volume1/DEV/trading}
BRANCH=${BRANCH:-main}
URL=https://github.com/svr-net/trading.git

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || die "Docker is not installed (Synology: Package Center, Container Manager)."
docker info >/dev/null 2>&1 || die "Docker is not running, or this user may not use it (run with sudo)."

git_() {  # git, or git in a container (alpine/git) on the folder above the clone
  if command -v git >/dev/null 2>&1; then git "$@"; return; fi
  parent=$(dirname "$TARGET")
  docker run --rm -v "$parent:$parent" -w "$parent" -e HOME=/tmp alpine/git:2.47.2 -c safe.directory='*' "$@"
}

mkdir -p "$(dirname "$TARGET")"
if [ -d "$TARGET/.git" ]; then
  say "Updating $TARGET ($BRANCH)"
  git_ -C "$TARGET" fetch -q origin "$BRANCH"
  git_ -C "$TARGET" checkout -q "$BRANCH"
  git_ -C "$TARGET" merge -q --ff-only "origin/$BRANCH" || die "the clone has local changes; keep your settings in docker/ibkr/nas/ only"
else
  if [ -e "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then die "$TARGET exists and is not a clone; move it away or set TARGET"; fi
  say "Cloning into $TARGET"
  git_ clone -q --branch "$BRANCH" "$URL" "$TARGET"
fi
say "At $(git_ -C "$TARGET" log -1 --format='%h %s' 2>/dev/null || echo "$BRANCH")"

IBKR_DIR="$TARGET/docker/ibkr/nas" REPO_DIR="$TARGET" sh "$TARGET/docker/ibkr/setup-nas.sh"
