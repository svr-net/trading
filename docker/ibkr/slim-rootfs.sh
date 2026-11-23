#!/bin/bash
# The slim image's whole filesystem (docker/ibkr/Dockerfile.ibkr's last stage is FROM scratch with only this): run in
# the IB Gateway image as root, it copies IB Gateway, IBC and the bridge with the binaries the container runs and
# every shared library they load (ldd's closure of the binaries and of IB Gateway's own libraries, and what IB Gateway
# and Xvfb were seen to load at its login, 23 Nov), and the data those read. The OpenGL stack (Mesa, LLVM: 180 MB) is
# left out: Xvfb runs without GLX and JavaFX draws in software.
#   slim-rootfs.sh /rootfs
set -euo pipefail
R=$1
mkdir -p "$R/usr/bin" "$R/usr/sbin" "$R/usr/lib" "$R/usr/lib64"
for d in bin sbin lib lib64; do  # merged /usr: the same links in the rootfs, so a library is copied once
  if [ -L "/$d" ]; then ln -sfn "$(readlink "/$d")" "$R/$d"; elif [ -d "/$d" ]; then mkdir -p "$R/$d"; fi
done
MULTI=$(gcc -dumpmachine 2>/dev/null || dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || uname -m | sed 's/$/-linux-gnu/')
LIBDIR=/usr/lib/$MULTI

copy() {  # copy a path (and the chain of symlinks to it) into the rootfs, keeping its place
  local p=$1
  [ -e "$p" ] || return 0
  while [ -L "$p" ]; do
    mkdir -p "$R$(dirname "$p")"
    cp -P "$p" "$R$p" 2>/dev/null || true
    local t
    t=$(readlink "$p")
    case "$t" in /*) p=$t ;; *) p=$(dirname "$p")/$t ;; esac
  done
  mkdir -p "$R$(dirname "$p")"
  cp -a "$p" "$R$p" 2>/dev/null || true
}
skip() { case "$1" in *libgallium* | *libLLVM* | *libllvm* | */dri/* | *libvulkan* | *libgbm* ) return 0 ;; esac; return 1; }
declare -A seen
libs() {  # the ldd closure of each file given
  local f l
  for f in "$@"; do
    while read -r l; do
      [ -n "$l" ] || continue
      skip "$l" && continue
      [ -n "${seen[$l]:-}" ] && continue
      seen[$l]=1
      copy "$l"
    done < <(ldd "$f" 2>/dev/null | awk '/=> \//{print $3} /^[ \t]*\/[^ ]+ \(/{print $1}')
  done
}

# The binaries the container runs (bash for the start scripts and IBC's, busybox for the small tools)
BIN="bash Xvfb xkbcomp envsubst setpriv openssl curl ps pgrep pkill"
for b in $BIN; do p=$(command -v "$b"); copy "$p"; libs "$p"; done
copy "$(command -v busybox)"
mkdir -p "$R/bin" "$R/usr/bin"
for a in $(busybox --list); do
  case "$a" in sh | ash | bash | ps | pgrep | pkill | setpriv | openssl | wget | busybox) continue ;; esac
  [ -e "$R/usr/bin/$a" ] || [ -e "$R/bin/$a" ] || ln -s "$(command -v busybox)" "$R/usr/bin/$a"
done
[ -e "$R/bin/sh" ] || ln -s "$(command -v bash)" "$R/bin/sh"  # bash: the full ps, pkill (procps), not busybox's
[ -e "$R/bin/bash" ] || ln -s "$(command -v bash)" "$R/bin/bash"

# IB Gateway (its own Java runtime too), IBC, the base image's scripts' place
copy /home/ibgateway
libs $(find /home/ibgateway/Jts -name '*.so*' -type f ! -path '*jxbrowser*')
# What IB Gateway and Xvfb loaded at the login screen (JavaFX on GTK 3, the X libraries), with their closure
for l in libgtk-3.so.0 libgdk-3.so.0 libatk-bridge-2.0.so.0 libXtst.so.6 libXi.so.6 libXrender.so.1 libXrandr.so.2 \
  libXinerama.so.1 libXcursor.so.1 libXcomposite.so.1 libXdamage.so.1 libXxf86vm.so.1 libxkbcommon.so.0 libX11-xcb.so.1 \
  libfontconfig.so.1 libfreetype.so.6 libharfbuzz.so.0 libpangoft2-1.0.so.0 libcairo-gobject.so.2 libgdk_pixbuf-2.0.so.0 \
  libepoxy.so.0 libXRes.so.1 libnss_files.so.2 libnss_dns.so.2 libresolv.so.2; do
  for p in "$LIBDIR/$l" /lib/"$MULTI"/"$l"; do [ -e "$p" ] && { copy "$p"; libs "$p"; }; done
done
# GTK's image loaders and its module caches (the gateway's windows load icons)
for d in "$LIBDIR/gdk-pixbuf-2.0" "$LIBDIR/glycin-loaders" "$LIBDIR/gtk-3.0"; do
  [ -d "$d" ] && { copy "$d"; libs $(find "$d" -name '*.so*' -type f); find "$d" -type f -perm -u+x ! -name '*.so*' | while read -r x; do libs "$x"; done; }
done
for d in /usr/libexec/glycin-loaders; do [ -d "$d" ] && { copy "$d"; find "$d" -type f | while read -r x; do libs "$x"; done; }; done

# Data: the keyboard maps (Xvfb), fonts and their configuration (Java and GTK), GLib's schemas, time zones, CA roots
for d in /usr/share/X11/xkb /usr/share/X11/locale /etc/fonts /usr/share/fonts/truetype/dejavu /usr/share/fonts/truetype/liberation \
  /usr/share/fonts/truetype/liberation2 /usr/share/glib-2.0/schemas /usr/share/zoneinfo /etc/ssl/certs/ca-certificates.crt \
  /usr/share/mime/mime.cache /usr/share/icons/hicolor/index.theme /etc/localtime /etc/nsswitch.conf /etc/host.conf; do copy "$d"; done
copy /var/cache/fontconfig
mkdir -p "$R/etc"
printf 'root:x:0:0:root:/root:/bin/bash\nibgateway:x:1000:1000:ibgateway:/home/ibgateway:/bin/bash\n' >"$R/etc/passwd"
printf 'root:x:0:\nibgateway:x:1000:\n' >"$R/etc/group"
grep -q '^hosts:' "$R/etc/nsswitch.conf" 2>/dev/null || printf 'passwd: files\ngroup: files\nhosts: files dns\n' >"$R/etc/nsswitch.conf"
mkdir -p "$R/tmp" "$R/root" "$R/run" "$R/data/history" "$R/var/tmp"
chmod 1777 "$R/tmp" "$R/var/tmp"
chown -R 1000:1000 "$R/home/ibgateway" "$R/data"
du -sh "$R"
