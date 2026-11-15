# syntax=docker/dockerfile:1
# The native tools (the strategy engine ofm_strategy, the loaders ofm_eod, the scan ofm_scan) on Alpine, built in
# layers so that a code change rebuilds only its own:
#   toolchain  Alpine with the compiler and the libraries (libgit2, libcurl, nlohmann-json, zlib): cached, never
#              rebuilt unless this list changes
#   build      the sources only (CMakeLists.txt, apps, include, strategies, backends): the layer a change rebuilds
#   check      every strategy's checks on a generated market (tools/check/strategy_check.sh) and the scan's (ofm_scan test)
#   tools      the runtime image the scan runs: the three programs and their runtime libraries (only once the checks pass)
#   out        just the programs, exported with --output
#
#   docker buildx build -f docker/native/Dockerfile --target tools -t ofm-tools --load .
ARG ALPINE=3.20
FROM alpine:${ALPINE} AS toolchain
# An extra certificate authority only where the build's network re-signs HTTPS (a proxy): the build secret "ca", if given.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then cat /run/secrets/ca >> /etc/ssl/certs/ca-certificates.crt; fi \
 && apk add --no-cache build-base cmake ninja pkgconf libgit2-dev curl-dev nlohmann-json zlib-dev ca-certificates jq

FROM toolchain AS build
WORKDIR /src
COPY CMakeLists.txt ./
COPY include/ include/
COPY strategies/ strategies/
COPY backends/ backends/
COPY apps/ apps/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null \
 && cmake --build build \
 && ./build/ofm_strategy --list

FROM build AS check
COPY tools/check/strategy_check.sh tools/check/
RUN sh tools/check/strategy_check.sh build/ofm_strategy && build/ofm_scan test && touch /checked

FROM alpine:${ALPINE} AS tools
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then cat /run/secrets/ca >> /etc/ssl/certs/ca-certificates.crt; fi \
 && apk add --no-cache libgit2 libcurl zlib libstdc++ ca-certificates
# Made only when the checks passed (the check stage's mark).
COPY --from=check /checked /etc/ofm-checked
COPY --from=build /src/build/ofm_strategy /src/build/ofm_eod /src/build/ofm_scan /usr/local/bin/
WORKDIR /work

FROM scratch AS out
COPY --from=build /src/build/ofm_strategy /src/build/ofm_eod /src/build/ofm_scan /
