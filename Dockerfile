# syntax=docker/dockerfile:1
#
# Multi-stage build of the self-adaptive trading library, its WebAssembly module and the web front end.
#
#   docker build -t trading-sat .                    # standalone web front end on nginx, default target
#   docker run --rm -p 8080:80 trading-sat           # -> http://localhost:8080
#
#   docker build --target native -t trading-sat:native .   # C++ build + unit tests
#   docker run --rm trading-sat:native                     # runs the end-to-end demo
#
#   docker build --target e2e .                     # browser tests of every page (Playwright,
#                                                   # WebGPU on SwiftShader); behind a TLS-intercepting
#                                                   # proxy add --secret id=ca,src=<proxy CA bundle>
#   docker build --target wasm-artifacts --output web/wasm .   # export sat.js/sat.wasm to the host
#   docker build --target standalone-artifacts --output dist .  # export the standalone site + zip
#
# Every build stage runs its tests, so a successful build means they passed.

ARG EMSDK_VERSION=4.0.10
ARG PLAYWRIGHT_VERSION=1.56.1
ARG NODE_VERSION=22

# ---------------------------------------------------------------- native C++ library
FROM ubuntu:24.04 AS native-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY tests tests
COPY examples examples
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build \
 && ctest --test-dir build --output-on-failure \
 && cmake --install build --prefix /opt/sat

FROM ubuntu:24.04 AS native
COPY --from=native-build /opt/sat /opt/sat
COPY --from=native-build /src/build/self_adaptive_trading_demo /usr/local/bin/
COPY --from=native-build /src/build/sat_tests /usr/local/bin/
CMD ["self_adaptive_trading_demo"]

# ---------------------------------------------------------------- WebAssembly module
FROM emscripten/emsdk:${EMSDK_VERSION} AS wasm
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY wasm wasm
COPY web web
# Rebuild from source: never ship the committed binaries from the build context.
RUN rm -f web/wasm/sat.js web/wasm/sat.wasm \
 && emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build-wasm -j"$(nproc)" \
 && node web/tests/wasm.test.mjs

FROM scratch AS wasm-artifacts
COPY --from=wasm /src/web/wasm/sat.js /src/web/wasm/sat.wasm /

# ---------------------------------------------------------------- standalone (copy-deployable) site
# One folder of static files with the .wasm embedded: no ES modules, no fetch, no server
# configuration. Open index.html from disk or copy the folder to any static host.
FROM node:${NODE_VERSION}-alpine AS standalone
ARG GIT_COMMIT=unknown
WORKDIR /src
COPY tools/standalone/package.json tools/standalone/package-lock.json tools/standalone/
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca; fi \
 && cd tools/standalone && npm ci --no-audit --no-fund
COPY tools/standalone/build.mjs tools/standalone/
COPY --from=wasm /src/web web
RUN GIT_COMMIT=${GIT_COMMIT} node tools/standalone/build.mjs --out /dist

FROM scratch AS standalone-artifacts
COPY --from=standalone /dist /

# ---------------------------------------------------------------- browser end-to-end tests
FROM mcr.microsoft.com/playwright:v${PLAYWRIGHT_VERSION}-noble AS e2e
ARG PLAYWRIGHT_VERSION
WORKDIR /web
COPY --from=wasm /src/web /web
COPY --from=standalone /dist/standalone /standalone
# Behind a TLS-intercepting proxy, pass its CA: docker build --secret id=ca,src=/path/ca.pem ...
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca; fi \
 && npm init -y >/dev/null && npm install --no-audit --no-fund playwright@${PLAYWRIGHT_VERSION}
# Development site over HTTP, then the standalone build opened from disk and served over HTTP.
RUN node tests/e2e.mjs \
 && node tests/e2e.mjs --root /standalone --file \
 && node tests/e2e.mjs --root /standalone

# ---------------------------------------------------------------- web front end (default)
FROM nginx:1.27-alpine AS web
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=standalone /dist/standalone /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
