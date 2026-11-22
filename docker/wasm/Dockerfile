# The strategy engine (apps/ofm_strategy, with every strategy in strategies/ and the serial back end) as WebAssembly
# for the page's Lab: it runs in the browser, on the phone, on the bars the bridge serves. Built in layers so that a change to the model rebuilds only the last one:
#   toolchain  the pinned Emscripten, and its system libraries (libc++, libc, with SIMD and LTO) built once by
#              a warm-up compile with the same flags: cached, never rebuilt unless the flags or version change
#   build      the engine's sources only (apps/ofm_strategy.cpp, include/strategy, strategies/, backends/serial.cpp):
#              the only layer a strategy change rebuilds
#   gpu        the same engine with the webgpu back end (backends/webgpu.cpp, WebGPU's C API): the nested Monte Carlo's
#              trade kernel (include/strategy/mc_wgsl.hpp) on the browser's GPU; Asyncify lets the engine wait for the
#              GPU's answers. The Lab loads it only where the browser has WebGPU
#   out        just ofm_strategy.js/.wasm and ofm_strategy_gpu.js/.wasm, exported with --output
#
#   docker buildx build -f docker/wasm/Dockerfile --target out --output type=local,dest=web/lab .
#
# Single-threaded (GitHub Pages cannot send the headers WebAssembly threads need); -O3, link-time
# optimisation and 128-bit SIMD.
ARG EMSDK_VERSION=3.1.74
FROM emscripten/emsdk:${EMSDK_VERSION} AS toolchain
ENV FLAGS="-std=c++17 -O3 -flto -msimd128 -DNDEBUG"
ENV LINK="-sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createOfmStrategy -sINVOKE_RUN=0 -sEXIT_RUNTIME=1 \
 -sEXPORTED_RUNTIME_METHODS=callMain,FS,ENV -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB \
 -sENVIRONMENT=web,worker,node -sFILESYSTEM=1 -sSTACK_SIZE=4MB"
WORKDIR /warm
RUN printf '%s\n' '#include <algorithm>' '#include <array>' '#include <atomic>' '#include <cmath>' '#include <cstdio>' \
      '#include <fstream>' '#include <functional>' '#include <map>' '#include <set>' '#include <sstream>' '#include <string>' \
      '#include <thread>' '#include <vector>' \
      'int main(int c, char** v) { std::vector<double> x(c); std::map<std::string, int> m; std::ifstream f(v[0]);' \
      '  std::ostringstream s; s << x.size() << m.size(); std::printf("%s", s.str().c_str()); return 0; }' > warm.cpp \
 && em++ $FLAGS warm.cpp -o warm.js $LINK && rm -rf /warm

FROM toolchain AS build
WORKDIR /src
COPY include/ include/
COPY strategies/ strategies/
COPY backends/serial.cpp backends/
COPY apps/ofm_strategy.cpp apps/
RUN mkdir -p /out && em++ $FLAGS -Iinclude apps/ofm_strategy.cpp strategies/*.cpp backends/serial.cpp -o /out/ofm_strategy.js $LINK \
 && ls -l /out

FROM build AS gpu
COPY backends/webgpu.cpp backends/
RUN em++ $FLAGS -DOFM_WEBGPU -Iinclude apps/ofm_strategy.cpp strategies/*.cpp backends/serial.cpp backends/webgpu.cpp -o /out/ofm_strategy_gpu.js $LINK \
      -sUSE_WEBGPU=1 -sASYNCIFY=1 -sASYNCIFY_STACK_SIZE=1048576 -sEXPORT_NAME=createOfmStrategyGpu \
 && ls -l /out

FROM scratch AS out
COPY --from=gpu /out/ /
