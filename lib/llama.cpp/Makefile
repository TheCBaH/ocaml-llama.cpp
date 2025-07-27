build-ggml: static.build-ggml shared.build-ggml
build-llama: static.build-llama shared.build-llama

OPTS.static=-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
TARGET.GGML.static=ggml/src/libggml.a
TARGET.GGML.shared=bin/libggml.so
FINISH.GGML.static=printf "create libggml.a\n$(foreach l, ggml ggml-base ggml-cpu, addlib build.ggml-ml.static/ggml/src/lib${l}.a)\nsave\nend" | ar -M;
ROOT=${CURDIR}/../..
LLAMA.CPP?=${ROOT}/vendored/llama.cpp
TARGET.LLAMA.CPP.static=src/libllama.a
TARGET.LLAMA.CPP.shared=bin/libllama.so
BIN_DIR=.

%.ggml-config:
	set -eux;cd ${LLAMA.CPP};\
	 cmake -B ${BIN_DIR}/build.ggml-ml.$(basename $@) -G Ninja -DCMAKE_BUILD_TYPE=Release $(OPTS.$(basename $@))

%.llama-config:
	set -eux;cd ${LLAMA.CPP};\
	 cmake -B ${BIN_DIR}/build.llama-ml.$(basename $@) -G Ninja -DCMAKE_BUILD_TYPE=Release $(OPTS.$(basename $@))

%.build-ggml: %.ggml-config
	set -eux;cd ${LLAMA.CPP};\
	 cmake --build ${BIN_DIR}/build.ggml-ml.$(basename $@) --target $(TARGET.GGML.$(basename $@)) -j $$(getconf _NPROCESSORS_ONLN);\
	 $(FINISH.GGML.$(basename $@))\
	 true

%.build-llama: %.llama-config
	set -eux;cd ${LLAMA.CPP};\
	 cmake --build ${BIN_DIR}/build.llama-ml.$(basename $@) --target $(TARGET.LLAMA.CPP.$(basename $@)) -j $$(getconf _NPROCESSORS_ONLN);\
	 true

TARGET.LLAMA.CPP.simple=bin/llama-simple
simple: BIN_DIR=$(abspath ${ROOT})/_build/llama-ml.simple
simple: MODEL=${ROOT}/models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf
simple:
	${MAKE} simple.build-llama BIN_DIR=${BIN_DIR}
	${BIN_DIR}/build.llama-ml.simple/${TARGET.LLAMA.CPP.simple} -m ${MODEL}
