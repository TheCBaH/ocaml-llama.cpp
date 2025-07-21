
default: build

build:
	opam exec -- dune $@

runtest:
	opam exec -- dune $@ --auto-promote

static:
	opam exec -- dune build --profile static

format:
	opam exec dune fmt

run:
	opam exec dune exec ./main.exe

top:
	opam exec dune exec ./example_top.exe

utop:
	opam exec dune utop

clean:
	opam exec dune $@

models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q5_K_M.gguf

models/ggml-org/gemma-3-1b-it-Q4_K_M.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

models/ggml-org/qwen2.5-coder-0.5b-q8_0.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF/resolve/main/qwen2.5-coder-0.5b-q8_0.gguf

models/tiiuae/Falcon3-1B-Instruct-q3_k_m.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/tiiuae/Falcon3-1B-Instruct-GGUF/resolve/main/Falcon3-1B-Instruct-q3_k_m.gguf

MODEL=models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf
models: models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf ${MODEL}
simple: models
	mkdir -p _build/json
	opam exec -- dune exec -- src/simple.exe $(if ${WITH_GRAPH},-g _build/json) ${MODEL}

simple.all:
	$(foreach m,\
 models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf\
 models/ggml-org/gemma-3-1b-it-Q4_K_M.gguf\
 models/ggml-org/qwen2.5-coder-0.5b-q8_0.gguf\
 models/tiiuae/Falcon3-1B-Instruct-q3_k_m.gguf\
  , ${MAKE} MODEL=$m WITH_GRAPH=y simple &&) true

llama-simple: ${MODEL}
	${MAKE} -C lib/llama.cpp simple MODEL=$(abspath ${MODEL})

model-explorer.install:
	python3 -m pip --no-cache-dir install ai-edge-model-explorer

patch:
	git -C vendored/llama.cpp apply < patches/graph_callback_llama.patch

sync:
	git -C vendored/llama.cpp apply --3way < patches/graph_callback_llama.patch

create.patch:
	git -C vendored/llama.cpp diff HEAD > patches/graph_callback_llama.patch

LLAMA.CPP_API.REV=7b50d589a863c7631135c1226f6eab65cb406212
api.diff:
	git -C vendored/llama.cpp/ fetch --depth 1000
	git -C vendored/llama.cpp/ diff ${LLAMA.CPP_API.REV}...HEAD include >$@

GGML_API.REV=0a5a3b5cdfd887cf0f8e09d9ff89dee130cfcdde
ggml_api.diff:
	git -C vendored/llama.cpp/ fetch --depth 1000
	git -C vendored/llama.cpp/ diff ${GGML_API.REV}...HEAD ggml/include >$@

OCAML_GGML.REV=fe208f522906ae27f38098a2527c5c01da609c84
OCAML_GGML.REV_SYNC=42aa6bdead5350f06d3cf36d70f70855e6673605
ggml.sync:
	git fetch https://github.com/TheCBaH/ocaml-ggml.git
	git checkout ${OCAML_GGML.REV_SYNC} lib/ggml
	git diff ${OCAML_GGML.REV}...FETCH_HEAD lib/ggml | git apply --check
	git diff ${OCAML_GGML.REV}...FETCH_HEAD lib/ggml | git apply --3way

.PHONY: api.diff default clean format models create.patch patch run top utop model-explorer.install
