# README
## Linux (Ubuntu) dependencies

The following will setup all dependencies and caches on a fresh Ubuntu instance

```bash
apt update
DEBIAN_FRONTEND=noninteractive apt install --assume-yes git clang curl libssl-dev llvm libudev-dev make pkg-config protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "~/.cargo/env"
rustc --version 
rustup toolchain install stable --target x86_64-unknown-linux-gnu,wasm32-unknown-unknown,wasm32-wasip2
rustup default stable
rustup component add clippy
rustup component add rustfmt
cargo install --git https://github.com/rust-lang/mdBook.git mdbook
curl https://wasmtime.dev/install.sh -sSf | bash
mkdir -p ~/.cache/hf
cp -a $GITHUB_WORKSPACE/.cache/hf/. ~/.cache/hf/
curl -L -o ~/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/model.safetensors https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors?download=true -sSf
curl -L -o ~/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin?download=true -sSf
curl -L -o ~/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf https://huggingface.co/sudomoniker/all-MiniLM-L6-v2-Q8_0-GGUF/resolve/main/all-minilm-l6-v2-q8_0.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf https://huggingface.co/Segilmez06/SmolLM2-135M-Instruct-Q4_K_M-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf https://huggingface.co/tensorblock/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf?download=true -sSf
```

## Tests

The following runs all tests with all CPU, GPU, and WASM features and targets
see [script](ci.sh)

## Additional cache resources for benchmarking

The following will setup additional dependencies and cache resources

```bash
# cargo install cargo-criterion
curl -L -o ~/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q8_0.gguf https://huggingface.co/Segilmez06/SmolLM2-135M-Instruct-Q4_K_M-GGUF/resolve/main/smollm2-135m-instruct-q8_0.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-f16.gguf https://huggingface.co/MaziyarPanahi/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.fp16.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-3b-instruct-q5_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true -sSf
mkdir ~/.cache/metrics
mkdir ./target/criterion/metrics
```

## Benchmarks

The following runs all benchmarks with all CPU, GPU, and WASM features and targets

```bash
cargo bench --bench candle_asset -p phymes-ml --no-default-features --features wsl,gpu,candle -- --sample-size 10
cargo bench --bench candle_asset -p phymes-ml --no-default-features --features wsl,candle -- --sample-size 10
cargo bench --bench candle_asset -p phymes-ml --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run
for file in target/wasm32-wasip2/release/deps/candle_asset-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --dir=$HOME/.cache/metrics --dir=./target/criterion --env=HOME=$HOME "$file" --bench --sample-size 10; done
cargo bench --bench chat -p phymes-ml --no-default-features --features wsl,gpu,candle -- --sample-size 10
cargo bench --bench chat -p phymes-ml --no-default-features --features wsl,candle -- --sample-size 10
cargo bench --bench chat -p phymes-ml --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run
for file in target/wasm32-wasip2/release/deps/chat-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --dir=$HOME/.cache/metrics --dir=./target/criterion --env=HOME=$HOME "$file" --bench --sample-size 10; done
cargo bench --bench chat_agent_session -p phymes-agents --no-default-features --features wsl,gpu,candle -- --sample-size 10
cargo bench --bench chat_agent_session -p phymes-agents --no-default-features --features wsl,candle -- --sample-size 10
cargo bench --bench chat_agent_session -p phymes-agents --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run
for file in target/wasm32-wasip2/release/deps/chat_agent_session-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --dir=$HOME/.cache/metrics --dir=./target/criterion --env=HOME=$HOME "$file" --bench --sample-size 10; done
cargo bench --bench doc_rag_session -p phymes-agents --no-default-features --features wsl,gpu,candle -- --sample-size 10
cargo bench --bench doc_rag_session -p phymes-agents --no-default-features --features wsl,candle -- --sample-size 10
cargo bench --bench doc_rag_session -p phymes-agents --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run
for file in target/wasm32-wasip2/release/deps/doc_rag_session-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --dir=$HOME/.cache/metrics --dir=./target/criterion --env=HOME=$HOME "$file" --bench --sample-size 10; done
cargo bench --bench candle_ops -p phymes-data --no-default-features --features wsl,gpu,candle -- --sample-size 10 --measurement-time 1
cargo bench --bench candle_ops -p phymes-data --no-default-features --features wsl,candle -- --sample-size 10 --measurement-time 1
cargo bench --bench candle_ops -p phymes-data --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run
for file in target/wasm32-wasip2/release/deps/candle_ops-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --dir=$HOME/.cache/metrics --dir=./target/criterion --env=HOME=$HOME "$file" --bench --sample-size 10; done
mv ~/.cache/metrics/* ./target/criterion/metrics/
```

## Semantic versioning

The following will change the version of all `Cargo.toml` and `Cargo.lock` files

```bash
export RELEASE_VERSION="0.3.0"
export PACKAGES="phymes-app phymes-agents phymes-ml phymes-data phymes-core phymes-server phymes-diagnostics"
for p in $PACKAGES; do cd $p;  awk -v ver="$RELEASE_VERSION" '/^version = / {sub(/= "[^"]*"/, "= \""ver"\""); print; next} {print}' Cargo.toml > Cargo.toml.new;  mv Cargo.toml.new Cargo.toml; cd ..; awk -v ver="$RELEASE_VERSION" -v package="$p" '"^name = \"\"package\"\"$" {print; getline; sub(/version = "[^"]*"/, "version = \""ver"\""); print; next} {print}' Cargo.lock > Cargo.lock.new; mv Cargo.lock.new Cargo.lock; done
```

### Known issues
`Cargo.lock` does not update correctly. Workaround is to delete it and regenerate it.

## Release builds

The following will build each of the releases for distribution

```bash
# Web app with NVIDIA CUDA GPU support using native Candle
dx bundle -p phymes-app --platform web --release
cargo build --package phymes-server --features wsl,gpu,candle,hf_hub --release
mv target/release/phymes-server target/dx/phymes-app/release/web/public/
tar -czf phymes-web-candle-cuda12.6.2-ubuntu24.04.tar.gz -C target/dx/phymes-app/release/web/public .

# Linux desktop app with NVIDIA CUDA GPU support using native Candle
cargo build -p phymes-app --features desktop --release
cargo build --package phymes-server --features wsl,gpu,candle,hf_hub --release
tar -czf phymes-desktop-candle-cuda12.6.2-ubuntu24.04.tar.gz target/release/phymes-app target/release/phymes-server

# Web app without GPU support using OpenAI API
dx bundle -p phymes-app --platform web --release
cargo build --package phymes-server --no-default-features --features wsl,api --release
mv target/release/phymes-server target/dx/phymes-app/release/web/public/
tar -czf phymes-web-openai-ubuntu24.04.tar.gz -C target/dx/phymes-app/release/web/public .

# Linux desktop app without GPU support using OpenAI API
cargo build -p phymes-app --features desktop --release
cargo build --package phymes-server --no-default-features --features wsl,api --release
tar -czf phymes-desktop-openai-ubuntu24.04.tar.gz target/release/phymes-app target/release/phymes-server

# WASM app without GPU support using native Candle
cargo build -p phymes-server --no-default-features --features wasip2,candle --target wasm32-wasip2 --release
tar -czf phymes-candle-wasm32-wasip2.tar.gz -C target/wasm32-wasip2/release ./phymes-server.wasm
```