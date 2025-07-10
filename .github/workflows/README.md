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
curl -L -o ~/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/model.safetensors  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors?download=true -sSf
curl -L -o ~/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin?download=true -sSf
curl -L -o ~/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_0.gguf  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf  https://huggingface.co/Segilmez06/SmolLM2-135M-Instruct-Q4_K_M-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf?download=true -sSf
curl -L -o ~/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf  https://huggingface.co/tensorblock/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf?download=true -sSf
```
## Tests

The following runs all tests with all CPU, GPU, and WASM features and targets

```bash
cargo test
dx build -p phymes-app
cargo check -p phymes-server --features wsl,gpu,candle --all-targets
cargo test -p phymes-server --features wsl,gpu,candle
cargo test -p phymes-agents --features wsl,gpu,candle
cargo run --package phymes-agents --features wsl,gpu,candle --release --example chat -- --weights-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json" --weights-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf" --tokenizer-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json" --tokenizer-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json" --candle-asset "SmoLM2-135M-chat"
cargo run --package phymes-agents --features wsl,gpu,candle --release --example chatagent
cargo check --all-targets
cargo check -p phymes-core --all-targets --features wsl
cargo check -p phymes-core --all-targets --features wasip2
cargo check -p phymes-agents --all-targets --features wsl
cargo check -p phymes-agents --all-targets --features wasip2
cargo check -p phymes-server --all-targets --features wsl
cargo check -p phymes-server --all-targets --features wasip2
cargo check -p phymes-app --all-targets --features mobile
cargo check -p phymes-app --all-targets --features desktop
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo check -p phymes-core --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-core --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_core-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo build -p phymes-core --target wasm32-wasip2 --no-default-features --features wasip2 --release --example addrows
wasmtime run target/wasm32-wasip2/release/examples/addrows.wasm
cargo build -p phymes-agents --no-default-features --features wasip2,candle --target wasm32-unknown-unknown
cargo test -p phymes-agents --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_agents-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME "$file"; done
cargo build --package phymes-agents --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example chat
cargo build --package phymes-agents --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example chatagent
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/release/examples/chatagent.wasm
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/release/examples/chat.wasm --weights-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json" --weights-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf" --tokenizer-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json" --tokenizer-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json" --candle-asset "SmoLM2-135M-chat"
cargo build -p phymes-server --no-default-features --features wasip2,candle --target wasm32-unknown-unknown
cargo test -p phymes-server --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_server-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME "$file"; done
cargo build -p phymes-server --no-default-features --features wasip2,candle --target wasm32-wasip2 --release
mdbook test phymes-book
mdbook build phymes-book
cargo doc --document-private-items --no-deps -p phymes-core
cargo doc --document-private-items --no-deps -p phymes-agents
cargo doc --document-private-items --no-deps -p phymes-server
cargo doc --document-private-items --no-deps -p phymes-app
```