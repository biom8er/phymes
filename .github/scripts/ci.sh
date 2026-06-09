#!/usr/bin/env bash
# This script logs all stdout and stderr to a file while still displaying them.

# Exit immediately if a command exits with a non-zero status
set -e

# Define log file (with timestamp to avoid overwriting)
LOG_FILE="./target/ci_log_$(date +'%Y-%m-%d_%H-%M-%S').txt"

# Redirect all output (stdout & stderr) to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Local CI run started. All output will be saved to: $LOG_FILE"
echo "-----------------------------------------------"

# Tests and examples
echo "Tests and examples for default features for Linux targets."
echo "-----------------------------------------------"
cargo test
dx build -p phymes-app

# GPU tests require CUDA or Metal
echo "Tests and examples for gpu feature for Linux targets."
echo "-----------------------------------------------"
cargo check --features gpu --all-targets
cargo test --features gpu
cargo run --package phymes-processor --features gpu --release --example chat -- --weights-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json" --messages "messages" --weights-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf" --tokenizer-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json" --tokenizer-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json" --candle-asset "SmoLM2-135M-chat"
cargo run --package phymes-network --features gpu --release --example chat_agent_network
cargo run --package phymes-network --features gpu --release --example doc_rag_network
cargo run --package phymes-network --features gpu --release --example tool_agent_network

# API with Candle tests require Docker and an internet connection
echo "Tests and examples for api and candle features for Linux targets."
echo "-----------------------------------------------"
cargo check --features api --all-targets
cargo test  --features api

echo "Compilation checks for Linux targets."
echo "-----------------------------------------------"
cargo check --all-targets
cargo check -p phymes-diagnostics --all-targets --no-default-features --features wsl
cargo check -p phymes-subject --all-targets --no-default-features --features wsl
cargo check -p phymes-subject --all-targets --no-default-features --features wsl,api
cargo check -p phymes-schemas --all-targets --no-default-features --features wsl
cargo check -p phymes-schemas --all-targets --no-default-features --features wsl,api
cargo check -p phymes-event --all-targets --no-default-features --features wsl
cargo check -p phymes-event --all-targets --no-default-features --features wsl,api
cargo check -p phymes-message --all-targets --no-default-features --features wsl
cargo check -p phymes-message --all-targets --no-default-features --features wsl,api
cargo check -p phymes-data --all-targets --no-default-features --features wsl
cargo check -p phymes-data --all-targets --no-default-features --features wsl,api
cargo check -p phymes-ml --all-targets --no-default-features --features wsl
cargo check -p phymes-ml --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-ml --all-targets --no-default-features --features wsl,api
cargo check -p phymes-streams --all-targets --no-default-features --features wsl
cargo check -p phymes-streams --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-streams --all-targets --no-default-features --features wsl,api
cargo check -p phymes-processor --all-targets --no-default-features --features wsl
cargo check -p phymes-processor --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-processor --all-targets --no-default-features --features wsl,api
cargo check -p phymes-task --all-targets --no-default-features --features wsl
cargo check -p phymes-task --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-task --all-targets --no-default-features --features wsl,api
cargo check -p phymes-network --all-targets --no-default-features --features wsl
cargo check -p phymes-network --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-network --all-targets --no-default-features --features wsl,api
cargo check -p phymes-templates --all-targets --no-default-features --features wsl
cargo check -p phymes-templates --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-templates --all-targets --no-default-features --features wsl,api
cargo check -p phymes-server --all-targets --no-default-features --features wsl
cargo check -p phymes-server --all-targets --no-default-features --features wsl,hf_hub,candle
cargo check -p phymes-server --all-targets --no-default-features --features wsl,api
cargo check -p phymes-app --all-targets --no-default-features --features mobile
cargo check -p phymes-app --all-targets --no-default-features --features desktop
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check

# # API without Candle tests require API key from OpenAI or NVIDIA or NVIDIA self-hosted NIMS
# echo "Tests and examples for api features for Linux targets."
# echo "-----------------------------------------------"
# cargo test --no-default-features --features wsl,api -p phymes-diagnostics --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-subject --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-schemas --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-event --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-message --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-data --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-ml --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-streams --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-processor --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-task --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-network --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-templates --all-targets
# cargo test --no-default-features --features wsl,api -p phymes-server --all-targets

echo "Tests and examples for WASM target"
echo "-----------------------------------------------"
cargo check -p phymes-diagnostics --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-diagnostics --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_diagnostics-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-subject --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-subject --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_subject-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-schemas --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-schemas --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_schemas-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-event --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-event --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_event-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-message --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-message --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_message-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-data --no-default-features --features wasip2,candle --target wasm32-unknown-unknown
cargo test -p phymes-data --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_data-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-ml --no-default-features --features wasip2,candle --target wasm32-unknown-unknown
cargo test -p phymes-ml --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_network-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME "$file"; done
cargo check -p phymes-streams --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-streams --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_streams-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-processor --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-processor --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_processor-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo build --package phymes-processor --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example chat
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/release/examples/chat.wasm --messages "messages" --weights-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json" --weights-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf" --tokenizer-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json" --tokenizer-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json" --candle-asset "SmoLM2-135M-chat"
cargo check -p phymes-task --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-task --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_task-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-network --no-default-features --features wasip2,candle --target wasm32-unknown-unknown
cargo test -p phymes-network --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_network-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME "$file"; done
cargo build --package phymes-network --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example chat_agent_network
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/release/examples/chat_agent_network.wasm
cargo build --package phymes-network --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example doc_rag_network
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/release/examples/doc_rag_network.wasm
cargo build --package phymes-network --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example tool_agent_network
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/release/examples/tool_agent_network.wasm
cargo check -p phymes-templates --features wasip2 --no-default-features --target wasm32-unknown-unknown
cargo test -p phymes-templates --features wasip2 --no-default-features --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_templates-*.wasm; do [ -f "$file" ] && wasmtime "$file"; done
cargo check -p phymes-server --no-default-features --features wasip2,candle --target wasm32-unknown-unknown
cargo test -p phymes-server --no-default-features --features wasip2,candle --target wasm32-wasip2 --no-run --release
for file in target/wasm32-wasip2/release/deps/phymes_server-*.wasm; do [ -f "$file" ] && wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME "$file"; done
cargo build -p phymes-server --no-default-features --features wasip2,candle --target wasm32-wasip2 --release

echo "Tests and examples for WASM target"
echo "-----------------------------------------------"
mdbook test phymes-book
mdbook build phymes-book
cargo doc --document-private-items --no-deps -p phymes-diagnostics
cargo doc --document-private-items --no-deps -p phymes-subject
cargo doc --document-private-items --no-deps -p phymes-schemas
cargo doc --document-private-items --no-deps -p phymes-event
cargo doc --document-private-items --no-deps -p phymes-message
cargo doc --document-private-items --no-deps -p phymes-data
cargo doc --document-private-items --no-deps -p phymes-ml
cargo doc --document-private-items --no-deps -p phymes-streams
cargo doc --document-private-items --no-deps -p phymes-processor
cargo doc --document-private-items --no-deps -p phymes-task
cargo doc --document-private-items --no-deps -p phymes-network
cargo doc --document-private-items --no-deps -p phymes-templates
cargo doc --document-private-items --no-deps -p phymes-server
cargo doc --document-private-items --no-deps -p phymes-app

echo "-----------------------------------------------"
echo "Local CI run finished."

cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check