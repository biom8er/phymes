## Introduction

We welcome and encourage contributions of all kinds, such as:

1. Tickets with issue reports of feature requests
2. Documentation improvements
3. Code (PR or PR Review)

In addition to submitting new PRs, we have a healthy tradition of community
members helping review each other's PRs. Doing so is a great way to help the
community as well as get more familiar with Rust and the relevant codebases.

## Finding and Creating Issues to Work On

You can find a curated [good-first-issue] list to help you get started.

Phymes is an open contribution project, and thus there is no particular
project imposed deadline for completing any issue or any restriction on who can
work on an issue, nor how many people can work on an issue at the same time.

If someone is already working on an issue that you want or need but hasn't
been able to finish it yet, you should feel free to work on it as well. In
general it is both polite and will help avoid unnecessary duplication of work if
you leave a note on an issue when you start working on it.

If you want to work on an issue which is not already assigned to someone else
and there are no comment indicating that someone is already working on that
issue then you can assign the issue to yourself by submitting a single word
comment `take`. This will assign the issue to yourself. However, if you are
unable to make progress you should unassign the issue by using the `unassign me`
link at the top of the issue page (and ask for help if are stuck) so that
someone else can get involved in the work.

If you plan to work on a new feature that doesn't have an existing ticket, it is
a good idea to open a ticket to discuss the feature. Advanced discussion often
helps avoid wasted effort by determining early if the feature is a good fit for
Phymes before too much time is invested. It also often helps to discuss your
ideas with the community to get feedback on implementation.

[good-first-issue]: https://github.com/biom8er/phymes/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

## Developer's guide to Phymes

<!--- ANCHOR: developing --->

### Setting Up Your Build Environment

Install the Rust tool chain:

https://www.rust-lang.org/tools/install

An example bash script for installing the Rust tool chain for Linux is the following:

```bash
apt update
DEBIAN_FRONTEND=noninteractive apt install --assume-yes git clang curl libssl-dev llvm libudev-dev make pkg-config protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup toolchain install stable --target x86_64-unknown-linux-gnu
rustup default stable
rustc --version
```

Also, make sure your Rust tool chain is up-to-date, because we always use the latest stable version of Rust to test this project.

```bash
rustup update stable
```

### Setting up GPU acceleration with CUDA

Install CUDA for linux:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

GPU acceleration with CUDA is currently only support for Linux (including WSL2) at this time. An example bash script for installing CUDA for WSL2 is the following:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install cuda-toolkit-12-6
```

Please replace the repo and cuda versions accordingly. Check the Cuda installation

```bash
nvcc --version
nvidia-smi --query-gpu=compute_cap --format=csv
```

Install CuDNN backend for linux:

https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html

An example bash script for install CuDNN for Linux is the following:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install cudnn
```

Please replace the repo and cuda versions accordingly.

### Setting up NVIDIA NIMs for local deployment

Obtain an NGC API key following the [instructions](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/quickstart.md#obtain-an-api-key).

Install the NVIDIA Container Toolkit following the [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Check that the installation was successful by running the following:

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The NGC catalogue can be viewed using `NGC CLI`. Install `NGC` following the [instructions](https://org.ngc.nvidia.com/setup/installers/cli)

Alternatively, the NGC catalogue can be viewed [online](https://build.nvidia.com/). For example, the open-source Llama3.2 model can be deployed locally following the [instructions](https://build.nvidia.com/meta/llama-3.2-1b-instruct/deploy), and alternatively accessed via the NVIDIA NIMs API if available (see the NIMs LLM [API](NIMs LLM API https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html) for OpenAPI schema). 

### Setting up WASM build environment

Add the following wasm32 compilation targets from the nightly Rust toolchain:

```bash
rustup update nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
rustup target add wasm32-wasip2 --toolchain nightly
```

In addition, we recommend using [wasmtime] for running wasi components

```bash
curl https://wasmtime.dev/install.sh -sSf | bash
```

[wasmtime]: https://github.com/bytecodealliance/wasmtime

### Setting up Dioxus

The front-end application is built using [dioxus](https://dioxuslabs.com) to enable creating web, desktop, and mobile applications using Rust

```bash
cargo install dioxus-cli
```

### How to compile

This is a standard cargo project with workspaces. To build the different workspaces, you need to have `rust` and `cargo` and you will need to specify workspaces using the using the `-p`, `--project` flag:

```bash
cargo build -p phymes-core
```

CPU, GPU, and WASM-specific compilation features are gated behind feature flags `wsl`, `gpu`, and `wasip2` respectively. The use of embedded Candle or OpenAI API token services are gated behind the feature flag `candle`, which indicates to use embedded candle models.

The following will build the `phymes-agents` workspace with different configurations of CPU and GPU acceleration for Tensor and Token services:

```bash
# Native CPU for tensor operations and local/remote OpenAI API token services
cargo build -p phymes-agents --features wsl --release

# Native CPU for tensor operations and embedded Candle for token services
cargo build -p phymes-agents --features wsl,candle --release

# GPU support for tensor operations and local/remote OpenAI API token services
cargo build -p phymes-agents --features wsl,gpu --release

# GPU support for tensor operations and embedded Candle for token services
cargo build -p phymes-agents --features wsl,gpu,candle --release
```

Please ensure that all CUDA related environmental variables are setup correctly for GPU acceleration. Most errors related to missing CUDA or CuDNN libraries are related to missing environmental variables particularly on WSL2.

```bash
export PATH=$PATH:/usr/local/cuda/bin:/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
```

The following will build the phymes-agents workspace as a WASIp2 component:

```bash
cargo build -p phymes-agents --target wasm32-wasip2 --no-default-features --features wasip2,candle --release
```

Mixing and matching features that are compilation target specific and compilation targets will result in build errors.

You can also use rust's official docker image:

```bash
docker run --rm -v $(pwd):/phymes -it rust /bin/bash -c "cd /phymes && rustup component add rustfmt && cargo build -p phymes-core"
```

From here on, this is a pure Rust project and `cargo` can be used to run tests, benchmarks, docs and examples as usual.

### Setting up the cache for running tests and examples

Many of the tests (and examples if running without the GPU or on WASM) depend upon a local cache of model assets to run. The following bash script can be used to prepare the local assets:

```bash
# ensure your home environmental variable is set
echo $HOME

# make the cache directory
mkdir -p $HOME/.cache/hf

# copy over the cache files from the root of the GitHub repository
cp -a .cache/hf/. $HOME/.cache/hf/

# download the model assets manually from HuggingFace
curl -L -o $HOME/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/model.safetensors  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors?download=true -sSf
curl -L -o $HOME/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin?download=true -sSf
curl -L -o $HOME/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_0.gguf  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf?download=true -sSf
curl -L -o $HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf  https://huggingface.co/Segilmez06/SmolLM2-135M-Instruct-Q4_K_M-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf?download=true -sSf
curl -L -o $HOME/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf  https://huggingface.co/tensorblock/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf?download=true -sSf
```

### Setting up local OpenAI API endpoints

Instead of using token credits with remote OpenAI API endpoints, it is possible to run the tests and examples locally using self-hosted open-source NVIDIA NIMs. Modify the following code depending upon the model(s) to be locally deployed:

```bash
# Text Generation Inference with Llama 3.2 (terminal 1)
export NGC_API_KEY=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
export LOCAL_NIM_CACHE=$HOME/.cache/nim
docker run -it --rm --gpus all --shm-size=16GB -e NGC_API_KEY -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" -u $(id -u) -p 8000:8000 nvcr.io/nim/meta/llama-3.2-1b-instruct:1.8.6

# Text Embedding Inference with Llama 3.2 (terminal 2)
export NGC_API_KEY=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
export LOCAL_NIM_CACHE=$HOME/.cache/nim
docker run -it --rm --gpus all --shm-size=16GB -e NGC_API_KEY -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" -u $(id -u) -p 8001:8000 nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
```

Note that the tests and examples assume that the local OpenAI API endpoints for NVIDIA NIMs are `http://0.0.0.0:8000/v1` for Text Generation Inference (TGI, Chat) and `http://0.0.0.0:8001/v1` for Text Embedding Inference (TEI, Embed), respectively. The defaults can be overwritten by setting the environmental variables for the TEI and TGI endpoints, respectively.

```bash
# URL of the local TGI NIMs deployment
export CHAT_API_URL=http://0.0.0.0:8000/v1

# URL of the local TEI NIMs deployment
export EMBED_API_URL=http://0.0.0.0:8001/v1
```

Also, be sure to add your `NGC_API_KEY` to the environmental variables before running tests or examples in a different terminal.

```bash
# NVIDIA API Key
export NGC_API_KEY=nvapi-...
```

### Running the tests

Run tests using the Rust standard `cargo test` command:

```bash
# run all unit and integration tests with default features
cargo test

# run tests for the phymes-core crate with all features enabled
cargo test -p phymes-core --all-features

# run a specific test for the phymes-core crate with the wsl feature enabled
# and printing to the console
cargo test test_session_update_state -p phymes-core --features wsl -- --no-capture

# run the doc tests
cargo test --doc
```

You can find up-to-date information on the current CI tests in [.github/workflows](https://github.com/biom8er/phymes/tree/main/.github/workflows). The phymes-server, phymes-core, and phymes-agents crates have unit tests. Please note that many of the tests will in the phymes-agents crate do not run on the CPU due to the amount of time that it takes to run them. To run all tests in the phymes-agents create, either enable GPU acceleration with Candle using `--features wsl,gpu,candle` feature flag, or with OpenAI API local/remote token services using `--feature wsl,openai_api` or `--feature wsl,gpu,openai_api` feature flags depending upon GPU availability.

```bash
# run tests for the phymes-core crate
cargo test --package phymes-core --features wsl --release

# run tests for the phymes-agents crate with GPU acceleration with Candle assets
cargo test --package phymes-agents --features wsl,gpu,candle --release
# or run tests for the phymes-agents crate on the CPU with OpenAI API token services
cargo test --package phymes-agents --features wsl --release

# run tests for the phymes-server crate
cargo test --package phymes-server --features wsl --release
```

The tests can also be ran for WASM components. However, the WASM debug output is essentially useless, so it is recommend to debug the tests natively before testing on WASM

```bash
# build tests for the phymes-core crate
cargo test --package phymes-core --target wasm32-wasip2 --no-default-features --features wasip2 --no-run

# build tests for the phymes-core crate using wasmtime
# be sure to replace the -26200b790e92721b with your systems unique hash
wasmtime run target/wasm32-wasip2/debug/deps/phymes-core-26200b790e92721b.wasm

# run tests for the phymes-agents crate with GPU acceleration
cargo test --package phymes-agents --target wasm32-wasip2 --no-default-features --features wasip2,candle --no-run

# build tests for the phymes-agents crate using wasmtime
# be sure to replace the -9ce9c7c7142d7db7 with your systems unique hash
wasmtime --dir=$HOME/.cache/hf --env=HOME=$HOME target/wasm32-wasip2/debug/deps/phymes-agents-9ce9c7c7142d7db7.wasm

# run tests for the phymes-server crate
cargo test -p phymes-server --features wasip2-candle --no-default-features --target wasm32-wasip2 --no-run

# build tests for the phymes-server crate using wasmtime
# be sure to replace the -48a453bb50fd01da with your systems unique hash
wasmtime --dir=$HOME/.cache --env=HOME=$HOME target/wasm32-wasip2/debug/deps/phymes_server-48a453bb50fd01da.wasm
```

### Running the examples

Run examples using the Rust standard `cargo run` command. A few simple examples are provided for the phymes-core and phymes-agents crates to provide new users a starting point for building application using the crates

```bash
# run examples for the phymes-core crate
cargo run --package phymes-core --features wsl --release --example addrows

# run examples for the phymes-agents crate with GPU acceleration with Candle assets
cargo run --package phymes-agents --features wsl,gpu,candle --release --example chat -- --candle-asset SmoLM2-135M-chat
cargo run --package phymes-agents --features wsl,gpu,candle --release --example chatagent

# or run examples for the phymes-agents crate on the CPU with OpenAI API token services
cargo run --package phymes-agents --features wsl --release --example chat -- --openai-asset Llama-3.2-1b-instruct
cargo run --package phymes-agents --features wsl --release --example chatagent
```

The examples can also be ran using WASM. However, all assets needed to run the example need to be provided locally unlike native where we can rely on the HuggingFace API to download and cache models for us. The following bash script can be used to build the examples in wasm and run the examples using wasmtime:

```bash
# build examples for the phymes-core crate
cargo build --package phymes-core --target wasm32-wasip2 --no-default-features --features wasip2 --release --example addrows

# run the examples for the phymes-core crate
wasmtime run target/wasm32-wasip2/release/examples/addrows.wasm

# build the chat example for the phymes-agents crate
cargo build --package phymes-agents --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example chat

# run the chat example for the phymes-agents crate
wasmtime --dir="$HOME/.cache/hf" --env=HOME=$HOME target/wasm32-wasip2/release/examples/chat.wasm --weights-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json" --weights-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf" --tokenizer-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json" --tokenizer-config-file "$HOME/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json" --candle-asset "SmoLM2-135M-chat"

# build the chatagent example for the phymes-agents crate
cargo build --package phymes-agents --target wasm32-wasip2 --no-default-features --features wasip2,candle --release --example chatagent

# run the chatagent example for the phymes-agents crate
wasmtime --dir="$HOME/.cache/hf" --env=HOME=$HOME target/wasm32-wasip2/release/examples/chatagent.wasm
```

### Clippy lints

We use `clippy` for checking lints during development, and CI runs `clippy` checks.

Run the following to check for `clippy` lints:

```bash
cargo clippy -p phymes-core --tests --examples --features wsl -- -D warnings
cargo clippy -p phymes-agents --tests --examples --features wsl -- -D warnings
cargo clippy -p phymes-server --tests --examples --features wsl -- -D warnings
cargo clippy -p phymes-app --tests --examples -- -D warnings

```

If you use Visual Studio Code with the `rust-analyzer` plugin, you can enable `clippy` to run each time you save a file. See https://users.rust-lang.org/t/how-to-use-clippy-in-vs-code-with-rust-analyzer/41881.

One of the concerns with `clippy` is that it often produces a lot of false positives, or that some recommendations may hurt readability. We do not have a policy of which lints are ignored, but if you disagree with a `clippy` lint, you may disable the lint and briefly justify it.

Search for `allow(clippy::` in the codebase to identify lints that are ignored/allowed. We currently prefer ignoring lints on the lowest unit possible.

- If you are introducing a line that returns a lint warning or error, you may disable the lint on that line.
- If you have several lints on a function or module, you may disable the lint on the function or module.
- If a lint is pervasive across multiple modules, you may disable it at the crate level.

### Rustfmt Formatting

We use `rustfmt` for formatting during development, and CI runs `rustfmt` checks.

Run the following to check for `rustfmt` changes (before submitting a PR!):

```bash
cargo fmt --all -- --check
```

The individual workspaces can then be formatted using `rustfmt`:

```bash
cargo fmt -p phymes-core --all
cargo fmt -p phymes-agents --all
cargo fmt -p phymes-server --all
cargo fmt -p phymes-app --all
```

### Rustdocs and mdBook for documentation

We use `doc` for API documentation hosted on crates.io and [mdBook](https://github.com/rust-lang/mdBook) for the guide and tutorial static website with a mermaid preprocessor [mdbook-mermaid](https://lib.rs/crates/mdbook-mermaid) is used for generating mermaid diagrams hosted on GitHub Pages.

Run the following to create the API documentation using `doc`:

```bash
cargo doc --document-private-items --no-deps -p phymes-core --features wsl
cargo doc --document-private-items --no-deps -p phymes-agents --features wsl
cargo doc --document-private-items --no-deps -p phymes-server --features wsl
cargo doc --document-private-items --no-deps -p phymes-app
```

Please visit the mdBook [guide](https://rust-lang.github.io/mdBook/guide/installation.html) for installation and usage instructions. Also, please visit [mdbook-mermaid](https://lib.rs/crates/mdbook-mermaid) for installation instructions. Run the following to create the the guide and tutorials using `mdBook`:

```bash
mdbook build phymes-book
```

### Running Benchmarks

In progress...

Running benchmarks are a good way to test the performance of a change. As benchmarks usually take a long time to run, we recommend running targeted tests instead of the full suite.

```bash
# run all benchmarks
cargo bench

# run phymes-core benchmarks
cargo bench -p phymes-core

# run benchmark for the add_rows function within the phymes-core crate
cargo bench -p phymes-core --bench add_rows
```

To set the baseline for your benchmarks, use the --save-baseline flag:

```bash
git checkout main

cargo bench -p phymes-core --bench add_rows -- --save-baseline main

git checkout feature

cargo bench -p phymes-core --bench add_rows -- --baseline main
```

### Running the CI locally

Continuous integration and deployment are orchestrated using GitHub [actions](https://docs.github.com/en/actions) on each pull request (PR) to the `main` branch. Unfortunately, debugging the CI/CD can be quite difficult and time consuming, so we recommend testing locally using a self-hosted [runner](https://github.com/biom8er/phymes/settings/actions/runners/new?). Since caching is not supported with `act`, alternative GitHub Action files for downloading resources is provided in the `.github.act` folder.

First, follow the instructions for downloading, configuring, and using the self-hosted runner.

Second, be sure to change `runs-on: ubuntu-latest` to `runs-on: self-hosted` in the YAML for all workflow files for each job. 

Third, Run the actions-runner. Now, when you open a PR, the CI will run locally on your machine.

<!--- ANCHOR_END: developing --->

## Deploying the phymes application

<!--- ANCHOR: deploying --->

The `phymes-core`, `phymes-agents`, `phymes-server`, `phymes-app` crates form a full-stack application that can run Agentic AI workflows, Graph algorithms, or Simulate networks at scale using a web, desktop, or mobile interface. Both the frontend and server need to be built in `--release` mode for improved performance and security.

### Web

First, build the frontend application using dioxus

```bash
dx bundle -p phymes-app --release
```

Second, build the server with the desired features.

```bash
# GPU support and Candle token service
cargo build --package phymes-server --features wsl,gpu,candle --release

# Or OpenAI API
cargo build --package phymes-server --features wsl --release
```

Third, move the server executable to the same directory as the web assets

```bash
mv target/release/phymes-server target/dx/phymes-app/release/web/public/phymes-server
```

Fourth, run the application and navigate to http://127.0.0.1:4000/

```bash
cd target/dx/phymes-app/release/web/public
./phymes-server
```

### Desktop

First, build the frontend application
```bash
cargo build -p phymes-app --features desktop --release
```

Second, build the phymes-server application with the desired features.

```bash
# GPU support and Candle token service
cargo build --package phymes-server --features wsl,gpu,candle --release

# Or OpenAI API
cargo build --package phymes-server --features wsl --release
```

Third, launch the `phymes-app` executable

```bash
./target/release/phymes-app
```

Fourth, launch the `phymes-server` executable

```bash
./target/release/phymes-server
```

### Mobile

In progress...

### WASM

First, build the phymes-server application with Candle token services.

```bash
cargo build --package phymes-server --no-default-features --features wasip2-candle --target wasm32-wasip2 --release
```

Second, iteratively query the phymes-server using `wasmtime`.

```bash
# Sign-in and get our JWT token
wastime --dir=$HOME/.cache phymes-server.wasm --route app/v1/sign_in --basic_auth EMAIL:PASSWORD
# mock response {"email":"EMAIL","jwt":"JWTTOKEN","session_plans":["Chat","DocChat","ToolChat"]}

# Get information about the different subjects
wastime --dir=$HOME/.cache phymes-server.wasm curl --route app/v1/subjects_info --bearer_auth JWTTOKEN --data '{"session_name":"myemail@gmail.comChat","subject_name":"","format":""}'

# Chat request
# Be sure to replace EMAIL and JWTTOKEN with your actual email and JWT token!
# Note that the session_name = email + session_plan
wastime --dir=$HOME/.cache phymes-server.wasm curl --route app/v1/chat --bearer_auth JWTTOKEN --data '{"content": "Write a python function to count prime numbers", "session_name": "EMAILChat", "subject_name": "messages"}'
```

## Deploying with local or remote OpenAI API compatible token service endpoints

OpenAI API compatible token service endpoints are supported for local (e.g., NVIDIA NIMs) or remote (e.g., OpenAI or NVIDIA NIMs). Please see the [guide](CONTRIBUTING.md#developers-guide-to-phymes) for local NVIDIA NIMs token service deployment.

Before running the `phymes-server`, setup the environmental variables as needed to access the local or remote token service endpoint as described in the [guide](CONTRIBUTING.md#developers-guide-to-phymes), or specify the endpoint urls in the `CandleEmbedConfig` and `CandleChatConfig`, respectively.

## Developing and debugging the phymes application

We recommend debugging the application using two terminals: one for `phymes-app` and another for `phymes-server`. Dioxus provides a great development loop for front-end application development with nifty hot-reloading features, but requires it's own dedicated terminal to run. Tokio provides an industry grade server along with nifty security features. During development (specifically, debug mode), the server permissions are relaxed to enable iterative debugging of the application. The `phymes-core`, `phymes-agents`, and `phymes-server` all use the Tracing crates for tracing and logging functionality. The packages and verbosity of console logging can be specified on the command line using the `RUST_LOG` environmental variable.

In the first terminal:

```bash
dx serve -p phymes-app
```

In the second terminal:

```bash
# default log level
cargo run -p phymes-server --features wsl,gpu

# only INFO level logs
RUST_LOG=phymes_server=INFO cargo run -p phymes-server --features wsl,gpu

# debug level logs for phymes-server, phymes-core, and phymes-agents
RUST_LOG=phymes_server=DEBUG,phymes_core=DEBUG,phymes_agents=DEBUG cargo run -p phymes-server --features wsl,gpu
```

<!--- ANCHOR_END: deploying --->