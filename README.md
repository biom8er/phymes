# PHYMES: Parallel HYpergraph MEssaging Streams

[![CI Status](https://github.com/biom8er/phymes/actions/workflows/main.yml/badge.svg)](https://github.com/biom8er/phymes/actions/workflows/main.yml)
[![Latest version](https://img.shields.io/crates/v/phymes-core.svg)](https://crates.io/crates/phymes-core)
[![Documentation](https://docs.rs/phymes-core/badge.svg)](https://docs.rs/phymes-core)
[![License](https://img.shields.io/github/license/base-org/node?color=blue)](https://github.com/biom8er/phymes/blob/main/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/biom8er/phymes/blob/main/LICENSE-APACHE)

<!--- ANCHOR: introduction --->

## Introduction

🤔 What is PHYMES?

PHYMES (Parallel HYpergraph MEssaging Streams) is a subject-based message passing algorithm based on directed hypergraphs which provide the expressivity needed to model the heterogeneity and complexity of the real world. 

🤔 What can PHYMES do?

Phymes can be used to build scalable Agentic AI workflows, (hyper)-graph algorithms, and world simulators. Examples for building a chat bot, a tool calling agent, and document RAG agent are provided using embedded token/tensor services or local/remote token/tensor services using OpenAI compatible APIs.

🤔 Why PHYMES?

🔐 written 100% in [Rust] for performance, safety, and security.<br>
🌎 deployable on any platform (Linux, MacOs, Win, Android, and iOS) and in the browser (WebAssembly).<br>
💪 scalable to massive data sets using columnar in memory format, parallel and stream native processing, and GPU acceleration.<br>
🧩 interoperable with existing stacks by interfacing with cross-platform [Arrow] and [WASM]/[WASI].<br>
🔎 instrumented with tracing and customizable metrics to debug (hyper-)graph workflows faster.<br>

🤔 Who and what inspired PHYMES?

The implementation of Phymes takes inspiration from [DataFusion], [Pregel], and [PyG]. See GitHub Pages for guides and tutorials.

🙏 PHYMES would not be possible if it were not for the amazing open-sources projects that it is built on top of including [Arrow] and [Candle] with full-stack support from [Tokio], [Dioxus], and [Wasmtime].

[Rust]: https://www.rust-lang.org/
[Arrow]: https://arrow.apache.org/
[Candle]: https://www.rust-lang.org/
[Tokio]: https://tokio.rs/
[Dioxus]: https://dioxuslabs.com/
[DataFusion]: https://github.com/apache/datafusion
[Pregel]: https://dl.acm.org/doi/10.1145/1807167.1807184
[PyG]: https://github.com/pyg-team/pytorch_geometric
[WASM]: https://webassembly.org/
[WASI]: https://github.com/WebAssembly/WASI
[contributing]: CONTRIBUTING.md

<!--- ANCHOR_END: introduction --->

<!--- ANCHOR: installation1 --->

## Installation

Precompiled bundles for different Arch, OS, CUDA versions, and Token and Tensor services (e.g. for Agentic AI workflows) are provided on the [releases] page. 

| Arch | OS | CUDA | Token service |
| ---- | -- | ---- | ------------- |
| x86_64-unknown-linux-gnu | ubuntu22.04, ubuntu24.04 | 12.6.2, 12.9.1 | candle, openai_api |
| wasm32-wasip2 | n/a | n/a | candle |
| wasm32-unknown-unknown | n/a | n/a | candle |

Token services for agentic AI workflows can embedded in the application using `candle` or accessed locally e.g., self-hosted NVIDIA NIMs docker containers or remotely e.g., OpenAI, NVIDIA NIMs, etc. that adhere to the OpenAI API schema using `openai_api`. Tensor services are embedded in the application using `candle` with CPU vectorization and GPU acceleration support.

To install the phymes application, download the precompiled bundle that matches your system and needs, and unzip the bundle. Double click on `phymes-server` to start the server, then navigate to http://127.0.0.1:4000/ to view the web application. 

<!--- ANCHOR_END: installation1 --->

<video controls>
  <source src="./phymes-book/assets/2025-07-05_phymes-app_ui_1080p.mp4" type="video/mp4">
</video>

<!--- ANCHOR: installation2 --->

Alternatively, you can make REST API requests against the server using e.g., `curl`.

```bash
# Sign-in and get our JWT token
curl -X POST -u EMAIL:PASSWORD http://localhost:4000/app/v1/sign_in
# mock response {"email":"EMAIL","jwt":"JWTTOKEN","session_plans":["Chat","DocChat","ToolChat"]}

# Chat request
# Be sure to replace EMAIL and JWTTOKEN with your actual email and JWT token!
# Note that the session_name = email + session_plan
curl -H "Content-Type: application/json" -H "Authorization: Bearer JWTTOKEN" -d '{"content": "Write a python function to count prime numbers", "session_name": "EMAILChat", "subject_name": "messages"}' http://localhost:4000/app/v1/chat
```

Before running the `phymes-server`, setup the environmental variables *as needed* to access the local or remote OpenAI API token service endpoints.

```bash
# OpenAI API Key
export OPENAI_API_KEY=sk-proj-...

# NVIDIA API Key
export NGC_API_KEY=nvapi-...

# URL of the local/remote TGI OpenAI or NIMs deployment
export CHAT_API_URL=http://0.0.0.0:8000/v1

# URL of the local/remote TEI OpenAI or NIMs deployment
export EMBED_API_URL=http://0.0.0.0:8001/v1
```

WASM builds of `phymes-server` can be ran as stateless functions for embedded application using [wasmtime] or serverless applications.

<!--- ANCHOR_END: installation2 --->

<video controls>
  <source src="./phymes-book/assets/2025-07-05_phymes-app_server_1080p.mp4" type="video/mp4">
</video>

<!--- ANCHOR: installation3 --->

```bash
# Sign-in and get our JWT token
wastime phymes-server.wasm -- --route app/v1/sign_in --basic-auth EMAIL:PASSWORD
# mock response {"email":"EMAIL","jwt":"JWTTOKEN","session_plans":["Chat","DocChat","ToolChat"]}

# Chat request
# Be sure to replace EMAIL and JWTTOKEN with your actual email and JWT token!
# Note that the session_name = email + session_plan
wastime phymes-server.wasm curl -- --route app/v1/chat --bearer-auth JWTTOKEN --data '{"content": "Write a python function to count prime numbers", "session_name": "EMAILChat", "subject_name": "messages"}'
```

See [contributing] guide for detailed installation and build instructions.

[releases]: https://github.com/biom8er/phymes/releases
[Wasmtime]: https://github.com/bytecodealliance/wasmtime

<!--- ANCHOR_END: installation3 --->

<!--- ANCHOR: repository --->

## Repository

The [`phymes-core`], [`phymes-agents`], [`phymes-server`], [`phymes-app`] crates form a full-stack application that can run Agentic AI workflows, (Hyper-)Graph algorithms, and/or Simulate complex real world networks at scale using a web, desktop, or mobile interface.

| Crate | Description | Latest API Docs | README |
| ----- | ----------- | --------------- | ------ |
| [`phymes-core`] | Core hypergraph messaging functionality | [docs.rs](https://docs.rs/phymes-core/latest) | [README](phymes-core-readme) |
| [`phymes-agents`] | Support for AI agents and GPU accelerated data analytics | [docs.rs](https://docs.rs/phymes-agents/latest) | [README](phymes-agents-readme) |
| [`phymes-server`] | Server that runs the Agentic AI hypergraph messaging services  | [docs.rs](https://docs.rs/phymes-server/latest) | [README](phymes-server-readme) |
| [`phymes-app`] | Frontend UI for dynamically interacting with the Agentic AI hypergraph messaging services  | [docs.rs](https://docs.rs/phymes-app/latest) | [README](phymes-app-readme) |

[`phymes-core`]: https://crates.io/phymes-core/arrow
[`phymes-agents`]: https://crates.io/crates/phymes-agents
[`phymes-server`]: https://crates.io/crates/phymes-server
[`phymes-app`]: https://crates.io/crates/phymes-app
[arrow-rs-object-store repository]: https://github.com/apache/arrow-rs-object-store

<!--- ANCHOR_END: repository --->

## Roadmap

1. More examples for running hypergraph algorithms and simulators using `phymes-core`, and production agentic AI examples e.g., NVIDIA RAG [Blue Print](https://github.com/NVIDIA-AI-Blueprints/rag) within `phymes-agent`.
2. Improved GPU accelerated ETL operators including joins and aggregations [see](https://arxiv.org/pdf/2312.00720)
3. Better test coverage of `phymes-server` and `phymes-app` which also require a refactor particularly of `phymes-app` components
4. Proper application database and sign-in user journey
5. Better OpenAI (and non-OpenAI) API token service coverage e.g. [rust-genai], support for building Model Context Provider ([MCP]) e.g. [rust-sdk], and integrations with other external databases e.g. [rig]
6. See [issues] for more...

[rust-genai]: https://github.com/jeremychone/rust-genai
[MCP]: https://modelcontextprotocol.io/specification
[rust-sdk]: https://github.com/modelcontextprotocol/rust-sdk
[rig]: https://github.com/0xPlaygrounds/rig

## Community

The best place to engage with the Biom8er phymes community is on [GitHub Discussions][discussions]. New features and bug fix requests should be submitted via [GitHub issues][issues] which acts as the system of record for development. Design and more technical discussions should also take place on GitHub issues.

[issues]: https://github.com/apache/arrow-rs/issues
[discussions]: https://github.com/apache/arrow-rs/discussions