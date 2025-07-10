# Tutorial

### Security recommendations

`phymes-server` should be built in `--release` mode for use in production. When built in `debug` mode, the CORS security is relaxed to enable interactive debugging of the `phymes-app`.

### Performance recommendations

It is strongly recommend to enable CUDA and CuDNN for NVIDIA GPU acceleration when running token and tensor services locally. If a GPU is not available for running token services, it is strongly recommended to use an API such as OpenAI, NVIDIA NIMs, etc with your API access key passed as an environmental variable instead. Native acceleration for Intel and Apple chipsets are enabled by default when detected. SIMD128 vector instructions for WASM runtimes are enabled by default.

### WASM compatibility

Phymes-server can be built for the wasm32-wasip2 and wasm32-unknown-unknown targets (without support for serving HTML and without encryption but with support for APIs). See the developer guide for instructions on running the wasm32-wasip2 CLI application. See an [example](https://github.com/tokio-rs/axum/blob/main/examples/simple-router-wasm/src/main.rs) of how to embed the wasm32-unknown-unknown library in a serverless application.

WASI [HTTP](https://github.com/sunfishcode/hello-wasi-http/) can be used in conjunction with `wasmtime serve` to forward requests to WASM components. However, this has not been implemented yet.