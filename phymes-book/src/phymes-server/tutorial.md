# Tutorial

### Security recommendations

`phymes-server` should be built in `--release` mode for in production. When built in `debug` mode, the CORS security is relaxed to enable interactive debugging of the `phymes-app`.

### Performance recommendations

It is strongly recommend to enable CUDA and CuDNN for NVIDIA GPU acceleration when running token and tensor services locally. If a GPU is not available for running token services, it is strongly recommended to use an API such as OpenAI, NVIDIA NIMs, etc with your API access key passed as an environmental variable instead. Native acceleration for Intel and Apple chipsets are enabled by default when detected. SIMD128 vector instructions for WASM runtimes are enabled by default.

### WASM compatibility

Phymes-server can be built for the wasm32-wasip2 target (without support for serving HTML and without encryption but with support for APIs) by specifying the "tokio_unstable" feature flag. See discussion https://github.com/tokio-rs/tokio/discussions/6526.

```bash
RUSTFLAGS="--cfg tokio_unstable" cargo build --release -p phymes-server --features wasip2 --target wasm32-wasip2
wasmtime run -S tcp-listen=127.0.0.1:4000 ./wasi-server.wasm
```

Unfortunately, it appears that the latest version of Tokio does not build for target `wasm32-wasip2`. 

WASI HTTP can be used using `wasmtime serve` to forward requests to Wasm components. See example https://github.com/sunfishcode/hello-wasi-http/. However, this has not been implemented yet.

Phymes-server can also be built for the wasm32-unknown-unknown target (ang without without support for serving HTML and without encryption but with support for APIs) for serverless applications. See example https://github.com/tokio-rs/axum/blob/main/examples/simple-router-wasm/src/main.rs.