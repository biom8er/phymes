# Model Assets
## Synopsis

This tutorial describes how the [Candle](https://github.com/huggingface/candle) model [assets](https://github.com/biom8er/phymes/blob/main/phymes-agents/src/candle_assets/candle_assets.rs) are used to support text generation inference (TGI) and text embedding inference (TEI) services that are needed for agentic AI. A `chat` model that provides TGI via the command line is provided in the [examples](https://github.com/biom8er/phymes/blob/main/phymes-agents/examples/chat/main.rs).

## Tutorial
### Candle assets

The candle assets that enable TGI and TEI include the model weights, configs, and [tokenizer](https://github.com/huggingface/tokenizers). Pytorch .bin, SafeTensor model.safetensor, and .gguf formats are supported. All assets can be downloaded using the HuggingFace [API](https://github.com/huggingface/hf-hub). `phymes-agents` provides a unified interface that hides away the nuances of different model architectures, quantizations, etc. to provide a more streamlined experience when working with different models similar to other agentic AI libraries.

#### Text generation inference (TGI)

The TGI model classes supported currently include Llama and Qwen and their quantized versions. Please reach out if other model classes are needed.

#### Text embedding inference (TEI)

The TEI model classes supported currently include BERT and QWEN and their quantized versions. Please reach out if other model classes are needed.

#### WASM compatibility

TGI, TEI, and Tensor operations are all supported in WASM with simd128 vectorization acceleration when supported by the CPU. Note that the SafeTensor format cannot be used with WASM. The maximum model weight memory cannot exceed 2 GB. The HuggingFace API can also not be used.

### OpenAI API compatible assets

Complementary to the embedded AI functionality of Candle-based assets, support for local e.g., self-hosted NVIDIA NIMs docker containers or remote e.g., OpenAI, NVIDIA NIMs, etc. OpenAI API compatible assets are also provided. Please see the contributing guide for more details on using OpenAI API compatible endpoints.

#### WASM compatibility

API requests over the wire are not supported in WASM. Projects such as [WASMCloud](https://github.com/wasmCloud/wasmCloud) and [WRPC](https://github.com/bytecodealliance/wrpc) would enable the use of OpenAI API requests in a hybrid native, cloud, and WASM context. PHYMES will look to support OpenAI API requests using WRPC in the future.