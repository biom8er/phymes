# Model Assets
## Synopsis

This tutorial describes how the [Candle](https://github.com/huggingface/candle) model [assets](../../../phymes-agents/src/session_plans/tool_agent_session.rs) are used to support text generation inference (TGI) and text embedding inference (TEI) services that are needed for agentic AI. In addition, this tutorial also describes how the Candle `Tensor` class can provide GPU accelerated ETL operations such as sorting, joining, and group aggregation to build powerful tools and complete ETL pipelines that can be integrated with agentic AI. A `chat` model that provides TGI via the command line is provided in the [examples](../../../phymes-agents/examples/chat/main.rs).

## Tutorial
### Assets

The assets that enable TGI and TEI include the model weights, configs, and [tokenizer](https://github.com/huggingface/tokenizers). Pytorch .bin, SafeTensor model.safetensor, and .gguf formats are supported. All assets can be downloaded using the HuggingFace [API](https://github.com/huggingface/hf-hub). `phymes-agents` provides a unified interface that hides away the nuances of different model architectures, quantizations, etc. to provide a more streamlined experience when working with different models similar to other agentic AI libraries.

#### Text generation inference (TGI)

The TGI model classes supported currently include Llama and Qwen and their quantized versions. Please reach out if other model classes are needed.

#### Text embedding inference (TEI)

The TEI model classes supported currently include BERT and QWEN and their quantized versions. Please reach out if other model classes are needed.

#### WASM compatibility

TGI, TEI, and Tensor operations are all supported in WASM with simd128 vectorization acceleration when supported by the CPU. Note that the SafeTensor format cannot be used with WASM. The maximum model weight memory cannot exceed 2 GB. The HuggingFace API can also not be used.

### Tensor operations

The `Tensor` class combined with Arrow's `Compute` library provides the primitives for select, sort, join, and aggregate operations with CPU and GPU accelerated that can be combined into complete ETL pipelines. Custom operations such as document chunking required for document RAG can also be created. Operations are either Unary or Binary, and composed into complex execution graphs analogous to database query plans. All available functions are wrapped into a unified interface that supports tool calling with agents.