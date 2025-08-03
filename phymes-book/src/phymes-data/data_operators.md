# Data Operators
## Synopsis

This tutorial describes how the [Candle](https://github.com/huggingface/candle) `Tensor` class can provide GPU accelerated Data operations such as sorting, joining, and group aggregation to build powerful tools and complete Data pipelines that can be integrated with agentic AI. A `etl` executable that provides Data operators over tabular data via the command line is provided in the [examples](https://github.com/biom8er/phymes/blob/main/phymes-data/examples/etl/main.rs).

## Tutorial
### Tensor operations

The `Tensor` class combined with Arrow's `Compute` library provides the primitives for select, sort, join, and aggregate operations with CPU and GPU accelerated that can be combined into complete Data pipelines over columnar tables. Custom operations such as document chunking required for document RAG can also be created. Operations are either Unary or Binary, and composed into complex execution graphs analogous to database query plans that operate over colunar tables. All available functions are wrapped into a unified interface that supports tool calling with agents.

The following primitive and non-primitive Rust types are supported for all data operations: u8, u32, i64, f32, f64, and String. Candle `Tensor` also supports bf16 and f16 types, but these are not yet fully supported by Apache `Arrow` at the time of writing. In addition, nested types (either fixed sized list or variable sized lists) in combination with the supported primitive and non-primitive Rust types are also supported e.g., `Vec<Vec<f32>>` or `Vec<Vec<u32>>` that are often used for embeddings.

#### WASM compatibility

Tensor operations are supported in WASM with simd128 vectorization acceleration when supported by the CPU.