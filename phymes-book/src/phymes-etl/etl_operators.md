# ETL Operators
## Synopsis

This tutorial describes how the [Candle](https://github.com/huggingface/candle) `Tensor` class can provide GPU accelerated ETL operations such as sorting, joining, and group aggregation to build powerful tools and complete ETL pipelines that can be integrated with agentic AI. A `etl` executable that provides ETL operators over tabular data via the command line is provided in the [examples](https://github.com/biom8er/phymes/blob/main/phymes-etl/examples/etl/main.rs).

## Tutorial
### Tensor operations

The `Tensor` class combined with Arrow's `Compute` library provides the primitives for select, sort, join, and aggregate operations with CPU and GPU accelerated that can be combined into complete ETL pipelines over columnar tables. Custom operations such as document chunking required for document RAG can also be created. Operations are either Unary or Binary, and composed into complex execution graphs analogous to database query plans that operate over colunar tables. All available functions are wrapped into a unified interface that supports tool calling with agents.

#### WASM compatibility

Tensor operations are supported in WASM with simd128 vectorization acceleration when supported by the CPU.