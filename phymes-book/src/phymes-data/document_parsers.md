# Document Parsers
## Synopsis

This tutorial describes how the [lopdf](https://github.com/J-F-Liu/lopdf) crate provides PDF parsing functionality that can be combined with Data pipelines and Agentic AI. A `pdf` executable that provides PDF parsing is provided in the [examples](https://github.com/biom8er/phymes/blob/main/phymes-data/examples/etl/main.rs).

## Tutorial
### PDF parsers

Text is extracted from PDF documents and returned as columnar tables following the schema used for document processing in the `phymes-agents` session plans. Simple cleaning of the extracted text is provided by default, e.g., removing extra white spaces and joining lines and paragraphs. While the sequence of text is maintained, the hierarchy is lost and would require a proper OCR solution such as the NVIDIA NIMS [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse/modelcard) or [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) which would also provide image and table parsing and can be hosted as microservices and called via API in the future.

#### WASM compatibility

Text extraction from PDF documents is supported in WASM with simd128 vectorization acceleration when supported by the CPU.