# PHYMES: Parallel HYpergraph MEssaging Streams

Extract Transform Load (Data) crate

<!--- ANCHOR: synopsis --->

## Synopsis

The PHYMES etl crate implements the functionality for data wrangling and exploratory data analysis that are often used as function or tool calls in Agentic AI workflows. Specifically, the crate implements functionality to convert documents into columnar tables as implemented in `phymes-subject`, Data operators over columnar tables, and data visualization from columnar tables. Functionality for general tensor operations using the [candle](https://github.com/huggingface/candle) crates from [HuggingFace](https://huggingface.co/), PDF parsing using the [lopdf](https://github.com/J-F-Liu/lopdf) crate, and data visualization using the [plotly](https://github.com/plotly/plotly.rs) crate. All tensor operations can be GPU accelerated using CUDA and CuDNN.

<!--- ANCHOR_END: synopsis --->