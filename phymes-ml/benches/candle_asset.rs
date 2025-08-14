use candle_core::DType;
use criterion::{Criterion, criterion_group, criterion_main};
use phymes_core::session::common_traits::{TokenProcessorTrait, TokenWrapper, device};
use phymes_ml::{
    candle_assets::available_candle_assets::{AvailableCandleAssets, load_model_asset_path, load_tokenizer},
    candle_chat::{
        chat_config::CandleChatConfig,
        chat_processor::{process_logits_sampler, process_prompt_chat},
    },
    candle_embed::embed_config::CandleEmbedConfig,
};

fn benchmark_build_candle_chat_asset(c: &mut Criterion) {
    // Cases for different chat configurations
    let config_template = CandleChatConfig {
        max_tokens: 1000,
        temperature: 0.8,
        seed: 299792458,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        ..Default::default()
    };
    let mut config_smollm2_1 = config_template.clone();
    config_smollm2_1.weights_config_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.weights_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.tokenizer_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.tokenizer_config_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.candle_asset = Some(AvailableCandleAssets::SmolLM2_135MChat);
    let mut config_smollm2_3 = config_smollm2_1.clone();
    config_smollm2_3.weights_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-f16.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    let mut config_qwen2p5_1 = config_template.clone();
    config_qwen2p5_1.weights_config_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.weights_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.tokenizer_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.tokenizer_config_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.candle_asset = Some(AvailableCandleAssets::QwenV2p5_0p5bChat);
    let mut config_qwen2p5_2 = config_qwen2p5_1.clone();
    config_qwen2p5_2.weights_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_2.candle_asset = Some(AvailableCandleAssets::QwenV2p5_3bChat);
    let mut config_qwen2p5_3 = config_qwen2p5_1.clone();
    config_qwen2p5_3.weights_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-3b-instruct-q5_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_3.candle_asset = Some(AvailableCandleAssets::QwenV2p5_3bChat);

    let config_vec = [
        config_smollm2_1,
        config_smollm2_3,
        config_qwen2p5_1,
        config_qwen2p5_2,
        config_qwen2p5_3,
    ];

    // Get the target and GPU configuration
    let wasm = if cfg!(target_arch = "wasm32") {
        "wasm"
    } else {
        "native"
    };
    let gpu = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };
    let candle = if cfg!(feature = "candle") {
        "candle"
    } else {
        "openai_api"
    };

    // Benchmark each configuration with each user content sequentially
    for config in config_vec.iter() {
        // Extract file name without path and extension
        let weight_filename = if let Some(weights_file) = config.weights_file.as_ref() {
            if let Some(file_name) = std::path::Path::new(weights_file)
                .file_stem()
                .and_then(|stem| stem.to_str())
            {
                file_name
            } else {
                config
                    .candle_asset
                    .as_ref()
                    .map_or("unknown", |a| a.get_name())
            }
        } else {
            config
                .candle_asset
                .as_ref()
                .map_or("unknown", |a| a.get_name())
        };

        // Create a unique identifier for the benchmark
        let id = format!("build-chat_{weight_filename}_{wasm}_{gpu}_{candle}");
        c.bench_function(id.as_str(), |b| {
            b.iter(|| {
                // Build the asset
                let device = device(config.cpu).unwrap();
                let _asset = config
                    .candle_asset
                    .unwrap()
                    .build(
                        config.weights_config_file.clone(),
                        config.tokenizer_file.clone(),
                        config.weights_file.clone(),
                        config.tokenizer_config_file.clone(),
                        DType::F32,
                        device,
                    )
                    .unwrap();
            });
        });
    }
}

fn benchmark_build_candle_embed_asset(c: &mut Criterion) {
    // Cases for different embed configurations
    let config_minilmv2_1 = CandleEmbedConfig {
        weights_config_file: Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        weights_file: Some(format!(
            // "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin",
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        tokenizer_file: Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        tokenizer_config_file: Some(format!(
            "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        candle_asset: Some(
            // crate::candle_assets::candle_which::WhichCandleAsset::BertEmbed,
            AvailableCandleAssets::QuantizedBertEmbed,
        ),
        ..Default::default()
    };
    let config_qwen2_1 = CandleEmbedConfig {
        weights_config_file: Some(format!(
            "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        weights_file: Some(format!(
            "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        tokenizer_file: Some(format!(
            "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer.json",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        tokenizer_config_file: Some(format!(
            "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer_config.json",
            std::env::var("HOME").unwrap_or("".to_string())
        )),
        candle_asset: Some(AvailableCandleAssets::QwenV2_1p5bEmbed),
        ..Default::default()
    };

    let config_vec = [config_minilmv2_1, config_qwen2_1];

    // Get the target and GPU configuration
    let wasm = if cfg!(target_arch = "wasm32") {
        "wasm"
    } else {
        "native"
    };
    let gpu = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };
    let candle = if cfg!(feature = "candle") {
        "candle"
    } else {
        "openai_api"
    };

    // Benchmark each configuration with each user content sequentially
    for config in config_vec.iter() {
        // Extract file name without path and extension
        let weight_filename = if let Some(weights_file) = config.weights_file.as_ref() {
            if let Some(file_name) = std::path::Path::new(weights_file)
                .file_stem()
                .and_then(|stem| stem.to_str())
            {
                file_name
            } else {
                config
                    .candle_asset
                    .as_ref()
                    .map_or("unknown", |a| a.get_name())
            }
        } else {
            config
                .candle_asset
                .as_ref()
                .map_or("unknown", |a| a.get_name())
        };

        // Create a unique identifier for the benchmark
        let id = format!("build-embed_{weight_filename}_{wasm}_{gpu}_{candle}");
        c.bench_function(id.as_str(), |b| {
            b.iter(|| {
                // Build the asset
                let device = device(config.cpu).unwrap();
                let _asset = config
                    .candle_asset
                    .unwrap()
                    .build(
                        config.weights_config_file.clone(),
                        config.tokenizer_file.clone(),
                        config.weights_file.clone(),
                        config.tokenizer_config_file.clone(),
                        DType::F32,
                        device,
                    )
                    .unwrap();
            });
        });
    }
}

fn benchmark_process_prompt_chat(c: &mut Criterion) {
    // Prompts for tool or no tool calls
    let prompt_no_tool = "\"\"\\n\\n<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"".to_string();
    let prompt_tool = "\"\"\\n<|im_start|>system\\n\\n\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\\n\\n\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n\\n\\n\\n{\"function\":{\"description\":\"Get the current weather\",\"name\":\"get_current_weather\",\"parameters\":{\"properties\":{\"format\":{\"description\":\"The temperature unit to use. Infer this from the users location.\",\"enum_values\":[\"celsius\",\"fahrenheit\"],\"type\":\"string\"},\"location\":{\"description\":\"The city and state, e.g. San Francisco, CA\",\"type\":\"string\"}},\"required\":[\"location\",\"format\"],\"type\":\"object\"}},\"type\":\"function\"}\\n\\n\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nHi!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nHello how can I help?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>user\\nWhat is Deep Learning?<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\nmagic!<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"".to_string();
    let prompts = [("no-tool", prompt_no_tool), ("tool", prompt_tool)];

    // Different tokenizers (includes embedding tokenizers!)
    let path_minilmv2: Option<String> = Some(format!(
        "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    let path_smollm2: Option<String> = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    let path_qwen2p5: Option<String> = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    let paths = [
        ("all-MiniLM-L6-v2", path_minilmv2),
        ("SmolLM2-135M", path_smollm2),
        ("Qwen2-0.5B", path_qwen2p5),
    ];
    let repo = "".to_string();
    let filename = "tokenizer.json".to_string();
    let revision = "main".to_string();
    let max_seq_length: Option<usize> = Some(2048);

    // Get the target and GPU configuration
    let wasm = if cfg!(target_arch = "wasm32") {
        "wasm"
    } else {
        "native"
    };
    let gpu = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };
    let candle = if cfg!(feature = "candle") {
        "candle"
    } else {
        "openai_api"
    };

    for (prompt_name, prompt) in prompts.iter() {
        for (path_name, path) in paths.iter() {
            // Create a unique identifier for the benchmark
            let id = format!("prompt-chat_{prompt_name}_{path_name}_{wasm}_{gpu}_{candle}");
            c.bench_function(id.as_str(), |b| {
                b.iter(|| {
                    let tokenizer =
                        load_tokenizer(load_model_asset_path(path, &repo, &filename, &revision))
                            .unwrap();
                    let (_, _, _) =
                        process_prompt_chat(prompt.clone(), &tokenizer, 1000, max_seq_length)
                            .unwrap();
                });
            });
        }
    }
}

fn benchmark_candle_chat_forward(c: &mut Criterion) {
    // Cases for different chat configurations
    let config_template = CandleChatConfig {
        max_tokens: 100,
        temperature: 0.8,
        seed: 299792458,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        ..Default::default()
    };
    let mut config_smollm2_1 = config_template.clone();
    config_smollm2_1.weights_config_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.weights_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.tokenizer_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.tokenizer_config_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_smollm2_1.candle_asset = Some(AvailableCandleAssets::SmolLM2_135MChat);
    let mut config_smollm2_3 = config_smollm2_1.clone();
    config_smollm2_3.weights_file = Some(format!(
        "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-f16.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    let mut config_qwen2p5_1 = config_template.clone();
    config_qwen2p5_1.weights_config_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.weights_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.tokenizer_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.tokenizer_config_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_1.candle_asset = Some(AvailableCandleAssets::QwenV2p5_0p5bChat);
    let mut config_qwen2p5_2 = config_qwen2p5_1.clone();
    config_qwen2p5_2.weights_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_2.candle_asset = Some(AvailableCandleAssets::QwenV2p5_3bChat);
    let mut config_qwen2p5_3 = config_qwen2p5_1.clone();
    config_qwen2p5_3.weights_file = Some(format!(
        "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-3b-instruct-q5_k_m.gguf",
        std::env::var("HOME").unwrap_or("".to_string())
    ));
    config_qwen2p5_3.candle_asset = Some(AvailableCandleAssets::QwenV2p5_3bChat);

    let config_vec = [
        config_smollm2_1,
        config_smollm2_3,
        config_qwen2p5_1,
        config_qwen2p5_2,
        config_qwen2p5_3,
    ];

    // Code generation prompt
    let prompt = "\"\"\\n\\n<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n\\n\\n\\n\\n\\n<|im_start|>user\\Write a python function to count prime numbers up to N. Please include complete docstrings as well as comments when needed. Please provide an example using the functions in the docstrings.<|im_end|>\\n\\n\\n\\n\\n<|im_start|>assistant\\n\\n\\n\"\"";

    // Get the target and GPU configuration
    let wasm = if cfg!(target_arch = "wasm32") {
        "wasm"
    } else {
        "native"
    };
    let gpu = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };
    let candle = if cfg!(feature = "candle") {
        "candle"
    } else {
        "openai_api"
    };

    // Benchmark each configuration sequentially
    for config in config_vec.iter() {
        // Extract file name without path and extension
        let weight_filename = if let Some(weights_file) = config.weights_file.as_ref() {
            if let Some(file_name) = std::path::Path::new(weights_file)
                .file_stem()
                .and_then(|stem| stem.to_str())
            {
                file_name
            } else {
                config
                    .candle_asset
                    .as_ref()
                    .map_or("unknown", |a| a.get_name())
            }
        } else {
            config
                .candle_asset
                .as_ref()
                .map_or("unknown", |a| a.get_name())
        };

        // Build the asset
        let device = device(config.cpu).unwrap();
        let mut asset = config
            .candle_asset
            .unwrap()
            .build(
                config.weights_config_file.clone(),
                config.tokenizer_file.clone(),
                config.weights_file.clone(),
                config.tokenizer_config_file.clone(),
                DType::F32,
                device,
            )
            .unwrap();

        // Build the logits processor
        let mut logits_processor =
            process_logits_sampler(config.temperature, config.seed, config.top_k, config.top_p);

        // Create the prompt tokens
        let (prompt_tokens, to_sample, mut tos) = process_prompt_chat(
            prompt.to_string(),
            &asset.tokenizer,
            config.max_tokens,
            asset.tokenizer_config.model_max_length,
        )
        .unwrap();
        println!("prompt length: {}", prompt_tokens.len());
        assert_eq!(to_sample, config.max_tokens - 1);

        // Create a unique identifier for the benchmark
        let id = format!("chat-f-prompt_{weight_filename}_{wasm}_{gpu}_{candle}");
        c.bench_function(id.as_str(), |b| {
            b.iter(|| {
                // Process the prompt
                let mut next_token = 0;
                for (pos, token) in prompt_tokens.iter().enumerate() {
                    let logits = asset
                        .forward(&TokenWrapper::D1(vec![*token]), pos, None, true)
                        .unwrap();
                    let logits = logits.squeeze(0).unwrap();
                    if pos == prompt_tokens.len() - 1 {
                        next_token = logits_processor.sample(&logits).unwrap();
                    }
                }
                let _text = tos.next_token(next_token).unwrap();
            });
        });

        // Initialize the index
        let mut index = prompt_tokens.len();

        // Create a unique identifier for the benchmark
        let id = format!("chat-f-samples_{weight_filename}_{wasm}_{gpu}_{candle}");
        c.bench_function(id.as_str(), |b| {
            b.iter(|| {
                for _iter in 0..to_sample {
                    let logits = asset
                        .forward(
                            &TokenWrapper::D1(vec![*tos.tokens().last().unwrap()]),
                            index,
                            None,
                            true,
                        )
                        .unwrap();
                    let logits = logits.squeeze(0).unwrap();
                    let logits = if config.repeat_penalty == 1. {
                        logits
                    } else {
                        let start_at = tos.tokens().len().saturating_sub(config.repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            config.repeat_penalty,
                            &tos.tokens()[start_at..],
                        )
                        .unwrap()
                    };
                    let next_token = logits_processor.sample(&logits).unwrap();
                    let _text = tos.next_token(next_token).unwrap();

                    // Update the index
                    index += 1;
                }
            });
        });
    }
}

criterion_group!(
    benches,
    benchmark_build_candle_chat_asset,
    benchmark_build_candle_embed_asset,
    benchmark_process_prompt_chat,
    benchmark_candle_chat_forward
);
criterion_main!(benches);
