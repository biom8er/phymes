use criterion::{Criterion, criterion_group, criterion_main};
use phymes_agents::{candle_assets::device::device, candle_ops::{ops_config::{CandleOpsConfig, CandleOpsStreamManager}, ops_which::WhichCandleOps}};

fn benchmark_candle_ops_processor(c: &mut Criterion) {
    // Cases for dataset sizes
    // xs = 1, s = 100, m = 1e3, l = 1e6, xl = 1e12
    let data_vec = [
        ("xs-lhs-xs-rhs", ""),
        ("xs-lhs-s-rhs", ""),
        ("xs-lhs-m-rhs", ""),
        ("xs-lhs-l-rhs", ""),
        ("xs-lhs-xl-rhs", ""),
        ("s-lhs-xs-rhs", ""),
        ("s-lhs-s-rhs", ""),
        ("s-lhs-m-rhs", ""),
        ("s-lhs-l-rhs", ""),
        ("s-lhs-xl-rhs", ""),
        ("m-lhs-xs-rhs", ""),
        ("m-lhs-s-rhs", ""),
        ("m-lhs-m-rhs", ""),
        ("m-lhs-l-rhs", ""),
        ("m-lhs-xl-rhs", ""),
        ("l-lhs-xs-rhs", ""),
        ("l-lhs-s-rhs", ""),
        ("l-lhs-m-rhs", ""),
        ("l-lhs-l-rhs", ""),
        ("l-lhs-xl-rhs", ""),
        ("xl-lhs-xs-rhs", ""),
        ("xl-lhs-s-rhs", ""),
        ("xl-lhs-m-rhs", ""),
        ("xl-lhs-l-rhs", ""),
        ("xl-lhs-xl-rhs", ""),
    ];

    // Cases for stream and accumulation options
    let stream_vec = [
        CandleOpsStreamManager::AccumulateLHSStreamRHS,
        CandleOpsStreamManager::AccumulateLHSAccumulateRHS,
        CandleOpsStreamManager::StreamLHSStreamRHS,
        CandleOpsStreamManager::StreamLHSAccumulateRHS,
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

    // Benchmark each case sequentially
    for (data_size, data) in data_vec.iter() {
        for stream in stream_vec.iter() {
            // Create a unique identifier for the benchmark
            let id = format!("rel-sim-score_{data_size}_{}_{gpu}_{candle}", stream.get_name());

            // Build the input messages

            // Build the ops config
            let config = CandleOpsConfig {
                stream: stream.to_owned(),
                which: WhichCandleOps::RelativeSimilarityScore,
                ..Default::default()
            };
            // ...

            // Build the metrics

            // Build the runtime environment

            c.bench_function(id.as_str(), |b| {
                b.iter(|| {
                    #[cfg(feature = "wasip2")]
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap();
                    #[cfg(not(feature = "wasip2"))]
                    let rt = tokio::runtime::Runtime::new().unwrap();

                    // Make the stream and run
                    let ops_stream = CandleOpStream::new(
                        messages,
                        config_table.clone().to_record_batch_stream(),
                        Arc::clone(&runtime_env),
                        baseline_metrics,
                    ).unwrap();
                    let _result = rt.block_on(async {
                        ops_stream.try_collect::<Vec<_>>().await
                    });
                });
            });
        }
    }
}

criterion_group!(
    benches,
    benchmark_candle_ops_processor,
);
criterion_main!(benches);