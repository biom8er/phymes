use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use parking_lot::Mutex;
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    session::{
        common_traits::{device, BuildableTrait, BuilderTrait},
        runtime_env::RuntimeEnv,
        session_context::get_metrics_as_pivot_table,
    },
    table::{
        arrow_table::{
            test_table::TestTableSizes, ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait
        },
        arrow_table_publish::ArrowTablePublish, arrow_table_subscribe::{AllTableNamesSubscribe, SubscribeTrait},
    },
    task::arrow_message::{
        ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
        ArrowOutgoingMessageTrait,
    },
};
use phymes_data::{
    candle_data::{
        data_config::{DataConfig, DataStreamManager},
        data_processor::CandleDataProcessor,
        tensor_service::CandleTensorService,
    },
    candle_operators::which_operator::WhichCandleOperator,
};

fn benchmark_candle_ops_processor(c: &mut Criterion) {
    // Cases for dataset sizes
    // xs = 1, s = 100, m = 1e3, l = 1e6, xl = 1e12
    let data_size_vec = [
        // ("xs", "xs"),
        // ("xs", "s"),
        // ("xs", "m"),
        // ("s", "xs"),
        ("s", "s"),
        // ("s", "m"),
        // ("m", "m"),
    ];

    // Cases for stream and accumulation options
    let stream_vec = [
        // CandleOpsStreamManager::AccumulateLHSStreamRHS,
        DataStreamManager::AccumulateLHSAccumulateRHS,
        // CandleOpsStreamManager::StreamLHSStreamRHS,
        // CandleOpsStreamManager::StreamLHSAccumulateRHS,
    ];

    // Cases for the ops functions
    let ops_configs_vec = [
        DataConfig {
            which: WhichCandleOperator::RelativeSimilarityScore,
            lhs_pk: "id".to_string(),
            lhs_fk: "title".to_string(),
            lhs_values: "embedding".to_string(),
            rhs_pk: Some("id".to_string()),
            rhs_fk: Some("title".to_string()),
            rhs_values: Some("embedding".to_string()),
            ..Default::default()
        },
        DataConfig {
            which: WhichCandleOperator::SortColumnAndIndices,
            lhs_pk: "id".to_string(),
            lhs_fk: "title".to_string(),
            lhs_values: "score".to_string(),
            rhs_pk: Some("id".to_string()),
            rhs_fk: Some("title".to_string()),
            rhs_values: Some("score".to_string()),
            op_kwargs: Some("{\"asc\": false}".to_string()),
            ..Default::default()
        },
        // DataConfig {
        //     which: WhichCandleOperator::ChunkDocuments,
        //     lhs_pk: "id".to_string(),
        //     lhs_fk: "title".to_string(),
        //     lhs_values: "text".to_string(),
        //     rhs_pk: Some("id".to_string()),
        //     rhs_fk: Some("title".to_string()),
        //     rhs_values: Some("text".to_string()),
        //     op_kwargs: Some("{\"chunk_size\": 512, \"chunk_overlap\": 64}".to_string()),
        //     ..Default::default()
        // },
        DataConfig {
            which: WhichCandleOperator::JoinInner,
            lhs_pk: "title".to_string(),
            lhs_fk: "id".to_string(),
            rhs_pk: Some("title".to_string()),
            rhs_fk: Some("id".to_string()),
            ..Default::default()
        },
        DataConfig {
            which: WhichCandleOperator::GroupByAndAggregate,
            lhs_pk: "id".to_string(),
            lhs_fk: "id".to_string(),
            lhs_values: "[\"title\",\"collection\"]".to_string(),
            op_kwargs: Some(
                "{\"agg_columns\": [id, text, score], \"agg_operators\": [Sum, Count, Max]}"
                    .to_string(),
            ),
            ..Default::default()
        },
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
    let mut metrics_vec = Vec::new();
    for (lhs_size, rhs_size) in data_size_vec.iter() {
        let lhs_name = format!("{lhs_size}-lhs");
        let rhs_name = format!("{rhs_size}-rhs");
        for stream in stream_vec.iter() {
            for config in ops_configs_vec.iter() {
                // Build the runtime environment
                let device = device(false).unwrap();
                let service = CandleTensorService::new(device);
                let runtime_env = RuntimeEnv {
                    token_service: None,
                    tensor_service: Some(Box::new(service)),
                    name: "service".to_string(),
                    memory_limit: None,
                    time_limit: None,
                };
                let runtime_env = Arc::new(Mutex::new(runtime_env));

                // Update the config
                let mut config = config.clone();
                config.stream = stream.to_owned();
                config.lhs_name = lhs_name.to_owned();
                config.rhs_name = Some(rhs_name.to_owned());

                // Create a unique identifier for the benchmark
                let id = format!(
                    "{}_{lhs_size}-{rhs_size}_{wasm}_{gpu}_{candle}",
                    config.which.get_name(),
                );
                // let id = format!(
                //     "{}_{lhs_size}-{rhs_size}_{}_{wasm}_{gpu}_{candle}",
                //     config.which.get_name(),
                //     stream.get_name()
                // );
                let mut iter = 0;
                c.bench_function(id.as_str(), |b| {
                    b.iter(|| {
                        // Build the metrics
                        let metrics = ArrowTaskMetricsSet::new();
                        let sample_id = format!("{id}_{iter}");
                        let name = format!("ops-processor_{id}_{iter}");

                        // Build the input messages
                        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
                        let _ = messages.insert(
                            lhs_name.to_owned(),
                            ArrowOutgoingMessage::get_builder()
                                .with_name(lhs_name.as_str())
                                .with_publisher("s1")
                                .with_subject("d1")
                                .with_update(&ArrowTablePublish::None)
                                .with_message(
                                    TestTableSizes::new_from_name(lhs_size)
                                        .unwrap()
                                        .get_test_table(lhs_name.as_str())
                                        .unwrap()
                                        .to_record_batch_stream(),
                                )
                                .build()
                                .unwrap(),
                        );
                        let _ = messages.insert(
                            rhs_name.to_owned(),
                            ArrowOutgoingMessage::get_builder()
                                .with_name(rhs_name.as_str())
                                .with_publisher("s1")
                                .with_subject("d1")
                                .with_update(&ArrowTablePublish::None)
                                .with_message(
                                    TestTableSizes::new_from_name(rhs_size)
                                        .unwrap()
                                        .get_test_table(rhs_name.as_str())
                                        .unwrap()
                                        .to_record_batch_stream(),
                                )
                                .build()
                                .unwrap(),
                        );

                        // Build the ops config
                        let config_table = ArrowTable::get_builder()
                            .with_name(name.as_str())
                            .with_json(&serde_json::to_vec(&config).unwrap(), 1)
                            .unwrap()
                            .build()
                            .unwrap();
                        let _ = messages.insert(
                            name.to_owned(),
                            ArrowOutgoingMessage::get_builder()
                                .with_name(name.as_str())
                                .with_publisher("")
                                .with_subject("")
                                .with_update(&ArrowTablePublish::None)
                                .with_message(config_table.to_record_batch_stream())
                                .build()
                                .unwrap(),
                        );

                        // Handle the runtime
                        #[cfg(feature = "wasip2")]
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        #[cfg(not(feature = "wasip2"))]
                        let rt = tokio::runtime::Runtime::new().unwrap();

                        // Start the timer
                        let baseline_metrics = BaselineMetrics::new(&metrics, sample_id.as_str());
                        let timer = baseline_metrics.elapsed_compute().timer();

                        // Make the stream and run
                        let _result = rt.block_on(async {
                            let ops_processor = CandleDataProcessor::new_with_pub_sub_for(
                                name.as_str(),
                                &[ArrowTablePublish::Replace {
                                    table_name: "results".to_string(),
                                }],
                                &[],
                                &[],
                                AllTableNamesSubscribe::new_box(),
                            );
                            let mut ops_stream = ops_processor
                                .process(messages, metrics.clone(), runtime_env.clone())
                                .unwrap();
                            ops_stream
                                .remove("results")
                                .unwrap()
                                .get_message_own()
                                .try_collect::<Vec<_>>()
                                .await
                                .unwrap()
                        });

                        // Stop the timer
                        timer.done();
                        baseline_metrics.done();

                        // Collect the metrics
                        metrics_vec.push(metrics);

                        // Increment the iteration counter
                        iter += 1;
                        println!("iteration {iter}");
                    });
                });
            }
        }
    }

    // Export the metrics to CSV
    println!("exporting metrics");
    let metrics_table = get_metrics_as_pivot_table(&metrics_vec, "metrics").unwrap();
    let target_dir = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let pathname =
        format!("{target_dir}/.cache/metrics/benchmark_ops_processor_{wasm}_{gpu}_{candle}.csv");
    let path = std::path::Path::new(pathname.as_str());
    let prefix = path.parent().unwrap();
    std::fs::create_dir_all(prefix).unwrap();
    let mut file = std::fs::File::create(pathname).unwrap();
    metrics_table.to_csv_file(&mut file, b',', true).unwrap();
}

criterion_group!(benches, benchmark_candle_ops_processor,);
criterion_main!(benches);
