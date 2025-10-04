use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use parking_lot::Mutex;
use phymes_core::{
    metrics::{get_metrics_as_pivot_table, ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    session::{
        common_traits::{device, BuildableTrait, BuilderTrait},
        runtime_env::RuntimeEnv,
    },
    table::{
        table_trait::{
            test_table::TestTableSizes, Table, TableBuilderTrait, TableTrait
        },
        table_publish::TablePublish,
        table_subscribe::{AllTableNamesSubscribe, SubscribeTrait, TableSubscribe},
    },
    task::{
        message::{MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage},
        processor::ProcessorTrait,
    },
};
use phymes_data::{
    candle_data::{
        data_config::{DataAggregatorOperator, DataComparatorOperator, DataComparatorPredicate, DataConfig, DataStreamManager},
        data_processor::CandleDataProcessor,
        tensor_service::CandleTensorService,
    },
    candle_operators::available_candle_operators::AvailableCandleOperators,
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
            operator: AvailableCandleOperators::RelativeSimilarityScore,
            lhs_pk: "id".to_string(),
            lhs_fk: "title".to_string(),
            lhs_values: vec!["embedding".to_string()],
            rhs_pk: Some("id".to_string()),
            rhs_fk: Some("title".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            ..Default::default()
        },
        DataConfig {
            operator: AvailableCandleOperators::SortColumnAndIndices,
            lhs_pk: "id".to_string(),
            lhs_fk: "title".to_string(),
            lhs_values: vec!["score".to_string()],
            rhs_pk: Some("id".to_string()),
            rhs_fk: Some("title".to_string()),
            rhs_values: Some(vec!["score".to_string()]),
            asc: Some(false),
            ..Default::default()
        },
        // DataConfig {
        //     which: WhichCandleOperator::ChunkDocuments,
        //     lhs_pk: "id".to_string(),
        //     lhs_fk: "title".to_string(),
        //     lhs_values: vec!["text".to_string()],
        //     rhs_pk: Some("id".to_string()),
        //     rhs_fk: Some("title".to_string()),
        //     rhs_values: Some(vec!["text".to_string()]),
        //     chunk_size: Some(512),
        //     chunk_overlap: Some(64),
        //     ..Default::default()
        // },
        DataConfig {
            operator: AvailableCandleOperators::JoinInner,
            lhs_pk: "title".to_string(),
            lhs_fk: "id".to_string(),
            rhs_pk: Some("title".to_string()),
            rhs_fk: Some("id".to_string()),
            ..Default::default()
        },
        DataConfig {
            operator: AvailableCandleOperators::GroupByAndAggregate,
            lhs_pk: "id".to_string(),
            lhs_fk: "id".to_string(),
            lhs_values: vec!["title".to_string(),"collection".to_string()],
            agg_columns: Some(vec!["id".to_string(), "text".to_string(), "score".to_string()]),
            agg_operators: Some(vec![DataAggregatorOperator::Sum, DataAggregatorOperator::Count, DataAggregatorOperator::Max]),
            ..Default::default()
        },
        DataConfig {
            operator: AvailableCandleOperators::FilterColumnsAndIndices,
            lhs_pk: "id".to_string(),
            lhs_values: vec!["title".to_string(),"id".to_string()],
            cmp_columns: Some(vec!["title".to_string(),"id".to_string()]),
            cmp_operators: Some(vec![DataComparatorOperator::Like, DataComparatorOperator::Equals]),
            cmp_predicate: Some(DataComparatorPredicate::All),
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
                    config.operator,
                );
                // let id = format!(
                //     "{}_{lhs_size}-{rhs_size}_{}_{wasm}_{gpu}_{candle}",
                //     config.operator,
                //     stream
                // );
                let mut iter = 0;
                c.bench_function(id.as_str(), |b| {
                    b.iter(|| {
                        // Build the metrics
                        let metrics = ArrowTaskMetricsSet::new();
                        let sample_id = format!("{id}_{iter}");
                        let name = format!("ops-processor_{id}_{iter}");

                        // Build the input messages
                        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
                        let _ = messages.insert(
                            lhs_name.to_owned(),
                            SendableRecordBatchStreamMessage::get_builder()
                                .with_name(lhs_name.as_str())
                                .with_publisher("s1")
                                .with_subject("d1")
                                .with_update(&TablePublish::None)
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
                            SendableRecordBatchStreamMessage::get_builder()
                                .with_name(rhs_name.as_str())
                                .with_publisher("s1")
                                .with_subject("d1")
                                .with_update(&TablePublish::None)
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
                        let config_table = Table::get_builder()
                            .with_name(name.as_str())
                            .with_json(&serde_json::to_vec(&config).unwrap(), 1)
                            .unwrap()
                            .build()
                            .unwrap();
                        let _ = messages.insert(
                            name.to_owned(),
                            SendableRecordBatchStreamMessage::get_builder()
                                .with_name(name.as_str())
                                .with_publisher("")
                                .with_subject("")
                                .with_update(&TablePublish::None)
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
                            let ops_processor = CandleDataProcessor::new_arc_with_pub_sub(
                                name.as_str(),
                                &[TablePublish::Replace {
                                    table_name: "results".to_string(),
                                }],
                                &[
                                    TableSubscribe::AlwaysFullTable {
                                        table_name: lhs_name.clone(),
                                    },
                                    TableSubscribe::AlwaysFullTable {
                                        table_name: rhs_name.clone(),
                                    },
                                ],
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
