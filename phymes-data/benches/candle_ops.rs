use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use phymes_core::{
    BuildableTrait, BuilderTrait, MessageBuilderTrait, ProcessorTrait, RuntimeEnv,
    SendableRecordBatchStreamMessage, Table, TableBuilderTrait, TablePublication, TableTrait,
    from_diagnostics_to_tables, test_table::TestTableSizes,
};
use phymes_data::{
    AvailableCandleOperators, CandleDataProcessor, DataAggregatorOperator, DataComparatorOperator,
    DataComparatorPredicate, DataConfig, DataStreamManager,
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, HashMap, MetricBuilderTrait,
    SpanBuilder,
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
        // (DataStreamManager::Accumulate, DataStreamManager::Stream),
        (DataStreamManager::Accumulate, DataStreamManager::Accumulate), // (DataStreamManager::Stream, DataStreamManager::Stream),
                                                                        // (DataStreamManager::Stream, DataStreamManager::Accumulate),
    ];

    // Cases for the ops functions
    let ops_configs_vec = [
        DataConfig {
            operator: AvailableCandleOperators::VectorDistance,
            lhs_pk: Some("id".to_string()),
            lhs_fk: Some("title".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("id".to_string()),
            rhs_fk: Some("title".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            ..Default::default()
        },
        DataConfig {
            operator: AvailableCandleOperators::Sort,
            lhs_pk: Some("id".to_string()),
            lhs_fk: Some("title".to_string()),
            lhs_values: Some(vec!["score".to_string()]),
            rhs_pk: Some("id".to_string()),
            rhs_fk: Some("title".to_string()),
            rhs_values: Some(vec!["score".to_string()]),
            asc: Some(false),
            ..Default::default()
        },
        // DataConfig {
        //     which: WhichCandleOperator::ChunkDocuments,
        //     lhs_pk: Some("id".to_string()),
        //     lhs_fk: Some("title".to_string()),
        //     lhs_values: vec!["text".to_string()],
        //     rhs_pk: Some("id".to_string()),
        //     rhs_fk: Some("title".to_string()),
        //     rhs_values: Some(vec!["text".to_string()]),
        //     chunk_size: Some(512),
        //     chunk_overlap: Some(64),
        //     ..Default::default()
        // },
        DataConfig {
            operator: AvailableCandleOperators::Join,
            lhs_pk: Some("title".to_string()),
            lhs_values: Some(vec!["score".to_string()]),
            rhs_pk: Some("title".to_string()),
            rhs_fk: Some("id".to_string()),
            ..Default::default()
        },
        DataConfig {
            operator: AvailableCandleOperators::GroupBy,
            lhs_pk: Some("id".to_string()),
            lhs_values: Some(vec!["title".to_string(), "collection".to_string()]),
            agg_columns: Some(vec![
                "id".to_string(),
                "text".to_string(),
                "score".to_string(),
            ]),
            agg_operators: Some(vec![
                DataAggregatorOperator::Sum,
                DataAggregatorOperator::Count,
                DataAggregatorOperator::Max,
            ]),
            ..Default::default()
        },
        DataConfig {
            operator: AvailableCandleOperators::Filter,
            lhs_pk: Some("id".to_string()),
            lhs_values: Some(vec!["title".to_string(), "id".to_string()]),
            cmp_columns: Some(vec!["title".to_string(), "id".to_string()]),
            cmp_operators: Some(vec![
                DataComparatorOperator::Like,
                DataComparatorOperator::Equals,
            ]),
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
        "api"
    };

    // Benchmark each case sequentially
    let mut metrics_vec = Vec::new();
    for (lhs_size, rhs_size) in data_size_vec.iter() {
        let lhs_name = format!("{lhs_size}-lhs");
        let rhs_name = format!("{rhs_size}-rhs");
        for stream in stream_vec.iter() {
            for config in ops_configs_vec.iter() {
                // Build the runtime environment
                let runtime_env = RuntimeEnv::get_builder().with_name("rt").build().unwrap();
                let runtime_env = Arc::new(runtime_env);

                // Update the config
                let mut config = config.clone();
                config.lhs_stream = stream.0.to_owned();
                config.rhs_stream = Some(stream.1.to_owned());
                config.lhs_name = Some(lhs_name.to_owned());
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
                        let sample_id = format!("{id}_{iter}");
                        let name = format!("ops-processor_{id}_{iter}");
                        let span = SpanBuilder::default()
                            .with_span(sample_id.as_str())
                            .build()
                            .unwrap();
                        let diagnostics = Diagnostics::new();
                        let diagnostic_builder =
                            DiagnosticBuilder::new(&diagnostics).with_span(&span);

                        // Build the input messages
                        let mut messages =
                            HashMap::<String, SendableRecordBatchStreamMessage>::new();
                        let _ = messages.insert(
                            lhs_name.to_owned(),
                            SendableRecordBatchStreamMessage::get_builder()
                                .with_name(lhs_name.as_str())
                                .with_publisher("s1")
                                .with_subject("d1")
                                .with_update(&TablePublication::None)
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
                                .with_update(&TablePublication::None)
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
                                .with_update(&TablePublication::None)
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
                        let baseline_metrics = diagnostic_builder.clone().baseline_metrics(
                            line!(),
                            file!(),
                            &sample_id,
                        );
                        let timer = baseline_metrics.elapsed_compute().timer();

                        // Make the stream and run
                        let _result = rt.block_on(async {
                            let ops_processor = CandleDataProcessor::new(name.as_str(), "");
                            let mut ops_stream = ops_processor
                                .process(messages, Some(&diagnostic_builder), runtime_env.clone())
                                .unwrap();
                            ops_stream
                                .remove("results")
                                .unwrap()
                                .message
                                .take()
                                .unwrap()
                                .try_collect::<Vec<_>>()
                                .await
                                .unwrap()
                        });

                        // Stop the timer
                        timer.done();
                        baseline_metrics.done();

                        // Collect the metrics
                        metrics_vec.push(diagnostics);

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
    let (metrics_table, _traces_table, _events_table) =
        from_diagnostics_to_tables(&metrics_vec).unwrap();
    if let Some(metrics_table) = metrics_table {
        let target_dir = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let pathname = format!(
            "{target_dir}/.cache/metrics/benchmark_ops_processor_{wasm}_{gpu}_{candle}.csv"
        );
        let path = std::path::Path::new(pathname.as_str());
        let prefix = path.parent().unwrap();
        std::fs::create_dir_all(prefix).unwrap();
        let mut file = std::fs::File::create(pathname).unwrap();
        metrics_table.to_csv_file(&mut file, b',', true).unwrap();
    }
}

criterion_group!(benches, benchmark_candle_ops_processor,);
criterion_main!(benches);
