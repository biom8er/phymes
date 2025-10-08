use std::sync::Arc;

use arrow::{array::{ArrayRef, UInt64Array, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// `span_name` will most likely be the processor name
/// `span_id` connects the trace data to the metrics data
pub fn create_metrics_fields() -> Fields {
    let field_names = ["task_name", "metric_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    // let field_names = ["span_id", "id"];
    // fields_vec.extend(field_names
    //     .iter()
    //     .map(|f| Field::new(*f, DataType::UInt32, false))
    //     .collect::<Vec<_>>());
    fields_vec.push(Field::new("metric_value", DataType::UInt64, false));
    Fields::from(fields_vec)
}

pub fn create_metrics_batch(
    task_name: Vec<String>,
    metric_name: Vec<String>,
    metric_value: Vec<u64>,
) -> Result<RecordBatch> {
    let task_name_arr: ArrayRef = Arc::new(StringArray::from(task_name));
    let metric_name_arr: ArrayRef = Arc::new(StringArray::from(metric_name));
    let metric_value_arr: ArrayRef = Arc::new(UInt64Array::from(metric_value));
    let batch = RecordBatch::try_from_iter(vec![
        ("task_name", task_name_arr),
        ("metric_name", metric_name_arr),
        ("metric_value", metric_value_arr),
    ])?;
    Ok(batch)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricSubject {
    pub task_name: String,
    pub metric_name: String,
    pub metric_value: u64,
}

pub fn create_metrics_mermaid_gantt_fields() -> Fields {
    let field_names = ["processor_traces", "elapsed_compute", "output_rows"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_metrics_mermaid_gantt_batch(
    processor_traces: Vec<String>,
    elapsed_compute: Vec<String>,
    output_rows: Vec<String>,
) -> Result<RecordBatch> {
    let processor_traces_arr: ArrayRef = Arc::new(StringArray::from(processor_traces));
    let elapsed_compute_arr: ArrayRef = Arc::new(StringArray::from(elapsed_compute));
    let output_rows_arr: ArrayRef = Arc::new(StringArray::from(output_rows));
    let batch = RecordBatch::try_from_iter(vec![
        ("processor_traces", processor_traces_arr),
        ("elapsed_compute", elapsed_compute_arr),
        ("output_rows", output_rows_arr),
    ])?;
    Ok(batch)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricMermaidGanttSubject {
    pub processor_traces: String,
    pub metric_name: String,
    pub output_rows: String,
}