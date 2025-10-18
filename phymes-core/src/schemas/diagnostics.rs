use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray}, compute::{kernels::numeric::{add, sub}, min}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use phymes_diagnostics::{Diagnostics, DiagnosticsType, JSONObjectTrait};
use serde::{Deserialize, Serialize};

use crate::{schemas::available_subjects::{AvailableSubjects, AvailableSubjectsTrait}, session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, session_context::SessionContextTableNames}, table::table_trait::{Table, TableBuilderTrait, TableTrait}};

pub fn create_span_fields() -> Vec<Field> {
    let field_names = ["span_name", "parent_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["span_id", "parent_id"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt64, false))
        .collect::<Vec<_>>());
    fields_vec
}

pub fn create_current_context_fields() -> Vec<Field> {
    let field_names = ["file", "thread", "function"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["line"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    let field_names = ["timestamp"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>());
    fields_vec
}

pub fn create_diagnostic_span_fields() -> Vec<Field> {
    let field_names = ["labels"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["id"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>());
    fields_vec.extend(create_span_fields());
    fields_vec.extend(create_current_context_fields());
    fields_vec
}

pub fn create_metrics_fields() -> Fields {
    let field_names = ["metric_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["metric_value"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>());
    fields_vec.extend(create_diagnostic_span_fields());
    Fields::from(fields_vec)
}

pub fn create_metrics_pivot_fields_vec() -> Vec<Field> {
    let field_names = ["start_timestamp", "end_timestamp", "elapsed_compute", "output_rows"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>();
    fields_vec
}

pub fn create_metrics_pivot_fields() -> Fields {
    let mut fields_vec = create_diagnostic_span_fields();
    fields_vec.extend(create_metrics_pivot_fields_vec());
    Fields::from(fields_vec)
}

pub fn create_metrics_pivot_norm_time_fields() -> Fields {
    let field_names = ["start_time_norm", "end_time_norm", "duration"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>();
    fields_vec.extend(create_diagnostic_span_fields());
    fields_vec.extend(create_metrics_pivot_fields_vec());
    Fields::from(fields_vec)
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

pub fn create_traces_fields() -> Fields {
    let field_names = ["tracer_type", "tracer_event", "message_name", "subject_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec.extend(create_diagnostic_span_fields());
    Fields::from(fields_vec)
}

pub fn create_events_fields() -> Fields {
    let field_names = ["event_level", "record_name", "record_value"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec.extend(create_diagnostic_span_fields());
    Fields::from(fields_vec)
}

/// Pivot the metrics table
pub fn pivot_metrics_table(table: Table, table_name: &str) -> Result<Table> {

    // extract out values from metrics
    let span_names_vec = table.get_column_as_vec_nonprimitive::<String>("span_name")?;
    let span_ids_vec = table.get_column_as_vec_primitive::<i64>("span_id")?;
    let parent_names_vec = table.get_column_as_vec_nonprimitive::<String>("parent_name")?;
    let parent_ids_vec = table.get_column_as_vec_primitive::<i64>("parent_id")?;
    let metric_names_vec = table.get_column_as_vec_nonprimitive::<String>("metric_name")?;
    let metric_values_vec = table.get_column_as_vec_primitive::<i64>("metric_value")?;

    // find the unique metric names
    let mut unique_metric_names: Vec<String> = metric_names_vec
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_metric_names.sort();

    // find the unique span names
    let unique_span_names_hashset = span_names_vec
        .iter()
        .zip(span_ids_vec.iter())
        .zip(parent_names_vec.iter())
        .zip(parent_ids_vec.iter())
        .map(|(((a, b), c), d)| (a, b, c, d))
        .collect::<std::collections::HashSet<_>>();
    let mut unique_span_names = unique_span_names_hashset
        .iter()
        .collect::<Vec<_>>();
    unique_span_names.sort_by(|a, b| a.0.cmp(&b.0));
    unique_span_names.sort_by(|a, b| a.2.cmp(&b.2));

    // create the pivot table columns and initialize with span names and metric IDs
    let mut pivot_columns = Vec::new();
    let span_names: ArrayRef = Arc::new(StringArray::from(
        unique_span_names
            .iter()
            .map(|(name, _, _, _)| name.to_string())
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("span_name", span_names));
    let span_ids: ArrayRef = Arc::new(Int64Array::from(
        unique_span_names
            .iter()
            .map(|(_, id, _, _)| id.to_owned().to_owned())
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("span_id", span_ids));
    let parent_names: ArrayRef = Arc::new(StringArray::from(
        unique_span_names
            .iter()
            .map(|(_, _, name, _)| name.to_string())
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("parent_name", parent_names));
    let parent_ids: ArrayRef = Arc::new(Int64Array::from(
        unique_span_names
            .iter()
            .map(|(_, _, _, id)| id.to_owned().to_owned())
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("parent_id", parent_ids));

    // Extract the metric values for each unique metric name and span name
    for metric_name in unique_metric_names.iter() {
        let mut pivot_metric_values = Vec::<i64>::new();
        for (span_name, span_id, parent_name, parent_id) in unique_span_names.iter() {
            // find the matching metric and span name
            let mut found = false;
            for i in 0..span_names_vec.len() {
                if metric_names_vec.get(i).unwrap() == metric_name
                    && span_names_vec.get(i).unwrap() == *span_name
                    && span_ids_vec.get(i).unwrap() == *span_id
                    && parent_names_vec.get(i).unwrap() == *parent_name
                    && parent_ids_vec.get(i).unwrap() == *parent_id
                {
                    pivot_metric_values.push(metric_values_vec.get(i).unwrap().to_owned());
                    found = true;
                    break;
                }
            }
            if !found {
                pivot_metric_values.push(0); // default value if not found
            }
        }

        // create the named array for this metric
        let metric_values: ArrayRef = Arc::new(Int64Array::from(pivot_metric_values));
        pivot_columns.push((metric_name, metric_values));
    }

    // create the record batch
    let batch = RecordBatch::try_from_iter(pivot_columns)?;

    // create the table
    Table::get_builder()
        .with_name(table_name)
        .with_record_batches(vec![batch])?
        .build()
}

/// Get the metrics for a single session as a table
pub fn from_diagnostics_to_tables(diagnostics_vec: &[Diagnostics]) -> Result<(Option<Table>, Option<Table>, Option<Table>)> {

    // Extract out the diagnostics and partition into metrics, traces, and events
    let mut metrics_vec = Vec::new();
    let mut traces_vec = Vec::new();
    let mut events_vec = Vec::new();
    for diagnostics in diagnostics_vec {
        metrics_vec.extend(diagnostics.clone_inner().filter_by_diagnostic_type(DiagnosticsType::Metric).to_json_object());
        traces_vec.extend(diagnostics.clone_inner().filter_by_diagnostic_type(DiagnosticsType::Trace).to_json_object());
        events_vec.extend(diagnostics.clone_inner().filter_by_diagnostic_type(DiagnosticsType::Event).to_json_object());
    }

    // Wrap the metrics, traces, and events into tables
    let metrics_table = if metrics_vec.is_empty() {
        None
    } else {
        let values = metrics_vec.into_iter().map(|o| serde_json::Value::from(o)).collect::<Vec<_>>();
        let table = Table::get_builder()
            .with_name(SessionContextTableNames::Metrics.to_string().as_str())
            .with_schema(AvailableSubjects::Metrics.to_schema())
            .with_json_values(&values)?
            .build()?;
        Some(table)
    };
    let traces_table = if traces_vec.is_empty() {
        None
    } else {
        let values = traces_vec.into_iter().map(|o| serde_json::Value::from(o)).collect::<Vec<_>>();
        let table = Table::get_builder()
            .with_name(SessionContextTableNames::Traces.to_string().as_str())
            .with_schema(AvailableSubjects::Traces.to_schema())
            .with_json_values(&values)?
            .build()?;
        Some(table)
    };
    let events_table = if events_vec.is_empty() {
        None
    } else {
        let values = events_vec.into_iter().map(|o| serde_json::Value::from(o)).collect::<Vec<_>>();
        let table = Table::get_builder()
            .with_name(SessionContextTableNames::Events.to_string().as_str())
            .with_schema(AvailableSubjects::Events.to_schema())
            .with_json_values(&values)?
            .build()?;
        Some(table)
    };
    Ok((metrics_table, traces_table, events_table))
}

/// Add normalized start and end time for use in gantt or barplot visualizations
pub fn get_metrics_as_gantt_table(pivot_table: Table, table_name: &str,) -> Result<Table> {
    // determine the minimum start time
    let start_time_arr: ArrayRef = pivot_table.get_column_as_array("start_timestamp");
    let start_time_arr_prim = start_time_arr
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let min_start_time = min(start_time_arr_prim).unwrap();
    let min_start_time_arr: ArrayRef = Arc::new(Int64Array::from_value(
        min_start_time,
        start_time_arr.len(),
    ));

    // normalize the start time
    let normalized_start_time_arr = sub(&start_time_arr, &min_start_time_arr).unwrap();

    // normalize the end time
    let end_time_arr: ArrayRef = pivot_table.get_column_as_array("end_timestamp");
    let duration_arr = sub(&end_time_arr, &start_time_arr).unwrap();
    let normalized_end_time_arr = add(&normalized_start_time_arr, &duration_arr).unwrap();

    // add the start_time_norm and end_time_norm columns to the table
    let mut batch_vec = vec![
        ("start_time_norm", normalized_start_time_arr),
        ("end_time_norm", normalized_end_time_arr),
    ];
    let schema = pivot_table.get_schema();
    for field in schema.fields().iter() {
        batch_vec.push((field.name(), pivot_table.get_column_as_array(field.name())));
    }
    let batch = RecordBatch::try_from_iter(batch_vec)?;
    Table::get_builder()
        .with_name(table_name)
        .with_record_batches(vec![batch])?
        .build()
}

/// export the metrics as a mermaid gantt chart
///
/// # Notes
///
/// * chart 1: Processor traces based on normalized start and end times
/// * chart 2 and 3: Elapsed compute and output rows, respectively, as barcharts
pub fn get_metrics_as_mermaid_gantt(pivot_table: Table) -> Result<Table> {
    // initialize the diagram headers and vecs
    let header_str = "gantt\n\tdateFormat\tx\n\taxisFormat\t%s\n\ttitle\t".to_string();
    let mut processor_traces_vec = vec![
        header_str.to_string(),
        "Processor Traces\n\n\tsection Traces[ns]\n".to_string(),
    ];
    let mut elapsed_compute_vec = vec![
        header_str.to_string(),
        "Elapsed compute\n\n\tsection Time[ns]\n".to_string(),
    ];
    let mut output_rows_vec = vec![
        header_str.to_string(),
        "Row count\n\n\tsection Counts\n".to_string(),
    ];

    // extract the gantt data
    let span_name = pivot_table.get_column_as_vec_str("span_name");
    // let span_id = pivot_table.get_column_as_vec_primitive::<i64>("span_id")?;
    let parent_name = pivot_table.get_column_as_vec_str("parent_name");
    // let parent_id = pivot_table.get_column_as_vec_primitive::<i64>("parent_id")?;
    // let id = pivot_table.get_column_as_vec_primitive::<i64>("id")?;
    let start_time_norm = pivot_table.get_column_as_vec_primitive::<i64>("start_time_norm")?;
    let end_time_norm = pivot_table.get_column_as_vec_primitive::<i64>("end_time_norm")?;
    let elapsed_compute = pivot_table.get_column_as_vec_primitive::<i64>("elapsed_compute")?;
    let output_rows = pivot_table.get_column_as_vec_primitive::<i64>("output_rows")?;
    let combined = span_name
        .iter()
        .zip(parent_name.iter())
        .zip(start_time_norm.iter())
        .zip(end_time_norm.iter())
        .zip(elapsed_compute.iter())
        .zip(output_rows.iter())
        .map(|(((((a, b), c), d), e), f)| (a, b, c, d, e, f))
        .collect::<Vec<_>>();

    // create the gantt script lines
    for (sn, pn, stn, etn, ec, or) in combined {
        processor_traces_vec.push(format!("\t{sn}-{pn}:{stn},\t{etn}\n"));
        elapsed_compute_vec.push(format!("\t{sn}-{pn}:0,\t{ec}\n"));
        output_rows_vec.push(format!("\t{sn}-{pn}:0,\t{or}\n"));
    }

    // make the final strings
    let processor_traces = processor_traces_vec.join("");
    let elapsed_compute = elapsed_compute_vec.join("");
    let output_rows = output_rows_vec.join("");

    // create the record batch
    let batch = create_metrics_mermaid_gantt_batch(vec![processor_traces], vec![elapsed_compute], vec![output_rows])?;

    // create the table
    Table::get_builder()
        .with_name(pivot_table.get_name())
        .with_record_batches(vec![batch])?
        .build()
}
