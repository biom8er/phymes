use std::sync::Arc;

use crate::{
    metrics::{HashMap, SpanMetricsSet}, schemas::metrics::{create_metrics_batch, create_metrics_mermaid_gantt_batch}, session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::table_trait::{Table, TableBuilderTrait, TableTrait}
};
use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt64Array},
    compute::{
        kernels::numeric::{add, sub},
        min,
    },
};

/// Get the metrics for multiple sessions as a pivot table
/// 
/// # Notes
/// * Aggregation is over the `span_id` and NOT `span_name` which should uniquely identify the span
/// * Aggregation is also over the `metric_name`
pub fn get_metrics_as_pivot_table(
    metrics_vec: &[SpanMetricsSet],
    table_name: &str,
) -> Result<Table> {
    // extract out values from metrics
    let mut span_metrics_count: HashMap<(String, String), usize> = HashMap::new();
    let mut parent_names_vec = Vec::<String>::new();
    let mut parent_ids_vec = Vec::<u64>::new();
    let mut span_names_vec = Vec::<String>::new();
    let mut span_ids_vec = Vec::<u64>::new();
    let mut ids_vec = Vec::<u64>::new();
    let mut metric_names_vec = Vec::<String>::new();
    let mut metric_values_vec = Vec::<u64>::new();
    for metrics in metrics_vec.iter() {
        for metric in metrics.clone_inner().iter() {
            // Count the number of unique span and metric combinations
            let span_name = metric.span_name().to_string();
            let span_id = metric.span_id().to_owned();
            let metric_name = metric.value().name().to_string();
            if let Some(count) =
                span_metrics_count.get_mut(&(span_name.clone(), metric_name.clone()))
            {
                *count += 1;
            } else {
                span_metrics_count.insert((span_name.clone(), metric_name.clone()), 1);
            }

            // Record the span name, metric name, and value
            parent_names_vec.push(metric.parent_name().clone().unwrap_or_default());
            parent_ids_vec.push(metric.parent_id().unwrap_or_default().to_owned());
            span_names_vec.push(span_name);
            span_ids_vec.push(span_id);
            ids_vec.push(metric.id().to_owned());
            metric_names_vec.push(metric_name);
            metric_values_vec.push(metric.value().as_usize() as u64);
        }
    }

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
    let span_ids: ArrayRef = Arc::new(UInt64Array::from(
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
    let parent_ids: ArrayRef = Arc::new(UInt64Array::from(
        unique_span_names
            .iter()
            .map(|(_, _, _, id)| id.to_owned().to_owned())
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("parent_id", parent_ids));

    // Extract the metric values for each unique metric name and span name
    for metric_name in unique_metric_names.iter() {
        let mut pivot_metric_values = Vec::<u64>::new();
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
        let metric_values: ArrayRef = Arc::new(UInt64Array::from(pivot_metric_values));
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
pub fn get_metrics_as_table(metrics_vec: &[SpanMetricsSet], table_name: &str) -> Result<Table> {
    // extract out values from metrics
    let mut span_names_vec = Vec::<String>::new();
    let mut span_ids_vec = Vec::<u64>::new();
    let mut parent_names_vec = Vec::<String>::new();
    let mut parent_ids_vec = Vec::<u64>::new();
    let mut ids_vec = Vec::<u64>::new();
    let mut metric_names_vec = Vec::<String>::new();
    let mut metric_values_vec = Vec::<u64>::new();
    for metrics in metrics_vec.iter() {
        for metric in metrics.clone_inner().iter() {
            span_names_vec.push(metric.span_name().to_string());
            span_ids_vec.push(metric.span_id().to_owned());
            parent_names_vec.push(metric.parent_name().clone().unwrap_or_default());
            parent_ids_vec.push(metric.parent_id().unwrap_or_default().to_owned());
            ids_vec.push(metric.id().to_owned());
            metric_names_vec.push(metric.value().name().to_string());
            metric_values_vec.push(metric.value().as_usize() as u64);
        }
    }

    // create the record batch
    let batch = create_metrics_batch(
        span_names_vec, 
        metric_names_vec, 
        parent_names_vec, 
        span_ids_vec, 
        ids_vec, 
        parent_ids_vec, 
        metric_values_vec)?;

    // create the table
    Table::get_builder()
        .with_name(table_name)
        .with_record_batches(vec![batch])?
        .build()
}

/// Add normalized start and end time for use in gantt or barplot visualizations
pub fn get_metrics_as_gantt_table(pivot_table: Table, table_name: &str,) -> Result<Table> {
    // determine the minimum start time
    let start_time_arr: ArrayRef = pivot_table.get_column_as_array("start_timestamp");
    let start_time_arr_prim = start_time_arr
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let min_start_time = min(start_time_arr_prim).unwrap();
    let min_start_time_arr: ArrayRef = Arc::new(UInt64Array::from_value(
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
    // let span_id = pivot_table.get_column_as_vec_primitive::<u64>("span_id")?;
    let parent_name = pivot_table.get_column_as_vec_str("parent_name");
    // let parent_id = pivot_table.get_column_as_vec_primitive::<u64>("parent_id")?;
    // let id = pivot_table.get_column_as_vec_primitive::<u64>("id")?;
    let start_time_norm = pivot_table.get_column_as_vec_primitive::<u64>("start_time_norm")?;
    let end_time_norm = pivot_table.get_column_as_vec_primitive::<u64>("end_time_norm")?;
    let elapsed_compute = pivot_table.get_column_as_vec_primitive::<u64>("elapsed_compute")?;
    let output_rows = pivot_table.get_column_as_vec_primitive::<u64>("output_rows")?;
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
