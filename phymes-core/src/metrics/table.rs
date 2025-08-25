use std::sync::Arc;

use crate::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    table::arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
};
use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt64Array},
    compute::{
        kernels::numeric::{add, sub},
        min,
    },
};

/// Get the metrics for multiple sessions as a table
pub fn get_metrics_as_pivot_table(
    metrics_vec: &[ArrowTaskMetricsSet],
    table_name: &str,
) -> Result<ArrowTable> {
    // extract out values from metrics
    let mut task_metrics_count: HashMap<(String, String), usize> = HashMap::new();
    let mut task_names_vec = Vec::<(String, usize)>::new();
    let mut metric_names_vec = Vec::<String>::new();
    let mut metric_values_vec = Vec::<u64>::new();
    for metrics in metrics_vec.iter() {
        for metric in metrics.clone_inner().iter() {
            // Count the number of unique task and metric combinations
            let task_name = metric.task().as_ref().unwrap().to_string();
            let metric_name = metric.value().name().to_string();
            if let Some(count) =
                task_metrics_count.get_mut(&(task_name.clone(), metric_name.clone()))
            {
                *count += 1;
            } else {
                task_metrics_count.insert((task_name.clone(), metric_name.clone()), 1);
            }

            // Record the task name, metric name, and value
            task_names_vec.push((
                task_name.clone(),
                *task_metrics_count
                    .get(&(task_name.clone(), metric_name.clone()))
                    .unwrap(),
            ));
            metric_names_vec.push(metric_name);
            metric_values_vec.push(metric.value().as_usize() as u64);
        }
    }

    // find the unique metric names and task names
    let mut unique_metric_names: Vec<String> = metric_names_vec
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_metric_names.sort();
    let mut unique_task_names: Vec<(String, usize)> = task_names_vec
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_task_names.sort_by(|a, b| a.1.cmp(&b.1));
    unique_task_names.sort_by(|a, b| a.0.cmp(&b.0));

    // create the pivot table columns and initialize with task names and replicate counts
    let mut pivot_columns = Vec::new();
    let task_names: ArrayRef = Arc::new(StringArray::from(
        unique_task_names
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("task_name", task_names));
    let replicate_couns: ArrayRef = Arc::new(UInt64Array::from(
        unique_task_names
            .iter()
            .map(|(_, count)| *count as u64)
            .collect::<Vec<_>>(),
    ));
    pivot_columns.push(("replicate_count", replicate_couns));

    // Extract the metric values for each unique metric name and task name
    for metric_name in unique_metric_names.iter() {
        let mut pivot_metric_values = Vec::<u64>::new();
        for (task_name, replicate_count) in unique_task_names.iter() {
            // find the matching metric and task name
            let mut found = false;
            for i in 0..task_names_vec.len() {
                if metric_names_vec.get(i).unwrap() == metric_name
                    && task_names_vec.get(i).unwrap().0 == *task_name
                    && task_names_vec.get(i).unwrap().1 == *replicate_count
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
    ArrowTable::get_builder()
        .with_name(table_name)
        .with_record_batches(vec![batch])?
        .build()
}

/// Get the metrics for a single session as a table
pub fn get_metrics_as_table(metrics: ArrowTaskMetricsSet, table_name: &str) -> Result<ArrowTable> {
    // extract out values from metrics
    let mut task_names_vec = Vec::<String>::new();
    let mut metric_names_vec = Vec::<String>::new();
    let mut metric_values_vec = Vec::<u64>::new();
    // let mut metrics_sorted = metrics.clone_inner().iter().map(|m| Arc::clone(m)).collect::<Vec<_>>();
    // metrics_sorted.sort_by(|a, b| a.task().as_ref().unwrap().cmp(b.task().as_ref().unwrap()));
    // metrics_sorted.sort_by(|a, b| a.value().name().to_string().cmp(&b.value().name().to_string()));
    for metric in metrics.clone_inner().iter() {
        task_names_vec.push(metric.task().as_ref().unwrap().to_string());
        metric_names_vec.push(metric.value().name().to_string());
        metric_values_vec.push(metric.value().as_usize() as u64);
    }

    if let Some(val) = metrics.clone_inner().elapsed_compute() {
        task_names_vec.push("All".to_string());
        metric_names_vec.push("elapsed_compute".to_string());
        metric_values_vec.push(val as u64);
    }

    if let Some(val) = metrics.clone_inner().output_rows() {
        task_names_vec.push("All".to_string());
        metric_names_vec.push("output_rows".to_string());
        metric_values_vec.push(val as u64);
    }

    // create the record batch
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names_vec));
    let metric_names: ArrayRef = Arc::new(StringArray::from(metric_names_vec));
    let metric_values: ArrayRef = Arc::new(UInt64Array::from(metric_values_vec));
    let batch = RecordBatch::try_from_iter(vec![
        ("task_name", task_names),
        ("metric_name", metric_names),
        ("metric_value", metric_values),
    ])?;

    // create the table
    ArrowTable::get_builder()
        .with_name(table_name)
        .with_record_batches(vec![batch])?
        .build()
}

/// Add normalized start and end time for use in gantt or barplot visualizations
pub fn get_metrics_as_gantt_table(pivot_table: ArrowTable) -> Result<ArrowTable> {
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
    ArrowTable::get_builder()
        .with_name(pivot_table.get_name())
        .with_record_batches(vec![batch])?
        .build()
}

/// export the metrics as a mermaid gantt chart
///
/// # Notes
///
/// * chart 1: Processor traces based on normalized start and end times
/// * chart 2 and 3: Elapsed compute and output rows, respectively, as barcharts
pub fn get_metrics_as_mermaid_gantt(pivot_table: ArrowTable) -> Result<ArrowTable> {
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
    let task_name = pivot_table.get_column_as_vec_str("task_name");
    let replicate_count = pivot_table.get_column_as_vec_primitive::<u64>("replicate_count")?;
    let start_time_norm = pivot_table.get_column_as_vec_primitive::<u64>("start_time_norm")?;
    let end_time_norm = pivot_table.get_column_as_vec_primitive::<u64>("end_time_norm")?;
    let elapsed_compute = pivot_table.get_column_as_vec_primitive::<u64>("elapsed_compute")?;
    let output_rows = pivot_table.get_column_as_vec_primitive::<u64>("output_rows")?;
    let combined = task_name
        .iter()
        .zip(replicate_count.iter())
        .zip(start_time_norm.iter())
        .zip(end_time_norm.iter())
        .zip(elapsed_compute.iter())
        .zip(output_rows.iter())
        .map(|(((((a, b), c), d), e), f)| (a, b, c, d, e, f))
        .collect::<Vec<_>>();

    // create the gantt script lines
    for (tn, rc, stn, etn, ec, or) in combined {
        processor_traces_vec.push(format!("\t{tn}-{rc}:{stn},\t{etn}\n"));
        elapsed_compute_vec.push(format!("\t{tn}-{rc}:0,\t{ec}\n"));
        output_rows_vec.push(format!("\t{tn}-{rc}:0,\t{or}\n"));
    }

    // make the final strings
    let processor_traces = processor_traces_vec.join("");
    let elapsed_compute = elapsed_compute_vec.join("");
    let output_rows = output_rows_vec.join("");

    // create the record batch
    let processor_traces: ArrayRef = Arc::new(StringArray::from(vec![processor_traces]));
    let elapsed_compute: ArrayRef = Arc::new(StringArray::from(vec![elapsed_compute]));
    let output_rows: ArrayRef = Arc::new(StringArray::from(vec![output_rows]));
    let batch = RecordBatch::try_from_iter(vec![
        ("processor_traces", processor_traces),
        ("elapsed_compute", elapsed_compute),
        ("output_rows", output_rows),
    ])?;

    // create the table
    ArrowTable::get_builder()
        .with_name(pivot_table.get_name())
        .with_record_batches(vec![batch])?
        .build()
}
