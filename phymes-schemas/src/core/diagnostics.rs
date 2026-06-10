use std::{fmt::Display, sync::Arc};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Fields},
};
use phymes_diagnostics::{Diagnostics, DiagnosticsType, JSONObjectTrait};
use phymes_subject::{BuildableTrait, BuilderTrait, Subject, SubjectBuilderTrait};
use serde::{Deserialize, Serialize};

use crate::{AvailableSchemaTrait, AvailableSubjects};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum DiagnosticsVisualizations {
    /// Traces as a sequence diagram
    TraceSequenceDiagram,
    /// Events as a kanban diagram
    EventKanban,
    /// Error as a kanban diagram
    ErrorKanban,
    /// Metrics (processor traces) as a gantt chart
    #[default]
    MetricProcessorTracesGantt,
    /// Metrics (elapsed compute) as a gantt chart
    MetricElapsedComputeGantt,
    /// Metrics (output rows) as a gantt chart
    MetricOutputRowsGantt,
}

impl Display for DiagnosticsVisualizations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TraceSequenceDiagram => write!(f, "TraceSequenceDiagram"),
            Self::EventKanban => write!(f, "EventKanban"),
            Self::ErrorKanban => write!(f, "ErrorKanban"),
            Self::MetricProcessorTracesGantt => write!(f, "MetricProcessorTracesGantt"),
            Self::MetricElapsedComputeGantt => write!(f, "MetricElapsedComputeGantt"),
            Self::MetricOutputRowsGantt => write!(f, "MetricOutputRowsGantt"),
        }
    }
}

fn create_span_fields() -> Vec<Field> {
    let field_names = ["span_name", "parent_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["span_id", "parent_id"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    fields_vec
}

fn create_current_context_fields() -> Vec<Field> {
    let field_names = ["file", "thread", "function"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["line"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    let field_names = ["timestamp"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    fields_vec
}

fn create_diagnostic_span_fields() -> Vec<Field> {
    let field_names = ["labels"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["id"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    fields_vec.extend(create_span_fields());
    fields_vec.extend(create_current_context_fields());
    fields_vec
}

pub(crate) fn create_metrics_fields() -> Fields {
    let field_names = ["metric_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["metric_value"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    fields_vec.extend(create_diagnostic_span_fields());
    Fields::from(fields_vec)
}

fn create_metrics_pivot_fields_vec() -> Vec<Field> {
    vec![
        Field::new("span_name", DataType::Utf8, false),
        Field::new("span_id", DataType::Int64, false),
        Field::new("parent_name", DataType::Utf8, false),
        Field::new("parent_id", DataType::Int64, false),
        Field::new("elapsed_compute-metric_value-Sum", DataType::Int64, false),
        Field::new("end_timestamp-metric_value-Sum", DataType::Int64, false),
        Field::new("output_rows-metric_value-Sum", DataType::Int64, false),
        Field::new("start_timestamp-metric_value-Sum", DataType::Int64, false),
    ]
}

pub(crate) fn create_metrics_pivot_fields() -> Fields {
    Fields::from(create_metrics_pivot_fields_vec())
}

pub(crate) fn create_metrics_pivot_norm_time_fields() -> Fields {
    let field_names = [
        "start_timestamp-metric_value-Sum-normalized",
        "end_timestamp-metric_value-Sum-normalized",
        "duration",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Int64, false))
        .collect::<Vec<_>>();
    fields_vec.extend(create_metrics_pivot_fields_vec());
    Fields::from(fields_vec)
}

pub(crate) fn create_metrics_mermaid_gantt_fields() -> Fields {
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

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricMermaidGanttSubject {
    pub processor_traces: String,
    pub metric_name: String,
    pub output_rows: String,
}

pub(crate) fn create_traces_fields() -> Fields {
    let field_names = [
        "tracer_type",
        "tracer_event",
        "message_name",
        "subject_name",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["tracer_timestamp"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    fields_vec.extend(create_diagnostic_span_fields());
    Fields::from(fields_vec)
}

pub(crate) fn create_events_fields() -> Fields {
    let field_names = ["event_level", "record_name", "record_value"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec.extend(create_diagnostic_span_fields());
    Fields::from(fields_vec)
}

/// Get the metrics for a single session as a table
pub fn from_diagnostics_to_tables(
    diagnostics_vec: &[Diagnostics],
) -> Result<(Option<Subject>, Option<Subject>, Option<Subject>)> {
    // Extract out the diagnostics and partition into metrics, traces, and events
    let mut metrics_vec = Vec::new();
    let mut traces_vec = Vec::new();
    let mut events_vec = Vec::new();
    for diagnostics in diagnostics_vec {
        metrics_vec.extend(
            diagnostics
                .clone_inner()
                .filter_by_diagnostic_type(DiagnosticsType::Metric)
                .to_json_object(),
        );
        traces_vec.extend(
            diagnostics
                .clone_inner()
                .filter_by_diagnostic_type(DiagnosticsType::Trace)
                .to_json_object(),
        );
        events_vec.extend(
            diagnostics
                .clone_inner()
                .filter_by_diagnostic_type(DiagnosticsType::Event)
                .to_json_object(),
        );
    }

    // Wrap the metrics, traces, and events into tables
    let metrics_table = if metrics_vec.is_empty() {
        None
    } else {
        let values = metrics_vec
            .into_iter()
            .map(serde_json::Value::from)
            .collect::<Vec<_>>();
        let table = Subject::get_builder()
            .with_name(AvailableSubjects::SessionMetrics.to_string().as_str())
            .with_schema(AvailableSubjects::SessionMetrics.to_schema())
            .with_json_values(&values)?
            .build()?;
        Some(table)
    };
    let traces_table = if traces_vec.is_empty() {
        None
    } else {
        let values = traces_vec
            .into_iter()
            .map(serde_json::Value::from)
            .collect::<Vec<_>>();
        let table = Subject::get_builder()
            .with_name(AvailableSubjects::SessionTraces.to_string().as_str())
            .with_schema(AvailableSubjects::SessionTraces.to_schema())
            .with_json_values(&values)?
            .build()?;
        Some(table)
    };
    let events_table = if events_vec.is_empty() {
        None
    } else {
        let values = events_vec
            .into_iter()
            .map(serde_json::Value::from)
            .collect::<Vec<_>>();
        let table = Subject::get_builder()
            .with_name(AvailableSubjects::SessionEvents.to_string().as_str())
            .with_schema(AvailableSubjects::SessionEvents.to_schema())
            .with_json_values(&values)?
            .build()?;
        Some(table)
    };
    Ok((metrics_table, traces_table, events_table))
}
