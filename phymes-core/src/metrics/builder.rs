//! Builder for creating arbitrary metrics

use std::{borrow::Cow, sync::Arc};

use super::{SpanMetricsSet, Count, Gauge, Label, Metric, MetricValue, Time, Timestamp};

/// Structure for constructing metrics, counters, timers, etc.
///
/// Note the use of `Cow<..>` is to avoid allocations in the common
/// case of constant strings
///
/// ```rust
///  use phymes_core::metrics::*;
///
///  let metrics = SpanMetricsSet::new();
///  let span_name = "1";
///  let span_id = 1;
///
///  // Create the standard output_rows metric
///  let output_rows = MetricBuilder::new(&metrics).output_rows(partition);
///
///  // Create a operator specific counter with some labels
///  let num_bytes = MetricBuilder::new(&metrics)
///    .with_new_label("filename", "my_awesome_file.parquet")
///    .counter("num_bytes", partition);
///
/// ```
pub struct MetricBuilder<'a> {
    /// Location that the metric created by this builder will be added do
    metrics: &'a SpanMetricsSet,

    /// The parent span name of execution
    parent_name: Option<String>,

    /// The parent id name of execution
    parent_id: Option<u64>,

    /// To which span of execution does these metrics apply?
    span_name: Option<String>,

    /// A unique ID identifying the span that this metric
    /// is a part of
    span_id: Option<u64>,

    /// arbitrary name=value pairs identifying this metric
    labels: Vec<Label>,
}

impl<'a> MetricBuilder<'a> {
    /// Create a new `MetricBuilder` that will register the result of `build()` with the `metrics`
    pub fn new(metrics: &'a SpanMetricsSet) -> Self {
        Self {
            metrics,
            parent_name: None,
            parent_id: None,
            span_name: None,
            span_id: None,
            labels: vec![],
        }
    }

    /// Add a label to the metric being constructed
    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    /// Add a label to the metric being constructed
    pub fn with_new_label(
        self,
        name: impl Into<Cow<'static, str>>,
        value: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.with_label(Label::new(name.into(), value.into()))
    }

    pub fn with_parent(mut self, parent_name: &str, parent_id: u64) -> Self {
        self.parent_name = Some(parent_name.to_string());
        self.parent_id = Some(parent_id);
        self
    }

    pub fn with_span(mut self, span_name: &str, span_id: u64) -> Self {
        self.span_name = Some(span_name.to_string());
        self.span_id = Some(span_id);
        self
    }

    /// Consume self and create a metric of the specified value
    /// registered with the MetricsSet
    pub fn build(self, value: MetricValue) {
        let Self {
            labels,
            parent_name,
            parent_id,
            span_name,
            span_id,
            metrics,
        } = self;
        let metric = Arc::new(Metric::new_with_labels(value, parent_name.as_deref(), parent_id, span_name.as_deref().unwrap(), span_id.unwrap(), labels));
        metrics.register(metric);
    }

    /// Consume self and create a new counter for recording output rows
    pub fn output_rows(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Count {
        let count = Count::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::OutputRows(count.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::OutputRows(count.clone()));
        }
        count
    }

    /// Consume self and create a new counter for recording the number of spills
    /// triggered by an operator
    pub fn spill_count(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Count {
        let count = Count::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::OutputRows(count.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::OutputRows(count.clone()));
        }
        count
    }

    /// Consume self and create a new counter for recording the total spilled bytes
    /// triggered by an operator
    pub fn spilled_bytes(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Count {
        let count = Count::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::SpilledBytes(count.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::SpilledBytes(count.clone()));
        }
        count
    }

    /// Consume self and create a new counter for recording the total spilled rows
    /// triggered by an operator
    pub fn spilled_rows(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Count {
        let count = Count::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::SpilledRows(count.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::SpilledRows(count.clone()));
        }
        count
    }

    /// Consume self and create a new gauge for reporting current memory usage
    pub fn mem_used(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Gauge {
        let gauge = Gauge::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::CurrentMemoryUsage(gauge.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::CurrentMemoryUsage(gauge.clone()));
        }
        gauge
    }

    /// Consumes self and creates a new [`Count`] for recording some
    /// arbitrary metric of an operator.
    pub fn counter(self, counter_name: impl Into<Cow<'static, str>>, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Count {
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .global_counter(counter_name)
        } else {
            self.with_span(span_name, span_id)
                .global_counter(counter_name)
        }
    }

    /// Consumes self and creates a new [`Gauge`] for reporting some
    /// arbitrary metric of an operator.
    pub fn gauge(self, gauge_name: impl Into<Cow<'static, str>>, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Gauge {
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .global_gauge(gauge_name)
        } else {
            self.with_span(span_name, span_id)
                .global_gauge(gauge_name)
        }
    }

    /// Consumes self and creates a new [`Count`] for recording a
    /// metric of an overall operator (not per task)
    pub fn global_counter(self, counter_name: impl Into<Cow<'static, str>>) -> Count {
        let count = Count::new();
        self.build(MetricValue::Count {
            name: counter_name.into(),
            count: count.clone(),
        });
        count
    }

    /// Consumes self and creates a new [`Gauge`] for reporting a
    /// metric of an overall operator (not per task)
    pub fn global_gauge(self, gauge_name: impl Into<Cow<'static, str>>) -> Gauge {
        let gauge = Gauge::new();
        self.build(MetricValue::Gauge {
            name: gauge_name.into(),
            gauge: gauge.clone(),
        });
        gauge
    }

    /// Consume self and create a new Timer for recording the elapsed
    /// CPU time spent by an operator
    pub fn elapsed_compute(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Time {
        let time = Time::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::ElapsedCompute(time.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::ElapsedCompute(time.clone()));
        }
        time
    }

    /// Consumes self and creates a new Timer for recording some
    /// subset of an operators execution time.
    pub fn subset_time(self, subset_name: impl Into<Cow<'static, str>>, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Time {
        let time = Time::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::Time {
                    name: subset_name.into(),
                    time: time.clone(),
                });
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::Time {
                    name: subset_name.into(),
                    time: time.clone(),
                });
        }
        time
    }

    /// Consumes self and creates a new Timestamp for recording the
    /// starting time of execution for a task
    pub fn start_timestamp(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Timestamp {
        let timestamp = Timestamp::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::StartTimestamp(timestamp.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::StartTimestamp(timestamp.clone()));
        }
        timestamp
    }

    /// Consumes self and creates a new Timestamp for recording the
    /// ending time of execution for a task
    pub fn end_timestamp(self, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Timestamp {
        let timestamp = Timestamp::new();
        if let (Some(parent_name), Some(parent_id)) = (parent_name, parent_id) {
            self.with_parent(parent_name, parent_id)
                .with_span(span_name, span_id)
                .build(MetricValue::EndTimestamp(timestamp.clone()));
        } else {
            self.with_span(span_name, span_id)
                .build(MetricValue::EndTimestamp(timestamp.clone()));
        }
        timestamp
    }
}
