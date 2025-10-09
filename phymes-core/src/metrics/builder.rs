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
///  let metrics = ArrowTaskMetricsSet::new();
///  let partition = 1;
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

    /// Optional partition id
    partition_id: Option<usize>,

    /// arbitrary name=value pairs identifying this metric
    labels: Vec<Label>,
}

impl<'a> MetricBuilder<'a> {
    /// Create a new `MetricBuilder` that will register the result of `build()` with the `metrics`
    pub fn new(metrics: &'a SpanMetricsSet) -> Self {
        Self {
            metrics,
            partition_id: None,
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

    /// Set the task of the metric being constructed
    pub fn with_partition_id(mut self, partition_id: usize) -> Self {
        self.partition_id = Some(partition_id);
        self
    }

    /// Consume self and create a metric of the specified value
    /// registered with the MetricsSet
    pub fn build(self, value: MetricValue) {
        let Self {
            labels,
            partition_id,
            metrics,
        } = self;
        let metric = Arc::new(Metric::new_with_labels(value, partition_id, labels));
        metrics.register(metric);
    }

    /// Consume self and create a new counter for recording output rows
    pub fn output_rows(self, partition_id: usize) -> Count {
        let count = Count::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::OutputRows(count.clone()));
        count
    }

    /// Consume self and create a new counter for recording the number of spills
    /// triggered by an operator
    pub fn spill_count(self, partition_id: usize) -> Count {
        let count = Count::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::SpillCount(count.clone()));
        count
    }

    /// Consume self and create a new counter for recording the total spilled bytes
    /// triggered by an operator
    pub fn spilled_bytes(self, partition_id: usize) -> Count {
        let count = Count::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::SpilledBytes(count.clone()));
        count
    }

    /// Consume self and create a new counter for recording the total spilled rows
    /// triggered by an operator
    pub fn spilled_rows(self, partition_id: usize) -> Count {
        let count = Count::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::SpilledRows(count.clone()));
        count
    }

    /// Consume self and create a new gauge for reporting current memory usage
    pub fn mem_used(self, partition_id: usize) -> Gauge {
        let gauge = Gauge::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::CurrentMemoryUsage(gauge.clone()));
        gauge
    }

    /// Consumes self and creates a new [`Count`] for recording some
    /// arbitrary metric of an operator.
    pub fn counter(self, counter_name: impl Into<Cow<'static, str>>, partition_id: usize) -> Count {        
        self.with_partition_id(partition_id)
            .global_counter(counter_name)
    }

    /// Consumes self and creates a new [`Gauge`] for reporting some
    /// arbitrary metric of an operator.
    pub fn gauge(self, gauge_name: impl Into<Cow<'static, str>>, partition_id: usize) -> Gauge {
        self.with_partition_id(partition_id)
            .global_gauge(gauge_name)
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
    pub fn elapsed_compute(self, partition_id: usize) -> Time {
        let time = Time::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::ElapsedCompute(time.clone()));
        time
    }

    /// Consumes self and creates a new Timer for recording some
    /// subset of an operators execution time.
    pub fn subset_time(self, subset_name: impl Into<Cow<'static, str>>, partition_id: usize) -> Time {
        let time = Time::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::Time {
            name: subset_name.into(),
            time: time.clone(),
        });
        time
    }

    /// Consumes self and creates a new Timestamp for recording the
    /// starting time of execution for a task
    pub fn start_timestamp(self, partition_id: usize) -> Timestamp {
        let timestamp = Timestamp::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::StartTimestamp(timestamp.clone()));
        timestamp
    }

    /// Consumes self and creates a new Timestamp for recording the
    /// ending time of execution for a task
    pub fn end_timestamp(self, partition_id: usize) -> Timestamp {
        let timestamp = Timestamp::new();
        self.with_partition_id(partition_id)
            .build(MetricValue::EndTimestamp(timestamp.clone()));
        timestamp
    }
}
