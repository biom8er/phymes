//! Metrics for recording information about execution

mod baseline;
mod builder;
mod common;
mod instant;
mod table;
mod value;

use anyhow::Result;
use parking_lot::Mutex;
use std::{
    borrow::Cow, fmt::{Debug, Display}, sync::Arc
};

pub use common::{HashMap, HashSet};

// public exports
pub use baseline::{BaselineMetrics, RecordOutput};
pub use builder::MetricBuilder;
pub use table::{
    get_metrics_as_gantt_table, get_metrics_as_mermaid_gantt, get_metrics_as_pivot_table,
    get_metrics_as_table,
};
pub use value::{Count, Gauge, MetricValue, ScopedTimerGuard, Time, Timestamp};

/// Something that tracks a value of interest (metric)
///
/// Typically [Metric]s are not created directly, but instead
/// are created using [MetricBuilder] or methods on
/// [SpanMetricsSet].
///
/// ```
///  use phymes_core::metrics::*;
///
///  let metrics = SpanMetricsSet::new();
///  assert!(metrics.clone_inner().output_rows().is_none());
///
///  // Create a counter to increment using the MetricBuilder
///  let output_rows = MetricBuilder::new(&metrics)
///      .output_rows("1");
///
///  // Counter can be incremented
///  output_rows.add(13);
///
///  // The value can be retrieved directly:
///  assert_eq!(output_rows.value(), 13);
///
///  // As well as from the metrics set
///  assert_eq!(metrics.clone_inner().output_rows(), Some(13));
/// ```

#[derive(Debug)]
pub struct Metric {
    /// The value of the metric
    value: MetricValue,

    /// arbitrary name=value pairs identifying this metric
    labels: Vec<Label>,

    /// A unique ID identifying this metric
    id: u64,

    /// The parent span name of execution
    parent_name: Option<String>,

    /// The parent id name of execution
    parent_id: Option<u64>,

    /// To which span of execution does these metrics apply?
    span_name: String,

    /// A unique ID identifying the span that this metric
    /// is a part of
    span_id: u64,
}

impl Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value.name())?;

        let mut iter = [self.span_name.to_string()]
            .into_iter()
            .map(|span_name| Label::new("span_name", span_name))
            .chain(self.labels().iter().cloned())
            .peekable();

        // print out the labels specially
        if iter.peek().is_some() {
            write!(f, "{{")?;

            let mut is_first = true;
            for i in iter {
                if !is_first {
                    write!(f, ", ")?;
                } else {
                    is_first = false;
                }

                write!(f, "{i}")?;
            }

            write!(f, "}}")?;
        }

        // and now the value
        write!(f, "={}", self.value)
    }
}

impl Metric {
    /// Create a new [`Metric`]. Consider using [`MetricBuilder`]
    /// rather than this function directly.
    pub fn new(value: MetricValue, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64) -> Self {
        Self {
            value,
            labels: vec![],
            parent_name: parent_name.map(String::from),
            parent_id,
            span_name: span_name.to_string(),
            span_id,
            id: create_random_id().unwrap(),
        }
    }

    /// Create a new [`Metric`]. Consider using [`MetricBuilder`]
    /// rather than this function directly.
    pub fn new_with_labels(value: MetricValue, parent_name: Option<&str>, parent_id: Option<u64>, span_name: &str, span_id: u64, labels: Vec<Label>) -> Self {
        Self {
            value,
            labels,
            parent_name: parent_name.map(String::from),
            parent_id,
            span_name: span_name.to_string(),
            span_id,
            id: create_random_id().unwrap(),
        }
    }

    /// Add a new label to this metric
    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    /// What labels are present for this metric?
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Return a reference to the value of this metric
    pub fn value(&self) -> &MetricValue {
        &self.value
    }

    /// Return a mutable reference to the value of this metric
    pub fn value_mut(&mut self) -> &mut MetricValue {
        &mut self.value
    }

    /// Return a reference to the parent name
    pub fn parent_name(&self) -> &Option<String> {
        &self.parent_name
    }

    /// Return a reference to the span ID
    pub fn parent_id(&self) -> &Option<u64> {
        &self.parent_id
    }

    /// Return a reference to the span name
    pub fn span_name(&self) -> &str {
        &self.span_name
    }

    /// Return a reference to the span ID
    pub fn span_id(&self) -> &u64 {
        &self.span_id
    }

    /// Return a reference to the ID
    pub fn id(&self) -> &u64 {
        &self.id
    }

}

/// A snapshot of the metrics for a particular ([Processor]).
///
/// [Processor]: crate::task::processor::Processor
#[derive(Default, Debug, Clone)]
pub struct MetricsSet {
    metrics: Vec<Arc<Metric>>,    
}

impl MetricsSet {
    /// Create a new container of metrics
    pub fn new() -> Self {
        MetricsSet::default()
    }

    /// Add the specified metric
    pub fn push(&mut self, metric: Arc<Metric>) {
        self.metrics.push(metric)
    }

    /// Returns an iterator across all metrics
    pub fn iter(&self) -> impl Iterator<Item = &Arc<Metric>> {
        self.metrics.iter()
    }

    /// Sort the order of metrics so the "most useful" show up first
    pub fn sorted_for_display(mut self) -> Self {
        self.metrics.sort_unstable_by_key(|metric| {
            (
                metric.value().display_sort_key(),
                metric.value().name().to_owned(),
            )
        });
        self
    }

    /// Remove all timestamp metrics (for more compact display)
    pub fn timestamps_removed(self) -> Self {
        let Self { metrics } = self;

        let metrics = metrics
            .into_iter()
            .filter(|m| !m.value.is_timestamp())
            .collect::<Vec<_>>();

        Self { metrics }
    }
}

impl Display for MetricsSet {
    /// Format the [`MetricsSet`] as a single string
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut is_first = true;
        for i in self.metrics.iter() {
            if !is_first {
                write!(f, ", ")?;
            } else {
                is_first = false;
            }

            write!(f, "{i}")?;
        }
        Ok(())
    }
}

/// A set of [Metric]s for an individual "operator" (e.g. `&dyn
/// Processor`).
///
/// This structure is intended as a convenience for [Processor]
/// implementations so they can generate different streams for multiple
/// processes but easily report them together.
///
/// Each `clone()` of this structure will add metrics to the same
/// underlying metrics set
///
/// [Processor]: crate::task::processor::Processor
#[derive(Default, Debug, Clone)]
pub struct SpanMetricsSet {
    inner: Arc<Mutex<MetricsSet>>,
}

impl SpanMetricsSet {
    /// Create a new empty shared metrics set
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MetricsSet::new())),
        }
    }

    /// Add the specified metric to the underlying metric set
    pub fn register(&self, metric: Arc<Metric>) {
        self.inner.lock().push(metric)
    }

    /// Return a clone of the inner [`MetricsSet`]
    pub fn clone_inner(&self) -> MetricsSet {
        let guard = self.inner.lock();
        (*guard).clone()
    }

    /// Clear the metrics
    pub fn clear(&mut self) {
        self.inner.try_lock().unwrap().metrics.clear();
    }
}

/// `name=value` pairs identifying a metric. This concept is called various things
/// in various different systems:
///
/// "labels" in
/// [prometheus](https://prometheus.io/docs/concepts/data_model/) and
/// "tags" in
/// [InfluxDB](https://docs.influxdata.com/influxdb/v1.8/write_protocols/line_protocol_tutorial/)
/// , "attributes" in [open
/// telemetry]<https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/metrics/datamodel.md>,
/// etc.
///
/// As the name and value are expected to mostly be constant strings,
/// use a [`Cow`] to avoid copying / allocations in this common case.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    name: Cow<'static, str>,
    value: Cow<'static, str>,
}

impl Label {
    /// Create a new [`Label`]
    pub fn new(name: impl Into<Cow<'static, str>>, value: impl Into<Cow<'static, str>>) -> Self {
        let name = name.into();
        let value = value.into();
        Self { name, value }
    }

    /// Returns the name of this label
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Returns the value of this label
    pub fn value(&self) -> &str {
        self.value.as_ref()
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}

/// Create a (pseudo)random ID
pub fn create_random_id() -> Result<u64> {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf)?;
    let id = u64::from_ne_bytes(buf);
    Ok(id)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::{TimeZone, Utc};

    use super::*;

    #[test]

    #[test]
    fn test_display_no_labels_with_span() {
        let count = Count::new();
        count.add(44);
        let value = MetricValue::OutputRows(count);
        let metric = Metric::new(value, None, None, "1", 1);

        assert_eq!("output_rows{span=1}=44", metric.to_string())
    }

    #[test]
    fn test_display_labels_and_span() {
        let count = Count::new();
        count.add(66);
        let value = MetricValue::OutputRows(count);
        let label = Label::new("foo", "bar");
        let metric = Metric::new_with_labels(value,  None, None, "1", 1, vec![label]);

        assert_eq!("output_rows{span=1, foo=bar}=66", metric.to_string())
    }

    #[test]
    fn test_sorted_for_display() {
        let metrics = SpanMetricsSet::new();
        MetricBuilder::new(&metrics).end_timestamp(None, None, "", 0);
        MetricBuilder::new(&metrics).start_timestamp(None, None, "", 0);
        MetricBuilder::new(&metrics).elapsed_compute(None, None, "", 0);
        MetricBuilder::new(&metrics).counter("the_second_counter", None, None, "", 0);
        MetricBuilder::new(&metrics).counter("the_counter", None, None, "", 0);
        MetricBuilder::new(&metrics).counter("the_third_counter", None, None, "", 0);
        MetricBuilder::new(&metrics).subset_time("the_time", None, None, "", 0);
        MetricBuilder::new(&metrics).output_rows(None, None, "", 0);
        let metrics = metrics.clone_inner();

        fn metric_names(metrics: &MetricsSet) -> String {
            let n = metrics.iter().map(|m| m.value().name()).collect::<Vec<_>>();
            n.join(", ")
        }

        assert_eq!(
            "end_timestamp, start_timestamp, elapsed_compute, the_second_counter, the_counter, the_third_counter, the_time, output_rows",
            metric_names(&metrics)
        );

        let metrics = metrics.sorted_for_display();
        assert_eq!(
            "output_rows, elapsed_compute, the_counter, the_second_counter, the_third_counter, the_time, start_timestamp, end_timestamp",
            metric_names(&metrics)
        );
    }
}
