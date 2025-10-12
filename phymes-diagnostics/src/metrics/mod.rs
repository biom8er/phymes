//! Metrics for recording information about execution

mod baseline;
mod builder;
mod common;
mod instant;
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
pub use value::{Count, Gauge, MetricValue, ScopedTimerGuard, Time, Timestamp};
pub use instant::{create_timestamp_micros, create_timestamp_str, convert_timestamp_micros_to_str};

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
///      .with_span("my_span", 0)
///      .output_rows();
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

    /// Convenience: return the number of rows produced, aggregated
    /// across tasks or `None` if no metric is present
    pub fn output_rows(&self) -> Option<usize> {
        self.sum(|metric| matches!(metric.value(), MetricValue::OutputRows(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: return the count of spills, aggregated
    /// across tasks or `None` if no metric is present
    pub fn spill_count(&self) -> Option<usize> {
        self.sum(|metric| matches!(metric.value(), MetricValue::SpillCount(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: return the total byte size of spills, aggregated
    /// across tasks or `None` if no metric is present
    pub fn spilled_bytes(&self) -> Option<usize> {
        self.sum(|metric| matches!(metric.value(), MetricValue::SpilledBytes(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: return the total rows of spills, aggregated
    /// across tasks or `None` if no metric is present
    pub fn spilled_rows(&self) -> Option<usize> {
        self.sum(|metric| matches!(metric.value(), MetricValue::SpilledRows(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: return the amount of elapsed CPU time spent,
    /// aggregated across tasks or `None` if no metric is present
    pub fn elapsed_compute(&self) -> Option<usize> {
        self.sum(|metric| matches!(metric.value(), MetricValue::ElapsedCompute(_)))
            .map(|v| v.as_usize())
    }

    /// Sums the values for metrics for which `f(metric)` returns
    /// `true`, and returns the value. Returns `None` if no metrics match
    /// the predicate.
    pub fn sum<F>(&self, mut f: F) -> Option<MetricValue>
    where
        F: FnMut(&Metric) -> bool,
    {
        let mut iter = self
            .metrics
            .iter()
            .filter(|metric| f(metric.as_ref()))
            .peekable();

        let mut accum = match iter.peek() {
            None => {
                return None;
            }
            Some(metric) => metric.value().new_empty(),
        };

        iter.for_each(|metric| accum.aggregate(metric.value()));

        Some(accum)
    }

    /// Returns the sum of all the metrics with the specified name
    /// in the returned set.
    pub fn sum_by_name(&self, metric_name: &str) -> Option<MetricValue> {
        self.sum(|m| match m.value() {
            MetricValue::Count { name, .. } => name == metric_name,
            MetricValue::Time { name, .. } => name == metric_name,
            MetricValue::OutputRows(_) => false,
            MetricValue::ElapsedCompute(_) => false,
            MetricValue::SpillCount(_) => false,
            MetricValue::SpilledBytes(_) => false,
            MetricValue::SpilledRows(_) => false,
            MetricValue::CurrentMemoryUsage(_) => false,
            MetricValue::Gauge { name, .. } => name == metric_name,
            MetricValue::StartTimestamp(_) => false,
            MetricValue::EndTimestamp(_) => false,
        })
    }

    /// Returns a new derived `MetricsSet` where all metrics
    /// that had the same name have been
    /// aggregated together. The resulting `MetricsSet` has all
    /// metrics with `span=None`
    pub fn aggregate_by_name(&self) -> Self {
        let mut map = HashMap::new();

        // There are all sorts of ways to make this more efficient
        for metric in &self.metrics {
            let key = metric.value.name();
            map.entry(key)
                .and_modify(|accum: &mut Metric| {
                    accum.value_mut().aggregate(metric.value());
                })
                .or_insert_with(|| {
                    // accumulate with no task
                    let mut accum = Metric::new(metric.value().new_empty(), None, None, "", 0);
                    accum.value_mut().aggregate(metric.value());
                    accum
                });
        }

        let new_metrics = map
            .into_iter()
            .map(|(_k, v)| Arc::new(v))
            .collect::<Vec<_>>();

        Self {
            metrics: new_metrics,
        }
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
    fn test_display_no_labels_with_span() {
        let count = Count::new();
        count.add(44);
        let value = MetricValue::OutputRows(count);
        let metric = Metric::new(value, None, None, "1", 1);

        assert_eq!("output_rows{span_name=1}=44", metric.to_string())
    }

    #[test]
    fn test_display_labels_and_span() {
        let count = Count::new();
        count.add(66);
        let value = MetricValue::OutputRows(count);
        let label = Label::new("foo", "bar");
        let metric = Metric::new_with_labels(value,  None, None, "1", 1, vec![label]);

        assert_eq!("output_rows{span_name=1, foo=bar}=66", metric.to_string())
    }

    #[test]
    fn test_output_rows() {
        let metrics = SpanMetricsSet::new();
        assert!(metrics.clone_inner().output_rows().is_none());

        let builder = MetricBuilder::new(&metrics);
        let task = 1;
        let output_rows = builder.clone().with_span(task.to_string().as_str(), task).output_rows();
        output_rows.add(13);

        let output_rows = builder.clone().with_span((task + 1).to_string().as_str(), task + 1).output_rows();
        output_rows.add(7);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 20);
    }

    #[test]
    fn test_elapsed_compute() {
        let metrics = SpanMetricsSet::new();
        assert!(metrics.clone_inner().elapsed_compute().is_none());

        let builder = MetricBuilder::new(&metrics);
        let task = 1;
        let elapsed_compute =
            builder.clone().with_span(task.to_string().as_str(), task).elapsed_compute();
        elapsed_compute.add_duration(Duration::from_nanos(1234));

        let elapsed_compute =
            builder.clone().with_span((task + 1).to_string().as_str(), task + 1).elapsed_compute();
        elapsed_compute.add_duration(Duration::from_nanos(6));
        assert_eq!(metrics.clone_inner().elapsed_compute().unwrap(), 1240);
    }

    #[test]
    fn test_sum() {
        let metrics = SpanMetricsSet::new();
        let builder = MetricBuilder::new(&metrics);

        let count1 = builder
            .clone()
            .with_span("1", 1)
            .with_new_label("foo", "bar")
            .counter("my_counter");
        count1.add(1);

        let count2 = builder
            .clone()
            .with_span("2", 2)
            .counter("my_counter");
        count2.add(2);

        let metrics = metrics.clone_inner();
        assert!(metrics.sum(|_| false).is_none());

        let expected_count = Count::new();
        expected_count.add(3);
        let expected_sum = MetricValue::Count {
            name: "my_counter".into(),
            count: expected_count,
        };

        assert_eq!(metrics.sum(|_| true), Some(expected_sum));
    }

    #[test]
    #[should_panic(expected = "Mismatched metric types. Can not aggregate Count")]
    fn test_bad_sum() {
        // can not add different kinds of metrics
        let metrics = SpanMetricsSet::new();
        let builder = MetricBuilder::new(&metrics);

        let count = builder
            .clone()
            .with_span("1", 1)
            .counter("my_metric");
        count.add(1);

        let time = builder
            .clone()
            .with_span("1", 1)
            .subset_time("my_metric");
        time.add_duration(Duration::from_nanos(10));

        // expect that this will error out
        metrics.clone_inner().sum(|_| true);
    }

    #[test]
    fn test_aggregate_by_name() {
        let metrics = SpanMetricsSet::new();

        // Note cpu_time1 has labels but it is still aggregated with metrics 2 and 3
        let elapsed_compute1 = MetricBuilder::new(&metrics)
            .with_new_label("foo", "bar")
            .with_span("1", 1)
            .elapsed_compute();
        elapsed_compute1.add_duration(Duration::from_nanos(12));

        let elapsed_compute2 = MetricBuilder::new(&metrics).with_span("2", 2).elapsed_compute();
        elapsed_compute2.add_duration(Duration::from_nanos(34));

        let elapsed_compute3 = MetricBuilder::new(&metrics).with_span("4", 4).elapsed_compute();
        elapsed_compute3.add_duration(Duration::from_nanos(56));

        let output_rows = MetricBuilder::new(&metrics).with_span("1", 1).output_rows(); // output rows
        output_rows.add(56);

        let aggregated = metrics.clone_inner().aggregate_by_name();

        // cpu time should be aggregated:
        let elapsed_computes = aggregated
            .iter()
            .filter(|metric| matches!(metric.value(), MetricValue::ElapsedCompute(_)))
            .collect::<Vec<_>>();
        assert_eq!(elapsed_computes.len(), 1);
        assert_eq!(elapsed_computes[0].value().as_usize(), 12 + 34 + 56);
        assert_eq!(elapsed_computes[0].span_name(), "");
        assert_eq!(elapsed_computes[0].span_id(), &0);

        // output rows should
        let output_rows = aggregated
            .iter()
            .filter(|metric| matches!(metric.value(), MetricValue::OutputRows(_)))
            .collect::<Vec<_>>();
        assert_eq!(output_rows.len(), 1);
        assert_eq!(output_rows[0].value().as_usize(), 56);
        assert_eq!(output_rows[0].span_name(), "");
        assert_eq!(output_rows[0].span_id(), &0);
    }

    #[test]
    #[should_panic(expected = "Mismatched metric types. Can not aggregate Count")]
    fn test_aggregate_task_bad_sum() {
        let metrics = SpanMetricsSet::new();
        let builder = MetricBuilder::new(&metrics).with_span("1", 1);

        let count = builder.clone().counter("my_metric");
        count.add(1);

        let time = builder.clone().subset_time("my_metric");
        time.add_duration(Duration::from_nanos(10));

        // can't aggregate time and count -- expect a panic
        metrics.clone_inner().aggregate_by_name();
    }

    #[test]
    fn test_aggregate_task_timestamps() {
        let metrics = SpanMetricsSet::new();

        // 1431648000000000 == 1970-01-17 13:40:48 UTC
        let t1 = Utc.timestamp_nanos(1431648000000000);
        // 1531648000000000 == 1970-01-18 17:27:28 UTC
        let t2 = Utc.timestamp_nanos(1531648000000000);
        // 1631648000000000 == 1970-01-19 21:14:08 UTC
        let t3 = Utc.timestamp_nanos(1631648000000000);
        // 1731648000000000 == 1970-01-21 01:00:48 UTC
        let t4 = Utc.timestamp_nanos(1731648000000000);

        let builder = MetricBuilder::new(&metrics).with_span("1", 1);
        let start_timestamp0 = builder.clone().start_timestamp();
        start_timestamp0.set(t1);
        let end_timestamp0 = builder.clone().end_timestamp();
        end_timestamp0.set(t2);
        let start_timestamp1 = builder.clone().start_timestamp();
        start_timestamp1.set(t3);
        let end_timestamp1 = builder.clone().end_timestamp();
        end_timestamp1.set(t4);

        // aggregate
        let aggregated = metrics.clone_inner().aggregate_by_name();

        let mut ts = aggregated
            .iter()
            .filter(|metric| {
                matches!(metric.value(), MetricValue::StartTimestamp(_))
                    && metric.labels().is_empty()
            })
            .collect::<Vec<_>>();
        assert_eq!(ts.len(), 1);
        match ts.remove(0).value() {
            MetricValue::StartTimestamp(ts) => {
                // expect earliest of t1, t2
                assert_eq!(ts.value(), Some(t1));
            }
            _ => {
                panic!("Not a timestamp");
            }
        };

        let mut ts = aggregated
            .iter()
            .filter(|metric| {
                matches!(metric.value(), MetricValue::EndTimestamp(_)) && metric.labels().is_empty()
            })
            .collect::<Vec<_>>();
        assert_eq!(ts.len(), 1);
        match ts.remove(0).value() {
            MetricValue::EndTimestamp(ts) => {
                // expect latest of t3, t4
                assert_eq!(ts.value(), Some(t4));
            }
            _ => {
                panic!("Not a timestamp");
            }
        };
    }

    #[test]
    fn test_sorted_for_display() {
        let metrics = SpanMetricsSet::new();
        MetricBuilder::new(&metrics).with_span("", 0).end_timestamp();
        MetricBuilder::new(&metrics).with_span("", 0).start_timestamp();
        MetricBuilder::new(&metrics).with_span("", 0).elapsed_compute();
        MetricBuilder::new(&metrics).with_span("", 0).counter("the_second_counter");
        MetricBuilder::new(&metrics).with_span("", 0).counter("the_counter");
        MetricBuilder::new(&metrics).with_span("", 0).counter("the_third_counter");
        MetricBuilder::new(&metrics).with_span("", 0).subset_time("the_time");
        MetricBuilder::new(&metrics).with_span("", 0).output_rows();
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
