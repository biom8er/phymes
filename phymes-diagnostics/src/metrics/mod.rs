//! Metrics for recording information about execution

mod baseline;
mod builder;
mod common;
mod instant;
mod value;

use parking_lot::Mutex;
use std::{
    borrow::Cow, fmt::{Debug, Display}, sync::Arc
};

pub use common::{HashMap, HashSet};

// public exports
pub use baseline::{BaselineMetrics, RecordOutput};
pub use builder::MetricBuilder;
pub use value::{Count, Gauge, Metric, ScopedTimerGuard, Time, Timestamp};
pub use instant::create_timestamp_micros;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::{TimeZone, Utc};

    use super::*;

    #[test]
    fn test_display_no_labels_with_span() {
        let count = Count::new();
        count.add(44);
        let value = Metric::OutputRows(count);
        let metric = Metric::new(value, None, None, "1", 1);

        assert_eq!("output_rows{span_name=1}=44", metric.to_string())
    }

    #[test]
    fn test_display_labels_and_span() {
        let count = Count::new();
        count.add(66);
        let value = Metric::OutputRows(count);
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
        let expected_sum = Metric::Count {
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
            .filter(|metric| matches!(metric.value(), Metric::ElapsedCompute(_)))
            .collect::<Vec<_>>();
        assert_eq!(elapsed_computes.len(), 1);
        assert_eq!(elapsed_computes[0].value().as_usize(), 12 + 34 + 56);
        assert_eq!(elapsed_computes[0].span_name(), "");
        assert_eq!(elapsed_computes[0].span_id(), &0);

        // output rows should
        let output_rows = aggregated
            .iter()
            .filter(|metric| matches!(metric.value(), Metric::OutputRows(_)))
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
                matches!(metric.value(), Metric::StartTimestamp(_))
                    && metric.labels().is_empty()
            })
            .collect::<Vec<_>>();
        assert_eq!(ts.len(), 1);
        match ts.remove(0).value() {
            Metric::StartTimestamp(ts) => {
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
                matches!(metric.value(), Metric::EndTimestamp(_)) && metric.labels().is_empty()
            })
            .collect::<Vec<_>>();
        assert_eq!(ts.len(), 1);
        match ts.remove(0).value() {
            Metric::EndTimestamp(ts) => {
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
