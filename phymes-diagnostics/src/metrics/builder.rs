//! Builder for creating arbitrary metrics

use std::borrow::Cow;

use crate::{diagnostics::{AvailableDiagnostics, DiagnosticBuilder, DiagnosticBuilderTrait}, metrics::BaselineMetrics};

use super::{Count, Gauge, Metric, Time, Timestamp};

/// Trait extension constructing metrics, counters, timers, etc.
pub trait MetricBuilderTrait: DiagnosticBuilderTrait {
    /// Consume self and create a new counter for recording output rows
    fn output_rows(self, line: u32, file: &str, function: &str) -> Count;

    /// Consume self and create a new counter for recording the number of spills
    /// triggered by an operator
    fn spill_count(self, line: u32, file: &str, function: &str) -> Count;

    /// Consume self and create a new counter for recording the total spilled bytes
    /// triggered by an operator
    fn spilled_bytes(self, line: u32, file: &str, function: &str) -> Count;    

    /// Consume self and create a new counter for recording the total spilled rows
    /// triggered by an operator
    fn spilled_rows(self, line: u32, file: &str, function: &str) -> Count;

    /// Consume self and create a new gauge for reporting current memory usage
    fn mem_used(self, line: u32, file: &str, function: &str) -> Gauge;    

    /// Consumes self and creates a new [`Count`] for recording a
    /// metric of an overall operator (not per task)
    fn counter(self, counter_name: impl Into<Cow<'static, str>>, line: u32, file: &str, function: &str) -> Count;

    /// Consumes self and creates a new [`Gauge`] for reporting a
    /// metric of an overall operator (not per task)
    fn gauge(self, gauge_name: impl Into<Cow<'static, str>>, line: u32, file: &str, function: &str) -> Gauge;

    /// Consume self and create a new Timer for recording the elapsed
    /// CPU time spent by an operator
    fn elapsed_compute(self, line: u32, file: &str, function: &str) -> Time;

    /// Consumes self and creates a new Timer for recording some
    /// subset of an operators execution time.
    fn subset_time(self, subset_name: impl Into<Cow<'static, str>>, line: u32, file: &str, function: &str) -> Time;    

    /// Consumes self and creates a new Timestamp for recording the
    /// starting time of execution for a task
    fn start_timestamp(self, line: u32, file: &str, function: &str) -> Timestamp;

    /// Consumes self and creates a new Timestamp for recording the
    /// ending time of execution for a task
    fn end_timestamp(self, line: u32, file: &str, function: &str) -> Timestamp;

    /// Consumes self and crease a new [BaselineMetrics]
    fn baseline_metrics(self, line: u32, file: &str, function: &str) -> BaselineMetrics;
}

impl MetricBuilderTrait for DiagnosticBuilder {
    fn output_rows(self, line: u32, file: &str, function: &str) -> Count {
        let count = Count::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::OutputRows(count.clone()));
        self.build(&diagnostic, line, file, function);
        count
    }

    fn spill_count(self, line: u32, file: &str, function: &str) -> Count {
        let count = Count::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::SpillCount(count.clone()));
        self.build(&diagnostic, line, file, function);
        count
    }

    fn spilled_bytes(self, line: u32, file: &str, function: &str) -> Count {
        let count = Count::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::SpilledBytes(count.clone()));
        self.build(&diagnostic, line, file, function);
        count
    }

    fn spilled_rows(self, line: u32, file: &str, function: &str) -> Count {
        let count = Count::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::SpilledRows(count.clone()));
        self.build(&diagnostic, line, file, function);
        count
    }

    fn mem_used(self, line: u32, file: &str, function: &str) -> Gauge {
        let gauge = Gauge::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::CurrentMemoryUsage(gauge.clone()));
        self.build(&diagnostic, line, file, function);
        gauge
    }

    fn counter(self, counter_name: impl Into<Cow<'static, str>>, line: u32, file: &str, function: &str) -> Count {
        let count = Count::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::Count {
            name: counter_name.into(),
            count: count.clone(),
        });
        self.build(&diagnostic, line, file, function);
        count
    }

    fn gauge(self, gauge_name: impl Into<Cow<'static, str>>, line: u32, file: &str, function: &str) -> Gauge {
        let gauge = Gauge::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::Gauge {
            name: gauge_name.into(),
            gauge: gauge.clone(),
        });
        self.build(&diagnostic, line, file, function);
        gauge
    }

    fn elapsed_compute(self, line: u32, file: &str, function: &str) -> Time {
        let time = Time::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::ElapsedCompute(time.clone()));
        self.build(&diagnostic, line, file, function);
        time
    }

    fn subset_time(self, subset_name: impl Into<Cow<'static, str>>, line: u32, file: &str, function: &str) -> Time {
        let time = Time::new();

        let diagnostic = AvailableDiagnostics::Metric(Metric::Time {
            name: subset_name.into(),
            time: time.clone(),
        });
        self.build(&diagnostic, line, file, function);
        time
    }

    fn start_timestamp(self, line: u32, file: &str, function: &str) -> Timestamp {
        let timestamp = Timestamp::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::StartTimestamp(timestamp.clone()));
        self.build(&diagnostic, line, file, function);
        timestamp
    }

    fn end_timestamp(self, line: u32, file: &str, function: &str) -> Timestamp {
        let timestamp = Timestamp::new();
        let diagnostic = AvailableDiagnostics::Metric(Metric::EndTimestamp(timestamp.clone()));
        self.build(&diagnostic, line, file, function);
        timestamp
    }

    fn baseline_metrics(self, line: u32, file: &str, function: &str) -> BaselineMetrics {
        let start_time = self.clone().start_timestamp(line, file, function);
        let end_time = self.clone().end_timestamp(line, file, function);
        let elapsed_compute = self.clone().elapsed_compute(line, file, function);
        let output_rows = self.output_rows(line, file, function);
        BaselineMetrics::new(start_time, end_time, elapsed_compute, output_rows)
    }
}