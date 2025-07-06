//! Metrics common for almost all operators

use std::task::Poll;

use arrow::record_batch::RecordBatch;

use super::{ArrowTaskMetricsSet, Count, MetricBuilder, Time, Timestamp};
use anyhow::Result;

/// Helper for creating and tracking common "baseline" metrics for
/// each operator
///
/// Example:
/// ```
/// use phymes_core::metrics::{BaselineMetrics, ArrowTaskMetricsSet};
/// let metrics = ArrowTaskMetricsSet::new();
///
/// let task = "2";
/// let baseline_metrics = BaselineMetrics::new(&metrics, task);
///
/// // during execution, in CPU intensive operation:
/// let timer = baseline_metrics.elapsed_compute().timer();
/// // .. do CPU intensive work
/// timer.done();
///
/// // when operator is finished:
/// baseline_metrics.done();
/// ```
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    /// end_time is set when `ExecutionMetrics::done()` is called
    end_time: Timestamp,

    /// amount of time the operator was actively trying to use the CPU
    elapsed_compute: Time,

    /// output rows: the total output rows
    output_rows: Count,
}

impl BaselineMetrics {
    /// Create a new BaselineMetric structure, and set `start_time` to now
    pub fn new(metrics: &ArrowTaskMetricsSet, task: &str) -> Self {
        let start_time = MetricBuilder::new(metrics).start_timestamp(task);
        start_time.record();

        Self {
            end_time: MetricBuilder::new(metrics).end_timestamp(task),
            elapsed_compute: MetricBuilder::new(metrics).elapsed_compute(task),
            output_rows: MetricBuilder::new(metrics).output_rows(task),
        }
    }

    /// Returns a [`BaselineMetrics`] that updates the same `elapsed_compute` ignoring
    /// all other metrics
    ///
    /// This is useful when an operator offloads some of its intermediate work to separate tasks
    /// that as a result won't be recorded by [`Self::record_poll`]
    pub fn intermediate(&self) -> BaselineMetrics {
        Self {
            end_time: Default::default(),
            elapsed_compute: self.elapsed_compute.clone(),
            output_rows: Default::default(),
        }
    }

    /// return the metric for cpu time spend in this operator
    pub fn elapsed_compute(&self) -> &Time {
        &self.elapsed_compute
    }

    /// return the metric for the total number of output rows produced
    pub fn output_rows(&self) -> &Count {
        &self.output_rows
    }

    /// Records the fact that this operator's execution is complete
    /// (recording the `end_time` metric).
    ///
    /// Note care should be taken to call `done()` manually if
    /// `BaselineMetrics` is not `drop`ped immediately upon operator
    /// completion, as async streams may not be dropped immediately
    /// depending on the consumer.
    pub fn done(&self) {
        self.end_time.record()
    }

    /// Record that some number of rows have been produced as output
    ///
    /// See the [`RecordOutput`] for conveniently recording record
    /// batch output for other thing
    pub fn record_output(&self, num_rows: usize) {
        self.output_rows.add(num_rows);
    }

    /// If not previously recorded `done()`, record
    pub fn try_done(&self) {
        if self.end_time.value().is_none() {
            self.end_time.record()
        }
    }

    /// Process a poll result of a stream producing output for an
    /// operator, recording the output rows and stream done time and
    /// returning the same poll result
    pub fn record_poll(
        &self,
        poll: Poll<Option<Result<RecordBatch>>>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if let Poll::Ready(maybe_batch) = &poll {
            match maybe_batch {
                Some(Ok(batch)) => {
                    batch.record_output(self);
                }
                Some(Err(_)) => self.done(),
                None => self.done(),
            }
        }
        poll
    }
}

impl Drop for BaselineMetrics {
    fn drop(&mut self) {
        self.try_done()
    }
}

/// Trait for things that produce output rows as a result of execution.
pub trait RecordOutput {
    /// Record that some number of output rows have been produced
    ///
    /// Meant to be composable so that instead of returning `batch`
    /// the operator can return `batch.record_output(baseline_metrics)`
    fn record_output(self, bm: &BaselineMetrics) -> Self;
}

impl RecordOutput for usize {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        bm.record_output(self);
        self
    }
}

impl RecordOutput for RecordBatch {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        bm.record_output(self.num_rows());
        self
    }
}

impl RecordOutput for &RecordBatch {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        bm.record_output(self.num_rows());
        self
    }
}

impl RecordOutput for Option<&RecordBatch> {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        if let Some(record_batch) = &self {
            record_batch.record_output(bm);
        }
        self
    }
}

impl RecordOutput for Option<RecordBatch> {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        if let Some(record_batch) = &self {
            record_batch.record_output(bm);
        }
        self
    }
}

impl RecordOutput for Result<RecordBatch> {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        if let Ok(record_batch) = &self {
            record_batch.record_output(bm);
        }
        self
    }
}
