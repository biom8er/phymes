//! Value representation of metrics

use std::{
    fmt::Display,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::metrics::instant::Instant;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;

/// A counter to record things such as number of input or output rows
///
/// Note `clone`ing counters update the same underlying metrics
#[derive(Debug, Clone)]
pub struct Count {
    /// value of the metric counter
    value: Arc<AtomicUsize>,
}

impl PartialEq for Count {
    fn eq(&self, other: &Self) -> bool {
        self.value().eq(&other.value())
    }
}

impl Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl Default for Count {
    fn default() -> Self {
        Self::new()
    }
}

impl Count {
    /// create a new counter
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add `n` to the metric's value
    pub fn add(&self, n: usize) {
        // relaxed ordering for operations on `value` poses no issues
        // we're purely using atomic ops with no associated memory ops
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn value(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }
}

/// A gauge is the simplest metrics type. It just returns a value.
/// For example, you can easily expose current memory consumption with a gauge.
///
/// Note `clone`ing gauge update the same underlying metrics
#[derive(Debug, Clone)]
pub struct Gauge {
    /// value of the metric gauge
    value: Arc<AtomicUsize>,
}

impl PartialEq for Gauge {
    fn eq(&self, other: &Self) -> bool {
        self.value().eq(&other.value())
    }
}

impl Display for Gauge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

impl Gauge {
    /// create a new gauge
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add `n` to the metric's value
    pub fn add(&self, n: usize) {
        // relaxed ordering for operations on `value` poses no issues
        // we're purely using atomic ops with no associated memory ops
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Sub `n` from the metric's value
    pub fn sub(&self, n: usize) {
        // relaxed ordering for operations on `value` poses no issues
        // we're purely using atomic ops with no associated memory ops
        self.value.fetch_sub(n, Ordering::Relaxed);
    }

    /// Set metric's value to maximum of `n` and current value
    pub fn set_max(&self, n: usize) {
        self.value.fetch_max(n, Ordering::Relaxed);
    }

    /// Set the metric's value to `n` and return the previous value
    pub fn set(&self, n: usize) -> usize {
        // relaxed ordering for operations on `value` poses no issues
        // we're purely using atomic ops with no associated memory ops
        self.value.swap(n, Ordering::Relaxed)
    }

    /// Get the current value
    pub fn value(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }
}

/// Measure a potentially non contiguous duration of time
#[derive(Debug, Clone)]
pub struct Time {
    /// elapsed time, in nanoseconds
    nanos: Arc<AtomicUsize>,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Time {
    fn eq(&self, other: &Self) -> bool {
        self.value().eq(&other.value())
    }
}

impl Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let duration = Duration::from_nanos(self.value() as u64);
        write!(f, "{duration:?}")
    }
}

impl Time {
    /// Create a new [`Time`] wrapper suitable for recording elapsed
    /// times for operations.
    pub fn new() -> Self {
        Self {
            nanos: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add elapsed nanoseconds since `start`to self
    pub fn add_elapsed(&self, start: Instant) {
        self.add_duration(start.elapsed());
    }

    /// Add duration of time to self
    ///
    /// Note: this will always increment the recorded time by at least 1 nanosecond
    /// to distinguish between the scenario of no values recorded, in which
    /// case the value will be 0, and no measurable amount of time having passed,
    /// in which case the value will be small but not 0.
    ///
    /// This is based on the assumption that the timing logic in most cases is likely
    /// to take at least a nanosecond, and so this is reasonable mechanism to avoid
    /// ambiguity, especially on systems with low-resolution monotonic clocks
    pub fn add_duration(&self, duration: Duration) {
        let more_nanos = duration.as_nanos() as usize;
        self.nanos.fetch_add(more_nanos.max(1), Ordering::Relaxed);
    }

    /// Add the number of nanoseconds of other `Time` to self
    pub fn add(&self, other: &Time) {
        self.add_duration(Duration::from_nanos(other.value() as u64))
    }

    /// return a scoped guard that adds the amount of time elapsed
    /// between its creation and its drop or call to `stop` to the
    /// underlying metric.
    pub fn timer(&self) -> ScopedTimerGuard<'_> {
        ScopedTimerGuard {
            inner: self,
            start: Some(Instant::now()),
        }
    }

    /// Get the number of nanoseconds record by this Time metric
    pub fn value(&self) -> usize {
        self.nanos.load(Ordering::Relaxed)
    }
}

/// Stores a single timestamp, stored as the number of nanoseconds
/// elapsed from Jan 1, 1970 UTC
#[derive(Debug, Clone)]
pub struct Timestamp {
    /// Time thing started
    timestamp: Arc<Mutex<Option<DateTime<Utc>>>>,
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::new()
    }
}

impl Timestamp {
    /// Create a new timestamp and sets its value to 0
    pub fn new() -> Self {
        Self {
            timestamp: Arc::new(Mutex::new(None)),
        }
    }

    /// Sets the timestamps value to the current time
    pub fn record(&self) {
        self.set(Utc::now())
    }

    /// Sets the timestamps value to a specified time
    pub fn set(&self, now: DateTime<Utc>) {
        *self.timestamp.lock() = Some(now);
    }

    /// return the timestamps value at the last time `record()` was
    /// called.
    ///
    /// Returns `None` if `record()` has not been called
    pub fn value(&self) -> Option<DateTime<Utc>> {
        *self.timestamp.lock()
    }

    /// sets the value of this timestamp to the minimum of this and other
    pub fn update_to_min(&self, other: &Timestamp) {
        let min = match (self.value(), other.value()) {
            (None, None) => None,
            (Some(v), None) => Some(v),
            (None, Some(v)) => Some(v),
            (Some(v1), Some(v2)) => Some(if v1 < v2 { v1 } else { v2 }),
        };

        *self.timestamp.lock() = min;
    }

    /// sets the value of this timestamp to the maximum of this and other
    pub fn update_to_max(&self, other: &Timestamp) {
        let max = match (self.value(), other.value()) {
            (None, None) => None,
            (Some(v), None) => Some(v),
            (None, Some(v)) => Some(v),
            (Some(v1), Some(v2)) => Some(if v1 < v2 { v2 } else { v1 }),
        };

        *self.timestamp.lock() = max;
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        self.value().eq(&other.value())
    }
}

impl Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.value() {
            None => write!(f, "NONE"),
            Some(v) => {
                write!(f, "{v}")
            }
        }
    }
}

/// RAAI structure that adds all time between its construction and
/// destruction to the CPU time or the first call to `stop` whichever
/// comes first
pub struct ScopedTimerGuard<'a> {
    inner: &'a Time,
    start: Option<Instant>,
}

impl ScopedTimerGuard<'_> {
    /// Stop the timer timing and record the time taken
    pub fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            self.inner.add_elapsed(start)
        }
    }

    /// Restarts the timer recording from the current time
    pub fn restart(&mut self) {
        self.start = Some(Instant::now())
    }

    /// Stop the timer, record the time taken and consume self
    pub fn done(mut self) {
        self.stop()
    }
}

impl Drop for ScopedTimerGuard<'_> {
    fn drop(&mut self) {
        self.stop()
    }
}
