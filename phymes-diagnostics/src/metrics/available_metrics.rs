use std::{borrow::Cow, fmt::Display};

use serde_json::{Map, Value};

use crate::{
    diagnostics::JSONObjectTrait,
    metrics::{Count, Gauge, Time, Timestamp},
};

/// Possible Metrics to track a value of interest (metric)
///
/// Among other differences, the metric types have different ways to
/// logically interpret their underlying values and some metrics are
/// so common they are given special treatment.
#[derive(Debug, Clone, PartialEq)]
pub enum Metric {
    /// Number of output rows produced: "output_rows" metric
    OutputRows(Count),
    /// Elapsed Compute Time: the wall clock time spent in "cpu
    /// intensive" work.
    ///
    /// This measurement represents, roughly:
    /// ```
    /// use std::time::Instant;
    /// let start = Instant::now();
    /// // ...CPU intensive work here...
    /// let elapsed_compute = (Instant::now() - start).as_nanos();
    /// ```
    ///
    /// Note 1: Does *not* include time other operators spend
    /// computing input.
    ///
    /// Note 2: *Does* includes time when the thread could have made
    /// progress but the OS did not schedule it (e.g. due to CPU
    /// contention), thus making this value different than the
    /// classical definition of "cpu_time", which is the time reported
    /// from `clock_gettime(CLOCK_THREAD_CPUTIME_ID, ..)`.
    ElapsedCompute(Time),
    /// Number of spills produced: "spill_count" metric
    SpillCount(Count),
    /// Total size of spilled bytes produced: "spilled_bytes" metric
    SpilledBytes(Count),
    /// Total size of spilled rows produced: "spilled_rows" metric
    SpilledRows(Count),
    /// Current memory used
    CurrentMemoryUsage(Gauge),
    /// Operator defined count.
    Count {
        /// The provided name of this metric
        name: Cow<'static, str>,
        /// The value of the metric
        count: Count,
    },
    /// Operator defined gauge.
    Gauge {
        /// The provided name of this metric
        name: Cow<'static, str>,
        /// The value of the metric
        gauge: Gauge,
    },
    /// Operator defined time
    Time {
        /// The provided name of this metric
        name: Cow<'static, str>,
        /// The value of the metric
        time: Time,
    },
    /// The time at which execution started
    StartTimestamp(Timestamp),
    /// The time at which execution ended
    EndTimestamp(Timestamp),
}

impl Metric {
    /// Return the value of the metric as a usize value
    pub fn as_usize(&self) -> usize {
        match self {
            Self::OutputRows(count) => count.value(),
            Self::SpillCount(count) => count.value(),
            Self::SpilledBytes(bytes) => bytes.value(),
            Self::SpilledRows(count) => count.value(),
            Self::CurrentMemoryUsage(used) => used.value(),
            Self::ElapsedCompute(time) => time.value(),
            Self::Count { count, .. } => count.value(),
            Self::Gauge { gauge, .. } => gauge.value(),
            Self::Time { time, .. } => time.value(),
            Self::StartTimestamp(timestamp) => timestamp
                .value()
                .and_then(|ts| ts.timestamp_nanos_opt())
                .map(|nanos| nanos as usize)
                .unwrap_or(0),
            Self::EndTimestamp(timestamp) => timestamp
                .value()
                .and_then(|ts| ts.timestamp_nanos_opt())
                .map(|nanos| nanos as usize)
                .unwrap_or(0),
        }
    }

    /// create a new MetricValue with the same type as `self` suitable
    /// for accumulating
    pub fn new_empty(&self) -> Self {
        match self {
            Self::OutputRows(_) => Self::OutputRows(Count::new()),
            Self::SpillCount(_) => Self::SpillCount(Count::new()),
            Self::SpilledBytes(_) => Self::SpilledBytes(Count::new()),
            Self::SpilledRows(_) => Self::SpilledRows(Count::new()),
            Self::CurrentMemoryUsage(_) => Self::CurrentMemoryUsage(Gauge::new()),
            Self::ElapsedCompute(_) => Self::ElapsedCompute(Time::new()),
            Self::Count { name, .. } => Self::Count {
                name: name.clone(),
                count: Count::new(),
            },
            Self::Gauge { name, .. } => Self::Gauge {
                name: name.clone(),
                gauge: Gauge::new(),
            },
            Self::Time { name, .. } => Self::Time {
                name: name.clone(),
                time: Time::new(),
            },
            Self::StartTimestamp(_) => Self::StartTimestamp(Timestamp::new()),
            Self::EndTimestamp(_) => Self::EndTimestamp(Timestamp::new()),
        }
    }

    /// Aggregates the value of other to `self`. panic's if the types
    /// are mismatched or aggregating does not make sense for this
    /// value
    ///
    /// Note this is purposely marked `mut` (even though atomics are
    /// used) so Rust's type system can be used to ensure the
    /// appropriate API access. `MetricValues` should be modified
    /// using the original [`Count`] or [`Time`] they were created
    /// from.
    pub fn aggregate(&mut self, other: &Self) {
        match (self, other) {
            (Self::OutputRows(count), Self::OutputRows(other_count))
            | (Self::SpillCount(count), Self::SpillCount(other_count))
            | (Self::SpilledBytes(count), Self::SpilledBytes(other_count))
            | (Self::SpilledRows(count), Self::SpilledRows(other_count))
            | (
                Self::Count { count, .. },
                Self::Count {
                    count: other_count, ..
                },
            ) => count.add(other_count.value()),
            (Self::CurrentMemoryUsage(gauge), Self::CurrentMemoryUsage(other_gauge))
            | (
                Self::Gauge { gauge, .. },
                Self::Gauge {
                    gauge: other_gauge, ..
                },
            ) => gauge.add(other_gauge.value()),
            (Self::ElapsedCompute(time), Self::ElapsedCompute(other_time))
            | (
                Self::Time { time, .. },
                Self::Time {
                    time: other_time, ..
                },
            ) => time.add(other_time),
            // timestamps are aggregated by min/max
            (Self::StartTimestamp(timestamp), Self::StartTimestamp(other_timestamp)) => {
                timestamp.update_to_min(other_timestamp);
            }
            // timestamps are aggregated by min/max
            (Self::EndTimestamp(timestamp), Self::EndTimestamp(other_timestamp)) => {
                timestamp.update_to_max(other_timestamp);
            }
            m @ (_, _) => {
                panic!(
                    "Mismatched metric types. Can not aggregate {:?} with value {:?}",
                    m.0, m.1
                )
            }
        }
    }

    /// Returns a number by which to sort metrics by display. Lower
    /// numbers are "more useful" (and displayed first)
    pub fn display_sort_key(&self) -> u8 {
        match self {
            Self::OutputRows(_) => 0,     // show first
            Self::ElapsedCompute(_) => 1, // show second
            Self::SpillCount(_) => 2,
            Self::SpilledBytes(_) => 3,
            Self::SpilledRows(_) => 4,
            Self::CurrentMemoryUsage(_) => 5,
            Self::Count { .. } => 6,
            Self::Gauge { .. } => 7,
            Self::Time { .. } => 8,
            Self::StartTimestamp(_) => 9, // show timestamps last
            Self::EndTimestamp(_) => 10,
        }
    }

    /// returns true if this metric has a timestamp value
    pub fn is_timestamp(&self) -> bool {
        matches!(self, Self::StartTimestamp(_) | Self::EndTimestamp(_))
    }
}

impl Display for Metric {
    /// Prints the value of this event
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::OutputRows(_) => write!(f, "output_rows"),
            Self::SpillCount(_) => write!(f, "spill_count"),
            Self::SpilledBytes(_) => write!(f, "spilled_bytes"),
            Self::SpilledRows(_) => write!(f, "spilled_rows"),
            Self::CurrentMemoryUsage(_) => write!(f, "mem_used"),
            Self::ElapsedCompute(_) => write!(f, "elapsed_compute"),
            Self::Count { name, .. } => write!(f, "{name}"),
            Self::Gauge { name, .. } => write!(f, "{name}"),
            Self::Time { name, .. } => write!(f, "{name}"),
            Self::StartTimestamp(_) => write!(f, "start_timestamp"),
            Self::EndTimestamp(_) => write!(f, "end_timestamp"),
        }
    }
}

impl JSONObjectTrait for Metric {
    fn to_json_object(&self) -> Vec<Map<String, Value>> {
        let mut map = Map::new();
        map.insert("metric_name".to_string(), self.to_string().into());
        map.insert("metric_value".to_string(), self.as_usize().into());
        vec![map]
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;

    #[test]
    fn test_metric_values() {
        let count = Count::new();
        let metric = Metric::OutputRows(count.clone());
        count.add(1);
        assert_eq!(metric.as_usize(), 1);
        assert!(!metric.is_timestamp());

        let guage = Gauge::new();
        let metric = Metric::CurrentMemoryUsage(guage.clone());
        guage.add(1);
        assert_eq!(metric.as_usize(), 1);
        assert!(!metric.is_timestamp());

        let time = Time::new();
        let metric = Metric::ElapsedCompute(time.clone());
        time.add_duration(std::time::Duration::from_nanos(1));
        assert_eq!(metric.as_usize(), 1);
        assert!(!metric.is_timestamp());

        let timestamp = Timestamp::new();
        let metric = Metric::StartTimestamp(timestamp.clone());
        // 1431648000000000 == 1970-01-17 13:40:48 UTC
        let t1 = Utc.timestamp_nanos(1431648000000000);
        timestamp.set(t1);
        #[cfg(not(target_family = "wasm"))]
        assert_eq!(metric.as_usize(), 1431648000000000);
        assert!(metric.is_timestamp());
    }
}
