//! WASM-compatible `Instant` wrapper.

#[cfg(target_family = "wasm")]
/// DataFusion wrapper around [`std::time::Instant`]. Uses [`web_time::Instant`]
/// under `wasm` feature gate. It provides the same API as [`std::time::Instant`].
pub type Instant = web_time::Instant;

#[allow(clippy::disallowed_types)]
#[cfg(not(target_family = "wasm"))]
/// DataFusion wrapper around [`std::time::Instant`]. This is only a type alias.
pub type Instant = std::time::Instant;

use chrono::{DateTime, Utc};

/// Generate a timestamp that can be added to the message table
pub fn create_timestamp_str() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.format("%a %b %e %T %Y").to_string()
}

/// Generate a timestamp that can be added to the message table
pub fn create_timestamp_micros() -> i64 {
    let now: DateTime<Utc> = Utc::now();
    now.timestamp_micros()
}

/// Convert timestamp in micro seconds to a formatted string
pub fn convert_timestamp_micros_to_str(timestamp_micros: i64) -> String {
    // Convert microseconds to seconds and nanoseconds
    let datetime = DateTime::from_timestamp(
        timestamp_micros / 1_000_000,                    // seconds
        ((timestamp_micros % 1_000_000) * 1_000) as u32, // nanoseconds
    )
    .unwrap();

    // Format as a string
    datetime.format("%a %b %e %T %Y").to_string()
}
