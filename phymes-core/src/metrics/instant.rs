//! WASM-compatible `Instant` wrapper.

#[cfg(target_family = "wasm")]
/// DataFusion wrapper around [`std::time::Instant`]. Uses [`web_time::Instant`]
/// under `wasm` feature gate. It provides the same API as [`std::time::Instant`].
pub type Instant = web_time::Instant;

#[allow(clippy::disallowed_types)]
#[cfg(not(target_family = "wasm"))]
/// DataFusion wrapper around [`std::time::Instant`]. This is only a type alias.
pub type Instant = std::time::Instant;
