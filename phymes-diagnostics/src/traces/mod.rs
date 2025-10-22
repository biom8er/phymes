mod current_context;
mod span;
mod tracer;
mod builder;
mod available_traces;

pub use tracer::{TraceableTrait, TraceRecord, Tracer};
pub use current_context::CurrentContext;
pub use span::{Span, SpanBuilder, create_random_id};
pub use available_traces::Trace;
pub use builder::TraceBuilderTrait;

// DM: linting does not realize that it is used in `diagnostic_set` tests
#[allow(unused_imports)]
pub(crate) use available_traces::available_tracers_tests::Message;