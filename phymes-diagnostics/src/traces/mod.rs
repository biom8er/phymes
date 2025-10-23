mod available_traces;
mod builder;
mod current_context;
mod span;
mod tracer;

pub use available_traces::Trace;
pub use builder::TraceBuilderTrait;
pub use current_context::CurrentContext;
pub use span::{Span, SpanBuilder, create_random_id};
pub use tracer::{TraceRecord, TraceableTrait, Tracer};

// DM: linting does not realize that it is used in `diagnostic_set` tests
#[allow(unused_imports)]
pub(crate) use available_traces::available_tracers_tests::Message;
