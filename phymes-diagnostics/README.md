# PHYMES: Parallel HYpergraph MEssaging Streams

Diagnostics crate

<!--- ANCHOR: synopsis --->

## Synopsis

The PHYMES diagnostics crate provides tools to observe, monitor, and record the behavior of PHYMES networks for debugging and optimization. The tools provided include traces, events, and metrics. Traces track the flow of subject messages through tasks and processors. Events add context to enable building a comprehensive timeline of what happened, when it happened, and why it happened. Metrics focus on aggregating numerical data over time from events to provide an overview of system performance and resource utilization. 

The diagnostics are intended to be implemented in such a way that you "Pay" only for what you use: Networks, Tasks, and Processors can be instrumented to record diagnostics, but are only invoked when explicitly provided the resources by the implementer to record the diagnostics. The diagnostics are designed real-time telemetry in multi-thread environments using an `Arc<Mutex>` pattern instead of `MSPSC` channels.

<!--- ANCHOR_END: synopsis --->