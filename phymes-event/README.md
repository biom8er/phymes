# PHYMES: Parallel HYpergraph MEssaging Streams

Core crate

<!--- ANCHOR: synopsis --->

## Synopsis

The PHYMES core crate implements the core algorithm for parallel and streaming hypergraph message passing. Similar to Pregel, messages are passed to computational nodes at each superstep. The messaging passing continues until a criteria is met or a maximum number of supersteps have been reached. Unlike Pregel, PHYMES does not operate over an undirected graph, but over a directed hypergraph using a publish-subscribe model where subjects are nodes and tasks are hyperedges and the directionality is determined by whether subjects are subscribed to or published on. In short, tasks subscribe to subjects, process subjects, and then publish their results to subjects.

<!--- ANCHOR_END: synopsis --->