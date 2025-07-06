# Phymes glossary

### Session

Manifestation of particular task hypergraph that a user can publish subjects to and stream subscribed subjects from. The `SessionContext` maintains the metrics, tasks, state, and runtime (discussed below). The `SessionStream` manages the subscribing and publishing of subjects by tasks by iteratively running a "super step" via `SessionStreamStep` which invokes tasks for which all of their subscribed subjects have been updated and then updates the subjects with the published results of each ran task.

### State

The application state are subjects (i.e., tables) that implement the `ArrowTableTrait`. Subjects include data tables, configurations (e.g., single row tables where the schema defines the parameters and the first row defines the values for the parameters), and aggregated metrics. Subjects can be subscribed to and published to from multiple tasks, which results in a hypergraph structure.

### Metrics

Recorded per `ArrowProcessorTrait` and aggregated at the query level. Examples of baseline metrics include runtime and processed table row counts. Additional metrics can be defined by the user.

### Tasks

The unit of operation. Tasks are hyperedges in the task hypergraph. Tasks are often referenced as Agents or Tools in the agentic AI world or as execution plan partitions in the database world. The trait `ArrowTaskTrait` defines the behavior of the task through the `run` method which is responsible for subscribing to subjects, processing them by calling a series of structs implementing the `ArrowProcessorTrait`, and then publishing outgoing messages to subjects. The struct that implements the `ArrowTaskTrait` includes a name for the task, metrics to track the task processes, runtime information, access to the complete or a subset of the complete state, and a `ArrowProcessorTrait`. The processing of messages is implemented by the `ArrowProcessorTrait`, which recieves the incoming message, metrics, state, and runtime from the `ArrowTaskTrait`.

### Process

The processing of messages is implemented by the `ArrowProcessorTrait`, which recieves the incoming message, metrics, state, and runtime from the `ArrowTaskTrait`. The processor operates over streams of `RecordBatchs` which enables scaling across multiple nodes, and allows for calling native or remote (via RPC) components or provider services. Tasks can call a series of structs implementing `ArrowProcessorTrait` forming a complex computational chain over subscribed messagings. Processes are intended to be quasi-stateless, and therefore all state and runtime artifacts are passed as parameters to the `process` method. For example, the configurations for the processor method are passed as state tables, the runtime artifacts are set in the runtime, and the metrics for recording the time and number of rows processed and other user defined criteria are all passed as input paramters. The only state maintained by the processors are the subscribed subjects, subjects to publish, and subjects to pass through to the next process in the chain.

### Messages

The unit of communication between tasks. `ArrowIncomingMessage`es are sent to tasks in uncompressed tables or compressed IPC format; while `ArrowOutgoingMessage`es are sent from tasks in a streaming format to enable lazy and parallel execution of the task processor computational chain. The streaming format is used as both input and output for processors to enable scaling to massive data sizes that can be efficiently distributed across multiple nodes.

### Tables

The unit of data. Tables implement the `ArrowTableTrait` and facilitate the manipulation of Arrow `RecordBatch`es. `RecordBatch`es are a highly efficient columnar data storage format that facilitate computation on the CPU and transfer to the GPU for accelerated computation. Tables can be seralized when sending over the wire, and converted to and from various formats including JSON as needed. 

### Runtime Env

Defines how Processes are ran. For AI/ML tasks, the `RuntimeEnv` may include model weights loaded into GPU memory. For ETL tasks, the `RuntimeEnv` may include CPU settings and memory or wall clock constraints. Similar to state, the `RuntimeEnv` artifacts are intended to be shared between different processes. For example, a single LLM model weights maybe used by multiple processors in parallel or re-used between queries to the same session without having to reload the weights into memory each time. The runtime environment can also be thought of as services that processes can share. For example, instead of hosting an LLM locally, the runtime can define an API to call to generate tokens remotely which provides flexibility and scalability depending upon the needs of the application.