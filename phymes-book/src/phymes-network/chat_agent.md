# Session Plan: Chat Agent

## Synopsis

This tutorial describes how the [Chat Agent Session Plan](https://github.com/biom8er/phymes/blob/main/phymes-network/src/session_plans/chat_agent_session.rs) uses the [phymes-agent](https://github.com/biom8er/phymes/blob/main/phymes-network/README.md) and [phymes-subject](https://github.com/biom8er/phymes/blob/main/phymes-subject/README.md) crates to build a simple chat agent. The `chat agent` is provided in the [examples](https://github.com/biom8er/phymes/blob/main/phymes-network/examples/chat_agent_session/main.rs).

## Tutorial

The simplest agentic AI architecture is that of a chat agent, which can be modeled as a static directed acyclic graph.

```mermaid
sequenceDiagram
    autonumber
    user ->> TGI: messages
    TGI ->> user: messages
```

The session is composed of 2 tasks: 1. the user, and 2. Text generation inference (TGI).

![sign-in](../assets/2025-07-05_phymes-app_docchat-no-rag_messaging.png)

The session starts when the user uploads their query to the session (1). The TGI task generates a response for the user based on the query (2). The session ends when there are no further updates to the subjects. If the user publishes a follow-up message, the session will pick-up where it left off.

```mermaid
flowchart TD
	subgraph chat_task_1
		messages-subject-.AllRecordBatches.->chat_processor_1-subscribe
		-subject--None-->chat_processor_1-subscribe
		chat_processor_1-subject--AllRecordBatches-->chat_processor_1-subscribe
		chat_processor_1-subscribe-->chat_processor_1-processor
		chat_processor_1-processor-->chat_processor_1-publish
		chat_processor_1-publish--ExtendChunks-->messages-subject
	end
	subgraph session_1
		messages-subject-.LastRecordBatch.->session_1-subscribe
		session_1-subscribe-->session_1-processor
		session_1-processor-->session_1-publish
		session_1-publish--Extend-->messages-subject
	end
	rt_1-rt-->chat_task_1
	rt_default-rt-->session_1
	chat_processor_1-processor@{shape: rect, label: CandleChatProcessor}
	session_1-processor@{shape: rect, label: ArrowProcessorEcho}
	rt_1-rt@{shape: subproc, label: rt_1}
	rt_default-rt@{shape: subproc, label: rt_default}
	chat_processor_1-subject@{shape: doc, label: chat_processor_1}
	messages-subject@{shape: doc, label: messages}
	chat_processor_1-publish@{shape: fork}
	session_1-publish@{shape: fork}
	chat_processor_1-subscribe@{shape: diamond, label: All}
	session_1-subscribe@{shape: diamond, label: All}
```

Under the hood, the states of the application are determined by the subjects that are subscribed to and published on by the User, and TGI tasks. Each task is composed of one or more processes that are chained together to execute the task. Each processor listens for changes on their subscribed subjects and publishes their results to subjects. Each task runs once the subscription criteria for all of its child processors are satisfied.

At each superstep of the session, subscribed subjects are allocated to tasks, tasks are ran in parallel, and the subjects for which tasks publish on are updated sequentially.

While not shown in the flowchart above, each processor subscribes to a special subject usually called the config which specifies all of the parameters for the processor. And like any other subject, the config can also be updated dynamically during the execution of the session.

The decision to chain multiple processors into a single task or to allocated each processor to its own task is up to the needs of the user. Chaining multple processors can be more performant and efficient because fewer subscription and publishing copies and updates, respectively are needed. However, allocating each processor to its own task is easier to debug since the output of each task can be easily verified. Also, any processor that requires an external API call has to be allocated to its own task as chaining of streams breaks the poll on the external API call.

## Next steps

The [Chat Agent Session Plan](https://github.com/biom8er/phymes/phymes-network/src/session_plans/chat_agent_session.rs) comes with a number of default configurations including the model, number of tokens to sample, temperature of sampling, etc. that can be modified by the user.