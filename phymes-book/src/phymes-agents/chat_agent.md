# Session Plan: Chat Agent

## Synopsis

This tutorial describes how the [Chat Agent Session Plan](https://github.com/biom8er/phymes/blob/main/phymes-agents/src/session_plans/chat_agent_session.rs) uses the [phymes-agent](https://github.com/biom8er/phymes/blob/main/phymes-agents/README.md) and [phymes-core](https://github.com/biom8er/phymes/blob/main/phymes-core/README.md) crates to build a simple chat agent. The `chat agent` is provided in the [examples](https://github.com/biom8er/phymes/blob/main/phymes-agents/examples/chat_agent_session/main.rs).

## Tutorial

The simplest agentic AI architecture is that of a chat agent, which can be modeled as a static directed acyclic graph.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> chat_agent: Query
    chat_agent --> [*]: Response
```

The session starts with a query to the chat_agent from the user, and the chat_agent runs text generation inference to respond back to the user.

Under the hood, the states of the application are determined by the subjects that are subscribed to and published on by the user and the chat_agent.

```mermaid
sequenceDiagram
    user->>messages: 1
    messages->>chat_agent: 2
    config->>chat_agent: 3
    chat_agent->>messages: 4
    messages->>user: 5
```

The sequence of actions are the following:

1. The user publishes to messages subject
2. The chat_agent subscribes to messages subject when there is a change to the messages subject table.
3. The chat_agent subscribes to configs subject no matter if there is a change or not because the configs provide the parameters for running the chat_agent.
4. The chat_agent performs text generation inference based on the messages subject content and publishes the results to the messages subject.
5. The user subscribes to messages subject where there is a change to the messages subject table.

![sign-in](../assets/2025-07-05_phymes-app_docchat-no-rag_messaging.png)

The session ends because there are no further updates to the subjects. If the user publishes a follow-up message, the session will pick-up where it left off with the chat_agent responding to the updated message content.

```mermaid
flowchart TD
	subgraph chat_task_1
		messages-subject-.FullTable.->chat_processor_1-subscribe
		-subject--None-->chat_processor_1-subscribe
		chat_processor_1-subject--FullTable-->chat_processor_1-subscribe
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

## Next steps

The [Chat Agent Session Plan](https://github.com/biom8er/phymes/phymes-agents/src/session_plans/chat_agent_session.rs) comes with a number of default configurations including the model, number of tokens to sample, temperature of sampling, etc. that can be modified by the user.