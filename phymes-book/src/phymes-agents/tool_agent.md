# Session Plan: Tool Agent
## Synopsis

This tutorial describes how the [Tool Agent Session Plan](https://github.com/biom8er/phymes/blob/main/phymes-agents/src/session_plans/tool_agent_session.rs) uses the [phymes-agent](https://github.com/biom8er/phymes/blob/main/phymes-agents/README.md) and [phymes-core](https://github.com/biom8er/phymes/blob/main/phymes-core/README.md) crates to build a tool calling agent.

## Tutorial

The tool agent adds stochasticity to the agentic AI architecture of the chat agent by conditionally calling an external tool if needed, which can be modeled as a conditional directed cyclic graph.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> chat_agent: Query
    chat_agent --> [*]: Response
    chat_agent --> tool_task: Invoke tool
    tool_task --> chat_agent: Tool call
```

The session starts with a query to the chat_agent from the user. Next, the chat_agent may call one or more tools to answer the user query. Next, the tool calls are executed in parallel by the tool_task. Finally, the results of the tool calls are provided to the chat_agent to ground the text generation inference to respond back to the user.

Under the hood, the states of the application are determined by the subjects that are subscribed to and published on by the user, tool_task and chat_agent.

```mermaid
sequenceDiagram
    user->>messages: 1
    messages-->>chat_agent: 2
    tools-->>chat_agent: 3a
    config->>chat_agent: 3b
    chat_agent->>tool_calls: 4
    tool_calls->>tool_task: 5
    data->>tool_task: 6
    tool_task->>messages: 7
    messages-->>chat_agent: 8
    tools-->>chat_agent: 9a
    config->>chat_agent: 9b
    chat_agent->>messages: 10
    messages->>user: 11
```

The sequence of actions are the following:

1. The user publishes to messages subject
2. The chat_agent subscribes to messages subject when there is a change to the messages subject table.
3. The chat_agent subscribes to configs and tools subjects no matter if there is a change or not because the configs provide the parameters for running the chat_agent and the tools describes the schema for the tool calls.
4. The chat_agent performs text generation inference based on the messages subject content and tool schemas, and publishes the results to either messages or tool_calls subject.
5. The tool_task subscribes to the tool_calls subject when there is a change to the tool_calls subject table
6. The tool_task subscribes to the data subject (one or more data tables needed to execute the tool call) no matter if there is a change or not because the data subjects provide the data tables needed for running the tool_task.
7. The tool_task retrieves the needed data tables to execute the tool_calls, executes the tool_calls, and publishes the results to the messages subject.
8. The chat_agent subscribes to messages subject when there is a change to the messages subject table, which has now been updated with the results of the tool_calls.
9. The chat_agent subscribes to configs and tools subjects no matter if there is a change or not because the configs provide the parameters for running the chat_agent and the tools describes the schema for the tool calls.
10. The chat_agent performs text generation inference based on the messages subject content, tool schemas, and results of the tool_calls, and publishes the results to either messages or tool_calls subject.
11. The user subscribes to messages subject where there is a change to the messages subject table.

The session ends because there are no further updates to the subjects. If the user publishes a follow-up message the session will pick-up where it left off with the chat_agent responding to the updated message and tool_calls content.

```mermaid
flowchart TD
	subgraph message_aggregator_task_1
		user_messages-subject--FullTable-->message_aggregator_processor_1-subscribe
		tool_messages-subject-.LastRecordBatch.->message_aggregator_processor_1-subscribe
		assistant_messages-subject--FullTable-->message_aggregator_processor_1-subscribe
		message_aggregator_processor_1-subject--LastRecordBatch-->message_aggregator_processor_1-subscribe
		message_aggregator_processor_1-subscribe-->message_aggregator_processor_1-processor
		message_aggregator_processor_1-processor-->message_aggregator_processor_1-publish
		message_aggregator_processor_1-publish--Replace-->chat_task_1-subject
	end
	subgraph message_aggregator_task_2
		user_messages-subject-.LastRecordBatch.->message_aggregator_processor_2-subscribe
		assistant_messages-subject-.LastRecordBatch.->message_aggregator_processor_2-subscribe
		message_aggregator_processor_2-subject--LastRecordBatch-->message_aggregator_processor_2-subscribe
		message_aggregator_processor_2-subscribe-->message_aggregator_processor_2-processor
		message_aggregator_processor_2-processor-->message_aggregator_processor_2-publish
		message_aggregator_processor_2-publish--Extend-->messages-subject
	end
	subgraph chat_task_1
		chat_task_1-subject-.FullTable.->chat_processor_1-subscribe
		tools-subject--FullTable-->chat_processor_1-subscribe
		chat_processor_1-subject--FullTable-->chat_processor_1-subscribe
		chat_processor_1-subscribe-->chat_processor_1-processor
		chat_processor_1-processor-->chat_processor_1-publish
		chat_processor_1-publish--Replace-->message_parser_task_1-subject
	end
	subgraph message_parser_task_1
		message_parser_task_1-subject-.FullTable.->message_parser_processor_1-subscribe
		message_parser_processor_1-subject--FullTable-->message_parser_processor_1-subscribe
		message_parser_processor_1-subscribe-->message_parser_processor_1-processor
		message_parser_processor_1-processor-->message_parser_processor_1-publish
		message_parser_processor_1-publish--Extend-->assistant_messages-subject
		message_parser_processor_1-publish--Extend-->SortColumnAndIndices-subject
		message_parser_processor_1-publish--Extend-->HumanInTheLoop-subject
	end
	subgraph SortColumnAndIndices
		SortColumnAndIndices-subject-.LastRecordBatch.->SortColumnAndIndices-subscribe
		available_data_1-subject--FullTable-->SortColumnAndIndices-subscribe
		SortColumnAndIndices-subscribe-->SortColumnAndIndices-processor
		SortColumnAndIndices-processor-->SortColumnAndIndices-publish
		SortColumnAndIndices-publish--Replace-->available_data_1-subject
		summary_processor_1-subject--LastRecordBatch-->summary_processor_1-subscribe
		available_data_1-subject--LastRecordBatch-->summary_processor_1-subscribe
		summary_processor_1-subscribe-->summary_processor_1-processor
		summary_processor_1-processor-->summary_processor_1-publish
		summary_processor_1-publish--Extend-->tool_messages-subject
	end
	subgraph HumanInTheLoop
		HumanInTheLoop-subject-.LastRecordBatch.->HumanInTheLoop-subscribe
		HumanInTheLoop-subscribe-->HumanInTheLoop-processor
		HumanInTheLoop-processor-->HumanInTheLoop-publish
		HumanInTheLoop-publish--Extend-->assistant_messages-subject
		summary_processor_2-subject--LastRecordBatch-->summary_processor_2-subscribe
		assistant_messages-subject--LastRecordBatch-->summary_processor_2-subscribe
		summary_processor_2-subscribe-->summary_processor_2-processor
		summary_processor_2-processor-->summary_processor_2-publish
		summary_processor_2-publish--Extend-->assistant_messages-subject
	end
	subgraph session_context_1
		assistant_messages-subject-.LastRecordBatch.->session_context_1-subscribe
		session_context_1-subscribe-->session_context_1-processor
		session_context_1-processor-->session_context_1-publish
		session_context_1-publish--Extend-->user_messages-subject
		session_context_1-publish--Extend-->assistant_messages-subject
	end
	tool_rt_1-rt-->message_aggregator_task_1
	message_aggregator_rt_1-rt-->message_aggregator_task_2
	chat_rt_1-rt-->chat_task_1
	chat_rt_1-rt-->message_parser_task_1
	tool_rt_1-rt-->SortColumnAndIndices
	tool_rt_1-rt-->HumanInTheLoop
	rt_default-rt-->session_context_1
	message_aggregator_processor_1-processor@{shape: rect, label: MessageAggregatorProcessor}
	message_aggregator_processor_2-processor@{shape: rect, label: MessageAggregatorProcessor}
	chat_processor_1-processor@{shape: rect, label: CandleChatProcessor}
	message_parser_processor_1-processor@{shape: rect, label: MessageParserProcessor}
	SortColumnAndIndices-processor@{shape: rect, label: CandleDataProcessor}
	HumanInTheLoop-processor@{shape: rect, label: CandleDataProcessor}
	summary_processor_1-processor@{shape: rect, label: DataSummaryProcessor}
	summary_processor_2-processor@{shape: rect, label: DataSummaryProcessor}
	session_context_1-processor@{shape: rect, label: ArrowProcessorEcho}
	chat_rt_1-rt@{shape: subproc, label: chat_rt_1}
	message_aggregator_rt_1-rt@{shape: subproc, label: message_aggregator_rt_1}
	rt_default-rt@{shape: subproc, label: rt_default}
	tool_rt_1-rt@{shape: subproc, label: tool_rt_1}
	HumanInTheLoop-subject@{shape: doc, label: HumanInTheLoop}
	SortColumnAndIndices-subject@{shape: doc, label: SortColumnAndIndices}
	assistant_messages-subject@{shape: doc, label: assistant_messages}
	available_data_1-subject@{shape: doc, label: available_data_1}
	chat_processor_1-subject@{shape: doc, label: chat_processor_1}
	chat_task_1-subject@{shape: doc, label: chat_task_1}
	message_aggregator_processor_1-subject@{shape: doc, label: message_aggregator_processor_1}
	message_aggregator_processor_2-subject@{shape: doc, label: message_aggregator_processor_2}
	message_parser_processor_1-subject@{shape: doc, label: message_parser_processor_1}
	message_parser_task_1-subject@{shape: doc, label: message_parser_task_1}
	messages-subject@{shape: doc, label: messages}
	summary_processor_1-subject@{shape: doc, label: summary_processor_1}
	summary_processor_2-subject@{shape: doc, label: summary_processor_2}
	tool_messages-subject@{shape: doc, label: tool_messages}
	tools-subject@{shape: doc, label: tools}
	user_messages-subject@{shape: doc, label: user_messages}
	HumanInTheLoop-publish@{shape: fork}
	SortColumnAndIndices-publish@{shape: fork}
	chat_processor_1-publish@{shape: fork}
	message_aggregator_processor_1-publish@{shape: fork}
	message_aggregator_processor_2-publish@{shape: fork}
	message_parser_processor_1-publish@{shape: fork}
	session_context_1-publish@{shape: fork}
	summary_processor_1-publish@{shape: fork}
	summary_processor_2-publish@{shape: fork}
	HumanInTheLoop-subscribe@{shape: diamond, label: All}
	SortColumnAndIndices-subscribe@{shape: diamond, label: All}
	chat_processor_1-subscribe@{shape: diamond, label: All}
	message_aggregator_processor_1-subscribe@{shape: diamond, label: ChatContentSubscribe}
	message_aggregator_processor_2-subscribe@{shape: diamond, label: Any}
	message_parser_processor_1-subscribe@{shape: diamond, label: All}
	session_context_1-subscribe@{shape: diamond, label: All}
	summary_processor_1-subscribe@{shape: diamond, label: All}
	summary_processor_2-subscribe@{shape: diamond, label: All}
```

## Next steps

The [Tool Agent Session Plan](https://github.com/biom8er/phymes/blob/main/phymes-agents/src/session_plans/tool_agent_session.rs) comes with a number of default configurations including the model, number of tokens to sample, temperature of sampling, etc. that can be modified by the user. A trivial use case is provided for sorting an array in a table that can be used as a starting point for creating more complex realistic use cases involving tools that manipulate (large) tabular data.