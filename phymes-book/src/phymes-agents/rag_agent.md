# Seesion Plan: Retrieval Augmented Generation (RAG) Agent
## Synopsis

This tutorial describes how the [Document RAG Agent Session Plan](https://github.com/biom8er/phymes/blob/main/phymes-agents/src/session_plans/document_rag_session.rs) uses the [phymes-agent](https://github.com/biom8er/phymes/blob/main/phymes-agents/README.md) and [phymes-core](https://github.com/biom8er/phymes/blob/main/phymes-core/README.md) crates to build a tool calling agent.

## Tutorial

The document RAG agent adds a complex document parsing, embedding, and retrieval Data pipeline to the agentic AI architecture of the chat agent.

```mermaid
sequenceDiagram
    autonumber
    user ->> TEI: documents
    user ->> TEI: query 
    user ->> TGI: user_message
    TEI ->> retrieval: embedded_docs
    TEI ->> retrieval: embedded_query
    retrieval ->> TGI: top_k_docs
    TGI ->> user: assistant_message
```

The session is composed of 4 tasks: 1. the user, 2. Text embedding inference (TEI), 3. Retrieval, and 4. Text generation inference (TGI). 

The session starts when the user publishes documents (1), and a query (2 and 3) to the session.

![documents](../assets/2025-07-05_phymes-app_docchat-documents_subjects.png)

The TEI task then chunks the documents, and embeds the chunks and query (4 and 5). The Retrieval task finds the document chunks that best match the query and returns the top K chunks (6). The TGI task generates a response for the user based on the top K chunks and the query (7).

![doc-rag-response](../assets/2025-07-05_phymes-app_docchat_messaging.png)

The session ends when there are no further updates to the subjects. If the user publishes a follow-up message or uploads new documents, the session will pick-up where it left off.

```mermaid
flowchart TD
	subgraph message_aggregator_task_1
		user_messages-subject-.FullTable.->message_aggregator_1-subscribe
		top_k-subject-.FullTable.->message_aggregator_1-subscribe
		assistant_messages-subject--FullTable-->message_aggregator_1-subscribe
		message_aggregator_1-subject--LastRecordBatch-->message_aggregator_1-subscribe
		message_aggregator_1-subscribe-->message_aggregator_1-processor
		message_aggregator_1-processor-->message_aggregator_1-publish
		message_aggregator_1-publish--Replace-->chat_task_1-subject
	end
	subgraph message_aggregator_task_2
		user_messages-subject-.LastRecordBatch.->message_aggregator_2-subscribe
		assistant_messages-subject-.LastRecordBatch.->message_aggregator_2-subscribe
		message_aggregator_2-subject--LastRecordBatch-->message_aggregator_2-subscribe
		message_aggregator_2-subscribe-->message_aggregator_2-processor
		message_aggregator_2-processor-->message_aggregator_2-publish
		message_aggregator_2-publish--Extend-->messages-subject
	end
	subgraph chat_task_1
		chat_task_1-subject-.FullTable.->chat_processor_1-subscribe
		-subject--None-->chat_processor_1-subscribe
		chat_processor_1-subject--FullTable-->chat_processor_1-subscribe
		chat_processor_1-subscribe-->chat_processor_1-processor
		chat_processor_1-processor-->chat_processor_1-publish
		chat_processor_1-publish--ExtendChunks-->assistant_messages-subject
	end
	subgraph embed_documents_task_1
		documents-subject-.FullTable.->chunk_documents_processor_1-subscribe
		chunk_documents_processor_1-subject--FullTable-->chunk_documents_processor_1-subscribe
		chunk_documents_processor_1-subscribe-->chunk_documents_processor_1-processor
		chunk_documents_processor_1-processor-->chunk_documents_processor_1-publish
		chunk_documents_processor_1-publish--Replace-->chunk_documents_task_1-subject
		chunk_documents_task_1-subject--FullTable-->embed_documents_processor_1-subscribe
		embed_documents_processor_1-subject--FullTable-->embed_documents_processor_1-subscribe
		embed_documents_processor_1-subscribe-->embed_documents_processor_1-processor
		embed_documents_processor_1-processor-->embed_documents_processor_1-publish
		embed_documents_processor_1-publish--Replace-->doc_embeddings-subject
	end
	subgraph embed_query_task_1
		queries-subject-.FullTable.->embed_query_processor_1-subscribe
		embed_query_processor_1-subject--FullTable-->embed_query_processor_1-subscribe
		embed_query_processor_1-subscribe-->embed_query_processor_1-processor
		embed_query_processor_1-processor-->embed_query_processor_1-publish
		embed_query_processor_1-publish--Replace-->q_embeddings-subject
	end
	subgraph vs_task_1
		doc_embeddings-subject--FullTable-->rel_sim_processor_1-subscribe
		q_embeddings-subject-.FullTable.->rel_sim_processor_1-subscribe
		rel_sim_processor_1-subject--FullTable-->rel_sim_processor_1-subscribe
		rel_sim_processor_1-subscribe-->rel_sim_processor_1-processor
		rel_sim_processor_1-processor-->rel_sim_processor_1-publish
		rel_sim_processor_1-publish--Replace-->tmp_scores-subject
		sort_scores_processor_1-subject--FullTable-->sort_scores_processor_1-subscribe
		tmp_scores-subject--FullTable-->sort_scores_processor_1-subscribe
		sort_scores_processor_1-subscribe-->sort_scores_processor_1-processor
		sort_scores_processor_1-processor-->sort_scores_processor_1-publish
		sort_scores_processor_1-publish--Replace-->tmp_scores-subject
		documents-subject--FullTable-->chunk_documents_processor_2-subscribe
		chunk_documents_processor_2-subject--FullTable-->chunk_documents_processor_2-subscribe
		chunk_documents_processor_2-subscribe-->chunk_documents_processor_2-processor
		chunk_documents_processor_2-processor-->chunk_documents_processor_2-publish
		chunk_documents_processor_2-publish--Replace-->documents-subject
		documents-subject--FullTable-->join_scores_chunks_processor_1-subscribe
		tmp_scores-subject--FullTable-->join_scores_chunks_processor_1-subscribe
		join_scores_chunks_processor_1-subject--FullTable-->join_scores_chunks_processor_1-subscribe
		join_scores_chunks_processor_1-subscribe-->join_scores_chunks_processor_1-processor
		join_scores_chunks_processor_1-processor-->join_scores_chunks_processor_1-publish
		join_scores_chunks_processor_1-publish--Replace-->tmp_scores_chunks_join-subject
		top_k_processor_1-subject--FullTable-->top_k_processor_1-subscribe
		tmp_scores_chunks_join-subject--FullTable-->top_k_processor_1-subscribe
		top_k_processor_1-subscribe-->top_k_processor_1-processor
		top_k_processor_1-processor-->top_k_processor_1-publish
		top_k_processor_1-publish--Replace-->top_k-subject
	end
	subgraph session_context_1
		assistant_messages-subject-.LastRecordBatch.->session_context_1-subscribe
		session_context_1-subscribe-->session_context_1-processor
		session_context_1-processor-->session_context_1-publish
		session_context_1-publish--Extend-->user_messages-subject
		session_context_1-publish--Extend-->documents-subject
		session_context_1-publish--Extend-->queries-subject
		session_context_1-publish--Extend-->assistant_messages-subject
	end
	vs_rt_1-rt-->message_aggregator_task_1
	vs_rt_1-rt-->message_aggregator_task_2
	chat_rt_1-rt-->chat_task_1
	embed_documents_rt_1-rt-->embed_documents_task_1
	embed_query_rt_1-rt-->embed_query_task_1
	vs_rt_1-rt-->vs_task_1
	rt_default-rt-->session_context_1
	message_aggregator_1-processor@{shape: rect, label: MessageAggregatorProcessor}
	message_aggregator_2-processor@{shape: rect, label: MessageAggregatorProcessor}
	chat_processor_1-processor@{shape: rect, label: CandleChatProcessor}
	chunk_documents_processor_1-processor@{shape: rect, label: CandleDataProcessor}
	embed_documents_processor_1-processor@{shape: rect, label: CandleEmbedProcessor}
	embed_query_processor_1-processor@{shape: rect, label: CandleEmbedProcessor}
	rel_sim_processor_1-processor@{shape: rect, label: CandleDataProcessor}
	sort_scores_processor_1-processor@{shape: rect, label: CandleDataProcessor}
	chunk_documents_processor_2-processor@{shape: rect, label: CandleDataProcessor}
	join_scores_chunks_processor_1-processor@{shape: rect, label: CandleDataProcessor}
	top_k_processor_1-processor@{shape: rect, label: PackTabular}
	session_context_1-processor@{shape: rect, label: ArrowProcessorEcho}
	chat_rt_1-rt@{shape: subproc, label: chat_rt_1}
	embed_documents_rt_1-rt@{shape: subproc, label: embed_documents_rt_1}
	embed_query_rt_1-rt@{shape: subproc, label: embed_query_rt_1}
	rt_default-rt@{shape: subproc, label: rt_default}
	vs_rt_1-rt@{shape: subproc, label: vs_rt_1}
	assistant_messages-subject@{shape: doc, label: assistant_messages}
	chat_processor_1-subject@{shape: doc, label: chat_processor_1}
	chat_task_1-subject@{shape: doc, label: chat_task_1}
	chunk_documents_processor_1-subject@{shape: doc, label: chunk_documents_processor_1}
	chunk_documents_processor_2-subject@{shape: doc, label: chunk_documents_processor_2}
	chunk_documents_task_1-subject@{shape: doc, label: chunk_documents_task_1}
	doc_embeddings-subject@{shape: doc, label: doc_embeddings}
	documents-subject@{shape: doc, label: documents}
	embed_documents_processor_1-subject@{shape: doc, label: embed_documents_processor_1}
	embed_query_processor_1-subject@{shape: doc, label: embed_query_processor_1}
	join_scores_chunks_processor_1-subject@{shape: doc, label: join_scores_chunks_processor_1}
	message_aggregator_1-subject@{shape: doc, label: message_aggregator_1}
	message_aggregator_2-subject@{shape: doc, label: message_aggregator_2}
	messages-subject@{shape: doc, label: messages}
	q_embeddings-subject@{shape: doc, label: q_embeddings}
	queries-subject@{shape: doc, label: queries}
	rel_sim_processor_1-subject@{shape: doc, label: rel_sim_processor_1}
	sort_scores_processor_1-subject@{shape: doc, label: sort_scores_processor_1}
	tmp_scores-subject@{shape: doc, label: tmp_scores}
	tmp_scores_chunks_join-subject@{shape: doc, label: tmp_scores_chunks_join}
	top_k-subject@{shape: doc, label: top_k}
	top_k_processor_1-subject@{shape: doc, label: top_k_processor_1}
	user_messages-subject@{shape: doc, label: user_messages}
	chat_processor_1-publish@{shape: fork}
	chunk_documents_processor_1-publish@{shape: fork}
	chunk_documents_processor_2-publish@{shape: fork}
	embed_documents_processor_1-publish@{shape: fork}
	embed_query_processor_1-publish@{shape: fork}
	join_scores_chunks_processor_1-publish@{shape: fork}
	message_aggregator_1-publish@{shape: fork}
	message_aggregator_2-publish@{shape: fork}
	rel_sim_processor_1-publish@{shape: fork}
	session_context_1-publish@{shape: fork}
	sort_scores_processor_1-publish@{shape: fork}
	top_k_processor_1-publish@{shape: fork}
	chat_processor_1-subscribe@{shape: diamond, label: All}
	chunk_documents_processor_1-subscribe@{shape: diamond, label: All}
	chunk_documents_processor_2-subscribe@{shape: diamond, label: All}
	embed_documents_processor_1-subscribe@{shape: diamond, label: All}
	embed_query_processor_1-subscribe@{shape: diamond, label: All}
	join_scores_chunks_processor_1-subscribe@{shape: diamond, label: All}
	message_aggregator_1-subscribe@{shape: diamond, label: All}
	message_aggregator_2-subscribe@{shape: diamond, label: All}
	rel_sim_processor_1-subscribe@{shape: diamond, label: All}
	session_context_1-subscribe@{shape: diamond, label: All}
	sort_scores_processor_1-subscribe@{shape: diamond, label: All}
	top_k_processor_1-subscribe@{shape: diamond, label: All}
```

Under the hood, the states of the application are determined by the subjects that are subscribed to and published on by the User, TEI, Retrieval, and TGI tasks. Each task is composed of one or more processes that are chained together to execute the task. Each processor listens for changes on their subscribed subjects and publishes their results to subjects. Each task runs once the subscription criteria for all of its child processors are satisfied.

At each superstep of the session, subscribed subjects are allocated to tasks, tasks are ran in parallel, and the subjects for which tasks publish on are updated sequentially.

While not shown in the flowchart above, each processor subscribes to a special subject usually called the config which specifies all of the parameters for the processor. And like any other subject, the config can also be updated dynamically during the execution of the session.

The decision to chain multiple processors into a single task or to allocated each processor to its own task is up to the needs of the user. Chaining multple processors can be more performant and efficient because fewer subscription and publishing copies and updates, respectively are needed. However, allocating each processor to its own task is easier to debug since the output of each task can be easily verified. Also, any processor that requires an external API call has to be allocated to its own task as chaining of streams breaks the poll on the external API call.

## Next steps

The [Document RAG Agent Session Plan](https://github.com/biom8er/phymes/blob/main/phymes-agents/src/session_plans/document_rag_agent_session.rs) comes with a number of default configurations including the model, number of tokens to sample, temperature of sampling, etc. that can be modified by the user. The session plan can be used with embedded Candle models or OpenAI API endpoints for token services.