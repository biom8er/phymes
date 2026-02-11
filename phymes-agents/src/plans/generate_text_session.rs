/// A session for generating text in a structured format from a prompt
///   with support for tools
///
/// # Notes
pub struct GenerateTextSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl<'a> Default for GenerateTextSession<'a> {
    fn default() -> Self {
        Self {
            session_context_name: "generate_text_session",
        }
    }
}

impl<'a> GenerateTextSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	%% ------------------------------------------------------------------------------
	%% Message aggregation for text generation
	%% ------------------------------------------------------------------------------
	subgraph aggregate_messages_generate_text_t
		UserMessages-subject-->|FullTable|aggregate_messages_generate_text_p-subscribe
		ToolMessages-subject-.->|LastRecordBatch|aggregate_messages_generate_text_p-subscribe
		SessionErrors-subject-.->|LastRecordBatch|aggregate_messages_generate_text_p-subscribe
		AssistantMessages-subject-->|FullTable|aggregate_messages_generate_text_p-subscribe
		aggregate_messages_generate_text_p-subscribe-->aggregate_messages_generate_text_p-processor
		aggregate_messages_generate_text_p-processor-->aggregate_messages_generate_text_p-publish
		aggregate_messages_generate_text_p-publish-->|Replace|aggregate_messages_generate_text_s-subject
	end
	generate_text_r-rt@{shape: subproc, label: generate_text_r}
	generate_text_r-rt-->aggregate_messages_generate_text_t
	UserMessages-subject@{shape: doc, label: UserMessages}
	ToolMessages-subject@{shape: doc, label: ToolMessages}
	SessionErrors-subject@{shape: doc, label: SessionErrors}
	AssistantMessages-subject@{shape: doc, label: AssistantMessages}
	aggregate_messages_generate_text_p-processor@{shape: rect, label: MessageAggregatorProcessor}
	aggregate_messages_generate_text_p-publish@{shape: fork}
	aggregate_messages_generate_text_p-subscribe@{shape: diamond, label: ChatContentSubscribe}
	aggregate_messages_generate_text_s-subject@{shape: doc, label: aggregate_messages_generate_text_s}
	%% ------------------------------------------------------------------------------
	%% Message aggregation for the UI
	%% ------------------------------------------------------------------------------
	subgraph aggregate_messages_user_interface_t
		UserMessages-subject-.->|LastRecordBatch|aggregate_messages_user_interface_p-subscribe
		AssistantMessages-subject-.->|LastRecordBatch|aggregate_messages_user_interface_p-subscribe
		aggregate_messages_user_interface_p-subscribe-->aggregate_messages_user_interface_p-processor
		aggregate_messages_user_interface_p-processor-->aggregate_messages_user_interface_p-publish
		aggregate_messages_user_interface_p-publish-->|Extend|AggregatedMessages-subject
	end
	generate_text_r-rt-->aggregate_messages_user_interface_t
	aggregate_messages_user_interface_p-processor@{shape: rect, label: MessageAggregatorProcessor}
	aggregate_messages_user_interface_p-publish@{shape: fork}
	aggregate_messages_user_interface_p-subscribe@{shape: diamond, label: Any}
	AggregatedMessages-subject@{shape: doc, label: AggregatedMessages}
	%% ------------------------------------------------------------------------------
	%% Text generation
	%% ------------------------------------------------------------------------------
	subgraph generate_text_inference_t
		aggregate_messages_generate_text_s-subject-.->|FullTable|generate_text_inference_p-subscribe
		Tools-subject-->|FullTable|generate_text_inference_p-subscribe
		generate_text_inference_p-subscribe-->generate_text_inference_p-processor
		generate_text_inference_p-processor-->generate_text_inference_p-publish
		generate_text_inference_p-publish-->|Replace|generate_text_inference_s-subject
	end
	generate_text_inference_r-rt@{shape: subproc, label: generate_text_inference_r}
	generate_text_inference_r-rt-->generate_text_inference_t
	Tools-subject@{shape: doc, label: Tools}
	generate_text_inference_p-processor@{shape: rect, label: CandleChatProcessor}
	generate_text_inference_p-publish@{shape: fork}
	generate_text_inference_p-subscribe@{shape: diamond, label: All}
	generate_text_inference_s-subject@{shape: doc, label: generate_text_inference_s}
	%% ------------------------------------------------------------------------------
	%% Parse generated text
	%% ------------------------------------------------------------------------------
	subgraph parse_generated_text_t
		generate_text_inference_s-subject-.->|FullTable|parse_generated_text_p-subscribe
		parse_generated_text_p-subscribe-->parse_generated_text_p-processor
		parse_generated_text_p-processor-->parse_generated_text_p-publish
		parse_generated_text_p-publish-->|Extend|AssistantMessages-subject
	end
	generate_text_r-rt-->parse_generated_text_t
	parse_generated_text_p-processor@{shape: rect, label: MessageParserProcessor}
	parse_generated_text_p-publish@{shape: fork}
	parse_generated_text_p-subscribe@{shape: diamond, label: All}
	%% ------------------------------------------------------------------------------"#
	}

    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    UserMessages["UserMessages"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
	ToolMessages["ToolMessages"] {
	    Utf8 role
	    Utf8 content
	    Int64 timestamp
	}
    SessionErrors["SessionErrors"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    AssistantMessages["AssistantMessages"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    aggregate_messages_generate_text_p["aggregate_messages_generate_text_p"] {
        Boolean asc "true"
        Boolean cpu "false"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Sort"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    aggregate_messages_user_interface_p["aggregate_messages_user_interface_p"] {
        Boolean asc "true"
        Boolean cpu "false"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Sort"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    AggregatedMessages["AggregatedMessages"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    Tools["Tools"] {
        Utf8 tool_id
        Utf8 tool
    }
    aggregate_messages_generate_text_s["aggregate_messages_generate_text_s"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    generate_text_inference_p["generate_text_inference_p"] {
        Utf8 candle_asset "QwenV2p5_1p5bChat"
        Boolean cpu "false"
        Float64 frequency_penalty "0.0"
        Int64 max_tokens "1000"
        Utf8 messages "aggregate_messages_generate_text_s"
        Int64 repeat_last_n "64"
        Float64 repeat_penalty "1.1"
        Int64 seed "299792458"
        Boolean split_prompt "false"
        Float64 temperature "0.8"
        Utf8 tokenizer_config_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json"
        Utf8 tokenizer_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json"
        Utf8 tools "Tools"
        Utf8 weights_config_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json"
        Utf8 weights_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    }
    generate_text_inference_s["generate_text_inference_s"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    parse_generated_text_p["parse_generated_text_p"] {
        Utf8 candle_asset "QwenV2p5_1p5bChat"
        Boolean cpu "false"
        Float64 frequency_penalty "0.0"
        Int64 max_tokens "1000"
        Utf8 messages "generate_text_inference_s"
        Int64 repeat_last_n "64"
        Float64 repeat_penalty "1.1"
        Int64 seed "299792458"
        Boolean split_prompt "false"
        Float64 temperature "0.8"
        Utf8 tokenizer_config_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json"
        Utf8 tokenizer_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json"
        Utf8 weights_config_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json"
        Utf8 weights_file "/home/dmccloskey/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    }"#
	}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        AvailableSubjects, AvailableSubjectsTrait, BuildableTrait, BuilderTrait, ChatBuilderTraitExt, IPCMessage, MappableTrait, MessageBuilderTrait, TableBuilder, TableBuilderTrait, TablePublication, TableTrait, create_documents_batch, create_documents_embeddings_batch, create_query_embeddings_batch, create_tools_record_batch
    };
    use phymes_data::{AvailableCandleOperators, ToolTrait};
    use phymes_diagnostics::HashMap;

    use crate::{
        AvailableInterfaceSubjects, SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream, create_message_map
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_no_tools() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(generate_text_session.as_mermaid_erdiagram(), false, true)?
        .with_name(generate_text_session.session_context_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
		.add_next_supersteps()?
		.build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

		// User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&TablePublication::Extend {
                table_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;
		
        let message_map = create_message_map(vec![chat_message]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // {
        //     // Debug any errors
        //     let subjects_reading = session_ctx_arc.read();
        //     let table_reading = subjects_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        // }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AssistantMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_str("content");
			let assistant_content = column.first().unwrap();
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::ToolMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AggregatedMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 2);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            assert_eq!(column.last().unwrap(), assistant_content);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("aggregate_messages_generate_text_s")
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("generate_text_inference_s")
                .unwrap()
                .read();
			assert!(table_reading.count_rows() > 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_tool_call() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(generate_text_session.as_mermaid_erdiagram(), false, true)?
        .with_name(generate_text_session.session_context_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
		.add_next_supersteps()?
		.build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

		// Add the target tool subjects to the session for testing
		let _ = session_ctx_arc.write().state.insert(
			AvailableCandleOperators::Sort.to_string(), 
			Arc::new(RwLock::new(AvailableSubjects::Bytes.to_table(Some(AvailableCandleOperators::Sort.to_string().as_str()), None)?))
		);
		let _ = session_ctx_arc.write().state.insert(
			AvailableCandleOperators::HumanInTheLoop.to_string(), 
			Arc::new(RwLock::new(AvailableSubjects::Bytes.to_table(Some(AvailableCandleOperators::HumanInTheLoop.to_string().as_str()), None)?))
		);

		// Tools data
        let tool_ids = vec![
            AvailableCandleOperators::Sort.to_string(),
            AvailableCandleOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableCandleOperators::Sort.to_json_tool_schema(),
            AvailableCandleOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        let table = TableBuilder::new()
            .with_name(AvailableSubjects::Tools.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tool_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&TablePublication::Extend {
                table_name: table.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;

		// User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&TablePublication::Extend {
                table_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;
		
        let message_map = create_message_map(vec![tool_message, chat_message]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // {
        //     // Debug any errors
        //     let subjects_reading = session_ctx_arc.read();
        //     let table_reading = subjects_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        // }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AssistantMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::ToolMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AggregatedMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("aggregate_messages_generate_text_s")
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("generate_text_inference_s")
                .unwrap()
                .read();
			assert!(table_reading.count_rows() > 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get(AvailableCandleOperators::Sort.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("values");
            assert_eq!(column.first().unwrap(), &"{\"arguments\":{\"lhs_name\":\"available_data_1\",\"lhs_pk\":\"lhs_pk\",\"lhs_values\":[\"score\"]},\"name\":\"Sort\"}");
            let table_reading = session_reading
                .get_states()
                .get(AvailableCandleOperators::HumanInTheLoop.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_tool_response() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(generate_text_session.as_mermaid_erdiagram(), false, true)?
        .with_name(generate_text_session.session_context_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
		.add_next_supersteps()?
		.build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

		// Add the target tool subjects to the session for testing
		let _ = session_ctx_arc.write().state.insert(
			AvailableCandleOperators::Sort.to_string(), 
			Arc::new(RwLock::new(AvailableSubjects::Bytes.to_table(Some(AvailableCandleOperators::Sort.to_string().as_str()), None)?))
		);
		let _ = session_ctx_arc.write().state.insert(
			AvailableCandleOperators::HumanInTheLoop.to_string(), 
			Arc::new(RwLock::new(AvailableSubjects::Bytes.to_table(Some(AvailableCandleOperators::HumanInTheLoop.to_string().as_str()), None)?))
		);

		// Tools data
        let tool_ids = vec![
            AvailableCandleOperators::Sort.to_string(),
            AvailableCandleOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableCandleOperators::Sort.to_json_tool_schema(),
            AvailableCandleOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        let table = TableBuilder::new()
            .with_name(AvailableSubjects::Tools.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tool_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&TablePublication::Extend {
                table_name: table.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;

		// User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&TablePublication::Extend {
                table_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;

		// Tool response
        let tool = AvailableInterfaceSubjects::ToolMessages.to_table_builder(None)
            .append_new_user_query_str("[{\"lhs_pk\":\"c\",\"score\":1.0}, {\"lhs_pk\":\"b\",\"score\":2.0}, {\"lhs_pk\":\"a\",\"score\":3.0}]", "tool")?
            .build()?;
        let tool_response = IPCMessage::get_builder()
            .with_message(tool.to_ipc_stream()?)
            .with_subject(tool.get_name())
            .with_update(&TablePublication::Extend {
                table_name: tool.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;
		
        let message_map = create_message_map(vec![tool_message, chat_message, tool_response]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // {
        //     // Debug any errors
        //     let subjects_reading = session_ctx_arc.read();
        //     let table_reading = subjects_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        // }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AssistantMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_str("content");
			let assistant_content = column.first().unwrap();
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AggregatedMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 2);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            assert_eq!(column.last().unwrap(), assistant_content);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("aggregate_messages_generate_text_s")
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 2);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            assert_eq!(column.last().unwrap(), &"tool");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            assert_eq!(column.last().unwrap(), &"[{\"lhs_pk\":\"c\",\"score\":1.0}, {\"lhs_pk\":\"b\",\"score\":2.0}, {\"lhs_pk\":\"a\",\"score\":3.0}]");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("generate_text_inference_s")
                .unwrap()
                .read();
			assert!(table_reading.count_rows() > 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get(AvailableCandleOperators::Sort.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
            let table_reading = session_reading
                .get_states()
                .get(AvailableCandleOperators::HumanInTheLoop.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_error_response() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(generate_text_session.as_mermaid_erdiagram(), false, true)?
        .with_name(generate_text_session.session_context_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
		.add_next_supersteps()?
		.build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

		// Add the target tool subjects to the session for testing
		let _ = session_ctx_arc.write().state.insert(
			AvailableCandleOperators::Sort.to_string(), 
			Arc::new(RwLock::new(AvailableSubjects::Bytes.to_table(Some(AvailableCandleOperators::Sort.to_string().as_str()), None)?))
		);
		let _ = session_ctx_arc.write().state.insert(
			AvailableCandleOperators::HumanInTheLoop.to_string(), 
			Arc::new(RwLock::new(AvailableSubjects::Bytes.to_table(Some(AvailableCandleOperators::HumanInTheLoop.to_string().as_str()), None)?))
		);

		// Tools data
        let tool_ids = vec![
            AvailableCandleOperators::Sort.to_string(),
            AvailableCandleOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableCandleOperators::Sort.to_json_tool_schema(),
            AvailableCandleOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        let table = TableBuilder::new()
            .with_name(AvailableSubjects::Tools.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tool_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&TablePublication::Extend {
                table_name: table.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;

		// User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&TablePublication::Extend {
                table_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;

		// Error response
        let tool = AvailableSubjects::SessionErrors.to_table_builder(None)
            .append_new_user_query_str("lhs_name `available_data_1` was not found. Available options are [`available_data_0`, `available_data_2`, `available_data_3`].", "tool")?
            .build()?;
        let tool_response = IPCMessage::get_builder()
            .with_message(tool.to_ipc_stream()?)
            .with_subject(tool.get_name())
            .with_update(&TablePublication::Extend {
                table_name: tool.get_name().to_string(),
            })
            .with_publisher(generate_text_session.session_context_name)
            .make_name()?
            .build()?;
		
        let message_map = create_message_map(vec![tool_message, chat_message, tool_response]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // {
        //     // Debug any errors
        //     let subjects_reading = session_ctx_arc.read();
        //     let table_reading = subjects_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        // }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AssistantMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_str("content");
			let assistant_content = column.first().unwrap();
            assert!(assistant_content.contains("available_data_0"));
            assert!(assistant_content.contains("available_data_1"));
            assert!(assistant_content.contains("available_data_2"));
            assert!(assistant_content.contains("available_data_3"));
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::AggregatedMessages.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 2);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            assert_eq!(column.last().unwrap(), assistant_content);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("aggregate_messages_generate_text_s")
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 2);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"user");
            assert_eq!(column.last().unwrap(), &"tool");
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(column.first().unwrap(), &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.");
            assert_eq!(column.last().unwrap(), &"lhs_name `available_data_1` was not found. Available options are [`available_data_0`, `available_data_2`, `available_data_3`].");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get("generate_text_inference_s")
                .unwrap()
                .read();
			assert!(table_reading.count_rows() > 1);
            let column = table_reading.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            assert_eq!(column.last().unwrap(), &"assistant");
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
			for t in column {
				assert!(t > 0);
			}
            let table_reading = session_reading
                .get_states()
                .get(AvailableCandleOperators::Sort.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
            let table_reading = session_reading
                .get_states()
                .get(AvailableCandleOperators::HumanInTheLoop.to_string().as_str())
                .unwrap()
                .read();
			assert_eq!(table_reading.count_rows(), 0);
			// assert_eq!(table_reading.count_rows(), 1);
			// let column = table_reading.get_column_as_vec_str("values");
            // assert_eq!(column.first().unwrap(), &"");

        }
        Ok(())
    }
}
