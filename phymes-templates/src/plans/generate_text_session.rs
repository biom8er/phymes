/// A session for generating text in a structured format from a prompt
///   with support for tools
///
/// # Notes
pub struct GenerateTextSession<'a> {
    /// Session
    pub network_name: &'a str,
    /// The Asset to use for Text Generation and related parameters
    pub candle_asset: Option<String>,
    pub openai_asset: Option<String>,
    pub weights_config_file: Option<String>,
    pub weights_file: Option<String>,
    pub tokenizer_file: Option<String>,
    pub tokenizer_config_file: Option<String>,
    pub api_url: Option<String>,
    /// The processor to use for text generation
    pub chat_processor: &'a str,
}

impl<'a> Default for GenerateTextSession<'a> {
    fn default() -> Self {
        let (
            candle_asset,
            openai_asset,
            weights_config_file,
            weights_file,
            tokenizer_file,
            tokenizer_config_file,
            api_url,
        ) = if cfg!(feature = "hf_hub") {
            (
                Some("QwenV2p5_1p5bChat".to_string()),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        } else if cfg!(all(feature = "api", not(feature = "candle"))) {
            (
                None,
                Some("MetaLlamaV3p2_1B".to_string()),
                None,
                None,
                None,
                None,
                Some("http://0.0.0.0:8000/v1".to_string()),
            )
        } else {
            (
                Some("SmolLM2_135MChat".to_string()),
                None,
                Some(format!(
                    "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                Some(format!(
                    "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                Some(format!(
                    "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                Some(format!(
                    "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                None,
            )
        };
        let chat_processor = if cfg!(all(feature = "api", not(feature = "candle"))) {
            "OpenAIChatProcessor"
        } else {
            "CandleChatProcessor"
        };
        Self {
            network_name: "generate_text_session",
            candle_asset,
            openai_asset,
            weights_config_file,
            weights_file,
            tokenizer_config_file,
            tokenizer_file,
            api_url,
            chat_processor,
        }
    }
}

impl<'a> GenerateTextSession<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        network_name: &'a str,
        candle_asset: Option<String>,
        openai_asset: Option<String>,
        weights_config_file: Option<String>,
        weights_file: Option<String>,
        tokenizer_file: Option<String>,
        tokenizer_config_file: Option<String>,
        api_url: Option<String>,
    ) -> Self {
        let chat_processor = if cfg!(all(feature = "api", not(feature = "candle"))) {
            "OpenAIChatProcessor"
        } else {
            "CandleChatProcessor"
        };
        GenerateTextSession {
            network_name,
            candle_asset,
            openai_asset,
            weights_config_file,
            weights_file,
            tokenizer_file,
            tokenizer_config_file,
            api_url,
            chat_processor,
        }
    }
    fn generate_text_inference_p(&self) -> String {
        let mut lines = Vec::new();
        if let Some(candle_asset) = &self.candle_asset {
            let line = format!(r#"Utf8 candle_asset "{candle_asset}""#);
            lines.push(line);
        }
        if let Some(openai_asset) = &self.openai_asset {
            let line = format!(r#"Utf8 openai_asset "{openai_asset}""#);
            lines.push(line);
        }
        if let Some(tokenizer_config_file) = &self.tokenizer_config_file {
            let line = format!(r#"Utf8 tokenizer_config_file "{tokenizer_config_file}""#);
            lines.push(line);
        }
        if let Some(tokenizer_file) = &self.tokenizer_file {
            let line = format!(r#"Utf8 tokenizer_file "{tokenizer_file}""#);
            lines.push(line);
        }
        if let Some(weights_config_file) = &self.weights_config_file {
            let line = format!(r#"Utf8 weights_config_file "{weights_config_file}""#);
            lines.push(line);
        }
        if let Some(weights_file) = &self.weights_file {
            let line = format!(r#"Utf8 weights_file "{weights_file}""#);
            lines.push(line);
        }
        if let Some(api_url) = &self.api_url {
            let line = format!(r#"Utf8 api_url "{api_url}""#);
            lines.push(line);
        }
        lines.join("\n\t\t")
    }
    fn parse_generated_text_p(&self) -> String {
        let mut lines = Vec::new();
        if let Some(candle_asset) = &self.candle_asset {
            let line = format!(r#"Utf8 candle_asset "{candle_asset}""#);
            lines.push(line);
        }
        if let Some(openai_asset) = &self.openai_asset {
            let line = format!(r#"Utf8 openai_asset "{openai_asset}""#);
            lines.push(line);
        }
        lines.join("\n\t\t")
    }
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        format!(
            r#"flowchart TD
	%% ------------------------------------------------------------------------------
	%% Message aggregation for text generation
	%% ------------------------------------------------------------------------------
	subgraph aggregate_messages_generate_text_t
		UserMessages-subject-->|AllRecordBatches|aggregate_messages_generate_text_p-subscribe
		ToolMessages-subject-.->|LastRecordBatch|aggregate_messages_generate_text_p-subscribe
		SessionErrors-subject-.->|LastRecordBatch|aggregate_messages_generate_text_p-subscribe
		AssistantMessages-subject-->|AllRecordBatches|aggregate_messages_generate_text_p-subscribe
		aggregate_messages_generate_text_p-subscribe-->aggregate_messages_generate_text_p-processor
		aggregate_messages_generate_text_p-processor-->aggregate_messages_generate_text_p-publish
		aggregate_messages_generate_text_p-publish-->|Replace|aggregate_messages_generate_text_s-subject
	end
	generate_text_r-rt@{{shape: subproc, label: generate_text_r}}
	generate_text_r-rt-->aggregate_messages_generate_text_t
	UserMessages-subject@{{shape: doc, label: UserMessages}}
	ToolMessages-subject@{{shape: doc, label: ToolMessages}}
	SessionErrors-subject@{{shape: doc, label: SessionErrors}}
	AssistantMessages-subject@{{shape: doc, label: AssistantMessages}}
	aggregate_messages_generate_text_p-processor@{{shape: rect, label: AggregatorProcessor}}
	aggregate_messages_generate_text_p-publish@{{shape: fork}}
	aggregate_messages_generate_text_p-subscribe@{{shape: diamond, label: ChatContentSubscribe}}
	aggregate_messages_generate_text_s-subject@{{shape: doc, label: aggregate_messages_generate_text_s}}
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
	aggregate_messages_user_interface_p-processor@{{shape: rect, label: AggregatorProcessor}}
	aggregate_messages_user_interface_p-publish@{{shape: fork}}
	aggregate_messages_user_interface_p-subscribe@{{shape: diamond, label: Any}}
	AggregatedMessages-subject@{{shape: doc, label: AggregatedMessages}}
	%% ------------------------------------------------------------------------------
	%% Text generation
	%% ------------------------------------------------------------------------------
	subgraph generate_text_inference_t
		aggregate_messages_generate_text_s-subject-.->|AllRecordBatches|generate_text_inference_p-subscribe
		Tools-subject-->|AllRecordBatches|generate_text_inference_p-subscribe
		generate_text_inference_p-subscribe-->generate_text_inference_p-processor
		generate_text_inference_p-processor-->generate_text_inference_p-publish
		generate_text_inference_p-publish-->|Replace|generate_text_inference_s-subject
	end
	generate_text_r-rt-->generate_text_inference_t
	Tools-subject@{{shape: doc, label: Tools}}
	generate_text_inference_p-processor@{{shape: rect, label: {}}}
	generate_text_inference_p-publish@{{shape: fork}}
	generate_text_inference_p-subscribe@{{shape: diamond, label: All}}
	generate_text_inference_s-subject@{{shape: doc, label: generate_text_inference_s}}
	%% ------------------------------------------------------------------------------
	%% Parse generated text
	%% ------------------------------------------------------------------------------
	subgraph parse_generated_text_t
		generate_text_inference_s-subject-.->|AllRecordBatches|parse_generated_text_p-subscribe
		parse_generated_text_p-subscribe-->parse_generated_text_p-processor
		parse_generated_text_p-processor-->parse_generated_text_p-publish
		parse_generated_text_p-publish-->|Extend|AssistantMessages-subject
	end
	generate_text_r-rt-->parse_generated_text_t
	parse_generated_text_p-processor@{{shape: rect, label: MessageParserProcessor}}
	parse_generated_text_p-publish@{{shape: fork}}
	parse_generated_text_p-subscribe@{{shape: diamond, label: All}}
	%% ------------------------------------------------------------------------------"#,
            self.chat_processor
        )
    }

    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        format!(
            r#"erDiagram
    UserMessages["UserMessages"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
	ToolMessages["ToolMessages"] {{
	    Utf8 role
	    Utf8 content
	    Int64 timestamp
	}}
    SessionErrors["SessionErrors"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    AssistantMessages["AssistantMessages"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    aggregate_messages_generate_text_p["aggregate_messages_generate_text_p"] {{
        Boolean asc "true"
        Boolean cpu "false"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Sort"
        Utf8 lhs_stream "Accumulate"
    }}
    aggregate_messages_user_interface_p["aggregate_messages_user_interface_p"] {{
        Boolean asc "true"
        Boolean cpu "false"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Sort"
        Utf8 lhs_stream "Accumulate"
    }}
    AggregatedMessages["AggregatedMessages"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    Tools["Tools"] {{
        Utf8 tool_id
        Utf8 tool
    }}
    aggregate_messages_generate_text_s["aggregate_messages_generate_text_s"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    generate_text_inference_p["generate_text_inference_p"] {{
        Boolean cpu "false"
        Float64 frequency_penalty "0.0"
        Int64 max_tokens "1000"
        Utf8 messages "aggregate_messages_generate_text_s"
        Utf8 tools "Tools"
        Int64 repeat_last_n "64"
        Float64 repeat_penalty "1.1"
        Int64 seed "299792458"
        Boolean split_prompt "false"
        Float64 temperature "0.8"
        {}
    }}
    generate_text_inference_s["generate_text_inference_s"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    parse_generated_text_p["parse_generated_text_p"] {{
        Boolean cpu "false"
        Float64 frequency_penalty "0.0"
        Utf8 messages "generate_text_inference_s"
        Int64 max_tokens "1000"
        Int64 repeat_last_n "64"
        Float64 repeat_penalty "1.1"
        Int64 seed "299792458"
        Boolean split_prompt "false"
        Float64 temperature "0.8"
        {}
    }}"#,
            self.generate_text_inference_p(),
            self.parse_generated_text_p()
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait,
        SubjectPlan, SubjectPlanBuilderTrait, SubjectTrait,
    };
    use phymes_data::{AvailableOperators, ToolTrait};
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait, create_message_map};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, SessionStream,
    };
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait,
        create_tools_record_batch,
    };
    use phymes_streams::ChatBuilderTraitExt;
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_no_tools() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::default();
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(generate_text_session.network_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        let message_map = create_message_map(vec![chat_message]);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // Run the session
            let session_stream = SessionStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

            assert_eq!(response.len(), 0);

            {
                // Test supsersteps
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AssistantMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_str("content");
                let assistant_content = column.first().unwrap();
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AggregatedMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 2);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                assert_eq!(column.last().unwrap(), assistant_content);
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "aggregate_messages_generate_text_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("aggregate_messages_generate_text_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "generate_text_inference_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("generate_text_inference_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert!(subject.count_rows() > 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_tool_call() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::new(
            "generate_text_session",
            Some("QwenV2p5_1p5bChat".to_string()),
            None,
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            None,
        );
        let mut network_builder = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(generate_text_session.network_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
        .add_next_supersteps()?;

        // Add the target tool subjects to the session for testing
        let mut subjects = network_builder.subjects.take().unwrap();
        let tool = AvailableSubjects::Bytes
            .to_subject(Some(AvailableOperators::Sort.to_string().as_str()), None)?;
        subjects.push(SubjectPlan::get_builder().with_subject(tool).build()?);
        let tool = AvailableSubjects::Bytes.to_subject(
            Some(AvailableOperators::HumanInTheLoop.to_string().as_str()),
            None,
        )?;
        subjects.push(SubjectPlan::get_builder().with_subject(tool).build()?);

        let (network, session_messages) = network_builder
            .with_subjects(subjects)
            // DM: needed for the session to build, the target tool subjects need to be called by at least 1 task
            .add_session_interface(Some(&[
                AvailableOperators::Sort.to_string().as_str(),
                AvailableOperators::HumanInTheLoop.to_string().as_str(),
            ]))?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Tools data
        let tool_ids = vec![
            AvailableOperators::Sort.to_string(),
            AvailableOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableOperators::Sort.to_json_tool_schema(),
            AvailableOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        let table = SubjectBuilder::new()
            .with_name(AvailableSubjects::Tools.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tool_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&Publication::Extend {
                subject_name: table.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        // User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        let message_map = create_message_map(vec![tool_message, chat_message]);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // Run the session
            let session_stream = SessionStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

            assert_eq!(response.len(), 1); // Due to session interface

            {
                // Test supsersteps
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AggregatedMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "aggregate_messages_generate_text_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("aggregate_messages_generate_text_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "generate_text_inference_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("generate_text_inference_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert!(subject.count_rows() > 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableOperators::HumanInTheLoop.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableOperators::Sort.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(AvailableOperators::Sort.to_string().as_str())
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject
                    .get_column_as_vec_nested_primitive::<u8>("bytes")?
                    .into_iter()
                    .map(|b| String::from_utf8(b).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    column.first().unwrap(),
                    &"{\"lhs_name\":\"available_data_1\",\"lhs_pk\":\"lhs_pk\",\"lhs_values\":[\"score\"],\"operator\":\"Sort\"}"
                );
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_tool_response() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::default();
        let mut network_builder = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(generate_text_session.network_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
        .add_next_supersteps()?;

        // Add the target tool subjects to the session for testing
        let mut subjects = network_builder.subjects.take().unwrap();
        let tool = AvailableSubjects::Bytes
            .to_subject(Some(AvailableOperators::Sort.to_string().as_str()), None)?;
        subjects.push(SubjectPlan::get_builder().with_subject(tool).build()?);
        let tool = AvailableSubjects::Bytes.to_subject(
            Some(AvailableOperators::HumanInTheLoop.to_string().as_str()),
            None,
        )?;
        subjects.push(SubjectPlan::get_builder().with_subject(tool).build()?);
        let (network, session_messages) = network_builder
            .with_subjects(subjects)
            // DM: needed for the session to build, the target tool subjects need to be called by at least 1 task
            .add_session_interface(Some(&[
                AvailableOperators::Sort.to_string().as_str(),
                AvailableOperators::HumanInTheLoop.to_string().as_str(),
            ]))?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Tools data
        let tool_ids = vec![
            AvailableOperators::Sort.to_string(),
            AvailableOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableOperators::Sort.to_json_tool_schema(),
            AvailableOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        let table = SubjectBuilder::new()
            .with_name(AvailableSubjects::Tools.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tool_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&Publication::Extend {
                subject_name: table.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        // User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        // Tool response
        let tool = AvailableInterfaceSubjects::ToolMessages.to_subject_builder(None)
            .append_new_user_query_str("[{\"lhs_pk\":\"c\",\"score\":1.0}, {\"lhs_pk\":\"b\",\"score\":2.0}, {\"lhs_pk\":\"a\",\"score\":3.0}]", "tool")?
            .build()?;
        let tool_response = IPCMessage::get_builder()
            .with_message(tool.to_ipc_stream()?)
            .with_subject(tool.get_name())
            .with_update(&Publication::Extend {
                subject_name: tool.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        let message_map = create_message_map(vec![tool_message, chat_message, tool_response]);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // Run the session
            let session_stream = SessionStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

            assert_eq!(response.len(), 2);

            {
                // Test supsersteps
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AssistantMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_str("content");
                let assistant_content = column.first().unwrap();
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AggregatedMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 2);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                assert_eq!(column.last().unwrap(), assistant_content);
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "aggregate_messages_generate_text_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("aggregate_messages_generate_text_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 2);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                assert_eq!(column.last().unwrap(), &"tool");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                assert_eq!(
                    column.last().unwrap(),
                    &"[{\"lhs_pk\":\"c\",\"score\":1.0}, {\"lhs_pk\":\"b\",\"score\":2.0}, {\"lhs_pk\":\"a\",\"score\":3.0}]"
                );
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "generate_text_inference_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("generate_text_inference_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert!(subject.count_rows() > 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableOperators::Sort.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableOperators::HumanInTheLoop.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_generate_text_session_error_response() -> Result<()> {
        // Initialize the session
        let generate_text_session = GenerateTextSession::new(
            "generate_text_session",
            Some("QwenV2p5_1p5bChat".to_string()),
            None,
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            None,
        );
        let mut network_builder = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(generate_text_session.network_name)
        .add_processor_subjects()?
        .with_diagnostics(true)
        .add_next_tasks()?
        .add_next_supersteps()?;

        // Add the target tool subjects to the session for testing
        let mut subjects = network_builder.subjects.take().unwrap();
        let tool = AvailableSubjects::Bytes
            .to_subject(Some(AvailableOperators::Sort.to_string().as_str()), None)?;
        subjects.push(SubjectPlan::get_builder().with_subject(tool).build()?);
        let tool = AvailableSubjects::Bytes.to_subject(
            Some(AvailableOperators::HumanInTheLoop.to_string().as_str()),
            None,
        )?;
        subjects.push(SubjectPlan::get_builder().with_subject(tool).build()?);
        let (network, session_messages) = network_builder
            .with_subjects(subjects)
            // DM: needed for the session to build, the target tool subjects need to be called by at least 1 task
            .add_session_interface(Some(&[
                AvailableOperators::Sort.to_string().as_str(),
                AvailableOperators::HumanInTheLoop.to_string().as_str(),
            ]))?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Tools data
        let tool_ids = vec![
            AvailableOperators::Sort.to_string(),
            AvailableOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableOperators::Sort.to_json_tool_schema(),
            AvailableOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        let table = SubjectBuilder::new()
            .with_name(AvailableSubjects::Tools.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tool_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(table.get_name())
            .with_update(&Publication::Extend {
                subject_name: table.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        // User message
        let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        // Error response
        let tool = AvailableSubjects::SessionErrors.to_subject_builder(None)
            .append_new_user_query_str("lhs_name `available_data_1` was not found. Available options are [`available_data_0`, `available_data_2`, `available_data_3`].", "tool")?
            .build()?;
        let tool_response = IPCMessage::get_builder()
            .with_message(tool.to_ipc_stream()?)
            .with_subject(tool.get_name())
            .with_update(&Publication::Extend {
                subject_name: tool.get_name().to_string(),
            })
            .with_publisher(generate_text_session.network_name)
            .make_name()?
            .build()?;

        let message_map = create_message_map(vec![tool_message, chat_message, tool_response]);
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // Run the session
            let session_stream = SessionStream::new(message_map, Arc::clone(&network_arc));
            let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

            assert_eq!(response.len(), 1);

            {
                // Test supsersteps
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AssistantMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                println!("{}", String::from_utf8(subject.to_csv(b',', true)?)?);
                assert_eq!(subject.count_rows(), 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_str("content");
                let assistant_content = column.first().unwrap();
                // assert!(assistant_content.contains("available_data_0")); //DM : response does not always contain the available subjects
                assert!(assistant_content.contains("available_data_1"));
                // assert!(assistant_content.contains("available_data_2"));
                // assert!(assistant_content.contains("available_data_3"));
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name(
                        AvailableInterfaceSubjects::AggregatedMessages
                            .to_string()
                            .as_str(),
                    )
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 2);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                assert_eq!(column.last().unwrap(), assistant_content);
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "aggregate_messages_generate_text_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("aggregate_messages_generate_text_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert_eq!(subject.count_rows(), 2);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"user");
                assert_eq!(column.last().unwrap(), &"tool");
                let column = subject.get_column_as_vec_str("content");
                assert_eq!(
                    column.first().unwrap(),
                    &"Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`."
                );
                assert_eq!(
                    column.last().unwrap(),
                    &"lhs_name `available_data_1` was not found. Available options are [`available_data_0`, `available_data_2`, `available_data_3`]."
                );
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: "generate_text_inference_s".to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                let subject = Subject::get_builder()
                    .with_name("generate_text_inference_s")
                    .with_record_batches(batches)?
                    .build()?;
                assert!(subject.count_rows() > 1);
                let column = subject.get_column_as_vec_str("role");
                assert_eq!(column.first().unwrap(), &"assistant");
                assert_eq!(column.last().unwrap(), &"assistant");
                let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
                for t in column {
                    assert!(t > 0);
                }
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableOperators::Sort.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
                let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                    subject_name: AvailableOperators::HumanInTheLoop.to_string(),
                }
                .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
                .unwrap()
                .try_collect()
                .await?;
                assert_eq!(batches.len(), 0);
            }
        }
        Ok(())
    }
}
