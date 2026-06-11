use phymes_network::{NetworkBuilder, NetworkBuilderMermaidTrait};
use phymes_subject::BuilderTrait;

use crate::{EmbedTextNetworkBuilder, ExtractPDFNetworkBuilder, GetPdfNetworkBuilderStaticWSubject, RetrieveTextNetworkBuilder};

/// Retrieve Text PDF network
pub struct RetrieveTextPDFNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
}

impl Default for RetrieveTextPDFNetworkBuilder {
    fn default() -> Self {
        // DM, todo: provide option to read from disc (WASM) instead of HTTP request (Non-WASM)
        // Get PDF network
        let retrieve_text_pdf_network_builder = GetPdfNetworkBuilderStaticWSubject::default().inner.build_dynamic();

        // Extract PDF session
        let extract_pdf_network_builder = ExtractPDFNetworkBuilder::default().inner.take().unwrap();
        let retrieve_text_pdf_network_builder = retrieve_text_pdf_network_builder
            .extend(extract_pdf_network_builder)
            .unwrap();

        // Embed text session
        let embed_text_network = EmbedTextNetworkBuilder::default();
        let embed_text_network_builder = NetworkBuilder::from_mermaid_flowchart(
            &embed_text_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &embed_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(embed_text_network.network_name);
        let retrieve_text_pdf_network_builder = retrieve_text_pdf_network_builder
            .extend(embed_text_network_builder)
            .unwrap();

        // Retrieve text session
        let retrieve_text_network = RetrieveTextNetworkBuilder::default();
        let retrieve_text_builder = NetworkBuilder::from_mermaid_flowchart(
            retrieve_text_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            retrieve_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(retrieve_text_network.network_name);
        let retrieve_text_pdf_network_builder = retrieve_text_pdf_network_builder
            .extend(retrieve_text_builder)
            .unwrap();

        RetrieveTextPDFNetworkBuilder {
            inner: Some(retrieve_text_pdf_network_builder.with_name("retrieve_text_pdf_network")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream};
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, create_chat_record_batch, create_queries_batch
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use crate::{
        DynamicTaskNetworkNames, extended_diagnostic_subjects, write_diagnostic_subjects_to_csv,
    };

    use super::*;

    // `cargo test -p phymes-templates test_open_alex_network_v_rust --features api,gpu,hf_hub --release -- --nocapture`
    // #[ignore = "In progress... Optimizing PDF and OWL parsing..."]
    #[tokio::test]
    async fn test_retrieve_text_pdf_network() -> Result<()> {
        // Initialize the session
        let retrieve_text_pdf_network_builder = RetrieveTextPDFNetworkBuilder::default().inner.take().unwrap();
        let network_name = retrieve_text_pdf_network_builder.name.clone().unwrap();
        let (network, session_messages) = retrieve_text_pdf_network_builder
            .with_runtime_env(
                RuntimeEnv::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_max_steps(20)
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test session data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        let subject_name_lhs = "http_client_request_pdf_s";
        let id = "2508.18700";
        let get_url = format!("pdf/{id}");
        let role = vec!["user".to_string()];
        let content = vec![get_url];
        let timestamp = vec![0_i64];
        let batch = create_chat_record_batch(role, content, timestamp)?;
        let messages = AvailableInterfaceSubjects::UserMessages
            .to_subject_builder(Some(subject_name_lhs))
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            messages.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(messages.get_name())
                .with_subject(messages.get_name())
                .with_update(&Publication::Replace {
                    subject_name: messages.get_name().to_string(),
                })
                .with_message(messages.to_ipc_stream()?)
                .with_publisher(network_arc.get_name())
                .make_name()?
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the session (embed documents)
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        // Make the test session data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Make the query data
        let query_ids = vec!["0".to_string()];
        let text =vec!["What is the problem with ID-based embeddings?".to_string()];
        let batch = create_queries_batch(query_ids, text)?;
        let queries = AvailableInterfaceSubjects::UserQueries
            .to_subject_builder(None)
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message_map.insert(
            queries.get_name().to_string(),
            IPCMessage::get_builder()
                .with_message(queries.to_ipc_stream()?)
                .with_subject(queries.get_name())
                .with_update(&Publication::Extend {
                    subject_name: queries.get_name().to_string(),
                })
                .with_publisher(network_arc.get_name())
                .make_name()?
                .build()?,
        );

        // 2. Run the session (embed queries and retrieve)
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        let extended_diagnostic_subjects = extended_diagnostic_subjects();
        let subject_names = extended_diagnostic_subjects
            .iter()
            .map(|s| s.as_str())
            .chain(["EmbeddingScores", "Documents", "UserQueries"])
            .collect::<Vec<_>>();
        write_diagnostic_subjects_to_csv(
            &subject_names,
            network_arc.runtime_env(),
            network_arc.get_name(),
        )
        .await?;

        assert_eq!(response.len(), 0);

        // Test PDF extraction, embedding, and retrieval
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::EmbeddingScores.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableSubjects::EmbeddingScores.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        dbg!(&subject.count_rows());
        // assert_eq!(subject.count_rows(), );
        let column = subject.get_column_as_vec_str("chunk_id");
        // dbg!(column.first().unwrap());
        assert_eq!(column.first().unwrap(), &"https://arxiv.org/pdf/2508.187002PdfText { op: 158, bt: 54, tm: PdfTm { a: 0.0, b: 0.0, c: 0.0, d: 0.0, x: 53.798, y: 281.913 }, td: PdfTd { x: 0, y: 0 }, font: PdfFont { font_name: \"F194\", font_subtype: \"CVYSYC+LinLibertineTB\", base_font: \"Type1\" }, font_size: 8, text: \"\" }_0");
        let column = subject.get_column_as_vec_str("query_id");
        // dbg!(column.first().unwrap());
        assert_eq!(column.first().unwrap(), &"0");
        let column = subject.get_column_as_vec_primitive::<f32>("score")?;
        for t in column {
            assert!(t > 0.15); // Threshold used for filtering
        }

        Ok(())
    }
}
