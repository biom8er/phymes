use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::HTTPClientRequestStream;
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};

use crate::ProcessorTrait;

#[derive(Debug)]
pub struct HTTPClientRequestProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for HTTPClientRequestProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for HTTPClientRequestProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn line_and_file(&self) -> (u32, String) {
        (line!(), file!().to_string())
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the stream
        let out = Box::pin(HTTPClientRequestStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.cloned(),
        )?);

        // Prepare the message builder
        let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let builder = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.get_name())
            .with_message(out);
        let _ = builder_map.insert(self.get_name().to_string(), builder);

        Ok(builder_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use phymes_data::{DocumentExtractType, DocumentFilterType, extract_pdf};
    use phymes_diagnostics::{
        DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, HashMap, SpanBuilder,
    };
    use phymes_event::Publication;
    use phymes_schemas::{create_chat_record_batch, open_alex, semantic_scholar};
    use phymes_streams::{HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType};
    use phymes_subject::{SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
    use serde_json::{Map, Value};

    #[tokio::test]
    async fn test_http_client_processor_open_alex_get_message_from_message() -> Result<()> {
        // Case 1: GET from messages
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // OpenAlex request filters
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "publication_year".to_string(),
            Value::String("2020".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Works,
            ..Default::default()
        };

        // Config for the HTTP Processor
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch = create_chat_record_batch(
            vec!["user".to_string()],
            vec![open_alex_request.to_get_query()?],
            vec![0],
        )?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        assert!(snippet.contains("https://openalex.org/W3038568908"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_open_alex_get_message_from_config() -> Result<()> {
        // Case 1: GET from messages
        let name = "HTTPClientRequestProcessor";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // OpenAlex request filters
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "publication_year".to_string(),
            Value::String("2020".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Works,
            ..Default::default()
        };

        // Config for the HTTP Processor
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        assert!(snippet.contains("https://openalex.org/W3038568908"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_open_alex_get_blob_from_message() -> Result<()> {
        // Case 1: GET from messages
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // OpenAlex request filters
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "publication_year".to_string(),
            Value::String("2020".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Works,
            ..Default::default()
        };

        // Config for the HTTP Processor
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch = create_chat_record_batch(
            vec!["user".to_string()],
            vec![open_alex_request.to_get_query()?],
            vec![0],
        )?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("filename");
        assert_eq!(
            result,
            ["https://api.openalex.org//works?page=1&per-page=1&filter=publication_year:\"2020\""]
        );
        let result = table.get_column_as_vec_str("extension");
        assert_eq!(result, ["application/json"]);
        let result = table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let snippet = String::from_utf8(result)?;
        assert!(snippet.contains("https://openalex.org/W3038568908"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_e_utils_e_search() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build ESearch query
        let mesh_term = "Diabetes Mellitus";
        let year_from = 2020;
        let year_to = 2023;
        let journal_filter = Some("Lancet");
        let mut query = format!("{mesh_term}[MeSH Terms]");
        if let Some(journal) = journal_filter {
            query.push_str(&format!(" AND \"{journal}\"[Journal]"));
        }

        let esearch_url = format!(
            "db=pubmed&term={}&retmode=json&retmax=5&mindate={}&maxdate={}",
            urlencoding::encode(&query),
            year_from,
            year_to
        );

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?".to_string(),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch =
            create_chat_record_batch(vec!["user".to_string()], vec![esearch_url], vec![0])?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content").join("");
        assert!(result.contains("\"idlist\":["));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_e_utils_e_fetch() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build EFetch query
        let ids = ["37997144", "37997132", "37997130", "37997120", "37997092"].join(",");
        let efetch_url = format!("db=pubmed&id={ids}&retmode=xml");

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?".to_string(),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch =
            create_chat_record_batch(vec!["user".to_string()], vec![efetch_url], vec![0])?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("filename");
        assert_eq!(
            result,
            [
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=37997144,37997132,37997130,37997120,37997092&retmode=xml"
            ]
        );
        let result = table.get_column_as_vec_str("extension");
        assert_eq!(result, ["text/xml; charset=UTF-8"]);
        let result = table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let snippet = String::from_utf8(result)?;
        assert!(snippet.contains("MedlineCitation"));
        assert!(snippet.contains("!DOCTYPE PubmedArticleSet PUBLIC"));
        assert!(snippet.contains("https://dtd.nlm.nih.gov/ncbi/pubmed/"));

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_pdf_download() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build pathname for download
        let id = "2508.18700";
        let download_url = format!("pdf/{id}");

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://arxiv.org/".to_string(),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch =
            create_chat_record_batch(vec!["user".to_string()], vec![download_url], vec![0])?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;
        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(result, ["tool"]);
        let filenames = table.get_column_as_vec_str("filename");
        assert_eq!(filenames, ["https://arxiv.org/pdf/2508.18700"]);
        let result = table.get_column_as_vec_str("extension");
        assert_eq!(result, ["application/pdf"]);

        // Check the PDF
        let pdf_batch = extract_pdf(
            "filename",
            "bytes",
            table.get_record_batches(),
            &DocumentFilterType::Default,
            &DocumentExtractType::Default,
        )?;
        let table = SubjectBuilder::new()
            .with_record_batches(vec![pdf_batch])?
            .with_name("")
            .build()?;
        let result = table.get_column_as_vec_str("chunk_id");
        assert_eq!(
            result,
            [
                "https://arxiv.org/pdf/2508.18700_1",
                "https://arxiv.org/pdf/2508.18700_2",
                "https://arxiv.org/pdf/2508.18700_3"
            ]
        );
        let result = table.get_column_as_vec_str("document_id");
        assert_eq!(
            result,
            [
                "https://arxiv.org/pdf/2508.18700",
                "https://arxiv.org/pdf/2508.18700",
                "https://arxiv.org/pdf/2508.18700"
            ]
        );
        let result = table.get_column_as_vec_str("text");
        let snippet = result.first().unwrap().to_string();
        assert_eq!(
            snippet[..100],
            *"Taming the One-Epoch Phenomenon in Online Recommendation System by Two-stage Contrastive ID Pre-trai"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_semantic_scholar() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 30,
            request_type: HTTPClientRequestType::Post,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            content_type: Some("application/json".to_string()),
            base_url: "https://api.semanticscholar.org/recommendations/v1/papers/?".to_string(),
            json: Some("fields=title,url,authors&limit=3".to_string()),
            subject_name: Some(messages.to_string()),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the request body
        let req_body = semantic_scholar::RecommendationsRequest {
            positive_papers: Some(vec!["649def34f8be52c8b66281af98ae884c09aef38b".to_string()]),
            negative_papers: Some(vec!["ArXiv:1805.02262".to_string()]),
        };
        let req_body_json = serde_json::to_vec(&req_body)?;
        let req_body_table = SubjectBuilder::new()
            .with_name(messages)
            .with_json(&req_body_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(req_body_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        let snippet = result.first().unwrap().to_string();
        assert!(snippet.contains("{\"paperId\":"));

        Ok(())
    }

    #[tokio::test]
    #[ignore = "for generating data to test OpenAlex parsers"]
    async fn test_http_client_processor_open_alex_test_data() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Author
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("has_orcid".to_string(), Value::String("true".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Authors,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Author: {snippet}");

        // Institution
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("country_code".to_string(), Value::String("us".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Institutions,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Institution: {snippet}");

        // Topic
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "display_name.search".to_string(),
            Value::String("artificial+intelligence".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Topics,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Topic: {snippet}");

        // Award
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "funder.id".to_string(),
            Value::String("F4320306076".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Awards,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Award: {snippet}");

        // Funder
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("country_code".to_string(), Value::String("us".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Funders,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Funder: {snippet}");

        // Publisher
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert(
            "display_name.search".to_string(),
            Value::String("elsevier".to_string()),
        );
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Publishers,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Publisher: {snippet}");

        // Source
        let mut filter = Map::<String, Value>::new();
        let _ = filter.insert("has_issn".to_string(), Value::String("true".to_string()));
        let open_alex_request = open_alex::OpenAlexRequest {
            page: Some(1),
            per_page: Some(1),
            filter: Some(filter),
            entity: open_alex::OpenAlexRequestEntity::Sources,
            ..Default::default()
        };
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: format!("{}?", open_alex_request.to_base_url()),
            json: Some(open_alex_request.to_get_query()?),
            request_schema: HTTPClientRequestSchemas::Messages,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor =
            HTTPClientRequestProcessor::new(name, HTTPClientRequestProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_http_client_processor_open_alex_get_config")
            .with_record_batches(result)?
            .build()?;
        let result = table.get_column_as_vec_string("content")?;
        let snippet = result.first().unwrap().to_string();
        println!("Source: {snippet}");

        Ok(())
    }
}
