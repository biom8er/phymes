use std::{
    fmt::Write,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready}, time::Duration,
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, PublishAndSubscribeTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, Table, TableBuilderTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription, TableTrait, create_chat_record_batch, remove_message_by_subject
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
    create_timestamp_micros,
};
use reqwest::{Client, Response, header::{CONTENT_TYPE, USER_AGENT}};
use serde_json::{Map, Value};
use tracing::{Level, event};

use crate::{DataConfigTrait, external_operators::http_client_config::{HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType, e_utils_schemas, open_alex_schemas, semantic_scholar_schemas}};

/// The state of the HTTP Client API request
///
/// # Notes
/// * We need to capture each stage of the request so that the connection 
///   is not dropped during repeated polling of the stream.
pub enum HTTPClientRequestState {
    NotStarted,
    Connecting(Pin<Box<dyn Future<Output = Result<Response, reqwest::Error>> + Send + 'static>>),
    ToText(Pin<Box<dyn Future<Output = Result<String, reqwest::Error>> + Send + 'static>>),
    Done,
}

/// Error reporting method for Reqwest error
pub(crate) fn error_report(mut err: &(dyn std::error::Error + 'static)) -> String {
    let mut s = format!("{}", err);
    while let Some(src) = err.source() {
        let _ = write!(s, "\n\nCaused by: {}", src);
        err = src;
    }
    s
}

#[derive(Debug)]
pub struct HTTPClientRequestProcessor {
    name: String,
    r#type: String,
    publications: Vec<TablePublication>,
    subscriptions: Vec<TableSubscription>,
    subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
}

impl MappableTrait for HTTPClientRequestProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PublishAndSubscribeTrait for HTTPClientRequestProcessor {
    fn get_publications(&self) -> Vec<&TablePublication> {
        self.publications.iter().collect::<Vec<_>>()
    }

    fn get_subscriptions(&self) -> Vec<&TableSubscription> {
        self.subscriptions.iter().collect::<Vec<_>>()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe_policy
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ProcessorTrait for HTTPClientRequestProcessor {
    fn new(
        name: &str,
        r#type: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe_policy,
        }
    }

    fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait {
        self.subscribe_policy.as_ref()
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        // Trace the inbox
        let trace = if let Some(diagnostic_builder) = diagnostic_builder {
            let trace_builder = diagnostic_builder.clone().to_child(self.get_name())?;
            let trace = trace_builder
                .clone()
                .messages(line!(), file!(), self.get_name());
            trace.enter(&message.values().collect::<Vec<_>>());
            Some((trace, trace_builder))
        } else {
            None
        };

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Extract out the subscribed messages
        let mut subscriptions = Vec::new();
        for subs in self.subscriptions.iter() {
            if subs.get_table_name() != self.get_name() {
                match remove_message_by_subject(subs.get_table_name(), &mut message) {
                    Some(m) => {
                        subscriptions.push(m);
                    }
                    None => {
                        event!(
                            Level::WARN,
                            "Subscription {} not provided for {}.",
                            subs.get_table_name(),
                            self.get_name()
                        );
                    }
                }
            }
        }
        if subscriptions.len() > 1 {
            return Err(anyhow!("More than one subscription was found."));
        } else if subscriptions.is_empty() {
            return Err(anyhow!("No subscriptions were found."));
        }

        // Run the stream
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(HTTPClientRequestStream::new(
            subscriptions.swap_remove(0).get_message_own(),
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
            .make_name()?
            .build()?;
        let _ = message.insert(out_m.get_name().to_string(), out_m);

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&message.values().collect::<Vec<_>>());
        }
        Ok(message)
    }
}

pub struct HTTPClientRequestStream {
    /// Output schema
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    _runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<HTTPClientConfig>,
    /// State of the OpenAI API request
    state: HTTPClientRequestState,
}

impl HTTPClientRequestStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            message_stream,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            state: HTTPClientRequestState::NotStarted,
        })
    }

    /// Initialize the config for text generation inference
    fn init_config(&mut self, config_table: Table) -> Result<()> {
        if self.config.is_none() {
            let config = HTTPClientConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }
}

impl Stream for HTTPClientRequestStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Iterate through each state until the API request is completed
        match &mut self.state {
            HTTPClientRequestState::NotStarted => {
                // Initialize the config
                if self.config.is_none() {
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    let config_table = Table::get_builder()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    self.init_config(config_table)?;
                }

                // Collect the message data in a streaming fashion
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                    if batch.num_rows() > 0 {
                        batches.push(batch);
                        break;
                    }                    
                }

                // The poll ends when there are no more batches
                if batches.is_empty() {
                    self.state = HTTPClientRequestState::Done;
                    return Poll::Ready(None)
                }
                let messages = Table::get_builder()
                    .with_name("messages")
                    .with_record_batches(batches)?
                    .build()?;

                // Create HTTP client with timeout
                let client = Client::builder()
                    .timeout(Duration::from_secs(self.config.as_ref().unwrap().timeout.try_into()?))
                    .build()?;

                // Make the request
                // DM: A future optimization maybe to treat each row as a parallel API request
                let fut = match self.config.as_ref().unwrap().request_type {
                    HTTPClientRequestType::Get => {
                        // Join the `content` fields together for the case of multiple rows
                        let query_str = messages.get_column_as_vec_str("content").join("");
                        
                        // Prioritize the message data over the config when building the url
                        let query_url = if query_str.is_empty() {
                            None
                        } else {
                            Some(query_str)
                        };
                        let url = self.config.as_ref().unwrap().url(query_url.as_deref());

                        // Make the request
                        let mut client = client.get(url);
                        if let Ok(token) = self.config.as_ref().unwrap().api_key() {
                            client = client.bearer_auth(token)
                        }
                        client.header(USER_AGENT, self.config.as_ref().unwrap().content_type.clone().ok_or(anyhow!("Content type (header value) needs to be specified for GET requests."))?)
                            .send()
                    },
                    HTTPClientRequestType::Post => {
                        // Extract the table as a JSON object
                        // DM: currently, only the last row is used similar to configs...
                        let mut json_object = messages.to_json_object()?;

                        // Prioritize the message data over the config when building the JSON body and url
                        let (json_data, url) = if json_object.len() > 0 {
                            let json_data = json_object.pop().unwrap();
                            let url = self.config.as_ref().unwrap().url(None);
                            (json_data, url)
                        } else if let Some(json_str) = self.config.as_ref().unwrap().json.as_ref() {
                            let json_data = serde_json::from_str::<Map<String, Value>>(json_str)?;
                            let url = self.config.as_ref().unwrap().base_url.clone();
                            (json_data, url)
                        } else {
                            self.state = HTTPClientRequestState::Done;
                            return Poll::Ready(Some(Err(anyhow!("POST json data was not found in the messages nor in the config."))))
                        };
                        dbg!(&json_data);
                        
                        // Make the request
                        let mut client = client.post(url);                        
                        if let Ok(token) = self.config.as_ref().unwrap().api_key() {
                            client = client.bearer_auth(token)
                        }
                        client.header(CONTENT_TYPE, self.config.as_ref().unwrap().content_type.clone().ok_or(anyhow!("Content type needs to be specified for POST requests."))?)
                            .json(&json_data)
                            .send()
                    },
                    _ => {
                        self.state = HTTPClientRequestState::Done;
                        return Poll::Ready(Some(Err(anyhow!("Request type {} is not supported yet.", self.config.as_ref().unwrap().request_type))))
                    }
                };

                // Update the request state and poll next
                self.state = HTTPClientRequestState::Connecting(Box::pin(fut));
                self.poll_next(cx)
            }
            HTTPClientRequestState::Connecting(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(response) => {
                    let fut = response.text();
                    self.state = HTTPClientRequestState::ToText(Box::pin(fut));
                    self.poll_next(cx)
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
                }
            },
            HTTPClientRequestState::ToText(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(text) => {
                    // Initialize the metrics
                    let baseline_metrics =
                        if let Some(diagnostic_builder) = &self.diagnostic_builder {
                            Some(
                                diagnostic_builder
                                    .clone()
                                    .to_child("HTTPClientRequestStream")?
                                    .baseline_metrics(line!(), file!(), "poll_next"),
                            )
                        } else {
                            None
                        };
                    let _timer = baseline_metrics
                        .as_ref()
                        .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                    // Parse the response
                    let batch = match self.config.as_ref().unwrap().request_schema {
                        HTTPClientRequestSchemas::None => create_chat_record_batch(
                                vec!["tool".to_string()],
                                vec![text],
                                vec![create_timestamp_micros()],
                            )?,
                        HTTPClientRequestSchemas::OpenAlex => {
                            let parsed = match serde_json::from_str::<open_alex_schemas::OpenAlexResponse>(&text) {
                                Ok(parsed) => parsed,
                                Err(err) => {
                                    self.state = HTTPClientRequestState::Done;
                                    return Poll::Ready(Some(Err(anyhow!("A parsing error {err:?} was encountered when parsing response {text} for schema {}.", self.config.as_ref().unwrap().request_schema))))
                                }
                            };
                            let content = parsed.results.into_iter().map(|w| serde_json::to_string(&w).unwrap()).collect::<Vec<_>>();
                            let roles = content.iter().map(|_| "tool".to_string()).collect::<Vec<_>>();
                            let timestamps = content.iter().map(|_| create_timestamp_micros()).collect::<Vec<_>>();
                            create_chat_record_batch(
                                roles,
                                content,
                                timestamps,
                            )?
                        }
                        HTTPClientRequestSchemas::ESearch => {
                            let parsed = match serde_json::from_str::<e_utils_schemas::ESearchResponse>(&text) {
                                Ok(parsed) => parsed,
                                Err(err) => {
                                    self.state = HTTPClientRequestState::Done;
                                    return Poll::Ready(Some(Err(anyhow!("A parsing error {err:?} was encountered when parsing response {text} for schema {}.", self.config.as_ref().unwrap().request_schema))))
                                }
                            };
                            let content = parsed.esearchresult.idlist;
                            let roles = content.iter().map(|_| "tool".to_string()).collect::<Vec<_>>();
                            let timestamps = content.iter().map(|_| create_timestamp_micros()).collect::<Vec<_>>();
                            create_chat_record_batch(
                                roles,
                                content,
                                timestamps,
                            )?
                        }
                        HTTPClientRequestSchemas::EFetch => {
                            let cleaned_text = text.replace("<sup>", "")
                                .replace("</sup>", "")
                                .replace("<sub>", "")
                                .replace("</sub>", "");
                            let parsed = match serde_json::from_str::<e_utils_schemas::PubmedArticleSet>(&cleaned_text) {
                                Ok(parsed) => parsed,
                                Err(err) => {
                                    self.state = HTTPClientRequestState::Done;
                                    return Poll::Ready(Some(Err(anyhow!("A parsing error {err:?} was encountered when parsing response {text} for schema {}.", self.config.as_ref().unwrap().request_schema))))
                                }
                            };
                            let content = parsed.articles.into_iter().map(|w| serde_json::to_string(&w).unwrap()).collect::<Vec<_>>();
                            let roles = content.iter().map(|_| "tool".to_string()).collect::<Vec<_>>();
                            let timestamps = content.iter().map(|_| create_timestamp_micros()).collect::<Vec<_>>();
                            create_chat_record_batch(
                                roles,
                                content,
                                timestamps,
                            )?
                        }
                        HTTPClientRequestSchemas::SemanticScholarRecomendations => {
                            let parsed = match serde_json::from_str::<semantic_scholar_schemas::RecommendationsResponse>(&text) {
                                Ok(parsed) => parsed,
                                Err(err) => {
                                    self.state = HTTPClientRequestState::Done;
                                    return Poll::Ready(Some(Err(anyhow!("A parsing error {err:?} was encountered when parsing response {text} for schema {}.", self.config.as_ref().unwrap().request_schema))))
                                }
                            };
                            let content = parsed.papers.into_iter().map(|w| serde_json::to_string(&w).unwrap()).collect::<Vec<_>>();
                            let roles = content.iter().map(|_| "tool".to_string()).collect::<Vec<_>>();
                            let timestamps = content.iter().map(|_| create_timestamp_micros()).collect::<Vec<_>>();
                            create_chat_record_batch(
                                roles,
                                content,
                                timestamps,
                            )?
                        }
                        _ => {
                            self.state = HTTPClientRequestState::Done;
                            return Poll::Ready(Some(Err(anyhow!("Request schema {} is not supported yet.", self.config.as_ref().unwrap().request_schema))))
                        }
                        
                    };

                    // Reset the state to poll the next batch
                    self.state = HTTPClientRequestState::NotStarted;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    if let Some(baseline_metrics) = &baseline_metrics {
                        baseline_metrics.record_poll(poll)
                    } else {
                        poll
                    }
                }
                Err(err) => {
                    self.state = HTTPClientRequestState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
                }
            },
            HTTPClientRequestState::Done => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for HTTPClientRequestStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use crate::external_operators::http_client_config::semantic_scholar_schemas;

    use super::*;
    use futures::TryStreamExt;
    use phymes_core::{AvailableTableSubscribePolicies, ChatBuilderTraitExt, RuntimeEnvTrait, TableBuilder};
    use phymes_diagnostics::{DiagnosticBuilder, Diagnostics, HashMap, SpanBuilder};

    #[tokio::test]
    async fn test_http_client_processor_open_alex() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt")));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the http client processor config
        let year = "2020";
        let per_page = 5; // 50 Max allowed by OpenAlex
        let page = 1;
        let query_url = format!("filter=publication_year:{year}&per-page={per_page}&page={page}");
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            content_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://api.openalex.org/works".to_string(),
            // json: Some(query_url),
            request_schema: HTTPClientRequestSchemas::OpenAlex,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(
                &query_url,
                "user",
            )?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor = HTTPClientRequestProcessor::new(
            name,
            HTTPClientRequestProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: http_client_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response        
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool", "tool", "tool", "tool", "tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(*result.first().unwrap(), "{\"id\":\"https://openalex.org/W3038568908\",\"display_name\":\"Radiation Resistant Camera System for Monitoring Deuterium Plasma Discharges in the Large Helical Device\",\"publication_year\":2020,\"publication_date\":\"2020-06-08\",\"doi\":\"https://doi.org/10.1585/pfr.15.2402039\",\"language\":\"en\",\"type_\":null,\"cited_by_count\":801215,\"authorships\":[{\"author\":{\"id\":\"https://openalex.org/A5039600762\",\"display_name\":\"M. Shoji\"},\"institutions\":[{\"id\":\"https://openalex.org/I4210108322\",\"display_name\":\"National Institute for Fusion Science\"},{\"id\":\"https://openalex.org/I199525922\",\"display_name\":\"National Institutes of Natural Sciences\"}]}],\"concepts\":[{\"id\":\"https://openalex.org/C153385146\",\"display_name\":\"Radiation\",\"level\":2,\"score\":0.7057818174362183},{\"id\":\"https://openalex.org/C82706917\",\"display_name\":\"Plasma\",\"level\":2,\"score\":0.5598242878913879},{\"id\":\"https://openalex.org/C192562407\",\"display_name\":\"Materials science\",\"level\":0,\"score\":0.5517664551734924},{\"id\":\"https://openalex.org/C120665830\",\"display_name\":\"Optics\",\"level\":1,\"score\":0.5239154100418091},{\"id\":\"https://openalex.org/C138081364\",\"display_name\":\"Shield\",\"level\":2,\"score\":0.5098416209220886},{\"id\":\"https://openalex.org/C152568617\",\"display_name\":\"Neutron\",\"level\":2,\"score\":0.4559711515903473},{\"id\":\"https://openalex.org/C116915560\",\"display_name\":\"Nuclear engineering\",\"level\":1,\"score\":0.3836207985877991},{\"id\":\"https://openalex.org/C121332964\",\"display_name\":\"Physics\",\"level\":0,\"score\":0.32291728258132935},{\"id\":\"https://openalex.org/C185544564\",\"display_name\":\"Nuclear physics\",\"level\":1,\"score\":0.13794386386871338},{\"id\":\"https://openalex.org/C127313418\",\"display_name\":\"Geology\",\"level\":0,\"score\":0.05549171566963196},{\"id\":\"https://openalex.org/C5900021\",\"display_name\":\"Petrology\",\"level\":1,\"score\":0.0},{\"id\":\"https://openalex.org/C127413603\",\"display_name\":\"Engineering\",\"level\":0,\"score\":0.0}],\"mesh\":[],\"host_venue\":null,\"open_access\":{\"is_oa\":true,\"oa_status\":\"diamond\",\"oa_url\":\"https://www.jstage.jst.go.jp/article/pfr/15/0/15_2402039/_pdf\"},\"abstract_inverted_index\":{\"(LHD).\":[16],\"(neutrons\":[34],\"10%\":[79],\"2017,\":[54],\"CCD\":[146],\"Device\":[15],\"FY\":[53],\"For\":[86],\"Helical\":[14],\"Investigation\":[137],\"Large\":[13],\"MCNP-6\":[101],\"Radiation\":[0],\"Thanks\":[122],\"The\":[37,64],\"This\":[17,181],\"all\":[84],\"also\":[166,183],\"and\":[35,111],\"appeared\":[60],\"been\":[67,135],\"blocks\":[82],\"borated\":[80],\"box,\":[93],\"box.\":[121],\"boxes\":[71,76],\"bright\":[57,154,170],\"by\":[100,173],\"calculated\":[99],\"camera\":[2],\"camera,\":[164],\"cameras\":[38,65,133],\"cameras.\":[197],\"campaigns\":[27],\"change\":[113],\"code,\":[102],\"consist\":[73],\"constructed\":[5],\"contributed\":[20],\"contributes\":[185],\"covered\":[77],\"design\":[89],\"deuterium\":[8],\"directions.\":[85],\"disappear\":[172],\"discharge\":[45],\"discharges\":[10],\"distribution\":[97],\"due\":[31],\"during\":[24],\"emission\":[50],\"energy\":[116],\"even\":[41],\"experimental\":[26],\"extension\":[127,189],\"flux\":[96,110,161],\"for\":[6],\"functioned\":[40],\"further\":[188],\"gamma-rays).\":[36],\"generally\":[156],\"has\":[19,134],\"have\":[66],\"highly\":[184],\"image\":[147,179],\"images.\":[63],\"in\":[11,42,52,69,83,118],\"increases\":[157],\"indicates\":[167],\"influence\":[140],\"installed\":[68],\"lead\":[75],\"lifetime\":[130,192],\"maximum\":[48],\"monitoring\":[7],\"neutron\":[49],\"number\":[152],\"of\":[74,90,107,114,128,131,138,141,153,190,193],\"on\":[61,144,177],\"operation\":[23],\"optimization,\":[125],\"optimizing\":[87],\"phenomenon\":[182],\"plasma\":[9,44],\"polyethylene\":[81],\"problems\":[30],\"process\":[176],\"radiation\":[33,95,109,143,160,195],\"rate\":[51],\"realized.\":[136],\"reduction\":[106],\"resistant\":[1,196],\"reveals\":[104],\"safe\":[22],\"self-annealing\":[175],\"sensor\":[148],\"sensor.\":[180],\"serious\":[29],\"shield\":[70,92,120],\"shows\":[149],\"significant\":[126],\"some\":[56,169],\"specks\":[58,155,171],\"spectra\":[117],\"steadily\":[39],\"system\":[3,18],\"temporarily\":[59],\"that\":[150,168],\"the\":[12,43,47,62,88,91,94,105,108,112,115,119,124,129,132,139,142,145,151,159,163,174,178,187,191,194],\"though\":[55],\"to\":[21,32,123,162,186],\"two\":[25],\"was\":[4,98],\"which\":[72,103,165],\"with\":[46,78,158],\"without\":[28]}}");

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_e_utils() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt")));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Build ESearch query
        let mesh_term = "Diabetes Mellitus";
        let year_from = 2020;
        let year_to = 2023;
        let journal_filter = Some("Lancet");
        let mut query = format!("{}[MeSH Terms]", mesh_term);
        if let Some(journal) = journal_filter {
            query.push_str(&format!(" AND \"{}\"[Journal]", journal));
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
            content_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi".to_string(),
            request_schema: HTTPClientRequestSchemas::ESearch,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(
                &esearch_url,
                "user",
            )?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor = HTTPClientRequestProcessor::new(
            name,
            HTTPClientRequestProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: http_client_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response        
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool","tool","tool","tool","tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["37997144","37997132","37997130","37997120","37997092"]);        

        // Build EFetch query
        let ids = table.get_column_as_vec_str("content").join(",");
        let efetch_url = format!(
            "db=pubmed&id={}&retmode=xml",
            ids
        );

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            content_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi".to_string(),
            request_schema: HTTPClientRequestSchemas::EFetch,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str(
                &efetch_url,
                "user",
            )?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor = HTTPClientRequestProcessor::new(
            name,
            HTTPClientRequestProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: http_client_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response        
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool","tool","tool","tool","tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(*result.first().unwrap(), "{\"MedlineCitation\":{\"Article\":{\"ArticleTitle\":\"Effectiveness of interventions for improving physical activity level in working-age people (aged 18-60 years) with type 2 diabetes: a systematic review and meta-analysis.\",\"Abstract\":{\"AbstractText\":[\"The increasing prevalence of type 2 diabetes in working-age people imposes a substantial societal burden. Although physical activity is crucial for diabetes management, limited evidence exists to inform optimal strategies for promoting physical activity in this population. We aimed to determine and compare the effectiveness of interventions for increasing physical activity level in working-age people with diabetes.\",\"In this systematic review and meta-analysis, we searched Web of Science, the Cochrane Library, Medline, Embase, PsycINFO, ClinicalTrials.gov, and ICTRP for papers published between Jan 1, 1931, and June 30, 2022, in English. Search terms included \\\"physical activity\\\", \\\"diabetes\\\", and \\\"randomised controlled trial\\\". We included trials reporting the effects of interventions on physical activity level (objectively or subjectively measured) in people with type 2 diabetes aged 18-60 years. Two independent reviewers conducted summary data extraction and quality assessment. We used pairwise random-effects, frequentist network meta-analyses, and meta-regression to obtain pooled effects. Heterogeneity was evaluated using I2 statistic. The risk of bias and certainty of evidence were assessed using the Cochrane risk-of-bias 2 tool and the Grading of Recommendations Assessment, Development, and Evaluation. This study is registered with PROSPERO (CRD42022323165).\",\"We identified 52 trials (6257 participants) from 21 countries (32 Asia, ten North America, eight Europe, one Australia, one Africa). The overall risk of bias was classified as \\\"some concerns\\\" for included studies. Four types of interventions (structured exercise training, physical activity education, psychological intervention, physical activity education plus psychological intervention) were identified. Compared with control groups, the interventions showed significant effects in objectively measured (standardised mean difference 0·77, 95% CI 0·27-1·27, low certainty), subjectively measured (0·88, 0·40-1·35, very low certainty), and overall physical activity (0·82, 0·48-1·16, moderate certainty). Physical activity education exerted large effect in overall physical activity compared with control groups. Psychological intervention exerted large effects in overall physical activity compared with other interventions. Heterogeneity was high (I2=96-97%). Intervention setting (p=0·04) and facilitator (p=0·03) showed effects on heterogeneity.\",\"Psychologically modelled education might be the most beneficial way of promoting physical activity. Intervention setting and facilitator type should be considered when designing interventions for improving physical activity level in working-age people with type 2 diabetes. Limitations of this review include restriction to the English language and considerable heterogeneity between studies.\",\"King's-China Scholarship Council PhD Scholarship (202108440151).\"]},\"Journal\":{\"Title\":\"Lancet (London, England)\",\"ISSN\":\"1474-547X\",\"JournalIssue\":{\"PubDate\":{\"Year\":\"2023\",\"Month\":\"Nov\",\"Day\":null},\"Volume\":\"402 Suppl 1\",\"Issue\":null}},\"AuthorList\":{\"Author\":[{\"LastName\":\"Zhao\",\"ForeName\":\"Xiaoyan\",\"AffiliationInfo\":[{\"Affiliation\":\"Florence Nightingale Faculty of Nursing & Midwifery, King's College London, London, UK. Electronic address: xiaoyan.zhao@kcl.ac.uk.\"}]},{\"LastName\":\"Duaso\",\"ForeName\":\"Maria\",\"AffiliationInfo\":[{\"Affiliation\":\"Florence Nightingale Faculty of Nursing & Midwifery, King's College London, London, UK.\"}]},{\"LastName\":\"Ghazaleh\",\"ForeName\":\"Haya Abu\",\"AffiliationInfo\":[{\"Affiliation\":\"Florence Nightingale Faculty of Nursing & Midwifery, King's College London, London, UK.\"}]},{\"LastName\":\"Cheng\",\"ForeName\":\"Li\",\"AffiliationInfo\":[{\"Affiliation\":\"School of Nursing, Sun Yat-sen University, Guangzhou, China.\"}]},{\"LastName\":\"Forbes\",\"ForeName\":\"Angus\",\"AffiliationInfo\":[{\"Affiliation\":\"Florence Nightingale Faculty of Nursing & Midwifery, King's College London, London, UK.\"}]}]},\"Pagination\":{\"MedlinePgn\":\"S97\"},\"ELocationID\":[{\"$value\":\"10.1016/S0140-6736(23)02145-1\",\"EIdType\":null},{\"$value\":\"S0140-6736(23)02145-1\",\"EIdType\":null}]},\"MeshHeadingList\":{\"MeshHeading\":[{\"DescriptorName\":\"Humans\"},{\"DescriptorName\":\"Diabetes Mellitus, Type 2\"},{\"DescriptorName\":\"Exercise\"},{\"DescriptorName\":\"Africa\"},{\"DescriptorName\":\"Asia\"},{\"DescriptorName\":\"Australia\"}]}},\"PubmedData\":{\"ArticleIdList\":{\"ArticleId\":[{\"$value\":\"37997144\",\"IdType\":null},{\"$value\":\"10.1016/S0140-6736(23)02145-1\",\"IdType\":null},{\"$value\":\"S0140-6736(23)02145-1\",\"IdType\":null}]}}}");

        Ok(())
    }

    #[tokio::test]
    async fn test_http_client_processor_semantic_scholar() -> Result<()> {
        let name = "HTTPClientRequestProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt")));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the http client processor config
        let http_client_config = HTTPClientConfig {
            timeout: 30,
            request_type: HTTPClientRequestType::Post,
            content_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://api.semanticscholar.org/recommendations/v1/papers/".to_string(),
            json: Some("fields=title,url,authors&limit=3".to_string()),
            request_schema: HTTPClientRequestSchemas::SemanticScholarRecomendations,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&http_client_config_json, 1)?
            .build()?;

        // Make the request body
        let req_body = semantic_scholar_schemas::RecommendationsRequest {
            positive_papers: Some(vec![
                "649def34f8be52c8b66281af98ae884c09aef38b".to_string(),
            ]),
            negative_papers: Some(vec![
                "ArXiv:1805.02262".to_string(),
            ])
        };
        let req_body_json = serde_json::to_vec(&req_body)?;
        let req_body_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(req_body_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            http_client_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher("")
                .with_subject(http_client_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(http_client_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the http client processor
        let processor = HTTPClientRequestProcessor::new(
            name,
            HTTPClientRequestProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: http_client_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response        
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool", "tool", "tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(*result.first().unwrap(), "{\"paperId\":\"65947afa8f1a6673ae9d4932b1262dca776c097c\",\"title\":\"A systematic review of relation extraction task since the emergence of Transformers\",\"abstract\":null,\"year\":null,\"venue\":null,\"publicationTypes\":null,\"publicationDate\":null,\"doi\":null,\"arxivId\":null,\"url\":\"https://www.semanticscholar.org/paper/65947afa8f1a6673ae9d4932b1262dca776c097c\",\"isOpenAccess\":null,\"openAccessPdf\":null,\"citationCount\":null,\"influentialCitationCount\":null,\"isHighlyCited\":null,\"referenceCount\":null,\"fieldsOfStudy\":null,\"authors\":[{\"authorId\":\"2284776050\",\"name\":\"Célian Ringwald\",\"aliases\":null,\"affiliations\":null,\"homepage\":null,\"paperCount\":null,\"citationCount\":null,\"hIndex\":null,\"url\":null},{\"authorId\":\"2287849105\",\"name\":\"Fabien L. Gandon\",\"aliases\":null,\"affiliations\":null,\"homepage\":null,\"paperCount\":null,\"citationCount\":null,\"hIndex\":null,\"url\":null},{\"authorId\":\"2239966124\",\"name\":\"Catherine Faron-Zucker\",\"aliases\":null,\"affiliations\":null,\"homepage\":null,\"paperCount\":null,\"citationCount\":null,\"hIndex\":null,\"url\":null},{\"authorId\":\"2287787061\",\"name\":\"Franck Michel\",\"aliases\":null,\"affiliations\":null,\"homepage\":null,\"paperCount\":null,\"citationCount\":null,\"hIndex\":null,\"url\":null},{\"authorId\":\"1514280691\",\"name\":\"Hanna Abi Akl\",\"aliases\":null,\"affiliations\":null,\"homepage\":null,\"paperCount\":null,\"citationCount\":null,\"hIndex\":null,\"url\":null}],\"tldr\":null,\"externalIds\":null,\"publicationVenue\":null,\"journal\":null}");

        Ok(())
    }
}
