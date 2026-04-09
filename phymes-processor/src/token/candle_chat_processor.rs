use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, Subject, SubjectBuilder,
    SubjectBuilderTrait, SubjectTrait,
};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_event::Publication;
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::CandleChatConfig;
use phymes_ml::TokenStreamTrait;
use phymes_streams::CandleChatStream;
use tracing::{Level, event, instrument};

use crate::{ProcessorTrait, TokenStreamTraitExt};

/// Processor for text generation inference (TGI) using Candle models
#[derive(Debug)]
pub struct CandleChatProcessor {
    name: String,
    r#type: String,
    token_service: Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>,
}

impl MappableTrait for CandleChatProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for CandleChatProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
            token_service: Arc::new(Mutex::new(None)),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn line_and_file(&self) -> (u32, String) {
        (line!(), file!().to_string())
    }

    #[instrument(skip(self, message, diagnostic_builder, runtime_env))]
    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => {
                return Err(anyhow!(
                    "Config not provided for {}. Available messages are {:?}",
                    self.get_name(),
                    message.keys()
                ));
            }
        };

        // Run the chat stream
        let out = Box::pin(CandleChatStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            Arc::clone(self.token_service()),
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

impl TokenStreamTraitExt for CandleChatProcessor {
    fn token_service(&self) -> &Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>> {
        &self.token_service
    }
}

pub mod bench_chat_processor {
    #[cfg(all(not(feature = "candle"), feature = "api"))]
    use crate::OpenAIChatProcessor;
    use phymes_streams::ChatBuilderTraitExt;

    use super::*;

    /// Run the chat processor with a given config and return the message history
    pub async fn bench_chat_processor(
        diagnostic_builder: Option<&DiagnosticBuilder>,
        config: &CandleChatConfig,
        user_content: &str,
        name: &str,
    ) -> Result<Subject> {
        // State for the chat processor config
        let candle_chat_config_json = serde_json::to_vec(config)?;
        let candle_chat_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = SubjectBuilder::new()
            .with_name(&config.messages)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(user_content, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config.messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(&config.messages)
                .with_publisher("")
                .with_subject(&config.messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            candle_chat_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(candle_chat_config_table.get_name())
                .with_publisher("")
                .with_subject(candle_chat_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(candle_chat_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        #[allow(unused_variables)]
        let chat_processor = CandleChatProcessor::new(name, CandleChatProcessor::get_static_name());
        #[cfg(all(not(feature = "candle"), feature = "api"))]
        let chat_processor = OpenAIChatProcessor::new(name, OpenAIChatProcessor::get_static_name());
        let mut stream = chat_processor.process(
            message,
            diagnostic_builder,
            Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?),
        )?;

        // Update the chat history with the response
        let (message_builder, _stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove(name).unwrap().message.take().unwrap(),
                1000,
            )
            .await?;
        message_builder.clone().build()
    }
}

#[cfg(test)]
mod tests {
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_ml::AvailableCandleAssets;
    use phymes_streams::ChatBuilderTraitExt;

    use super::*;

    #[tokio::test]
    async fn test_candle_chat_processor() -> Result<(), Box<dyn std::error::Error>> {
        let name = "CandleChatProcessor";
        let messages = "messages";

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the chat processor config
        let candle_chat_config = CandleChatConfig {
            messages: messages.to_string(),
            max_tokens: 1000,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(AvailableCandleAssets::SmolLM2_135MChat),
            ..Default::default()
        };

        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config)?;
        let candle_chat_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
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
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            candle_chat_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(candle_chat_config_table.get_name())
                .with_publisher("")
                .with_subject(candle_chat_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(candle_chat_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        let chat_processor = CandleChatProcessor::new(name, "");
        let mut stream = chat_processor.process(
            message,
            Some(&diagnostic_builder),
            Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?),
        )?;

        // DM: Skip actually running the tests as they take too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // Update the chat history with the response
            let (message_builder, _stream) = message_builder
                .append_chat_response_sendable_record_batch_stream(
                    &mut stream.remove(name).unwrap().message.take().unwrap(),
                    1000,
                )
                .await?;
            let messages = message_builder.clone().build()?;
            let json_data = messages.to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{}: {}", row["role"], row["content"])
                }
            }

            // Expected
            // "\nimport math\n\ndef count_primes(n):\n    \"\"\"\n    Finds all prime numbers up to n and counts them.\n    \n    Args:\n        n (int): The upper limit of the range to find primes in.\n\n    Returns:\n        int: The total number of prime numbers found.\n    \"\"\"\n\n    # Initialize a boolean array that indicates the primality of each number\n    is_prime = [True for _ in range(n + 1)]\n\n    # Set initial values based on small numbers and even numbers\n    i, p = 2, 3\n    while i * i <= n:\n        if is_prime[i]:\n            j = (i * i)\n            while j <= n:\n                is_prime[j] = False\n                j += i\n        i += 1\n\n    # Count the number of primality\n    return sum(1 for num in range(2, n + 1) if is_prime[num])"

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "system");
            assert_eq!(
                json_data.first().unwrap().get("content").unwrap(),
                "You are a helpful assistant."
            );
            assert_eq!(json_data.get(1).unwrap().get("role").unwrap(), "user");
            assert_eq!(
                json_data.get(1).unwrap().get("content").unwrap(),
                "Write a function to count prime numbers up to N."
            );
            assert_eq!(json_data.get(2).unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.get(2).unwrap().get("content").is_some());
        }

        Ok(())
    }
}
