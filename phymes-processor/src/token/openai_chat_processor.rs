use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::OpenAIChatStream;
use tracing::{Level, event};

use crate::ProcessorTrait;

#[derive(Debug)]
pub struct OpenAIChatProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for OpenAIChatProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for OpenAIChatProcessor {
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

        // Re-index the messages by the subject name which needs to be unique at this stage
        let message = message
            .into_iter()
            .map(|(_k, v)| (v.get_subject().to_string(), v))
            .collect::<HashMap<_, _>>();

        // Run the chat stream
        let out = Box::pin(OpenAIChatStream::new(
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

#[cfg(all(not(feature = "candle"), feature = "api"))]
#[cfg(test)]
mod tests {
    use super::*;
    use phymes_subject::{ChatBuilderTraitExt, Publication, SubjectBuilder};
    use phymes_diagnostics::{DiagnosticBuilder, Diagnostics, HashMap, SpanBuilder};

    use crate::AvailableOpenAIAssets;

    #[tokio::test]
    async fn test_openai_chat_processor() -> Result<()> {
        let name = "OpenAIChatProcessor";
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
            api_url: Some("http://0.0.0.0:8000/v1".to_string()),
            openai_asset: Some(AvailableOpenAIAssets::MetaLlamaV3p2_1B),
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
        let chat_processor = OpenAIChatProcessor::new(name, OpenAIChatProcessor::get_static_name());
        let mut stream = chat_processor.process(
            message,
            Some(&diagnostic_builder),
            Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?),
        )?;

        // Update the chat history with the response
        let (message_builder, _stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove(messages).unwrap().message.take().unwrap(),
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
        // "**Counting Prime Numbers Up to N**\n=====================================\n\nHere is a Python function that counts prime numbers up to a given number `N`:\n\n```python\ndef count_prime_numbers(n):\n    \"\"\"\n    Returns the count of prime numbers up to n.\n\n    Args:\n        n (int): The upper limit (exclusive) for counting prime numbers.\n\n    Returns:\n        int: The count of prime numbers up to n.\n    \"\"\"\n    def is_prime(num):\n        \"\"\"\n        Checks if a number is prime.\n\n        Args:\n            num (int): The number to check.\n\n        Returns:\n            bool: True if the number is prime, False otherwise.\n        \"\"\"\n        if num < 2:\n            return False\n        for i in range(2, int(num ** 0.5) + 1):\n            if num % i == 0:\n                return False\n        return True\n\n    count = 0\n    for i in range(2, n):\n        if is_prime(i):\n            count += 1\n    return count\n```\n\n**Example Use Cases**\n---------------------\n\n```python\n# Count prime numbers up to 20\nprint(count_prime_numbers(20))  # Output: 8\n\n# Count prime numbers up to 50\nprint(count_prime_numbers(50))  # Output: 15\n```\n\nThis function works by defining a helper function `is_prime` that checks whether a given number is prime or not. It then uses a simple loop to iterate from 2 to `n-1`, and increments the count each time it finds a prime number. The final count is returned by the main function `count_prime_numbers`."

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

        Ok(())
    }
}
