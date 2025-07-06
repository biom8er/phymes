use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use crate::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::ArrowTableSubscribe,
        stream::{RecordBatchStream, SendableRecordBatchStream},
    },
    task::arrow_message::{
        ArrowMessageBuilderTrait, ArrowMessageTrait, ArrowOutgoingMessage,
        ArrowOutgoingMessageBuilderTrait, ArrowOutgoingMessageTrait,
    },
};

use super::arrow_processor::ArrowProcessorTrait;

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Schema, SchemaRef},
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tracing::{Level, event};

/// Processor that aggregates messages
///
/// # Notes
///
/// - There is no guarantee that the order of incoming
///   messages is preserved
/// - All incoming meessages MUST have the same schema
#[derive(Default, Debug)]
pub struct ArrowAggregatorProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl ArrowAggregatorProcessor {
    pub fn new_with_pub_sub_for(
        name: &str,
        publications: &[ArrowTablePublish],
        subscriptions: &[ArrowTableSubscribe],
        forward: &[&str],
    ) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            forward: forward.iter().map(|s| s.to_string()).collect(),
        })
    }
}

impl MappableTrait for ArrowAggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for ArrowAggregatorProcessor {
    fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: vec![ArrowTablePublish::None],
            subscriptions: vec![ArrowTableSubscribe::None],
            forward: Vec::new(),
        })
    }

    fn get_publications(&self) -> &[ArrowTablePublish] {
        &self.publications
    }

    fn get_subscriptions(&self) -> &[ArrowTableSubscribe] {
        &self.subscriptions
    }

    fn get_forward_subscriptions(&self) -> &[String] {
        self.forward.as_slice()
    }

    fn process(
        &self,
        message: OutgoingMessageMap,
        metrics: ArrowTaskMetricsSet,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Assert that each message has the same schema
        let mut schema = Arc::new(Schema::empty());
        let mut subject = String::new();
        let mut update = ArrowTablePublish::None;
        let mut input = Vec::with_capacity(message.len());
        for (i, (_k, v)) in message.into_iter().enumerate() {
            if i == 0 {
                schema = Arc::clone(&v.get_message().schema());
                subject = v.get_subject().to_owned();
                update = v.get_update().clone();
            }
            if !schema.eq(&v.get_message().schema()) {
                return Err(anyhow!(
                    "There is a mismatch between schemas in the messages provided to {}.",
                    self.get_name()
                ));
            }
            if update != *v.get_update() {
                return Err(anyhow!(
                    "There is a mismatch between updates in the messages provided to {}.",
                    self.get_name()
                ));
            }
            input.push(v.get_message_own());
        }

        // Make the outbox and send
        let out = Box::pin(ArrowAggregatorStream {
            schema,
            input,
            runtime_env: Arc::clone(&runtime_env),
            baseline_metrics: BaselineMetrics::new(&metrics, self.get_name()),
        });
        let out_m = ArrowOutgoingMessage::get_builder()
            .with_name(self.get_name())
            .with_publisher(self.get_name())
            .with_subject(subject.as_str())
            .with_message(out)
            .with_update(&update)
            .build()?;
        let mut outbox = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = outbox.insert(self.get_name().to_string(), out_m);
        Ok(outbox)
    }
}

#[allow(dead_code)]
pub struct ArrowAggregatorStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    input: Vec<SendableRecordBatchStream>,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
}

impl Stream for ArrowAggregatorStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.input.is_empty() {
            Poll::Ready(None)
        } else {
            // Initialize the metrics
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // Collect the input
            let mut batches = Vec::new();
            for i in self.input.as_mut_slice().iter_mut() {
                while let Some(Ok(batch)) = ready!(i.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
            }

            // Clear the input so that any subsequent pools will return None
            self.input.clear();

            // Concatenate into a single record batch
            let batch = ArrowTable::get_builder()
                .with_name("")
                .with_record_batches(batches)?
                .build()?
                .concat_record_batches()?
                .get_record_batches_own()
                .first()
                .unwrap()
                .to_owned();

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl RecordBatchStream for ArrowAggregatorStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use crate::table::arrow_table::{ArrowTableBuilder, test_table::make_test_table};

    use super::*;

    #[tokio::test]
    async fn test_arrow_aggregator_processor() -> Result<()> {
        // Case 1: mismatch between update targets
        let mut message_1 = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message_1.insert(
            "m1".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m1")
                .with_publisher("s1")
                .with_subject("task_1")
                .with_update(&ArrowTablePublish::Extend {
                    table_name: "t1".to_string(),
                })
                .with_message(make_test_table("t1", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("task_1")
                .with_update(&ArrowTablePublish::Extend {
                    table_name: "t2".to_string(),
                })
                .with_message(make_test_table("t2", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            token_service: None,
            tensor_service: None,
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        }));

        // Create the aggregator and run
        let agg_arc_1 = ArrowAggregatorProcessor::new_arc("aggregator_processor");
        match agg_arc_1.process(message_1, metrics.clone(), Arc::clone(&runtime_env)) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "There is a mismatch between updates in the messages provided to aggregator_processor."
            ),
        }

        // Case 2: no mismatches
        let mut message_1 = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message_1.insert(
            "m1".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m1")
                .with_publisher("s1")
                .with_subject("task_1")
                .with_update(&ArrowTablePublish::Extend {
                    table_name: "t1".to_string(),
                })
                .with_message(make_test_table("t1", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("task_1")
                .with_update(&ArrowTablePublish::Extend {
                    table_name: "t1".to_string(),
                })
                .with_message(make_test_table("t1", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );

        // Create the aggregator and run
        let agg_arc_1 = ArrowAggregatorProcessor::new_arc("aggregator_processor");
        let mut agg_stream =
            agg_arc_1.process(message_1, metrics.clone(), Arc::clone(&runtime_env))?;

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            agg_stream
                .remove("aggregator_processor")
                .unwrap()
                .get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 24);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 24);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        Ok(())
    }
}
