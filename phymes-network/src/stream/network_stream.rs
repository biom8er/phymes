use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::Result;
use futures::{FutureExt, Stream};
use phymes_diagnostics::HashMap;
use phymes_message::{IPCMessage, IPCMessageMap};
use tracing::{Level, event};

use crate::{Network, NetworkStreamStep, NetworkStreamStepTrait};

pub struct NetworkStream {
    /// The network context
    network: Arc<Network>,
    /// The next superstep
    #[allow(clippy::type_complexity)]
    next_step: Option<Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>>,
    /// The maximum number of supersteps
    max_steps: usize,
    /// The current dynamic step
    step: usize,
}

impl NetworkStream {
    /// New [NetworkStream]
    pub fn new(messages: IPCMessageMap, network: Arc<Network>) -> Self {
        let max_steps = network.get_max_steps();
        let step = 0;
        #[allow(clippy::type_complexity)]
        let next_step: Option<
            Pin<Box<dyn Future<Output = Result<Option<IPCMessageMap>>> + Send>>,
        > = Some(Box::pin(NetworkStreamStep::run_superstep(
            Arc::clone(&network),
            messages,
        )));

        Self {
            network,
            next_step,
            max_steps,
            step,
        }
    }
}

impl Stream for NetworkStream {
    type Item = Result<IPCMessageMap>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // An internal while loop is needed to skip empty messages without causing an infinite poll
        while self.step < self.max_steps {
            // Poll the next superstep
            let messages = if let Some(fut) = self.next_step.as_mut() {
                match ready!(fut.poll_unpin(cx)) {
                    Ok(Some(messages)) => messages,
                    Ok(None) => return Poll::Ready(None),
                    Err(err) => {
                        event!(Level::ERROR, "{err:?}");
                        HashMap::<String, IPCMessage>::new()
                    }
                }
            } else {
                return Poll::Ready(None);
            };

            // Prepare the next superstep
            self.next_step = Some(Box::pin(NetworkStreamStep::run_superstep(
                Arc::clone(&self.network),
                HashMap::<String, IPCMessage>::new(),
            )));
            self.step += 1;

            // Return the superstep result
            if messages.is_empty() {
                // Skip empty results
                continue;
            } else {
                return Poll::Ready(Some(Ok(messages)));
            }
        }
        event!(
            Level::DEBUG,
            "Maximum iterations {} exeeded.",
            self.max_steps
        );
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.max_steps))
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use phymes_event::{Publication, Subscription};
    use phymes_message::MessageTrait;
    use phymes_schemas::AvailableSubjects;
    use phymes_subject::{
        BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::{SubscriptionTrait, test_task};

    use super::*;
    use crate::{NetworkBuilderAppsTrait, NetworkBuilderTrait, test_network_builder};

    #[tokio::test]
    async fn test_network_stream_replace_state_update_sequential_tasks() -> Result<()> {
        // Build the network
        let (network, network_messages) =
            test_network_builder::make_test_network_builder_sequential("network_1", 2)?
                .with_diagnostics(true)
                .add_network_interface(Some(&["state_1"]))?
                .add_next_tasks()?
                .add_next_supersteps()?
                .build_with_tables()?;
        let network_arc = Arc::new(network);
        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;
        let messages = test_task::make_test_input_message(
            "task_1",
            "network_1",
            "state_1",
            "state_1",
            &Publication::Replace {
                subject_name: "state_1".to_string(),
            },
            true,
        )?;
        let network_stream = NetworkStream::new(messages, Arc::clone(&network_arc));
        let mut response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        // Check the response
        assert_eq!(response.len(), 1);
        assert_eq!(response.last().unwrap().len(), 1);
        assert_eq!(
            response
                .last()
                .unwrap()
                .get("from_network_1_on_state_1")
                .unwrap()
                .get_name(),
            "from_network_1_on_state_1"
        );
        assert_eq!(
            response
                .last()
                .unwrap()
                .get("from_network_1_on_state_1")
                .unwrap()
                .get_publisher(),
            "network_1"
        );
        assert_eq!(
            response
                .last()
                .unwrap()
                .get("from_network_1_on_state_1")
                .unwrap()
                .get_subject(),
            "state_1"
        );
        assert_eq!(
            *response
                .last()
                .unwrap()
                .get("from_network_1_on_state_1")
                .unwrap()
                .get_update(),
            Publication::Extend {
                subject_name: "state_1".to_string()
            }
        );
        let bytes = response
            .pop()
            .unwrap()
            .remove("from_network_1_on_state_1")
            .unwrap()
            .get_message_own();
        let partitions = SubjectBuilder::new_from_ipc_stream(&bytes)?
            .with_name("")
            .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 7); // DM, Check!(): changed from 4

        // check the network and network
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "state_1".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 6);
        assert_eq!(subscriptions.last().unwrap().num_rows(), 7);

        // Check the traces, events, and metrics tables
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::NetworkTasksRunLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 3); // DM, Check!(): changed from 5
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsChangeLog.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 5); // DM, Check!(): changed from 8
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SubjectsNumRows.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::NetworkMetrics.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 1);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::NetworkErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 0);
        let subscriptions: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::NetworkSupersteps.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), "network_1")?
        .unwrap()
        .try_collect()
        .await?;
        assert_eq!(subscriptions.len(), 2);

        Ok(())
    }
}
