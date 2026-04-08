use std::sync::Arc;

use anyhow::Result;
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_event::{Publication, Subscription};
use phymes_message::{MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilderMap, SendableRecordBatchStreamMessageMap, remove_message_by_subject};
use phymes_diagnostics::HashMap;

use crate::SubscriptionTrait;

/// Subscribe to the subject
///
/// # Notes
///
/// * The update for messages is heuristically determined by first trying to match on the subject name,
///   then by taking the first provided publication, and defaulting to None when no publications are provided
/// * Each message is given a unique name to prevent collisions when multiple processors
///   subscribe to the same table.
///
/// # Arguments
///
/// * `subscriptions` - `HashMap<String, bool>` where the key is the subscription table name
///   and the value is whether the table has been updated or not
/// * `publications`
/// * `runtime_env`
/// * `session_name`
/// * `messages`
///
/// # Returns
/// [SendableRecordBatchStreamMessageMap] with unique names to prevent collisions in the `HashMap`
pub fn subscribe_to_subject(
    subscriptions: &[Subscription],
    publications: &[Publication],
    runtime_env: &Arc<RuntimeEnv>,
    session_name: &str,
    messages: &mut SendableRecordBatchStreamMessageMap,
) -> Result<SendableRecordBatchStreamMessageMap> {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for subscription in subscriptions.iter() {
        // 1. Check for subscriptions in the message stream
        if let Some(message) = remove_message_by_subject(subscription.subject_name(), messages) {
            let _ = map.insert(message.get_name().to_string(), message);
        // 2. Check for subscriptions in the subjects
        } else {
            // A. Check for a matching subject in the publications
            let update = publications
                .iter()
                .filter(|p| p.subject_name() == subscription.subject_name())
                .collect::<Vec<_>>();
            let update = if let Some(update) = update.first() {
                update
            // B. Default to None
            } else {
                &Publication::None
            };

            // C. Get the subject
            let stream = subscription
                .subscribe_to_subject(runtime_env, session_name)?
                .unwrap();
            let message = SendableRecordBatchStreamMessage::get_builder()
                .with_publisher("Subjects")
                .with_subject(subscription.subject_name())
                .with_update(update)
                .with_message(stream)
                .make_random_name()?
                .build()?;
            let _ = map.insert(message.get_name().to_string(), message);
        }
    }
    Ok(map)
}

/// Publish messages to the subject
///
/// # Note
///
/// * The publisher is updated to the publishing task/processor
/// * A unique name to protect against collisions when building the final message map is added
/// * The update for messages matched to the first publication with the same subject name
pub fn build_and_publish_to_stream(
    publisher_name: &str,
    publications: &[&Publication],
    messages: SendableRecordBatchStreamMessageBuilderMap,
) -> Result<SendableRecordBatchStreamMessageMap> {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for (_name, message) in messages.into_iter() {
        // Try to find a matching publication based on the subject
        let updates = if let Some(subject) = message.subject.as_ref() {
            publications
                .iter()
                .filter(|p| p.subject_name() == subject)
                .collect::<Vec<_>>()
        } else {
            publications.iter().collect::<Vec<_>>()
        };

        // Build the messages with publications
        let builds = if let Some(publication) = updates.first() {
            message
                .with_subject(publication.subject_name())
                .with_publisher(publisher_name)
                .with_update(publication)
                .make_name()?
                .build()?
        // Skip messages that are not in the publications
        } else {
            continue;
        };

        let _ = map.insert(builds.get_name().to_string(), builds);
    }
    Ok(map)
}

/// Update the name of the publisher
pub fn update_publisher(
    publisher_name: &str,
    messages: SendableRecordBatchStreamMessageMap,
) -> Result<SendableRecordBatchStreamMessageMap> {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for (_name, message) in messages.into_iter() {
        let out = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(publisher_name)
            .with_subject(message.get_subject())
            .with_update(message.get_update())
            .with_message(message.get_message_own())
            .make_name()?
            .build()?;
        let _ = map.insert(out.get_name().to_string(), out);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use crate::{PublicationTrait, test_task};
    use futures::TryStreamExt;
    use phymes_core::{SubjectPlanTrait, SubjectTrait, test_subject};
    use phymes_message::SendableRecordBatchStreamMessageBuilder;

    use super::*;

    #[tokio::test]
    async fn test_subscribe_to_subject() -> Result<()> {
        // --- Case 1: from subjects ---
        // Create the subscriptions/publications
        let subject_name = "test_table";
        let config_name = "test_config";
        let subscriptions = vec![
            Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name.to_string(),
            },
            Subscription::AlwaysAllRecordBatches {
                subject_name: config_name.to_string(),
            },
        ];
        let publications = vec![Publication::Extend {
            subject_name: subject_name.to_string(),
        }];

        // Create the runtime environment
        let runtime_env = Arc::new(RuntimeEnv::default());

        // Create the Tables
        let subjects = test_task::make_subjects(subject_name, config_name)?;
        for subject in subjects {
            let _publication: Vec<_> = Publication::Extend {
                subject_name: subject.get_name().to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                subject.subject_own().get_record_batches_own(),
                0,
                "",
                "test_session",
            )?
            .unwrap()
            .try_collect()
            .await?;
        }

        // Create the stream
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();

        // Test
        let mut messages = subscribe_to_subject(
            &subscriptions,
            &publications,
            &runtime_env,
            "test_session",
            &mut stream,
        )?;
        assert_eq!(messages.len(), 2);
        assert_eq!(
            remove_message_by_subject(subject_name, &mut messages)
                .unwrap()
                .get_subject(),
            subject_name
        );
        assert_eq!(
            remove_message_by_subject(config_name, &mut messages)
                .unwrap()
                .get_subject(),
            config_name
        );

        // --- Case 2: from subjects and messages ---
        // Create the subscriptions/publications
        let table_name_2 = "test_table_2";
        let subscriptions = vec![
            Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name.to_string(),
            },
            Subscription::OnUpdateAllRecordBatches {
                subject_name: table_name_2.to_string(),
            },
            Subscription::AlwaysAllRecordBatches {
                subject_name: config_name.to_string(),
            },
        ];

        // Create the stream
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let m = SendableRecordBatchStreamMessageBuilder::default()
            .with_subject(table_name_2)
            .with_publisher("")
            .with_update(&Publication::None)
            .with_message(
                test_subject::make_test_subject(table_name_2, 1, 0, 1)?.to_record_batch_stream(),
            )
            .make_random_name()?
            .build()?;
        let _ = stream.insert(m.get_name().to_string(), m);

        // Test
        let mut messages = subscribe_to_subject(
            &subscriptions,
            &publications,
            &runtime_env,
            "test_session",
            &mut stream,
        )?;
        assert_eq!(messages.len(), 3);
        assert_eq!(
            remove_message_by_subject(subject_name, &mut messages)
                .unwrap()
                .get_subject(),
            subject_name
        );
        assert_eq!(
            remove_message_by_subject(config_name, &mut messages)
                .unwrap()
                .get_subject(),
            config_name
        );
        assert_eq!(
            remove_message_by_subject(table_name_2, &mut messages)
                .unwrap()
                .get_subject(),
            table_name_2
        );

        // --- Case 3: from subjects and messages but missing the messages table ---
        // Create the subscriptions/publications
        let subscriptions = vec![
            Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_name.to_string(),
            },
            Subscription::AlwaysAllRecordBatches {
                subject_name: config_name.to_string(),
            },
        ];

        // Create the stream
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let m = SendableRecordBatchStreamMessageBuilder::default()
            .with_subject(table_name_2)
            .with_publisher("")
            .with_update(&Publication::None)
            .with_message(
                test_subject::make_test_subject(table_name_2, 1, 0, 1)?.to_record_batch_stream(),
            )
            .make_random_name()?
            .build()?;
        let _ = stream.insert(m.get_name().to_string(), m);

        // Test
        let mut messages = subscribe_to_subject(
            &subscriptions,
            &publications,
            &runtime_env,
            "test_session",
            &mut stream,
        )?;
        assert_eq!(messages.len(), 2);
        assert_eq!(
            remove_message_by_subject(subject_name, &mut messages)
                .unwrap()
                .get_subject(),
            subject_name
        );
        assert_eq!(
            remove_message_by_subject(config_name, &mut messages)
                .unwrap()
                .get_subject(),
            config_name
        );

        Ok(())
    }

    #[test]
    fn test_update_publisher() -> Result<()> {
        let subject_name = "test_table";
        let task_name = "test_task";

        // Case 2: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_publisher("s1")
            .with_subject(subject_name)
            .with_update(&Publication::Extend {
                subject_name: subject_name.to_string(),
            })
            .with_message(
                test_subject::make_test_subject(subject_name, 1, 8, 2)?.to_record_batch_stream(),
            )
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let inbox = update_publisher(task_name, messages)?;
        assert_eq!(inbox.len(), 1);
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            task_name
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            subject_name
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            Publication::Extend {
                subject_name: subject_name.to_string()
            }
        );
        Ok(())
    }

    #[test]
    fn test_build_and_publish_to_stream() -> Result<()> {
        let subject_name = "test_table";
        let task_name = "test_task";
        let publications = [Publication::Extend {
            subject_name: subject_name.to_string(),
        }];

        // Case 1: Message does not have a subject
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_message(test_subject::make_test_subject("d1", 1, 8, 2)?.to_record_batch_stream());
        let _ = messages.insert("test_message".to_string(), message);
        let inbox = build_and_publish_to_stream(
            task_name,
            &publications.iter().collect::<Vec<_>>(),
            messages,
        )?;
        assert_eq!(inbox.len(), 1);
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            task_name
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            subject_name
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            Publication::Extend {
                subject_name: subject_name.to_string()
            }
        );

        // Case 2: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_subject("d1")
            .with_message(test_subject::make_test_subject("d1", 1, 8, 2)?.to_record_batch_stream());
        let _ = messages.insert("test_message".to_string(), message);
        let inbox = build_and_publish_to_stream(
            task_name,
            &publications.iter().collect::<Vec<_>>(),
            messages,
        )?;
        assert_eq!(inbox.len(), 0);

        // Case 3: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_message(
                test_subject::make_test_subject(subject_name, 1, 8, 2)?.to_record_batch_stream(),
            );
        let _ = messages.insert("test_message".to_string(), message);
        let inbox = build_and_publish_to_stream(
            task_name,
            &publications.iter().collect::<Vec<_>>(),
            messages,
        )?;
        assert_eq!(inbox.len(), 1);
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            task_name
        );
        assert_eq!(
            inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            subject_name
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            Publication::Extend {
                subject_name: subject_name.to_string()
            }
        );

        Ok(())
    }
}
