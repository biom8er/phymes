use anyhow::Result;
use phymes_diagnostics::HashMap;

use crate::{
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap,
    TablePublication, TableSubscription, TableSubscriptionTrait,
    message::SendableRecordBatchStreamMessageBuilderMap, remove_message_by_subject,
};

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
/// * `state` - [StateMap] with the subjects
///
/// # Returns
/// [SendableRecordBatchStreamMessageMap] with unique names to prevent collisions in the `HashMap`
pub fn subscribe_to_subject(
    subscriptions: &[TableSubscription],
    publications: &[TablePublication],
    subjects: &StateMap,
    messages: &mut SendableRecordBatchStreamMessageMap,
) -> Result<SendableRecordBatchStreamMessageMap> {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for subscription in subscriptions.iter() {
        // 1. Check for subscriptions in the message stream
        if let Some(message) = remove_message_by_subject(subscription.get_table_name(), messages) {
            let _ = map.insert(message.get_name().to_string(), message);
        // 2. Check for subscriptions in the subjects
        } else if let Some(table) = subjects.get(subscription.get_table_name())
            && let Some(stream) = table.read().subscribe_to_table(subscription)
        {
            // a. check for a matching subject in the publications
            let update = publications
                .iter()
                .filter(|p| p.get_table_name() == subscription.get_table_name())
                .collect::<Vec<_>>();
            let update = if let Some(u) = update.first() {
                u
            // // b. use the first publication provided
            // // DM: fails message check for consistency between subject and publish subject
            // } else if let Some(u) = publications.first() {
            //     u
            // c. default to None
            } else {
                &TablePublication::None
            };
            let message = SendableRecordBatchStreamMessage::get_builder()
                .with_publisher("State")
                .with_subject(subscription.get_table_name())
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
    publications: &[&TablePublication],
    messages: SendableRecordBatchStreamMessageBuilderMap,
) -> Result<SendableRecordBatchStreamMessageMap> {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for (_name, message) in messages.into_iter() {
        // Try to find a matching publication based on the subject
        let updates = if let Some(subject) = message.subject.as_ref() {
            publications
                .iter()
                .filter(|p| p.get_table_name() == subject)
                .collect::<Vec<_>>()
        } else {
            publications.iter().collect::<Vec<_>>()
        };

        // Build the messages with publications
        let builds = if let Some(publication) = updates.first() {
            message
                .with_subject(publication.get_table_name())
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
    use crate::{SendableRecordBatchStreamMessageBuilder, TableTrait, test_table, test_task};

    use super::*;

    #[test]
    fn test_subscribe_to_subject() -> Result<()> {
        // Case 1: from subjects
        let table_name = "test_table";
        let config_name = "test_config";
        let subscriptions = vec![
            TableSubscription::OnUpdateFullTable {
                table_name: table_name.to_string(),
            },
            TableSubscription::AlwaysFullTable {
                table_name: config_name.to_string(),
            },
        ];
        let publications = vec![TablePublication::Extend {
            table_name: table_name.to_string(),
        }];
        let subjects = test_task::make_state(table_name, config_name)?;
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let mut messages =
            subscribe_to_subject(&subscriptions, &publications, &subjects, &mut stream)?;
        assert_eq!(messages.len(), 2);
        assert_eq!(
            remove_message_by_subject(table_name, &mut messages)
                .unwrap()
                .get_subject(),
            table_name
        );
        assert_eq!(
            remove_message_by_subject(config_name, &mut messages)
                .unwrap()
                .get_subject(),
            config_name
        );

        // Case 2: from subjects and messages
        let table_name_2 = "test_table_2";
        let subscriptions = vec![
            TableSubscription::OnUpdateFullTable {
                table_name: table_name.to_string(),
            },
            TableSubscription::OnUpdateFullTable {
                table_name: table_name_2.to_string(),
            },
            TableSubscription::AlwaysFullTable {
                table_name: config_name.to_string(),
            },
        ];
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let m = SendableRecordBatchStreamMessageBuilder::default()
            .with_subject(table_name_2)
            .with_publisher("")
            .with_update(&TablePublication::None)
            .with_message(
                test_table::make_test_table(table_name_2, 1, 0, 1)?.to_record_batch_stream(),
            )
            .make_random_name()?
            .build()?;
        let _ = stream.insert(m.get_name().to_string(), m);
        let mut messages =
            subscribe_to_subject(&subscriptions, &publications, &subjects, &mut stream)?;
        assert_eq!(messages.len(), 3);
        assert_eq!(
            remove_message_by_subject(table_name, &mut messages)
                .unwrap()
                .get_subject(),
            table_name
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

        // Case 3: from subjects and messages but missing the messages table
        let subscriptions = vec![
            TableSubscription::OnUpdateFullTable {
                table_name: table_name.to_string(),
            },
            TableSubscription::AlwaysFullTable {
                table_name: config_name.to_string(),
            },
        ];
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let m = SendableRecordBatchStreamMessageBuilder::default()
            .with_subject(table_name_2)
            .with_publisher("")
            .with_update(&TablePublication::None)
            .with_message(
                test_table::make_test_table(table_name_2, 1, 0, 1)?.to_record_batch_stream(),
            )
            .make_random_name()?
            .build()?;
        let _ = stream.insert(m.get_name().to_string(), m);
        let mut messages =
            subscribe_to_subject(&subscriptions, &publications, &subjects, &mut stream)?;
        assert_eq!(messages.len(), 2);
        assert_eq!(
            remove_message_by_subject(table_name, &mut messages)
                .unwrap()
                .get_subject(),
            table_name
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
        let table_name = "test_table";
        let task_name = "test_task";

        // Case 2: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_publisher("s1")
            .with_subject(table_name)
            .with_update(&TablePublication::Extend {
                table_name: table_name.to_string(),
            })
            .with_message(
                test_table::make_test_table(table_name, 1, 8, 2)?.to_record_batch_stream(),
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
            table_name
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            TablePublication::Extend {
                table_name: table_name.to_string()
            }
        );
        Ok(())
    }

    #[test]
    fn test_build_and_publish_to_stream() -> Result<()> {
        let table_name = "test_table";
        let task_name = "test_task";
        let publications = [TablePublication::Extend {
            table_name: table_name.to_string(),
        }];

        // Case 1: Message does not have a subject
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_message(test_table::make_test_table("d1", 1, 8, 2)?.to_record_batch_stream());
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
            table_name
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            TablePublication::Extend {
                table_name: table_name.to_string()
            }
        );

        // Case 2: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_subject("d1")
            .with_message(test_table::make_test_table("d1", 1, 8, 2)?.to_record_batch_stream());
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
                test_table::make_test_table(table_name, 1, 8, 2)?.to_record_batch_stream(),
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
            table_name
        );
        assert_eq!(
            *inbox
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_update(),
            TablePublication::Extend {
                table_name: table_name.to_string()
            }
        );

        Ok(())
    }
}
