use anyhow::Result;
use phymes_diagnostics::HashMap;
use tracing::{Level, event};

use crate::{
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, TablePublication, TableSubscription, TableSubscriptionTrait, remove_message_by_subject
};

/// Subscribe to the subject
///
/// # Notes
///
/// * The update is taken from the first matching publication that is found as default,
///   but each processor should add the update when called.
/// * Each message is given a unique name to prevent collisions when multiple processors
///   subscribe to the same table.
///
/// # Arguments
///
/// * `updates` - `HashMap<String, bool>` where the key is the subscription table name
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
        } else if let Some(table) = subjects.get(subscription.get_table_name()) {
            if let Some(stream) = table.read().subscribe_to_table(subscription) {
                let update = publications
                    .iter()
                    .filter(|p| p.get_table_name() == subscription.get_table_name())
                    .collect::<Vec<_>>();
                let update = match update.first() {
                    Some(u) => u,
                    None => &TablePublication::None,
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
    }
    Ok(map)
}

/// Publish messages to the subject
///
/// # Note
///
/// * The publisher is updated to the publishing task/processor
/// * A unique name to protect against collisions when building the final message map is added
pub fn publish_to_subject(
    publisher_name: &str,
    publications: &[&TablePublication],
    messages: SendableRecordBatchStreamMessageMap,
) -> SendableRecordBatchStreamMessageMap {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for (name, message) in messages.into_iter() {
        let update = publications.iter()
            .filter(|p| p.get_table_name() == message.get_subject())
            .collect::<Vec<_>>();

        // Skip messages that are not in the publications
        if update.is_empty() {
            event!(
                Level::ERROR,
                "No publications found for message {name} on {} from {} during {publisher_name}",
                message.get_subject(),
                message.get_publisher(),
            );
            continue;
        }

        // Build the output message
        let out = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(publisher_name)
            .with_subject(message.get_subject())
            .with_update(update.first().unwrap())
            .with_message(message.get_message_own())
            .make_name()
            .unwrap()
            .build()
            .unwrap();
        let _ = map.insert(out.get_name().to_string(), out);
    }
    map
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
        let subscriptions = vec![TableSubscription::OnUpdateFullTable { table_name: table_name.to_string() }, TableSubscription::AlwaysFullTable { table_name: config_name.to_string()}];
        let publications = vec![TablePublication::Extend {table_name: table_name.to_string()}];
        let subjects = test_task::make_state(table_name, config_name)?;
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let mut messages = subscribe_to_subject(
            &subscriptions,
            &publications,
            &subjects,
            &mut stream
        )?;
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
        let subscriptions = vec![TableSubscription::OnUpdateFullTable { table_name: table_name.to_string() }, TableSubscription::OnUpdateFullTable { table_name: table_name_2.to_string() }, TableSubscription::AlwaysFullTable { table_name: config_name.to_string()}];
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let m = SendableRecordBatchStreamMessageBuilder::default()
            .with_subject(table_name_2)
            .with_publisher("")
            .with_update(&TablePublication::None)
            .with_message(test_table::make_test_table(table_name_2, 1, 0, 1)?.to_record_batch_stream())
            .make_random_name()?
            .build()?;
        let _ = stream.insert(m.get_name().to_string(), m);
        let mut messages = subscribe_to_subject(
            &subscriptions,
            &publications,
            &subjects,
            &mut stream
        )?;
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
        let subscriptions = vec![TableSubscription::OnUpdateFullTable { table_name: table_name.to_string() }, TableSubscription::AlwaysFullTable { table_name: config_name.to_string()}];
        let mut stream = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let m = SendableRecordBatchStreamMessageBuilder::default()
            .with_subject(table_name_2)
            .with_publisher("")
            .with_update(&TablePublication::None)
            .with_message(test_table::make_test_table(table_name_2, 1, 0, 1)?.to_record_batch_stream())
            .make_random_name()?
            .build()?;
        let _ = stream.insert(m.get_name().to_string(), m);
        let mut messages = subscribe_to_subject(
            &subscriptions,
            &publications,
            &subjects,
            &mut stream
        )?;
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
    fn test_publish_to_subject() -> Result<()> {
        let table_name = "test_table";
        let task_name = "test_task";
        let publications = vec![TablePublication::Extend {table_name: table_name.to_string()}];

        // Case 1: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_publisher("s1")
            .with_subject("d1")
            .with_update(&TablePublication::Extend {
                table_name: "d1".to_string(),
            })
            .with_message(test_table::make_test_table("d1", 1, 8, 2)?.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let inbox = publish_to_subject(task_name, &publications.iter().collect::<Vec<_>>(), messages);
        assert_eq!(inbox.len(), 0);

        // Case 2: Message has subject that the task does not publish on
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("test_message")
            .with_publisher("s1")
            .with_subject(table_name)
            .with_update(&TablePublication::Extend {
                table_name: table_name.to_string(),
            })
            .with_message(test_table::make_test_table(table_name, 1, 8, 2)?.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let inbox = publish_to_subject(task_name, &publications.iter().collect::<Vec<_>>(), messages);
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