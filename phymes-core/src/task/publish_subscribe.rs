use anyhow::Result;
use phymes_diagnostics::HashMap;

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
    is_subscription_updated: &[bool],
    publications: &[TablePublication],
    subjects: &StateMap,
    messages: &mut SendableRecordBatchStreamMessageMap,
) -> Result<SendableRecordBatchStreamMessageMap> {
    let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    for (subscription, is_subscription_updated) in subscriptions.iter()
        .zip(is_subscription_updated.iter()) {
        // Check for subscriptions in the subjects
        if let Some(table) = subjects.get(subscription.get_table_name()) {
            // OnUpdate... tables are not subscribed to if they have not been updated
            if let Some(stream) = table.read().subscribe_to_table(subscription, *is_subscription_updated) {
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
        // Check for subscriptions in the message stream
        } else if let Some(message) = remove_message_by_subject(subscription.get_table_name(), messages) {
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
fn publish_to_subject(
    publisher_name: &str,
    publications: &[TablePublication],
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
                "No publications found for message {} on {} from {} during {}",
                &name,
                message.get_subject(),
                message.get_publisher(),
                self.get_name()
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

/// For task or processor objects that publish and
/// subscribe to messages
pub trait PublishAndSubscribeTrait {
    /// Get an immutable list of subscription subject names
    fn get_subscriptions(&self) -> Vec<&TableSubscription>;

    /// Get an immutable list of publication subject names
    fn get_publications(&self) -> Vec<&TablePublication>;

    /// Get subscriptions from the state
    /// DM: Change name to `from_subscriptions_to_messages`
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
    fn get_subscriptions_from_state(
        &self,
        subscriptions: &[TableSubscription],
        is_subscription_updated: &[bool],
        publications: &[TablePublication],
        state: &StateMap,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let combined = subscriptions.iter()
            .zip(is_subscription_updated.iter())
            .collect::<Vec<_>>();
        for (subscription, is_subscription_updated) in combined {
            // default or dummy tables may not be found in the state so we just ignore them
            if let Some(table) = state.get(subscription.get_table_name()) {
                // OnUpdate... tables are not subscribed to if they have not been updated
                if let Some(message) = table.read().subscribe_to_table(subscription, *is_subscription_updated) {
                    let update = publications
                        .iter()
                        .filter(|p| p.get_table_name() == subscription.get_table_name())
                        .collect::<Vec<_>>();
                    let update = match update.first() {
                        Some(u) => u,
                        None => &TablePublication::None,
                    };
                    let out = SendableRecordBatchStreamMessage::get_builder()
                        .with_publisher("State")
                        .with_subject(subscription.get_table_name())
                        .with_update(update)
                        .with_message(message)
                        .make_random_name()
                        .unwrap()
                        .build()
                        .unwrap();
                    let _ = map.insert(out.get_name().to_string(), out);
                }
            }
        }
        Ok(map)
    }

    // Check if the criteria for reading the subscriptions has been full-filled
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool;
}
