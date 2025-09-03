use crate::{
    metrics::HashMap,
    session::common_traits::{BuildableTrait, BuilderTrait, SendableRecordBatchStreamMessageMap, StateMap},
    table::{
        table_publish::TablePublish,
        table_subscribe::{TableSubscribe, TableSubscribeTrait},
    },
    task::message::{
        MessageBuilderTrait, SendableRecordBatchStreamMessage,
    },
};

/// For task or processor objects that publish and
/// subscribe to messages
pub trait PubSubTrait {
    /// Get an immutable list of subscription subject names
    fn get_subscriptions(&self) -> Vec<&TableSubscribe>;

    /// Get an immutable list of publication subject names
    fn get_publications(&self) -> Vec<&TablePublish>;

    /// Get subscriptions from the state
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
    /// [OutgoingMessageMap] with unique names to prevent collisions in the `HashMap`
    fn get_subscriptions_from_state(
        &self,
        updates: &HashMap<String, bool>,
        state: &StateMap,
    ) -> SendableRecordBatchStreamMessageMap {
        let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for subscription in self.get_subscriptions().iter() {
            let updated = updates.get(subscription.get_table_name()).unwrap_or(&false);
            // default or dummy tables may not be found in the state so we just ignore them
            if let Some(table) = state.get(subscription.get_table_name()) {
                // OnUpdate... tables are not subscribed to if they have not been updated
                if let Some(message) = table
                    .try_read()
                    .unwrap()
                    .subscribe_table(subscription, *updated)
                {
                    let publications = self.get_publications();
                    let update = publications
                        .iter()
                        .filter(|p| p.get_table_name() == subscription.get_table_name())
                        .collect::<Vec<_>>();
                    let update = match update.first() {
                        Some(u) => u,
                        None => &TablePublish::None,
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
                    // DM: need to migrate map key to `make_random_name` to prevent hash collisions
                    //  when the same table is subscribed to by multiple processors in the chain...
                    let _ = map.insert(subscription.get_table_name().to_string(), out);
                }
            }
        }
        map
    }

    // Check if the criteria for reading the subscriptions has been full-filled
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool;
}
