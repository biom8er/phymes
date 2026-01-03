use anyhow::Result;
use phymes_diagnostics::HashMap;

use crate::{
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, TablePublication, TableSubscription, TableSubscriptionTrait
};

/// For task or processor objects that publish and
/// subscribe to messages
pub trait PublishAndSubscribeTrait {
    /// Get an immutable list of subscription subject names
    fn get_subscriptions(&self) -> Vec<&TableSubscription>;

    /// Get an immutable list of publication subject names
    fn get_publications(&self) -> Vec<&TablePublication>;

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
    /// [SendableRecordBatchStreamMessageMap] with unique names to prevent collisions in the `HashMap`
    fn get_subscriptions_from_state(
        &self,
        subscription_names: &[&str], 
        subscription_table_names: &[&str], 
        is_updated: &[bool],
        state: &StateMap,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        let mut map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let combined = subscription_names.iter()
            .zip(subscription_table_names.iter())
            .zip(is_updated.iter())
            .map(|((a, b), c)| (a, b, c))
            .collect::<Vec<_>>();
        for (subscription_name, subscription_table_name, is_updated) in combined {
            // default or dummy tables may not be found in the state so we just ignore them
            if let Some(table) = state.get(*subscription_table_name) {
                let subscription = TableSubscription::from_str_fuzzy(subscription_name, subscription_table_name)?;
                // OnUpdate... tables are not subscribed to if they have not been updated
                if let Some(message) = table.read().subscribe_table(&subscription, *is_updated) {
                    let publications = self.get_publications();
                    let update = publications
                        .iter()
                        .filter(|p| &p.get_table_name() == subscription_table_name)
                        .collect::<Vec<_>>();
                    let update = match update.first() {
                        Some(u) => u,
                        None => &TablePublication::None,
                    };
                    let out = SendableRecordBatchStreamMessage::get_builder()
                        .with_publisher("State")
                        .with_subject(subscription_table_name)
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
