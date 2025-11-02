use phymes_diagnostics::HashMap;
use std::fmt::Debug;

use crate::{AvailableSubjects, TableSubscription, session::{MappableTrait, StateMap}};

use super::table_trait::{Table, TableTrait};

/// Determine when all subscriptions are ready
pub trait TableSubscribePolicyTrait: MappableTrait + Debug + Send + Sync {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscription],
        updates: &HashMap<String, bool>,
        state: &StateMap,
    ) -> bool;
    fn new_box() -> Box<dyn TableSubscribePolicyTrait>
    where
        Self: Sized;
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait>;
}

/// Always subscribe (dummy subscription check for testing)
#[derive(Default, Debug, Clone)]
pub struct AlwaysSubscribe;

impl TableSubscribePolicyTrait for AlwaysSubscribe {
    fn check_subscriptions(
        &self,
        _subscriptions: &[TableSubscription],
        _updates: &HashMap<String, bool>,
        _state: &StateMap,
    ) -> bool {
        true
    }
    fn new_box() -> Box<dyn TableSubscribePolicyTrait>
    where
        Self: Sized,
    {
        Box::new(Self)
    }
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AlwaysSubscribe {
    fn get_static_name() -> &'static str {
        "Always"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when any matching table name has been updated
#[derive(Default, Debug, Clone)]
pub struct AnyTableNameSubscribe;

impl TableSubscribePolicyTrait for AnyTableNameSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscription],
        updates: &HashMap<String, bool>,
        _state: &StateMap,
    ) -> bool {
        let mut is_update_count: usize = 0;
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                is_update_count += 1;
                if *updates.get(subscription.get_table_name()).unwrap_or(&false) {
                    return true;
                }
            }
        }
        is_update_count == 0
    }
    fn new_box() -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AnyTableNameSubscribe {
    fn get_static_name() -> &'static str {
        "Any"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when all matching table names has been updated
#[derive(Default, Debug, Clone)]
pub struct AllTableNamesSubscribe;

impl TableSubscribePolicyTrait for AllTableNamesSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscription],
        updates: &HashMap<String, bool>,
        _state: &StateMap,
    ) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update()
                & !updates.get(subscription.get_table_name()).unwrap_or(&true)
            {
                return false;
            }
        }
        true
    }
    fn new_box() -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AllTableNamesSubscribe {
    fn get_static_name() -> &'static str {
        "All"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when any matching table schema has been updated
#[derive(Default, Debug, Clone)]
pub struct AnyTableSchemaSubscribe;

impl TableSubscribePolicyTrait for AnyTableSchemaSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscription],
        updates: &HashMap<String, bool>,
        state: &StateMap,
    ) -> bool {
        let mut is_update_count: usize = 0;
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                is_update_count += 1;
                for (table_name, update) in updates {
                    if state
                        .get(subscription.get_table_name())
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_schema()
                        .eq(&state
                            .get(table_name)
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .get_schema())
                        & *update
                    {
                        return true;
                    }
                }
            }
        }
        is_update_count == 0
    }
    fn new_box() -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AnyTableSchemaSubscribe {
    fn get_static_name() -> &'static str {
        "AnySchema"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when all matching table schemas has been updated
#[derive(Default, Debug, Clone)]
pub struct AllTableSchemasSubscribe;

impl TableSubscribePolicyTrait for AllTableSchemasSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscription],
        updates: &HashMap<String, bool>,
        state: &StateMap,
    ) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                for (table_name, update) in updates {
                    if state
                        .get(subscription.get_table_name())
                        .unwrap()
                        .try_read()
                        .unwrap()
                        .get_schema()
                        .eq(&state
                            .get(table_name)
                            .unwrap()
                            .try_read()
                            .unwrap()
                            .get_schema())
                        & !*update
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
    fn new_box() -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AllTableSchemasSubscribe {
    fn get_static_name() -> &'static str {
        "AllSchema"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Custom subscription to pull in all of the relevant content for the chat
#[derive(Default, Debug, Clone)]
pub struct ChatContentSubscribe {
    user_message_table_name: String,
    tool_message_table_name: String,
    error_message_table_name: String,
}

#[allow(dead_code)]
impl ChatContentSubscribe {
    pub fn new_box_with_table_names(
        user_message_table_name: &str,
        tool_message_table_name: &str,
        error_message_table_name: &str,
    ) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(Self {
            user_message_table_name: user_message_table_name.to_string(),
            tool_message_table_name: tool_message_table_name.to_string(),
            error_message_table_name: error_message_table_name.to_string(),
        })
    }
}

impl TableSubscribePolicyTrait for ChatContentSubscribe {
    fn check_subscriptions(
        &self,
        _subscriptions: &[TableSubscription],
        updates: &HashMap<String, bool>,
        _state: &StateMap,
    ) -> bool {
        let user = updates.get(&self.user_message_table_name).unwrap_or(&false);
        let tool = updates.get(&self.tool_message_table_name).unwrap_or(&false);
        let error = updates.get(&self.error_message_table_name).unwrap_or(&false);
        *tool || *user || *error
    }
    fn new_box() -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(Self {
            // DM: dangerous as the strings needs to stay syncronized with the actual table names
            // in `AvailableinterfaceSubjects` and `AvailableinterfaceSubjects`
            user_message_table_name: "UserMessages".to_string(),
            tool_message_table_name: "ToolMessages".to_string(),
            error_message_table_name: AvailableSubjects::SessionErrors.to_string(),
        })
    }
    fn clone_boxed(&self) -> Box<dyn TableSubscribePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for ChatContentSubscribe {
    fn get_name(&self) -> &str {
        "ChatContentSubscribe"
    }
}

mod test_subscribe {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::table::table_trait::test_table::make_test_table;

    use super::*;

    #[allow(dead_code)]
    pub fn make_test_state() -> StateMap {
        let mut state = HashMap::<String, Arc<RwLock<Table>>>::new();
        state.insert(
            "t1".to_string(),
            Arc::new(RwLock::new(make_test_table("t1", 1, 0, 1).unwrap())),
        );
        state.insert(
            "t2".to_string(),
            Arc::new(RwLock::new(make_test_table("t2", 1, 0, 1).unwrap())),
        );
        state.insert(
            "t3".to_string(),
            Arc::new(RwLock::new(make_test_table("t3", 1, 0, 1).unwrap())),
        );
        state
    }

    #[allow(dead_code)]
    pub fn make_test_subscriptions(use_table_name: bool) -> Vec<TableSubscription> {
        if use_table_name {
            vec![
                TableSubscription::OnUpdateLastRecordBatch {
                    table_name: "t1".to_string(),
                },
                TableSubscription::OnUpdateLastRecordBatch {
                    table_name: "t2".to_string(),
                },
                TableSubscription::AlwaysLastRecordBatch {
                    table_name: "t3".to_string(),
                },
            ]
        } else {
            vec![
                TableSubscription::OnUpdateLastRecordBatch {
                    table_name: "t3".to_string(),
                },
                TableSubscription::OnUpdateLastRecordBatch {
                    table_name: "t3".to_string(),
                },
                TableSubscription::AlwaysLastRecordBatch {
                    table_name: "t3".to_string(),
                },
            ]
        }
    }

    #[allow(dead_code)]
    pub fn make_test_updates(is_any: bool) -> HashMap<String, bool> {
        let mut updates = HashMap::<String, bool>::new();
        updates.insert("t1".to_string(), true);
        if is_any {
            updates.insert("t2".to_string(), false);
        } else {
            updates.insert("t2".to_string(), true);
        }
        updates.insert("t3".to_string(), false);
        updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_subscribe() {
        let state = test_subscribe::make_test_state();
        let subscriptions = test_subscribe::make_test_subscriptions(true);
        let updates = test_subscribe::make_test_updates(true);
        let sub = AlwaysSubscribe::new_box();
        assert!(sub.check_subscriptions(&subscriptions, &updates, &state));
    }

    #[test]
    fn test_any_tablename_subscribe() {
        let state = test_subscribe::make_test_state();
        let subscriptions = test_subscribe::make_test_subscriptions(true);
        let updates = test_subscribe::make_test_updates(true);
        let sub = AnyTableNameSubscribe::new_box();
        assert!(sub.check_subscriptions(&subscriptions, &updates, &state));
        let updates = test_subscribe::make_test_updates(false);
        assert!(sub.check_subscriptions(&subscriptions, &updates, &state));
    }

    #[test]
    fn test_all_tablename_subscribe() {
        let state = test_subscribe::make_test_state();
        let subscriptions = test_subscribe::make_test_subscriptions(true);
        let updates = test_subscribe::make_test_updates(true);
        let sub = AllTableNamesSubscribe::new_box();
        assert!(!sub.check_subscriptions(&subscriptions, &updates, &state));
        let updates = test_subscribe::make_test_updates(false);
        assert!(sub.check_subscriptions(&subscriptions, &updates, &state));
    }

    #[test]
    fn test_any_schema_subscribe() {
        let state = test_subscribe::make_test_state();
        let subscriptions = test_subscribe::make_test_subscriptions(false);
        let updates = test_subscribe::make_test_updates(true);
        let sub = AnyTableSchemaSubscribe::new_box();
        assert!(sub.check_subscriptions(&subscriptions, &updates, &state));
        let updates = test_subscribe::make_test_updates(false);
        assert!(sub.check_subscriptions(&subscriptions, &updates, &state));
    }

    #[test]
    fn test_all_schema_subscribe() {
        let state = test_subscribe::make_test_state();
        let subscriptions = test_subscribe::make_test_subscriptions(false);
        let updates = test_subscribe::make_test_updates(true);
        let sub = AllTableSchemasSubscribe::new_box();
        assert!(!sub.check_subscriptions(&subscriptions, &updates, &state));
        let updates = test_subscribe::make_test_updates(false);
        assert!(!sub.check_subscriptions(&subscriptions, &updates, &state));
    }
}
