use anyhow::{Result, anyhow};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::session::common_traits::{MappableTrait, StateMap};

use super::{
    table_trait::{Table, TableTrait},
    stream::SendableRecordBatchStream,
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum TableSubscribe {
    /// Only when the subject has been updated
    OnUpdateFullTable { table_name: String },
    /// Only when the subject has been updated
    /// and just the last RecordBatch
    OnUpdateLastRecordBatch { table_name: String },
    /// Always read the full table
    AlwaysFullTable { table_name: String },
    /// Always read just the last record batch
    AlwaysLastRecordBatch { table_name: String },
    /// No download
    #[default]
    None,
    /// Custom subscription function
    Custom(String),
}

impl TableSubscribe {
    pub fn get_table_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { table_name: tn } => tn,
            Self::OnUpdateLastRecordBatch { table_name: tn } => tn,
            Self::AlwaysFullTable { table_name: tn } => tn,
            Self::AlwaysLastRecordBatch { table_name: tn } => tn,
            Self::None => "",
            Self::Custom(_name) => "",
        }
    }

    #[allow(dead_code)]
    fn get_full_name(&self) -> String {
        match self {
            Self::OnUpdateFullTable { table_name: tn } => format!("OnUpdateFullTable-{tn}"),
            Self::OnUpdateLastRecordBatch { table_name: tn } => {
                format!("OnUpdateLastRecordBatch-{tn}")
            }
            Self::AlwaysFullTable { table_name: tn } => format!("AlwaysFullTable-{tn}"),
            Self::AlwaysLastRecordBatch { table_name: tn } => format!("AlwaysLastRecordBatch-{tn}"),
            Self::None => "None".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    pub fn is_update(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { table_name: _tn }
            | Self::OnUpdateLastRecordBatch { table_name: _tn } => true,
            Self::AlwaysFullTable { table_name: _tn }
            | Self::AlwaysLastRecordBatch { table_name: _tn } => false,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    pub fn get_short_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { table_name: _tn } => "FullTable",
            Self::OnUpdateLastRecordBatch { table_name: _tn } => "LastRecordBatch",
            Self::AlwaysFullTable { table_name: _tn } => "FullTable",
            Self::AlwaysLastRecordBatch { table_name: _tn } => "LastRecordBatch",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }

    pub fn from_str(name: &str, subject: &str) -> Result<TableSubscribe> {
        let subscription = if name.contains("OnUpdateFullTable") {
            TableSubscribe::OnUpdateFullTable {
                table_name: subject.to_string(),
            }
        } else if name.contains("AlwaysFullTable") {
            TableSubscribe::AlwaysFullTable {
                table_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateLastRecordBatch") {
            TableSubscribe::OnUpdateLastRecordBatch {
                table_name: subject.to_string(),
            }
        } else if name.contains("AlwaysLastRecordBatch") {
            TableSubscribe::AlwaysLastRecordBatch {
                table_name: subject.to_string(),
            }
        } else if name.contains("None") {
            TableSubscribe::None {}
        } else {
            return Err(anyhow!(
                "Variant for ArrowTableSubscribe {name} with subject {subject} was not recognized."
            ));
        };
        Ok(subscription)
    }
}

impl MappableTrait for TableSubscribe {
    fn get_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { table_name: _tn } => "OnUpdateFullTable",
            Self::OnUpdateLastRecordBatch { table_name: _tn } => "OnUpdateLastRecordBatch",
            Self::AlwaysFullTable { table_name: _tn } => "AlwaysFullTable",
            Self::AlwaysLastRecordBatch { table_name: _tn } => "AlwaysLastRecordBatch",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }
}

/// Subscribe to an arrow table
pub trait TableSubscribeTrait: TableTrait {
    /// Implement the subscription
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_table(
        &self,
        subscribe: &TableSubscribe,
        updated: bool,
    ) -> Option<SendableRecordBatchStream>;
}

impl TableSubscribeTrait for Table {
    fn subscribe_table(
        &self,
        subscribe: &TableSubscribe,
        updated: bool,
    ) -> Option<SendableRecordBatchStream> {
        match subscribe {
            TableSubscribe::AlwaysFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            TableSubscribe::AlwaysLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            TableSubscribe::OnUpdateFullTable { table_name: _ } => {
                if updated {
                    Some(self.to_record_batch_stream())
                } else {
                    None
                }
            }
            TableSubscribe::OnUpdateLastRecordBatch { table_name: _ } => {
                if updated {
                    Some(self.to_record_batch_stream_last_record_batch())
                } else {
                    None
                }
            }
            TableSubscribe::None => None,
            TableSubscribe::Custom(_) => None,
        }
    }
}

/// Helper function to convert a [String] to a [SubscribeTrait]
///
/// # Notes
/// * This method will eventually be on an enum of all concrete
///   [SubscribeTrait] implementations
/// * Comparison by `contains` can be dangerous so order matters
pub fn from_str_to_subscribe(line: &str) -> Result<Box<dyn SubscribeTrait>> {
    let subscribe = if line.contains(AllTableSchemasSubscribe::get_static_name()) {
        AllTableSchemasSubscribe::new_box()
    } else if line.contains(AnyTableSchemaSubscribe::get_static_name()) {
        AnyTableSchemaSubscribe::new_box()
    } else if line.contains(AllTableNamesSubscribe::get_static_name()) {
        AllTableNamesSubscribe::new_box()
    } else if line.contains(AnyTableNameSubscribe::get_static_name()) {
        AnyTableNameSubscribe::new_box()
    } else if line.contains(AlwaysSubscribe::get_static_name()) {
        AlwaysSubscribe::new_box()
    } else if line.contains(ChatContentSubscribe::get_static_name()) {
        ChatContentSubscribe::new_box()
    } else {
        return Err(anyhow!("Subscribe policy {line} was not recognized."));
    };
    Ok(subscribe)
}

/// Determine when all subscriptions are ready
pub trait SubscribeTrait: MappableTrait + Debug + Send + Sync {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscribe],
        updates: &HashMap<String, bool>,
        state: &StateMap,
    ) -> bool;
    fn new_box() -> Box<dyn SubscribeTrait>
    where
        Self: Sized;
}

/// Always subscribe (dummy subscription check for testing)
#[derive(Default, Debug)]
pub struct AlwaysSubscribe;

impl SubscribeTrait for AlwaysSubscribe {
    fn check_subscriptions(
        &self,
        _subscriptions: &[TableSubscribe],
        _updates: &HashMap<String, bool>,
        _state: &StateMap,
    ) -> bool {
        true
    }
    fn new_box() -> Box<dyn SubscribeTrait>
    where
        Self: Sized,
    {
        Box::new(Self)
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
#[derive(Default, Debug)]
pub struct AnyTableNameSubscribe;

impl SubscribeTrait for AnyTableNameSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscribe],
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
    fn new_box() -> Box<dyn SubscribeTrait> {
        Box::new(Self {})
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
#[derive(Default, Debug)]
pub struct AllTableNamesSubscribe;

impl SubscribeTrait for AllTableNamesSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscribe],
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
    fn new_box() -> Box<dyn SubscribeTrait> {
        Box::new(Self {})
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
#[derive(Default, Debug)]
pub struct AnyTableSchemaSubscribe;

impl SubscribeTrait for AnyTableSchemaSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscribe],
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
    fn new_box() -> Box<dyn SubscribeTrait> {
        Box::new(Self {})
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
#[derive(Default, Debug)]
pub struct AllTableSchemasSubscribe;

impl SubscribeTrait for AllTableSchemasSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[TableSubscribe],
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
    fn new_box() -> Box<dyn SubscribeTrait> {
        Box::new(Self {})
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
#[derive(Default, Debug)]
pub struct ChatContentSubscribe {
    user_message_table_name: String,
    tool_message_table_name: String,
}

impl ChatContentSubscribe {
    pub fn new_box_with_table_names(user_message_table_name: &str, tool_message_table_name: &str) -> Box<dyn SubscribeTrait> {
        Box::new(Self {
            user_message_table_name: user_message_table_name.to_string(),
            tool_message_table_name: tool_message_table_name.to_string(),
        })
    }
}

impl SubscribeTrait for ChatContentSubscribe {
    fn check_subscriptions(
        &self,
        _subscriptions: &[TableSubscribe],
        updates: &HashMap<String, bool>,
        _state: &StateMap,
    ) -> bool {
        let user = updates.get(&self.user_message_table_name).unwrap_or(&false);
        let tool = updates.get(&self.tool_message_table_name).unwrap_or(&false);
        *tool || *user
    }
    fn new_box() -> Box<dyn SubscribeTrait> {
        Box::new(Self {
            // DM: dangerous as the strings needs to stay syncronized with the actual table names
            // in `AvailableinterfaceSubjects` and `AvailableinterfaceSubjects`
            user_message_table_name: "UserMessages".to_string(),
            tool_message_table_name: "ToolMessages".to_string(),
        })
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
    pub fn make_test_subscriptions(use_table_name: bool) -> Vec<TableSubscribe> {
        if use_table_name {
            vec![
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t1".to_string(),
                },
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t2".to_string(),
                },
                TableSubscribe::AlwaysLastRecordBatch {
                    table_name: "t3".to_string(),
                },
            ]
        } else {
            vec![
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t3".to_string(),
                },
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t3".to_string(),
                },
                TableSubscribe::AlwaysLastRecordBatch {
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
