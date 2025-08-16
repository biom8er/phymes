use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{metrics::HashMap, session::common_traits::{MappableTrait, StateMap}};

use super::{
    arrow_table::{ArrowTable, ArrowTableTrait},
    stream::SendableRecordBatchStream,
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum ArrowTableSubscribe {
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

impl ArrowTableSubscribe {
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
    fn get_full_name(&self) -> String {        
        match self {
            Self::OnUpdateFullTable { table_name: tn } => format!("OnUpdateFullTable-{tn}"),
            Self::OnUpdateLastRecordBatch { table_name: tn } => format!("OnUpdateLastRecordBatch-{tn}"),
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
}

impl MappableTrait for ArrowTableSubscribe {
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
pub trait ArrowTableSubscribeTrait: ArrowTableTrait {
    /// Implement the subscription
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_table(
        &self,
        subscribe: &ArrowTableSubscribe,
        updated: bool,
    ) -> Option<SendableRecordBatchStream>;
}

impl ArrowTableSubscribeTrait for ArrowTable {
    fn subscribe_table(
        &self,
        subscribe: &ArrowTableSubscribe,
        updated: bool,
    ) -> Option<SendableRecordBatchStream> {
        match subscribe {
            ArrowTableSubscribe::AlwaysFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            ArrowTableSubscribe::AlwaysLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            ArrowTableSubscribe::OnUpdateFullTable { table_name: _ } => {
                if updated {
                    Some(self.to_record_batch_stream())
                } else {
                    None
                }
            }
            ArrowTableSubscribe::OnUpdateLastRecordBatch { table_name: _ } => {
                if updated {
                    Some(self.to_record_batch_stream_last_record_batch())
                } else {
                    None
                }
            }
            ArrowTableSubscribe::None => None,
            ArrowTableSubscribe::Custom(_) => None,
        }
    }
}

/// Determine when all subscriptions are ready
pub trait SubscribeTrait: MappableTrait + Debug + Send + Sync {
    fn check_subscriptions(
        &self,
        subscriptions: &[ArrowTableSubscribe],
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
        _subscriptions: &[ArrowTableSubscribe],
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
        subscriptions: &[ArrowTableSubscribe],
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
        subscriptions: &[ArrowTableSubscribe],
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
        subscriptions: &[ArrowTableSubscribe],
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
        subscriptions: &[ArrowTableSubscribe],
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

mod test_subscribe {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::table::arrow_table::test_table::make_test_table;

    use super::*;

    #[allow(dead_code)]
    pub fn make_test_state() -> StateMap {
        let mut state = HashMap::<String, Arc<RwLock<ArrowTable>>>::new();
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
    pub fn make_test_subscriptions(use_table_name: bool) -> Vec<ArrowTableSubscribe> {
        if use_table_name {
            vec![
                ArrowTableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t1".to_string(),
                },
                ArrowTableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t2".to_string(),
                },
                ArrowTableSubscribe::AlwaysLastRecordBatch {
                    table_name: "t3".to_string(),
                },
            ]
        } else {
            vec![
                ArrowTableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t3".to_string(),
                },
                ArrowTableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "t3".to_string(),
                },
                ArrowTableSubscribe::AlwaysLastRecordBatch {
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
