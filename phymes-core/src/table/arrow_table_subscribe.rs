use serde::{Deserialize, Serialize};

use crate::{metrics::HashMap, session::common_traits::StateMap};

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
}

/// Subscribe to an arrow table
pub trait ArrowTableSubscribeTrait: ArrowTableTrait {
    fn subscribe_table(&self, subscribe: &ArrowTableSubscribe)
    -> Option<SendableRecordBatchStream>;
}

impl ArrowTableSubscribeTrait for ArrowTable {
    fn subscribe_table(
        &self,
        subscribe: &ArrowTableSubscribe,
    ) -> Option<SendableRecordBatchStream> {
        match subscribe {
            ArrowTableSubscribe::AlwaysFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            ArrowTableSubscribe::AlwaysLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            ArrowTableSubscribe::OnUpdateFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            ArrowTableSubscribe::OnUpdateLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            ArrowTableSubscribe::None => None,
            ArrowTableSubscribe::Custom(_) => None,
        }
    }
}

/// Determine when all subscriptions are ready
pub trait SubscribeTrait {
    fn check_subscriptions(&self, subscriptions: &[ArrowTableSubscribe], updates: &HashMap<String, bool>, state: &StateMap) -> bool;
}

/// Subscribe when any matching table name has been updated
pub struct AnyTableNameSubscribe;

impl SubscribeTrait for AnyTableNameSubscribe {
    fn check_subscriptions(&self, subscriptions: &[ArrowTableSubscribe], updates: &HashMap<String, bool>, _state: &StateMap) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() & updates.get(subscription.get_table_name()).unwrap_or(&false) {
                return true;
            }
        }
        false
    }
}

/// Subscribe when all matching table names has been updated
pub struct AllTableNamesSubscribe;

impl SubscribeTrait for AllTableNamesSubscribe {
    fn check_subscriptions(&self, subscriptions: &[ArrowTableSubscribe], updates: &HashMap<String, bool>, _state: &StateMap) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() & !updates.get(subscription.get_table_name()).unwrap_or(&true) {
                return false;
            }
        }
        true
    }
}

/// Subscribe when any matching table schema has been updated
pub struct AnyTableSchemaSubscribe;

impl SubscribeTrait for AnyTableSchemaSubscribe {
    fn check_subscriptions(&self, subscriptions: &[ArrowTableSubscribe], updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                for (table_name, update) in updates {
                    if state.get(subscription.get_table_name()).unwrap().try_read().unwrap().get_schema().eq(
                        &state.get(table_name).unwrap().try_read().unwrap().get_schema())
                        & *update {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Subscribe when all matching table schemas has been updated
pub struct AllTableSchemasSubscribe;

impl SubscribeTrait for AllTableSchemasSubscribe {
    fn check_subscriptions(&self, subscriptions: &[ArrowTableSubscribe], updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                for (table_name, update) in updates {
                    if state.get(subscription.get_table_name()).unwrap().try_read().unwrap().get_schema().eq(
                        &state.get(table_name).unwrap().try_read().unwrap().get_schema())
                        & !*update {
                        return false;
                    }
                }
            }
        }
        true
    }
}