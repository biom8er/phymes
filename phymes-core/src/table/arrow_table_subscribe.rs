use serde::{Deserialize, Serialize};

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
