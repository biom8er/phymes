use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::session::MappableTrait;

use super::{
    stream::SendableRecordBatchStream,
    table_trait::{Table, TableTrait},
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum TableSubscription {
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

impl TableSubscription {
    /// The `table_name` of the variant
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
    /// Full name for the [TableSubscription] that includes the `table_name` and other information
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

    /// Short name for the [TableSubscription] that omits the `table_name` and other information
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

    /// New [TableSubscription] from a short name identifying the variant and the `table_name`
    pub fn from_str_fuzzy(name: &str, subject: &str) -> Result<TableSubscription> {
        let subscription = if name.contains("OnUpdateFullTable") {
            TableSubscription::OnUpdateFullTable {
                table_name: subject.to_string(),
            }
        } else if name.contains("AlwaysFullTable") {
            TableSubscription::AlwaysFullTable {
                table_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateLastRecordBatch") {
            TableSubscription::OnUpdateLastRecordBatch {
                table_name: subject.to_string(),
            }
        } else if name.contains("AlwaysLastRecordBatch") {
            TableSubscription::AlwaysLastRecordBatch {
                table_name: subject.to_string(),
            }
        } else if name.contains("None") {
            TableSubscription::None {}
        } else {
            return Err(anyhow!(
                "Variant for ArrowTableSubscribe {name} with subject {subject} was not recognized."
            ));
        };
        Ok(subscription)
    }

    /// New [TableSubscription] from a short name identifying the variant, the subject `table_name`
    ///   and the mermaid.js flowchart diagram link type
    pub fn from_str_mermaid(line: &str, subject: &str) -> Result<TableSubscription> {
        if line.contains("|") & line.contains("-.->") & line.contains("FullTable") {
            Ok(TableSubscription::OnUpdateFullTable {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("FullTable") {
            Ok(TableSubscription::AlwaysFullTable {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("LastRecordBatch")
        {
            Ok(TableSubscription::OnUpdateLastRecordBatch {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("LastRecordBatch")
        {
            Ok(TableSubscription::AlwaysLastRecordBatch {
                table_name: subject.to_string(),
            })
        } else if line.contains("None") {
            Ok(TableSubscription::None {})
        } else {
            Err(anyhow!(
                "Variant for TablePublication with subject {subject} was not recognized in string slice {line}."
            ))
        }
    }
}

impl MappableTrait for TableSubscription {
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
pub trait TableSubscriptionTrait: TableTrait {
    /// Implement the subscription
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_table(
        &self,
        subscribe: &TableSubscription,
        updated: bool,
    ) -> Option<SendableRecordBatchStream>;
}

impl TableSubscriptionTrait for Table {
    fn subscribe_table(
        &self,
        subscribe: &TableSubscription,
        updated: bool,
    ) -> Option<SendableRecordBatchStream> {
        // Skip tables for which there are no rows
        if self.count_rows() == 0 {
            return None;
        }

        // Match on the subscribe policy
        match subscribe {
            TableSubscription::AlwaysFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            TableSubscription::AlwaysLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            TableSubscription::OnUpdateFullTable { table_name: _ } => {
                if updated {
                    Some(self.to_record_batch_stream())
                } else {
                    None
                }
            }
            TableSubscription::OnUpdateLastRecordBatch { table_name: _ } => {
                if updated {
                    Some(self.to_record_batch_stream_last_record_batch())
                } else {
                    None
                }
            }
            TableSubscription::None => None,
            TableSubscription::Custom(_) => None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_subscription_from_str_mermaid() -> Result<()> {
        let line = "message_parsing-subject-->|FullTable|message_parser-subscribe";
        let subject = "message_parser";
        let publication = TableSubscription::AlwaysFullTable { table_name: subject.to_string() };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|FullTable|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateFullTable { table_name: subject.to_string() };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|LastRecordBatch|message_parser-subscribe";
        let publication = TableSubscription::AlwaysLastRecordBatch { table_name: subject.to_string() };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|LastRecordBatch|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateLastRecordBatch { table_name: subject.to_string() };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        Ok(())
    }
}