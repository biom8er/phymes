use anyhow::{Result, anyhow};
use arrow::datatypes::Schema;
use phymes_diagnostics::{TraceableTrait, Tracer};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

use crate::{RecordBatchStreamAdapter, runtime_env::MappableTrait};

use super::{
    stream::SendableRecordBatchStream,
    table_trait::{Table, TableTrait},
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum TableSubscription {
    /// Only when the subject has been updated, copy the full table
    OnUpdateFullTable { table_name: String },
    /// Only when the subject has been updated, and just a copy of the last RecordBatch
    OnUpdateLastRecordBatch { table_name: String },
    /// Only when the subject has been updated, but don't copy or take any data
    ///   which is useful for ensuring a task is triggered after another task
    OnUpdateEmpty { table_name: String },
    /// Always copy the full table
    AlwaysFullTable { table_name: String },
    /// Always copy just the last record batch
    AlwaysLastRecordBatch { table_name: String },
    /// Only when the subject has been updated, drain the full table
    OnUpdateFullTableDrain { table_name: String },
    /// Only when the subject has been updated, and just pop the last RecordBatch
    OnUpdateLastRecordBatchPop { table_name: String },
    /// Always drain the full table
    AlwaysFullTableDrain { table_name: String },
    /// Always pop just the last record batch
    AlwaysLastRecordBatchPop { table_name: String },
    /// No reading of the table
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
            Self::OnUpdateEmpty { table_name: tn } => tn,
            Self::AlwaysFullTable { table_name: tn } => tn,
            Self::AlwaysLastRecordBatch { table_name: tn } => tn,
            Self::OnUpdateFullTableDrain { table_name: tn } => tn,
            Self::OnUpdateLastRecordBatchPop { table_name: tn } => tn,
            Self::AlwaysFullTableDrain { table_name: tn } => tn,
            Self::AlwaysLastRecordBatchPop { table_name: tn } => tn,
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
            Self::OnUpdateEmpty { table_name: tn } => format!("OnUpdateEmpty-{tn}"),
            Self::AlwaysFullTable { table_name: tn } => format!("AlwaysFullTable-{tn}"),
            Self::AlwaysLastRecordBatch { table_name: tn } => format!("AlwaysLastRecordBatch-{tn}"),
            Self::OnUpdateFullTableDrain { table_name: tn } => format!("OnUpdateFullTableDrain-{tn}"),
            Self::OnUpdateLastRecordBatchPop { table_name: tn } => {
                format!("OnUpdateLastRecordBatchPop-{tn}")
            }
            Self::AlwaysFullTableDrain { table_name: tn } => format!("AlwaysFullTableDrain-{tn}"),
            Self::AlwaysLastRecordBatchPop { table_name: tn } => {
                format!("AlwaysLastRecordBatchPop-{tn}")
            }
            Self::None => "None".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    /// Is the subscription triggered by a table update?
    pub fn is_update(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { table_name: _tn }
            | Self::OnUpdateLastRecordBatch { table_name: _tn }
            | Self::OnUpdateFullTableDrain { table_name: _tn }
            | Self::OnUpdateLastRecordBatchPop { table_name: _tn }
            | Self::OnUpdateEmpty { table_name: _tn } => true,
            Self::AlwaysFullTable { table_name: _tn }
            | Self::AlwaysLastRecordBatch { table_name: _tn }
            | Self::AlwaysFullTableDrain { table_name: _tn }
            | Self::AlwaysLastRecordBatchPop { table_name: _tn } => false,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Does the subscription result in a clone of the table?
    pub fn is_clone(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { table_name: _tn }
            | Self::OnUpdateLastRecordBatch { table_name: _tn }
            | Self::OnUpdateEmpty { table_name: _tn }
            | Self::AlwaysFullTable { table_name: _tn }
            | Self::AlwaysLastRecordBatch { table_name: _tn } => true,
            Self::OnUpdateFullTableDrain { table_name: _tn }
            | Self::OnUpdateLastRecordBatchPop { table_name: _tn }
            | Self::AlwaysFullTableDrain { table_name: _tn }
            | Self::AlwaysLastRecordBatchPop { table_name: _tn } => false,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Does the subscription result in mutating the table?
    pub fn is_mut(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { table_name: _tn }
            | Self::OnUpdateLastRecordBatch { table_name: _tn }
            | Self::OnUpdateEmpty { table_name: _tn }
            | Self::AlwaysFullTable { table_name: _tn }
            | Self::AlwaysLastRecordBatch { table_name: _tn } => false,
            Self::OnUpdateFullTableDrain { table_name: _tn }
            | Self::OnUpdateLastRecordBatchPop { table_name: _tn }
            | Self::AlwaysFullTableDrain { table_name: _tn }
            | Self::AlwaysLastRecordBatchPop { table_name: _tn } => true,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Short name for the [TableSubscription] that omits the `table_name` and other information
    pub fn get_short_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { table_name: _tn } => "FullTable",
            Self::OnUpdateLastRecordBatch { table_name: _tn } => "LastRecordBatch",
            Self::OnUpdateEmpty { table_name: _tn } => "Empty",
            Self::AlwaysFullTable { table_name: _tn } => "FullTable",
            Self::AlwaysLastRecordBatch { table_name: _tn } => "LastRecordBatch",
            Self::OnUpdateFullTableDrain { table_name: _tn } => "FullTableDrain",
            Self::OnUpdateLastRecordBatchPop { table_name: _tn } => "LastRecordBatchPop",
            Self::AlwaysFullTableDrain { table_name: _tn } => "FullTableDrain",
            Self::AlwaysLastRecordBatchPop { table_name: _tn } => "LastRecordBatchPop",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }

    /// New [TableSubscription] from a short name identifying the variant and the `table_name`
    pub fn from_str_fuzzy(name: &str, subject: &str) -> Result<TableSubscription> {
        let subscription = if name.contains("OnUpdateFullTableDrain") {
            TableSubscription::OnUpdateFullTableDrain {
                table_name: subject.to_string(),
            }
        } else if name.contains("AlwaysFullTableDrain") {
            TableSubscription::AlwaysFullTableDrain {
                table_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateLastRecordBatchPop") {
            TableSubscription::OnUpdateLastRecordBatchPop {
                table_name: subject.to_string(),
            }
        } else if name.contains("AlwaysLastRecordBatchPop") {
            TableSubscription::AlwaysLastRecordBatchPop {
                table_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateFullTable") {
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
        } else if name.contains("OnUpdateEmpty") {
            TableSubscription::OnUpdateEmpty {
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
        if line.contains("|") & line.contains("-.->") & line.contains("FullTableDrain") {
            Ok(TableSubscription::OnUpdateFullTableDrain {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("FullTableDrain") {
            Ok(TableSubscription::AlwaysFullTableDrain {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("LastRecordBatchPop") {
            Ok(TableSubscription::OnUpdateLastRecordBatchPop {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("LastRecordBatchPop") {
            Ok(TableSubscription::AlwaysLastRecordBatchPop {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("FullTable") {
            Ok(TableSubscription::OnUpdateFullTable {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("FullTable") {
            Ok(TableSubscription::AlwaysFullTable {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("LastRecordBatch") {
            Ok(TableSubscription::OnUpdateLastRecordBatch {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("LastRecordBatch") {
            Ok(TableSubscription::AlwaysLastRecordBatch {
                table_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("Empty") {
            Ok(TableSubscription::OnUpdateEmpty {
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
            Self::OnUpdateEmpty { table_name: _tn } => "OnUpdateEmpty",
            Self::AlwaysFullTable { table_name: _tn } => "AlwaysFullTable",
            Self::AlwaysLastRecordBatch { table_name: _tn } => "AlwaysLastRecordBatch",
            Self::OnUpdateFullTableDrain { table_name: _tn } => "OnUpdateFullTableDrain",
            Self::OnUpdateLastRecordBatchPop { table_name: _tn } => "OnUpdateLastRecordBatchPop",
            Self::AlwaysFullTableDrain { table_name: _tn } => "AlwaysFullTableDrain",
            Self::AlwaysLastRecordBatchPop { table_name: _tn } => "AlwaysLastRecordBatchPop",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }
}

impl TraceableTrait for TableSubscription {
    fn to_trace(&self) -> Tracer {
        Tracer::new(self.get_short_name(), self.get_table_name())
    }
}

/// Subscribe to an arrow table
pub trait TableSubscriptionTrait: TableTrait {
    /// Implement the subscription
    ///
    /// # Notes
    ///
    /// * Empty tables are skipped
    /// * `TableSubscription` where `is_clone` = false are skipped
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_to_table(
        &self,
        subscribe: &TableSubscription,
    ) -> Option<SendableRecordBatchStream>;

    /// Implement the subscription mutating the table
    ///
    /// # Notes
    ///
    /// * Empty tables are skipped
    /// * `TableSubscription` where `is_mut` = false are skipped
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_to_table_mut(
        &mut self,
        subscribe: &TableSubscription,
    ) -> Option<SendableRecordBatchStream>;
}

impl TableSubscriptionTrait for Table {
    fn subscribe_to_table(
        &self,
        subscribe: &TableSubscription,
    ) -> Option<SendableRecordBatchStream> {
        if self.count_rows() == 0 {
            return None;
        }
        match subscribe {
            TableSubscription::AlwaysFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            TableSubscription::AlwaysLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            TableSubscription::OnUpdateFullTable { table_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            TableSubscription::OnUpdateLastRecordBatch { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            TableSubscription::OnUpdateEmpty { table_name: _ } => {
                let schema = Schema::empty();
                let stream = futures::stream::iter(Vec::new().into_iter().map(Ok));
                Some(Box::pin(RecordBatchStreamAdapter::new(
                    Arc::new(schema),
                    stream,
                )))
            }
            TableSubscription::AlwaysFullTableDrain { table_name: _ } => None,
            TableSubscription::AlwaysLastRecordBatchPop { table_name: _ } => None,
            TableSubscription::OnUpdateFullTableDrain { table_name: _ } => None,
            TableSubscription::OnUpdateLastRecordBatchPop { table_name: _ } => None,
            TableSubscription::None => None,
            TableSubscription::Custom(_) => None,
        }
    }
    fn subscribe_to_table_mut(
        &mut self,
        subscribe: &TableSubscription,
    ) -> Option<SendableRecordBatchStream> {
        if self.count_rows() == 0 {
            return None;
        }
        match subscribe {
            TableSubscription::AlwaysFullTableDrain { table_name: _ } => {
                Some(self.to_record_batch_stream_drain())
            }
            TableSubscription::AlwaysLastRecordBatchPop { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch_pop())
            }
            TableSubscription::OnUpdateFullTableDrain { table_name: _ } => {
                Some(self.to_record_batch_stream_drain())
            }
            TableSubscription::OnUpdateLastRecordBatchPop { table_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch_pop())
            }
            TableSubscription::OnUpdateEmpty { table_name: _ } => None,
            TableSubscription::AlwaysFullTable { table_name: _ } => None,
            TableSubscription::AlwaysLastRecordBatch { table_name: _ } => None,
            TableSubscription::OnUpdateFullTable { table_name: _ } => None,
            TableSubscription::OnUpdateLastRecordBatch { table_name: _ } => None,
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
        let publication = TableSubscription::AlwaysFullTable {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|FullTable|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateFullTable {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|LastRecordBatch|message_parser-subscribe";
        let publication = TableSubscription::AlwaysLastRecordBatch {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|LastRecordBatch|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateLastRecordBatch {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|Empty|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateEmpty {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|FullTableDrain|message_parser-subscribe";
        let subject = "message_parser";
        let publication = TableSubscription::AlwaysFullTableDrain {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|FullTableDrain|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateFullTableDrain {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|LastRecordBatchPop|message_parser-subscribe";
        let publication = TableSubscription::AlwaysLastRecordBatchPop {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|LastRecordBatchPop|message_parser-subscribe";
        let publication = TableSubscription::OnUpdateLastRecordBatchPop {
            table_name: subject.to_string(),
        };
        let test = TableSubscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        Ok(())
    }
}
