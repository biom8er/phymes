use anyhow::{Result, anyhow};
use arrow::datatypes::Schema;
use phymes_diagnostics::{TraceableTrait, Tracer};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

use crate::{RecordBatchStreamAdapter, MappableTrait, SendableRecordBatchStream, Subject, SubjectTrait};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum Subscription {
    /// Only when the subject has been updated, copy the full table
    OnUpdateFullTable { subject_name: String },
    /// Only when the subject has been updated, and just a copy of the last RecordBatch
    OnUpdateLastRecordBatch { subject_name: String },
    /// Only when the subject has been updated, but don't copy or take any data
    ///   which is useful for ensuring a task is triggered after another task
    OnUpdateEmpty { subject_name: String },
    /// Always copy the full table
    AlwaysFullTable { subject_name: String },
    /// Always copy just the last record batch
    AlwaysLastRecordBatch { subject_name: String },
    /// Only when the subject has been updated, drain the full table
    OnUpdateFullTableDrain { subject_name: String },
    /// Only when the subject has been updated, and just pop the last RecordBatch
    OnUpdateLastRecordBatchPop { subject_name: String },
    /// Always drain the full table
    AlwaysFullTableDrain { subject_name: String },
    /// Always pop just the last record batch
    AlwaysLastRecordBatchPop { subject_name: String },
    /// No reading of the table
    #[default]
    None,
    /// Custom subscription function
    Custom(String),
}

impl Subscription {
    /// The `subject_name` of the variant
    pub fn get_table_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { subject_name: tn } => tn,
            Self::OnUpdateLastRecordBatch { subject_name: tn } => tn,
            Self::OnUpdateEmpty { subject_name: tn } => tn,
            Self::AlwaysFullTable { subject_name: tn } => tn,
            Self::AlwaysLastRecordBatch { subject_name: tn } => tn,
            Self::OnUpdateFullTableDrain { subject_name: tn } => tn,
            Self::OnUpdateLastRecordBatchPop { subject_name: tn } => tn,
            Self::AlwaysFullTableDrain { subject_name: tn } => tn,
            Self::AlwaysLastRecordBatchPop { subject_name: tn } => tn,
            Self::None => "",
            Self::Custom(_name) => "",
        }
    }

    #[allow(dead_code)]
    /// Full name for the [Subscription] that includes the `subject_name` and other information
    fn get_full_name(&self) -> String {
        match self {
            Self::OnUpdateFullTable { subject_name: tn } => format!("OnUpdateFullTable-{tn}"),
            Self::OnUpdateLastRecordBatch { subject_name: tn } => {
                format!("OnUpdateLastRecordBatch-{tn}")
            }
            Self::OnUpdateEmpty { subject_name: tn } => format!("OnUpdateEmpty-{tn}"),
            Self::AlwaysFullTable { subject_name: tn } => format!("AlwaysFullTable-{tn}"),
            Self::AlwaysLastRecordBatch { subject_name: tn } => format!("AlwaysLastRecordBatch-{tn}"),
            Self::OnUpdateFullTableDrain { subject_name: tn } => {
                format!("OnUpdateFullTableDrain-{tn}")
            }
            Self::OnUpdateLastRecordBatchPop { subject_name: tn } => {
                format!("OnUpdateLastRecordBatchPop-{tn}")
            }
            Self::AlwaysFullTableDrain { subject_name: tn } => format!("AlwaysFullTableDrain-{tn}"),
            Self::AlwaysLastRecordBatchPop { subject_name: tn } => {
                format!("AlwaysLastRecordBatchPop-{tn}")
            }
            Self::None => "None".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    /// Is the subscription triggered by a table update?
    pub fn is_update(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { subject_name: _tn }
            | Self::OnUpdateLastRecordBatch { subject_name: _tn }
            | Self::OnUpdateFullTableDrain { subject_name: _tn }
            | Self::OnUpdateLastRecordBatchPop { subject_name: _tn }
            | Self::OnUpdateEmpty { subject_name: _tn } => true,
            Self::AlwaysFullTable { subject_name: _tn }
            | Self::AlwaysLastRecordBatch { subject_name: _tn }
            | Self::AlwaysFullTableDrain { subject_name: _tn }
            | Self::AlwaysLastRecordBatchPop { subject_name: _tn } => false,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Does the subscription result in a clone of the table?
    pub fn is_clone(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { subject_name: _tn }
            | Self::OnUpdateLastRecordBatch { subject_name: _tn }
            | Self::OnUpdateEmpty { subject_name: _tn }
            | Self::AlwaysFullTable { subject_name: _tn }
            | Self::AlwaysLastRecordBatch { subject_name: _tn } => true,
            Self::OnUpdateFullTableDrain { subject_name: _tn }
            | Self::OnUpdateLastRecordBatchPop { subject_name: _tn }
            | Self::AlwaysFullTableDrain { subject_name: _tn }
            | Self::AlwaysLastRecordBatchPop { subject_name: _tn } => false,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Does the subscription result in mutating the table?
    pub fn is_mut(&self) -> bool {
        match self {
            Self::OnUpdateFullTable { subject_name: _tn }
            | Self::OnUpdateLastRecordBatch { subject_name: _tn }
            | Self::OnUpdateEmpty { subject_name: _tn }
            | Self::AlwaysFullTable { subject_name: _tn }
            | Self::AlwaysLastRecordBatch { subject_name: _tn } => false,
            Self::OnUpdateFullTableDrain { subject_name: _tn }
            | Self::OnUpdateLastRecordBatchPop { subject_name: _tn }
            | Self::AlwaysFullTableDrain { subject_name: _tn }
            | Self::AlwaysLastRecordBatchPop { subject_name: _tn } => true,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Short name for the [Subscription] that omits the `subject_name` and other information
    pub fn get_short_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { subject_name: _tn } => "FullTable",
            Self::OnUpdateLastRecordBatch { subject_name: _tn } => "LastRecordBatch",
            Self::OnUpdateEmpty { subject_name: _tn } => "Empty",
            Self::AlwaysFullTable { subject_name: _tn } => "FullTable",
            Self::AlwaysLastRecordBatch { subject_name: _tn } => "LastRecordBatch",
            Self::OnUpdateFullTableDrain { subject_name: _tn } => "FullTableDrain",
            Self::OnUpdateLastRecordBatchPop { subject_name: _tn } => "LastRecordBatchPop",
            Self::AlwaysFullTableDrain { subject_name: _tn } => "FullTableDrain",
            Self::AlwaysLastRecordBatchPop { subject_name: _tn } => "LastRecordBatchPop",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }

    /// New [Subscription] from a short name identifying the variant and the `subject_name`
    pub fn from_str_fuzzy(name: &str, subject: &str) -> Result<Subscription> {
        let subscription = if name.contains("OnUpdateFullTableDrain") {
            Subscription::OnUpdateFullTableDrain {
                subject_name: subject.to_string(),
            }
        } else if name.contains("AlwaysFullTableDrain") {
            Subscription::AlwaysFullTableDrain {
                subject_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateLastRecordBatchPop") {
            Subscription::OnUpdateLastRecordBatchPop {
                subject_name: subject.to_string(),
            }
        } else if name.contains("AlwaysLastRecordBatchPop") {
            Subscription::AlwaysLastRecordBatchPop {
                subject_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateFullTable") {
            Subscription::OnUpdateFullTable {
                subject_name: subject.to_string(),
            }
        } else if name.contains("AlwaysFullTable") {
            Subscription::AlwaysFullTable {
                subject_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateLastRecordBatch") {
            Subscription::OnUpdateLastRecordBatch {
                subject_name: subject.to_string(),
            }
        } else if name.contains("AlwaysLastRecordBatch") {
            Subscription::AlwaysLastRecordBatch {
                subject_name: subject.to_string(),
            }
        } else if name.contains("OnUpdateEmpty") {
            Subscription::OnUpdateEmpty {
                subject_name: subject.to_string(),
            }
        } else if name.contains("None") {
            Subscription::None {}
        } else {
            return Err(anyhow!(
                "Variant for ArrowTableSubscribe {name} with subject {subject} was not recognized."
            ));
        };
        Ok(subscription)
    }

    /// New [Subscription] from a short name identifying the variant, the subject `subject_name`
    ///   and the mermaid.js flowchart diagram link type
    pub fn from_str_mermaid(line: &str, subject: &str) -> Result<Subscription> {
        if line.contains("|") & line.contains("-.->") & line.contains("FullTableDrain") {
            Ok(Subscription::OnUpdateFullTableDrain {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("FullTableDrain") {
            Ok(Subscription::AlwaysFullTableDrain {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("LastRecordBatchPop") {
            Ok(Subscription::OnUpdateLastRecordBatchPop {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("LastRecordBatchPop") {
            Ok(Subscription::AlwaysLastRecordBatchPop {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("FullTable") {
            Ok(Subscription::OnUpdateFullTable {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("FullTable") {
            Ok(Subscription::AlwaysFullTable {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("LastRecordBatch") {
            Ok(Subscription::OnUpdateLastRecordBatch {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("LastRecordBatch") {
            Ok(Subscription::AlwaysLastRecordBatch {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-.->") & line.contains("Empty") {
            Ok(Subscription::OnUpdateEmpty {
                subject_name: subject.to_string(),
            })
        } else if line.contains("None") {
            Ok(Subscription::None {})
        } else {
            Err(anyhow!(
                "Variant for Publication with subject {subject} was not recognized in string slice {line}."
            ))
        }
    }
}

impl MappableTrait for Subscription {
    fn get_name(&self) -> &str {
        match self {
            Self::OnUpdateFullTable { subject_name: _tn } => "OnUpdateFullTable",
            Self::OnUpdateLastRecordBatch { subject_name: _tn } => "OnUpdateLastRecordBatch",
            Self::OnUpdateEmpty { subject_name: _tn } => "OnUpdateEmpty",
            Self::AlwaysFullTable { subject_name: _tn } => "AlwaysFullTable",
            Self::AlwaysLastRecordBatch { subject_name: _tn } => "AlwaysLastRecordBatch",
            Self::OnUpdateFullTableDrain { subject_name: _tn } => "OnUpdateFullTableDrain",
            Self::OnUpdateLastRecordBatchPop { subject_name: _tn } => "OnUpdateLastRecordBatchPop",
            Self::AlwaysFullTableDrain { subject_name: _tn } => "AlwaysFullTableDrain",
            Self::AlwaysLastRecordBatchPop { subject_name: _tn } => "AlwaysLastRecordBatchPop",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }
}

impl TraceableTrait for Subscription {
    fn to_trace(&self) -> Tracer {
        Tracer::new(self.get_short_name(), self.get_table_name())
    }
}

/// Subscribe to an arrow table
pub trait TableSubscriptionTrait: SubjectTrait {
    /// Implement the subscription
    ///
    /// # Notes
    ///
    /// * Empty tables are skipped
    /// * `Subscription` where `is_clone` = false are skipped
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_to_table(
        &self,
        subscribe: &Subscription,
    ) -> Option<SendableRecordBatchStream>;

    /// Implement the subscription mutating the table
    ///
    /// # Notes
    ///
    /// * Empty tables are skipped
    /// * `Subscription` where `is_mut` = false are skipped
    ///
    /// # Arguments
    ///
    /// * `updated` - whether the table has been updated or not
    /// * `subscribe` - `ArrowTableSubscribe` the subscription enum
    fn subscribe_to_table_mut(
        &mut self,
        subscribe: &Subscription,
    ) -> Option<SendableRecordBatchStream>;
}

impl TableSubscriptionTrait for Subject {
    fn subscribe_to_table(
        &self,
        subscribe: &Subscription,
    ) -> Option<SendableRecordBatchStream> {
        if self.count_rows() == 0 {
            return None;
        }
        match subscribe {
            Subscription::AlwaysFullTable { subject_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            Subscription::AlwaysLastRecordBatch { subject_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            Subscription::OnUpdateFullTable { subject_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            Subscription::OnUpdateLastRecordBatch { subject_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            Subscription::OnUpdateEmpty { subject_name: _ } => {
                let schema = Schema::empty();
                let stream = futures::stream::iter(Vec::new().into_iter().map(Ok));
                Some(Box::pin(RecordBatchStreamAdapter::new(
                    Arc::new(schema),
                    stream,
                )))
            }
            Subscription::AlwaysFullTableDrain { subject_name: _ } => None,
            Subscription::AlwaysLastRecordBatchPop { subject_name: _ } => None,
            Subscription::OnUpdateFullTableDrain { subject_name: _ } => None,
            Subscription::OnUpdateLastRecordBatchPop { subject_name: _ } => None,
            Subscription::None => None,
            Subscription::Custom(_) => None,
        }
    }
    fn subscribe_to_table_mut(
        &mut self,
        subscribe: &Subscription,
    ) -> Option<SendableRecordBatchStream> {
        if self.count_rows() == 0 {
            return None;
        }
        match subscribe {
            Subscription::AlwaysFullTableDrain { subject_name: _ } => {
                Some(self.to_record_batch_stream_drain())
            }
            Subscription::AlwaysLastRecordBatchPop { subject_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch_pop())
            }
            Subscription::OnUpdateFullTableDrain { subject_name: _ } => {
                Some(self.to_record_batch_stream_drain())
            }
            Subscription::OnUpdateLastRecordBatchPop { subject_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch_pop())
            }
            Subscription::OnUpdateEmpty { subject_name: _ } => None,
            Subscription::AlwaysFullTable { subject_name: _ } => None,
            Subscription::AlwaysLastRecordBatch { subject_name: _ } => None,
            Subscription::OnUpdateFullTable { subject_name: _ } => None,
            Subscription::OnUpdateLastRecordBatch { subject_name: _ } => None,
            Subscription::None => None,
            Subscription::Custom(_) => None,
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
        let publication = Subscription::AlwaysFullTable {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|FullTable|message_parser-subscribe";
        let publication = Subscription::OnUpdateFullTable {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|LastRecordBatch|message_parser-subscribe";
        let publication = Subscription::AlwaysLastRecordBatch {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|LastRecordBatch|message_parser-subscribe";
        let publication = Subscription::OnUpdateLastRecordBatch {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|Empty|message_parser-subscribe";
        let publication = Subscription::OnUpdateEmpty {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|FullTableDrain|message_parser-subscribe";
        let subject = "message_parser";
        let publication = Subscription::AlwaysFullTableDrain {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|FullTableDrain|message_parser-subscribe";
        let publication = Subscription::OnUpdateFullTableDrain {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-->|LastRecordBatchPop|message_parser-subscribe";
        let publication = Subscription::AlwaysLastRecordBatchPop {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|LastRecordBatchPop|message_parser-subscribe";
        let publication = Subscription::OnUpdateLastRecordBatchPop {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        Ok(())
    }
}
