use anyhow::{Result, anyhow};
use arrow::datatypes::Schema;
use object_store::{ObjectStore, path::Path};
use phymes_diagnostics::{TraceableTrait, Tracer};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

use crate::{RecordBatchStreamAdapter, MappableTrait, SendableRecordBatchStream, Subject, SubjectTrait};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum Subscription {
    /// Only when the subject has been updated, copy the full table
    OnUpdateAllRecordBatches { subject_name: String },
    /// Only when the subject has been updated, and just a copy of the last RecordBatch
    OnUpdateLastRecordBatch { subject_name: String },
    /// Only when the subject has been updated, but don't copy or take any data
    ///   which is useful for ensuring a task is triggered after another task
    OnUpdateEmpty { subject_name: String },
    /// Always copy the full table
    AlwaysAllRecordBatches { subject_name: String },
    /// Always copy just the last record batch
    AlwaysLastRecordBatch { subject_name: String },
    /// No reading of the table
    #[default]
    None,
    /// Custom subscription function
    Custom(String),
}

impl Subscription {
    /// The `subject_name` of the variant
    pub fn subject_name(&self) -> &str {
        match self {
            Self::OnUpdateAllRecordBatches { subject_name: tn } => tn,
            Self::OnUpdateLastRecordBatch { subject_name: tn } => tn,
            Self::OnUpdateEmpty { subject_name: tn } => tn,
            Self::AlwaysAllRecordBatches { subject_name: tn } => tn,
            Self::AlwaysLastRecordBatch { subject_name: tn } => tn,
            Self::None => "",
            Self::Custom(_name) => "",
        }
    }

    #[allow(dead_code)]
    /// Full name for the [Subscription] that includes the `subject_name` and other information
    fn full_name(&self) -> String {
        match self {
            Self::OnUpdateAllRecordBatches { subject_name: tn } => format!("OnUpdateAllRecordBatches-{tn}"),
            Self::OnUpdateLastRecordBatch { subject_name: tn } => {
                format!("OnUpdateLastRecordBatch-{tn}")
            }
            Self::OnUpdateEmpty { subject_name: tn } => format!("OnUpdateEmpty-{tn}"),
            Self::AlwaysAllRecordBatches { subject_name: tn } => format!("AlwaysAllRecordBatches-{tn}"),
            Self::AlwaysLastRecordBatch { subject_name: tn } => format!("AlwaysLastRecordBatch-{tn}"),
            Self::None => "None".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    /// Is the subscription triggered by a table update?
    pub fn is_update(&self) -> bool {
        match self {
            Self::OnUpdateAllRecordBatches { subject_name: _tn }
            | Self::OnUpdateLastRecordBatch { subject_name: _tn }
            | Self::OnUpdateEmpty { subject_name: _tn } => true,
            Self::AlwaysAllRecordBatches { subject_name: _tn }
            | Self::AlwaysLastRecordBatch { subject_name: _tn } => false,
            Self::None => false,
            Self::Custom(_name) => false,
        }
    }

    /// Short name for the [Subscription] that omits the `subject_name` and other information
    pub fn short_name(&self) -> &str {
        match self {
            Self::OnUpdateAllRecordBatches { subject_name: _tn } => "AllRecordBatches",
            Self::OnUpdateLastRecordBatch { subject_name: _tn } => "LastRecordBatch",
            Self::OnUpdateEmpty { subject_name: _tn } => "Empty",
            Self::AlwaysAllRecordBatches { subject_name: _tn } => "AllRecordBatches",
            Self::AlwaysLastRecordBatch { subject_name: _tn } => "LastRecordBatch",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }

    /// New [Subscription] from a short name identifying the variant and the `subject_name`
    pub fn from_str_fuzzy(name: &str, subject: &str) -> Result<Subscription> {
        let subscription = if name.contains("OnUpdateAllRecordBatches") {
            Subscription::OnUpdateAllRecordBatches {
                subject_name: subject.to_string(),
            }
        } else if name.contains("AlwaysAllRecordBatches") {
            Subscription::AlwaysAllRecordBatches {
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
        if line.contains("|") & line.contains("-.->") & line.contains("AllRecordBatchesDrain") {
            Ok(Subscription::OnUpdateAllRecordBatchesDrain {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("AllRecordBatchesDrain") {
            Ok(Subscription::AlwaysAllRecordBatchesDrain {
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
        } else if line.contains("|") & line.contains("-.->") & line.contains("AllRecordBatches") {
            Ok(Subscription::OnUpdateAllRecordBatches {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("AllRecordBatches") {
            Ok(Subscription::AlwaysAllRecordBatches {
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
            Self::OnUpdateAllRecordBatches { subject_name: _tn } => "OnUpdateAllRecordBatches",
            Self::OnUpdateLastRecordBatch { subject_name: _tn } => "OnUpdateLastRecordBatch",
            Self::OnUpdateEmpty { subject_name: _tn } => "OnUpdateEmpty",
            Self::AlwaysAllRecordBatches { subject_name: _tn } => "AlwaysAllRecordBatches",
            Self::AlwaysLastRecordBatch { subject_name: _tn } => "AlwaysLastRecordBatch",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }
}

impl TraceableTrait for Subscription {
    fn to_trace(&self) -> Tracer {
        Tracer::new(self.short_name(), self.subject_name())
    }
}

/// Subscribe to an arrow table
pub trait SubscriptionTrait {
    /// Implement the subscription
    ///
    /// # Notes
    ///
    /// * Empty tables are skipped
    /// * `Subscription` where `is_clone` = false are skipped
    ///
    /// # Arguments
    ///
    /// * `store` - `Arc<dyn ObjectStore>` the object store
    fn subscribe_to_subject(
        &self,
        store: &Arc<dyn ObjectStore>,
    ) -> Option<SendableRecordBatchStream>;
}

impl SubscriptionTrait for Subscription {
    fn subscribe_to_subject(
        &self,
        store: &Arc<dyn ObjectStore>,
    ) -> Option<SendableRecordBatchStream> {
        // 1. List the partitions (RecordBatches)
        
        // 2. Get all or only the most recent
        if self.count_rows() == 0 {
            return None;
        }
        match self {
            Self::AlwaysAllRecordBatches { subject_name: sn } => {
                let path = Path::from(sn.to_string());
                let list = store.list(Some(&path)).map_ok(|m| m.location).boxed();
            }
            Self::AlwaysLastRecordBatch { subject_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            Self::OnUpdateAllRecordBatches { subject_name: _ } => {
                Some(self.to_record_batch_stream())
            }
            Self::OnUpdateLastRecordBatch { subject_name: _ } => {
                Some(self.to_record_batch_stream_last_record_batch())
            }
            Self::OnUpdateEmpty { subject_name: _ } => {
                let schema = Schema::empty();
                let stream = futures::stream::iter(Vec::new().into_iter().map(Ok));
                Some(Box::pin(RecordBatchStreamAdapter::new(
                    Arc::new(schema),
                    stream,
                )))
            }
            Self::None => None,
            Self::Custom(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_subscription_from_str_mermaid() -> Result<()> {
        let line = "message_parsing-subject-->|AllRecordBatches|message_parser-subscribe";
        let subject = "message_parser";
        let publication = Subscription::AlwaysAllRecordBatches {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|AllRecordBatches|message_parser-subscribe";
        let publication = Subscription::OnUpdateAllRecordBatches {
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

        let line = "message_parsing-subject-->|AllRecordBatchesDrain|message_parser-subscribe";
        let subject = "message_parser";
        let publication = Subscription::AlwaysAllRecordBatchesDrain {
            subject_name: subject.to_string(),
        };
        let test = Subscription::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parsing-subject-.->|AllRecordBatchesDrain|message_parser-subscribe";
        let publication = Subscription::OnUpdateAllRecordBatchesDrain {
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
