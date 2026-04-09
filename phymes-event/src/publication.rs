use std::fmt::Display;

use anyhow::{Result, anyhow};
use phymes_subject::MappableTrait;
use phymes_schemas::DataFormat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq, Default)]
pub enum Publication {
    /// Push a new vector of record batches onto the table
    Extend { subject_name: String },
    /// Push a new vector of record batches onto the table
    /// after joining the chunks along the named column
    ExtendChunks {
        subject_name: String,
        col_name: String,
    },
    /// Push a new vector of record batches onto the table
    /// after deserializing bytes from a specified format
    /// DM: intended for internal routing of messages
    ExtendBytes {
        subject_name: String,
        col_name: String,
        serialize_format: DataFormat,
    },
    /// Replace the existing vector of record batches with a new one
    Replace { subject_name: String },
    /// Replace only the last record batch
    ReplaceLast { subject_name: String },
    /// No updates
    #[default]
    None,
    /// Custom update function
    Custom(String),
}

impl Publication {
    /// Short name for the [Publication] that omits the `subject_name` and other information
    pub fn short_name(&self) -> &str {
        match self {
            Self::Extend { subject_name: _tn } => "Extend",
            Self::ExtendChunks {
                subject_name: _tn,
                col_name: _cn,
            } => "ExtendChunks",
            Self::ExtendBytes {
                subject_name: _tn,
                col_name: _cn,
                serialize_format: _sf,
            } => "ExtendBytes",
            Self::Replace { subject_name: _tn } => "Replace",
            Self::ReplaceLast { subject_name: _tn } => "ReplaceLast",
            Self::None => "None",
            Self::Custom(name) => name,
        }
    }

    /// Full name for the [Publication] that includes the `subject_name` and other information
    pub fn full_name(&self) -> String {
        match self {
            Self::Extend { subject_name: tn } => format!("extend-{tn}"),
            Self::ExtendChunks {
                subject_name: tn,
                col_name: cn,
            } => format!("extend-chunks-{tn}-{cn}"),
            Self::ExtendBytes {
                subject_name: tn,
                col_name: cn,
                serialize_format: sf,
            } => format!("extend-values-{tn}-{cn}-{sf}"),
            Self::Replace { subject_name: tn } => format!("replace-{tn}"),
            Self::ReplaceLast { subject_name: tn } => format!("replace-last-{tn}"),
            Self::None => "none".to_string(),
            Self::Custom(name) => name.to_string(),
        }
    }

    /// The `subject_name` of the variant
    pub fn subject_name(&self) -> &str {
        match self {
            Self::Extend { subject_name: tn } => tn,
            Self::ExtendChunks {
                subject_name: tn,
                col_name: _cn,
            } => tn,
            Self::ExtendBytes {
                subject_name: tn,
                col_name: _cn,
                serialize_format: _sf,
            } => tn,
            Self::Replace { subject_name: tn } => tn,
            Self::ReplaceLast { subject_name: tn } => tn,
            Self::None => "",
            Self::Custom(_name) => "",
        }
    }

    /// New [Publication] from a short name identifying the variant and the `subject_name`
    pub fn from_str_fuzzy(name: &str, subject: &str) -> Result<Publication> {
        let publication = if name.contains("ExtendChunks") {
            Publication::ExtendChunks {
                subject_name: subject.to_string(),
                col_name: "content".to_string(),
            }
        } else if name.contains("Extend") {
            Publication::Extend {
                subject_name: subject.to_string(),
            }
        } else if name.contains("ReplaceLast") {
            Publication::ReplaceLast {
                subject_name: subject.to_string(),
            }
        } else if name.contains("Replace") {
            Publication::Replace {
                subject_name: subject.to_string(),
            }
        } else if name.contains("None") {
            Publication::None {}
        } else {
            return Err(anyhow!(
                "Variant for ArrowTablePublish {name} with subject {subject} was not recognized."
            ));
        };
        Ok(publication)
    }

    /// New [Publication] from a short name identifying the variant, the subject `subject_name`
    ///   and the mermaid.js flowchart diagram link type
    pub fn from_str_mermaid(line: &str, subject: &str) -> Result<Publication> {
        if line.contains("|") & line.contains("-->") & line.contains("ExtendChunks") {
            Ok(Publication::ExtendChunks {
                subject_name: subject.to_string(),
                col_name: "content".to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("Extend") {
            Ok(Publication::Extend {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("ReplaceLast") {
            Ok(Publication::ReplaceLast {
                subject_name: subject.to_string(),
            })
        } else if line.contains("|") & line.contains("-->") & line.contains("Replace") {
            Ok(Publication::Replace {
                subject_name: subject.to_string(),
            })
        } else if line.contains("None") {
            Ok(Publication::None {})
        } else {
            Err(anyhow!(
                "Variant for Publication with subject {subject} was not recognized in string slice {line}."
            ))
        }
    }
}

impl Display for Publication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extend { subject_name: _ } => write!(f, "Extend"),
            Self::Replace { subject_name: _ } => write!(f, "Replace"),
            Self::ReplaceLast { subject_name: _ } => write!(f, "ReplaceLast"),
            Self::None => write!(f, "None"),
            Self::ExtendChunks {
                subject_name: _,
                col_name: _,
            } => write!(f, "ExtendChunks"),
            Self::ExtendBytes {
                subject_name: _,
                col_name: _,
                serialize_format: _,
            } => write!(f, "ExtendBytes"),
            Self::Custom(_s) => write!(f, "Custom"),
        }
    }
}

impl MappableTrait for Publication {
    fn get_name(&self) -> &str {
        self.short_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_publication_from_str_mermaid() -> Result<()> {
        let line = "message_parser-publish-->|ExtendChunks|AssistantMessages-subject";
        let subject = "AssistantMessages";
        let publication = Publication::ExtendChunks {
            subject_name: subject.to_string(),
            col_name: "content".to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parser-publish-->|Extend|AssistantMessages-subject";
        let publication = Publication::Extend {
            subject_name: subject.to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parser-publish-->|ReplaceLast|AssistantMessages-subject";
        let publication = Publication::ReplaceLast {
            subject_name: subject.to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        let line = "message_parser-publish-->|Replace|AssistantMessages-subject";
        let publication = Publication::Replace {
            subject_name: subject.to_string(),
        };
        let test = Publication::from_str_mermaid(line, subject)?;
        assert_eq!(test, publication);

        Ok(())
    }
}
