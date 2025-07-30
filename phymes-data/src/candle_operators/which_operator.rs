use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
};

/// General dependencies
use clap::ValueEnum;
use futures::future::Join;
use phymes_core::{
    session::common_traits::BuilderTrait,
    table::arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait},
};
use serde::{Deserialize, Serialize};

use phymes_ai::openai_asset::{chat_completion, types};

use crate::candle_operators::{chunk_documents::ChunkDocuments, human_in_the_loop::HumanInTheLoop, join_inner::JoinInner, sort_scores_and_indices::SortScoresAndIndices};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum WhichCandleOperator {
    #[value(name = "relative-similarity-score")]
    #[serde(alias = "relative-similarity-score")]
    RelativeSimilarityScore,
    #[value(name = "sort-scores-and-indices")]
    #[serde(alias = "sort-scores-and-indices")]
    SortScoresAndIndices,
    #[value(name = "human-in-the-loop")]
    #[serde(alias = "human-in-the-loop")]
    HumanInTheLoops,
    #[value(name = "chunk-documents")]
    #[serde(alias = "chunk-documents")]
    ChunkDocuments,
    #[value(name = "join-inner")]
    #[serde(alias = "join-inner")]
    JoinInner,
    #[value(name = "extract-pdf-text")]
    #[serde(alias = "extract-pdf-text")]
    ExtractPDFText,
}

impl Default for WhichCandleOperator {
    fn default() -> Self {
        Self::RelativeSimilarityScore
    }
}

impl WhichCandleOperator {
    /// The name of the operation
    pub fn get_name(&self) -> &str {
        match self {
            Self::RelativeSimilarityScore => "relative-similarity-score",
            Self::SortScoresAndIndices => "sort-scores-and-indices",
            Self::HumanInTheLoops => "human-in-the-loop",
            Self::ChunkDocuments => "chunk-documents",
            Self::JoinInner => "join-inner",
        }
    }

    /// The description to use for the operation
    pub fn get_description(&self) -> &str {
        match self {
            Self::RelativeSimilarityScore => {
                "Compute the relative similarity score between two different lists of embedding vectors"
            }
            Self::SortScoresAndIndices => "Sort the the list of computed scores in ascending order",
            Self::HumanInTheLoops => {
                "Ask a question to clarify the user's query, ask a questionn to get additional information that the user did not provide, confirm a choice of tool, confirm arguments for a tool before answering the user's query or calling a tool, or provide the answer to the user's query."
            }
            Self::ChunkDocuments => "Chunk documents by splitting the document text",
            Self::JoinInner => "Join two tables on their foreign keys",
        }
    }

    /// The description to use for the operation
    pub fn get_json_tool_schema(&self) -> String {
        match self {
            Self::RelativeSimilarityScore
            | Self::SortScoresAndIndices
            | Self::ChunkDocuments
            | Self::JoinInner => {
                let mut properties = HashMap::new();
                properties.insert(
                    "lhs_name".to_string(),
                    Box::new(types::JSONSchemaDefine {
                        schema_type: Some(types::JSONSchemaType::String),
                        description: Some("The name of the left hand side table".to_string()),
                        ..Default::default()
                    }),
                );
                // properties.insert(
                //     "rhs_name".to_string(),
                //     Box::new(types::JSONSchemaDefine {
                //         schema_type: Some(types::JSONSchemaType::String),
                //         description: Some("The name of the right hand side table".to_string()),
                //         ..Default::default()
                //     }),
                // );
                properties.insert(
                    "lhs_pk".to_string(),
                    Box::new(types::JSONSchemaDefine {
                        schema_type: Some(types::JSONSchemaType::String),
                        description: Some(
                            "The primary key column identifier for the left hand side table"
                                .to_string(),
                        ),
                        ..Default::default()
                    }),
                );
                // properties.insert(
                //     "rhs_pk".to_string(),
                //     Box::new(types::JSONSchemaDefine {
                //         schema_type: Some(types::JSONSchemaType::String),
                //         description: Some("The primary key column identifier for the right hand side table".to_string()),
                //         ..Default::default()
                //     }),
                // );
                // properties.insert(
                //     "lhs_fk".to_string(),
                //     Box::new(types::JSONSchemaDefine {
                //         schema_type: Some(types::JSONSchemaType::String),
                //         description: Some("The foriegn key column identifier for the left hand side table".to_string()),
                //         ..Default::default()
                //     }),
                // );
                // properties.insert(
                //     "rhs_fk".to_string(),
                //     Box::new(types::JSONSchemaDefine {
                //         schema_type: Some(types::JSONSchemaType::String),
                //         description: Some("The foriegn key column identifier for the right hand side table".to_string()),
                //         ..Default::default()
                //     }),
                // );
                properties.insert(
                    "lhs_values".to_string(),
                    Box::new(types::JSONSchemaDefine {
                        schema_type: Some(types::JSONSchemaType::String),
                        description: Some(
                            "The values column identifier for the left hand side table".to_string(),
                        ),
                        ..Default::default()
                    }),
                );
                // properties.insert(
                //     "rhs_values".to_string(),
                //     Box::new(types::JSONSchemaDefine {
                //         schema_type: Some(types::JSONSchemaType::String),
                //         description: Some("The values column identifier for the right hand side table".to_string()),
                //         ..Default::default()
                //     }),
                // );
                let function = types::Function {
                    name: self.get_name().to_string(),
                    description: Some(self.get_description().to_string()),
                    parameters: types::FunctionParameters {
                        schema_type: types::JSONSchemaType::Object,
                        properties: Some(properties),
                        required: Some(vec![
                            "lhs_name".to_string(),
                            "lhs_pk".to_string(),
                            "lhs_values".to_string(),
                        ]),
                    },
                };
                let tool = chat_completion::Tool {
                    r#type: chat_completion::ToolType::Function,
                    function,
                };
                serde_json::to_string(&tool).unwrap()
            }
            Self::HumanInTheLoops => {
                let mut properties = HashMap::new();
                properties.insert(
                    "lhs_args".to_string(),
                    Box::new(types::JSONSchemaDefine {
                        schema_type: Some(types::JSONSchemaType::String),
                        description: Some("The question or answer for the user. Format lhs_arg value as JSON according to the schema {\"role\": \"assistant\", \"content\": \"`RESPONSE`\"} where `RESPONSE` is where you put your question or answer for the user".to_string()),
                        ..Default::default()
                    }),
                );
                let function = types::Function {
                    name: self.get_name().to_string(),
                    description: Some(self.get_description().to_string()),
                    parameters: types::FunctionParameters {
                        schema_type: types::JSONSchemaType::Object,
                        properties: Some(properties),
                        required: Some(vec!["lhs_args".to_string()]),
                    },
                };
                let tool = chat_completion::Tool {
                    r#type: chat_completion::ToolType::Function,
                    function,
                };
                serde_json::to_string(&tool).unwrap()
            }
        }
    }

    /// Return the operation based on the name
    pub fn new_from_name(name: &str) -> Option<Self> {
        if name == RelativeSimilarityScore::get_name() {
            Some(Self::RelativeSimilarityScore)
        } else if name == SortScoresAndIndices::get_name() {
            Some(Self::SortScoresAndIndices)
        } else if name == HumanInTheLoop::get_name() {
            Some(Self::HumanInTheLoops)
        } else if name == ChunkDocuments::get_name() {
            Some(Self::ChunkDocuments)
        } else if name == JoinInner::get_name() {
            Some(Self::JoinInner)
        } else {
            //Err(anyhow!("No Candle Operator found for {}.", name))
            None
        }
    }
}

pub fn convert_destinations_to_tools(name: &str, destinations: &[String]) -> Option<ArrowTable> {
    let mut tool_id_vec = Vec::new();
    let mut tool_vec = Vec::new();
    for destination in destinations.iter() {
        if let Some(ops) = WhichCandleOperator::new_from_name(destination) {
            tool_id_vec.push(ops.get_name().to_string());
            tool_vec.push(ops.get_json_tool_schema());
        }
    }
    if tool_id_vec.is_empty() {
        None
    } else {
        let tool_id: ArrayRef = Arc::new(StringArray::from(tool_id_vec));
        let tool: ArrayRef = Arc::new(StringArray::from(tool_vec));
        let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_id), ("tool", tool)]).unwrap();
        let table = ArrowTableBuilder::new()
            .with_name(name)
            .with_record_batches(vec![batch])
            .unwrap()
            .build()
            .unwrap();
        Some(table)
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::table::arrow_table::ArrowTableTrait;

    use super::*;

    #[test]
    fn test_convert_destinations_to_tools_all() -> Result<()> {
        let result = convert_destinations_to_tools(
            "test",
            &[
                "relative-similarity-score".to_string(),
                "sort-scores-and-indices".to_string(),
                "chunk-documents".to_string(),
                "join-inner".to_string(),
                "human-in-the-loop".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(
            result.get_column_as_str_vec("tool_id"),
            &[
                "relative-similarity-score",
                "sort-scores-and-indices",
                "chunk-documents",
                "join-inner",
                "human-in-the-loop",
            ]
        );
        let functions = result.get_column_as_str_vec("tool");
        assert!(functions.first().unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"relative-similarity-score\",\"description\":\"Compute the relative similarity score between two different lists of embedding vectors\"")
        );
        assert!(
            functions
                .first()
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.first().unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.first().unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.first().unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .first()
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(1).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"sort-scores-and-indices\",\"description\":\"Sort the the list of computed scores in ascending order\"")
        );
        assert!(
            functions
                .get(1)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(1).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(1).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(1).unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .get(1)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(2).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"chunk-documents\",\"description\":\"Chunk documents by splitting the document text\"")
        );
        assert!(
            functions
                .get(2)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(2).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(2).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(2).unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .get(2)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(3).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"join-inner\",\"description\":\"Join two tables on their foreign keys\"")
        );
        assert!(
            functions
                .get(3)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_name\":{\"type\":\"string\",\"description\":\"The name of the left hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_values\":{\"type\":\"string\",\"description\":\"The values column identifier for the left hand side table\"")
        );
        assert!(functions.get(3).unwrap().contains("\"lhs_pk\":{\"type\":\"string\",\"description\":\"The primary key column identifier for the left hand side table\"")
        );
        assert!(
            functions
                .get(3)
                .unwrap()
                .contains("\"required\":[\"lhs_name\",\"lhs_pk\",\"lhs_values\"]}}}")
        );

        assert!(functions.get(4).unwrap().contains("{\"type\":\"function\",\"function\":{\"name\":\"human-in-the-loop\",\"description\":\"Ask a question to clarify the user's query, ask a questionn to get additional information that the user did not provide, confirm a choice of tool, confirm arguments for a tool before answering the user's query or calling a tool, or provide the answer to the user's query.\"")
        );
        assert!(
            functions
                .get(4)
                .unwrap()
                .contains("\"parameters\":{\"type\":\"object\",\"properties\":{")
        );
        assert!(functions.get(4).unwrap().contains("\"lhs_args\":{\"type\":\"string\",\"description\":\"The question or answer for the user. Format lhs_arg value as JSON according to the schema {\\\"role\\\": \\\"assistant\\\", \\\"content\\\": \\\"`RESPONSE`\\\"} where `RESPONSE` is where you put your question or answer for the user\"")
        );
        assert!(
            functions
                .get(4)
                .unwrap()
                .contains("\"required\":[\"lhs_args\"]}}}")
        );

        Ok(())
    }

    #[test]
    fn test_convert_destinations_to_tools_missing() -> Result<()> {
        let result = convert_destinations_to_tools("test", &["missing".to_string()]);
        assert!(result.is_none());
        Ok(())
    }
}
