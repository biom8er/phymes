use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
};

/// General dependencies
use clap::ValueEnum;
use phymes_core::{
    session::common_traits::BuilderTrait,
    table::arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait},
};
use serde::{Deserialize, Serialize};

use crate::openai_asset::{chat_completion, types};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum WhichCandleOps {
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
}

impl Default for WhichCandleOps {
    fn default() -> Self {
        Self::RelativeSimilarityScore
    }
}

impl WhichCandleOps {
    /// Get the mandatory fields that are expected to be found in
    /// the LHS input schema
    pub fn get_schema_lhs_input(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        match self {
            Self::RelativeSimilarityScore => {
                let lhs_pk = Field::new(lhs_pk, DataType::Utf8, false);
                let lhs_fk = Field::new(lhs_fk, DataType::Utf8, false);
                let embed_size = list_size.unwrap_or(2);
                let list_data_type = DataType::FixedSizeList(
                    Arc::new(Field::new_list_field(DataType::Float32, false)),
                    embed_size.try_into().unwrap(),
                );
                assert_eq!(lhs_value, "embeddings");
                let lhs_value = Field::new(lhs_value, list_data_type, false);
                let mut fields = vec![lhs_pk, lhs_fk, lhs_value];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::SortScoresAndIndices => {
                assert_eq!(lhs_value, "score");
                let lhs_value = Field::new(lhs_value, DataType::Float32, false);
                let mut fields = vec![lhs_value];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::HumanInTheLoops => {
                let role = Field::new("role", DataType::Utf8, false);
                let content = Field::new("content", DataType::Utf8, false);
                let mut fields = vec![role, content];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::ChunkDocuments => {
                let lhs_pk = Field::new(lhs_pk, DataType::Utf8, false);
                let text = Field::new("text", DataType::Utf8, false);
                let mut fields = vec![lhs_pk, text];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::JoinInner => {
                let lhs_pk = Field::new(lhs_pk, DataType::Utf8, false);
                let lhs_fk = Field::new(lhs_fk, DataType::Utf8, false);
                let mut fields = vec![lhs_pk, lhs_fk];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
        }
    }

    /// Get the mandatory fields that are expected to be found in
    /// the RHS input schema
    pub fn get_schema_rhs_input(
        &self,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        match self {
            Self::RelativeSimilarityScore => {
                let rhs_pk = Field::new(rhs_pk, DataType::Utf8, false);
                let rhs_fk = Field::new(rhs_fk, DataType::Utf8, false);
                let embed_size = list_size.unwrap_or(2);
                let list_data_type = DataType::FixedSizeList(
                    Arc::new(Field::new_list_field(DataType::Float32, false)),
                    embed_size.try_into().unwrap(),
                );
                assert_eq!(rhs_values, "embeddings");
                let rhs_values = Field::new(rhs_values, list_data_type, false);
                let mut fields = vec![rhs_pk, rhs_fk, rhs_values];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::SortScoresAndIndices => None,
            Self::HumanInTheLoops => None,
            Self::ChunkDocuments => None,
            Self::JoinInner => {
                let rhs_pk = Field::new(rhs_pk, DataType::Utf8, false);
                let rhs_fk = Field::new(rhs_fk, DataType::Utf8, false);
                let mut fields = vec![rhs_pk, rhs_fk];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
        }
    }

    /// Get the mandatory fields that are expected to be found in
    /// the output schema
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn get_schema_output(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        list_size: Option<usize>,
        other: Option<Vec<Field>>,
    ) -> Option<SchemaRef> {
        match self {
            Self::RelativeSimilarityScore => {
                let lhs_pk = Field::new(lhs_pk, DataType::Utf8, false);
                let rhs_pk = Field::new(rhs_pk, DataType::Utf8, false);
                let score = Field::new("score", DataType::Float32, false);
                let mut fields = vec![lhs_pk, rhs_pk, score];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::SortScoresAndIndices => {
                let score = Field::new("score", DataType::Float32, false);
                let mut fields = vec![score];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::HumanInTheLoops => None,
            Self::ChunkDocuments => {
                let lhs_pk = Field::new(lhs_pk, DataType::Utf8, false);
                let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
                let text = Field::new("text", DataType::Float32, false);
                let mut fields = vec![lhs_pk, chunk_id, text];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
            Self::JoinInner => {
                let lhs_fk = Field::new(lhs_fk, DataType::Utf8, false);
                let rhs_fk = Field::new(rhs_fk, DataType::Utf8, false);
                let mut fields = vec![lhs_fk, rhs_fk];
                if let Some(other) = other {
                    fields.extend(other);
                }
                Some(Arc::new(Schema::new(fields)))
            }
        }
    }

    /// Check the expected mandatory fields for the LHS input
    #[allow(unused_variables)]
    pub fn check_schema_lhs_input(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> {
        match self {
            Self::RelativeSimilarityScore => {
                if other.column_with_name(lhs_pk).is_none() {
                    return Err(anyhow!(
                        "LHS input is missing column for lhs_pk {}.",
                        lhs_pk
                    ));
                }
                if other.column_with_name("embeddings").is_none() {
                    return Err(anyhow!("LHS input is missing column for embeddings."));
                }
                Ok(Some(true))
            }
            Self::SortScoresAndIndices => {
                if other.column_with_name("score").is_none() {
                    return Err(anyhow!("LHS input is missing column for score."));
                }
                Ok(Some(true))
            }
            Self::HumanInTheLoops => {
                if other.column_with_name("role").is_none() {
                    return Err(anyhow!("LHS input is missing column for role."));
                }
                if other.column_with_name("content").is_none() {
                    return Err(anyhow!("LHS input is missing column for content."));
                }
                Ok(Some(true))
            }
            Self::ChunkDocuments => {
                if other.column_with_name(lhs_pk).is_none() {
                    return Err(anyhow!(
                        "LHS input is missing column for lhs_pk {}.",
                        lhs_pk
                    ));
                }
                if other.column_with_name("text").is_none() {
                    return Err(anyhow!("LHS input is missing column for text."));
                }
                Ok(Some(true))
            }
            Self::JoinInner => {
                if other.column_with_name(lhs_fk).is_none() {
                    return Err(anyhow!(
                        "LHS input is missing column for lhs_fk {}.",
                        lhs_fk
                    ));
                }
                Ok(Some(true))
            }
        }
    }

    /// Check the expected mandatory fields for the RHS input
    #[allow(unused_variables)]
    pub fn check_schema_rhs_input(
        &self,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> {
        match self {
            Self::RelativeSimilarityScore => {
                if other.column_with_name(rhs_pk).is_none() {
                    return Err(anyhow!(
                        "RHS input is missing column for rhs_pk {}.",
                        rhs_pk
                    ));
                }
                if other.column_with_name("embeddings").is_none() {
                    return Err(anyhow!("RHS input is missing column for embeddings."));
                }
                Ok(Some(true))
            }
            Self::SortScoresAndIndices => Ok(None),
            Self::HumanInTheLoops => Ok(None),
            Self::ChunkDocuments => Ok(None),
            Self::JoinInner => {
                if other.column_with_name(rhs_fk).is_none() {
                    return Err(anyhow!(
                        "RHS input is missing column for rhs_fk {}.",
                        rhs_fk
                    ));
                }
                Ok(Some(true))
            }
        }
    }

    /// Check the expected mandatory fields for the output
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn check_schema_output(
        &self,
        lhs_pk: &str,
        lhs_fk: &str,
        lhs_value: &str,
        rhs_pk: &str,
        rhs_fk: &str,
        rhs_values: &str,
        other: SchemaRef,
    ) -> Result<Option<bool>> {
        match self {
            Self::RelativeSimilarityScore => {
                if other.column_with_name(lhs_pk).is_none() {
                    return Err(anyhow!("LHS output is missing column for lhs_pk."));
                }
                if other.column_with_name(rhs_pk).is_none() {
                    return Err(anyhow!("RHS output is missing column for rhs_pk."));
                }
                if other.column_with_name("embeddings").is_none() {
                    return Err(anyhow!("Output is missing column for embeddings."));
                }
                Ok(Some(true))
            }
            Self::SortScoresAndIndices => {
                if other.column_with_name("score").is_none() {
                    return Err(anyhow!("LHS output is missing column for score."));
                }
                Ok(Some(true))
            }
            Self::HumanInTheLoops => Ok(None),
            Self::ChunkDocuments => {
                if other.column_with_name(lhs_pk).is_none() {
                    return Err(anyhow!("LHS output is missing column for lhs_pk."));
                }
                if other.column_with_name("chunk_id").is_none() {
                    return Err(anyhow!("RHS output is missing column for chunk_id."));
                }
                if other.column_with_name("text").is_none() {
                    return Err(anyhow!("Output is missing column for text."));
                }
                Ok(Some(true))
            }
            Self::JoinInner => {
                if other.column_with_name(lhs_fk).is_none() {
                    return Err(anyhow!("LHS output is missing column for lhs_fk."));
                }
                if other.column_with_name(rhs_fk).is_none() {
                    return Err(anyhow!("RHS output is missing column for rhs_fk."));
                }
                Ok(Some(true))
            }
        }
    }

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
        if name == "relative-similarity-score" {
            Some(Self::RelativeSimilarityScore)
        } else if name == "sort-scores-and-indices" {
            Some(Self::SortScoresAndIndices)
        } else if name == "human-in-the-loop" {
            Some(Self::HumanInTheLoops)
        } else if name == "chunk-documents" {
            Some(Self::ChunkDocuments)
        } else if name == "join-inner" {
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
        if let Some(ops) = WhichCandleOps::new_from_name(destination) {
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
