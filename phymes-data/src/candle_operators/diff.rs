use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt32Array},
    datatypes::{Schema, SchemaRef},
};
use candle_core::Device;
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait};
use phymes_schemas::{
    Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType, Tool, ToolType
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tracing::instrument;

use crate::{
    DataJoinOperator, DiffType, PatchOperator, ToolTrait,
    compute_diff,
    candle_data::DataConfig,
    candle_operators::{DataOperatorTrait, join::join},
};

/// Inject a table into a string template
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Diff {
    lhs_values: Vec<String>,
    rhs_values: Vec<String>,
    lhs_pk: String,
    rhs_pk: String,
    diff: DiffType,
}

impl MappableTrait for Diff {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl ToolTrait for Diff {
    fn get_description(&self) -> String {
        "Compute the diff between two RecordBatches".to_string()
    }
    fn to_json_tool_schema(&self) -> String {
        let mut properties = HashMap::new();
        properties.insert(
            "lhs_name".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some("The name of the left hand side table".to_string()),
                ..Default::default()
            }),
        );
        properties.insert(
            "op_kwargs".to_string(),
            Box::new(JSONSchemaDefine {
                schema_type: Some(JSONSchemaType::String),
                description: Some(
                    "template, table_expression, and input_template in the form of a JSON object"
                        .to_string(),
                ),
                ..Default::default()
            }),
        );
        let function = Function {
            name: Self::get_static_name().to_string(),
            description: Some(self.get_description()),
            parameters: FunctionParameters {
                schema_type: JSONSchemaType::Object,
                properties: Some(properties),
                required: Some(vec!["lhs_name".to_string(), "op_kwargs".to_string()]),
            },
        };
        let tool = Tool {
            r#type: ToolType::Function,
            function,
        };
        serde_json::to_string(&tool).unwrap()
    }
}

impl DataOperatorTrait for Diff {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        diff(
            lhs_args,
            rhs_args.ok_or(anyhow!(
                "Missing `rhs_args` for `{}`.",
                Self::get_static_name()
            ))?,
            &self
                .lhs_values
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            &self
                .rhs_values
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            &self.lhs_pk,
            &self.rhs_pk,
            &self.diff,
            device,
        )
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let lhs_values = config.lhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let rhs_values = config.rhs_values.as_ref().cloned().ok_or(anyhow!(
            "Missing `rhs_values` for `{}`.",
            Self::get_static_name()
        ))?;
        let lhs_pk = config.lhs_pk.as_ref().cloned().ok_or(anyhow!(
            "Missing `lhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;
        let rhs_pk = config.rhs_pk.clone().ok_or(anyhow!(
            "Missing `rhs_pk` for `{}`.",
            Self::get_static_name()
        ))?;
        let diff = config
            .diff
            .ok_or(anyhow!("Missing `diff` for `{}`.", Self::get_static_name()))?;

        if rhs_values.len() != lhs_values.len() {
            return Err(anyhow!(
                "lhs_values length {} is not equal to the rhs_values length {}",
                lhs_values.len(),
                rhs_values.len()
            ));
        }

        // Make the object
        Ok(Diff {
            lhs_values,
            rhs_values,
            lhs_pk,
            rhs_pk,
            diff,
        })
    }
}

/// Helper function to project and transform [RecordBatch]es to a diff-ready [RecordBatch]
pub fn to_diff_record_batches(
    lhs_args: &[RecordBatch],
    lhs_values: &[&str],
    lhs_pk: &str,
    lhs_values_name: &str,
) -> Result<RecordBatch> {
    let lhs_concat = Subject::get_builder()
        .with_name("lhs concat diff")
        .with_record_batches(lhs_args.to_vec())?
        .build()?;
    let lhs_pk_arr = lhs_concat.get_column_as_array(lhs_pk)?;
    let lhs_values_arr = if lhs_values.len() > 1 {
        let lhs_values_str = lhs_concat
            .to_json_object()?
            .into_iter()
            .map(|r| {
                let r_no_pk = r
                    .into_iter()
                    .filter(|(k, _v)| k != lhs_pk)
                    .collect::<Map<_, _>>();
                serde_json::to_string(&r_no_pk).unwrap()
            })
            .collect::<Vec<_>>();
        Arc::new(StringArray::from(lhs_values_str))
    } else {
        lhs_concat.get_column_as_array(lhs_values.first().unwrap())?
    };
    let lhs_args_transformed = RecordBatch::try_from_iter(vec![
        (lhs_pk, lhs_pk_arr),
        (lhs_values_name, lhs_values_arr),
    ])?;
    Ok(lhs_args_transformed)
}

/// Helper function to JSONify the columns of a [RecordBatch]
pub fn to_json_object_columns(lhs_args: RecordBatch, lhs_values: &[&str]) -> Result<RecordBatch> {
    let schema = lhs_args.schema();
    let lhs_other = schema
        .fields()
        .iter()
        .filter_map(|f| {
            if lhs_values.contains(&f.name().as_str()) {
                None
            } else {
                Some(f.name())
            }
        })
        .collect::<Vec<_>>();
    let lhs_args_subject = Subject::get_builder()
        .with_name("patch lhs_args JSONize")
        .with_record_batches(vec![lhs_args])?
        .build()?;
    let lhs_args_other = lhs_args_subject
        .unzip_columns(&lhs_other.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
    let lhs_args_values = lhs_args_subject.unzip_columns(lhs_values)?;
    let lhs_args_values_str = Subject::get_builder()
        .with_name("patch lhs_args JSONize values String")
        .with_record_batches(vec![lhs_args_values])?
        .build()?
        .to_json_object()?
        .into_iter()
        .map(|s| serde_json::to_string(&s).unwrap())
        .collect::<Vec<_>>();
    let lhs_args_values_arr: ArrayRef = Arc::new(StringArray::from(lhs_args_values_str));
    let lhs_args_values_batch =
        RecordBatch::try_from_iter(vec![(lhs_values.first().unwrap(), lhs_args_values_arr)])?;
    let batch = Subject::get_builder()
        .with_name("patch lhs_args zip")
        .with_record_batches(vec![lhs_args_other])?
        .zip_columns(vec![lhs_args_values_batch])?
        .build()?
        .get_record_batches_mut()
        .pop()
        .unwrap();
    Ok(batch)
}

/// Helper function to de-JSONify the columns of a [RecordBatch]
pub fn from_json_object_columns(
    lhs_args: RecordBatch,
    lhs_values: &[&str],
    lhs_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let lhs_concat = Subject::get_builder()
        .with_name("lhs json diff")
        .with_record_batches(vec![lhs_args])?
        .build()?;
    let lhs_concat_schema = lhs_concat.get_schema();
    let lhs_other = lhs_concat_schema
        .fields()
        .iter()
        .filter_map(|f| {
            if lhs_values.contains(&f.name().as_str()) {
                None
            } else {
                Some(f.name())
            }
        })
        .collect::<Vec<_>>();
    let lhs_other_batch =
        lhs_concat.unzip_columns(&lhs_other.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
    let json_values = lhs_concat
        .get_column_as_vec_str(lhs_values.first().unwrap())
        .into_iter()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .collect::<Vec<_>>();
    let json_fields = lhs_schema
        .fields()
        .iter()
        .filter_map(|f| {
            if lhs_values.contains(&f.name().as_str()) {
                Some(f.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let json_schema = Schema::new(json_fields);
    let lhs_columns = lhs_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect::<Vec<_>>();
    let batch = Subject::get_builder()
        .with_name("")
        .with_schema(Arc::new(json_schema))
        .with_json_values(&json_values)?
        .zip_columns(vec![lhs_other_batch])?
        .reorder_columns(&lhs_columns)?
        .build()?
        .get_record_batches_mut()
        .pop()
        .unwrap();
    Ok(batch)
}

/// Generate Diffs/Patches between a LHS [RecordBatch] and RHS [RecordBatch]
///
/// # Notes
///
/// - Each [RecordBatch] is treated as a seperate "workspace" where each row
///   is the equivalent of a "file", and a patch is applied per "file" analogous to `git`
/// - `Create` operations will generate a new row
/// - `Delete` operations will remove a row
/// - `Update` operations will update the row in place
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es that the diff will be computed against (the reference)
/// * `rhs_args` - Slice of [RecordBatch]es that the diff will be computed for (the update)
/// * `lhs_values` - The name of the LHS columns to consider when computing the diff.
/// * `rhs_values` - The name of the RHS columns to consider when computing the diff.
/// * `lhs_pk` - The name of the column with the full path of the "file" or another unique identifier to match the patch on
/// * `rhs_pk` - The name of the column with the full path of the "file" or another unique identifier to match the patch on
/// * `diff` - The [DiffType] to create the diff
/// * `device` - The compute device
#[instrument(skip(
    lhs_args, rhs_args, lhs_values, rhs_values, lhs_pk, rhs_pk, diff, device
))]
#[allow(clippy::too_many_arguments)]
pub fn diff(
    lhs_args: &[RecordBatch],
    rhs_args: &[RecordBatch],
    lhs_values: &[&str],
    rhs_values: &[&str],
    lhs_pk: &str,
    rhs_pk: &str,
    diff: &DiffType,
    device: &Device,
) -> Result<RecordBatch> {
    // Project (and transform to json) the lhs_args and rhs_args
    let lhs_args_transformed = to_diff_record_batches(lhs_args, lhs_values, lhs_pk, "lhs_values")?;
    let rhs_args_transformed = to_diff_record_batches(rhs_args, rhs_values, rhs_pk, "rhs_values")?;

    // FullOuterJoin
    let full_outer_join = join(
        lhs_pk,
        std::slice::from_ref(&lhs_args_transformed),
        rhs_pk,
        std::slice::from_ref(&rhs_args_transformed),
        &DataJoinOperator::FullOuter,
        device,
    )?;

    // Compute the diff
    // Taking advantage of the FullOuterJoin order (Inner -> Update, LeftOuter -> Delete, RightOuter -> Create)
    let lhs_values_col = full_outer_join
        .column_by_name("lhs_values")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or_default())
        .collect::<Vec<_>>();
    let rhs_values_col = full_outer_join
        .column_by_name("rhs_values")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or_default())
        .collect::<Vec<_>>();
    let diff_operator_col: Result<Vec<(usize, String, String)>> = lhs_values_col
        .into_iter()
        .zip(rhs_values_col.into_iter())
        .enumerate()
        .filter_map(|(i, (l, r))| if l == r { None } else { Some((i, l, r)) })
        .map(|(i, l, r)| {
            if l.is_empty() {
                Ok((i, r.to_string(), PatchOperator::Create.to_string()))
            } else if r.is_empty() {
                Ok((i, String::new(), PatchOperator::Delete.to_string()))
            } else {
                let patch = compute_diff(l, r, diff)?;
                Ok((i, patch, PatchOperator::Update.to_string()))
            }
        })
        .collect();
    let ((diff_col, operator_col), indices): ((Vec<String>, Vec<String>), Vec<u32>) =
        diff_operator_col?
            .into_iter()
            .map(|(i, d, o)| ((d, o), i as u32))
            .unzip();

    // Create the record batch
    let diff_arr: ArrayRef = Arc::new(StringArray::from(diff_col));
    let operator_arr: ArrayRef = Arc::new(StringArray::from(operator_col));
    let indices_arr: ArrayRef = Arc::new(UInt32Array::from(indices));
    let lhs_pk_ref: ArrayRef = full_outer_join.column_by_name(lhs_pk).unwrap().to_owned();
    let lhs_pk_arr = arrow::compute::take(&lhs_pk_ref, &indices_arr, None)?;
    let batch = RecordBatch::try_from_iter(vec![
        (lhs_pk, lhs_pk_arr),
        ("diff", diff_arr),
        ("operator", operator_arr),
    ])?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, UInt32Array};

    use crate::device;

    use super::*;

    #[test]
    fn test_diff_all_dmp() -> Result<()> {
        // Create the mock repository
        let repo_pks = vec![0, 1, 2, 3, 4];
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
            r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
            "pub mod extra;",
            r#"mod todo;
pub use todo::Todo"#,
            "pub struct Todo {}",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_pks: ArrayRef = Arc::new(UInt32Array::from(repo_pks));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let repo_batch = RecordBatch::try_from_iter(vec![
            ("repo_pk", repo_pks),
            ("repo_path", repo_paths),
            ("code", code),
        ])?;

        // Create the modified repository
        let repo_pks = vec![0, 3, 4, 2, 5];
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/other.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
            "pub mod other;\nmod todo;\npub use todo::Todo",
            "pub struct Todo {}",
            "pub mod extra;",
            "pub struct Other {}"
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_pks: ArrayRef = Arc::new(UInt32Array::from(repo_pks));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let modified_batch = RecordBatch::try_from_iter(vec![
            ("repo_pk", repo_pks),
            ("repo_path", repo_paths),
            ("code", code),
        ])?;

        // Make the device
        let device = device(false)?;

        // --- PK = String ---
        // Diff the repository
        let result = diff(
            std::slice::from_ref(&repo_batch),
            std::slice::from_ref(&modified_batch),
            &["code"],
            &["code"],
            "repo_path",
            "repo_path",
            &DiffType::Dmp,
            &device,
        )?;

        // Check the results
        let result_table = Subject::get_builder()
            .with_name("test_diff")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_nonprimitive::<String>("repo_path")?;
        assert_eq!(
            test,
            [
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/other.rs",
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("diff")?;
        assert_eq!(
            test,
            [
                "@@ -1,8 +1,23 @@\n+pub mod other;%0A\n mod todo\n",
                "",
                "pub struct Other {}"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("operator")?;
        assert_eq!(test, ["Update", "Delete", "Create"]);

        // --- PK = UInt32 ---
        // Diff the repository
        let result = diff(
            std::slice::from_ref(&repo_batch),
            std::slice::from_ref(&modified_batch),
            &["code"],
            &["code"],
            "repo_pk",
            "repo_pk",
            &DiffType::Dmp,
            &device,
        )?;

        // Check the results
        let result_table = Subject::get_builder()
            .with_name("test_diff")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_primitive::<u32>("repo_pk")?;
        assert_eq!(test, [3, 1, 5]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("diff")?;
        assert_eq!(
            test,
            [
                "@@ -1,8 +1,23 @@\n+pub mod other;%0A\n mod todo\n",
                "",
                "pub struct Other {}"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("operator")?;
        assert_eq!(test, ["Update", "Delete", "Create"]);

        Ok(())
    }

    #[test]
    fn test_diff_all_map() -> Result<()> {
        // Make the test record batches
        let lhs_ids_vec_1 = vec!["0", "1"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_1));
        let lhs_metadata_vec_1: Vec<u32> = vec![1, 2];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_1));
        let lhs_text_vec_1 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_1));
        let lhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_ids_array: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec_2));
        let lhs_metadata_vec_2: Vec<u32> = vec![3, 4];
        let lhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(lhs_metadata_vec_2));
        let lhs_text_vec_2 = vec!["left", "left"];
        let lhs_text_array: ArrayRef = Arc::new(StringArray::from(lhs_text_vec_2));
        let lhs_batch_2 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", lhs_ids_array),
            ("lhs_text", lhs_text_array),
            ("lhs_metadata", lhs_metadata_array),
        ])?;
        let rhs_ids_vec_1 = vec!["0", "1", "2", "4"];
        let rhs_ids_array: ArrayRef = Arc::new(StringArray::from(rhs_ids_vec_1));
        let rhs_metadata_vec_1: Vec<u32> = vec![1, 8, 3, 10];
        let rhs_metadata_array: ArrayRef = Arc::new(UInt32Array::from(rhs_metadata_vec_1));
        let rhs_text_vec_1 = vec!["left", "right", "left", "right"];
        let rhs_text_array: ArrayRef = Arc::new(StringArray::from(rhs_text_vec_1));
        let rhs_batch_1 = RecordBatch::try_from_iter(vec![
            ("lhs_pk", rhs_ids_array),
            ("lhs_text", rhs_text_array),
            ("lhs_metadata", rhs_metadata_array),
        ])?;

        // Make the device
        let device = device(false)?;

        // Diff the record batches
        let result = diff(
            &[lhs_batch_1, lhs_batch_2],
            &[rhs_batch_1],
            &["lhs_text", "lhs_metadata"],
            &["lhs_text", "lhs_metadata"],
            "lhs_pk",
            "lhs_pk",
            &DiffType::Map,
            &device,
        )?;

        // Check the results
        let result_table = Subject::get_builder()
            .with_name("test_diff")
            .with_record_batches(vec![result])?
            .build()?;

        let test = result_table.get_column_as_vec_nonprimitive::<String>("lhs_pk")?;
        assert_eq!(test, ["1", "3", "4"]);
        let test = result_table.get_column_as_vec_nonprimitive::<String>("diff")?;
        assert_eq!(
            test,
            [
                "{\"lhs_metadata\":8,\"lhs_text\":\"right\"}",
                "",
                "{\"lhs_metadata\":10,\"lhs_text\":\"right\"}"
            ]
        );
        let test = result_table.get_column_as_vec_nonprimitive::<String>("operator")?;
        assert_eq!(test, ["Update", "Delete", "Create"]);

        Ok(())
    }
}
