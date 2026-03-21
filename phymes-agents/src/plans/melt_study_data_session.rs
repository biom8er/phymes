use std::fmt::Display;

use anyhow::{Result, anyhow};
use arrow::datatypes::DataType;
use clap::ValueEnum;
use phymes_core::{items_to_list, make_random_id};
use serde::{Deserialize, Serialize};

/// Sample types
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum SampleType {
    #[default]
    #[value(name = "Physical")]
    Physical,
    #[value(name = "Digital")]
    Digital,
}

impl Display for SampleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Physical => write!(f, "Physical"),
            Self::Digital => write!(f, "Digital"),
        }
    }
}

/// A session for melting a `Study Dataset` from a single workflow step
///
/// # Notes
///
/// * Intended to be used with a single workflow step:
///   i.e., usually involving just a single `protocol_id` and `operator_id`
/// * Only the minimal study data is extracted:
///   i.e., Joins between `variable_name` with `ontology_uri` and `units_uri` along along with
///   Joins between `sample_name` and `protocol_id` and `operator_id` are omitted
/// * The `sample_type` is not specified
/// * Adding a unique UUID for samples and variables i.e., `sample_id` and `variable_id` is omitted
///
/// # Todo
///
/// * Support for melting a `Study Dataset` from `StudySamplesMelt` and `StudyVariableMelt` input tables
///   instead of as struct parameters
pub struct MeltStudyDataSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// `sample_name` column name
    /// the `sample_id` column will be automatically generated from a Hash of the `sample_name`
    pub sample_name_col: &'a str,
    /// `study_id`, if None then a unique UInt32 ID will be provided
    pub study_id: u32,
    /// Variable or Feature names
    pub variable_names: &'a [&'a str],
    /// Variable or Feature data types
    pub data_types: &'a [DataType],
}

impl<'a> MeltStudyDataSession<'a> {
    /// New [MeltStudyDataSession]
    pub fn new(
        session_context_name: Option<&'a str>,
        sample_name_col: &'a str,
        study_id: Option<u32>,
        variable_names: &'a [&'a str],
        data_types: &'a [DataType],
    ) -> Result<Self> {
        let session_context_name = session_context_name.unwrap_or("melt_study_data_session");
        if variable_names.len() != data_types.len() {
            return Err(anyhow!(
                "variable_names `{variable_names:?}` length does not match data_types `{data_types:?}` length."
            ));
        }
        let study_id = if let Some(study_id) = study_id {
            study_id
        } else {
            make_random_id()? as u32
        };
        Ok(Self {
            session_context_name,
            sample_name_col,
            study_id,
            variable_names,
            data_types,
        })
    }

    /// Make the variable columns
    fn variable_columns(&self) -> Result<String> {
        items_to_list(self.variable_names)
    }

    /// Make the datatype columns
    fn data_type_columns(&self) -> Result<String> {
        let items = self
            .data_types
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Make cast operator columns for variables
    fn cast_operator_columns(&self) -> Result<String> {
        let items = self
            .data_types
            .iter()
            .map(|_| "Cast".to_string())
            .collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Make cast template columns for variables
    fn cast_templates_columns(&self) -> Result<String> {
        let items = self
            .data_types
            .iter()
            .map(|_| "".to_string())
            .collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// make column operator columns for variables
    fn column_operators_columns(&self) -> Result<String> {
        let items = self
            .data_types
            .iter()
            .map(|_| "None".to_string())
            .collect::<Vec<_>>();
        items_to_list(&items.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    }

    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD	
	%% ---------------------------------
	%% Study data extraction
	%% ---------------------------------
	subgraph study_data_extraction
	    UserCsv-subject-.->|LastRecordBatch|user_csv-subscribe
	    user_csv-subscribe-->user_csv-processor
	    user_csv-processor-->user_csv-publish
	    user_csv-publish-->|Replace|StudyData-subject
	    StudyData-subject-->|AllRecordBatches|study_data_cast-subscribe
	    study_data_cast-subscribe-->study_data_cast-processor
	    study_data_cast-processor-->study_data_cast-publish
	    study_data_cast-publish-->|Replace|StudyDataCast-subject
	    StudyDataCast-subject-->|AllRecordBatches|study_data_melt-subscribe
	    study_data_melt-subscribe-->study_data_melt-processor
	    study_data_melt-processor-->study_data_melt-publish
	    study_data_melt-publish-->|Replace|StudyDataMelt-subject
	end
	study_data_extraction-rt@{shape: subproc, label: study_data_extraction}
	study_data_extraction-rt-->study_data_extraction
	UserCsv-subject@{shape: doc, label: UserCsv}
	user_csv-processor@{shape: rect, label: ExtractTabular}
	user_csv-publish@{shape: fork}
	user_csv-subscribe@{shape: diamond, label: All}
	StudyData-subject@{shape: doc, label: StudyData}
	study_data_cast-processor@{shape: rect, label: Select}
	study_data_cast-publish@{shape: fork}
	study_data_cast-subscribe@{shape: diamond, label: All}
	StudyDataCast-subject@{shape: doc, label: StudyDataCast}
	study_data_melt-processor@{shape: rect, label: Melt}
	study_data_melt-publish@{shape: fork}
	study_data_melt-subscribe@{shape: diamond, label: All}
	StudyDataMelt-subject@{shape: doc, label: StudyDataMelt}
	%% ---------------------------------
	%% Study samples extraction
	%% ---------------------------------
	subgraph study_samples_extraction
	    UserCsv-subject-.->|LastRecordBatch|user_csv-subscribe
	    user_csv-subscribe-->user_csv-processor
	    user_csv-processor-->user_csv-publish
	    user_csv-publish-->|Replace|StudyData-subject
	    StudyData-subject-->|AllRecordBatches|study_samples_select-subscribe
	    study_samples_select-subscribe-->study_samples_select-processor
	    study_samples_select-processor-->study_samples_select-publish
	    study_samples_select-publish-->|Extend|StudySamplesMelt-subject
	end
	study_samples_extraction-rt@{shape: subproc, label: study_samples_extraction}
	study_samples_extraction-rt-->study_samples_extraction
	study_samples_select-processor@{shape: rect, label: Select}
	study_samples_select-publish@{shape: fork}
	study_samples_select-subscribe@{shape: diamond, label: All}
	StudySamplesMelt-subject@{shape: doc, label: StudySamplesMelt}
	%% ---------------------------------
	%% Samples Variables extraction
	%% ---------------------------------
	subgraph samples_variables_extraction
	    StudyDataMelt-subject-.->|AllRecordBatches|samples_variables_select-subscribe
	    samples_variables_select-subscribe-->samples_variables_select-processor
	    samples_variables_select-processor-->samples_variables_select-publish
	    samples_variables_select-publish-->|Replace|SamplesVariablesMelt-subject
	end
	samples_variables_extraction-rt@{shape: subproc, label: samples_variables_extraction}
	samples_variables_extraction-rt-->samples_variables_extraction
	samples_variables_select-processor@{shape: rect, label: Select}
	samples_variables_select-publish@{shape: fork}
	samples_variables_select-subscribe@{shape: diamond, label: All}
	SamplesVariablesMelt-subject@{shape: doc, label: SamplesVariablesMelt}
	%% ---------------------------------
	%% Study variables extraction
	%% TODO: need to add coalesce step because group by runs out of GPU memory
	%% ---------------------------------
	subgraph study_variables_extraction
	    StudyDataMelt-subject-.->|AllRecordBatches|study_variables_group_by-subscribe
	    study_variables_group_by-subscribe-->study_variables_group_by-processor
	    study_variables_group_by-processor-->study_variables_group_by-publish
	    study_variables_group_by-publish-->|Replace|StudyVariablesMeltGroupBy-subject
	    StudyVariablesMeltGroupBy-subject-->|AllRecordBatches|study_variables_select-subscribe
	    study_variables_select-subscribe-->study_variables_select-processor
	    study_variables_select-processor-->study_variables_select-publish
	    study_variables_select-publish-->|Replace|StudyVariablesMelt-subject
	end
	study_variables_extraction-rt@{shape: subproc, label: study_variables_extraction}
	study_variables_extraction-rt-->study_variables_extraction
	study_variables_group_by-processor@{shape: rect, label: GroupBy}
	study_variables_group_by-publish@{shape: fork}
	study_variables_group_by-subscribe@{shape: diamond, label: All}
	StudyVariablesMeltGroupBy-subject@{shape: doc, label: StudyVariablesMeltGroupBy}
	study_variables_select-processor@{shape: rect, label: Select}
	study_variables_select-publish@{shape: fork}
	study_variables_select-subscribe@{shape: diamond, label: All}
	StudyVariablesMelt-subject@{shape: doc, label: StudyVariablesMelt}
	%% ---------------------------------"#
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> Result<String> {
        let erdiagram = format!(
            r#"erDiagram
	study_data_cast["study_data_cast"] {{
	    List-Utf8 as_columns "['sample_name',{},'study_id']"
	    List-Utf8 cast_datatypes "['Utf8',{},'UInt32']"
	    List-Utf8 cast_operators "['Cast',{},'None']"
	    List-Utf8 cast_templates "['',{},'{}']"
	    List-Utf8 column_operators "['None',{},'Value']"
	    List-Utf8 lhs_values "['{}',{},'study_id']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyData"
	    Utf8 operator "Select"
	    Utf8 lhs_stream "Accumulate"
	}}
	study_data_melt["study_data_melt"] {{
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyDataCast"
	    List-Utf8 lhs_values "['study_id','sample_name']"
	    Utf8 operator "Melt"
	    List-Utf8 pvt_columns "[{}]"
	    Utf8 lhs_stream "Accumulate"
	}}
	StudyDataMelt["StudyDataMelt"] {{
	    UInt32 study_id
	    Utf8 sample_name
	    Utf8 variable
	    Utf8 value
	    Utf8 data_type
	}}
	study_samples_select["study_samples_select"] {{
	    List-Utf8 as_columns "['sample_name','']"
	    List-Utf8 cast_datatypes "['Utf8','UInt32']"
	    List-Utf8 cast_operators "['Cast','None']"
	    List-Utf8 cast_templates "['','{}']"
	    List-Utf8 column_operators "['None','Value']"
	    List-Utf8 lhs_values "['{}','study_id']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyData"
	    Utf8 operator "Select"
	    Utf8 lhs_stream "Accumulate"
	}}
	StudySamplesMelt["StudySamplesMelt"] {{
	    Utf8 sample_name
	    UInt32 study_id
	}}
	samples_variables_select["samples_variables_select"] {{
	    List-Utf8 as_columns "['','variable_name','','']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyDataMelt"
	    List-Utf8 lhs_values "['sample_name','variable','value','study_id']"
	    Utf8 operator "Select"
	    List-Utf8 rhs_values "['','','','']"
	    Utf8 lhs_stream "Accumulate"
	}}
	SamplesVariablesMelt["SamplesVariablesMelt"] {{
	    Utf8 sample_name
	    Utf8 variable_name
	    Utf8 value
	    UInt32 study_id
	}}
	study_variables_group_by["study_variables_group_by"] {{
	    List-Utf8 agg_columns "['data_type']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "true"
	    Utf8 lhs_name "StudyDataMelt"
	    List-Utf8 lhs_values "['study_id','variable']"
	    Utf8 operator "GroupBy"
	    Utf8 lhs_stream "Accumulate"
	}}
	study_variables_select["study_variables_select"] {{
	    List-Utf8 as_columns "['variable_name','data_type','']"
	    List-Utf8 lhs_values "['variable','data_type-First','study_id']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyVariablesMeltGroupBy"
	    Utf8 operator "Select"
	    Utf8 lhs_stream "Accumulate"
	}}
	StudyVariablesMelt["StudyVariablesMelt"] {{
	    Utf8 variable_name
	    Utf8 data_type
	    UInt32 study_id
	}}
	UserCsv["UserCsv"] {{
	    Utf8 filename
	    Utf8 extension
	    List-UInt8 bytes
	    Utf8 metadata
	    Int64 timestamp
	}}
	user_csv["user_csv"] {{
	    Boolean cpu "false"
	    Utf8 format "CsvDefault"
        Utf8 schema "Attachments"
	    Utf8 lhs_name "UserCsv"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractTabular"
        Utf8 encoding "None"
	    Utf8 lhs_stream "Accumulate"
	}}"#,
            self.cast_templates_columns()?,
            self.data_type_columns()?,
            self.cast_operator_columns()?,
            self.cast_templates_columns()?,
            self.study_id,
            self.column_operators_columns()?,
            self.sample_name_col,
            self.variable_columns()?,
            self.variable_columns()?,
            self.study_id,
            self.sample_name_col,
        );
        Ok(erdiagram)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        AttachmentBuilderTraitExt, AvailableSubjectsTrait, BuildableTrait, BuilderTrait, CsvFormat,
        IPCMessage, MappableTrait, MessageBuilderTrait, Subject, SubjectBuilderTrait, Publication,
        SubjectTrait,
    };
    use phymes_diagnostics::HashMap;

    use crate::{
        AvailableInterfaceSubjects, SessionContextBuilder, SessionContextBuilderAgentsTrait,
        SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream,
        create_message_map,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_melt_study_data_session() -> Result<()> {
        // Make the anticipated pivot table values
        let variable_names = &["Age", "Gender", "Ethnicity", "RFFT", "VAT", "BMI", "Statin"];
        let data_types = &[
            DataType::Int64,
            DataType::Int64,
            DataType::Int64,
            DataType::Int64,
            DataType::Int64,
            DataType::Float64,
            DataType::Int64,
        ];

        // Initialize the session
        let melt_study_data_session =
            MeltStudyDataSession::new(None, "Casenr", None, variable_names, data_types)?;
        // dbg!(&melt_study_data_session.as_mermaid_erdiagram()?);
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            melt_study_data_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            melt_study_data_session.as_mermaid_erdiagram()?.as_str(),
            false,
            true,
        )?
        .with_name(melt_study_data_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the tabular data
        let csv_format = CsvFormat::default();
        let sample_names = [
            "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let ages = vec![82, 82, 82, 82, 82, 82, 82, 82];
        let genders = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let ethnicities = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let rffs = vec![52, 40, 53, 33, 47, 35, 67, 25];
        let vats = vec![9, 11, 4, 10, 11, 7, 6, 10];
        let bmis = vec![
            31.8734311,
            24.25867407,
            26.0932752,
            26.3958034,
            23.70110632,
            30.54380794,
            26.0261749,
            26.72929708,
        ];
        let statins = vec![1, 1, 0, 1, 1, 1, 0, 0];
        let sample_names: ArrayRef = Arc::new(StringArray::from(sample_names));
        let ages: ArrayRef = Arc::new(Int64Array::from(ages));
        let genders: ArrayRef = Arc::new(Int64Array::from(genders));
        let ethnicities: ArrayRef = Arc::new(Int64Array::from(ethnicities));
        let rffs: ArrayRef = Arc::new(Int64Array::from(rffs));
        let vats: ArrayRef = Arc::new(Int64Array::from(vats));
        let bmis: ArrayRef = Arc::new(Float64Array::from(bmis));
        let statins: ArrayRef = Arc::new(Int64Array::from(statins));
        let batch = RecordBatch::try_from_iter(vec![
            ("Casenr", sample_names),
            ("Age", ages),
            ("Gender", genders),
            ("Ethnicity", ethnicities),
            ("RFFT", rffs),
            ("VAT", vats),
            ("BMI", bmis),
            ("Statin", statins),
        ])?;
        let table = Subject::get_builder()
            .with_name("PivotTable")
            .with_record_batches(vec![batch])?
            .build()?;
        let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
        let blob = AvailableInterfaceSubjects::UserCsv
            .to_subject_builder(None)
            .with_attachment(None, Some("csv"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&Publication::Extend {
                subject_name: blob.get_name().to_string(),
            })
            .with_publisher(melt_study_data_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![blob_message]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test session context
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .subjects()
                .get("StudySamplesMelt")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("sample_name");
            assert_eq!(
                column,
                [
                    "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095"
                ]
            );
            let column = table_reading.get_column_as_vec_primitive::<u32>("study_id")?;
            assert!(!column.is_empty());

            let table_reading = session_reading
                .subjects()
                .get("SamplesVariablesMelt")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("sample_name");
            assert_eq!(
                column,
                [
                    "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089",
                    "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091",
                    "4092", "4093", "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093",
                    "4094", "4095", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095",
                    "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4088", "4089",
                    "4090", "4091", "4092", "4093", "4094", "4095"
                ]
            );
            let column = table_reading.get_column_as_vec_str("variable_name");
            assert_eq!(
                column,
                [
                    "Age",
                    "Age",
                    "Age",
                    "Age",
                    "Age",
                    "Age",
                    "Age",
                    "Age",
                    "Gender",
                    "Gender",
                    "Gender",
                    "Gender",
                    "Gender",
                    "Gender",
                    "Gender",
                    "Gender",
                    "Ethnicity",
                    "Ethnicity",
                    "Ethnicity",
                    "Ethnicity",
                    "Ethnicity",
                    "Ethnicity",
                    "Ethnicity",
                    "Ethnicity",
                    "RFFT",
                    "RFFT",
                    "RFFT",
                    "RFFT",
                    "RFFT",
                    "RFFT",
                    "RFFT",
                    "RFFT",
                    "VAT",
                    "VAT",
                    "VAT",
                    "VAT",
                    "VAT",
                    "VAT",
                    "VAT",
                    "VAT",
                    "BMI",
                    "BMI",
                    "BMI",
                    "BMI",
                    "BMI",
                    "BMI",
                    "BMI",
                    "BMI",
                    "Statin",
                    "Statin",
                    "Statin",
                    "Statin",
                    "Statin",
                    "Statin",
                    "Statin",
                    "Statin"
                ]
            );
            let column = table_reading.get_column_as_vec_str("value");
            assert_eq!(
                column,
                [
                    "82",
                    "82",
                    "82",
                    "82",
                    "82",
                    "82",
                    "82",
                    "82",
                    "0",
                    "0",
                    "0",
                    "0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "52",
                    "40",
                    "53",
                    "33",
                    "47",
                    "35",
                    "67",
                    "25",
                    "9",
                    "11",
                    "4",
                    "10",
                    "11",
                    "7",
                    "6",
                    "10",
                    "31.8734311",
                    "24.25867407",
                    "26.0932752",
                    "26.3958034",
                    "23.70110632",
                    "30.54380794",
                    "26.0261749",
                    "26.72929708",
                    "1",
                    "1",
                    "0",
                    "1",
                    "1",
                    "1",
                    "0",
                    "0"
                ]
            );
            let column = table_reading.get_column_as_vec_primitive::<u32>("study_id")?;
            assert!(!column.is_empty());

            let table_reading = session_reading
                .subjects()
                .get("StudyVariablesMelt")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("variable_name");
            assert_eq!(
                column,
                ["Age", "BMI", "Ethnicity", "Gender", "RFFT", "Statin", "VAT"]
            );
            let column = table_reading.get_column_as_vec_str("data_type");
            assert_eq!(
                column,
                [
                    "Int64", "Float64", "Int64", "Int64", "Int64", "Int64", "Int64"
                ]
            );
            let column = table_reading.get_column_as_vec_primitive::<u32>("study_id")?;
            assert!(!column.is_empty());
        }

        Ok(())
    }
}
