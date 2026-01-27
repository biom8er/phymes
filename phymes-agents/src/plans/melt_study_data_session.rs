use anyhow::Result;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, IPCMessageMap,
    MessageBuilderTrait, Table, TableBuilderTrait, TablePublication, TableTrait,
    create_session_tasks_subscribe_publish_batch,
};

use crate::create_message_map;

/// A session for determining the next superstep task publications and subscriptions
pub struct MeltStudyDataSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// `sample_name` column name
    pub sample_name_col: &'a str,
    /// `study_id` column name
    pub study_id_col: Option<&'a str>,
    /// Variable or Feature names
    pub variable_names: &'a[&'a str],
    /// Variable or Feature data types
    pub data_types: &'a[&'a str],
    /// Variable or Feature ontology URIs
    pub uris: Option<&'a[&'a str]>,
    /// Variable or Feature unit ontology URIs
    pub units_uris: Option<&'a[&'a str]>,
}

impl Default for NextSuperstepSession<'_> {
    fn default() -> Self {
        NextSuperstepSession {
            session_context_name: "melt_study_data_session",
        }
    }
}

impl<'a> NextSuperstepSession<'a> {
    /// New
    pub fn new(session_context_name: &str, sample_name_col: &str, study_id_col: Option<&str>, variable_names: &[&str], data_types: &[&str], uris: Option<&[&str]>, units_uris: Option<&[&str]>) -> Self {
        Self {
            session_context_name,
            sample_name_col,
            study_id_col,
            variable_names,
            data_types,
            uris,
            units_uris,
        }
    }
    /// Return the pre-compiled task subscriptions and publications as messages
    pub fn as_task_messages(&self) -> Result<Vec<IPCMessageMap>> {
        // 1. Message to trigger the first superstep
        let task_names = vec!["max_superstep_t"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["group_by_session_superstep_p"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![vec!["OnUpdateFullTable", "AlwaysFullTable"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let subscription_table_names =
            vec![vec!["SessionSupersteps", "group_by_session_superstep_p"]]
                .into_iter()
                .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["SessionSuperstepMax"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| self.session_context_name.to_string())
            .collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Table::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(self.session_context_name)
            .make_name()?
            .build()?;
        let messages_1 = create_message_map(vec![tasks_publish_subscribe_message]);

        Ok(vec![messages_1])
    }

    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD	
	%% Study data extraction
	%% ---------------------------------
	subgraph study_data_extraction
	    UserCsv-subject-.->|LastRecordBatch|user_csv-subscribe
	    user_csv-subscribe-->user_csv-processor
	    user_csv-processor-->user_csv-publish
	    user_csv-publish-->|Replace|StudyData-subject
	    StudyData-subject-->|FullTable|study_data_cast-subscribe
	    study_data_cast-subscribe-->study_data_cast-processor
	    study_data_cast-processor-->study_data_cast-publish
	    study_data_cast-publish-->|Replace|StudyDataCast-subject
	    StudyDataCast-subject-->|FullTable|study_data_melt-subscribe
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
	    StudyData-subject-->|FullTable|study_samples_cast-subscribe
	    study_samples_cast-subscribe-->study_samples_cast-processor
	    study_samples_cast-processor-->study_samples_cast-publish
	    study_samples_cast-publish-->|Replace|StudySamplesCast-subject
	    StudySamplesCast-subject-->|FullTable|study_samples_select-subscribe
	    study_samples_select-subscribe-->study_samples_select-processor
	    study_samples_select-processor-->study_samples_select-publish
	    study_samples_select-publish-->|Extend|StudySamples-subject
	end
	study_samples_extraction-rt@{shape: subproc, label: study_samples_extraction}
	study_samples_extraction-rt-->study_samples_extraction
	study_samples_cast-processor@{shape: rect, label: Select}
	study_samples_cast-publish@{shape: fork}
	study_samples_cast-subscribe@{shape: diamond, label: All}
	StudySamplesCast-subject@{shape: doc, label: StudySamplesCast}
	study_samples_select-processor@{shape: rect, label: Select}
	study_samples_select-publish@{shape: fork}
	study_samples_select-subscribe@{shape: diamond, label: All}
	StudySamples-subject@{shape: doc, label: StudySamples}
	%% ---------------------------------
	
	%% Study variables extraction
	%% TODO: need to add coalesce step because filter runs out of GPU memory
	%% ---------------------------------
	subgraph study_variables_extraction
	    StudyDataMelt-subject-.->|FullTable|study_variables_cmp-subscribe
	    study_variables_cmp-subscribe-->study_variables_cmp-processor
	    study_variables_cmp-processor-->study_variables_cmp-publish
	    study_variables_cmp-publish-->|Replace|StudyVariablesCmp-subject
	    StudyVariablesCmp-subject-->|FullTable|study_variables_filter-subscribe
	    study_variables_filter-subscribe-->study_variables_filter-processor
	    study_variables_filter-processor-->study_variables_filter-publish
	    study_variables_filter-publish-->|Replace|StudyVariablesFiltered-subject
	    StudyVariablesFiltered-subject-->|FullTable|study_variables_select-subscribe
	    study_variables_select-subscribe-->study_variables_select-processor
	    study_variables_select-processor-->study_variables_select-publish
	    study_variables_select-publish-->|Replace|StudyVariablesSelect-subject
	end
	study_variables_extraction-rt@{shape: subproc, label: study_variables_extraction}
	study_variables_extraction-rt-->study_variables_extraction
	study_variables_cmp-processor@{shape: rect, label: Select}
	study_variables_cmp-publish@{shape: fork}
	study_variables_cmp-subscribe@{shape: diamond, label: All}
	StudyVariablesCmp-subject@{shape: doc, label: StudyVariablesCmp}
	study_variables_filter-processor@{shape: rect, label: Filter}
	study_variables_filter-publish@{shape: fork}
	study_variables_filter-subscribe@{shape: diamond, label: All}
	StudyVariablesFiltered-subject@{shape: doc, label: StudyVariablesFiltered}
	study_variables_select-processor@{shape: rect, label: Select}
	study_variables_select-publish@{shape: fork}
	study_variables_select-subscribe@{shape: diamond, label: All}
	StudyVariablesSelect-subject@{shape: doc, label: StudyVariablesSelect}
	%% ---------------------------------
	%% To samples variables
	%% ---------------------------------
	subgraph to_sample_variables
	    StudyVariablesSelect-subject-.->|FullTable|samples_variables_select-subscribe
	    samples_variables_select-subscribe-->samples_variables_select-processor
	    samples_variables_select-processor-->samples_variables_select-publish
	    samples_variables_select-publish-->|Replace|SamplesVariables-subject
	end
	to_sample_variables-rt@{shape: subproc, label: to_sample_variables}
	to_sample_variables-rt-->to_sample_variables
	samples_variables_select-processor@{shape: rect, label: Select}
	samples_variables_select-publish@{shape: fork}
	samples_variables_select-subscribe@{shape: diamond, label: All}
	SamplesVariables-subject@{shape: doc, label: SamplesVariables}
	%% ---------------------------------
	%% To study variables
	%% TODO: need to add coalesce step because group by runs out of GPU memory
	%% ---------------------------------
	subgraph to_study_variables
	    StudyVariablesSelect-subject-.->|FullTable|study_variables_group_by-subscribe
	    study_variables_group_by-subscribe-->study_variables_group_by-processor
	    study_variables_group_by-processor-->study_variables_group_by-publish
	    study_variables_group_by-publish-->|Replace|StudyVariablesGroupBy-subject
	    StudyVariablesGroupBy-subject-->|FullTable|study_variables_select_2-subscribe
	    study_variables_select_2-subscribe-->study_variables_select_2-processor
	    study_variables_select_2-processor-->study_variables_select_2-publish
	    study_variables_select_2-publish-->|Replace|StudyVariables-subject
	end
	to_study_variables-rt@{shape: subproc, label: to_study_variables}
	to_study_variables-rt-->to_study_variables
	study_variables_group_by-processor@{shape: rect, label: GroupBy}
	study_variables_group_by-publish@{shape: fork}
	study_variables_group_by-subscribe@{shape: diamond, label: All}
	StudyVariablesGroupBy-subject@{shape: doc, label: StudyVariablesGroupBy}
	study_variables_select_2-processor@{shape: rect, label: Select}
	study_variables_select_2-publish@{shape: fork}
	study_variables_select_2-subscribe@{shape: diamond, label: All}
	StudyVariables-subject@{shape: doc, label: StudyVariables}
	%% ---------------------------------"#
    }

    /// Return the Mermaid.js ER Diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
	study_data_cast["study_data_cast"] {
	    List-Utf8 as_columns "['sample_name','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','sample_id']"
	    List-Utf8 cast_datatypes "['Utf8','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Float64','Float64','Float64','Float64','Float64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','Int64','UInt32','UInt32']"
	    List-Utf8 cast_operators "['Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','Cast','None','Hash']"
	    List-Utf8 cast_templates "['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','0','']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None']"
	    List-Utf8 rhs_values "['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','']"
	    List-Utf8 lhs_values "['Casenr','Age','Gender','Ethnicity','Education','RFFT','VAT','CVD','DM','Smoking','Hypertension','BMI','SBP','DBP','MAP','eGFR','Albuminuria_1','Albuminuria_2','Chol','HDL','Statin','Solubility','Days','Years','DDD','FRS','PS','PSquint','GRS','Match_1','Match_2','study_id','sample_name']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyData"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_data_melt["study_data_melt"] {
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyDataCast"
	    List-Utf8 lhs_values "['sample_id','study_id','sample_name']"
	    Utf8 operator "Melt"
	    List-Utf8 pvt_columns "['Age','Gender','Ethnicity','Education','RFFT','VAT','CVD','DM','Smoking','Hypertension','BMI','SBP','DBP','MAP','eGFR','Albuminuria_1','Albuminuria_2','Chol','HDL','Statin','Solubility','Days','Years','DDD','FRS','PS','PSquint','GRS','Match_1','Match_2']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	StudyDataMelt["StudyDataMelt"] {
	    UInt32 sample_id
	    UInt32 study_id
	    Utf8 sample_name
	    Utf8 variable
	    Utf8 value
	    Utf8 data_type
	}
	study_samples_cast["study_samples_cast"] {
	    List-Utf8 as_columns "['sample_name','','id','','','']"
	    List-Utf8 cast_datatypes "['Utf8','UInt32','UInt32','UInt32','UInt32','Utf8']"
	    List-Utf8 cast_operators "['Cast','None','Hash','None','None','None']"
	    List-Utf8 cast_templates "['','0','','0','0','Physical']"
	    List-Utf8 column_operators "['None','None','None','None','None','None']"
	    List-Utf8 rhs_values "['','','','','','']"
	    List-Utf8 lhs_values "['Casenr','study_id','sample_name','protocol_id','operator_id','type']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyData"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_samples_select["study_samples_select"] {
	    List-Utf8 as_columns "['','','','','','']"
	    List-Utf8 cast_datatypes "['UInt32','Utf8','UInt32','UInt32','UInt32','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None','None','None']"
	    List-Utf8 cast_templates "['','','','','','']"
	    List-Utf8 column_operators "['None','None','None','None','None','None']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudySamplesCast"
	    List-Utf8 lhs_values "['id','sample_name','study_id','protocol_id','operator_id','type']"
	    Utf8 operator "Select"
	    List-Utf8 rhs_values "['','','','','','']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_variables_cmp["study_variables_cmp"] {
	    List-Utf8 as_columns "['','','','','','','Age','Gender','Ethnicity','Education','RFFT','VAT','CVD','DM','Smoking','Hypertension','BMI','SBP','DBP','MAP','eGFR','Albuminuria_1','Albuminuria_2','Chol','HDL','Statin','Solubility','Days','Years','DDD','FRS','PS','PSquint','GRS','Match_1','Match_2']"
	    List-Utf8 cast_datatypes "['UInt32','Utf8','UInt32','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None']"
	    List-Utf8 cast_templates "['','','','','','','Age','Gender','Ethnicity','Education','RFFT','VAT','CVD','DM','Smoking','Hypertension','BMI','SBP','DBP','MAP','eGFR','Albuminuria_1','Albuminuria_2','Chol','HDL','Statin','Solubility','Days','Years','DDD','FRS','PS','PSquint','GRS','Match_1','Match_2']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None','None']"
	    List-Utf8 rhs_values "['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','']"
	    List-Utf8 lhs_values "['sample_id','sample_name','study_id','variable','value','data_type','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name','sample_name']"
	    Boolean cpu "true"
	    Utf8 lhs_name "StudyDataMelt"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_variables_filter["study_variables_filter"] {
	    List-Utf8 cmp_columns "['Age','Gender','Ethnicity','Education','RFFT','VAT','CVD','DM','Smoking','Hypertension','BMI','SBP','DBP','MAP','eGFR','Albuminuria_1','Albuminuria_2','Chol','HDL','Statin','Solubility','Days','Years','DDD','FRS','PS','PSquint','GRS','Match_1','Match_2']"
	    List-Utf8 cmp_operators "['Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like','Like']"
	    Utf8 cmp_predicate "Any"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyVariablesCmp"
	    List-Utf8 lhs_values "['variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable','variable']"
	    Utf8 operator "Filter"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_variables_select["study_variables_select"] {
	    List-Utf8 as_columns "['','variable_id','','','variable_name','','']"
	    List-Utf8 cast_datatypes "['UInt32','UInt32','UInt32','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','Hash','None','None','None','None','None']"
	    List-Utf8 cast_templates "['','','','','','','']"
	    List-Utf8 column_operators "['None','None','None','None','None','None','None']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyVariablesFiltered"
	    List-Utf8 lhs_values "['sample_id','variable','study_id','sample_name','variable','value','data_type']"
	    Utf8 operator "Select"
	    List-Utf8 rhs_values "['','','','','','','']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	StudyVariablesSelect["StudyVariablesSelect"] {
	    UInt32 sample_id
	    UInt32 variable_id
	    UInt32 study_id
	    Utf8 sample_name
	    Utf8 variable_name
	    Utf8 value
	    Utf8 data_type
	}
	samples_variables_select["samples_variables_select"] {
	    List-Utf8 as_columns "['','','','']"
	    List-Utf8 cast_datatypes "['UInt32','UInt32','UInt32','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None']"
	    List-Utf8 cast_templates "['','','','']"
	    List-Utf8 column_operators "['None','None','None','None']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyVariablesSelect"
	    List-Utf8 lhs_values "['study_id','sample_id','variable_id','value']"
	    Utf8 operator "Select"
	    List-Utf8 rhs_values "['','','','']"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_variables_group_by["study_variables_group_by"] {
	    List-Utf8 agg_columns "['data_type']"
	    List-Utf8 agg_operators "['First']"
	    Boolean cpu "true"
	    Utf8 lhs_name "StudyVariablesSelect"
	    List-Utf8 lhs_values "['study_id','variable_name','variable_id']"
	    Utf8 operator "GroupBy"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	study_variables_select_2["study_variables_select_2"] {
	    List-Utf8 as_columns "['id','study_id','','','data_type','']"
	    List-Utf8 cast_datatypes "['UInt32','UInt32','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_operators "['None','None','None','None','None','None']"
	    List-Utf8 cast_templates "['','','','todo','','todo']"
	    List-Utf8 column_operators "['None','None','None','None','None','None']"
	    List-Utf8 rhs_values "['','','','','','']"
	    List-Utf8 lhs_values "['variable_id','study_id','variable_name','uri','data_type-First','units_uri']"
	    Boolean cpu "false"
	    Utf8 lhs_name "StudyVariablesGroupBy"
	    Utf8 operator "Select"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	UserCsv["UserCsv"] {
	    Utf8 filename
	    Utf8 extension
	    List-UInt8 bytes
	    Utf8 metadata
	    Int64 timestamp
	}
	user_csv["user_csv"] {
	    Boolean cpu "false"
	    Utf8 format "CsvDefault"
	    Utf8 lhs_name "UserCsv"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractTabular"
	    Utf8 stream "AccumulateLHSAccumulateRHS"
	}
	StudyVariables ||--|{ Study : "is part of"
	StudyVariables["StudyVariables"] {
	    UInt32 id
	    UInt32 study_id
	    Utf8 variable_name
	    Utf8 uri
	    Utf8 data_type
	    Utf8 units_uri
	}
	StudySamples ||--|{ Study : "is part of"
	StudySamples ||--|{ StudySourceMaterial : "is derived from"
	StudySamples ||--|{ StudyProtocols : "is derived from"
	StudySamples ||--|{ StudyVariables : "is derived from"
	StudySamples ||--|{ StudyAuthors : "is derived from"
	StudySamples ||--|{ StudySamples : "is derived from"
	StudySamples["StudySamples"] {
	    UInt32 id
	    Utf8 sample_name
	    UInt32 study_id
	    UInt32 protocol_id
	    UInt32 operator_id
	    Utf8 type
	}
	SamplesVariables ||--|{ Study : "is part of"
	SamplesVariables ||--|{ StudySamples: "is derived from"
	SamplesVariables ||--|{ StudyVariables : "is derived from"
	SamplesVariables["SamplesVariables"] {
	    UInt32 study_id
	    UInt32 sample_id
	    UInt32 variable_id
	    Utf8 value
	}"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, MessageBuilderTrait,
        TablePublication, TableTrait, create_session_supersteps_batch,
    };
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, create_message_map,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_melt_study_data_session() -> Result<()> {
        // Initialize the session
        let melt_study_data_session = NextSuperstepSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            melt_study_data_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            melt_study_data_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(melt_study_data_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test session data
        let session_names = ["session_1", "session_1", "session_1", "session_1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let supersteps = vec![0, 1, 2, 3];
        let batch = create_session_supersteps_batch(session_names, supersteps)?;
        let table = Table::get_builder()
            .with_name(AvailableSubjects::SessionSupersteps.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let superstep_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableSubjects::SessionSupersteps.to_string().as_str())
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionSupersteps.to_string(),
            })
            .with_publisher(melt_study_data_session.session_context_name)
            .make_name()?
            .build()?;
        let mut message_map = create_message_map(vec![superstep_message]);

        // Session Tasks
        let mut next_superstep_messages = melt_study_data_session
            .as_task_messages()?
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        // Run the session
        message_map.extend(next_superstep_messages.pop().unwrap());
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supserstep 1
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableSubjects::SessionSuperstepMax.to_string().as_str())
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["session_1"]);
            let column = table_reading.get_column_as_vec_primitive::<u32>("superstep-Max")?;
            assert_eq!(column, [3]);
        }

        Ok(())
    }
}
