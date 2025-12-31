/// A session for all subject associated tasks
///
/// # Notes
///
/// * Supported tasks include the following:
///
/// 1. Counting the number of rows per subject (i.e., updating the `SubjectNumRows` table)
///   after updates have been made to the `SubjectsChangeLog`
/// 2. Determining what tasks are ready to run for the next super step
/// 3. Retrieving the publications per task and processor that will run for the next super step
/// 4. Updating the `SubjectsChangeLog` cache with the most recent updates and `TasksRunLog` cache with the most recent task runs
///
/// * Caching is implemented to minimize memory and compute
pub struct SubjectsSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl Default for SubjectsSession<'_> {
    fn default() -> Self {
        SubjectsSession {
            session_context_name: "subject_session",
        }
    }
}

impl<'a> SubjectsSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        SubjectsSession {
            session_context_name,
            ..Default::default()
        }
    }
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    default_runtime_env_name-rt@{shape: subproc, label: default_runtime_env_name}

	subgraph extract_tasks_t
		UserCsv-subject-.->|FullTable|extract_tasks_p-subscribe
		extract_tasks_p-subscribe-->extract_tasks_p-processor
		extract_tasks_p-processor-->extract_tasks_p-publish
		extract_tasks_p-publish-->|Replace|SessionTasksInbox-subject
	end
	UserCsv-subject@{shape: doc, label: UserCsv}
	default_runtime_env_name-rt-->extract_tasks_t
	extract_tasks_p-subscribe@{shape: diamond, label: All}
	extract_tasks_p-processor@{shape: rect, label: ExtractTabular}
	extract_tasks_p-publish@{shape: fork}
	SessionTasksInbox-subject@{shape: doc, label: SessionTasksInbox}

	subgraph group_by_tasks_run_log_timestamp_t
		SessionTasksRunLog-subject-.->|LastRecordBatch|group_by_tasks_run_log_timestamp_p-subscribe
		group_by_tasks_run_log_timestamp_p-subscribe-->group_by_tasks_run_log_timestamp_p-processor
		group_by_tasks_run_log_timestamp_p-processor-->group_by_tasks_run_log_timestamp_p-publish
		group_by_tasks_run_log_timestamp_p-publish-->|Replace|group_by_tasks_run_log_timestamp_t-subject
		group_by_tasks_run_log_timestamp_t-subject-->|FullTable|select_tasks_run_log_timestamp_p-subscribe
		select_tasks_run_log_timestamp_p-subscribe-->select_tasks_run_log_timestamp_p-processor
		select_tasks_run_log_timestamp_p-processor-->select_tasks_run_log_timestamp_p-publish
		select_tasks_run_log_timestamp_p-publish-->|Replace|select_tasks_run_log_timestamp_t-subject
	end
	default_runtime_env_name-rt-->group_by_tasks_run_log_timestamp_t
	SessionTasksRunLog-subject@{shape: doc, label: SessionTasksRunLog}
	group_by_tasks_run_log_timestamp_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_run_log_timestamp_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_run_log_timestamp_p-publish@{shape: fork}
	group_by_tasks_run_log_timestamp_t-subject@{shape: doc, label: group_by_tasks_run_log_timestamp_t}
	select_tasks_run_log_timestamp_p-subscribe@{shape: diamond, label: All}
	select_tasks_run_log_timestamp_p-processor@{shape: rect, label: Select}
	select_tasks_run_log_timestamp_p-publish@{shape: fork}
	select_tasks_run_log_timestamp_t-subject@{shape: doc, label: select_tasks_run_log_timestamp_t}

	subgraph group_by_subject_change_log_delta_t
		SubjectsChangeLog-subject-.->|LastRecordBatch|group_by_subject_change_log_delta_p-subscribe
		group_by_subject_change_log_delta_p-subscribe-->group_by_subject_change_log_delta_p-processor
		group_by_subject_change_log_delta_p-processor-->group_by_subject_change_log_delta_p-publish
		group_by_subject_change_log_delta_p-publish-->|Replace|group_by_subject_change_log_delta_t-subject
		group_by_subject_change_log_delta_t-subject-->|FullTable|select_subject_change_log_delta_p-subscribe
		select_subject_change_log_delta_p-subscribe-->select_subject_change_log_delta_p-processor
		select_subject_change_log_delta_p-processor-->select_subject_change_log_delta_p-publish
		select_subject_change_log_delta_p-publish-->|Extend|SubjectsNumRows-subject
	end
	default_runtime_env_name-rt-->group_by_subject_change_log_delta_t
	SubjectsChangeLog-subject@{shape: doc, label: SubjectsChangeLog}
	group_by_subject_change_log_delta_p-subscribe@{shape: diamond, label: All}
	group_by_subject_change_log_delta_p-processor@{shape: rect, label: GroupBy}
	group_by_subject_change_log_delta_p-publish@{shape: fork}
	group_by_subject_change_log_delta_t-subject@{shape: doc, label: group_by_subject_change_log_delta_t}
	select_subject_change_log_delta_p-subscribe@{shape: diamond, label: All}
	select_subject_change_log_delta_p-processor@{shape: rect, label: Select}
	select_subject_change_log_delta_p-publish@{shape: fork}
	SubjectsNumRows-subject@{shape: doc, label: SubjectsNumRows}

	subgraph group_by_subjects_num_rows_t
		SubjectsNumRows-subject-.->|FullTable|group_by_subjects_num_rows_p-subscribe
		group_by_subjects_num_rows_p-subscribe-->group_by_subjects_num_rows_p-processor
		group_by_subjects_num_rows_p-processor-->group_by_subjects_num_rows_p-publish
		group_by_subjects_num_rows_p-publish-->|Replace|group_by_subjects_num_rows_t-subject
		group_by_subjects_num_rows_t-subject-->|FullTable|select_subjects_num_rows_delta_p-subscribe
		select_subjects_num_rows_delta_p-subscribe-->select_subjects_num_rows_delta_p-processor
		select_subjects_num_rows_delta_p-processor-->select_subjects_num_rows_delta_p-publish
		select_subjects_num_rows_delta_p-publish-->|Extend|SubjectsNumRows-subject
	end
	default_runtime_env_name-rt-->group_by_subjects_num_rows_t
	group_by_subjects_num_rows_p-subscribe@{shape: diamond, label: All}
	group_by_subjects_num_rows_p-processor@{shape: rect, label: GroupBy}
	group_by_subjects_num_rows_p-publish@{shape: fork}
	group_by_subjects_num_rows_t-subject@{shape: doc, label: group_by_subjects_num_rows_t}
	select_subjects_num_rows_delta_p-subscribe@{shape: diamond, label: All}
	select_subjects_num_rows_delta_p-processor@{shape: rect, label: Select}
	select_subjects_num_rows_delta_p-publish@{shape: fork}

	subgraph filter_processors_subscriptions_t
		SessionProcessors-subject-.->|FullTable|cmp_processors_subscriptions_p-subscribe
		cmp_processors_subscriptions_p-subscribe-->cmp_processors_subscriptions_p-processor
		cmp_processors_subscriptions_p-processor-->cmp_processors_subscriptions_p-publish
		cmp_processors_subscriptions_p-publish-->|Replace|cmp_processors_subscriptions_t-subject
		cmp_processors_subscriptions_t-subject-->|FullTable|filter_processors_subscriptions_p-subscribe
		filter_processors_subscriptions_p-subscribe-->filter_processors_subscriptions_p-processor
		filter_processors_subscriptions_p-processor-->filter_processors_subscriptions_p-publish
		filter_processors_subscriptions_p-publish-->|Replace|filter_processors_subscriptions_t-subject
		filter_processors_subscriptions_t-subject-->|FullTable|select_processors_subscriptions_p-subscribe
		select_processors_subscriptions_p-subscribe-->select_processors_subscriptions_p-processor
		select_processors_subscriptions_p-processor-->select_processors_subscriptions_p-publish
		select_processors_subscriptions_p-publish-->|Replace|select_processors_subscriptions_t-subject
	end
	default_runtime_env_name-rt-->filter_processors_subscriptions_t
	SessionProcessors-subject@{shape: doc, label: SessionProcessors}
	cmp_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	cmp_processors_subscriptions_p-processor@{shape: rect, label: Select}
	cmp_processors_subscriptions_p-publish@{shape: fork}
	cmp_processors_subscriptions_t-subject@{shape: doc, label: cmp_processors_subscriptions_t}
	filter_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	filter_processors_subscriptions_p-processor@{shape: rect, label: Filter}
	filter_processors_subscriptions_p-publish@{shape: fork}
	filter_processors_subscriptions_t-subject@{shape: doc, label: filter_processors_subscriptions_t}
	select_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	select_processors_subscriptions_p-processor@{shape: rect, label: Select}
	select_processors_subscriptions_p-publish@{shape: fork}
	select_processors_subscriptions_t-subject@{shape: doc, label: select_processors_subscriptions_t}

    %% `filter_tasks_processors_subscriptions_subjects_p` requires a custom operator...
	subgraph join_tasks_run_log_timestamp_t
		SubjectsChangeLog-subject-->|FullTable|group_by_subject_change_log_timestamp_p-subscribe
		group_by_subject_change_log_timestamp_p-subscribe-->group_by_subject_change_log_timestamp_p-processor
		group_by_subject_change_log_timestamp_p-processor-->group_by_subject_change_log_timestamp_p-publish
		group_by_subject_change_log_timestamp_p-publish-->|Replace|group_by_subject_change_log_timestamp_t-subject
		select_tasks_run_log_timestamp_t-subject-->|FullTable|join_tasks_run_log_timestamp_p-subscribe
		SessionTasksInbox-subject-.->|FullTable|join_tasks_run_log_timestamp_p-subscribe
		join_tasks_run_log_timestamp_p-subscribe-->join_tasks_run_log_timestamp_p-processor
		join_tasks_run_log_timestamp_p-processor-->join_tasks_run_log_timestamp_p-publish
		join_tasks_run_log_timestamp_p-publish-->|Replace|join_tasks_run_log_timestamp_t-subject
		join_tasks_run_log_timestamp_t-subject-->|FullTable|join_tasks_processors_p-subscribe
		SessionTasks-subject-->|FullTable|join_tasks_processors_p-subscribe
		join_tasks_processors_p-subscribe-->join_tasks_processors_p-processor
		join_tasks_processors_p-processor-->join_tasks_processors_p-publish
		join_tasks_processors_p-publish-->|Replace|join_tasks_processors_t-subject
		join_tasks_processors_t-subject-->|FullTable|join_tasks_processors_subscriptions_p-subscribe
		select_processors_subscriptions_t-subject-->|FullTable|join_tasks_processors_subscriptions_p-subscribe
		join_tasks_processors_subscriptions_p-subscribe-->join_tasks_processors_subscriptions_p-processor
		join_tasks_processors_subscriptions_p-processor-->join_tasks_processors_subscriptions_p-publish
		join_tasks_processors_subscriptions_p-publish-->|Replace|join_tasks_processors_subscriptions_t-subject
		join_tasks_processors_subscriptions_t-subject-->|FullTable|join_tasks_processors_subscriptions_subjects_p-subscribe
		group_by_subject_change_log_timestamp_t-subject-->|FullTable|join_tasks_processors_subscriptions_subjects_p-subscribe
		join_tasks_processors_subscriptions_subjects_p-subscribe-->join_tasks_processors_subscriptions_subjects_p-processor
		join_tasks_processors_subscriptions_subjects_p-processor-->join_tasks_processors_subscriptions_subjects_p-publish
		join_tasks_processors_subscriptions_subjects_p-publish-->|Replace|join_tasks_processors_subscriptions_subjects_t-subject
		join_tasks_processors_subscriptions_subjects_t-subject-->|FullTable|select_tasks_processors_subscriptions_subjects_p-subscribe
		select_tasks_processors_subscriptions_subjects_p-subscribe-->select_tasks_processors_subscriptions_subjects_p-processor
		select_tasks_processors_subscriptions_subjects_p-processor-->select_tasks_processors_subscriptions_subjects_p-publish
		select_tasks_processors_subscriptions_subjects_p-publish-->|Replace|select_tasks_processors_subscriptions_subjects_t-subject
		select_tasks_processors_subscriptions_subjects_t-subject-->|FullTable|filter_tasks_processors_subscriptions_subjects_p-subscribe
		filter_tasks_processors_subscriptions_subjects_p-subscribe-->filter_tasks_processors_subscriptions_subjects_p-processor
		filter_tasks_processors_subscriptions_subjects_p-processor-->filter_tasks_processors_subscriptions_subjects_p-publish
		filter_tasks_processors_subscriptions_subjects_p-publish-->|Replace|filter_tasks_processors_subscriptions_subjects_t-subject
	end
	default_runtime_env_name-rt-->join_tasks_run_log_timestamp_t
	group_by_subject_change_log_timestamp_p-subscribe@{shape: diamond, label: All}
	group_by_subject_change_log_timestamp_p-processor@{shape: rect, label: GroupBy}
	group_by_subject_change_log_timestamp_p-publish@{shape: fork}
	group_by_subject_change_log_timestamp_t-subject@{shape: doc, label: group_by_subject_change_log_timestamp_t}
	join_tasks_run_log_timestamp_p-subscribe@{shape: diamond, label: All}
	join_tasks_run_log_timestamp_p-processor@{shape: rect, label: Join}
	join_tasks_run_log_timestamp_p-publish@{shape: fork}
	join_tasks_run_log_timestamp_t-subject@{shape: doc, label: join_tasks_run_log_timestamp_t}
	SessionTasks-subject@{shape: doc, label: SessionTasks}
	join_tasks_processors_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_p-processor@{shape: rect, label: Join}
	join_tasks_processors_p-publish@{shape: fork}
	join_tasks_processors_t-subject@{shape: doc, label: join_tasks_processors_t}
	join_tasks_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_subscriptions_p-processor@{shape: rect, label: Join}
	join_tasks_processors_subscriptions_p-publish@{shape: fork}
	join_tasks_processors_subscriptions_t-subject@{shape: doc, label: join_tasks_processors_subscriptions_t}
	join_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: Join}
	join_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	join_tasks_processors_subscriptions_subjects_t-subject@{shape: doc, label: join_tasks_processors_subscriptions_subjects_t}
	select_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	select_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: Select}
	select_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	select_tasks_processors_subscriptions_subjects_t-subject@{shape: doc, label: select_tasks_processors_subscriptions_subjects_t}
	filter_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	filter_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: Filter}
	filter_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	filter_tasks_processors_subscriptions_subjects_t-subject@{shape: doc, label: filter_tasks_processors_subscriptions_subjects_t}

	subgraph filter_processors_publications_t
		SessionProcessors-subject-.->|FullTable|cmp_processors_publications_p-subscribe
		cmp_processors_publications_p-subscribe-->cmp_processors_publications_p-processor
		cmp_processors_publications_p-processor-->cmp_processors_publications_p-publish
		cmp_processors_publications_p-publish-->|Replace|cmp_processors_publications_t-subject
		cmp_processors_publications_t-subject-->|FullTable|filter_processors_publications_p-subscribe
		filter_processors_publications_p-subscribe-->filter_processors_publications_p-processor
		filter_processors_publications_p-processor-->filter_processors_publications_p-publish
		filter_processors_publications_p-publish-->|Replace|filter_processors_publications_t-subject
		filter_processors_publications_t-subject-->|FullTable|select_processors_publications_p-subscribe
		select_processors_publications_p-subscribe-->select_processors_publications_p-processor
		select_processors_publications_p-processor-->select_processors_publications_p-publish
		select_processors_publications_p-publish-->|Replace|select_processors_publications_t-subject
	end
	default_runtime_env_name-rt-->filter_processors_publications_t
	cmp_processors_publications_p-subscribe@{shape: diamond, label: All}
	cmp_processors_publications_p-processor@{shape: rect, label: Select}
	cmp_processors_publications_p-publish@{shape: fork}
	cmp_processors_publications_t-subject@{shape: doc, label: cmp_processors_publications_t}
	filter_processors_publications_p-subscribe@{shape: diamond, label: All}
	filter_processors_publications_p-processor@{shape: rect, label: Filter}
	filter_processors_publications_p-publish@{shape: fork}
	filter_processors_publications_t-subject@{shape: doc, label: filter_processors_publications_t}
	select_processors_publications_p-subscribe@{shape: diamond, label: All}
	select_processors_publications_p-processor@{shape: rect, label: Select}
	select_processors_publications_p-publish@{shape: fork}
	select_processors_publications_t-subject@{shape: doc, label: select_processors_publications_t}

	subgraph select_tasks_processors_publications_t
		filter_tasks_processors_subscriptions_subjects_t-subject-.->|FullTable|select_tasks_ready_to_run_p-subscribe
		select_tasks_ready_to_run_p-subscribe-->select_tasks_ready_to_run_p-processor
		select_tasks_ready_to_run_p-processor-->select_tasks_ready_to_run_p-publish
		select_tasks_ready_to_run_p-publish-->|Replace|select_tasks_ready_to_run_t-subject
		select_tasks_ready_to_run_t-subject-->|FullTable|join_tasks_ready_to_run_p-subscribe
		SessionTasks-subject-->|FullTable|join_tasks_ready_to_run_p-subscribe
		join_tasks_ready_to_run_p-subscribe-->join_tasks_ready_to_run_p-processor
		join_tasks_ready_to_run_p-processor-->join_tasks_ready_to_run_p-publish
		join_tasks_ready_to_run_p-publish-->|Replace|join_tasks_ready_to_run_t-subject
		join_tasks_ready_to_run_t-subject-->|FullTable|join_tasks_processors_publications_p-subscribe
		select_processors_publications_t-subject-->|FullTable|join_tasks_processors_publications_p-subscribe
		join_tasks_processors_publications_p-subscribe-->join_tasks_processors_publications_p-processor
		join_tasks_processors_publications_p-processor-->join_tasks_processors_publications_p-publish
		join_tasks_processors_publications_p-publish-->|Replace|join_tasks_processors_publications_t-subject
		join_tasks_processors_publications_t-subject-->|FullTable|select_tasks_processors_publications_p-subscribe
		select_tasks_processors_publications_p-subscribe-->select_tasks_processors_publications_p-processor
		select_tasks_processors_publications_p-processor-->select_tasks_processors_publications_p-publish
		select_tasks_processors_publications_p-publish-->|Replace|select_tasks_processors_publications_t-subject
	end
	default_runtime_env_name-rt-->select_tasks_processors_publications_t
	select_tasks_ready_to_run_p-subscribe@{shape: diamond, label: All}
	select_tasks_ready_to_run_p-processor@{shape: rect, label: Select}
	select_tasks_ready_to_run_p-publish@{shape: fork}
	select_tasks_ready_to_run_t-subject@{shape: doc, label: select_tasks_ready_to_run_t}
	join_tasks_ready_to_run_p-subscribe@{shape: diamond, label: All}
	join_tasks_ready_to_run_p-processor@{shape: rect, label: Join}
	join_tasks_ready_to_run_p-publish@{shape: fork}
	join_tasks_ready_to_run_t-subject@{shape: doc, label: join_tasks_ready_to_run_t}
	join_tasks_processors_publications_p-subscribe@{shape: diamond, label: All}
	join_tasks_processors_publications_p-processor@{shape: rect, label: Join}
	join_tasks_processors_publications_p-publish@{shape: fork}
	join_tasks_processors_publications_t-subject@{shape: doc, label: join_tasks_processors_publications_t}
	select_tasks_processors_publications_p-subscribe@{shape: diamond, label: All}
	select_tasks_processors_publications_p-processor@{shape: rect, label: Select}
	select_tasks_processors_publications_p-publish@{shape: fork}
	select_tasks_processors_publications_t-subject@{shape: doc, label: select_tasks_processors_publications_t}

	subgraph aggregate_tasks_processors_publications_t
        UserMessages-subject-.->|FullTable|aggregate_tasks_processors_publications_p-subscribe
        AssistantMessages-subject-.->|FullTable|aggregate_tasks_processors_publications_p-subscribe
		filter_tasks_processors_subscriptions_subjects_t-subject-.->|FullTable|aggregate_tasks_processors_publications_p-subscribe
		select_tasks_processors_publications_t-subject-.->|FullTable|aggregate_tasks_processors_publications_p-subscribe
		aggregate_tasks_processors_publications_p-subscribe-->aggregate_tasks_processors_publications_p-processor
		aggregate_tasks_processors_publications_p-processor-->aggregate_tasks_processors_publications_p-publish
		aggregate_tasks_processors_publications_p-publish-->|Replace|AssistantCsv-subject
	end
	default_runtime_env_name-rt-->aggregate_tasks_processors_publications_t
	UserMessages-subject@{shape: doc, label: UserMessages}
	AssistantMessages-subject@{shape: doc, label: AssistantMessages}
	aggregate_tasks_processors_publications_p-subscribe@{shape: diamond, label: Any}
	aggregate_tasks_processors_publications_p-processor@{shape: rect, label: AttachmentAggregatorProcessor}
	aggregate_tasks_processors_publications_p-publish@{shape: fork}
	AssistantCsv-subject@{shape: doc, label: AssistantCsv}"#
    }
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    UserCsv["UserCsv"] {
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }
    extract_tasks_p["extract_tasks_p"] {
        Boolean cpu "false"
        Utf8 format "CsvDefault"
        Utf8 lhs_name "UserCsv"
        List-Utf8 lhs_values "['bytes']"
        Utf8 operator "ExtractTabular"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasksInbox["SessionTasksInbox"] {
        Utf8 session_name
        Utf8 task_name
    }
    SessionTasksRunLog["SessionTasksRunLog"] {
        Utf8 session_name
        Utf8 task_name
        Int64 timestamp
    }
    group_by_tasks_run_log_timestamp_p["group_by_tasks_run_log_timestamp_p"] {
        List-Utf8 agg_columns "['timestamp']"
        List-Utf8 agg_operators "['Last']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionTasksRunLog"
        List-Utf8 lhs_values "['task_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_run_log_timestamp_p["select_tasks_run_log_timestamp_p"] {
        List-Utf8 as_columns "['','timestamp']"
        List-Utf8 cast_datatypes "['Utf8','Int64 ']"
        List-Utf8 cast_operators "['None','None']"
        List-Utf8 cast_templates "['','']"
        List-Utf8 column_operators "['None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_tasks_run_log_timestamp_t"
        List-Utf8 lhs_values "['task_name','timestamp-Last']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_run_log_timestamp_t["select_tasks_run_log_timestamp_t"] {
        Utf8 task_name
        Int64 timestamp
    }
    SessionProcessors["SessionProcessors"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_names
        Utf8 subscribe_type
        UInt8 is_subscription
    }
    cmp_processors_subscriptions_p["cmp_processors_subscriptions_p"] {
        List-Utf8 as_columns "['','','','','','','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 cast_operators "['None','None','None','None','None','None','None','None']"
        List-Utf8 cast_templates "['','','','','','','','1']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription','subscription']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','','','','','','','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    filter_processors_subscriptions_p["filter_processors_subscriptions_p"] {
        List-Utf8 cmp_columns "['subscription']"
        List-Utf8 cmp_operators "['Equals']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_processors_subscriptions_t"
        List-Utf8 lhs_values "['is_subscription']"
        Utf8 operator "Filter"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_subscriptions_p["select_processors_subscriptions_p"] {
        List-Utf8 as_columns "['','','','','','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8']"
        List-Utf8 cast_operators "['None','None','None','None','None','None','None']"
        List-Utf8 cast_templates "['','','','','','','']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_subscriptions_t"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','','','','','','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_subscriptions_t["select_processors_subscriptions_t"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_names
        Utf8 subscribe_type
        UInt8 is_subscription
    }
    cmp_processors_publications_p["cmp_processors_publications_p"] {
        List-Utf8 as_columns "['','','','','','','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 cast_operators "['None','None','None','None','None','None','None','None']"
        List-Utf8 cast_templates "['','','','','','','','0']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription','publication']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','','','','','','','']"
    }
    filter_processors_publications_p["filter_processors_publications_p"] {
        List-Utf8 cmp_columns "['publication']"
        List-Utf8 cmp_operators "['Equals']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_processors_publications_t"
        List-Utf8 lhs_values "['is_subscription']"
        Utf8 operator "Filter"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_publications_p["select_processors_publications_p"] {
        List-Utf8 as_columns "['','','','','','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8']"
        List-Utf8 cast_operators "['None','None','None','None','None','None','None']"
        List-Utf8 cast_templates "['','','','','','','']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_publications_t"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','','','','','','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_publications_t["select_processors_publications_t"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_names
        Utf8 subscribe_type
        UInt8 is_subscription
    }
    SubjectsChangeLog["SubjectsChangeLog"] {
        Utf8 subject_name
        Utf8 task_name
        Utf8 session_name
        Int64 num_rows_delta
        Int64 timestamp
    }
    group_by_subject_change_log_delta_p["group_by_subject_change_log_delta_p"] {
        List-Utf8 agg_columns "['num_rows_delta']"
        List-Utf8 agg_operators "['Sum']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsChangeLog"
        List-Utf8 lhs_values "['subject_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_subject_change_log_delta_p["select_subject_change_log_delta_p"] {
        List-Utf8 as_columns "['','num_rows']"
        List-Utf8 cast_datatypes "['Utf8','Int64']"
        List-Utf8 cast_operators "['None','None']"
        List-Utf8 cast_templates "['','']"
        List-Utf8 column_operators "['None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_subject_change_log_delta_t"
        List-Utf8 lhs_values "['subject_name','num_rows_delta-Sum']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SubjectsNumRows["SubjectsNumRows"] {
        Utf8 subject_name
        Int64 num_rows
    }
    group_by_subjects_num_rows_p["group_by_subjects_num_rows_p"] {
        List-Utf8 agg_columns "['num_rows']"
        List-Utf8 agg_operators "['Sum']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsNumRows"
        List-Utf8 lhs_values "['subject_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_subjects_num_rows_delta_p["select_subjects_num_rows_delta_p"] {
        List-Utf8 as_columns "['','num_rows']"
        List-Utf8 cast_datatypes "['Utf8','Int64 ']"
        List-Utf8 cast_operators "['None','None']"
        List-Utf8 cast_templates "['','']"
        List-Utf8 column_operators "['','']"
        Boolean cpu "false"
        Utf8 lhs_name "group_by_subjects_num_rows_t"
        List-Utf8 lhs_values "['subject_name','num_rows-Sum']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','num_rows_delta-Sum']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    group_by_subject_change_log_timestamp_p["group_by_subject_change_log_timestamp_p"] {
        List-Utf8 agg_columns "['timestamp']"
        List-Utf8 agg_operators "['Last']"
        Boolean cpu "false"
        Utf8 lhs_name "SubjectsChangeLog"
        List-Utf8 lhs_values "['subject_name','task_name','session_name']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    join_tasks_run_log_timestamp_p["join_tasks_run_log_timestamp_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "task_name"
        Utf8 lhs_name "select_tasks_run_log_timestamp_t"
        Utf8 lhs_pk "task_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "task_name"
        Utf8 rhs_name "SessionTasksInbox"
        Utf8 rhs_pk "task_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasks["SessionTasks"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 runtime_env_name
    }
    join_tasks_processors_p["join_tasks_processors_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "task_name"
        Utf8 lhs_name "join_tasks_run_log_timestamp_t"
        Utf8 lhs_pk "task_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "task_name"
        Utf8 rhs_name "SessionTasks"
        Utf8 rhs_pk "task_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    join_tasks_processors_subscriptions_p["join_tasks_processors_subscriptions_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "join_tasks_processors_t"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "select_processors_subscriptions_t"
        Utf8 rhs_pk "processor_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    join_tasks_processors_subscriptions_subjects_p["join_tasks_processors_subscriptions_subjects_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "subject_name"
        Utf8 lhs_name "join_tasks_processors_subscriptions_t"
        Utf8 lhs_pk "subject_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "subject_name"
        Utf8 rhs_name "group_by_subject_change_log_timestamp_t"
        Utf8 rhs_pk "subject_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_subscriptions_subjects_p["select_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 as_columns "['','','','','','','','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Int64','Int64']"
        List-Utf8 cast_operators "['None','None','None','None','None','None','None','None','None']"
        List-Utf8 cast_templates "['','','','','','','','','']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_subscriptions_subjects_t"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','timestamp','timestamp-Last']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','','','','','','','','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    filter_tasks_processors_subscriptions_subjects_p["filter_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 cmp_columns "['timestamp-Last']"
        List-Utf8 cmp_operators "['GreaterThanOrEqualTo']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "lhs_name"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Filter"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    filter_tasks_processors_subscriptions_subjects_t["filter_tasks_processors_subscriptions_subjects_t"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
    }
    select_tasks_ready_to_run_p["select_tasks_ready_to_run_p"] {
        List-Utf8 as_columns "['as_columns']"
        List-Utf8 cast_datatypes "['Utf8']"
        List-Utf8 cast_operators "['None']"
        List-Utf8 cast_templates "['cast_template']"
        List-Utf8 column_operators "['None']"
        Boolean cpu "false"
        Utf8 lhs_name "lhs_name"
        List-Utf8 lhs_values "['lhs_values']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['rhs_values']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    join_tasks_ready_to_run_p["join_tasks_ready_to_run_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "task_name"
        Utf8 lhs_name "select_tasks_ready_to_run_t"
        Utf8 lhs_pk "task_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "task_name"
        Utf8 rhs_name "SessionTasks"
        Utf8 rhs_pk "task_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    join_tasks_processors_publications_p["join_tasks_processors_publications_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "join_tasks_ready_to_run_t"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "select_processors_publications_t"
        Utf8 rhs_pk "processor_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_publications_p["select_tasks_processors_publications_p"] {
        List-Utf8 as_columns "['','','','','','','']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8']"
        List-Utf8 cast_operators "['None','None','None','None','None','None','None']"
        List-Utf8 cast_templates "['','','','','','','']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None']"
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_publications_t"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type']"
        Utf8 operator "Select"
        List-Utf8 rhs_values "['','','','','','','']"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_publications_t["select_tasks_processors_publications_t"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
    }
    AssistantMessages["AssistantMessages"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    UserMessages["UserMessages"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    aggregate_tasks_processors_publications_p["aggregate_tasks_processors_publications_p"] {
        Boolean asc "true"
        Boolean cpu "false"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Sort"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    AssistantCsv["AssistantCsv"] {
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{BuilderTrait, IPCMessage, MappableTrait, MessageTrait, TableTrait};
    use phymes_diagnostics::HashMap;

    use crate::{SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream, SessionStreamState, create_message_map};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_subjects_session() -> Result<()> {
        // Initialize the session
        let subjects_session = SubjectsSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(subjects_session.as_mermaid_flowchart(), true)?
            .with_state_from_mermaid_erdiagram(subjects_session.as_mermaid_erdiagram(), true, true)?
            .with_name(subjects_session.session_context_name)            
            .with_diagnostics(true)
            .add_processor_subjects().unwrap()
            .add_session_interface(None).unwrap()
            .build_with_tables()?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));
        
        // Create the messages
        let message_map = create_message_map(vec![]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
        let mut response: Vec<HashMap<String, IPCMessage>> =
            session_stream.try_collect().await?;


        Ok(())
    }
}
