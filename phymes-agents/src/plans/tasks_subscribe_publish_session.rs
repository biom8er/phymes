/// A session for determining the next superstep task publications and subscriptions
pub struct TasksSubscribePublishSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl Default for TasksSubscribePublishSession<'_> {
    fn default() -> Self {
        TasksSubscribePublishSession {
            session_context_name: "tasks_publish_subscribe_session",
        }
    }
}

impl<'a> TasksSubscribePublishSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        TasksSubscribePublishSession {
            session_context_name,
            ..Default::default()
        }
    }
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
    default_runtime_env_name-rt@{shape: subproc, label: default_runtime_env_name}

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
		select_tasks_run_log_timestamp_t-subject-.->|FullTable|join_tasks_run_log_timestamp_p-subscribe
		SessionTasks-subject-->|FullTable|join_tasks_run_log_timestamp_p-subscribe
		join_tasks_run_log_timestamp_p-subscribe-->join_tasks_run_log_timestamp_p-processor
		join_tasks_run_log_timestamp_p-processor-->join_tasks_run_log_timestamp_p-publish
		join_tasks_run_log_timestamp_p-publish-->|Replace|join_tasks_run_log_timestamp_t-subject
		join_tasks_run_log_timestamp_t-subject-->|FullTable|join_tasks_processors_subscriptions_p-subscribe
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
		filter_tasks_processors_subscriptions_subjects_p-publish-->|Replace|SessionTasksSubscribe-subject
	end
	default_runtime_env_name-rt-->join_tasks_run_log_timestamp_t
	SubjectsChangeLog-subject@{shape: doc, label: SubjectsChangeLog}
	group_by_subject_change_log_timestamp_p-subscribe@{shape: diamond, label: All}
	group_by_subject_change_log_timestamp_p-processor@{shape: rect, label: GroupBy}
	group_by_subject_change_log_timestamp_p-publish@{shape: fork}
	group_by_subject_change_log_timestamp_t-subject@{shape: doc, label: group_by_subject_change_log_timestamp_t}
	SessionTasks-subject@{shape: doc, label: SessionTasks}
	join_tasks_run_log_timestamp_p-subscribe@{shape: diamond, label: All}
	join_tasks_run_log_timestamp_p-processor@{shape: rect, label: Join}
	join_tasks_run_log_timestamp_p-publish@{shape: fork}
	join_tasks_run_log_timestamp_t-subject@{shape: doc, label: join_tasks_run_log_timestamp_t}
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
	SessionTasksSubscribe-subject@{shape: doc, label: SessionTasksSubscribe}

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
		SessionTasksSubscribe-subject-.->|FullTable|select_tasks_ready_to_run_p-subscribe
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
		select_tasks_processors_publications_p-publish-->|Replace|SessionTasksSubscribePublish-subject
	end
	default_runtime_env_name-rt-->SessionTasksSubscribePublish
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
	SessionTasksSubscribePublish-subject@{shape: doc, label: SessionTasksSubscribePublish}"#
    }
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    extract_tasks_p["extract_tasks_p"] {
        Boolean cpu "false"
        Utf8 format "CsvDefault"
        Utf8 lhs_name "UserCsv"
        List-Utf8 lhs_values "['bytes']"
        Utf8 operator "ExtractTabular"
        Utf8 stream "AccumulateLHSAccumulateRHS"
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
        Boolean cpu "false"
        Utf8 lhs_name "group_by_tasks_run_log_timestamp_t"
        List-Utf8 lhs_values "['task_name','timestamp-Last']"
        Utf8 operator "Select"
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
        List-Utf8 as_columns "['','','','','','','','subscription']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','Ones']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription','subscription']"
        Utf8 operator "Select"
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
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_subscriptions_t"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription']"
        Utf8 operator "Select"
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
        List-Utf8 as_columns "['','','','','','','','publication']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','Zeros']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription','publication']"
        Utf8 operator "Select"
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
        Boolean cpu "false"
        Utf8 lhs_name "filter_processors_publications_t"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_names','subscribe_type','is_subscription']"
        Utf8 operator "Select"
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
        Utf8 rhs_name "SessionTasks"
        Utf8 rhs_pk "task_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasks["SessionTasks"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 runtime_env_name
    }
    join_tasks_processors_subscriptions_p["join_tasks_processors_subscriptions_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "join_tasks_run_log_timestamp_t"
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
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_subscriptions_subjects_t"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','timestamp','timestamp-Last']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    filter_tasks_processors_subscriptions_subjects_p["filter_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 cmp_columns "['timestamp-Last']"
        List-Utf8 cmp_operators "['GreaterThanOrEqualTo']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "select_tasks_processors_subscriptions_subjects_t"
        List-Utf8 lhs_values "['timestamp']"
        Utf8 operator "Filter"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasksSubscribe["SessionTasksSubscribe"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 subscription_name
        Utf8 subscription_table_name
        Utf8 subscribe_type
    }
    select_tasks_ready_to_run_p["select_tasks_ready_to_run_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "SessionTasksSubscribe"
        List-Utf8 lhs_values "['session_name','task_name']"
        Utf8 operator "Select"
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
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_publications_t"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasksSubscribePublish["SessionTasksSubscribePublish"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, MappableTrait, MessageBuilderTrait, MessageTrait, Table, TableBuilderTrait, TablePublication, TableTrait, create_session_tasks_subscribe_publish_batch};
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, create_message_map, plans::user_session_inner,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_tasks_subscribe_publish_session() -> Result<()> {
        // Initialize the session
        let tasks_publish_subscribe_session = TasksSubscribePublishSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            tasks_publish_subscribe_session.as_mermaid_flowchart(),
            false,
            )?
            .with_state_from_mermaid_erdiagram(tasks_publish_subscribe_session.as_mermaid_erdiagram(), true, true)?
            .with_name(tasks_publish_subscribe_session.session_context_name)
            .with_diagnostics(true)
            .add_processor_subjects()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test session data
        let (user_session_ctx, _user_session_stream) = user_session_inner::user_session()?;

        let mut message_map = {
            let usss = user_session_ctx.read();
            let table = usss
                .get_states()
                .get(AvailableSubjects::SessionProcessors.to_string().as_str())
                .unwrap()
                .read();
            let session_processor_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionProcessors.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SessionProcessors.to_string(),
                })
                .with_publisher(tasks_publish_subscribe_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_states()
                .get(AvailableSubjects::SessionTasks.to_string().as_str())
                .unwrap()
                .read();
            let session_tasks_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionTasks.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SessionTasks.to_string(),
                })
                .with_publisher(tasks_publish_subscribe_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_states()
                .get(AvailableSubjects::SessionTasksRunLog.to_string().as_str())
                .unwrap()
                .read();
            let session_tasks_run_log_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SessionTasksRunLog.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SessionTasksRunLog.to_string(),
                })
                .with_publisher(tasks_publish_subscribe_session.session_context_name)
                .make_name()?
                .build()?;
            let table = usss
                .get_states()
                .get(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .unwrap()
                .read();
            let subjects_change_log_message = IPCMessage::get_builder()
                .with_message(table.to_ipc_stream()?)
                .with_subject(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .with_update(&TablePublication::Replace {
                    table_name: AvailableSubjects::SubjectsChangeLog.to_string(),
                })
                .with_publisher(tasks_publish_subscribe_session.session_context_name)
                .make_name()?
                .build()?;
            create_message_map(vec![
                session_processor_message,
                session_tasks_message,
                session_tasks_run_log_message,
                subjects_change_log_message,
            ])
        };
        dbg!(&message_map);

        // 1. Message to trigger the first superstep
        let session_names = (0..8).map(|_| tasks_publish_subscribe_session.session_context_name.to_string()).collect::<Vec<_>>();
        let task_names = vec!["group_by_tasks_run_log_timestamp_t", "group_by_tasks_run_log_timestamp_t",
            "filter_processors_subscriptions_t", "filter_processors_subscriptions_t", "filter_processors_subscriptions_t",
            "filter_processors_publications_t", "filter_processors_publications_t", "filter_processors_publications_t",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_names = vec!["group_by_tasks_run_log_timestamp_p", "select_tasks_run_log_timestamp_p",
            "cmp_processors_subscriptions_p", "filter_processors_subscriptions_p", "select_processors_subscriptions_p",
            "cmp_processors_publications_p", "filter_processors_publications_p", "select_processors_publications_p",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Select",
            "Select", "Filter","Select",
            "Select", "Filter","Select",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subscription_names = vec![vec!["OnUpdateLastRecordBatch","AlwaysFullTable"], vec!["AlwaysFullTable","AlwaysFullTable"],
            vec!["OnUpdateFullTable","AlwaysFullTable"], vec!["AlwaysFullTable","AlwaysFullTable"], vec!["AlwaysFullTable","AlwaysFullTable"],
            vec!["OnUpdateFullTable","AlwaysFullTable"], vec!["AlwaysFullTable","AlwaysFullTable"], vec!["AlwaysFullTable","AlwaysFullTable"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let subscription_table_names = vec![vec!["SessionTasksRunLog", "group_by_tasks_run_log_timestamp_p"], vec!["group_by_tasks_run_log_timestamp_t", "select_tasks_run_log_timestamp_p"],
            vec!["SessionProcessors", "cmp_processors_subscriptions_p"], vec!["cmp_processors_subscriptions_t", "filter_processors_subscriptions_p"], vec!["filter_processors_subscriptions_t", "select_processors_subscriptions_p"],
            vec!["SessionProcessors", "cmp_processors_publications_p"], vec!["cmp_processors_publications_t", "filter_processors_publications_p"], vec!["select_processors_publications_t", "select_processors_publications_p"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"],
            vec!["Replace"], vec!["Replace"], vec!["Replace"],
            vec!["Replace"], vec!["Replace"], vec!["Replace"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_table_names = vec![vec!["group_by_tasks_run_log_timestamp_t"], vec!["select_tasks_run_log_timestamp_t"],
            vec!["cmp_processors_subscriptions_t"], vec!["filter_processors_subscriptions_t"], vec!["select_processors_subscriptions_t"],
            vec!["cmp_processors_publications_t"], vec!["select_processors_publications_t"], vec!["select_processors_publications_t"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(session_names, task_names, processor_names, processor_types, subscription_names, subscription_table_names, publication_names, publication_table_names)?;
        let table = Table::get_builder()
            .with_name(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(tasks_publish_subscribe_session.session_context_name)
            .make_name()?
            .build()?;
        let _ = message_map.insert(tasks_publish_subscribe_message.get_name().to_string(), tasks_publish_subscribe_message);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        let session_reading = session_ctx_arc.read();
        let table_reading = session_reading.get_states().get("select_tasks_run_log_timestamp_t").unwrap().read();
        let column = table_reading.get_column_as_vec_str("task_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
        for timestamp in column {
            assert!(timestamp > 0);
        }

        let table_reading = session_reading.get_states().get("select_processors_publications_t").unwrap().read();
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("publication_subscription_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("publication_subscription_table_names");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("subscribe_type");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_primitive::<u8>("is_subscription")?;
        assert_eq!(column, [0]);

        let table_reading = session_reading.get_states().get("select_processors_publications_t").unwrap().read();
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("publication_subscription_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("publication_subscription_table_names");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("subscribe_type");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_primitive::<u8>("is_subscription")?;
        assert_eq!(column, [0]);

        // 2. Message to trigger the second superstep
        let session_names = (0..6).map(|_| tasks_publish_subscribe_session.session_context_name.to_string()).collect::<Vec<_>>();
        let task_names = vec!["join_tasks_run_log_timestamp_t", 
            "join_tasks_run_log_timestamp_t", 
            "join_tasks_run_log_timestamp_t",
            "join_tasks_run_log_timestamp_t", 
            "join_tasks_run_log_timestamp_t", 
            "join_tasks_run_log_timestamp_t",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_names = vec!["group_by_subject_change_log_timestamp_p", 
            "join_tasks_run_log_timestamp_p",
            "join_tasks_processors_subscriptions_p",            
            "join_tasks_processors_subscriptions_subjects_p", 
            "select_tasks_processors_subscriptions_subjects_p", 
            "filter_tasks_processors_subscriptions_subjects_p",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", 
            "Join",            
            "Join",
            "Join", 
            "Select",
            "Filter", 
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subscription_names = vec![vec!["AlwaysFullTable","AlwaysFullTable"],
            vec!["OnUpdateFullTable","AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let subscription_table_names = vec![vec!["SubjectsChangeLog", "group_by_subject_change_log_timestamp_p"], 
            vec!["select_tasks_run_log_timestamp_t", "SessionTasks", "join_tasks_run_log_timestamp_p"], 
            vec!["join_tasks_run_log_timestamp_t", "select_processors_subscriptions_t", "join_tasks_processors_subscriptions_p"], 
            vec!["join_tasks_processors_subscriptions_t", "group_by_subject_change_log_timestamp_t", "join_tasks_processors_subscriptions_subjects_p"],             
            vec!["join_tasks_processors_subscriptions_subjects_t", "select_tasks_processors_subscriptions_subjects_p"], 
            vec!["select_tasks_processors_subscriptions_subjects_t", "filter_tasks_processors_subscriptions_subjects_p"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], 
            vec!["Replace"],             
            vec!["Replace"],
            vec!["Replace"], 
            vec!["Replace"], 
            vec!["Replace"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_table_names = vec![vec!["group_by_subject_change_log_timestamp_t"], 
            vec!["join_tasks_run_log_timestamp_t"],             
            vec!["join_tasks_processors_subscriptions_t"],
            vec!["join_tasks_processors_subscriptions_subjects_t"], 
            vec!["select_tasks_processors_subscriptions_subjects_t"], 
            vec!["SessionTasksSubscribe"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(session_names, task_names, processor_names, processor_types, subscription_names, subscription_table_names, publication_names, publication_table_names)?;
        let table = Table::get_builder()
            .with_name(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(tasks_publish_subscribe_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![tasks_publish_subscribe_message]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        let session_reading = session_ctx_arc.read();
        let table_reading = session_reading.get_states().get("SessionTasksSubscribe").unwrap().read();
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("subscription_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("subscription_table_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("subscribe_type");
        assert_eq!(column, [""]);

        // 3. Message to trigger the third superstep
        let session_names = (0..4).map(|_| tasks_publish_subscribe_session.session_context_name.to_string()).collect::<Vec<_>>();
        let task_names = vec!["select_tasks_processors_publications_t", 
            "select_tasks_processors_publications_t", 
            "select_tasks_processors_publications_t",
            "select_tasks_processors_publications_t",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_names = vec!["select_tasks_ready_to_run_p", 
            "join_tasks_ready_to_run_p",
            "join_tasks_processors_publications_p",            
            "select_tasks_processors_publications_p", 
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_types = vec!["Select", 
            "Join",            
            "Join",
            "Select",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subscription_names = vec![vec!["OnUpdateFullTable","AlwaysFullTable"],
            vec!["AlwaysFullTable","AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable"], 
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let subscription_table_names = vec![vec!["SessionTasksSubscribe", "select_tasks_ready_to_run_p"], 
            vec!["select_tasks_ready_to_run_t", "SessionTasks", "join_tasks_ready_to_run_p"], 
            vec!["join_tasks_ready_to_run_t", "select_processors_publications_t", "join_tasks_processors_publications_p"], 
            vec!["select_processors_publications_t", "select_tasks_processors_publications_p"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], 
            vec!["Replace"],             
            vec!["Replace"],
            vec!["Replace"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_table_names = vec![vec!["select_tasks_ready_to_run_t"], 
            vec!["join_tasks_ready_to_run_t"],             
            vec!["join_tasks_processors_publications_t"],
            vec!["SessionTasksSubscribePublish"], 
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();

        let batch = create_session_tasks_subscribe_publish_batch(session_names, task_names, processor_names, processor_types, subscription_names, subscription_table_names, publication_names, publication_table_names)?;
        let table = Table::get_builder()
            .with_name(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_record_batches(vec![batch])?
            .build()?;
        let tasks_publish_subscribe_message = IPCMessage::get_builder()
            .with_message(table.to_ipc_stream()?)
            .with_subject(AvailableSubjects::SessionTasksSubscribePublish.to_string().as_str())
            .with_update(&TablePublication::Replace {
                table_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            })
            .with_publisher(tasks_publish_subscribe_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![tasks_publish_subscribe_message]);

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        let session_reading = session_ctx_arc.read();
        let table_reading = session_reading.get_states().get("SessionTasksSubscribePublish").unwrap().read();
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, [""]);
        let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        assert_eq!(column, [[""]]);
        let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        assert_eq!(column, [[""]]);
        let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        assert_eq!(column, [[""]]);
        let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        assert_eq!(column, [[""]]);

        Ok(())
    }
}
