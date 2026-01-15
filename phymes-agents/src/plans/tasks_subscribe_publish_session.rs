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
		select_tasks_processors_subscriptions_subjects_t-subject-->|FullTable|group_by_tasks_processors_subscriptions_p-subscribe
		group_by_tasks_processors_subscriptions_p-subscribe-->group_by_tasks_processors_subscriptions_p-processor
		group_by_tasks_processors_subscriptions_p-processor-->group_by_tasks_processors_subscriptions_p-publish
		group_by_tasks_processors_subscriptions_p-publish-->|Replace|SessionTasksSubscribeAggregate-subject
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
	group_by_tasks_processors_subscriptions_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_processors_subscriptions_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_processors_subscriptions_p-publish@{shape: fork}
	SessionTasksSubscribeAggregate-subject@{shape: doc, label: SessionTasksSubscribeAggregate}

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
		SessionTasksSubscribe-subject-.->|FullTable|group_by_tasks_processors_subscriptions_subjects_p-subscribe
		group_by_tasks_processors_subscriptions_subjects_p-subscribe-->group_by_tasks_processors_subscriptions_subjects_p-processor
		group_by_tasks_processors_subscriptions_subjects_p-processor-->group_by_tasks_processors_subscriptions_subjects_p-publish
		group_by_tasks_processors_subscriptions_subjects_p-publish-->|Replace|group_by_tasks_processors_subscriptions_subjects_t-subject
		select_processors_publications_t-subject-->|FullTable|group_by_tasks_processors_publications_p-subscribe
		group_by_tasks_processors_publications_p-subscribe-->group_by_tasks_processors_publications_p-processor
		group_by_tasks_processors_publications_p-processor-->group_by_tasks_processors_publications_p-publish
		group_by_tasks_processors_publications_p-publish-->|Replace|group_by_tasks_processors_publications_t-subject
		group_by_tasks_processors_subscriptions_subjects_t-subject-->|FullTable|join_tasks_processors_publications_p-subscribe
		group_by_tasks_processors_publications_t-subject-->|FullTable|join_tasks_processors_publications_p-subscribe
		join_tasks_processors_publications_p-subscribe-->join_tasks_processors_publications_p-processor
		join_tasks_processors_publications_p-processor-->join_tasks_processors_publications_p-publish
		join_tasks_processors_publications_p-publish-->|Replace|join_tasks_processors_publications_t-subject
		join_tasks_processors_publications_t-subject-->|FullTable|select_tasks_processors_publications_p-subscribe
		select_tasks_processors_publications_p-subscribe-->select_tasks_processors_publications_p-processor
		select_tasks_processors_publications_p-processor-->select_tasks_processors_publications_p-publish
		select_tasks_processors_publications_p-publish-->|Replace|SessionTasksSubscribePublish-subject
	end
	default_runtime_env_name-rt-->select_tasks_processors_publications_t
	SessionTasksSubscribe-subject@{shape: doc, label: SessionTasksSubscribe}
	group_by_tasks_processors_subscriptions_subjects_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_processors_subscriptions_subjects_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_processors_subscriptions_subjects_p-publish@{shape: fork}
	group_by_tasks_processors_subscriptions_subjects_t-subject@{shape: doc, label: group_by_tasks_processors_subscriptions_subjects_t}
	group_by_tasks_processors_publications_p-subscribe@{shape: diamond, label: All}
	group_by_tasks_processors_publications_p-processor@{shape: rect, label: GroupBy}
	group_by_tasks_processors_publications_p-publish@{shape: fork}
	group_by_tasks_processors_publications_t-subject@{shape: doc, label: group_by_tasks_processors_publications_t}
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
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }
    cmp_processors_subscriptions_p["cmp_processors_subscriptions_p"] {
        List-Utf8 as_columns "['','','','','','','','','subscription']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Ones']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription','subscription']"
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
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_subscriptions_t["select_processors_subscriptions_t"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        UInt8 is_subscription
    }
    cmp_processors_publications_p["cmp_processors_publications_p"] {
        List-Utf8 as_columns "['','','','','','','','','publication']"
        List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','Utf8','UInt8','UInt8']"
        List-Utf8 column_operators "['None','None','None','None','None','None','None','None','Zeros']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionProcessors"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription','publication']"
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
        List-Utf8 lhs_values "['session_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','is_subscription']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_processors_publications_t["select_processors_publications_t"] {
        Utf8 session_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 publication_subscription_name
        Utf8 publication_subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
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
        Utf8 lhs_fk "publication_subscription_table_name"
        Utf8 lhs_name "join_tasks_processors_subscriptions_t"
        Utf8 lhs_pk "publication_subscription_table_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "subject_name"
        Utf8 rhs_name "group_by_subject_change_log_timestamp_t"
        Utf8 rhs_pk "subject_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_subscriptions_subjects_p["select_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 as_columns "['','','','','subscription_name','subscription_table_name','','','','']"
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_subscriptions_subjects_t"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','publication_subscription_name','publication_subscription_table_name','subscribe_type','update_type','timestamp','timestamp-Last']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_subscriptions_subjects_t["select_tasks_processors_subscriptions_subjects_t"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 subscription_name
        Utf8 subscription_table_name
        Utf8 subscribe_type
        Utf8 update_type
        Int64 timestamp
        Int64 timestamp-Last
    }
    group_by_tasks_processors_subscriptions_p["group_by_tasks_processors_subscriptions_p"] {
        List-Utf8 agg_columns "['subscription_name','subscription_table_name','subscribe_type','update_type','timestamp','timestamp-Last']"
        List-Utf8 agg_operators "['List','List','Last','Last','List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_tasks_processors_subscriptions_subjects_t"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasksSubscribeAggregate["SessionTasksSubscribeAggregate"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_name-List
        List-Utf8 subscription_table_name-List
        Utf8 subscribe_type-Last
        Utf8 update_type-Last
        List-Int64 timestamp-List
        List-Int64 timestamp-Last-List
    }
    SessionTasksSubscribe["SessionTasksSubscribe"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        Utf8 subscription_name
        Utf8 subscription_table_name
    }
    group_by_tasks_processors_subscriptions_subjects_p["group_by_tasks_processors_subscriptions_subjects_p"] {
        List-Utf8 agg_columns "['subscription_name','subscription_table_name']"
        List-Utf8 agg_operators "['List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "SessionTasksSubscribe"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    group_by_tasks_processors_publications_p["group_by_tasks_processors_publications_p"] {
        List-Utf8 agg_columns "['publication_subscription_name','publication_subscription_table_name']"
        List-Utf8 agg_operators "['List','List']"
        Boolean cpu "false"
        Utf8 lhs_name "select_processors_publications_t"
        List-Utf8 lhs_values "['session_name','processor_name','processor_type']"
        Utf8 operator "GroupBy"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    join_tasks_processors_publications_p["join_tasks_processors_publications_p"] {
        Boolean cpu "false"
        Utf8 lhs_fk "processor_name"
        Utf8 lhs_name "group_by_tasks_processors_subscriptions_subjects_t"
        Utf8 lhs_pk "processor_name"
        Utf8 operator "Join"
        Utf8 rhs_fk "processor_name"
        Utf8 rhs_name "group_by_tasks_processors_publications_t"
        Utf8 rhs_pk "processor_name"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    select_tasks_processors_publications_p["select_tasks_processors_publications_p"] {
        Boolean cpu "false"
        Utf8 lhs_name "join_tasks_processors_publications_t"
        List-Utf8 as_columns "['','','','','subscription_names','subscription_table_names','publication_names','publication_table_names']"
        List-Utf8 lhs_values "['session_name','task_name','processor_name','processor_type','subscription_name-List','subscription_table_name-List','publication_subscription_name-List','publication_subscription_table_name-List']"
        Utf8 operator "Select"
        Utf8 stream "AccumulateLHSAccumulateRHS"
    }
    SessionTasksSubscribePublish["SessionTasksSubscribePublish"] {
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
        List-Utf8 publication_names
        List-Utf8 publication_table_names
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, MappableTrait, MessageBuilderTrait, Table, TableBuilderTrait, TablePublication, TableTrait, create_session_tasks_subscribe_publish_batch};
    use phymes_diagnostics::HashMap;

    use crate::{CustomAgentsBuilderTrait, SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream, UserSession, create_message_map};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_tasks_subscribe_publish_session() -> Result<()> {
        // Initialize the session
        let tasks_publish_subscribe_session = TasksSubscribePublishSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            tasks_publish_subscribe_session.as_mermaid_flowchart(),
            false,
            )?
            .with_state_from_mermaid_erdiagram(tasks_publish_subscribe_session.as_mermaid_erdiagram(), false, true)?
            .with_name(tasks_publish_subscribe_session.session_context_name)
            .with_diagnostics(true)
            .add_processor_subjects()?
            .with_max_iter(1) // DM: prevent continued execution after the final superstep for testing
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test session data
        let user_agent_session = UserSession::default();
        let user_session_ctx = Arc::new(RwLock::new(user_agent_session
            .build()
            .with_name(user_agent_session.session_context_name)
            .with_diagnostics(true)
            .build_with_tables()?
        ));

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

        // 1. Message to trigger the first superstep
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
            vec!["SessionProcessors", "cmp_processors_publications_p"], vec!["cmp_processors_publications_t", "filter_processors_publications_p"], vec!["filter_processors_publications_t", "select_processors_publications_p"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"],
            vec!["Replace"], vec!["Replace"], vec!["Replace"],
            vec!["Replace"], vec!["Replace"], vec!["Replace"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_table_names = vec![vec!["group_by_tasks_run_log_timestamp_t"], vec!["select_tasks_run_log_timestamp_t"],
            vec!["cmp_processors_subscriptions_t"], vec!["filter_processors_subscriptions_t"], vec!["select_processors_subscriptions_t"],
            vec!["cmp_processors_publications_t"], vec!["filter_processors_publications_t"], vec!["select_processors_publications_t"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let session_names = task_names.iter().map(|_| tasks_publish_subscribe_session.session_context_name.to_string()).collect::<Vec<_>>();

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
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        { // Test supserstep 1
            { // Debug any errors
                let subjects_reading = session_ctx_arc.read();
                let table_reading = subjects_reading
                    .get_states()
                    .get(AvailableSubjects::SessionErrors.to_string().as_str())
                    .unwrap()
                    .read();
                println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            }

            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading.get_states().get("select_tasks_run_log_timestamp_t").unwrap().read();
            let column = table_reading.get_column_as_vec_str("task_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_inbox_task_name", "filter_and_join_session_contexts_by_email_outbox_task_name", "filter_session_contexts_by_email_task_name", "filter_user_info_by_email_task_name", "join_session_contexts_with_mermaid_diagrams_task_name", "user_session"]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            for timestamp in column {
                assert!(timestamp > 0);
            }

            let table_reading = session_reading.get_states().get("select_processors_subscriptions_t").unwrap().read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_inbox_processor_name", "filter_and_join_session_contexts_by_email_inbox_processor_name", "filter_session_contexts_by_email_processor_name", "filter_session_contexts_by_email_processor_name", "filter_session_contexts_by_email_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "filter_user_info_by_email_processor_name", "filter_user_info_by_email_processor_name", "filter_user_info_by_email_processor_name", "filter_and_join_session_contexts_by_email_outbox_processor_name", "filter_and_join_session_contexts_by_email_outbox_processor_name", "filter_and_join_session_contexts_by_email_outbox_processor_name", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(column, ["ExtractTabular", "ExtractTabular", "Join", "Join", "Join", "Join", "Join", "Join", "Join", "Join", "Join", "DataSummaryProcessor", "DataSummaryProcessor", "DataSummaryProcessor", "ProcessorEcho"]);
            let column = table_reading.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(column, ["OnUpdateFullTable", "AlwaysFullTable", "AlwaysLastRecordBatch", "OnUpdateFullTable", "AlwaysFullTable", "AlwaysLastRecordBatch", "OnUpdateFullTable", "AlwaysFullTable", "AlwaysLastRecordBatch", "OnUpdateFullTable", "AlwaysFullTable", "AlwaysLastRecordBatch", "OnUpdateFullTable", "OnUpdateFullTable", "OnUpdateFullTable"]);
            let column = table_reading.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(column, ["UserJson", "filter_and_join_session_contexts_by_email_inbox_processor_name", "filter_session_contexts_by_email_processor_name", "UserInbox", "UserSessionContexts", "join_session_contexts_with_mermaid_diagrams_processor_name", "JoinUserInboxSessionContexts", "BuilderMermaid", "filter_user_info_by_email_processor_name", "UserInbox", "User", "filter_and_join_session_contexts_by_email_outbox_processor_name", "filter_user_info_by_email_table_name", "JoinUserInboxSessionContextsMermaid", "AssistantJson"]);
            let column = table_reading.get_column_as_vec_str("subscribe_type");
            assert_eq!(column, ["All", "All", "All", "All", "All", "All", "All", "All", "All", "All", "All", "Any", "Any", "Any", "All"]);
            let column = table_reading.get_column_as_vec_str("update_type");
            assert_eq!(column, ["TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate"]);
            let column = table_reading.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

            let table_reading = session_reading.get_states().get("select_processors_publications_t").unwrap().read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_inbox_processor_name", "filter_session_contexts_by_email_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "filter_user_info_by_email_processor_name", "filter_and_join_session_contexts_by_email_outbox_processor_name", "user_session", "user_session", "user_session", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(column, ["ExtractTabular", "Join", "Join", "Join", "DataSummaryProcessor", "ProcessorEcho", "ProcessorEcho", "ProcessorEcho", "ProcessorEcho", "ProcessorEcho"]);
            let column = table_reading.get_column_as_vec_str("publication_subscription_name");
            assert_eq!(column, ["Replace", "Replace", "Replace", "Replace", "Replace", "Extend", "Extend", "Extend", "Replace", "Replace"]);
            let column = table_reading.get_column_as_vec_str("publication_subscription_table_name");
            assert_eq!(column, ["UserInbox", "JoinUserInboxSessionContexts", "JoinUserInboxSessionContextsMermaid", "filter_user_info_by_email_table_name", "AssistantJson", "BuilderMermaid", "User", "UserSessionContexts", "UserJson", "AssistantJson"]);
            let column = table_reading.get_column_as_vec_str("subscribe_type");
            assert_eq!(column, ["All", "All", "All", "All", "Any", "All", "All", "All", "All", "All"]);
            let column = table_reading.get_column_as_vec_str("update_type");
            assert_eq!(column, ["TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate"]);
            let column = table_reading.get_column_as_vec_primitive::<u8>("is_subscription")?;
            assert_eq!(column, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        }

        // 2. Message to trigger the second superstep
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
            "group_by_tasks_processors_subscriptions_p"
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", 
            "Join",    
            "Join",
            "Join", 
            "Select",
            "GroupBy",
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
            vec!["select_tasks_processors_subscriptions_subjects_t", "group_by_tasks_processors_subscriptions_p"],
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
            vec!["SessionTasksSubscribeAggregate"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let session_names = task_names.iter().map(|_| tasks_publish_subscribe_session.session_context_name.to_string()).collect::<Vec<_>>();

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

        { // Test supserstep 2
            { // Debug any errors
                let subjects_reading = session_ctx_arc.read();
                let table_reading = subjects_reading
                    .get_states()
                    .get(AvailableSubjects::SessionErrors.to_string().as_str())
                    .unwrap()
                    .read();
                println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            }
            // { // Check metrics
            //     let subjects_reading = session_ctx_arc.read();
            //     let table_reading = subjects_reading
            //         .get_states()
            //         .get(AvailableSubjects::SessionMetrics.to_string().as_str())
            //         .unwrap()
            //         .read();
            //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            // }

            let session_reading = session_ctx_arc.read();
            // let table_reading = session_reading.get_states().get("select_tasks_processors_subscriptions_subjects_t").unwrap().read();
            // let column = table_reading.get_column_as_vec_str("session_name");
            // assert_eq!(column, ["user_session","user_session","user_session","user_session","user_session","user_session","user_session","user_session","user_session","user_session","user_session"]);
            // let column = table_reading.get_column_as_vec_str("task_name");
            // assert_eq!(column, ["user_session","user_session","join_session_contexts_with_mermaid_diagrams_task_name","join_session_contexts_with_mermaid_diagrams_task_name","filter_and_join_session_contexts_by_email_outbox_task_name","filter_user_info_by_email_task_name","filter_session_contexts_by_email_task_name","filter_user_info_by_email_task_name","filter_and_join_session_contexts_by_email_inbox_task_name","filter_session_contexts_by_email_task_name","filter_and_join_session_contexts_by_email_outbox_task_name"]);
            // let column = table_reading.get_column_as_vec_str("processor_name");
            // assert_eq!(column, ["user_session","user_session","join_session_contexts_with_mermaid_diagrams_processor_name","join_session_contexts_with_mermaid_diagrams_processor_name","filter_and_join_session_contexts_by_email_outbox_processor_name","filter_user_info_by_email_processor_name","filter_session_contexts_by_email_processor_name","filter_user_info_by_email_processor_name","filter_and_join_session_contexts_by_email_inbox_processor_name","filter_session_contexts_by_email_processor_name","filter_and_join_session_contexts_by_email_outbox_processor_name"]);
            // let column = table_reading.get_column_as_vec_str("processor_type");
            // assert_eq!(column, ["ProcessorEcho","ProcessorEcho","Join","Join","DataSummaryProcessor","Join","Join","Join","ExtractTabular","Join","DataSummaryProcessor"]);
            // let column = table_reading.get_column_as_vec_str("subscription_name");
            // assert_eq!(column, ["OnUpdateFullTable", "OnUpdateFullTable", "AlwaysFullTable", "OnUpdateFullTable", "OnUpdateFullTable", "AlwaysFullTable", "OnUpdateFullTable", "OnUpdateFullTable", "OnUpdateFullTable", "AlwaysFullTable", "OnUpdateFullTable"]);
            // let column = table_reading.get_column_as_vec_str("subscription_table_name");
            // assert_eq!(column, ["AssistantJson", "AssistantJson", "BuilderMermaid", "JoinUserInboxSessionContexts", "JoinUserInboxSessionContextsMermaid", "User", "UserInbox", "UserInbox", "UserJson", "UserSessionContexts", "filter_user_info_by_email_table_name"]);
            // let column = table_reading.get_column_as_vec_str("subscribe_type");
            // assert_eq!(column, ["All", "All", "All", "All", "Any", "All", "All", "All", "All", "All", "Any"]);
            // let column = table_reading.get_column_as_vec_str("update_type");
            // assert_eq!(column, ["TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate"]);
            // let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            // for timestamp in column {
            //     assert!(timestamp > 0);
            // }
            // let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp-Last")?;
            // for timestamp in column {
            //     assert!(timestamp > 0);
            // }            
            let table_reading = session_reading.get_states().get("SessionTasksSubscribeAggregate").unwrap().read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["user_session", "user_session", "user_session", "user_session", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("task_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_outbox_task_name", "filter_and_join_session_contexts_by_email_inbox_task_name", "filter_session_contexts_by_email_task_name", "filter_user_info_by_email_task_name", "join_session_contexts_with_mermaid_diagrams_task_name", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_outbox_processor_name", "filter_and_join_session_contexts_by_email_inbox_processor_name", "filter_session_contexts_by_email_processor_name", "filter_user_info_by_email_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(column, ["DataSummaryProcessor", "ExtractTabular", "Join", "Join", "Join", "ProcessorEcho"]);
            let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_name-List")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["OnUpdateFullTable", "OnUpdateFullTable", "OnUpdateFullTable", "OnUpdateFullTable", "AlwaysFullTable", "AlwaysFullTable", "OnUpdateFullTable", "AlwaysFullTable", "OnUpdateFullTable", "OnUpdateFullTable", "OnUpdateFullTable"]);
            let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_table_name-List")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["JoinUserInboxSessionContextsMermaid", "filter_user_info_by_email_table_name", "UserJson", "UserInbox", "UserSessionContexts", "User", "UserInbox", "BuilderMermaid", "JoinUserInboxSessionContexts", "AssistantJson", "AssistantJson"]);
            let column = table_reading.get_column_as_vec_str("subscribe_type-Last");
            assert_eq!(column, ["Any", "All", "All", "All", "All", "All"]);
            let column = table_reading.get_column_as_vec_str("update_type-Last");
            assert_eq!(column, ["TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate", "TableChangedSinceLastRunUpdate"]);
            let column = table_reading.get_column_as_vec_nested_primitive::<i64>("timestamp-List")?;
            for timestamps in column {
                for timestamp in timestamps {
                    assert!(timestamp > 0);
                }
            }
            let column = table_reading.get_column_as_vec_nested_primitive::<i64>("timestamp-Last-List")?;
            for timestamps in column {
                for timestamp in timestamps {
                    assert!(timestamp > 0);
                }
            }
        }

        // Calculate the tasks subscribe
        session_ctx_arc.read().tasks_subscribe()?;

        { // Test the tasks subscribe
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading.get_states().get("SessionTasksSubscribe").unwrap().read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, ["user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("task_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_outbox_task_name", "filter_and_join_session_contexts_by_email_outbox_task_name", "filter_and_join_session_contexts_by_email_inbox_task_name", "filter_session_contexts_by_email_task_name", "filter_session_contexts_by_email_task_name", "filter_user_info_by_email_task_name", "filter_user_info_by_email_task_name", "join_session_contexts_with_mermaid_diagrams_task_name", "join_session_contexts_with_mermaid_diagrams_task_name", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_outbox_processor_name", "filter_and_join_session_contexts_by_email_outbox_processor_name", "filter_and_join_session_contexts_by_email_inbox_processor_name", "filter_session_contexts_by_email_processor_name", "filter_session_contexts_by_email_processor_name", "filter_user_info_by_email_processor_name", "filter_user_info_by_email_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "join_session_contexts_with_mermaid_diagrams_processor_name", "user_session", "user_session"]);
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(column, ["DataSummaryProcessor", "DataSummaryProcessor", "ExtractTabular", "Join", "Join", "Join", "Join", "Join", "Join", "ProcessorEcho", "ProcessorEcho"]);
            let column = table_reading.get_column_as_vec_str("subscription_name");
            assert_eq!(column, [
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "AlwaysFullTable",
                "AlwaysFullTable",
                "OnUpdateFullTable",
                "AlwaysFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
            ]);
            let column = table_reading.get_column_as_vec_str("subscription_table_name");
            assert_eq!(column, [
                "JoinUserInboxSessionContextsMermaid",
                "filter_user_info_by_email_table_name",
                "UserJson",
                "UserInbox",
                "UserSessionContexts",
                "User",
                "UserInbox",
                "BuilderMermaid",
                "JoinUserInboxSessionContexts",
                "AssistantJson",
                "AssistantJson",
            ]);
        }

        // 3. Message to trigger the third superstep
        let task_names = vec!["select_tasks_processors_publications_t", 
            "select_tasks_processors_publications_t", 
            "select_tasks_processors_publications_t",
            "select_tasks_processors_publications_t",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_names = vec!["group_by_tasks_processors_subscriptions_subjects_p", 
            "group_by_tasks_processors_publications_p",
            "join_tasks_processors_publications_p",
            "select_tasks_processors_publications_p", 
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", 
            "GroupBy",
            "Join",
            "Select",
        ].into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let subscription_names = vec![vec!["OnUpdateFullTable","AlwaysFullTable"],
            vec!["AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable","AlwaysFullTable"], 
            vec!["AlwaysFullTable","AlwaysFullTable"], 
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let subscription_table_names = vec![vec!["SessionTasksSubscribe", "group_by_tasks_processors_subscriptions_subjects_p"], 
            vec!["select_processors_publications_t", "group_by_tasks_processors_publications_p"], 
            vec!["group_by_tasks_processors_subscriptions_subjects_t", "group_by_tasks_processors_publications_t", "join_tasks_processors_publications_p"], 
            vec!["join_tasks_processors_publications_t", "select_tasks_processors_publications_p"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], 
            vec!["Replace"],      
            vec!["Replace"],
            vec!["Replace"],
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let publication_table_names = vec![vec!["group_by_tasks_processors_subscriptions_subjects_t"], 
            vec!["group_by_tasks_processors_publications_t"],
            vec!["join_tasks_processors_publications_t"],
            vec!["SessionTasksSubscribePublish"], 
        ].into_iter().map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let session_names = task_names.iter().map(|_| tasks_publish_subscribe_session.session_context_name.to_string()).collect::<Vec<_>>();

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

        { // Test supserstep 3
            { // Debug errors
                let subjects_reading = session_ctx_arc.read();
                let table_reading = subjects_reading
                    .get_states()
                    .get(AvailableSubjects::SessionErrors.to_string().as_str())
                    .unwrap()
                    .read();
                println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            }

            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading.get_states().get("SessionTasksSubscribePublish").unwrap().read();
            let column = table_reading.get_column_as_vec_str("session_name");
            assert_eq!(column, [
                "user_session",
                "user_session",
                "user_session",
                "user_session",
                "user_session",
                "user_session"
            ]);
            let column = table_reading.get_column_as_vec_str("processor_name");
            assert_eq!(column, ["filter_and_join_session_contexts_by_email_inbox_processor_name",
                "filter_and_join_session_contexts_by_email_outbox_processor_name",
                "filter_session_contexts_by_email_processor_name",
                "filter_user_info_by_email_processor_name",
                "join_session_contexts_with_mermaid_diagrams_processor_name",
                "user_session",
            ]);
            let column = table_reading.get_column_as_vec_str("processor_type");
            assert_eq!(column, ["ExtractTabular",
                "DataSummaryProcessor",
                "Join",
                "Join",
                "Join",
                "ProcessorEcho",
            ]);
            let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["OnUpdateFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "AlwaysFullTable",
                "AlwaysFullTable",
                "OnUpdateFullTable",
                "AlwaysFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
                "OnUpdateFullTable",
            ]);
            let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["UserJson",
                "JoinUserInboxSessionContextsMermaid",
                "filter_user_info_by_email_table_name",
                "UserInbox",
                "UserSessionContexts",
                "User",
                "UserInbox",
                "BuilderMermaid",
                "JoinUserInboxSessionContexts",
                "AssistantJson",
                "AssistantJson",
            ]);
            let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["Replace",
                "Replace",
                "Replace",
                "Replace",
                "Replace",
                "Extend",
                "Extend",
                "Extend",
                "Replace",
                "Replace",
            ]);
            let column = table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
            let flattened = column.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flattened, ["UserInbox",
                "AssistantJson",
                "JoinUserInboxSessionContexts",
                "filter_user_info_by_email_table_name",
                "JoinUserInboxSessionContextsMermaid",
                "BuilderMermaid",
                "User",
                "UserSessionContexts",
                "UserJson",
                "AssistantJson",
            ]);
        }

        Ok(())
    }
}
