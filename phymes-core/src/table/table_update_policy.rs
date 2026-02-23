use std::fmt::Debug;

use phymes_diagnostics::HashMap;

use crate::{MappableTrait, StateMap, TableSubscription, TableTrait};

/// Determine when a table has been updated
pub trait TableUpdatePolicyTrait: MappableTrait + Debug + Send + Sync {
    /// Determine which tables have been updated with respect to the query processor
    ///
    /// # Notes
    /// * The output is the input to [TableUpdatePolicyTrait]
    ///
    /// # Arguments
    ///
    /// * `subscriptions` - Slice of `TableSubscription`s for the processor
    /// * `last_run` - timestamp of the last run
    /// * `subjects_change_log` - `HashMap` of the subjects and when they were changed last
    /// * `state` - `HashMap` of the subject tables
    fn determine_updates(
        &self,
        subscriptions: &[TableSubscription],
        last_run: &i64,
        subjects_change_log: &HashMap<String, i64>,
        state: &StateMap,
    ) -> HashMap<String, bool>;
    fn new_box() -> Box<dyn TableUpdatePolicyTrait>
    where
        Self: Sized;
    fn clone_boxed(&self) -> Box<dyn TableUpdatePolicyTrait>;
}

/// If a table has [RecordBatch]es, consider the table updated
/// 
/// [RecordBatch]: arrow::record_batch::RecordBatch
#[derive(Default, Debug, Clone)]
pub struct TableHasBatchesUpdate {}

impl TableUpdatePolicyTrait for TableHasBatchesUpdate {
    fn determine_updates(
        &self,
        subscriptions: &[TableSubscription],
        _last_run: &i64,
        _subjects_change_log: &HashMap<String, i64>,
        state: &StateMap,
    ) -> HashMap<String, bool> {
        subscriptions
            .iter()
            .map(|s| {
                if let Some(table) = state.get(s.get_table_name()) {
                    if !table.read().get_record_batches().is_empty() {
                        (s.get_table_name().to_string(), true)
                    } else {
                        (s.get_table_name().to_string(), false)
                    }
                } else {
                    (s.get_table_name().to_string(), false)
                }
            })
            .collect::<HashMap<_, _>>()
    }
    fn new_box() -> Box<dyn TableUpdatePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableUpdatePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for TableHasBatchesUpdate {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// If a table has been updated after the task/processor has ran, consider the table updated
#[derive(Default, Debug, Clone)]
pub struct TableChangedSinceLastRunUpdate {}

impl TableUpdatePolicyTrait for TableChangedSinceLastRunUpdate {
    fn determine_updates(
        &self,
        subscriptions: &[TableSubscription],
        last_run: &i64,
        subjects_change_log: &HashMap<String, i64>,
        _state: &StateMap,
    ) -> HashMap<String, bool> {
        subscriptions
            .iter()
            .map(|s| {
                if let Some(timestamp) = subjects_change_log.get(s.get_table_name()) {
                    if timestamp > last_run {
                        (s.get_table_name().to_string(), true)
                    } else {
                        (s.get_table_name().to_string(), false)
                    }
                } else {
                    (s.get_table_name().to_string(), false)
                }
            })
            .collect::<HashMap<_, _>>()
    }
    fn new_box() -> Box<dyn TableUpdatePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableUpdatePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for TableChangedSinceLastRunUpdate {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// if a table exists, consider the table updated
#[derive(Default, Debug, Clone)]
pub struct TableExistsUpdate {}

impl TableUpdatePolicyTrait for TableExistsUpdate {
    fn determine_updates(
        &self,
        subscriptions: &[TableSubscription],
        _last_run: &i64,
        _subjects_change_log: &HashMap<String, i64>,
        state: &StateMap,
    ) -> HashMap<String, bool> {
        subscriptions
            .iter()
            .map(|s| {
                if state.contains_key(s.get_table_name()) {
                    (s.get_table_name().to_string(), true)
                } else {
                    (s.get_table_name().to_string(), false)
                }
            })
            .collect::<HashMap<_, _>>()
    }
    fn new_box() -> Box<dyn TableUpdatePolicyTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn TableUpdatePolicyTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for TableExistsUpdate {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

mod test_update_policy {
    use super::*;

    #[allow(dead_code)]
    pub fn make_test_subjects_change_log() -> HashMap<String, i64> {
        let mut change_log = HashMap::<String, i64>::new();
        change_log.insert("t1".to_string(), 0);
        change_log.insert("t2".to_string(), 1);
        change_log.insert("t3".to_string(), 2);
        change_log
    }
}

#[cfg(test)]
mod tests {
    use crate::table::table_subscribe_policy::test_subscribe_policy;

    use super::*;

    #[test]
    fn test_table_exists_update() {
        let mut state = test_subscribe_policy::make_test_state();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let changes = test_update_policy::make_test_subjects_change_log();
        let up = TableExistsUpdate::new_box();

        let updates = up.determine_updates(&subscriptions, &0, &changes, &state);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), true);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);

        let _ = state.remove("t1").unwrap();

        let updates = up.determine_updates(&subscriptions, &0, &changes, &state);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);
    }

    #[test]
    fn test_table_has_batches_update() {
        let mut state = test_subscribe_policy::make_test_state();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let changes = test_update_policy::make_test_subjects_change_log();
        let up = TableHasBatchesUpdate::new_box();

        let updates = up.determine_updates(&subscriptions, &0, &changes, &state);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), true);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);

        state
            .get_mut("t1")
            .unwrap()
            .try_write()
            .unwrap()
            .get_record_batches_mut()
            .clear();

        let updates = up.determine_updates(&subscriptions, &0, &changes, &state);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);
    }

    #[test]
    fn test_table_changed_since_last_run_update() {
        let state = test_subscribe_policy::make_test_state();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let changes = test_update_policy::make_test_subjects_change_log();
        let up = TableChangedSinceLastRunUpdate::new_box();

        let updates = up.determine_updates(&subscriptions, &1, &changes, &state);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false);
        updates_test.insert("t2".to_string(), false);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);
    }
}
