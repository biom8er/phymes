use phymes_core::{MappableTrait, Subscription};
use phymes_diagnostics::HashMap;
use std::fmt::Debug;

/// Determine when a subject has been updated
pub trait UpdateEventTrait: MappableTrait + Debug + Send + Sync {
    /// Determine which subjects have been updated with respect to the query processor
    ///
    /// # Notes
    /// * The output is the input to [UpdateEventTrait]
    ///
    /// # Arguments
    ///
    /// * `subscriptions` - Slice of `Subscription`s for the processor
    /// * `last_run` - timestamp of the last run
    /// * `subjects_change_log` - `HashMap` of the subjects and when they were changed last
    fn determine_updates(
        &self,
        subscriptions: &[Subscription],
        last_run: &i64,
        subjects_change_log: &HashMap<String, i64>,
    ) -> HashMap<String, bool>;
    fn new_box() -> Box<dyn UpdateEventTrait>
    where
        Self: Sized;
    fn clone_boxed(&self) -> Box<dyn UpdateEventTrait>;
}

/// If a subject has [RecordBatch]es, consider the subject updated
///
/// [RecordBatch]: arrow::record_batch::RecordBatch
#[derive(Default, Debug, Clone)]
pub struct SubjectHasBatchesUpdate {}

impl UpdateEventTrait for SubjectHasBatchesUpdate {
    fn determine_updates(
        &self,
        subscriptions: &[Subscription],
        _last_run: &i64,
        subjects_change_log: &HashMap<String, i64>,
    ) -> HashMap<String, bool> {
        subscriptions
            .iter()
            .map(|s| {
                if let Some(timestamp) = subjects_change_log.get(s.subject_name()) {
                    if timestamp > &0_i64 {
                        (s.subject_name().to_string(), true)
                    } else {
                        (s.subject_name().to_string(), false)
                    }
                } else {
                    (s.subject_name().to_string(), false)
                }
            })
            .collect::<HashMap<_, _>>()
    }
    fn new_box() -> Box<dyn UpdateEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn UpdateEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for SubjectHasBatchesUpdate {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// If a subject has been updated after the task/processor has ran, consider the subject updated
#[derive(Default, Debug, Clone)]
pub struct SubjectChangedSinceLastRunUpdate {}

impl UpdateEventTrait for SubjectChangedSinceLastRunUpdate {
    fn determine_updates(
        &self,
        subscriptions: &[Subscription],
        last_run: &i64,
        subjects_change_log: &HashMap<String, i64>,
    ) -> HashMap<String, bool> {
        subscriptions
            .iter()
            .map(|s| {
                if let Some(timestamp) = subjects_change_log.get(s.subject_name()) {
                    if timestamp > last_run {
                        (s.subject_name().to_string(), true)
                    } else {
                        (s.subject_name().to_string(), false)
                    }
                } else {
                    (s.subject_name().to_string(), false)
                }
            })
            .collect::<HashMap<_, _>>()
    }
    fn new_box() -> Box<dyn UpdateEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn UpdateEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for SubjectChangedSinceLastRunUpdate {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// if a subject exists, consider the subject updated
#[derive(Default, Debug, Clone)]
pub struct SubjectExistsUpdate {}

impl UpdateEventTrait for SubjectExistsUpdate {
    fn determine_updates(
        &self,
        subscriptions: &[Subscription],
        _last_run: &i64,
        subjects_change_log: &HashMap<String, i64>,
    ) -> HashMap<String, bool> {
        subscriptions
            .iter()
            .map(|s| {
                if subjects_change_log.contains_key(s.subject_name()) {
                    (s.subject_name().to_string(), true)
                } else {
                    (s.subject_name().to_string(), false)
                }
            })
            .collect::<HashMap<_, _>>()
    }
    fn new_box() -> Box<dyn UpdateEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn UpdateEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for SubjectExistsUpdate {
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
    use crate::event::subscribe_event::test_subscribe_policy;

    use super::*;

    #[test]
    fn test_subject_exists_update() {
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let mut changes = test_update_policy::make_test_subjects_change_log();
        let up = SubjectExistsUpdate::new_box();

        let updates = up.determine_updates(&subscriptions, &0, &changes);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), true);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);

        let _ = changes.remove("t1").unwrap();

        let updates = up.determine_updates(&subscriptions, &0, &changes);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);
    }

    #[test]
    fn test_subject_has_batches_update() {
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let mut changes = test_update_policy::make_test_subjects_change_log();
        let up = SubjectHasBatchesUpdate::new_box();

        let updates = up.determine_updates(&subscriptions, &0, &changes);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false); // DM: need to change the logic so that this is `true`
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);

        let _ = changes.remove("t1").unwrap();

        let updates = up.determine_updates(&subscriptions, &0, &changes);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false);
        updates_test.insert("t2".to_string(), true);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);
    }

    #[test]
    fn test_subject_changed_since_last_run_update() {
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let changes = test_update_policy::make_test_subjects_change_log();
        let up = SubjectChangedSinceLastRunUpdate::new_box();

        let updates = up.determine_updates(&subscriptions, &1, &changes);
        let mut updates_test = HashMap::<String, bool>::new();
        updates_test.insert("t1".to_string(), false);
        updates_test.insert("t2".to_string(), false);
        updates_test.insert("t3".to_string(), true);
        assert_eq!(updates, updates_test);
    }
}
