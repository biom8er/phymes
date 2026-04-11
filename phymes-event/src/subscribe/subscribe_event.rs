use arrow::datatypes::SchemaRef;
use phymes_diagnostics::HashMap;
use phymes_subject::MappableTrait;
use std::fmt::Debug;

use crate::Subscription;

/// Determine when all subscriptions are ready
pub trait SubscribeEventTrait: MappableTrait + Debug + Send + Sync {
    /// Check if the subscriptions for a processor are ready to be subscribed to
    ///
    /// # Arguments
    /// * `subscriptions` - Slice of `Subscription`s for the processors
    /// * `updates` - `HashMap` of subscription subject names and if they were updated
    /// * `schemas` - `HashMap` of the subject schemas
    fn check_subscriptions(
        &self,
        subscriptions: &[Subscription],
        updates: &HashMap<String, bool>,
        schemas: &HashMap<String, SchemaRef>,
    ) -> bool;
    fn new_box() -> Box<dyn SubscribeEventTrait>
    where
        Self: Sized;
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait>;
}

/// Always subscribe (dummy subscription check for testing)
#[derive(Default, Debug, Clone)]
pub struct AlwaysSubscribe;

impl SubscribeEventTrait for AlwaysSubscribe {
    fn check_subscriptions(
        &self,
        _subscriptions: &[Subscription],
        _updates: &HashMap<String, bool>,
        _schemas: &HashMap<String, SchemaRef>,
    ) -> bool {
        true
    }
    fn new_box() -> Box<dyn SubscribeEventTrait>
    where
        Self: Sized,
    {
        Box::new(Self)
    }
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AlwaysSubscribe {
    fn get_static_name() -> &'static str {
        "Always"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when any matching table name has been updated
#[derive(Default, Debug, Clone)]
pub struct AnySubscribeNameSubscribe;

impl SubscribeEventTrait for AnySubscribeNameSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[Subscription],
        updates: &HashMap<String, bool>,
        _schemas: &HashMap<String, SchemaRef>,
    ) -> bool {
        let mut is_update_count: usize = 0;
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                is_update_count += 1;
                if *updates.get(subscription.subject_name()).unwrap_or(&false) {
                    return true;
                }
            }
        }
        is_update_count == 0
    }
    fn new_box() -> Box<dyn SubscribeEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AnySubscribeNameSubscribe {
    fn get_static_name() -> &'static str {
        "Any"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when all matching table names has been updated
#[derive(Default, Debug, Clone)]
pub struct AllSubjectNamesSubscribe;

impl SubscribeEventTrait for AllSubjectNamesSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[Subscription],
        updates: &HashMap<String, bool>,
        _schemas: &HashMap<String, SchemaRef>,
    ) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() & !updates.get(subscription.subject_name()).unwrap_or(&true)
            {
                return false;
            }
        }
        true
    }
    fn new_box() -> Box<dyn SubscribeEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AllSubjectNamesSubscribe {
    fn get_static_name() -> &'static str {
        "All"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when any matching table schema has been updated
#[derive(Default, Debug, Clone)]
pub struct AnySubjectSchemaSubscribe;

impl SubscribeEventTrait for AnySubjectSchemaSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[Subscription],
        updates: &HashMap<String, bool>,
        schemas: &HashMap<String, SchemaRef>,
    ) -> bool {
        let mut is_update_count: usize = 0;
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                is_update_count += 1;
                for (subject_name, update) in updates {
                    if schemas
                        .get(subscription.subject_name())
                        .unwrap()
                        .eq(schemas.get(subject_name).unwrap())
                        & *update
                    {
                        return true;
                    }
                }
            }
        }
        is_update_count == 0
    }
    fn new_box() -> Box<dyn SubscribeEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AnySubjectSchemaSubscribe {
    fn get_static_name() -> &'static str {
        "AnySchema"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Subscribe when all matching table schemas has been updated
#[derive(Default, Debug, Clone)]
pub struct AllSubjectSchemasSubscribe;

impl SubscribeEventTrait for AllSubjectSchemasSubscribe {
    fn check_subscriptions(
        &self,
        subscriptions: &[Subscription],
        updates: &HashMap<String, bool>,
        schemas: &HashMap<String, SchemaRef>,
    ) -> bool {
        for subscription in subscriptions.iter() {
            if subscription.is_update() {
                for (subject_name, update) in updates {
                    if schemas
                        .get(subscription.subject_name())
                        .unwrap()
                        .eq(schemas.get(subject_name).unwrap())
                        & !*update
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
    fn new_box() -> Box<dyn SubscribeEventTrait> {
        Box::new(Self {})
    }
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for AllSubjectSchemasSubscribe {
    fn get_static_name() -> &'static str {
        "AllSchema"
    }
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

/// Custom subscription to pull in all of the relevant content for the chat
#[derive(Default, Debug, Clone)]
pub struct ChatContentSubscribe {
    user_message_table_name: String,
    tool_message_table_name: String,
    error_message_table_name: String,
}

#[allow(dead_code)]
impl ChatContentSubscribe {
    pub fn new_box_with_table_names(
        user_message_table_name: &str,
        tool_message_table_name: &str,
        error_message_table_name: &str,
    ) -> Box<dyn SubscribeEventTrait> {
        Box::new(Self {
            user_message_table_name: user_message_table_name.to_string(),
            tool_message_table_name: tool_message_table_name.to_string(),
            error_message_table_name: error_message_table_name.to_string(),
        })
    }
}

impl SubscribeEventTrait for ChatContentSubscribe {
    fn check_subscriptions(
        &self,
        _subscriptions: &[Subscription],
        updates: &HashMap<String, bool>,
        _schemas: &HashMap<String, SchemaRef>,
    ) -> bool {
        // DM: default to false to prevent unwanted subscriptions
        let user = updates.get(&self.user_message_table_name).unwrap_or(&false);
        let tool = updates.get(&self.tool_message_table_name).unwrap_or(&false);
        let error = updates
            .get(&self.error_message_table_name)
            .unwrap_or(&false);
        // DM: assume the config is "other" which is always subscribed too
        let config = !updates.contains_key(&self.user_message_table_name)
            && !updates.contains_key(&self.tool_message_table_name)
            && !updates.contains_key(&self.error_message_table_name);
        *tool || *user || *error || config
    }
    fn new_box() -> Box<dyn SubscribeEventTrait> {
        Box::new(Self {
            // DM: dangerous as the strings needs to stay syncronized with the actual table names
            // in `AvailableinterfaceSubjects` and `AvailableinterfaceSubjects`
            user_message_table_name: "UserMessages".to_string(),
            tool_message_table_name: "ToolMessages".to_string(),
            error_message_table_name: "SessionErrors".to_string(),
        })
    }
    fn clone_boxed(&self) -> Box<dyn SubscribeEventTrait> {
        Box::new(self.clone())
    }
}

impl MappableTrait for ChatContentSubscribe {
    fn get_name(&self) -> &str {
        "ChatContentSubscribe"
    }
}

pub(crate) mod test_subscribe_policy {
    use phymes_subject::test_subject;

    use super::*;

    #[allow(dead_code)]
    pub fn make_test_subject_schemas() -> HashMap<String, SchemaRef> {
        let mut schemas = HashMap::<String, SchemaRef>::new();
        schemas.insert(
            "t1".to_string(),
            test_subject::make_test_subject_schema(0).unwrap(),
        );
        schemas.insert(
            "t2".to_string(),
            test_subject::make_test_subject_schema(0).unwrap(),
        );
        schemas.insert(
            "t3".to_string(),
            test_subject::make_test_subject_schema(0).unwrap(),
        );
        schemas
    }

    #[allow(dead_code)]
    pub fn make_test_subscriptions(use_table_name: bool) -> Vec<Subscription> {
        if use_table_name {
            vec![
                Subscription::OnUpdateLastRecordBatch {
                    subject_name: "t1".to_string(),
                },
                Subscription::OnUpdateLastRecordBatch {
                    subject_name: "t2".to_string(),
                },
                Subscription::AlwaysLastRecordBatch {
                    subject_name: "t3".to_string(),
                },
            ]
        } else {
            vec![
                Subscription::OnUpdateLastRecordBatch {
                    subject_name: "t3".to_string(),
                },
                Subscription::OnUpdateLastRecordBatch {
                    subject_name: "t3".to_string(),
                },
                Subscription::AlwaysLastRecordBatch {
                    subject_name: "t3".to_string(),
                },
            ]
        }
    }

    #[allow(dead_code)]
    pub fn make_test_updates(is_any: bool) -> HashMap<String, bool> {
        let mut updates = HashMap::<String, bool>::new();
        updates.insert("t1".to_string(), true);
        if is_any {
            updates.insert("t2".to_string(), false);
        } else {
            updates.insert("t2".to_string(), true);
        }
        updates.insert("t3".to_string(), false);
        updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_subscribe() {
        let schemas = test_subscribe_policy::make_test_subject_schemas();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let updates = test_subscribe_policy::make_test_updates(true);
        let sub = AlwaysSubscribe::new_box();
        assert!(sub.check_subscriptions(&subscriptions, &updates, &schemas));
    }

    #[test]
    fn test_any_tablename_subscribe() {
        let schemas = test_subscribe_policy::make_test_subject_schemas();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let updates = test_subscribe_policy::make_test_updates(true);
        let sub = AnySubscribeNameSubscribe::new_box();
        assert!(sub.check_subscriptions(&subscriptions, &updates, &schemas));
        let updates = test_subscribe_policy::make_test_updates(false);
        assert!(sub.check_subscriptions(&subscriptions, &updates, &schemas));
    }

    #[test]
    fn test_all_tablename_subscribe() {
        let schemas = test_subscribe_policy::make_test_subject_schemas();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(true);
        let updates = test_subscribe_policy::make_test_updates(true);
        let sub = AllSubjectNamesSubscribe::new_box();
        assert!(!sub.check_subscriptions(&subscriptions, &updates, &schemas));
        let updates = test_subscribe_policy::make_test_updates(false);
        assert!(sub.check_subscriptions(&subscriptions, &updates, &schemas));
    }

    #[test]
    fn test_any_schema_subscribe() {
        let schemas = test_subscribe_policy::make_test_subject_schemas();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(false);
        let updates = test_subscribe_policy::make_test_updates(true);
        let sub = AnySubjectSchemaSubscribe::new_box();
        assert!(sub.check_subscriptions(&subscriptions, &updates, &schemas));
        let updates = test_subscribe_policy::make_test_updates(false);
        assert!(sub.check_subscriptions(&subscriptions, &updates, &schemas));
    }

    #[test]
    fn test_all_schema_subscribe() {
        let schemas = test_subscribe_policy::make_test_subject_schemas();
        let subscriptions = test_subscribe_policy::make_test_subscriptions(false);
        let updates = test_subscribe_policy::make_test_updates(true);
        let sub = AllSubjectSchemasSubscribe::new_box();
        assert!(!sub.check_subscriptions(&subscriptions, &updates, &schemas));
        let updates = test_subscribe_policy::make_test_updates(false);
        assert!(!sub.check_subscriptions(&subscriptions, &updates, &schemas));
    }
}
