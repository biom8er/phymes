use anyhow::Result;
use phymes_data::items_to_list;

pub trait DynamicNetworkBuilderTrait {
    /// Subjects to listen for
    fn subject_names(&self) -> Vec<String>;

    /// ER Diagram subjects as `Bytes`
    fn erdiagram_subject_subscriptions(&self, subject_names: &[&str]) -> String {
        let mut subscriptions_vec = Vec::new();
        for subject_name in subject_names {
            let line = format!(
                r#"{subject_name}["{subject_name}"] {{
        List-UInt8 bytes
    }}"#
            );
            subscriptions_vec.push(line);
        }
        subscriptions_vec.join("\n\t")
    }

    /// List of subjects compatible with List-Utf8
    fn subject_columns(&self) -> Result<String> {
        items_to_list(
            &self
                .subject_names()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    }

    /// Flowchart subjects subscriptions part 1
    fn flowchart_subject_subscriptions_1(
        &self,
        subject_names: &[&str],
        processor: &str,
        subscription: &str,
    ) -> String {
        let mut subscriptions_vec = Vec::new();
        for subject_name in subject_names {
            let line = format!("{subject_name}-subject-.->|{subscription}|{processor}-subscribe");
            subscriptions_vec.push(line);
        }
        subscriptions_vec.join("\n\t\t")
    }

    /// Flowchart subjects subscriptions part 2
    fn flowchart_subject_subscriptions_2(&self, subject_names: &[&str]) -> String {
        let mut subscriptions_vec = Vec::new();
        for subject_name in subject_names {
            let line = format!("{subject_name}-subject@{{shape: doc, label: {subject_name}}}");
            subscriptions_vec.push(line);
        }
        subscriptions_vec.join("\n\t")
    }
}
