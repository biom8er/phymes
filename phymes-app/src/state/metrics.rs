pub fn get_metric_visualizations_by_metric_name(
    active_subject: &str,
    metric_names: &[&str],
    metric_visualizations: &[&str],
) -> Vec<String> {
    let indices = metric_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    metric_visualizations
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>()
}
