pub const SUBJECT_SCHEMA_HEADERS: [&str; 2] = ["Column", "Type"];

pub fn get_subject_schema_col_type_by_subject_name(
    active_subject: &str,
    subject_schema_names: &[&str],
    subject_schema_columns: &[&str],
    subject_schema_types: &[&str],
) -> (Vec<String>, Vec<String>) {
    let indices = subject_schema_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    let columns = subject_schema_columns
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    let types = subject_schema_types
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_string())
        .collect::<Vec<_>>();
    (columns, types)
}

pub fn get_subject_num_rows_by_subject_name(
    active_subject: &str,
    subject_names: &[&str],
    subject_num_rows: &[&usize],
) -> Vec<usize> {
    let indices = subject_names
        .iter()
        .enumerate()
        .filter(|(_i, s)| **s == active_subject)
        .map(|(i, _s)| i)
        .collect::<Vec<_>>();
    subject_num_rows
        .iter()
        .enumerate()
        .filter(|(i, _s)| indices.contains(i))
        .map(|(_i, s)| s.to_owned().to_owned())
        .collect::<Vec<_>>()
}
