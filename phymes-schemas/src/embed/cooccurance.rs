use arrow::datatypes::{DataType, Field};

// The context that the co-occurance was counted
pub enum CoOccuranceContext {
    Document,
    Paragraph,
    Sentence
}

fn create_cooccurrance_counts_fields() -> Vec<Field> {
    let field_names = ["entity_1", "entity_2", "context"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["count"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    fields_vec
}

/// Co-occurance parameters
/// 
/// # Notes
/// * w_document = document context co-occurance weight
/// * w_paragraph = document context co-occurance weight
/// * w_sentence = document context co-occurance weight
/// * alpha = score weight
/// * x_bar = Z-score background distribution mean
/// * s = Z-score background distribution variance
/// 
/// # Co-occurance calculations
/// * See <https://doi.org/10.1093/bioinformatics/btt677>
/// * Weighted count, C(i,j) = SUM[k=1, n](w_d*δ_d_k(i,j) + w_p*δ_p_k(i,j) + w_s*δ_s_k(i,j)) 
///   where δ is an indicator function taking into account whether the terms i,j co-occur within the same document (d), paragraph (p), or sentence (s). 
///   w is the co-occurrence weight here set to 1.0, 2.0, and 0.2, respectively.
/// * Score, S(i,j) = C(i, j)^α*((C(i,j)*C(_,_))/(C(i,_)*C(_,j)))^(1-α) 
///   where C(i,_) is the sum over j paired with i, C(_,j) is the sum over all i paired with j, C(_,_) is the sum over all pairs i and j, and α is set to 0.6.
/// * Z-score, Z(i,j) = (X_i - X_bar)/S which is calculated relative to a background distribution. 
///   "To this end, we assume that the empirically observed score distribution is a mixture of lower-scoring random background and the higher-scoring true signal. 
///   We model the background distribution as a Gaussian and estimate its mean as the mode of the mixture distribution. 
///   Because we have empirically observed that the 40th percentile in this case coincides with the mode, we estimate the variance based on the distance between the 20th and 40th percentiles."
fn create_cooccurrance_parameters_fields() -> Vec<Field> {
    let field_names = ["w_document", "w_paragraph", "w_sentence", "alpha", "x_bar", "s"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Float32, false))
        .collect::<Vec<_>>();
    fields_vec
}

fn create_cooccurrance_scores_fields() -> Vec<Field> {
    let field_names = ["entity_1", "entity_2"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["weighted_count", "score", "z_score"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Float32, false))
        .collect::<Vec<_>>());
    fields_vec
}