use std::{
    borrow::Cow, fmt::{Debug, Display}
};

/// `name=value` pairs identifying a metric. This concept is called various things
/// in various different systems:
///
/// "labels" in
/// [prometheus](https://prometheus.io/docs/concepts/data_model/) and
/// "tags" in
/// [InfluxDB](https://docs.influxdata.com/influxdb/v1.8/write_protocols/line_protocol_tutorial/)
/// , "attributes" in [open
/// telemetry]<https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/metrics/datamodel.md>,
/// etc.
///
/// As the name and value are expected to mostly be constant strings,
/// use a [`Cow`] to avoid copying / allocations in this common case.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    name: Cow<'static, str>,
    value: Cow<'static, str>,
}

impl Label {
    /// Create a new [`Label`]
    pub fn new(name: impl Into<Cow<'static, str>>, value: impl Into<Cow<'static, str>>) -> Self {
        let name = name.into();
        let value = value.into();
        Self { name, value }
    }

    /// Returns the name of this label
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Returns the value of this label
    pub fn value(&self) -> &str {
        self.value.as_ref()
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}