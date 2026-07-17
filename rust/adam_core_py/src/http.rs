use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::fmt;
use std::time::Duration;

pub(crate) fn percent_encode(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

#[derive(Debug)]
pub(crate) struct HttpFailure {
    pub status: Option<u16>,
    pub message: String,
}

impl fmt::Display for HttpFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

pub(crate) fn get_text_with_status(
    url: &str,
    timeout: Duration,
    service: &str,
) -> Result<String, HttpFailure> {
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let response = agent.get(url).call().map_err(|error| {
        let status = match &error {
            ureq::Error::Status(status, _) => Some(*status),
            ureq::Error::Transport(_) => None,
        };
        HttpFailure {
            status,
            message: format!("{service} request failed: {error}"),
        }
    })?;
    response.into_string().map_err(|error| HttpFailure {
        status: None,
        message: format!("{service} response read failed: {error}"),
    })
}

pub(crate) fn get_text(url: &str, timeout: Duration, service: &str) -> PyResult<String> {
    get_text_with_status(url, timeout, service)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
