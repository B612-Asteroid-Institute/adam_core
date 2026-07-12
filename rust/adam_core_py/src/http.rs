use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
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

pub(crate) fn get_text(url: &str, timeout: Duration, service: &str) -> PyResult<String> {
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let response = agent
        .get(url)
        .call()
        .map_err(|err| PyRuntimeError::new_err(format!("{service} request failed: {err}")))?;
    response
        .into_string()
        .map_err(|err| PyRuntimeError::new_err(format!("{service} response read failed: {err}")))
}
