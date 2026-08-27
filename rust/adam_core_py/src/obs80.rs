use adam_core_rs_coords::{
    optical_obs80_record_batch, parse_optical_obs80_file, parse_optical_obs80_line,
};
use arrow::pyarrow::ToPyArrow;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn parse_batch(raw: &str, strict: bool, file: bool) -> PyResult<arrow_array::RecordBatch> {
    let rows = if file {
        parse_optical_obs80_file(raw, strict).map_err(|error| value_error(error.to_string()))?
    } else {
        vec![parse_optical_obs80_line(raw).map_err(|error| value_error(error.to_string()))?]
    };
    optical_obs80_record_batch(&rows).map_err(|error| value_error(error.to_string()))
}

/// Parse MPC 80-column optical observations and return a quivr-compatible batch.
#[pyfunction]
#[pyo3(signature = (raw, strict=true, file=true))]
fn parse_optical_obs80_arrow<'py>(
    py: Python<'py>,
    raw: &str,
    strict: bool,
    file: bool,
) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| parse_batch(raw, strict, file))?;
    batch
        .to_pyarrow(py)
        .map_err(|error| value_error(error.to_string()))
}

/// Rust-owned parser and Arrow-assembly timing; PyO3 conversion is excluded.
#[pyfunction]
#[pyo3(signature = (raw, reps, trials, warmup_reps=1, strict=true, file=true))]
fn benchmark_parse_optical_obs80(
    raw: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    strict: bool,
    file: bool,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(value_error("reps and trials must be >= 1"));
    }
    let run = || -> PyResult<()> {
        black_box(parse_batch(raw, strict, file)?);
        Ok(())
    };
    let mut output = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            run()?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            run()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        output.push(samples);
    }
    Ok(output)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parse_optical_obs80_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_parse_optical_obs80, module)?)?;
    Ok(())
}
