use adam_core_rs_coords::{
    trajectory_mjd, trajectory_object_ids, trajectory_segment_index, validate_trajectory,
    TimeArray, TimeScale, TrajectoryData,
};
use arrow::pyarrow::FromPyArrow;
use arrow_array::{Array, ArrayRef, Int64Array, LargeStringArray, RecordBatch, StructArray};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn struct_array<'a>(array: &'a ArrayRef, label: &str) -> PyResult<&'a StructArray> {
    array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| value_error(format!("{label} must be a struct column")))
}

fn child<'a>(array: &'a StructArray, name: &str) -> PyResult<&'a ArrayRef> {
    array
        .column_by_name(name)
        .ok_or_else(|| value_error(format!("missing nested field {name:?}")))
}

fn required_strings(array: &ArrayRef, label: &str) -> PyResult<Vec<String>> {
    let values = array
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| value_error(format!("{label} must be a large string column")))?;
    (0..values.len())
        .map(|row| {
            if values.is_null(row) {
                Err(value_error(format!("{label} must not contain nulls")))
            } else {
                Ok(values.value(row).to_string())
            }
        })
        .collect()
}

fn required_ints(array: &ArrayRef, label: &str) -> PyResult<Vec<i64>> {
    let values = array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| value_error(format!("{label} must be an int64 column")))?;
    (0..values.len())
        .map(|row| {
            if values.is_null(row) {
                Err(value_error(format!("{label} must not contain nulls")))
            } else {
                Ok(values.value(row))
            }
        })
        .collect()
}

fn time_values(array: &ArrayRef, label: &str, scale: TimeScale) -> PyResult<TimeArray> {
    let time = struct_array(array, label)?;
    TimeArray::from_parts(
        scale,
        required_ints(child(time, "days")?, &format!("{label}.days"))?,
        required_ints(child(time, "nanos")?, &format!("{label}.nanos"))?,
    )
    .map_err(|error| value_error(error.to_string()))
}

fn metadata_scale(batch: &RecordBatch, name: &str) -> PyResult<TimeScale> {
    let schema = batch.schema();
    let value = schema
        .metadata()
        .get(name)
        .ok_or_else(|| value_error(format!("missing trajectory metadata {name:?}")))?;
    TimeScale::parse(value).map_err(|error| value_error(error.to_string()))
}

fn decode_trajectory(batch: &RecordBatch) -> PyResult<TrajectoryData> {
    let required = |name: &str| {
        batch
            .column_by_name(name)
            .ok_or_else(|| value_error(format!("missing trajectory field {name:?}")))
    };
    let coverage_start = time_values(
        required("coverage_start")?,
        "coverage_start",
        metadata_scale(batch, "coverage_start.scale")?,
    )?;
    let coverage_end = time_values(
        required("coverage_end")?,
        "coverage_end",
        metadata_scale(batch, "coverage_end.scale")?,
    )?;
    let orbit = struct_array(required("orbit")?, "orbit")?;
    let coordinates = struct_array(child(orbit, "coordinates")?, "orbit.coordinates")?;
    let orbit_times = time_values(
        child(coordinates, "time")?,
        "orbit.coordinates.time",
        metadata_scale(batch, "orbit.coordinates.time.scale")?,
    )?;
    let output = TrajectoryData {
        object_ids: required_strings(required("object_id")?, "object_id")?,
        segment_ids: required_strings(required("segment_id")?, "segment_id")?,
        coverage_start,
        coverage_end,
        orbit_times,
    };
    output.validate_lengths().map_err(value_error)?;
    Ok(output)
}

fn decode_batch(value: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    RecordBatch::from_pyarrow_bound(value)
        .map_err(|error| value_error(format!("invalid Trajectory batch: {error}")))
}

#[pyfunction]
fn trajectory_mjd_arrow<'py>(
    py: Python<'py>,
    trajectory_batch: &Bound<'py, PyAny>,
    field: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = decode_trajectory(&decode_batch(trajectory_batch)?)?;
    let times = match field {
        "coverage_start" => &data.coverage_start,
        "coverage_end" => &data.coverage_end,
        "epoch" => &data.orbit_times,
        _ => {
            return Err(value_error(format!(
                "unknown trajectory MJD field: {field}"
            )))
        }
    };
    Ok(trajectory_mjd(times).map_err(value_error)?.into_pyarray(py))
}

#[pyfunction]
fn trajectory_object_ids_arrow(trajectory_batch: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let data = decode_trajectory(&decode_batch(trajectory_batch)?)?;
    Ok(trajectory_object_ids(&data))
}

#[pyfunction]
fn trajectory_validate_arrow(trajectory_batch: &Bound<'_, PyAny>) -> PyResult<()> {
    let data = decode_trajectory(&decode_batch(trajectory_batch)?)?;
    validate_trajectory(&data).map_err(value_error)
}

#[pyfunction]
#[pyo3(signature = (trajectory_batch, time_mjd_tdb, object_id=None))]
fn trajectory_segment_index_arrow(
    trajectory_batch: &Bound<'_, PyAny>,
    time_mjd_tdb: f64,
    object_id: Option<&str>,
) -> PyResult<Option<i64>> {
    let data = decode_trajectory(&decode_batch(trajectory_batch)?)?;
    trajectory_segment_index(&data, time_mjd_tdb, object_id).map_err(value_error)
}

#[pyfunction]
#[pyo3(signature = (trajectory_batch, operation, reps, trials, warmup_reps=1, time_mjd_tdb=None, object_id=None))]
#[allow(clippy::too_many_arguments)]
fn benchmark_trajectory_arrow(
    trajectory_batch: &Bound<'_, PyAny>,
    operation: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    time_mjd_tdb: Option<f64>,
    object_id: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(value_error("reps and trials must be >= 1"));
    }
    let data = decode_trajectory(&decode_batch(trajectory_batch)?)?;
    let run = || -> PyResult<()> {
        match operation {
            "coverage_start_mjd" => {
                black_box(trajectory_mjd(&data.coverage_start).map_err(value_error)?);
            }
            "coverage_end_mjd" => {
                black_box(trajectory_mjd(&data.coverage_end).map_err(value_error)?);
            }
            "epoch_mjd" => {
                black_box(trajectory_mjd(&data.orbit_times).map_err(value_error)?);
            }
            "object_ids" => {
                black_box(trajectory_object_ids(&data));
            }
            "validate_coverage" => {
                black_box(validate_trajectory(&data).map_err(value_error)?);
            }
            "segment_for" => {
                black_box(
                    trajectory_segment_index(
                        &data,
                        time_mjd_tdb.ok_or_else(|| {
                            value_error("segment_for timing requires time_mjd_tdb")
                        })?,
                        object_id,
                    )
                    .map_err(value_error)?,
                );
            }
            _ => {
                return Err(value_error(format!(
                    "unknown trajectory operation: {operation}"
                )))
            }
        }
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
    module.add_function(wrap_pyfunction!(trajectory_mjd_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(trajectory_object_ids_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(trajectory_validate_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(trajectory_segment_index_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_trajectory_arrow, module)?)?;
    Ok(())
}
