use crate::rotation_period::{config_from_python, estimate_to_dict};
use adam_core_rs_coords::rotation_period::{
    estimate_rotation_period, estimate_rotation_period_grouped, RotationPeriodInput,
};
use adam_core_rs_coords::{
    calculate_phase_angle_into, rotate_equatorial_to_ecliptic_row, Epoch, TimeArray, TimeScale,
};
use adam_core_rs_spice::global_backend;
use ndarray::ArrayView2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeSet, HashMap};

struct PreparedObservations {
    input: RotationPeriodInput,
    filter: Vec<Option<String>>,
    session_id: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_observations(
    detection_exposure_ids: &[Option<String>],
    magnitude: &[f64],
    magnitude_sigma: &[f64],
    exposure_ids: &[String],
    exposure_codes: &[String],
    exposure_filters: &[Option<String>],
    exposure_days: &[i64],
    exposure_nanos: &[i64],
    exposure_duration: &[f64],
    exposure_time_scale: &str,
    object_positions: &ArrayView2<'_, f64>,
    object_days: &[i64],
    object_nanos: &[i64],
    object_time_scale: &str,
    object_frame: &str,
) -> PyResult<PreparedObservations> {
    let rows = detection_exposure_ids.len();
    if magnitude.len() != rows
        || magnitude_sigma.len() != rows
        || object_positions.nrows() != rows
        || object_positions.ncols() != 3
        || object_days.len() != rows
        || object_nanos.len() != rows
    {
        return Err(PyValueError::new_err(
            "all detection and object-coordinate inputs must have length N",
        ));
    }
    validate_exposure_lengths(
        exposure_ids.len(),
        exposure_codes,
        exposure_filters,
        exposure_days,
        exposure_nanos,
        exposure_duration,
    )?;
    if detection_exposure_ids.iter().any(Option::is_none) {
        return Err(PyValueError::new_err(
            "detections.exposure_id must be non-null to align exposures",
        ));
    }
    let aligned = align_exposures(detection_exposure_ids, exposure_ids)?;
    let codes = take(exposure_codes, &aligned);
    let filters = take(exposure_filters, &aligned);
    let days = take(exposure_days, &aligned);
    let nanos = take(exposure_nanos, &aligned);
    let duration = take(exposure_duration, &aligned);
    let observers = crate::spice::observer_positions_from_exposures(
        &codes,
        &days,
        &nanos,
        &duration,
        exposure_time_scale,
    )?;
    let object = ecliptic_positions(object_positions, object_frame)?;
    let mut phase_angle = vec![0.0; rows];
    calculate_phase_angle_into(&object, &observers, &mut phase_angle);
    let (r_au, delta_au) = distances(&object, &observers);
    validate_photometry(magnitude, &r_au, &delta_au, &phase_angle)?;

    let object_time = time_array(object_days, object_nanos, object_time_scale)?
        .rescale(TimeScale::Tdb)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let utc = time_array(&days, &nanos, exposure_time_scale)?
        .rescale(TimeScale::Utc)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let sessions = session_ids(&codes, &utc)?;
    let input = RotationPeriodInput {
        time: object_time,
        magnitude: magnitude.to_vec(),
        magnitude_sigma: magnitude_sigma.to_vec(),
        filter: filters
            .iter()
            .map(|value| {
                value
                    .clone()
                    .unwrap_or_else(|| "__missing_filter__".to_string())
            })
            .collect(),
        session_id: sessions.iter().cloned().map(Some).collect(),
        r_au,
        delta_au,
        phase_angle_deg: phase_angle,
    };
    Ok(PreparedObservations {
        input,
        filter: filters,
        session_id: sessions,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepared_from_python(
    detection_exposure_ids: Vec<Option<String>>,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    exposure_ids: Vec<String>,
    exposure_codes: Vec<String>,
    exposure_filters: Vec<Option<String>>,
    exposure_days: Vec<i64>,
    exposure_nanos: Vec<i64>,
    exposure_duration: Vec<f64>,
    exposure_time_scale: &str,
    object_positions: PyReadonlyArray2<'_, f64>,
    object_days: Vec<i64>,
    object_nanos: Vec<i64>,
    object_time_scale: &str,
    object_frame: &str,
) -> PyResult<PreparedObservations> {
    let magnitude = magnitude
        .as_slice()
        .map_err(|_| PyValueError::new_err("magnitude must be contiguous"))?;
    let magnitude_sigma = magnitude_sigma
        .as_slice()
        .map_err(|_| PyValueError::new_err("magnitude_sigma must be contiguous"))?;
    prepare_observations(
        &detection_exposure_ids,
        magnitude,
        magnitude_sigma,
        &exposure_ids,
        &exposure_codes,
        &exposure_filters,
        &exposure_days,
        &exposure_nanos,
        &exposure_duration,
        exposure_time_scale,
        &object_positions.as_array(),
        &object_days,
        &object_nanos,
        object_time_scale,
        object_frame,
    )
}

fn prepared_to_dict(py: Python<'_>, prepared: &PreparedObservations) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item(
        "time_days",
        prepared
            .input
            .time
            .epochs
            .iter()
            .map(|epoch| epoch.days)
            .collect::<Vec<_>>(),
    )?;
    output.set_item(
        "time_nanos",
        prepared
            .input
            .time
            .epochs
            .iter()
            .map(|epoch| epoch.nanos)
            .collect::<Vec<_>>(),
    )?;
    output.set_item("magnitude", &prepared.input.magnitude)?;
    output.set_item(
        "magnitude_sigma",
        prepared
            .input
            .magnitude_sigma
            .iter()
            .map(|&value| value.is_finite().then_some(value))
            .collect::<Vec<_>>(),
    )?;
    output.set_item("filter", &prepared.filter)?;
    output.set_item("session_id", &prepared.session_id)?;
    output.set_item("r_au", &prepared.input.r_au)?;
    output.set_item("delta_au", &prepared.input.delta_au)?;
    output.set_item("phase_angle_deg", &prepared.input.phase_angle_deg)?;
    Ok(output.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (
    detection_exposure_ids, magnitude, magnitude_sigma, exposure_ids, exposure_codes,
    exposure_filters, exposure_days, exposure_nanos, exposure_duration, exposure_time_scale,
    object_positions, object_days, object_nanos, object_time_scale, object_frame
))]
#[allow(clippy::too_many_arguments)]
fn rotation_period_observations_from_detections(
    py: Python<'_>,
    detection_exposure_ids: Vec<Option<String>>,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    exposure_ids: Vec<String>,
    exposure_codes: Vec<String>,
    exposure_filters: Vec<Option<String>>,
    exposure_days: Vec<i64>,
    exposure_nanos: Vec<i64>,
    exposure_duration: Vec<f64>,
    exposure_time_scale: &str,
    object_positions: PyReadonlyArray2<'_, f64>,
    object_days: Vec<i64>,
    object_nanos: Vec<i64>,
    object_time_scale: &str,
    object_frame: &str,
) -> PyResult<PyObject> {
    let prepared = prepared_from_python(
        detection_exposure_ids,
        magnitude,
        magnitude_sigma,
        exposure_ids,
        exposure_codes,
        exposure_filters,
        exposure_days,
        exposure_nanos,
        exposure_duration,
        exposure_time_scale,
        object_positions,
        object_days,
        object_nanos,
        object_time_scale,
        object_frame,
    )?;
    prepared_to_dict(py, &prepared)
}

#[pyfunction]
#[pyo3(signature = (
    detection_exposure_ids, magnitude, magnitude_sigma, exposure_ids, exposure_codes,
    exposure_filters, exposure_days, exposure_nanos, exposure_duration, exposure_time_scale,
    object_positions, object_days, object_nanos, object_time_scale, object_frame, object_ids,
    search_fidelity, fourier_orders, clip_sigma, min_rotations_in_span,
    max_frequency_cycles_per_day, frequency_grid_scale, max_search_period_hours,
    early_exit_on_insufficient, session_mode, auto_session_min_observations_per_group,
    auto_session_bic_improvement
))]
#[allow(clippy::too_many_arguments)]
fn rotation_period_estimate_from_detections<'py>(
    py: Python<'py>,
    detection_exposure_ids: Vec<Option<String>>,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    exposure_ids: Vec<String>,
    exposure_codes: Vec<String>,
    exposure_filters: Vec<Option<String>>,
    exposure_days: Vec<i64>,
    exposure_nanos: Vec<i64>,
    exposure_duration: Vec<f64>,
    exposure_time_scale: &str,
    object_positions: PyReadonlyArray2<'_, f64>,
    object_days: Vec<i64>,
    object_nanos: Vec<i64>,
    object_time_scale: &str,
    object_frame: &str,
    object_ids: Option<Vec<Option<String>>>,
    search_fidelity: String,
    fourier_orders: Option<Vec<usize>>,
    clip_sigma: f64,
    min_rotations_in_span: f64,
    max_frequency_cycles_per_day: f64,
    frequency_grid_scale: f64,
    max_search_period_hours: Option<f64>,
    early_exit_on_insufficient: bool,
    session_mode: String,
    auto_session_min_observations_per_group: usize,
    auto_session_bic_improvement: f64,
) -> PyResult<PyObject> {
    let prepared = prepared_from_python(
        detection_exposure_ids,
        magnitude,
        magnitude_sigma,
        exposure_ids,
        exposure_codes,
        exposure_filters,
        exposure_days,
        exposure_nanos,
        exposure_duration,
        exposure_time_scale,
        object_positions,
        object_days,
        object_nanos,
        object_time_scale,
        object_frame,
    )?;
    let config = config_from_python(
        search_fidelity,
        fourier_orders,
        clip_sigma,
        min_rotations_in_span,
        max_frequency_cycles_per_day,
        frequency_grid_scale,
        max_search_period_hours,
        early_exit_on_insufficient,
        session_mode,
        auto_session_min_observations_per_group,
        auto_session_bic_improvement,
    );
    if let Some(object_ids) = object_ids {
        let estimates = py
            .allow_threads(|| {
                estimate_rotation_period_grouped(&prepared.input, &object_ids, &config)
            })
            .map_err(PyValueError::new_err)?;
        let output = PyList::empty(py);
        for (object_id, estimate) in estimates {
            output.append((object_id, estimate_to_dict(py, estimate)?))?;
        }
        Ok(output.into_any().unbind())
    } else {
        let estimate = py
            .allow_threads(|| estimate_rotation_period(&prepared.input, &config))
            .map_err(PyValueError::new_err)?;
        Ok(estimate_to_dict(py, estimate)?.into_any().unbind())
    }
}

fn validate_exposure_lengths(
    rows: usize,
    codes: &[String],
    filters: &[Option<String>],
    days: &[i64],
    nanos: &[i64],
    duration: &[f64],
) -> PyResult<()> {
    if [
        codes.len(),
        filters.len(),
        days.len(),
        nanos.len(),
        duration.len(),
    ]
    .iter()
    .any(|&length| length != rows)
    {
        return Err(PyValueError::new_err(
            "all exposure-table inputs must have equal lengths",
        ));
    }
    Ok(())
}

fn align_exposures(
    detection_ids: &[Option<String>],
    exposure_ids: &[String],
) -> PyResult<Vec<usize>> {
    let mut exposure_index = HashMap::new();
    for (index, id) in exposure_ids.iter().enumerate() {
        exposure_index.entry(id.as_str()).or_insert(index);
    }
    let mut aligned = Vec::with_capacity(detection_ids.len());
    let mut missing = BTreeSet::new();
    for id in detection_ids {
        let id = id.as_deref().expect("nulls checked by caller");
        if let Some(&index) = exposure_index.get(id) {
            aligned.push(index);
        } else {
            missing.insert(id.to_string());
        }
    }
    if !missing.is_empty() {
        return Err(PyValueError::new_err(format!(
            "detections reference unknown exposure_id(s): {:?}",
            missing.into_iter().collect::<Vec<_>>()
        )));
    }
    Ok(aligned)
}

fn take<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index].clone()).collect()
}

fn ecliptic_positions(positions: &ArrayView2<'_, f64>, frame: &str) -> PyResult<Vec<f64>> {
    let mut output = Vec::with_capacity(3 * positions.nrows());
    for row in positions.rows() {
        let value = [row[0], row[1], row[2], 0.0, 0.0, 0.0];
        let value = match frame {
            "ecliptic" => value,
            "equatorial" => rotate_equatorial_to_ecliptic_row(&value),
            _ => {
                return Err(PyValueError::new_err(
                    "native rotation observation assembly supports ecliptic and equatorial object coordinates",
                ))
            }
        };
        output.extend_from_slice(&value[..3]);
    }
    Ok(output)
}

fn distances(object: &[f64], observers: &[f64]) -> (Vec<f64>, Vec<f64>) {
    object
        .chunks_exact(3)
        .zip(observers.chunks_exact(3))
        .map(|(object, observer)| {
            let r = object.iter().map(|value| value * value).sum::<f64>().sqrt();
            let delta = object
                .iter()
                .zip(observer)
                .map(|(object, observer)| (object - observer).powi(2))
                .sum::<f64>()
                .sqrt();
            (r, delta)
        })
        .unzip()
}

fn validate_photometry(
    magnitude: &[f64],
    r_au: &[f64],
    delta_au: &[f64],
    phase_angle: &[f64],
) -> PyResult<()> {
    if magnitude.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "detections.mag must be finite for rotation-period analysis",
        ));
    }
    if r_au.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(PyValueError::new_err(
            "invalid heliocentric distance(s) for rotation-period analysis",
        ));
    }
    if delta_au
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(PyValueError::new_err(
            "invalid observer distance(s) for rotation-period analysis",
        ));
    }
    if phase_angle.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "invalid phase angle(s) for rotation-period analysis",
        ));
    }
    Ok(())
}

fn time_array(days: &[i64], nanos: &[i64], scale: &str) -> PyResult<TimeArray> {
    let scale =
        TimeScale::parse(scale).map_err(|error| PyValueError::new_err(error.to_string()))?;
    TimeArray::new(
        scale,
        days.iter()
            .copied()
            .zip(nanos.iter().copied())
            .map(|(days, nanos)| Epoch::new(days, nanos))
            .collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn session_ids(codes: &[String], utc: &TimeArray) -> PyResult<Vec<String>> {
    let (longitude, cos_phi, sin_phi) = {
        let backend = global_backend()
            .lock()
            .map_err(|_| PyRuntimeError::new_err("adam-core SPICE backend lock is poisoned"))?;
        let mut longitude = Vec::with_capacity(codes.len());
        let mut cos_phi = Vec::with_capacity(codes.len());
        let mut sin_phi = Vec::with_capacity(codes.len());
        for code in codes {
            let site = backend.ground_observer_site(code).ok_or_else(|| {
                PyValueError::new_err(format!("{code} is not a valid MPC observatory code"))
            })?;
            longitude.push(site.longitude_deg);
            cos_phi.push(site.cos_phi);
            sin_phi.push(site.sin_phi);
        }
        (longitude, cos_phi, sin_phi)
    };
    let timezone_names = crate::coordinate_ops::observatory_timezone_values(
        &longitude,
        &cos_phi,
        &sin_phi,
        6.694_379_990_14e-3,
    )?;
    let timezones = crate::timestamp_ops::parse_timezones(&timezone_names)?;
    let days: Vec<i64> = utc.epochs.iter().map(|epoch| epoch.days).collect();
    let nanos: Vec<i64> = utc.epochs.iter().map(|epoch| epoch.nanos).collect();
    let nights = crate::timestamp_ops::observing_nights(&days, &nanos, &timezones)?;
    Ok(codes
        .iter()
        .zip(nights)
        .map(|(code, night)| format!("{code}:{night}"))
        .collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        rotation_period_observations_from_detections,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rotation_period_estimate_from_detections,
        m
    )?)?;
    Ok(())
}
