use crate::coordinate_ops::bench_result;
use crate::spice::observer_positions_from_exposures;
use adam_core_rs_coords::{fit_absolute_magnitude_grouped, fit_absolute_magnitude_rows};
use arrow::pyarrow::ToPyArrow;
use arrow_array::{ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[pyfunction]
fn fit_absolute_magnitude_rows_numpy<'py>(
    _py: Python<'py>,
    h_rows: PyReadonlyArray1<'py, f64>,
    sigma_rows: PyReadonlyArray1<'py, f64>,
) -> PyResult<(f64, f64, f64, f64, i64)> {
    let h = h_rows.as_array();
    let s = sigma_rows.as_array();
    if h.len() != s.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let h_slice = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let s_slice = s
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    let f = fit_absolute_magnitude_rows(h_slice, s_slice);
    Ok((f.h_hat, f.h_sigma, f.sigma_eff, f.chi2_red, f.n_used))
}

#[pyfunction]
fn fit_absolute_magnitude_grouped_numpy<'py>(
    py: Python<'py>,
    h_rows: PyReadonlyArray1<'py, f64>,
    sigma_rows: PyReadonlyArray1<'py, f64>,
    group_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let h = h_rows.as_array();
    let s = sigma_rows.as_array();
    let off = group_offsets.as_array();
    if h.len() != s.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let off_usize: Vec<usize> = off.iter().map(|&v| v as usize).collect();
    let h_slice = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let s_slice = s
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    let fits = fit_absolute_magnitude_grouped(h_slice, s_slice, &off_usize);
    let h_hat: Vec<f64> = fits.iter().map(|f| f.h_hat).collect();
    let h_sigma: Vec<f64> = fits.iter().map(|f| f.h_sigma).collect();
    let sigma_eff: Vec<f64> = fits.iter().map(|f| f.sigma_eff).collect();
    let chi2_red: Vec<f64> = fits.iter().map(|f| f.chi2_red).collect();
    let n_used: Vec<i64> = fits.iter().map(|f| f.n_used).collect();
    Ok((
        ndarray::Array1::from_vec(h_hat).into_pyarray(py),
        ndarray::Array1::from_vec(h_sigma).into_pyarray(py),
        ndarray::Array1::from_vec(sigma_eff).into_pyarray(py),
        ndarray::Array1::from_vec(chi2_red).into_pyarray(py),
        ndarray::Array1::from_vec(n_used).into_pyarray(py),
    ))
}

#[pyfunction]
fn calculate_phase_angle_numpy<'py>(
    py: Python<'py>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated phase output must be contiguous"))?;
        adam_core_rs_coords::calculate_phase_angle_into(obj_slice, obs_slice, out_slice);
    }
    Ok(out)
}

#[pyfunction]
fn calculate_apparent_magnitude_v_numpy<'py>(
    py: Python<'py>,
    h_v: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    let h_arr = h_v.as_array();
    let g_arr = g.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    if h_arr.len() != n || g_arr.len() != n {
        return Err(PyValueError::new_err(
            "h_v and g must each have length N for positions shape (N, 3)",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h_slice = h_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g_slice = g_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated magnitude output must be contiguous"))?;
        adam_core_rs_coords::calculate_apparent_magnitude_v_into(
            h_slice, obj_slice, obs_slice, g_slice, out_slice,
        );
    }
    Ok(out)
}

#[pyfunction]
fn calculate_apparent_magnitude_v_and_phase_angle_numpy<'py>(
    py: Python<'py>,
    h_v: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    let h_arr = h_v.as_array();
    let g_arr = g.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    if h_arr.len() != n || g_arr.len() != n {
        return Err(PyValueError::new_err(
            "h_v and g must each have length N for positions shape (N, 3)",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h_slice = h_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g_slice = g_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let mag_out = PyArray1::<f64>::zeros(py, n, false);
    let alpha_out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut mag_rw = numpy::PyArrayMethods::readwrite(&mag_out);
        let mut alpha_rw = numpy::PyArrayMethods::readwrite(&alpha_out);
        let mag_slice = mag_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated magnitude output must be contiguous"))?;
        let alpha_slice = alpha_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated phase output must be contiguous"))?;
        adam_core_rs_coords::calculate_apparent_magnitude_v_and_phase_angle_into(
            h_slice,
            obj_slice,
            obs_slice,
            g_slice,
            mag_slice,
            alpha_slice,
        );
    }
    Ok((mag_out, alpha_out))
}

fn bandpass_error(err: adam_core_rs_coords::SchemaError) -> PyErr {
    PyValueError::new_err(match err {
        adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => message,
        other => other.to_string(),
    })
}

fn validate_photometry_geometry(object: &[f64], observer: &[f64], rows: usize) -> PyResult<()> {
    let mut invalid = 0usize;
    for row in 0..rows {
        let base = row * 3;
        let r = (object[base] * object[base]
            + object[base + 1] * object[base + 1]
            + object[base + 2] * object[base + 2])
            .sqrt();
        let dx = object[base] - observer[base];
        let dy = object[base + 1] - observer[base + 1];
        let dz = object[base + 2] - observer[base + 2];
        let delta = (dx * dx + dy * dy + dz * dz).sqrt();
        if !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0 {
            invalid += 1;
        }
    }
    if invalid > 0 {
        return Err(PyValueError::new_err(format!(
            "Invalid photometry geometry for H-G model: {invalid} rows have non-finite or non-positive distances (r<=0 or delta<=0)."
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn predict_magnitudes_core(
    data_dir: &str,
    h: &[f64],
    object: &[f64],
    observer: &[f64],
    g: &[f64],
    target_filter_ids: &[String],
    reference_filter: &str,
    template_id: Option<&str>,
    mix: Option<(f64, f64)>,
) -> PyResult<Vec<f64>> {
    let n = h.len();
    let data = adam_core_rs_coords::bandpass_data(Path::new(data_dir)).map_err(bandpass_error)?;
    data.assert_filter_ids_have_curves(target_filter_ids)
        .map_err(bandpass_error)?;
    data.assert_filter_ids_have_curves(&[reference_filter.to_string()])
        .map_err(bandpass_error)?;
    let deltas = data.delta_table(template_id, mix).map_err(bandpass_error)?;
    if deltas.len() != data.filter_ids.len() {
        return Err(PyValueError::new_err(
            "Bandpass delta table length mismatch.",
        ));
    }
    let reference_id = data
        .filter_ids
        .iter()
        .position(|value| value == reference_filter)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown reference_filter for bandpass conversion: {reference_filter}"
            ))
        })?;
    let target_ids: Vec<i32> = target_filter_ids
        .iter()
        .map(|filter| {
            data.filter_ids
                .iter()
                .position(|value| value == filter)
                .map(|index| index as i32)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Unknown canonical filter_ids for bandpass prediction: ['{filter}']"
                    ))
                })
        })
        .collect::<PyResult<_>>()?;

    let mut invalid = 0usize;
    for row in 0..n {
        let base = row * 3;
        let r = (object[base] * object[base]
            + object[base + 1] * object[base + 1]
            + object[base + 2] * object[base + 2])
            .sqrt();
        let dx = object[base] - observer[base];
        let dy = object[base + 1] - observer[base + 1];
        let dz = object[base + 2] - observer[base + 2];
        let delta = (dx * dx + dy * dy + dz * dz).sqrt();
        if !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0 {
            invalid += 1;
        }
    }
    if invalid > 0 {
        return Err(PyValueError::new_err(format!(
            "Invalid photometry geometry for H-G model: {invalid} rows have non-finite or non-positive distances (r<=0 or delta<=0)."
        )));
    }

    let reference_delta = deltas[reference_id];
    let h_v: Vec<f64> = h.iter().map(|value| value - reference_delta).collect();
    let mut output = vec![0.0; n];
    adam_core_rs_coords::predict_magnitudes_bandpass_into(
        &h_v,
        object,
        observer,
        g,
        &target_ids,
        &deltas,
        &mut output,
    );
    Ok(output)
}

/// Complete post-observer prediction facade: Rust owns filter validation and
/// indexing, composition/reference conversion, geometry validation, and the
/// H-G + bandpass kernel. Python performs only scalar broadcasting and the
/// separately-governed observer-state provider call.
#[pyfunction]
#[pyo3(signature = (data_dir, h, object_pos, observer_pos, g, target_filter_ids, reference_filter, template_id=None, mix=None))]
#[allow(clippy::too_many_arguments)]
fn predict_magnitudes_fused_numpy<'py>(
    py: Python<'py>,
    data_dir: &str,
    h: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
    target_filter_ids: Vec<String>,
    reference_filter: &str,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let object = object_pos.as_array();
    let observer = observer_pos.as_array();
    let h = h.as_array();
    let g = g.as_array();
    if object.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = object.nrows();
    if observer.nrows() != n || observer.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) and match object_pos",
        ));
    }
    if h.len() != n || g.len() != n || target_filter_ids.len() != n {
        return Err(PyValueError::new_err(
            "h, g, and target_filter_ids must each have length N",
        ));
    }
    let object = object
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let observer = observer
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;

    let output = predict_magnitudes_core(
        data_dir,
        h,
        object,
        observer,
        g,
        &target_filter_ids,
        reference_filter,
        template_id.as_deref(),
        mix,
    )?;
    Ok(ndarray::Array1::from_vec(output).into_pyarray(py))
}

/// End-to-end prediction crossing for the ordinary Exposures provider path.
/// Exposure midpoint construction and ground/space observer state generation
/// stay inside Rust with filter resolution and photometry.
#[pyfunction]
#[pyo3(signature = (data_dir, h, object_pos, g, target_filter_ids, observer_codes, days, nanos, duration, time_scale, reference_filter, template_id=None, mix=None))]
#[allow(clippy::too_many_arguments)]
fn predict_magnitudes_complete_numpy<'py>(
    py: Python<'py>,
    data_dir: &str,
    h: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
    target_filter_ids: Vec<String>,
    observer_codes: Vec<String>,
    days: Vec<i64>,
    nanos: Vec<i64>,
    duration: Vec<f64>,
    time_scale: &str,
    reference_filter: &str,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let object = object_pos.as_array();
    let h = h.as_array();
    let g = g.as_array();
    if object.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = object.nrows();
    if h.len() != n
        || g.len() != n
        || target_filter_ids.len() != n
        || observer_codes.len() != n
        || days.len() != n
        || nanos.len() != n
        || duration.len() != n
    {
        return Err(PyValueError::new_err(
            "complete prediction inputs must each have length N",
        ));
    }
    let object = object
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let data = adam_core_rs_coords::bandpass_data(Path::new(data_dir)).map_err(bandpass_error)?;
    data.assert_filter_ids_have_curves(&target_filter_ids)
        .map_err(bandpass_error)?;
    data.assert_filter_ids_have_curves(&[reference_filter.to_string()])
        .map_err(bandpass_error)?;
    let observer =
        observer_positions_from_exposures(&observer_codes, &days, &nanos, &duration, time_scale)?;
    let output = predict_magnitudes_core(
        data_dir,
        h,
        object,
        &observer,
        g,
        &target_filter_ids,
        reference_filter,
        template_id.as_deref(),
        mix,
    )?;
    Ok(ndarray::Array1::from_vec(output).into_pyarray(py))
}

type FittedMagnitudeGroups = (
    Vec<String>,
    Vec<f64>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<i64>,
);

#[allow(clippy::too_many_arguments)]
fn fit_absolute_magnitude_core(
    data_dir: &str,
    magnitude: &[f64],
    sigma: &[f64],
    object: &[f64],
    observer: &[f64],
    observatory_codes: &[String],
    bands: &[String],
    g: f64,
    strict_band_mapping: bool,
    template_id: Option<&str>,
    mix: Option<(f64, f64)>,
    object_ids: Option<&[Option<String>]>,
) -> PyResult<FittedMagnitudeGroups> {
    let n = magnitude.len();
    validate_photometry_geometry(object, observer, n)?;
    let data = adam_core_rs_coords::bandpass_data(Path::new(data_dir)).map_err(bandpass_error)?;
    let canonical = data
        .map_to_canonical_filter_bands(observatory_codes, bands, !strict_band_mapping)
        .map_err(bandpass_error)?;
    let deltas = data.delta_table(template_id, mix).map_err(bandpass_error)?;
    let targets: Vec<i32> = canonical
        .iter()
        .map(|filter| {
            data.filter_ids
                .iter()
                .position(|value| value == filter)
                .map(|index| index as i32)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Unknown canonical filter_ids for bandpass prediction: ['{filter}']"
                    ))
                })
        })
        .collect::<PyResult<_>>()?;
    let h0 = vec![0.0; n];
    let g_rows = vec![g; n];
    let mut modeled = vec![0.0; n];
    adam_core_rs_coords::predict_magnitudes_bandpass_into(
        &h0,
        object,
        observer,
        &g_rows,
        &targets,
        &deltas,
        &mut modeled,
    );

    let fit_group = |rows: &[usize]| {
        let h_rows: Vec<f64> = rows
            .iter()
            .map(|&row| magnitude[row] - modeled[row])
            .collect();
        let sigma_rows: Vec<f64> = rows.iter().map(|&row| sigma[row]).collect();
        fit_absolute_magnitude_rows(&h_rows, &sigma_rows)
    };
    let optional = |value: f64| value.is_finite().then_some(value);

    if let Some(ids) = object_ids {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for row in 0..n {
            if magnitude[row].is_finite() && modeled[row].is_finite() {
                if let Some(id) = &ids[row] {
                    groups.entry(id.clone()).or_default().push(row);
                }
            }
        }
        let mut output: FittedMagnitudeGroups = Default::default();
        for (id, rows) in groups {
            let fit = fit_group(&rows);
            if !fit.h_hat.is_finite() {
                continue;
            }
            output.0.push(id);
            output.1.push(fit.h_hat);
            output.2.push(optional(fit.h_sigma));
            output.3.push(optional(fit.sigma_eff));
            output.4.push(optional(fit.chi2_red));
            output.5.push(fit.n_used);
        }
        Ok(output)
    } else {
        let rows: Vec<usize> = (0..n)
            .filter(|&row| magnitude[row].is_finite() && modeled[row].is_finite())
            .collect();
        if rows.is_empty() {
            return Err(PyValueError::new_err(
                "no valid rows: need finite detections.mag and forward-model magnitudes",
            ));
        }
        let fit = fit_group(&rows);
        if !fit.h_hat.is_finite() {
            return Err(PyValueError::new_err(
                "invalid weights derived from mag_sigma",
            ));
        }
        Ok((
            Vec::new(),
            vec![fit.h_hat],
            vec![optional(fit.h_sigma)],
            vec![optional(fit.sigma_eff)],
            vec![optional(fit.chi2_red)],
            vec![fit.n_used],
        ))
    }
}

/// Fused post-observer absolute-magnitude fit. Rust owns canonical-band
/// resolution, prediction, validity filtering, stable lexical grouping, all
/// fits, and null-sentinel conversion.
#[pyfunction]
#[pyo3(signature = (data_dir, magnitude, magnitude_sigma, object_pos, observer_pos, observatory_codes, bands, g, strict_band_mapping, template_id=None, mix=None, object_ids=None))]
#[allow(clippy::too_many_arguments)]
fn fit_absolute_magnitude_facade_numpy(
    data_dir: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    observatory_codes: Vec<String>,
    bands: Vec<String>,
    g: f64,
    strict_band_mapping: bool,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
    object_ids: Option<Vec<Option<String>>>,
) -> PyResult<FittedMagnitudeGroups> {
    let object = object_pos.as_array();
    let observer = observer_pos.as_array();
    let magnitude = magnitude.as_array();
    let sigma = magnitude_sigma.as_array();
    let n = object.nrows();
    if object.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    if observer.nrows() != n || observer.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) and match object_pos",
        ));
    }
    if magnitude.len() != n
        || sigma.len() != n
        || observatory_codes.len() != n
        || bands.len() != n
        || object_ids.as_ref().is_some_and(|ids| ids.len() != n)
    {
        return Err(PyValueError::new_err(
            "all absolute-magnitude fit inputs must have length N",
        ));
    }
    let object = object
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let observer = observer
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let magnitude = magnitude
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("magnitude must be contiguous"))?;
    let sigma = sigma
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("magnitude_sigma must be contiguous"))?;

    let data = adam_core_rs_coords::bandpass_data(Path::new(data_dir)).map_err(bandpass_error)?;
    let canonical = data
        .map_to_canonical_filter_bands(&observatory_codes, &bands, !strict_band_mapping)
        .map_err(bandpass_error)?;
    let deltas = data
        .delta_table(template_id.as_deref(), mix)
        .map_err(bandpass_error)?;
    let targets: Vec<i32> = canonical
        .iter()
        .map(|filter| {
            data.filter_ids
                .iter()
                .position(|value| value == filter)
                .map(|index| index as i32)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Unknown canonical filter_ids for bandpass prediction: ['{filter}']"
                    ))
                })
        })
        .collect::<PyResult<_>>()?;
    let h0 = vec![0.0; n];
    let g_rows = vec![g; n];
    let mut modeled = vec![0.0; n];
    adam_core_rs_coords::predict_magnitudes_bandpass_into(
        &h0,
        object,
        observer,
        &g_rows,
        &targets,
        &deltas,
        &mut modeled,
    );

    let fit_group = |rows: &[usize]| {
        let h_rows: Vec<f64> = rows
            .iter()
            .map(|&row| magnitude[row] - modeled[row])
            .collect();
        let sigma_rows: Vec<f64> = rows.iter().map(|&row| sigma[row]).collect();
        fit_absolute_magnitude_rows(&h_rows, &sigma_rows)
    };
    let optional = |value: f64| value.is_finite().then_some(value);

    if let Some(ids) = object_ids {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for row in 0..n {
            if magnitude[row].is_finite() && modeled[row].is_finite() {
                if let Some(id) = &ids[row] {
                    groups.entry(id.clone()).or_default().push(row);
                }
            }
        }
        let mut output: FittedMagnitudeGroups = Default::default();
        for (id, rows) in groups {
            let fit = fit_group(&rows);
            if !fit.h_hat.is_finite() {
                continue;
            }
            output.0.push(id);
            output.1.push(fit.h_hat);
            output.2.push(optional(fit.h_sigma));
            output.3.push(optional(fit.sigma_eff));
            output.4.push(optional(fit.chi2_red));
            output.5.push(fit.n_used);
        }
        Ok(output)
    } else {
        let rows: Vec<usize> = (0..n)
            .filter(|&row| magnitude[row].is_finite() && modeled[row].is_finite())
            .collect();
        if rows.is_empty() {
            return Err(PyValueError::new_err(
                "no valid rows: need finite detections.mag and forward-model magnitudes",
            ));
        }
        let fit = fit_group(&rows);
        if !fit.h_hat.is_finite() {
            return Err(PyValueError::new_err(
                "invalid weights derived from mag_sigma",
            ));
        }
        Ok((
            Vec::new(),
            vec![fit.h_hat],
            vec![optional(fit.h_sigma)],
            vec![optional(fit.sigma_eff)],
            vec![optional(fit.chi2_red)],
            vec![fit.n_used],
        ))
    }
}

fn fit_output_batch(output: FittedMagnitudeGroups, g: f64, grouped: bool) -> PyResult<RecordBatch> {
    let (ids, h, h_sigma, sigma_eff, chi2_red, n_used) = output;
    let rows = h.len();
    let physical_fields = Fields::from(vec![
        Field::new("H_v", DataType::Float64, true),
        Field::new("H_v_sigma", DataType::Float64, true),
        Field::new("G", DataType::Float64, true),
        Field::new("G_sigma", DataType::Float64, true),
        Field::new("sigma_eff", DataType::Float64, true),
        Field::new("chi2_red", DataType::Float64, true),
    ]);
    let physical_arrays: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(h)),
        Arc::new(Float64Array::from(h_sigma)),
        Arc::new(Float64Array::from(vec![Some(g); rows])),
        Arc::new(Float64Array::from(vec![None::<f64>; rows])),
        Arc::new(Float64Array::from(sigma_eff)),
        Arc::new(Float64Array::from(chi2_red)),
    ];
    if grouped {
        let physical = StructArray::new(physical_fields.clone(), physical_arrays, None);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("object_id", DataType::LargeUtf8, false),
                Field::new(
                    "physical_parameters",
                    DataType::Struct(physical_fields),
                    false,
                ),
                Field::new("n_fit_detections", DataType::Int64, false),
            ])),
            vec![
                Arc::new(LargeStringArray::from(ids)),
                Arc::new(physical),
                Arc::new(Int64Array::from(n_used)),
            ],
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))
    } else {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("H_v", DataType::Float64, true),
                Field::new("H_v_sigma", DataType::Float64, true),
                Field::new("G", DataType::Float64, true),
                Field::new("G_sigma", DataType::Float64, true),
                Field::new("sigma_eff", DataType::Float64, true),
                Field::new("chi2_red", DataType::Float64, true),
            ])),
            physical_arrays,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

/// End-to-end absolute-magnitude crossing for the ordinary Exposures
/// provider path. Rust owns exposure-id alignment, midpoint observer states,
/// band mapping, prediction, grouping, and fitting.
#[pyfunction]
#[pyo3(signature = (data_dir, magnitude, magnitude_sigma, object_pos, detection_exposure_ids, exposure_ids, exposure_codes, exposure_bands, exposure_days, exposure_nanos, exposure_duration, time_scale, g, strict_band_mapping, template_id=None, mix=None, object_ids=None))]
#[allow(clippy::too_many_arguments)]
fn fit_absolute_magnitude_complete_numpy(
    py: Python<'_>,
    data_dir: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    detection_exposure_ids: Vec<Option<String>>,
    exposure_ids: Vec<String>,
    exposure_codes: Vec<String>,
    exposure_bands: Vec<String>,
    exposure_days: Vec<i64>,
    exposure_nanos: Vec<i64>,
    exposure_duration: Vec<f64>,
    time_scale: &str,
    g: f64,
    strict_band_mapping: bool,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
    object_ids: Option<Vec<Option<String>>>,
) -> PyResult<PyObject> {
    let grouped = object_ids.is_some();
    let object = object_pos.as_array();
    let magnitude = magnitude.as_array();
    let sigma = magnitude_sigma.as_array();
    let n = object.nrows();
    let exposure_rows = exposure_ids.len();
    if object.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    if magnitude.len() != n
        || sigma.len() != n
        || detection_exposure_ids.len() != n
        || object_ids.as_ref().is_some_and(|ids| ids.len() != n)
    {
        return Err(PyValueError::new_err(
            "all absolute-magnitude fit inputs must have length N",
        ));
    }
    if exposure_codes.len() != exposure_rows
        || exposure_bands.len() != exposure_rows
        || exposure_days.len() != exposure_rows
        || exposure_nanos.len() != exposure_rows
        || exposure_duration.len() != exposure_rows
    {
        return Err(PyValueError::new_err(
            "all exposure-table inputs must have equal lengths",
        ));
    }
    if detection_exposure_ids.iter().any(Option::is_none) {
        return Err(PyValueError::new_err(
            "detections.exposure_id must be non-null to link to exposures",
        ));
    }
    let mut exposure_index: HashMap<&str, usize> = HashMap::new();
    for (row, id) in exposure_ids.iter().enumerate() {
        exposure_index.entry(id).or_insert(row);
    }
    let mut aligned = Vec::with_capacity(n);
    let mut missing = BTreeSet::new();
    for id in &detection_exposure_ids {
        let id = id.as_deref().expect("nulls checked above");
        if let Some(&row) = exposure_index.get(id) {
            aligned.push(row);
        } else {
            missing.insert(id.to_string());
        }
    }
    if !missing.is_empty() {
        let values: Vec<String> = missing.into_iter().collect();
        return Err(PyValueError::new_err(format!(
            "detections reference unknown exposure_id(s): {values:?}"
        )));
    }
    let codes: Vec<String> = aligned
        .iter()
        .map(|&row| exposure_codes[row].clone())
        .collect();
    let bands: Vec<String> = aligned
        .iter()
        .map(|&row| exposure_bands[row].clone())
        .collect();
    let days: Vec<i64> = aligned.iter().map(|&row| exposure_days[row]).collect();
    let nanos: Vec<i64> = aligned.iter().map(|&row| exposure_nanos[row]).collect();
    let duration: Vec<f64> = aligned.iter().map(|&row| exposure_duration[row]).collect();
    adam_core_rs_coords::bandpass_data(Path::new(data_dir))
        .map_err(bandpass_error)?
        .map_to_canonical_filter_bands(&codes, &bands, !strict_band_mapping)
        .map_err(bandpass_error)?;
    let observers =
        observer_positions_from_exposures(&codes, &days, &nanos, &duration, time_scale)?;
    let object = object
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let magnitude = magnitude
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("magnitude must be contiguous"))?;
    let sigma = sigma
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("magnitude_sigma must be contiguous"))?;
    let output = fit_absolute_magnitude_core(
        data_dir,
        magnitude,
        sigma,
        object,
        &observers,
        &codes,
        &bands,
        g,
        strict_band_mapping,
        template_id.as_deref(),
        mix,
        object_ids.as_deref(),
    )?;
    fit_output_batch(output, g, grouped)?
        .to_pyarrow(py)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Rust-owned timing for the three complete public photometry facades.
/// Samples include observer-state generation, mapping, geometry, prediction,
/// fitting/grouping, and Arrow table assembly; PyO3 conversion is excluded.
#[pyfunction]
#[pyo3(signature = (data_dir, reps, trials, warmup_reps=1))]
fn benchmark_complete_photometry_facades(
    data_dir: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let rows = 64usize;
    let codes = vec!["500".to_string(); rows];
    let days = vec![60_000i64; rows];
    let nanos = vec![0i64; rows];
    let duration = vec![0.0; rows];
    let filters = vec!["V".to_string(); rows];
    let object: Vec<f64> = (0..rows).flat_map(|_| [2.0, 0.0, 0.0]).collect();
    let h = vec![18.0; rows];
    let g_rows = vec![0.15; rows];
    let initial_observer =
        observer_positions_from_exposures(&codes, &days, &nanos, &duration, "tdb")?;
    let modeled = predict_magnitudes_core(
        data_dir,
        &h,
        &object,
        &initial_observer,
        &g_rows,
        &filters,
        "V",
        Some("C"),
        None,
    )?;
    let sigma = vec![0.1; rows];
    let grouped_ids: Vec<Option<String>> = (0..rows)
        .map(|row| Some(format!("object-{:02}", row % 8)))
        .collect();

    let prediction = bench_result(reps, trials, warmup_reps, || {
        let observer = observer_positions_from_exposures(&codes, &days, &nanos, &duration, "tdb")?;
        predict_magnitudes_core(
            data_dir,
            &h,
            &object,
            &observer,
            &g_rows,
            &filters,
            "V",
            Some("C"),
            None,
        )
    })?;
    let single = bench_result(reps, trials, warmup_reps, || {
        let observer = observer_positions_from_exposures(&codes, &days, &nanos, &duration, "tdb")?;
        let output = fit_absolute_magnitude_core(
            data_dir,
            &modeled,
            &sigma,
            &object,
            &observer,
            &codes,
            &filters,
            0.15,
            false,
            Some("C"),
            None,
            None,
        )?;
        fit_output_batch(output, 0.15, false)
    })?;
    let grouped = bench_result(reps, trials, warmup_reps, || {
        let observer = observer_positions_from_exposures(&codes, &days, &nanos, &duration, "tdb")?;
        let output = fit_absolute_magnitude_core(
            data_dir,
            &modeled,
            &sigma,
            &object,
            &observer,
            &codes,
            &filters,
            0.15,
            false,
            Some("C"),
            None,
            Some(&grouped_ids),
        )?;
        fit_output_batch(output, 0.15, true)
    })?;
    Ok(vec![prediction, single, grouped])
}

#[pyfunction]
fn predict_magnitudes_bandpass_numpy<'py>(
    py: Python<'py>,
    h_v: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
    target_ids: PyReadonlyArray1<'py, i32>,
    delta_table: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    let h_arr = h_v.as_array();
    let g_arr = g.as_array();
    let tid_arr = target_ids.as_array();
    let dt_arr = delta_table.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    if h_arr.len() != n || g_arr.len() != n || tid_arr.len() != n {
        return Err(PyValueError::new_err(
            "h_v, g, target_ids must each have length N for positions shape (N, 3)",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h_slice = h_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g_slice = g_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let tid_slice = tid_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("target_ids must be contiguous"))?;
    let dt_slice = dt_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("delta_table must be contiguous"))?;
    // SAFETY: the kernel below writes every element before the array is
    // exposed to Python. Avoiding NumPy's redundant zero-fill matters for
    // large public prediction batches.
    let out = unsafe { PyArray1::<f64>::new(py, [n], false) };
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated predict output must be contiguous"))?;
        adam_core_rs_coords::predict_magnitudes_bandpass_into(
            h_slice, obj_slice, obs_slice, g_slice, tid_slice, dt_slice, out_slice,
        );
    }
    Ok(out)
}

fn benchmark_samples<T, F>(
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    mut run_once: F,
) -> PyResult<Vec<Vec<f64>>>
where
    F: FnMut() -> T,
{
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once());
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            let output = run_once();
            black_box(&output);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

fn owned_positions(
    object_pos: &PyReadonlyArray2<'_, f64>,
    observer_pos: &PyReadonlyArray2<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, usize)> {
    let object = object_pos.as_array();
    let observer = observer_pos.as_array();
    if object.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = object.nrows();
    if observer.nrows() != n || observer.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    let object = object
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?
        .to_vec();
    let observer = observer
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?
        .to_vec();
    Ok((object, observer, n))
}

/// Rust-owned Instant timer for the phase-angle parity surface. NumPy/PyO3
/// extraction happens once above the timing loop; every sample allocates its
/// semantic output and calls the Rust kernel directly.
#[pyfunction]
#[pyo3(signature = (object_pos, observer_pos, reps, trials, warmup_reps=1))]
fn benchmark_calculate_phase_angle_numpy(
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut output = vec![0.0; n];
        adam_core_rs_coords::calculate_phase_angle_into(&object, &observer, &mut output);
        output
    })
}

#[pyfunction]
#[pyo3(signature = (h_v, object_pos, observer_pos, g, reps, trials, warmup_reps=1))]
fn benchmark_calculate_apparent_magnitude_v_numpy(
    h_v: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    let h = h_v.as_array();
    let g = g.as_array();
    if h.len() != n || g.len() != n {
        return Err(PyValueError::new_err("h_v and g must each have length N"));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut output = vec![0.0; n];
        adam_core_rs_coords::calculate_apparent_magnitude_v_into(
            h,
            &object,
            &observer,
            g,
            &mut output,
        );
        output
    })
}

#[pyfunction]
#[pyo3(signature = (h_v, object_pos, observer_pos, g, reps, trials, warmup_reps=1))]
fn benchmark_calculate_apparent_magnitude_v_and_phase_angle_numpy(
    h_v: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    let h = h_v.as_array();
    let g = g.as_array();
    if h.len() != n || g.len() != n {
        return Err(PyValueError::new_err("h_v and g must each have length N"));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut magnitude = vec![0.0; n];
        let mut phase_angle = vec![0.0; n];
        adam_core_rs_coords::calculate_apparent_magnitude_v_and_phase_angle_into(
            h,
            &object,
            &observer,
            g,
            &mut magnitude,
            &mut phase_angle,
        );
        (magnitude, phase_angle)
    })
}

#[pyfunction]
#[pyo3(signature = (h_v, object_pos, observer_pos, g, target_ids, delta_table, reps, trials, warmup_reps=1))]
fn benchmark_predict_magnitudes_bandpass_numpy(
    h_v: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    target_ids: PyReadonlyArray1<'_, i32>,
    delta_table: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    let h = h_v.as_array();
    let g = g.as_array();
    let targets = target_ids.as_array();
    let deltas = delta_table.as_array();
    if h.len() != n || g.len() != n || targets.len() != n {
        return Err(PyValueError::new_err(
            "h_v, g, and target_ids must each have length N",
        ));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let targets = targets
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("target_ids must be contiguous"))?;
    let deltas = deltas
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("delta_table must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut output = vec![0.0; n];
        adam_core_rs_coords::predict_magnitudes_bandpass_into(
            h,
            &object,
            &observer,
            g,
            targets,
            deltas,
            &mut output,
        );
        output
    })
}

/// Rust-owned timer for one-group fitting. The complete fit, including its
/// per-fit residual/MAD setup and allocations, is inside each semantic sample.
#[pyfunction]
#[pyo3(signature = (h_rows, sigma_rows, reps, trials, warmup_reps=1))]
fn benchmark_fit_absolute_magnitude_rows_numpy(
    h_rows: PyReadonlyArray1<'_, f64>,
    sigma_rows: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let h = h_rows.as_array();
    let sigma = sigma_rows.as_array();
    if h.len() != sigma.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let sigma = sigma
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        fit_absolute_magnitude_rows(h, sigma)
    })
}

/// Rust-owned timer for grouped fitting. Offset conversion/validation is input
/// preparation outside samples; each sample includes grouped orchestration,
/// per-group row-fit setup, Rayon collection, and result allocation.
#[pyfunction]
#[pyo3(signature = (h_rows, sigma_rows, group_offsets, reps, trials, warmup_reps=1))]
fn benchmark_fit_absolute_magnitude_grouped_numpy(
    h_rows: PyReadonlyArray1<'_, f64>,
    sigma_rows: PyReadonlyArray1<'_, f64>,
    group_offsets: PyReadonlyArray1<'_, i64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let h = h_rows.as_array();
    let sigma = sigma_rows.as_array();
    let offsets = group_offsets.as_array();
    if h.len() != sigma.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let sigma = sigma
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    let offsets = offsets
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("group_offsets must be contiguous"))?;
    if offsets.len() < 2
        || offsets[0] != 0
        || offsets.last().copied() != Some(h.len() as i64)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || offsets.iter().any(|&offset| offset < 0)
    {
        return Err(PyValueError::new_err(
            "group_offsets must be monotonic from 0 through the row count",
        ));
    }
    let offsets: Vec<usize> = offsets.iter().map(|&offset| offset as usize).collect();
    benchmark_samples(reps, trials, warmup_reps, || {
        fit_absolute_magnitude_grouped(h, sigma, &offsets)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_rows_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_grouped_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_phase_angle_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_apparent_magnitude_v_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        calculate_apparent_magnitude_v_and_phase_angle_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(predict_magnitudes_fused_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(predict_magnitudes_complete_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_facade_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_complete_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_complete_photometry_facades, m)?)?;
    m.add_function(wrap_pyfunction!(predict_magnitudes_bandpass_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calculate_phase_angle_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_calculate_apparent_magnitude_v_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_calculate_apparent_magnitude_v_and_phase_angle_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_predict_magnitudes_bandpass_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_fit_absolute_magnitude_rows_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_fit_absolute_magnitude_grouped_numpy,
        m
    )?)?;
    Ok(())
}
