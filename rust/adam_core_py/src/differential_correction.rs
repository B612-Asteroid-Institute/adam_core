//! PyO3 surface for the whitened-residual differential-correction kernels
//! (observation whitening, the autodiff 2-body Jacobian, Huber loss helpers)
//! and the CMC2003 outlier-rejection decision kernels. The scipy optimizer,
//! the N-body residual evaluation through a `Propagator`, and the CMC2003
//! refit loop stay in the Python veneer.

use adam_core_rs_coords::{
    cmc2003_apparitions, cmc2003_expected_residual_chi2, cmc2003_select,
    observation_whitening_matrices, robust_cost, robust_jacobian_scale, robust_weights,
    validate_loss, whiten_residual_pairs, whitened_2body_jacobian, whitened_2body_model_angles,
    Cmc2003SelectionOptions, LossType, TwoBodyJacobianTerms, TwoBodyModelConfig, WhiteningError,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn scalars(values: &PyReadonlyArray1<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    values
        .as_array()
        .as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| value_error(format!("{label} must be contiguous")))
}

fn rows_n(values: &PyReadonlyArray2<'_, f64>, width: usize, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    if view.ncols() != width {
        return Err(value_error(format!("{label} must have shape (N, {width})")));
    }
    view.as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| value_error(format!("{label} must be contiguous")))
}

fn rows6(values: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<Vec<[f64; 6]>> {
    Ok(rows_n(values, 6, label)?
        .chunks_exact(6)
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect())
}

fn state6(values: &PyReadonlyArray1<'_, f64>, label: &str) -> PyResult<[f64; 6]> {
    let values = scalars(values, label)?;
    if values.len() != 6 {
        return Err(value_error(format!("{label} must have shape (6,)")));
    }
    Ok([
        values[0], values[1], values[2], values[3], values[4], values[5],
    ])
}

fn covariance_rows(values: &PyReadonlyArray3<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    let shape = view.shape();
    if shape[1] != 6 || shape[2] != 6 {
        return Err(value_error(format!("{label} must have shape (N, 6, 6)")));
    }
    view.as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| value_error(format!("{label} must be contiguous")))
}

fn whitener_rows(values: &PyReadonlyArray3<'_, f64>, label: &str) -> PyResult<Vec<[f64; 4]>> {
    let view = values.as_array();
    let shape = view.shape();
    if shape[1] != 2 || shape[2] != 2 {
        return Err(value_error(format!("{label} must have shape (N, 2, 2)")));
    }
    let flat = view
        .as_slice()
        .ok_or_else(|| value_error(format!("{label} must be contiguous")))?;
    Ok(flat
        .chunks_exact(4)
        .map(|w| [w[0], w[1], w[2], w[3]])
        .collect())
}

fn loss_and_scale(loss: &str, f_scale: f64) -> PyResult<LossType> {
    validate_loss(loss, f_scale).map_err(value_error)
}

/// Per-observation whitening matrices `(N, 2, 2)`: inverse Cholesky factors of
/// the cos(lat)-corrected (lon, lat) covariance blocks. `ids` name the
/// offending observation in the error message.
#[pyfunction]
fn observation_whitening_matrices_numpy<'py>(
    py: Python<'py>,
    covariances: PyReadonlyArray3<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    ids: Vec<String>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let covariances = covariance_rows(&covariances, "covariances")?;
    let lat = scalars(&lat, "lat")?;
    if covariances.len() != lat.len() * 36 {
        return Err(value_error("covariances must have one row per latitude"));
    }
    let whiteners = observation_whitening_matrices(&covariances, &lat).map_err(|err| {
        let (row, message) = match err {
            WhiteningError::NonFinite { row } => (
                row,
                "has non-finite (lon, lat) covariance entries; least-squares fitting \
                 requires finite angular uncertainties for every observation.",
            ),
            WhiteningError::NotPositiveDefinite { row } => (
                row,
                "has a non-positive-definite (lon, lat) covariance block.",
            ),
        };
        let label = ids
            .get(row)
            .map(|id| format!("{id:?}"))
            .unwrap_or_else(|| row.to_string());
        value_error(format!("Observation {label} {message}"))
    })?;
    let n = whiteners.len();
    let flat: Vec<f64> = whiteners.into_iter().flatten().collect();
    ndarray::Array3::from_shape_vec((n, 2, 2), flat)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| value_error(err.to_string()))
}

/// Whiten `(N, 2)` cos(lat)-corrected (lon, lat) residual pairs into the flat
/// `(2N,)` component vector.
#[pyfunction]
fn whiten_residual_pairs_numpy<'py>(
    py: Python<'py>,
    whiteners: PyReadonlyArray3<'py, f64>,
    residual_pairs: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let whiteners = whitener_rows(&whiteners, "whiteners")?;
    let pairs = rows_n(&residual_pairs, 2, "residual_pairs")?;
    if pairs.len() != whiteners.len() * 2 {
        return Err(value_error("residual_pairs must have one row per whitener"));
    }
    Ok(whiten_residual_pairs(&whiteners, &pairs).into_pyarray(py))
}

#[allow(clippy::too_many_arguments)]
fn jacobian_terms(
    times_mjd_tdb: &PyReadonlyArray1<'_, f64>,
    observer_states: &PyReadonlyArray2<'_, f64>,
    sun_states: &PyReadonlyArray2<'_, f64>,
    cos_lats: &PyReadonlyArray1<'_, f64>,
    whiteners: &PyReadonlyArray3<'_, f64>,
    mu_helio: f64,
    mu_bary: f64,
) -> PyResult<TwoBodyJacobianTerms> {
    let terms = TwoBodyJacobianTerms {
        times_mjd_tdb: scalars(times_mjd_tdb, "times_mjd_tdb")?,
        observer_states: rows6(observer_states, "observer_states")?,
        sun_states: rows6(sun_states, "sun_states")?,
        cos_lats: scalars(cos_lats, "cos_lats")?,
        whiteners: whitener_rows(whiteners, "whiteners")?,
        mu_helio,
        mu_bary,
    };
    terms.validate().map_err(value_error)?;
    Ok(terms)
}

/// Predicted whitened (lon, lat) angles `(2N,)` of the 2-body model for a
/// heliocentric ecliptic epoch state (see `whitened_2body_jacobian_numpy`).
#[pyfunction]
#[pyo3(signature = (state, epoch_mjd_tdb, times_mjd_tdb, observer_states, sun_states, cos_lats, whiteners, mu_helio, mu_bary))]
#[allow(clippy::too_many_arguments)]
fn whitened_2body_model_angles_numpy<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    epoch_mjd_tdb: f64,
    times_mjd_tdb: PyReadonlyArray1<'py, f64>,
    observer_states: PyReadonlyArray2<'py, f64>,
    sun_states: PyReadonlyArray2<'py, f64>,
    cos_lats: PyReadonlyArray1<'py, f64>,
    whiteners: PyReadonlyArray3<'py, f64>,
    mu_helio: f64,
    mu_bary: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let state = state6(&state, "state")?;
    let terms = jacobian_terms(
        &times_mjd_tdb,
        &observer_states,
        &sun_states,
        &cos_lats,
        &whiteners,
        mu_helio,
        mu_bary,
    )?;
    let config = TwoBodyModelConfig::default();
    let angles = py
        .allow_threads(|| whitened_2body_model_angles(state, epoch_mjd_tdb, &terms, &config))
        .map_err(value_error)?;
    Ok(angles.into_pyarray(py))
}

/// Analytic Jacobian `(2N, 6)` of the whitened residual vector with respect to
/// the heliocentric ecliptic epoch state: the topocentric spherical
/// projection (with light time) chained through the 2-body state transition
/// matrix by forward-mode automatic differentiation over `Dual<6>`.
/// `observer_states` and `sun_states` are barycentric ecliptic `(N, 6)`.
#[pyfunction]
#[pyo3(signature = (state, epoch_mjd_tdb, times_mjd_tdb, observer_states, sun_states, cos_lats, whiteners, mu_helio, mu_bary))]
#[allow(clippy::too_many_arguments)]
fn whitened_2body_jacobian_numpy<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    epoch_mjd_tdb: f64,
    times_mjd_tdb: PyReadonlyArray1<'py, f64>,
    observer_states: PyReadonlyArray2<'py, f64>,
    sun_states: PyReadonlyArray2<'py, f64>,
    cos_lats: PyReadonlyArray1<'py, f64>,
    whiteners: PyReadonlyArray3<'py, f64>,
    mu_helio: f64,
    mu_bary: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let state = state6(&state, "state")?;
    let terms = jacobian_terms(
        &times_mjd_tdb,
        &observer_states,
        &sun_states,
        &cos_lats,
        &whiteners,
        mu_helio,
        mu_bary,
    )?;
    let config = TwoBodyModelConfig::default();
    let jacobian = py
        .allow_threads(|| whitened_2body_jacobian(state, epoch_mjd_tdb, &terms, &config))
        .map_err(value_error)?;
    let rows = 2 * terms.len();
    ndarray::Array2::from_shape_vec((rows, 6), jacobian)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| value_error(err.to_string()))
}

/// Objective minimized by `fit_least_squares` for the given loss (chi2 for
/// `linear`; the Huber cost, twice scipy's `cost`, for `huber`).
#[pyfunction]
fn robust_cost_numpy(
    residuals: PyReadonlyArray1<'_, f64>,
    loss: &str,
    f_scale: f64,
) -> PyResult<f64> {
    let loss = loss_and_scale(loss, f_scale)?;
    Ok(robust_cost(
        &scalars(&residuals, "residuals")?,
        loss,
        f_scale,
    ))
}

/// IRLS weight `rho'(z)` per whitened residual component at the solution.
#[pyfunction]
fn robust_weights_numpy<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray1<'py, f64>,
    loss: &str,
    f_scale: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let loss = loss_and_scale(loss, f_scale)?;
    Ok(robust_weights(&scalars(&residuals, "residuals")?, loss, f_scale).into_pyarray(py))
}

/// Row scaling turning the residual Jacobian into the Gauss-Newton Jacobian of
/// the robust cost (scipy's `sqrt(rho' + 2 rho'' r**2)` convention).
#[pyfunction]
fn robust_jacobian_scale_numpy<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray1<'py, f64>,
    loss: &str,
    f_scale: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let loss = loss_and_scale(loss, f_scale)?;
    Ok(robust_jacobian_scale(&scalars(&residuals, "residuals")?, loss, f_scale).into_pyarray(py))
}

/// Validate a `(loss, f_scale)` pair; raises `ValueError` with the legacy
/// messages.
#[pyfunction]
fn validate_robust_loss(loss: &str, f_scale: f64) -> PyResult<String> {
    Ok(loss_and_scale(loss, f_scale)?.as_str().to_string())
}

/// CMC2003 apparition index per observation: time-sorted groups split at gaps
/// longer than `gap_days`.
#[pyfunction]
fn cmc2003_apparitions_numpy<'py>(
    py: Python<'py>,
    mjd: PyReadonlyArray1<'py, f64>,
    gap_days: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(cmc2003_apparitions(&scalars(&mjd, "mjd")?, gap_days).into_pyarray(py))
}

type ExpectedChi2<'py> = (Bound<'py, PyArray1<f64>>, Vec<String>);

/// CMC2003 per-observation chi2 against the expected post-fit residual
/// covariance `I -/+ J_i C J_i^T` (whitened units) with the eigenvalue floor;
/// returns `(chi2, flags)`.
#[pyfunction]
#[pyo3(signature = (residuals, jacobian, covariance, selected, psd_floor_frac))]
fn cmc2003_expected_residual_chi2_numpy<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray2<'py, f64>,
    jacobian: PyReadonlyArray2<'py, f64>,
    covariance: Option<PyReadonlyArray2<'py, f64>>,
    selected: PyReadonlyArray1<'py, bool>,
    psd_floor_frac: f64,
) -> PyResult<ExpectedChi2<'py>> {
    let residuals = rows_n(&residuals, 2, "residuals")?;
    let jacobian = rows_n(&jacobian, 6, "jacobian")?;
    let covariance = match covariance {
        Some(values) => {
            let flat = rows_n(&values, 6, "covariance")?;
            if flat.len() != 36 {
                return Err(value_error("covariance must have shape (6, 6)"));
            }
            Some(flat)
        }
        None => None,
    };
    let selected = selected
        .as_array()
        .as_slice()
        .map(<[bool]>::to_vec)
        .ok_or_else(|| value_error("selected must be contiguous"))?;
    let (chi2, flags) = cmc2003_expected_residual_chi2(
        &residuals,
        &jacobian,
        covariance.as_deref(),
        &selected,
        psd_floor_frac,
    )
    .map_err(value_error)?;
    Ok((
        chi2.into_pyarray(py),
        flags
            .into_iter()
            .map(|flag| flag.as_str().to_string())
            .collect(),
    ))
}

type SelectionOutput<'py> = (Bound<'py, PyArray1<bool>>, usize, usize, Vec<String>);

/// One CMC2003 reject / re-include decision pass; returns
/// `(selected, n_rejected, n_recovered, flags)`.
#[pyfunction]
#[pyo3(signature = (chi2, selected, apparitions, chi2_reject, chi2_recover, chi2_frac, max_rejected_fraction, one_at_a_time, min_obs))]
#[allow(clippy::too_many_arguments)]
fn cmc2003_select_numpy<'py>(
    py: Python<'py>,
    chi2: PyReadonlyArray1<'py, f64>,
    selected: PyReadonlyArray1<'py, bool>,
    apparitions: PyReadonlyArray1<'py, i64>,
    chi2_reject: f64,
    chi2_recover: f64,
    chi2_frac: f64,
    max_rejected_fraction: f64,
    one_at_a_time: bool,
    min_obs: usize,
) -> PyResult<SelectionOutput<'py>> {
    let chi2 = scalars(&chi2, "chi2")?;
    let selected = selected
        .as_array()
        .as_slice()
        .map(<[bool]>::to_vec)
        .ok_or_else(|| value_error("selected must be contiguous"))?;
    let apparitions = apparitions
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| value_error("apparitions must be contiguous"))?;
    let options = Cmc2003SelectionOptions {
        chi2_reject,
        chi2_recover,
        chi2_frac,
        max_rejected_fraction,
        one_at_a_time,
        min_obs,
    };
    let selection =
        cmc2003_select(&chi2, &selected, &apparitions, &options).map_err(value_error)?;
    Ok((
        selection.selected.into_pyarray(py),
        selection.n_rejected,
        selection.n_recovered,
        selection
            .flags
            .into_iter()
            .map(|flag| flag.as_str().to_string())
            .collect(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(observation_whitening_matrices_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(whiten_residual_pairs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(whitened_2body_model_angles_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(whitened_2body_jacobian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(robust_cost_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(robust_weights_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(robust_jacobian_scale_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(validate_robust_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_apparitions_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_expected_residual_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_select_numpy, m)?)?;
    Ok(())
}
