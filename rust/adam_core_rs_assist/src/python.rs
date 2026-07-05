#![allow(clippy::useless_conversion)] // PyO3 0.22 macro expansion trips this lint on generated wrappers.

use crate::AssistPropagator as RustAssistPropagator;
use crate::{map_origin_code_to_assist_body, CollisionConditionSpec};
use adam_core_rs_coords::propagation::fit_orbit_least_squares_barycentric;
use adam_core_rs_coords::propagation::{
    CovariancePropagation, EpochPolicy, PropagationOptions, PropagationRequest, PropagationResult,
    Propagator,
};
use adam_core_rs_coords::types::Frame;
use adam_core_rs_coords::types::{SchemaResult, TimeScaleProvider};
use adam_core_rs_coords::{
    collapse_propagated_variants_to_orbits, create_sampled_orbit_variants, CoordinateBatch,
    CoordinateRepresentation, CovarianceBatch, CovarianceUnits, EphemerisOptions,
    EphemerisPhotometryOptions, EphemerisResult, ObjectId, ObservatoryCode, ObserverBatch,
    OrbitBatch, OrbitId, OrbitVariantBatch, OrbitVariantSamplingMethod, OriginArray, OriginId,
    SchemaError, TimeArray, TimeScale, VariantId,
};
use adam_core_rs_coords::{CoordinateValues, LeastSquaresConfig};
use adam_core_rs_spice::AdamCoreSpiceBackend;
use assist_rs::{AssistData, Ephemeris, Ias15AdaptiveMode, IntegratorConfig};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use std::sync::Arc;

struct PythonTimeProvider;

impl TimeScaleProvider for PythonTimeProvider {
    fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray> {
        Err(SchemaError::InvalidRecordBatch(format!(
            "adam_assist_rust cannot rescale {} to {} without an explicit provider",
            times.scale.as_str(),
            new_scale.as_str()
        )))
    }
}

#[pyclass]
struct NativeAssistPropagator {
    inner: RustAssistPropagator,
    spice: AdamCoreSpiceBackend,
}

#[pymethods]
impl NativeAssistPropagator {
    #[new]
    #[pyo3(signature = (planets_path, asteroids_path, *, min_dt=1.0e-9, initial_dt=1.0e-6, adaptive_mode=1, epsilon=1.0e-6))]
    fn new(
        planets_path: &str,
        asteroids_path: &str,
        min_dt: f64,
        initial_dt: f64,
        adaptive_mode: i32,
        epsilon: f64,
    ) -> PyResult<Self> {
        if min_dt <= 0.0 {
            return Err(PyValueError::new_err("min_dt must be positive"));
        }
        if initial_dt <= 0.0 {
            return Err(PyValueError::new_err("initial_dt must be positive"));
        }
        if min_dt > initial_dt {
            return Err(PyValueError::new_err(
                "min_dt must be smaller than initial_dt",
            ));
        }
        let adaptive_mode = parse_adaptive_mode(adaptive_mode)?;
        let ephem = Ephemeris::from_paths(Path::new(planets_path), Path::new(asteroids_path))
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to load ASSIST kernels: {err}"))
            })?;
        let integrator = IntegratorConfig {
            initial_dt: Some(initial_dt),
            min_dt: Some(min_dt),
            adaptive_mode: Some(adaptive_mode),
            epsilon: Some(epsilon),
        };
        let mut spice = AdamCoreSpiceBackend::new();
        spice.furnsh(Path::new(planets_path)).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to load SPICE planets kernel: {err}"))
        })?;
        Ok(Self {
            inner: RustAssistPropagator::with_integrator(
                Arc::new(AssistData::new(ephem)),
                integrator,
            ),
            spice,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        output_time_scale,
        lt_tol=1.0e-12,
        max_iter=1000,
        tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10,
        predict_magnitude_v=false,
        predict_phase_angle=false,
        h_v=None,
        g=None,
        chunk_size=None,
        thread_limit=None
    ))]
    fn generate_ephemeris<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        output_time_scale: &str,
        lt_tol: f64,
        max_iter: usize,
        tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
        predict_magnitude_v: bool,
        predict_phase_angle: bool,
        h_v: Option<Vec<Option<f64>>>,
        g: Option<Vec<Option<f64>>>,
        chunk_size: Option<usize>,
        thread_limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let orbit_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(orbit_states)?,
            Frame::parse(orbit_frame).map_err(py_value_error)?,
            OriginArray::new(
                orbit_origin_codes
                    .into_iter()
                    .map(OriginId::from_code)
                    .collect(),
            ),
            Some(time_array(
                orbit_time_scale,
                orbit_time_days,
                orbit_time_nanos,
            )?),
            None,
        )
        .map_err(py_value_error)?;
        let orbits = OrbitBatch::new(
            orbit_ids.into_iter().map(OrbitId).collect(),
            object_ids
                .into_iter()
                .map(|value| value.map(ObjectId))
                .collect(),
            orbit_coordinates,
        )
        .map_err(py_value_error)?;
        let observer_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(observer_states)?,
            Frame::parse(observer_frame).map_err(py_value_error)?,
            OriginArray::new(
                observer_origin_codes
                    .into_iter()
                    .map(OriginId::from_code)
                    .collect(),
            ),
            Some(time_array(
                observer_time_scale,
                observer_time_days,
                observer_time_nanos,
            )?),
            None,
        )
        .map_err(py_value_error)?;
        let observers = ObserverBatch::new(
            observer_codes.into_iter().map(ObservatoryCode).collect(),
            observer_coordinates,
        )
        .map_err(py_value_error)?;
        let options = EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size,
                thread_limit,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            lt_tol,
            max_iter,
            tol,
            stellar_aberration,
            max_lt_iter,
            output_time_scale: TimeScale::parse(output_time_scale).map_err(py_value_error)?,
            include_aberrated_coordinates: true,
            photometry: EphemerisPhotometryOptions {
                predict_magnitude_v,
                predict_phase_angle,
                h_v,
                g,
            },
        };
        let result = py
            .allow_threads(|| {
                self.inner.generate_ephemeris(
                    &orbits,
                    &observers,
                    &options,
                    &PythonTimeProvider,
                    &self.spice,
                )
            })
            .map_err(py_runtime_error)?;
        ephemeris_result_to_dict(py, &result)
    }

    /// Backend-generic least-squares orbit determination instantiated with
    /// the ASSIST propagator (bead personal-cmy.7). The Gauss-Newton driver
    /// lives in the permissive core (`fit_orbit_least_squares_barycentric`);
    /// this GPL boundary only supplies the propagator, mirroring the
    /// adam-assist packaging decision. Returns
    /// `(state (6,), covariance (36,), chi2, iterations, converged)` in the
    /// input orbit's frame/origin.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observed_values,
        observed_covariances,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        xtol=1e-12,
        ftol=1e-12,
        max_iterations=100,
        lt_tol=1.0e-12,
        eph_max_iter=1000,
        eph_tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10
    ))]
    fn fit_orbit_least_squares<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observed_values: PyReadonlyArray2<'py, f64>,
        observed_covariances: PyReadonlyArray2<'py, f64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        xtol: f64,
        ftol: f64,
        max_iterations: usize,
        lt_tol: f64,
        eph_max_iter: usize,
        eph_tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64, usize, bool)> {
        let orbit_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(orbit_states)?,
            Frame::parse(orbit_frame).map_err(py_value_error)?,
            OriginArray::new(
                orbit_origin_codes
                    .into_iter()
                    .map(OriginId::from_code)
                    .collect(),
            ),
            Some(time_array(
                orbit_time_scale,
                orbit_time_days,
                orbit_time_nanos,
            )?),
            None,
        )
        .map_err(py_value_error)?;
        let orbit = OrbitBatch::new(
            orbit_ids.into_iter().map(OrbitId).collect(),
            object_ids
                .into_iter()
                .map(|value| value.map(ObjectId))
                .collect(),
            orbit_coordinates,
        )
        .map_err(py_value_error)?;

        let observer_times =
            time_array(observer_time_scale, observer_time_days, observer_time_nanos)?;
        let observer_origins = OriginArray::new(
            observer_origin_codes
                .into_iter()
                .map(OriginId::from_code)
                .collect(),
        );
        let observed_rows = states_from_pyarray(observed_values)?;
        let n = observed_rows.len();
        let observed_cov = observed_covariances.as_array();
        if observed_cov.nrows() != n || observed_cov.ncols() != 36 {
            return Err(PyValueError::new_err(
                "observed_covariances must have shape (N, 36)",
            ));
        }
        let observed_cov_flat: Vec<f64> = observed_cov.iter().copied().collect();
        let covariance = CovarianceBatch::new(
            n,
            6,
            observed_cov_flat,
            CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
        )
        .map_err(py_value_error)?;
        let observed = CoordinateBatch::new(
            CoordinateValues::Spherical(observed_rows),
            Frame::parse(observer_frame).map_err(py_value_error)?,
            observer_origins.clone(),
            Some(observer_times.clone()),
            Some(covariance),
        )
        .map_err(py_value_error)?;

        let observer_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(observer_states)?,
            Frame::parse(observer_frame).map_err(py_value_error)?,
            observer_origins,
            Some(observer_times),
            None,
        )
        .map_err(py_value_error)?;
        let observers = ObserverBatch::new(
            observer_codes.into_iter().map(ObservatoryCode).collect(),
            observer_coordinates,
        )
        .map_err(py_value_error)?;

        let options = EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            lt_tol,
            max_iter: eph_max_iter,
            tol: eph_tol,
            stellar_aberration,
            max_lt_iter,
            output_time_scale: TimeScale::parse(observer_time_scale).map_err(py_value_error)?,
            include_aberrated_coordinates: false,
            photometry: EphemerisPhotometryOptions::default(),
        };
        let config = LeastSquaresConfig {
            xtol,
            ftol,
            max_iterations,
            lt_tol,
            ephemeris_max_iter: eph_max_iter,
            ephemeris_tol: eph_tol,
            stellar_aberration,
            max_lt_iter,
        };
        let fit = py
            .allow_threads(|| {
                fit_orbit_least_squares_barycentric(
                    &self.inner,
                    &orbit,
                    &observed,
                    &observers,
                    &options,
                    &config,
                    &PythonTimeProvider,
                    &self.spice,
                )
            })
            .map_err(|err| {
                PyRuntimeError::new_err(format!("assist least-squares fit failed: {err}"))
            })?;
        Ok((
            fit.state.to_vec(),
            fit.covariance.to_vec(),
            fit.chi2,
            fit.iterations,
            fit.converged,
        ))
    }

    /// Same-epoch collision detection mirroring
    /// `adam_assist.ASSISTPropagator._detect_collisions`. `states` must be
    /// barycentric equatorial (N, 6) at one shared TDB epoch; the epoch and
    /// horizon are TDB Julian dates computed by the Python boundary exactly
    /// as legacy does. Returns per-row survivor states/indices at the final
    /// executed (overshooting) step plus one impact record per
    /// condition-step detection, all in barycentric equatorial with TDB
    /// Julian-date times.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        states,
        epoch_jd_tdb,
        final_jd_tdb,
        condition_bodies,
        condition_distances_km,
        condition_stopping
    ))]
    fn detect_collisions<'py>(
        &self,
        py: Python<'py>,
        states: PyReadonlyArray2<'py, f64>,
        epoch_jd_tdb: f64,
        final_jd_tdb: f64,
        condition_bodies: Vec<String>,
        condition_distances_km: Vec<f64>,
        condition_stopping: Vec<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if condition_bodies.len() != condition_distances_km.len()
            || condition_bodies.len() != condition_stopping.len()
        {
            return Err(PyValueError::new_err(
                "collision condition arrays must share one length",
            ));
        }
        let mut conditions = Vec::with_capacity(condition_bodies.len());
        for ((code, distance_km), stopping) in condition_bodies
            .iter()
            .zip(condition_distances_km)
            .zip(condition_stopping)
        {
            let body = map_origin_code_to_assist_body(code).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unsupported collision object code for ASSIST ephemeris: {code}"
                ))
            })?;
            conditions.push(CollisionConditionSpec {
                body,
                distance_km,
                stopping,
            });
        }
        let states = states_from_pyarray(states)?;
        let output = py
            .allow_threads(|| {
                self.inner.detect_collisions_same_epoch(
                    &states,
                    epoch_jd_tdb,
                    final_jd_tdb,
                    &conditions,
                )
            })
            .map_err(|err| {
                PyRuntimeError::new_err(format!("assist collision detection failed: {err}"))
            })?;

        let dict = PyDict::new_bound(py);
        dict.set_item(
            "final_indices",
            output
                .final_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "final_states",
            shaped_states_array(py, &output.final_states)?,
        )?;
        dict.set_item("final_time_jd_tdb", output.final_time_jd_tdb)?;
        dict.set_item(
            "impact_indices",
            output
                .impact_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "impact_condition_indices",
            output
                .impact_condition_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "impact_states",
            shaped_states_array(py, &output.impact_states)?,
        )?;
        dict.set_item("impact_times_jd_tdb", output.impact_times_jd_tdb.clone())?;
        Ok(dict)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        states,
        origin_codes,
        frame,
        time_scale,
        time_days,
        time_nanos,
        target_scale,
        target_days,
        target_nanos,
        covariance,
        covariances=None,
        covariance_method="monte-carlo",
        num_samples=1000,
        seed=None,
        chunk_size=None,
        thread_limit=None,
        variant_ids=None,
        weights=None,
        weights_cov=None
    ))]
    fn propagate_orbits<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        states: PyReadonlyArray2<'py, f64>,
        origin_codes: Vec<String>,
        frame: &str,
        time_scale: &str,
        time_days: Vec<i64>,
        time_nanos: Vec<i64>,
        target_scale: &str,
        target_days: Vec<i64>,
        target_nanos: Vec<i64>,
        covariance: bool,
        covariances: Option<PyReadonlyArray2<'py, f64>>,
        covariance_method: &str,
        num_samples: usize,
        seed: Option<u64>,
        chunk_size: Option<usize>,
        thread_limit: Option<usize>,
        variant_ids: Option<Vec<Option<String>>>,
        weights: Option<Vec<Option<f64>>>,
        weights_cov: Option<Vec<Option<f64>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let state_rows = states_from_pyarray(states)?;
        let input_times = time_array(time_scale, time_days, time_nanos)?;
        let target_times = time_array(target_scale, target_days, target_nanos)?;
        let covariance_batch = match covariances {
            Some(covariances) => Some(covariance_from_pyarray(covariances, state_rows.len())?),
            None => None,
        };
        let coordinates = CoordinateBatch::cartesian(
            state_rows,
            Frame::parse(frame).map_err(py_value_error)?,
            OriginArray::new(origin_codes.into_iter().map(OriginId::from_code).collect()),
            Some(input_times),
            covariance_batch,
        )
        .map_err(py_value_error)?;
        let options = PropagationOptions {
            chunk_size,
            thread_limit,
            epoch_policy: EpochPolicy::CrossProduct,
            covariance: CovariancePropagation::None,
        };
        let result = match variant_ids {
            Some(variant_ids) => {
                if covariance {
                    return Err(PyValueError::new_err(
                        "covariance=True is not supported for VariantOrbits",
                    ));
                }
                let weights = weights
                    .ok_or_else(|| PyValueError::new_err("variant propagation requires weights"))?;
                let weights_cov = weights_cov.ok_or_else(|| {
                    PyValueError::new_err("variant propagation requires weights_cov")
                })?;
                let variants = OrbitVariantBatch::new(
                    orbit_ids.into_iter().map(OrbitId).collect(),
                    object_ids
                        .into_iter()
                        .map(|value| value.map(ObjectId))
                        .collect(),
                    variant_ids
                        .into_iter()
                        .map(|value| value.map(VariantId))
                        .collect(),
                    weights,
                    weights_cov,
                    coordinates,
                )
                .map_err(py_value_error)?;
                let request = PropagationRequest::new_variants(&variants, &target_times, options)
                    .map_err(py_runtime_error)?;
                py.allow_threads(|| self.inner.propagate(&request, &PythonTimeProvider))
                    .map_err(py_runtime_error)?
            }
            None => {
                let orbits = OrbitBatch::new(
                    orbit_ids.into_iter().map(OrbitId).collect(),
                    object_ids
                        .into_iter()
                        .map(|value| value.map(ObjectId))
                        .collect(),
                    coordinates,
                )
                .map_err(py_value_error)?;
                if covariance {
                    propagate_with_sampled_covariance(
                        py,
                        &self.inner,
                        &orbits,
                        &target_times,
                        options,
                        covariance_method,
                        num_samples,
                        seed,
                    )?
                } else {
                    let request = PropagationRequest::new(&orbits, &target_times, options)
                        .map_err(py_runtime_error)?;
                    py.allow_threads(|| self.inner.propagate(&request, &PythonTimeProvider))
                        .map_err(py_runtime_error)?
                }
            }
        };
        propagation_result_to_dict(py, &result)
    }
}

#[allow(clippy::too_many_arguments)]
fn propagate_with_sampled_covariance(
    py: Python<'_>,
    propagator: &RustAssistPropagator,
    orbits: &OrbitBatch,
    target_times: &TimeArray,
    mut options: PropagationOptions,
    covariance_method: &str,
    num_samples: usize,
    seed: Option<u64>,
) -> PyResult<PropagationResult> {
    let method = parse_covariance_method(covariance_method)?;
    if orbits.coordinates.covariance.is_none() {
        return Err(PyValueError::new_err(
            "covariance=True requires input coordinate covariance rows",
        ));
    }
    // Public covariance mirrors Python `adam_assist`: sample variants, propagate them,
    // and collapse to nominal covariance. The lower-level ASSIST STM transport
    // (`CovariancePropagation::Linearized`) is intentionally NOT used here; it stays a
    // separate typed Rust-trait surface (see `AssistPropagator::supports`). The public
    // path therefore forces `None` and owns covariance via sampled variants.
    options.covariance = CovariancePropagation::None;
    let variant_samples =
        create_sampled_orbit_variants(orbits, method, num_samples, seed, 1.0, 0.0, 0.0)
            .map_err(py_value_error)?;
    let nominal_request =
        PropagationRequest::new(orbits, target_times, options.clone()).map_err(py_runtime_error)?;
    let nominal = py
        .allow_threads(|| propagator.propagate(&nominal_request, &PythonTimeProvider))
        .map_err(py_runtime_error)?;
    let variant_request =
        PropagationRequest::new_variants(&variant_samples.variants, target_times, options)
            .map_err(py_runtime_error)?;
    let propagated_variants = py
        .allow_threads(|| propagator.propagate(&variant_request, &PythonTimeProvider))
        .map_err(py_runtime_error)?;
    collapse_propagated_variants_to_orbits(
        &nominal,
        &propagated_variants,
        &variant_samples.source_orbit_indices,
    )
    .map_err(py_runtime_error)
}

fn parse_covariance_method(value: &str) -> PyResult<OrbitVariantSamplingMethod> {
    match value {
        "auto" => Ok(OrbitVariantSamplingMethod::Auto),
        "sigma-point" => Ok(OrbitVariantSamplingMethod::SigmaPoint),
        "monte-carlo" => Ok(OrbitVariantSamplingMethod::MonteCarlo),
        other => Err(PyValueError::new_err(format!(
            "covariance_method must be one of 'auto', 'sigma-point', or 'monte-carlo'; got {other:?}"
        ))),
    }
}

fn parse_adaptive_mode(value: i32) -> PyResult<Ias15AdaptiveMode> {
    match value {
        0 => Ok(Ias15AdaptiveMode::Individual),
        1 => Ok(Ias15AdaptiveMode::Global),
        2 => Ok(Ias15AdaptiveMode::Prs23),
        3 => Ok(Ias15AdaptiveMode::Aarseth85),
        _ => Err(PyValueError::new_err(format!(
            "adaptive_mode must be one of 0, 1, 2, or 3; got {value}"
        ))),
    }
}

fn shaped_states_array<'py>(
    py: Python<'py>,
    states: &[[f64; 6]],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = states
        .iter()
        .flat_map(|state| state.iter().copied())
        .collect();
    let shaped = ndarray::Array2::from_shape_vec((states.len(), 6), flat)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to shape states: {err}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

fn states_from_pyarray(states: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 6]>> {
    let array = states.as_array();
    if array.ncols() != 6 {
        return Err(PyValueError::new_err(format!(
            "states must have shape (N, 6); got ({}, {})",
            array.nrows(),
            array.ncols()
        )));
    }
    let mut rows = Vec::with_capacity(array.nrows());
    for row in array.rows() {
        rows.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
    }
    Ok(rows)
}

fn time_array(scale: &str, days: Vec<i64>, nanos: Vec<i64>) -> PyResult<TimeArray> {
    TimeArray::from_parts(
        TimeScale::parse(scale).map_err(py_value_error)?,
        days,
        nanos,
    )
    .map_err(py_value_error)
}

fn covariance_from_pyarray(
    covariances: PyReadonlyArray2<'_, f64>,
    rows: usize,
) -> PyResult<CovarianceBatch> {
    let array = covariances.as_array();
    if array.nrows() != rows || array.ncols() != 36 {
        return Err(PyValueError::new_err(format!(
            "covariances must have shape ({rows}, 36); got ({}, {})",
            array.nrows(),
            array.ncols()
        )));
    }
    CovarianceBatch::new(
        rows,
        6,
        array.iter().copied().collect(),
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )
    .map_err(py_value_error)
}

fn propagation_result_to_dict<'py>(
    py: Python<'py>,
    result: &adam_core_rs_coords::propagation::PropagationResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    let (orbit_ids, object_ids, variant_ids, weights, weights_cov, coordinates) =
        if let Some(variants) = &result.variants {
            (
                variants
                    .orbit_id
                    .iter()
                    .map(|value| value.0.clone())
                    .collect::<Vec<_>>(),
                variants
                    .object_id
                    .iter()
                    .map(|value| value.as_ref().map(|item| item.0.clone()))
                    .collect::<Vec<_>>(),
                Some(
                    variants
                        .variant_id
                        .iter()
                        .map(|value| value.as_ref().map(|item| item.0.clone()))
                        .collect::<Vec<_>>(),
                ),
                Some(variants.weights.clone()),
                Some(variants.weights_cov.clone()),
                &variants.coordinates,
            )
        } else {
            (
                result
                    .orbits
                    .orbit_id
                    .iter()
                    .map(|value| value.0.clone())
                    .collect::<Vec<_>>(),
                result
                    .orbits
                    .object_id
                    .iter()
                    .map(|value| value.as_ref().map(|item| item.0.clone()))
                    .collect::<Vec<_>>(),
                None,
                None,
                None,
                &result.orbits.coordinates,
            )
        };

    let states = coordinates.values.cartesian().ok_or_else(|| {
        PyRuntimeError::new_err("assist propagation returned non-Cartesian coordinates")
    })?;
    let flat_states = states
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let shaped_states = ndarray::Array2::from_shape_vec((states.len(), 6), flat_states)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to shape states: {err}")))?;
    let times = coordinates.times.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err("assist propagation returned coordinates without times")
    })?;
    let input_orbit_indices = result
        .diagnostics
        .convergence
        .iter()
        .map(|row| row.input_orbit_index)
        .collect::<Vec<_>>();
    let validity = (0..result.validity.len())
        .map(|index| result.validity.is_valid(index))
        .collect::<Vec<_>>();
    let messages = result
        .diagnostics
        .convergence
        .iter()
        .map(|row| row.message.clone())
        .collect::<Vec<_>>();

    dict.set_item("orbit_id", orbit_ids)?;
    dict.set_item("object_id", object_ids)?;
    dict.set_item("variant_id", variant_ids)?;
    dict.set_item("weights", weights)?;
    dict.set_item("weights_cov", weights_cov)?;
    dict.set_item("states", shaped_states.into_pyarray_bound(py))?;
    match &coordinates.covariance {
        Some(covariance) => {
            let shaped_covariance = ndarray::Array2::from_shape_vec(
                (covariance.rows, covariance.dimension * covariance.dimension),
                covariance.values_row_major.clone(),
            )
            .map_err(|err| PyRuntimeError::new_err(format!("failed to shape covariance: {err}")))?;
            dict.set_item("covariances", shaped_covariance.into_pyarray_bound(py))?;
        }
        None => dict.set_item("covariances", py.None())?,
    }
    dict.set_item(
        "origin_codes",
        coordinates
            .origins
            .origins
            .iter()
            .map(OriginId::code)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("frame", coordinates.frame.as_str())?;
    dict.set_item("time_scale", times.scale.as_str())?;
    dict.set_item(
        "time_days",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.days)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "time_nanos",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.nanos)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("input_orbit_indices", input_orbit_indices)?;
    dict.set_item("validity", validity)?;
    dict.set_item("messages", messages)?;
    Ok(dict)
}

fn ephemeris_result_to_dict<'py>(
    py: Python<'py>,
    result: &EphemerisResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    let ephemeris = &result.ephemeris;
    dict.set_item(
        "orbit_id",
        ephemeris
            .orbit_id
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "object_id",
        ephemeris
            .object_id
            .iter()
            .map(|value| value.as_ref().map(|item| item.0.clone()))
            .collect::<Vec<_>>(),
    )?;
    let coordinates = &ephemeris.coordinates;
    let states = coordinates
        .values
        .spherical()
        .ok_or_else(|| PyRuntimeError::new_err("ephemeris coordinates must be spherical"))?;
    let flat = states
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let shaped = ndarray::Array2::from_shape_vec((states.len(), 6), flat).map_err(|err| {
        PyRuntimeError::new_err(format!("failed to shape ephemeris states: {err}"))
    })?;
    dict.set_item("states", shaped.into_pyarray_bound(py))?;
    dict.set_item(
        "origin_codes",
        coordinates
            .origins
            .origins
            .iter()
            .map(OriginId::code)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("frame", coordinates.frame.as_str())?;
    let times = coordinates
        .times
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("ephemeris coordinates missing times"))?;
    dict.set_item("time_scale", times.scale.as_str())?;
    dict.set_item(
        "time_days",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.days)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "time_nanos",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.nanos)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("light_time", ephemeris.light_time_days.clone())?;
    match &ephemeris.alpha_deg {
        Some(values) => dict.set_item("alpha", values.clone())?,
        None => dict.set_item("alpha", py.None())?,
    }
    match &ephemeris.predicted_magnitude_v {
        Some(values) => dict.set_item("predicted_magnitude_v", values.clone())?,
        None => dict.set_item("predicted_magnitude_v", py.None())?,
    }
    match &ephemeris.aberrated_coordinates {
        Some(aberrated) => {
            let aberrated_states = aberrated.values.cartesian().ok_or_else(|| {
                PyRuntimeError::new_err("aberrated coordinates must be Cartesian")
            })?;
            let aberrated_flat = aberrated_states
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let aberrated_shaped =
                ndarray::Array2::from_shape_vec((aberrated_states.len(), 6), aberrated_flat)
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!("failed to shape aberrated states: {err}"))
                    })?;
            dict.set_item("aberrated_states", aberrated_shaped.into_pyarray_bound(py))?;
            dict.set_item(
                "aberrated_origin_codes",
                aberrated
                    .origins
                    .origins
                    .iter()
                    .map(OriginId::code)
                    .collect::<Vec<_>>(),
            )?;
            let aberrated_times = aberrated
                .times
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("aberrated coordinates missing times"))?;
            dict.set_item("aberrated_time_scale", aberrated_times.scale.as_str())?;
            dict.set_item(
                "aberrated_time_days",
                aberrated_times
                    .epochs
                    .iter()
                    .map(|epoch| epoch.days)
                    .collect::<Vec<_>>(),
            )?;
            dict.set_item(
                "aberrated_time_nanos",
                aberrated_times
                    .epochs
                    .iter()
                    .map(|epoch| epoch.nanos)
                    .collect::<Vec<_>>(),
            )?;
        }
        None => {
            dict.set_item("aberrated_states", py.None())?;
            dict.set_item("aberrated_origin_codes", py.None())?;
            dict.set_item("aberrated_time_scale", py.None())?;
            dict.set_item("aberrated_time_days", py.None())?;
            dict.set_item("aberrated_time_nanos", py.None())?;
        }
    }
    let validity = (0..ephemeris.validity.len())
        .map(|index| ephemeris.validity.is_valid(index))
        .collect::<Vec<_>>();
    dict.set_item("validity", validity)?;
    Ok(dict)
}

fn py_value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn py_runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeAssistPropagator>()?;
    Ok(())
}
