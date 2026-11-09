use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use adam_core_rs_coords::{
    EvaluationConfig, IodRunConfig, IodStageConfig, ObservationPipeline, ObservationTransform,
    OdConfig, OdConfigError, OdProfile, ProfileIdentity, RefinementRunConfig,
    ResidualCovarianceOperation, ResourceBindings, ResourceRequirement, ResourceSlot,
    ASTEROID_INSTITUTE_OD_V1, CORE_OD_V1, EFCC18_BIAS_RESOURCE, V2_FULL_RESOURCE,
    V2_SIGMA_FILL_RESOURCE,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::Value;

pyo3::create_exception!(_rust_native, OdConfigurationError, PyValueError);
pyo3::create_exception!(_rust_native, OdResourceValidationError, PyValueError);

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn od_config_error(error: OdConfigError) -> PyErr {
    match error {
        OdConfigError::Configuration { .. } => OdConfigurationError::new_err(error.to_string()),
        OdConfigError::Resources(_) => OdResourceValidationError::new_err(error.to_string()),
    }
}

fn profile(name: &str) -> PyResult<OdProfile> {
    match name {
        "CORE_OD_V1" => Ok(CORE_OD_V1),
        "ASTEROID_INSTITUTE_OD_V1" => Ok(ASTEROID_INSTITUTE_OD_V1),
        _ => Err(value_error(format!("unknown OD profile {name:?}"))),
    }
}

fn profile_identity_json(identity: Option<ProfileIdentity>) -> Option<String> {
    identity.map(|identity| {
        serde_json::to_string(&serde_json::json!({
            "name": identity.name,
            "version": identity.version,
            "definition_sha256": identity.definition_sha256,
        }))
        .expect("profile identity is serializable")
    })
}

fn bindings(paths: BTreeMap<String, String>) -> PyResult<ResourceBindings> {
    let mut result = ResourceBindings::new();
    for (slot, path) in paths {
        let slot = match slot.as_str() {
            "station_catalog_uncertainty_table" => ResourceSlot::StationCatalogUncertaintyTable,
            "efcc18_catalog_bias_table" => ResourceSlot::Efcc18CatalogBiasTable,
            "station_band_residual_covariance_table" => {
                ResourceSlot::StationBandResidualCovarianceTable
            }
            _ => return Err(value_error(format!("unknown OD resource slot {slot:?}"))),
        };
        result = result.bind(slot, PathBuf::from(path));
    }
    Ok(result)
}

#[pyclass(name = "OdConfig", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyOdConfig {
    pub(crate) inner: Arc<OdConfig>,
}

#[pymethods]
impl PyOdConfig {
    #[getter]
    fn profile_name(&self) -> Option<String> {
        self.inner
            .profile_identity()
            .map(|profile| profile.name.to_string())
    }

    #[getter]
    fn profile_definition_sha256(&self) -> Option<String> {
        self.inner
            .profile_identity()
            .map(|profile| profile.definition_sha256.to_string())
    }

    #[getter]
    fn profile_identity_json(&self) -> Option<String> {
        profile_identity_json(self.inner.profile_identity())
    }

    #[getter]
    fn provenance_json(&self) -> String {
        self.inner.provenance_json()
    }

    fn for_iod(&self) -> PyIodRunConfig {
        PyIodRunConfig {
            inner: self.inner.for_iod(),
        }
    }

    fn for_refinement(&self) -> PyRefinementRunConfig {
        PyRefinementRunConfig {
            inner: self.inner.for_refinement(),
        }
    }

    fn for_evaluation(&self) -> PyEvaluationConfig {
        PyEvaluationConfig {
            inner: self.inner.for_evaluation(),
        }
    }

    fn transform_kinds(&self) -> Vec<String> {
        self.inner
            .observation_pipeline()
            .transform_kinds()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn __repr__(&self) -> String {
        match self.profile_name() {
            Some(profile) => format!("OdConfig(profile={profile:?})"),
            None => "OdConfig(explicit=True)".to_string(),
        }
    }
}

#[pyclass(name = "IodStageConfig", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyIodStageConfig {
    pub(crate) inner: IodStageConfig,
}

#[pymethods]
impl PyIodStageConfig {
    #[staticmethod]
    fn core_v1() -> Self {
        Self {
            inner: IodStageConfig::core_v1(),
        }
    }

    #[getter]
    fn canonical_json(&self) -> String {
        self.inner.canonical_json()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(value_error)
    }
}

#[pyclass(name = "IodRunConfig", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyIodRunConfig {
    pub(crate) inner: IodRunConfig,
}

#[pymethods]
impl PyIodRunConfig {
    #[getter]
    fn profile_name(&self) -> Option<String> {
        self.inner
            .profile_identity()
            .map(|profile| profile.name.to_string())
    }

    #[getter]
    fn profile_identity_json(&self) -> Option<String> {
        profile_identity_json(self.inner.profile_identity())
    }

    #[getter]
    fn provenance_json(&self) -> String {
        self.inner.provenance_json()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(value_error)
    }
}

#[pyclass(name = "RefinementRunConfig", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyRefinementRunConfig {
    pub(crate) inner: RefinementRunConfig,
}

#[pymethods]
impl PyRefinementRunConfig {
    #[staticmethod]
    fn core_v1() -> Self {
        Self {
            inner: RefinementRunConfig::core_v1(),
        }
    }

    #[getter]
    fn canonical_json(&self) -> String {
        self.inner.canonical_json()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(value_error)
    }
}

#[pyclass(name = "EvaluationConfig", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyEvaluationConfig {
    pub(crate) inner: EvaluationConfig,
}

#[pymethods]
impl PyEvaluationConfig {
    #[staticmethod]
    fn core_v1() -> Self {
        Self {
            inner: EvaluationConfig::core_v1(),
        }
    }

    #[getter]
    fn covariance_dimension(&self) -> usize {
        self.inner.covariance_dimension()
    }

    #[getter]
    fn canonical_json(&self) -> String {
        self.inner.canonical_json()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(value_error)
    }
}

#[pyfunction]
fn od_profile_definition(profile_name: &str) -> PyResult<String> {
    let profile = profile(profile_name)?;
    let value = serde_json::json!({
        "identity": {
            "name": profile.identity().name,
            "version": profile.identity().version,
            "definition_sha256": profile.identity().definition_sha256,
        },
        "definition": profile.definition_value(),
    });
    serde_json::to_string(&value).map_err(value_error)
}

#[pyfunction]
fn od_profile_requirements(profile_name: &str) -> PyResult<String> {
    let profile = profile(profile_name)?;
    let requirements: Vec<Value> = profile
        .requirements()
        .iter()
        .map(|requirement| requirement.canonical_value())
        .collect();
    serde_json::to_string(&requirements).map_err(value_error)
}

#[pyfunction]
fn bind_od_profile(
    profile_name: &str,
    resource_paths: BTreeMap<String, String>,
) -> PyResult<PyOdConfig> {
    let profile = profile(profile_name)?;
    let config = profile
        .bind(&bindings(resource_paths)?)
        .map_err(od_config_error)?;
    Ok(PyOdConfig {
        inner: Arc::new(config),
    })
}

fn requirement_for_kind(kind: &str) -> PyResult<ResourceRequirement> {
    match kind {
        "fill_missing_astrometric_uncertainty"
        | "vfc2017_astrometric_uncertainty_floor"
        | "vfc2017_astrometric_uncertainty_replace" => Ok(V2_SIGMA_FILL_RESOURCE),
        "efcc18_catalog_bias_correction" => Ok(EFCC18_BIAS_RESOURCE),
        "station_band_residual_covariance_adjustment"
        | "performance_weighted_covariance_scaling"
        | "measured_bias_astrometric_uncertainty_floor" => Ok(V2_FULL_RESOURCE),
        _ => Err(value_error(format!(
            "transform {kind:?} does not have a resource"
        ))),
    }
}

fn require_exact_keys(value: &Value, expected: &[&str]) -> PyResult<()> {
    let object = value
        .as_object()
        .ok_or_else(|| PyTypeError::new_err("transform declaration must be an object"))?;
    let mut actual: Vec<&str> = object.keys().map(String::as_str).collect();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(value_error(format!(
            "transform fields must be exactly {expected:?}; got {actual:?}"
        )));
    }
    Ok(())
}

fn usize_field(value: &Value, name: &str) -> PyResult<usize> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| value_error(format!("{name} must be a positive integer")))
}

fn parse_definitions(raw: &str) -> PyResult<Vec<ObservationTransform>> {
    let values: Vec<Value> = serde_json::from_str(raw)
        .map_err(|error| value_error(format!("invalid transform JSON: {error}")))?;
    if values.is_empty() {
        return Err(value_error(
            "empty transforms are ambiguous; use ObservationPipeline.as_supplied()",
        ));
    }
    values
        .into_iter()
        .map(|value| {
            let kind = value
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| PyTypeError::new_err("transform kind must be a string"))?;
            match kind {
                "fill_missing_astrometric_uncertainty" => {
                    require_exact_keys(&value, &["kind"])?;
                    Ok(ObservationTransform::FillMissingAstrometricUncertainty {
                        resource: requirement_for_kind(kind)?,
                    })
                }
                "vfc2017_astrometric_uncertainty_floor" => {
                    require_exact_keys(&value, &["fill_missing", "kind"])?;
                    let fill_missing = value
                        .get("fill_missing")
                        .and_then(Value::as_bool)
                        .ok_or_else(|| value_error("fill_missing must be a boolean"))?;
                    Ok(ObservationTransform::Vfc2017AstrometricUncertaintyFloor {
                        resource: requirement_for_kind(kind)?,
                        fill_missing,
                    })
                }
                "vfc2017_astrometric_uncertainty_replace" => {
                    require_exact_keys(&value, &["kind"])?;
                    Ok(ObservationTransform::Vfc2017AstrometricUncertaintyReplace {
                        resource: requirement_for_kind(kind)?,
                    })
                }
                "efcc18_catalog_bias_correction" => {
                    require_exact_keys(&value, &["exclude_astcats", "kind"])?;
                    let exclude_astcats = value
                        .get("exclude_astcats")
                        .and_then(Value::as_array)
                        .map(|values| {
                            values
                                .iter()
                                .map(|value| {
                                    value.as_str().map(str::to_string).ok_or_else(|| {
                                        PyTypeError::new_err(
                                            "exclude_astcats items must be strings",
                                        )
                                    })
                                })
                                .collect::<PyResult<Vec<_>>>()
                        })
                        .transpose()?
                        .ok_or_else(|| value_error("exclude_astcats is required"))?;
                    Ok(ObservationTransform::Efcc18CatalogBiasCorrection {
                        resource: requirement_for_kind(kind)?,
                        exclude_astcats,
                    })
                }
                "station_band_residual_covariance_adjustment" => {
                    require_exact_keys(&value, &["kind", "minimum_residual_count", "operation"])?;
                    let operation = match value.get("operation").and_then(Value::as_str) {
                        Some("add") => ResidualCovarianceOperation::Add,
                        Some("replace") => ResidualCovarianceOperation::Replace,
                        _ => return Err(value_error("operation must be 'add' or 'replace'")),
                    };
                    Ok(
                        ObservationTransform::StationBandResidualCovarianceAdjustment {
                            resource: requirement_for_kind(kind)?,
                            operation,
                            minimum_residual_count: usize_field(&value, "minimum_residual_count")?,
                        },
                    )
                }
                "performance_weighted_covariance_scaling" => {
                    require_exact_keys(&value, &["kind", "minimum_residual_count"])?;
                    Ok(ObservationTransform::PerformanceWeightedCovarianceScaling {
                        resource: requirement_for_kind(kind)?,
                        minimum_residual_count: usize_field(&value, "minimum_residual_count")?,
                    })
                }
                "measured_bias_astrometric_uncertainty_floor" => {
                    require_exact_keys(&value, &["kind", "minimum_residual_count"])?;
                    Ok(
                        ObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
                            resource: requirement_for_kind(kind)?,
                            minimum_residual_count: usize_field(&value, "minimum_residual_count")?,
                        },
                    )
                }
                "nightly_batch_deweighting" => {
                    require_exact_keys(&value, &["effective_observation_cap", "kind"])?;
                    Ok(ObservationTransform::NightlyBatchDeweighting {
                        effective_observation_cap: usize_field(
                            &value,
                            "effective_observation_cap",
                        )?,
                    })
                }
                _ => Err(value_error(format!(
                    "unknown observation transform {kind:?}"
                ))),
            }
        })
        .collect()
}

fn explicit_pipeline(
    transforms_json: Option<&str>,
    resource_paths: BTreeMap<String, String>,
) -> PyResult<ObservationPipeline> {
    let resource_bindings = bindings(resource_paths)?;
    match transforms_json {
        None => {
            if resource_bindings.slots().next().is_some() {
                return Err(OdResourceValidationError::new_err(
                    "as-supplied pipeline does not use resource bindings",
                ));
            }
            Ok(ObservationPipeline::as_supplied())
        }
        Some(raw) => {
            ObservationPipeline::bind_explicit(parse_definitions(raw)?, &resource_bindings)
                .map_err(od_config_error)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (transforms_json, resource_paths, iod, refinement))]
fn explicit_od_config(
    transforms_json: Option<&str>,
    resource_paths: BTreeMap<String, String>,
    iod: PyRef<'_, PyIodStageConfig>,
    refinement: PyRef<'_, PyRefinementRunConfig>,
) -> PyResult<PyOdConfig> {
    let config = OdConfig::explicit(
        explicit_pipeline(transforms_json, resource_paths)?,
        iod.inner.clone(),
        refinement.inner,
    )
    .map_err(od_config_error)?;
    Ok(PyOdConfig {
        inner: Arc::new(config),
    })
}

#[pyfunction]
#[pyo3(signature = (transforms_json, resource_paths, iod))]
fn explicit_iod_run_config(
    transforms_json: Option<&str>,
    resource_paths: BTreeMap<String, String>,
    iod: PyRef<'_, PyIodStageConfig>,
) -> PyResult<PyIodRunConfig> {
    let config = IodRunConfig::explicit(
        explicit_pipeline(transforms_json, resource_paths)?,
        iod.inner.clone(),
    )
    .map_err(od_config_error)?;
    Ok(PyIodRunConfig { inner: config })
}

#[pyfunction]
fn iod_stage_config_from_json(raw: &str) -> PyResult<PyIodStageConfig> {
    Ok(PyIodStageConfig {
        inner: IodStageConfig::from_json(raw).map_err(OdConfigurationError::new_err)?,
    })
}

#[pyfunction]
fn refinement_run_config_from_json(raw: &str) -> PyResult<PyRefinementRunConfig> {
    Ok(PyRefinementRunConfig {
        inner: RefinementRunConfig::from_json(raw).map_err(OdConfigurationError::new_err)?,
    })
}

#[pyfunction]
fn evaluation_config_from_json(raw: &str) -> PyResult<PyEvaluationConfig> {
    Ok(PyEvaluationConfig {
        inner: EvaluationConfig::from_json(raw).map_err(OdConfigurationError::new_err)?,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOdConfig>()?;
    m.add_class::<PyIodStageConfig>()?;
    m.add_class::<PyIodRunConfig>()?;
    m.add_class::<PyRefinementRunConfig>()?;
    m.add_class::<PyEvaluationConfig>()?;
    m.add_function(wrap_pyfunction!(od_profile_definition, m)?)?;
    m.add_function(wrap_pyfunction!(od_profile_requirements, m)?)?;
    m.add_function(wrap_pyfunction!(bind_od_profile, m)?)?;
    m.add_function(wrap_pyfunction!(explicit_od_config, m)?)?;
    m.add_function(wrap_pyfunction!(explicit_iod_run_config, m)?)?;
    m.add_function(wrap_pyfunction!(iod_stage_config_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(refinement_run_config_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation_config_from_json, m)?)?;
    m.add(
        "OdConfigurationError",
        m.py().get_type::<OdConfigurationError>(),
    )?;
    m.add(
        "OdResourceValidationError",
        m.py().get_type::<OdResourceValidationError>(),
    )?;
    Ok(())
}
