use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use adam_core_rs_coords::{
    EvaluationConfig, IodRunConfig, ObservationPipeline, ObservationTransform, OdConfig, OdProfile,
    RefinementRunConfig, ResidualCovarianceOperation, ResourceBindings, ResourceRequirement,
    ResourceSlot, ASTEROID_INSTITUTE_OD_V1, CORE_OD_V1, EFCC18_BIAS_RESOURCE, V2_FULL_RESOURCE,
    V2_SIGMA_FILL_RESOURCE,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::Value;

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn profile(name: &str) -> PyResult<OdProfile> {
    match name {
        "CORE_OD_V1" => Ok(CORE_OD_V1),
        "ASTEROID_INSTITUTE_OD_V1" => Ok(ASTEROID_INSTITUTE_OD_V1),
        _ => Err(value_error(format!("unknown OD profile {name:?}"))),
    }
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

#[pyclass(name = "OdConfig", frozen)]
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

#[pyclass(name = "IodRunConfig", frozen)]
#[derive(Clone)]
pub(crate) struct PyIodRunConfig {
    pub(crate) inner: IodRunConfig,
}

#[pymethods]
impl PyIodRunConfig {
    #[staticmethod]
    fn core_v1() -> Self {
        Self {
            inner: IodRunConfig::core_v1(),
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(value_error)
    }
}

#[pyclass(name = "RefinementRunConfig", frozen)]
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

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(value_error)
    }
}

#[pyclass(name = "EvaluationConfig", frozen)]
#[derive(Clone, Copy)]
pub(crate) struct PyEvaluationConfig {
    inner: EvaluationConfig,
}

#[pymethods]
impl PyEvaluationConfig {
    #[staticmethod]
    fn six_dimensional() -> Self {
        Self {
            inner: EvaluationConfig::six_dimensional(),
        }
    }

    #[getter]
    fn covariance_dimension(&self) -> usize {
        self.inner.covariance_dimension()
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
        .map_err(value_error)?;
    Ok(PyOdConfig {
        inner: Arc::new(config),
    })
}

fn requirement_for_kind(kind: &str) -> PyResult<ResourceRequirement> {
    match kind {
        "fill_missing_astrometric_uncertainty" => Ok(V2_SIGMA_FILL_RESOURCE),
        "efcc18_catalog_bias_correction" => Ok(EFCC18_BIAS_RESOURCE),
        "station_band_residual_covariance_adjustment" => Ok(V2_FULL_RESOURCE),
        _ => Err(value_error(format!(
            "transform {kind:?} does not have a resource"
        ))),
    }
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
                    Ok(ObservationTransform::FillMissingAstrometricUncertainty {
                        resource: requirement_for_kind(kind)?,
                    })
                }
                "efcc18_catalog_bias_correction" => {
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
                        .unwrap_or_default();
                    Ok(ObservationTransform::Efcc18CatalogBiasCorrection {
                        resource: requirement_for_kind(kind)?,
                        exclude_astcats,
                    })
                }
                "station_band_residual_covariance_adjustment" => {
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
                "nightly_batch_deweighting" => Ok(ObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap: usize_field(&value, "effective_observation_cap")?,
                }),
                _ => Err(value_error(format!(
                    "unknown observation transform {kind:?}"
                ))),
            }
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (transforms_json, resource_paths, iod, refinement, evaluation))]
fn explicit_od_config(
    transforms_json: Option<&str>,
    resource_paths: BTreeMap<String, String>,
    iod: PyRef<'_, PyIodRunConfig>,
    refinement: PyRef<'_, PyRefinementRunConfig>,
    evaluation: PyRef<'_, PyEvaluationConfig>,
) -> PyResult<PyOdConfig> {
    let pipeline = match transforms_json {
        None => ObservationPipeline::as_supplied(),
        Some(raw) => {
            ObservationPipeline::bind_explicit(parse_definitions(raw)?, &bindings(resource_paths)?)
                .map_err(value_error)?
        }
    };
    let config = OdConfig::explicit(
        pipeline,
        iod.inner.clone(),
        refinement.inner,
        evaluation.inner,
    )
    .map_err(value_error)?;
    Ok(PyOdConfig {
        inner: Arc::new(config),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOdConfig>()?;
    m.add_class::<PyIodRunConfig>()?;
    m.add_class::<PyRefinementRunConfig>()?;
    m.add_class::<PyEvaluationConfig>()?;
    m.add_function(wrap_pyfunction!(od_profile_definition, m)?)?;
    m.add_function(wrap_pyfunction!(od_profile_requirements, m)?)?;
    m.add_function(wrap_pyfunction!(bind_od_profile, m)?)?;
    m.add_function(wrap_pyfunction!(explicit_od_config, m)?)?;
    Ok(())
}
