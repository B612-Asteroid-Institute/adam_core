//! Complete, explicit orbit-determination configuration and profiles.
//!
//! This module is the authoritative Core 0.7 configuration boundary.  Python
//! bindings may locate files and marshal paths, but profile expansion,
//! validation, parsing, transform execution and provenance identity live here.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Array, Float64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::efcc18::Efcc18BiasTable;
use crate::observation_uncertainty::{
    BiasTable, BiasTableRow, Efcc18DebiasModel, EmpiricalCovarianceMode, EmpiricalCovarianceModel,
    NightBatchDeweightingModel, ObservationUncertaintyModel, OrbitDeterminationAstrometry,
};
use crate::propagation::IodConfig;
use crate::veres2017::{SigmaFillModel, VeresSigmaLookup, VeresSigmaRow};
use crate::{Cmc2003FitConfig, RefinementConfig, WhitenedFitConfig};

/// Stable semantic slots used by profiles.  Paths are deployment details and
/// are deliberately absent from scientific identity and provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceSlot {
    StationCatalogUncertaintyTable,
    Efcc18CatalogBiasTable,
    StationBandResidualCovarianceTable,
}

impl ResourceSlot {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StationCatalogUncertaintyTable => "station_catalog_uncertainty_table",
            Self::Efcc18CatalogBiasTable => "efcc18_catalog_bias_table",
            Self::StationBandResidualCovarianceTable => "station_band_residual_covariance_table",
        }
    }
}

/// Path-independent identity of one required scientific resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceRequirement {
    pub slot: ResourceSlot,
    pub product: &'static str,
    pub product_version: &'static str,
    pub schema_version: &'static str,
    pub sha256: &'static str,
    pub byte_size: u64,
    pub row_count: Option<usize>,
    pub catalog_date: Option<&'static str>,
    pub format: &'static str,
}

impl ResourceRequirement {
    pub fn validate(self) -> Result<(), String> {
        for (name, value) in [
            ("product", self.product),
            ("product_version", self.product_version),
            ("schema_version", self.schema_version),
            ("format", self.format),
        ] {
            if value.is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        if self.byte_size == 0 {
            return Err("byte_size must be positive".to_string());
        }
        if self.row_count == Some(0) {
            return Err("row_count must be positive when present".to_string());
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
        {
            return Err("sha256 must be exactly 64 lowercase hexadecimal characters".to_string());
        }
        let (schema, format) = match self.slot {
            ResourceSlot::StationCatalogUncertaintyTable => (
                "adam-observatory-sigma-table/v1",
                "application/vnd.apache.parquet",
            ),
            ResourceSlot::Efcc18CatalogBiasTable => ("jpl-efcc18-bias-dat/v1", "text/plain"),
            ResourceSlot::StationBandResidualCovarianceTable => (
                "adam-observatory-bias-table/v1",
                "application/vnd.apache.parquet",
            ),
        };
        if self.schema_version != schema {
            return Err(format!(
                "schema_version for {} must be {schema:?}",
                self.slot.as_str()
            ));
        }
        if self.format != format {
            return Err(format!(
                "format for {} must be {format:?}",
                self.slot.as_str()
            ));
        }
        Ok(())
    }

    pub fn canonical_value(self) -> Value {
        json!({
            "slot": self.slot.as_str(),
            "product": self.product,
            "product_version": self.product_version,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "row_count": self.row_count,
            "catalog_date": self.catalog_date,
            "format": self.format,
        })
    }
}

pub const EFCC18_BIAS_RESOURCE: ResourceRequirement = ResourceRequirement {
    slot: ResourceSlot::Efcc18CatalogBiasTable,
    product: "jpl-debias-2018-bias.dat",
    product_version: "2018",
    schema_version: "jpl-efcc18-bias-dat/v1",
    sha256: "ef6a50830bb83d8b1161e7acb391eed988ea9708e9fbcccc0c834da6109476e3",
    byte_size: 37_161_151,
    row_count: Some(49_152),
    catalog_date: None,
    format: "text/plain",
};

pub const V2_SIGMA_FILL_RESOURCE: ResourceRequirement = ResourceRequirement {
    slot: ResourceSlot::StationCatalogUncertaintyTable,
    product: "v2_sigma_fill",
    product_version: "v2_sigma_fill",
    schema_version: "adam-observatory-sigma-table/v1",
    sha256: "333855b153da9bc5f53f172cf691df0d85e7cb289942708a34f9b2d855a3a6f0",
    byte_size: 51_949,
    row_count: Some(2_474),
    catalog_date: None,
    format: "application/vnd.apache.parquet",
};

pub const V2_FULL_RESOURCE: ResourceRequirement = ResourceRequirement {
    slot: ResourceSlot::StationBandResidualCovarianceTable,
    product: "v2_full",
    product_version: "v2_full",
    schema_version: "adam-observatory-bias-table/v1",
    sha256: "96846a0e9717e0328ad5830fbab3012eab00546dccd1e5587a95c5cc6c712356",
    byte_size: 158_614,
    row_count: Some(1_123),
    catalog_date: Some("2026-06-23"),
    format: "application/vnd.apache.parquet",
};

/// Caller-owned concrete paths for semantic resource slots.
#[derive(Debug, Clone, Default)]
pub struct ResourceBindings {
    paths: BTreeMap<ResourceSlot, PathBuf>,
}

impl ResourceBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(mut self, slot: ResourceSlot, path: impl AsRef<Path>) -> Self {
        self.paths.insert(slot, path.as_ref().to_path_buf());
        self
    }

    pub fn get(&self, slot: ResourceSlot) -> Option<&Path> {
        self.paths.get(&slot).map(PathBuf::as_path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceProblem {
    pub slot: ResourceSlot,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceValidationError {
    pub problems: Vec<ResourceProblem>,
}

impl fmt::Display for ResourceValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OD resource validation failed:")?;
        for problem in &self.problems {
            writeln!(f, "  {}: {}", problem.slot.as_str(), problem.reason)?;
        }
        Ok(())
    }
}

impl std::error::Error for ResourceValidationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedResourceIdentity {
    pub requirement: ResourceRequirement,
    pub actual_sha256: String,
    pub actual_byte_size: u64,
}

impl ValidatedResourceIdentity {
    pub fn canonical_value(&self) -> Value {
        json!({
            "requirement": self.requirement.canonical_value(),
            "actual_sha256": self.actual_sha256,
            "actual_byte_size": self.actual_byte_size,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualCovarianceOperation {
    Add,
    Replace,
}

impl ResidualCovarianceOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Replace => "replace",
        }
    }
}

/// Path-free declarative transform definition. Resource paths are bound
/// separately, so the same definition has identical scientific identity in
/// Python and standalone Rust.
#[derive(Debug, Clone)]
pub enum ObservationTransform {
    FillMissingAstrometricUncertainty {
        resource: ResourceRequirement,
    },
    Efcc18CatalogBiasCorrection {
        resource: ResourceRequirement,
        exclude_astcats: Vec<String>,
    },
    StationBandResidualCovarianceAdjustment {
        resource: ResourceRequirement,
        operation: ResidualCovarianceOperation,
        minimum_residual_count: usize,
    },
    NightlyBatchDeweighting {
        effective_observation_cap: usize,
    },
}

impl ObservationTransform {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::FillMissingAstrometricUncertainty { resource } => {
                resource.validate()?;
                if resource.slot != ResourceSlot::StationCatalogUncertaintyTable {
                    return Err("FillMissingAstrometricUncertainty requires station_catalog_uncertainty_table".to_string());
                }
            }
            Self::Efcc18CatalogBiasCorrection { resource, .. } => {
                resource.validate()?;
                if resource.slot != ResourceSlot::Efcc18CatalogBiasTable {
                    return Err(
                        "Efcc18CatalogBiasCorrection requires efcc18_catalog_bias_table"
                            .to_string(),
                    );
                }
            }
            Self::StationBandResidualCovarianceAdjustment {
                resource,
                minimum_residual_count,
                ..
            } => {
                resource.validate()?;
                if resource.slot != ResourceSlot::StationBandResidualCovarianceTable {
                    return Err("StationBandResidualCovarianceAdjustment requires station_band_residual_covariance_table".to_string());
                }
                if *minimum_residual_count == 0 {
                    return Err("minimum_residual_count must be positive".to_string());
                }
            }
            Self::NightlyBatchDeweighting {
                effective_observation_cap,
            } => {
                if *effective_observation_cap == 0 {
                    return Err("effective_observation_cap must be positive".to_string());
                }
            }
        }
        Ok(())
    }
}

/// Validated, parsed transform state. It is never constructible without binding.
#[derive(Debug, Clone)]
enum BoundObservationTransform {
    FillMissingAstrometricUncertainty {
        lookup: VeresSigmaLookup,
        resource: ValidatedResourceIdentity,
    },
    Efcc18CatalogBiasCorrection {
        bias_table: Arc<Efcc18BiasTable>,
        exclude_astcats: Vec<String>,
        resource: ValidatedResourceIdentity,
    },
    StationBandResidualCovarianceAdjustment {
        table: BiasTable,
        operation: ResidualCovarianceOperation,
        minimum_residual_count: usize,
        resource: ValidatedResourceIdentity,
    },
    NightlyBatchDeweighting {
        effective_observation_cap: usize,
    },
}

impl BoundObservationTransform {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::StationBandResidualCovarianceAdjustment {
                minimum_residual_count,
                ..
            } if *minimum_residual_count == 0 => {
                Err("minimum_residual_count must be positive".to_string())
            }
            Self::NightlyBatchDeweighting {
                effective_observation_cap,
            } if *effective_observation_cap == 0 => {
                Err("effective_observation_cap must be positive".to_string())
            }
            _ => Ok(()),
        }
    }

    pub const fn kind(&self) -> &'static str {
        match self {
            Self::FillMissingAstrometricUncertainty { .. } => {
                "fill_missing_astrometric_uncertainty"
            }
            Self::Efcc18CatalogBiasCorrection { .. } => "efcc18_catalog_bias_correction",
            Self::StationBandResidualCovarianceAdjustment { .. } => {
                "station_band_residual_covariance_adjustment"
            }
            Self::NightlyBatchDeweighting { .. } => "nightly_batch_deweighting",
        }
    }

    #[doc(hidden)]
    pub fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        self.validate()?;
        match self {
            Self::FillMissingAstrometricUncertainty { lookup, .. } => SigmaFillModel {
                lookup: lookup.clone(),
            }
            .apply(observations),
            Self::Efcc18CatalogBiasCorrection {
                bias_table,
                exclude_astcats,
                ..
            } => Efcc18DebiasModel {
                bias_table: Arc::clone(bias_table),
                exclude_astcats: exclude_astcats.clone(),
            }
            .apply(observations),
            Self::StationBandResidualCovarianceAdjustment {
                table,
                operation,
                minimum_residual_count,
                ..
            } => EmpiricalCovarianceModel {
                table: table.clone(),
                mode: match operation {
                    ResidualCovarianceOperation::Add => EmpiricalCovarianceMode::Add,
                    ResidualCovarianceOperation::Replace => EmpiricalCovarianceMode::Replace,
                },
                min_resid_cov_n: *minimum_residual_count as f64,
            }
            .apply(observations),
            Self::NightlyBatchDeweighting {
                effective_observation_cap,
            } => NightBatchDeweightingModel::new(*effective_observation_cap)?.apply(observations),
        }
    }

    pub fn provenance_value(&self) -> Value {
        match self {
            Self::FillMissingAstrometricUncertainty { resource, .. } => json!({
                "kind": self.kind(), "resource": resource.canonical_value(),
            }),
            Self::Efcc18CatalogBiasCorrection {
                exclude_astcats,
                resource,
                ..
            } => json!({
                "kind": self.kind(), "exclude_astcats": exclude_astcats,
                "resource": resource.canonical_value(),
            }),
            Self::StationBandResidualCovarianceAdjustment {
                operation,
                minimum_residual_count,
                resource,
                ..
            } => json!({
                "kind": self.kind(), "operation": operation.as_str(),
                "minimum_residual_count": minimum_residual_count,
                "resource": resource.canonical_value(),
            }),
            Self::NightlyBatchDeweighting {
                effective_observation_cap,
            } => json!({
                "kind": self.kind(),
                "effective_observation_cap": effective_observation_cap,
            }),
        }
    }
}

/// An explicitly ordered, already-bound pipeline. The private representation
/// makes resource validation and parsing impossible to bypass. There is
/// deliberately no `Default` implementation.
#[derive(Debug, Clone)]
pub struct ObservationPipeline {
    transforms: Vec<BoundObservationTransform>,
}

impl ObservationPipeline {
    pub const fn as_supplied() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    fn ordered(transforms: Vec<BoundObservationTransform>) -> Result<Self, String> {
        if transforms.is_empty() {
            return Err(
                "an empty transform list is ambiguous; use ObservationPipeline::as_supplied()"
                    .to_string(),
            );
        }
        for transform in &transforms {
            transform.validate()?;
        }
        Ok(Self { transforms })
    }

    fn bound_transforms(&self) -> &[BoundObservationTransform] {
        &self.transforms
    }

    pub fn transform_kinds(&self) -> Vec<&'static str> {
        self.transforms
            .iter()
            .map(BoundObservationTransform::kind)
            .collect()
    }

    pub const fn is_as_supplied(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Execute exactly once in declaration order. Repeated variants are
    /// retained and executed; no sorting or deduplication occurs.
    pub fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        let mut changed = false;
        for transform in &self.transforms {
            changed |= transform.apply(observations)?;
        }
        Ok(changed)
    }

    #[doc(hidden)]
    pub fn native_models(&self) -> Result<Vec<Box<dyn ObservationUncertaintyModel>>, String> {
        self.transforms
            .iter()
            .map(
                |transform| -> Result<Box<dyn ObservationUncertaintyModel>, String> {
                    let model: Box<dyn ObservationUncertaintyModel> = match transform {
                        BoundObservationTransform::FillMissingAstrometricUncertainty {
                            lookup,
                            ..
                        } => Box::new(SigmaFillModel {
                            lookup: lookup.clone(),
                        }),
                        BoundObservationTransform::Efcc18CatalogBiasCorrection {
                            bias_table,
                            exclude_astcats,
                            ..
                        } => Box::new(Efcc18DebiasModel {
                            bias_table: Arc::clone(bias_table),
                            exclude_astcats: exclude_astcats.clone(),
                        }),
                        BoundObservationTransform::StationBandResidualCovarianceAdjustment {
                            table,
                            operation,
                            minimum_residual_count,
                            ..
                        } => Box::new(EmpiricalCovarianceModel {
                            table: table.clone(),
                            mode: match operation {
                                ResidualCovarianceOperation::Add => EmpiricalCovarianceMode::Add,
                                ResidualCovarianceOperation::Replace => {
                                    EmpiricalCovarianceMode::Replace
                                }
                            },
                            min_resid_cov_n: *minimum_residual_count as f64,
                        }),
                        BoundObservationTransform::NightlyBatchDeweighting {
                            effective_observation_cap,
                        } => Box::new(NightBatchDeweightingModel::new(*effective_observation_cap)?),
                    };
                    Ok(model)
                },
            )
            .collect()
    }

    pub fn provenance_value(&self) -> Value {
        Value::Array(
            self.transforms
                .iter()
                .map(BoundObservationTransform::provenance_value)
                .collect(),
        )
    }
}

impl ObservationPipeline {
    /// Bind an explicit path-free definition through the same validation and
    /// parsing path used by named profiles.
    pub fn bind_explicit(
        definitions: Vec<ObservationTransform>,
        bindings: &ResourceBindings,
    ) -> Result<Self, ResourceValidationError> {
        if definitions.is_empty() {
            return Err(ResourceValidationError {
                problems: vec![ResourceProblem {
                    slot: ResourceSlot::Efcc18CatalogBiasTable,
                    reason: "an empty transform list is ambiguous; use as_supplied".to_string(),
                }],
            });
        }
        for definition in &definitions {
            definition
                .validate()
                .map_err(|reason| ResourceValidationError {
                    problems: vec![ResourceProblem {
                        slot: ResourceSlot::Efcc18CatalogBiasTable,
                        reason: format!("invalid observation pipeline: {reason}"),
                    }],
                })?;
        }
        let requirements: Vec<_> = definitions
            .iter()
            .filter_map(|definition| match definition {
                ObservationTransform::FillMissingAstrometricUncertainty { resource }
                | ObservationTransform::Efcc18CatalogBiasCorrection { resource, .. }
                | ObservationTransform::StationBandResidualCovarianceAdjustment {
                    resource, ..
                } => Some(*resource),
                ObservationTransform::NightlyBatchDeweighting { .. } => None,
            })
            .collect();
        let validated = validate_resources(&requirements, bindings)?;
        let mut transforms = Vec::with_capacity(definitions.len());
        for definition in definitions {
            let transform = match definition {
                ObservationTransform::FillMissingAstrometricUncertainty { resource } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    let lookup = parse_sigma_parquet(&bound.bytes, resource)
                        .map_err(|error| one_resource_error(resource.slot, error))?;
                    BoundObservationTransform::FillMissingAstrometricUncertainty {
                        lookup,
                        resource: bound.identity.clone(),
                    }
                }
                ObservationTransform::Efcc18CatalogBiasCorrection {
                    resource,
                    exclude_astcats,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    let text = std::str::from_utf8(&bound.bytes).map_err(|error| {
                        one_resource_error(resource.slot, format!("resource is not UTF-8: {error}"))
                    })?;
                    let table = Efcc18BiasTable::parse_bias_dat(text)
                        .map_err(|error| one_resource_error(resource.slot, error))?;
                    BoundObservationTransform::Efcc18CatalogBiasCorrection {
                        bias_table: Arc::new(table),
                        exclude_astcats,
                        resource: bound.identity.clone(),
                    }
                }
                ObservationTransform::StationBandResidualCovarianceAdjustment {
                    resource,
                    operation,
                    minimum_residual_count,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    let table = parse_bias_parquet(&bound.bytes, resource)
                        .map_err(|error| one_resource_error(resource.slot, error))?;
                    BoundObservationTransform::StationBandResidualCovarianceAdjustment {
                        table,
                        operation,
                        minimum_residual_count,
                        resource: bound.identity.clone(),
                    }
                }
                ObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap,
                } => BoundObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap,
                },
            };
            transform
                .validate()
                .map_err(|reason| ResourceValidationError {
                    problems: vec![ResourceProblem {
                        slot: ResourceSlot::Efcc18CatalogBiasTable,
                        reason,
                    }],
                })?;
            transforms.push(transform);
        }
        ObservationPipeline::ordered(transforms).map_err(|reason| ResourceValidationError {
            problems: vec![ResourceProblem {
                slot: ResourceSlot::Efcc18CatalogBiasTable,
                reason,
            }],
        })
    }
}

#[derive(Debug, Clone)]
pub struct IodRunConfig {
    iod: IodConfig,
    correction_fit: WhitenedFitConfig,
}

fn fit_config_value(config: &WhitenedFitConfig) -> Value {
    json!({
        "loss": config.loss.as_str(),
        "f_scale": config.f_scale,
        "jacobian": config.jacobian.as_str(),
        "validate_covariance": config.validate_covariance,
        "xtol": config.xtol,
        "ftol": config.ftol,
        "gtol": config.gtol,
        "max_iterations": config.max_iterations,
        "max_nfev": config.max_nfev,
        "two_body": {
            "propagation_max_iter": config.two_body.propagation_max_iter,
            "propagation_tol": config.two_body.propagation_tol,
            "lt_tol": config.two_body.lt_tol,
            "ephemeris_max_iter": config.two_body.ephemeris_max_iter,
            "ephemeris_tol": config.two_body.ephemeris_tol,
            "max_lt_iter": config.two_body.max_lt_iter,
        },
    })
}

fn iod_config_value(config: &IodConfig) -> Value {
    json!({
        "min_obs": config.min_obs,
        "min_arc_length": config.min_arc_length,
        "contamination_percentage": config.contamination_percentage,
        "rchi2_threshold": config.rchi2_threshold,
        "observation_selection_method": config.observation_selection_method.as_str(),
        "light_time": config.light_time,
        "mu": config.mu,
        "speed_of_light": config.speed_of_light,
        "max_triplets": config.max_triplets,
        "max_pair_visits": config.max_pair_visits,
        "max_raw_pair_visits": config.max_raw_pair_visits,
        "max_middle_visits": config.max_middle_visits,
        "velocity_method": config.velocity_method.as_str(),
        "spans": {
            "target_span_days": config.spans.target_span_days,
            "min_span_days": config.spans.min_span_days,
            "max_span_days": config.spans.max_span_days,
            "dense_window_days": config.spans.dense_window_days,
            "edge_window_days": config.spans.edge_window_days,
            "max_triplets": config.spans.max_triplets,
        },
        "acceptance": config.acceptance.as_str(),
        "correction": {
            "max_corrections": config.correction.max_corrections,
            "max_converged": config.correction.max_converged,
            "growth": config.correction.growth,
            "min_stage_obs": config.correction.min_stage_obs,
            "max_stages": config.correction.max_stages,
            "stage_loss": config.correction.stage_loss.as_str(),
            "prefit_scope": config.correction.prefit_scope.as_str(),
        },
        "ranging": {
            "enabled": config.ranging.enabled,
            "rho_min_au": config.ranging.rho_min_au,
            "rho_max_au": config.ranging.rho_max_au,
            "n_rho": config.ranging.n_rho,
            "n_rho_dot": config.ranging.n_rho_dot,
            "max_corrections": config.ranging.max_corrections,
            "chunk_size": config.ranging.chunk_size,
            "max_arc_days": config.ranging.max_arc_days,
        },
    })
}

fn refinement_config_value(config: &RefinementConfig) -> Value {
    match config {
        RefinementConfig::WorstResidual(loop_config) => json!({
            "method": "worst_residual",
            "rchi2_threshold": loop_config.rchi2_threshold,
            "min_obs": loop_config.min_obs,
            "min_arc_length": loop_config.min_arc_length,
            "contamination_percentage": loop_config.contamination_percentage,
            "fit": fit_config_value(&loop_config.fit),
        }),
        RefinementConfig::Cmc2003(loop_config) => json!({
            "method": "cmc2003",
            "chi2_reject": loop_config.chi2_reject,
            "chi2_recover": loop_config.chi2_recover,
            "chi2_frac": loop_config.chi2_frac,
            "max_iterations": loop_config.max_iterations,
            "max_rejected_fraction": loop_config.max_rejected_fraction,
            "apparition_gap_days": loop_config.apparition_gap_days,
            "psd_floor_frac": loop_config.psd_floor_frac,
            "min_kept": loop_config.min_kept,
            "fit": fit_config_value(&loop_config.fit),
        }),
    }
}

impl IodRunConfig {
    pub fn new(iod: IodConfig, correction_fit: WhitenedFitConfig) -> Result<Self, String> {
        let config = Self {
            iod,
            correction_fit,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn core_v1() -> Self {
        Self::new(IodConfig::default(), WhitenedFitConfig::default())
            .expect("Core-v1 IOD settings are valid")
    }

    pub const fn iod(&self) -> &IodConfig {
        &self.iod
    }

    pub const fn correction_fit(&self) -> &WhitenedFitConfig {
        &self.correction_fit
    }

    pub fn canonical_value(&self) -> Value {
        json!({
            "search_correction_ranging": iod_config_value(&self.iod),
            "candidate_correction_fit": fit_config_value(&self.correction_fit),
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        self.iod.validate()?;
        self.correction_fit.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RefinementRunConfig {
    refinement: RefinementConfig,
}

impl RefinementRunConfig {
    pub fn new(refinement: RefinementConfig) -> Result<Self, String> {
        let config = Self { refinement };
        config.validate()?;
        Ok(config)
    }

    pub fn core_v1() -> Self {
        Self::new(RefinementConfig::Cmc2003(Cmc2003FitConfig::default()))
            .expect("Core-v1 refinement settings are valid")
    }

    pub const fn refinement(&self) -> &RefinementConfig {
        &self.refinement
    }

    pub fn canonical_value(&self) -> Value {
        refinement_config_value(&self.refinement)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.refinement.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationConfig {
    covariance_dimension: usize,
}

impl EvaluationConfig {
    pub fn new(covariance_dimension: usize) -> Result<Self, String> {
        let config = Self {
            covariance_dimension,
        };
        config.validate()?;
        Ok(config)
    }

    pub const fn six_dimensional() -> Self {
        Self {
            covariance_dimension: 6,
        }
    }

    pub const fn covariance_dimension(self) -> usize {
        self.covariance_dimension
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.covariance_dimension != 6 {
            return Err(format!(
                "complete OD evaluation requires covariance_dimension=6, got {}",
                self.covariance_dimension
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileIdentity {
    pub name: &'static str,
    pub version: u32,
    pub definition_sha256: &'static str,
}

/// Complete Core-owned OD configuration. There is deliberately no `Default`.
#[derive(Debug, Clone)]
pub struct OdConfig {
    observation_pipeline: ObservationPipeline,
    iod: IodRunConfig,
    refinement: RefinementRunConfig,
    evaluation: EvaluationConfig,
    profile: Option<ProfileIdentity>,
}

impl OdConfig {
    pub fn explicit(
        observation_pipeline: ObservationPipeline,
        iod: IodRunConfig,
        refinement: RefinementRunConfig,
        evaluation: EvaluationConfig,
    ) -> Result<Self, String> {
        let config = Self {
            observation_pipeline,
            iod,
            refinement,
            evaluation,
            profile: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.iod.validate()?;
        self.refinement.validate()?;
        self.evaluation.validate()?;
        for transform in self.observation_pipeline.bound_transforms() {
            transform.validate()?;
        }
        Ok(())
    }

    pub const fn observation_pipeline(&self) -> &ObservationPipeline {
        &self.observation_pipeline
    }

    pub const fn profile_identity(&self) -> Option<ProfileIdentity> {
        self.profile
    }

    pub fn for_iod(&self) -> IodRunConfig {
        self.iod.clone()
    }

    pub const fn for_refinement(&self) -> RefinementRunConfig {
        self.refinement
    }

    pub const fn for_evaluation(&self) -> EvaluationConfig {
        self.evaluation
    }

    pub fn provenance_json(&self) -> String {
        let value = json!({
            "schema": "adam-core-od-config/v1",
            "profile": self.profile.map(|profile| json!({
                "name": profile.name,
                "version": profile.version,
                "definition_sha256": profile.definition_sha256,
            })),
            "transforms": self.observation_pipeline.provenance_value(),
            "iod": self.iod.canonical_value(),
            "refinement": self.refinement.canonical_value(),
            "evaluation": {"covariance_dimension": self.evaluation.covariance_dimension},
        });
        serde_json::to_string(&value).expect("OD configuration provenance is serializable")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdProfileKind {
    Core,
    AsteroidInstitute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OdProfile {
    identity: ProfileIdentity,
    kind: OdProfileKind,
}

impl OdProfile {
    pub const fn identity(self) -> ProfileIdentity {
        self.identity
    }

    pub const fn kind(self) -> OdProfileKind {
        self.kind
    }
}

/// Public path-free profile. Selection is always explicit; it is not a
/// function-signature default.
pub const CORE_OD_V1: OdProfile = OdProfile {
    identity: ProfileIdentity {
        name: "CORE_OD_V1",
        version: 1,
        definition_sha256: "b56612e00fb2b8a929dc075e330675d80a53d39916453b3acd3e9080dd058d4d",
    },
    kind: OdProfileKind::Core,
};

/// Institute reference profile. Core documentation exposes only semantic
/// slots; deployment/package details remain outside this crate.
pub const ASTEROID_INSTITUTE_OD_V1: OdProfile = OdProfile {
    identity: ProfileIdentity {
        name: "ASTEROID_INSTITUTE_OD_V1",
        version: 1,
        definition_sha256: "9f17db56d1322ce7d779d81a54807970fad69629941fe77e403714024f0221fe",
    },
    kind: OdProfileKind::AsteroidInstitute,
};

impl OdProfile {
    pub fn definition_value(self) -> Value {
        let pipeline = match self.kind {
            OdProfileKind::Core => vec![
                json!({"kind": "efcc18_catalog_bias_correction", "exclude_astcats": Vec::<String>::new()}),
                json!({"kind": "nightly_batch_deweighting", "effective_observation_cap": 4}),
            ],
            OdProfileKind::AsteroidInstitute => vec![
                json!({"kind": "fill_missing_astrometric_uncertainty"}),
                json!({"kind": "efcc18_catalog_bias_correction", "exclude_astcats": Vec::<String>::new()}),
                json!({"kind": "station_band_residual_covariance_adjustment", "operation": "add", "minimum_residual_count": 30}),
                json!({"kind": "nightly_batch_deweighting", "effective_observation_cap": 4}),
            ],
        };
        json!({
            "schema": "adam-core-od-profile/v1",
            "name": self.identity.name,
            "version": self.identity.version,
            "pipeline": pipeline,
            "requirements": self.requirements().iter().map(|requirement| requirement.canonical_value()).collect::<Vec<_>>(),
            "iod": IodRunConfig::core_v1().canonical_value(),
            "refinement": RefinementRunConfig::core_v1().canonical_value(),
            "evaluation": {"covariance_dimension": 6},
        })
    }

    pub fn verify_definition_fingerprint(self) -> Result<(), String> {
        let canonical = serde_json::to_vec(&self.definition_value())
            .expect("profile definition is serializable");
        let actual = format!("{:x}", Sha256::digest(canonical));
        if actual != self.identity.definition_sha256 {
            return Err(format!(
                "profile definition fingerprint mismatch for {}: expected {}, got {actual}",
                self.identity.name, self.identity.definition_sha256
            ));
        }
        Ok(())
    }

    pub fn requirements(self) -> &'static [ResourceRequirement] {
        match self.kind {
            OdProfileKind::Core => &[EFCC18_BIAS_RESOURCE],
            OdProfileKind::AsteroidInstitute => &[
                V2_SIGMA_FILL_RESOURCE,
                EFCC18_BIAS_RESOURCE,
                V2_FULL_RESOURCE,
            ],
        }
    }

    pub fn bind(self, bindings: &ResourceBindings) -> Result<OdConfig, ResourceValidationError> {
        self.verify_definition_fingerprint()
            .map_err(|reason| ResourceValidationError {
                problems: vec![ResourceProblem {
                    slot: ResourceSlot::Efcc18CatalogBiasTable,
                    reason,
                }],
            })?;
        let validated = validate_resources(self.requirements(), bindings)?;
        let get = |slot| {
            validated
                .get(&slot)
                .expect("validated profile resource is present")
        };

        let efcc = get(ResourceSlot::Efcc18CatalogBiasTable);
        let bias_text = std::str::from_utf8(&efcc.bytes).map_err(|error| {
            one_resource_error(
                ResourceSlot::Efcc18CatalogBiasTable,
                format!("resource is not UTF-8: {error}"),
            )
        })?;
        let efcc_table = Efcc18BiasTable::parse_bias_dat(bias_text)
            .map_err(|error| one_resource_error(ResourceSlot::Efcc18CatalogBiasTable, error))?;

        let mut transforms = Vec::new();
        if self.kind == OdProfileKind::AsteroidInstitute {
            let sigma = get(ResourceSlot::StationCatalogUncertaintyTable);
            let lookup =
                parse_sigma_parquet(&sigma.bytes, V2_SIGMA_FILL_RESOURCE).map_err(|error| {
                    one_resource_error(ResourceSlot::StationCatalogUncertaintyTable, error)
                })?;
            transforms.push(
                BoundObservationTransform::FillMissingAstrometricUncertainty {
                    lookup,
                    resource: sigma.identity.clone(),
                },
            );
        }
        transforms.push(BoundObservationTransform::Efcc18CatalogBiasCorrection {
            bias_table: Arc::new(efcc_table),
            exclude_astcats: Vec::new(),
            resource: efcc.identity.clone(),
        });
        if self.kind == OdProfileKind::AsteroidInstitute {
            let bias = get(ResourceSlot::StationBandResidualCovarianceTable);
            let table = parse_bias_parquet(&bias.bytes, V2_FULL_RESOURCE).map_err(|error| {
                one_resource_error(ResourceSlot::StationBandResidualCovarianceTable, error)
            })?;
            transforms.push(
                BoundObservationTransform::StationBandResidualCovarianceAdjustment {
                    table,
                    operation: ResidualCovarianceOperation::Add,
                    minimum_residual_count: 30,
                    resource: bias.identity.clone(),
                },
            );
        }
        transforms.push(BoundObservationTransform::NightlyBatchDeweighting {
            effective_observation_cap: 4,
        });

        let refinement = RefinementConfig::Cmc2003(Cmc2003FitConfig::default());
        let config = OdConfig {
            observation_pipeline: ObservationPipeline::ordered(transforms)
                .expect("profile transform definitions are valid"),
            iod: IodRunConfig::core_v1(),
            refinement: RefinementRunConfig::new(refinement)
                .expect("profile refinement settings are valid"),
            evaluation: EvaluationConfig::six_dimensional(),
            profile: Some(self.identity),
        };
        config.validate().map_err(|error| ResourceValidationError {
            problems: vec![ResourceProblem {
                slot: ResourceSlot::Efcc18CatalogBiasTable,
                reason: format!("profile definition is invalid: {error}"),
            }],
        })?;
        Ok(config)
    }
}

#[derive(Debug, Clone)]
struct ValidatedResource {
    identity: ValidatedResourceIdentity,
    bytes: Vec<u8>,
}

fn one_resource_error(slot: ResourceSlot, reason: String) -> ResourceValidationError {
    ResourceValidationError {
        problems: vec![ResourceProblem { slot, reason }],
    }
}

fn validate_resources(
    requirements: &[ResourceRequirement],
    bindings: &ResourceBindings,
) -> Result<BTreeMap<ResourceSlot, ValidatedResource>, ResourceValidationError> {
    let mut problems = Vec::new();
    let mut validated = BTreeMap::new();
    for &requirement in requirements {
        if let Err(reason) = requirement.validate() {
            problems.push(ResourceProblem {
                slot: requirement.slot,
                reason: format!("invalid requirement: {reason}"),
            });
            continue;
        }
        let Some(path) = bindings.get(requirement.slot) else {
            problems.push(ResourceProblem {
                slot: requirement.slot,
                reason: "no path was bound".to_string(),
            });
            continue;
        };
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                problems.push(ResourceProblem {
                    slot: requirement.slot,
                    reason: format!("could not read bound file: {error}"),
                });
                continue;
            }
        };
        let actual_size = bytes.len() as u64;
        let actual_sha = format!("{:x}", Sha256::digest(&bytes));
        if actual_size != requirement.byte_size {
            problems.push(ResourceProblem {
                slot: requirement.slot,
                reason: format!(
                    "byte size mismatch: expected {}, got {}",
                    requirement.byte_size, actual_size
                ),
            });
        }
        if actual_sha != requirement.sha256 {
            problems.push(ResourceProblem {
                slot: requirement.slot,
                reason: format!(
                    "SHA-256 mismatch: expected {}, got {}",
                    requirement.sha256, actual_sha
                ),
            });
        }
        if actual_size == requirement.byte_size && actual_sha == requirement.sha256 {
            validated.insert(
                requirement.slot,
                ValidatedResource {
                    identity: ValidatedResourceIdentity {
                        requirement,
                        actual_sha256: actual_sha,
                        actual_byte_size: actual_size,
                    },
                    bytes,
                },
            );
        }
    }
    if problems.is_empty() {
        Ok(validated)
    } else {
        Err(ResourceValidationError { problems })
    }
}

fn parquet_batches(
    bytes: &[u8],
    requirement: ResourceRequirement,
) -> Result<Vec<RecordBatch>, String> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .map_err(|error| format!("invalid Parquet resource: {error}"))?;
    let schema = builder.schema().clone();
    validate_resource_schema(&schema, requirement)?;
    let reader = builder
        .with_batch_size(requirement.row_count.unwrap_or(1024).max(1))
        .build()
        .map_err(|error| format!("could not build Parquet reader: {error}"))?;
    let batches: Result<Vec<_>, _> = reader.collect();
    let batches = batches.map_err(|error| format!("could not read Parquet rows: {error}"))?;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if Some(rows) != requirement.row_count {
        return Err(format!(
            "row count mismatch: expected {:?}, got {rows}",
            requirement.row_count
        ));
    }
    Ok(batches)
}

fn validate_resource_schema(
    schema: &Schema,
    requirement: ResourceRequirement,
) -> Result<(), String> {
    let expected = match requirement.slot {
        ResourceSlot::StationCatalogUncertaintyTable => sigma_schema(),
        ResourceSlot::StationBandResidualCovarianceTable => bias_schema(),
        ResourceSlot::Efcc18CatalogBiasTable => return Ok(()),
    };
    if schema != &expected {
        return Err(format!(
            "Arrow schema mismatch for {}: expected {expected:?}, got {schema:?}",
            requirement.schema_version
        ));
    }
    Ok(())
}

fn sigma_schema() -> Schema {
    Schema::new(vec![
        Field::new("obs_code", DataType::LargeUtf8, true),
        Field::new("astcat", DataType::LargeUtf8, true),
        Field::new("sigma_ra_arcsec", DataType::Float64, true),
        Field::new("sigma_dec_arcsec", DataType::Float64, true),
        Field::new("n_obs", DataType::Int64, true),
        Field::new("n_objects", DataType::Int64, true),
        Field::new("level", DataType::LargeUtf8, true),
    ])
}

fn bias_schema() -> Schema {
    Schema::new(vec![
        Field::new("obs_code", DataType::LargeUtf8, false),
        Field::new("program_code", DataType::LargeUtf8, true),
        Field::new("band", DataType::LargeUtf8, true),
        Field::new("n_obs", DataType::Int64, false),
        Field::new("n_objects", DataType::Int64, false),
        Field::new("bias_ra_arcsec", DataType::Float64, false),
        Field::new("bias_ra_ci_low", DataType::Float64, false),
        Field::new("bias_ra_ci_high", DataType::Float64, false),
        Field::new("bias_dec_arcsec", DataType::Float64, false),
        Field::new("bias_dec_ci_low", DataType::Float64, false),
        Field::new("bias_dec_ci_high", DataType::Float64, false),
        Field::new("resid_var_ra", DataType::Float64, false),
        Field::new("resid_var_dec", DataType::Float64, false),
        Field::new("resid_cov_ra_dec", DataType::Float64, false),
        Field::new("resid_cov_n", DataType::Float64, false),
        Field::new("rms_ra_arcsec", DataType::Float64, false),
        Field::new("rms_dec_arcsec", DataType::Float64, false),
        Field::new("chi2_per_obs", DataType::Float64, false),
        Field::new("bias_significant", DataType::Boolean, false),
        Field::new("high_confidence", DataType::Boolean, false),
        Field::new("confidence_score", DataType::Float64, false),
    ])
}

fn large_strings<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a LargeStringArray, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column {name}"))?
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| format!("column {name} has wrong type"))
}

fn float64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column {name}"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("column {name} has wrong type"))
}

fn parse_sigma_parquet(
    bytes: &[u8],
    requirement: ResourceRequirement,
) -> Result<VeresSigmaLookup, String> {
    let batches = parquet_batches(bytes, requirement)?;
    let mut rows = Vec::with_capacity(requirement.row_count.unwrap_or(0));
    for batch in batches {
        let code = large_strings(&batch, "obs_code")?;
        let astcat = large_strings(&batch, "astcat")?;
        let sigma_ra = float64(&batch, "sigma_ra_arcsec")?;
        let sigma_dec = float64(&batch, "sigma_dec_arcsec")?;
        for row in 0..batch.num_rows() {
            if sigma_ra.is_null(row) || sigma_dec.is_null(row) {
                return Err(format!("sigma row {row} has a null sigma"));
            }
            rows.push(VeresSigmaRow {
                obs_code: (!code.is_null(row)).then(|| code.value(row).to_string()),
                astcat: (!astcat.is_null(row)).then(|| astcat.value(row).to_string()),
                sigma_ra_arcsec: sigma_ra.value(row),
                sigma_dec_arcsec: sigma_dec.value(row),
            });
        }
    }
    VeresSigmaLookup::new(&rows, Some(0.75))
}

fn parse_bias_parquet(bytes: &[u8], requirement: ResourceRequirement) -> Result<BiasTable, String> {
    let batches = parquet_batches(bytes, requirement)?;
    let mut codes = Vec::with_capacity(requirement.row_count.unwrap_or(0));
    let mut bands = Vec::with_capacity(requirement.row_count.unwrap_or(0));
    let mut rows = Vec::with_capacity(requirement.row_count.unwrap_or(0));
    for batch in batches {
        let code = large_strings(&batch, "obs_code")?;
        let band = large_strings(&batch, "band")?;
        let bias_ra = float64(&batch, "bias_ra_arcsec")?;
        let bias_dec = float64(&batch, "bias_dec_arcsec")?;
        let resid_ra = float64(&batch, "resid_var_ra")?;
        let resid_dec = float64(&batch, "resid_var_dec")?;
        let resid_cov = float64(&batch, "resid_cov_ra_dec")?;
        let resid_n = float64(&batch, "resid_cov_n")?;
        let chi2 = float64(&batch, "chi2_per_obs")?;
        for row in 0..batch.num_rows() {
            codes.push(code.value(row).to_string());
            bands.push((!band.is_null(row)).then(|| band.value(row).to_string()));
            rows.push(BiasTableRow {
                bias_ra_arcsec: bias_ra.value(row),
                bias_dec_arcsec: bias_dec.value(row),
                resid_var_ra: resid_ra.value(row),
                resid_var_dec: resid_dec.value(row),
                resid_cov_ra_dec: resid_cov.value(row),
                resid_cov_n: resid_n.value(row),
                chi2_per_obs: chi2.value(row),
            });
        }
    }
    BiasTable::new(codes, bands, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_are_explicit_and_path_free() {
        CORE_OD_V1.verify_definition_fingerprint().unwrap();
        ASTEROID_INSTITUTE_OD_V1
            .verify_definition_fingerprint()
            .unwrap();
        assert_eq!(CORE_OD_V1.requirements(), &[EFCC18_BIAS_RESOURCE]);
        assert_eq!(ASTEROID_INSTITUTE_OD_V1.requirements().len(), 3);
        for requirement in ASTEROID_INSTITUTE_OD_V1.requirements() {
            let canonical = requirement.canonical_value();
            assert!(canonical.get("path").is_none());
            assert!(!canonical.to_string().contains("file://"));
        }
    }

    #[test]
    fn as_supplied_is_explicit_and_empty_ordered_is_rejected() {
        assert!(ObservationPipeline::as_supplied().is_as_supplied());
        assert!(ObservationPipeline::ordered(Vec::new()).is_err());
    }

    #[test]
    fn missing_resources_are_aggregated_in_profile_order() {
        let error = ASTEROID_INSTITUTE_OD_V1
            .bind(&ResourceBindings::new())
            .unwrap_err();
        assert_eq!(error.problems.len(), 3);
        assert_eq!(
            error.problems[0].slot,
            ResourceSlot::StationCatalogUncertaintyTable
        );
        assert_eq!(error.problems[1].slot, ResourceSlot::Efcc18CatalogBiasTable);
        assert_eq!(
            error.problems[2].slot,
            ResourceSlot::StationBandResidualCovarianceTable
        );
    }

    #[test]
    #[ignore = "local acceptance check against separately packaged reference resources"]
    fn separately_packaged_reference_resources_parse() {
        let jpl = std::env::var("ADAM_TEST_EFCC18_BIAS_DAT").unwrap();
        let sigma = std::env::var("ADAM_TEST_SIGMA_FILL_PARQUET").unwrap();
        let full = std::env::var("ADAM_TEST_RESIDUAL_COVARIANCE_PARQUET").unwrap();
        let bindings = ResourceBindings::new()
            .bind(ResourceSlot::Efcc18CatalogBiasTable, jpl)
            .bind(ResourceSlot::StationCatalogUncertaintyTable, sigma)
            .bind(ResourceSlot::StationBandResidualCovarianceTable, full);
        let config = ASTEROID_INSTITUTE_OD_V1.bind(&bindings).unwrap();
        assert_eq!(config.observation_pipeline.transform_kinds().len(), 4);
    }

    #[test]
    fn corrupt_resources_are_aggregated_before_parsing() {
        let root = std::env::temp_dir().join(format!("adam-core-od-config-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let bindings = ResourceBindings::new()
            .bind(
                ResourceSlot::StationCatalogUncertaintyTable,
                root.join("sigma.parquet"),
            )
            .bind(ResourceSlot::Efcc18CatalogBiasTable, root.join("bias.dat"))
            .bind(
                ResourceSlot::StationBandResidualCovarianceTable,
                root.join("residual.parquet"),
            );
        for path in bindings.paths.values() {
            std::fs::write(path, b"not the required scientific resource").unwrap();
        }
        let error = ASTEROID_INSTITUTE_OD_V1.bind(&bindings).unwrap_err();
        let slots: std::collections::BTreeSet<_> =
            error.problems.iter().map(|problem| problem.slot).collect();
        assert_eq!(slots.len(), 3);
        assert!(error
            .problems
            .iter()
            .all(|problem| problem.reason.contains("mismatch")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repeated_transforms_preserve_order() {
        let pipeline = ObservationPipeline::bind_explicit(
            vec![
                ObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap: 4,
                },
                ObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap: 1,
                },
            ],
            &ResourceBindings::new(),
        )
        .unwrap();
        assert_eq!(
            pipeline.transform_kinds(),
            vec!["nightly_batch_deweighting", "nightly_batch_deweighting"]
        );
    }
}
