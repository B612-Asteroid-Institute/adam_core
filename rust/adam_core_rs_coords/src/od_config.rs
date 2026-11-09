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
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::efcc18::Efcc18BiasTable;
use crate::observation_uncertainty::{
    BiasTable, BiasTableRow, Efcc18DebiasModel, EmpiricalCovarianceMode, EmpiricalCovarianceModel,
    NightBatchDeweightingModel, ObservationUncertaintyModel, OrbitDeterminationAstrometry,
    PerformanceWeightedModel, SigmaFloorModel,
};
use crate::propagation::{
    CorrectionConfig, IodAcceptance, IodConfig, IterativeFitConfig, JacobianMethod,
    ObservationSelectionMethod, PrefitScope, RangingConfig, SpanSearchConfig, VelocityMethod,
};
use crate::veres2017::{
    SigmaFillModel, VeresFloorModel, VeresReplaceModel, VeresSigmaLookup, VeresSigmaRow,
};
use crate::{Cmc2003FitConfig, LossType, RefinementConfig, TwoBodyModelConfig, WhitenedFitConfig};

/// Stable semantic slots used by profiles.  Paths are deployment details and
/// are deliberately absent from scientific identity and provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceSlot {
    /// Station/catalog astrometric-uncertainty lookup used to fill missing values.
    StationCatalogUncertaintyTable,
    /// EFCC18 catalog-bias table used for deterministic debiasing.
    Efcc18CatalogBiasTable,
    /// Station/band empirical residual-covariance lookup.
    StationBandResidualCovarianceTable,
}

impl ResourceSlot {
    /// Stable snake-case slot name used in configuration provenance.
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
    /// Semantic slot occupied by the resource.
    pub slot: ResourceSlot,
    /// Stable scientific product name.
    pub product: &'static str,
    /// Exact product version.
    pub product_version: &'static str,
    /// Expected schema identity.
    pub schema_version: &'static str,
    /// Expected lowercase SHA-256 of the complete file.
    pub sha256: &'static str,
    /// Expected file size in bytes.
    pub byte_size: u64,
    /// Expected logical row count when the format is tabular.
    pub row_count: Option<usize>,
    /// Catalog date when supplied by the scientific product.
    pub catalog_date: Option<&'static str>,
    /// Expected media type.
    pub format: &'static str,
}

impl ResourceRequirement {
    /// Validate this declaration's identity, size and slot-specific schema contract.
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

    /// Return the path-free canonical JSON value persisted in provenance.
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

/// Exact public EFCC18 bias-table requirement used by Core OD V1.
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

/// Exact station/catalog uncertainty requirement used by the Institute profile.
pub const V2_SIGMA_FILL_RESOURCE: ResourceRequirement = ResourceRequirement {
    slot: ResourceSlot::StationCatalogUncertaintyTable,
    product: "v2_sigma_fill",
    product_version: "v2_sigma_fill",
    schema_version: "adam-observatory-sigma-table/v1",
    sha256: "95a6d12c0dc1f6278a869e2e0d7d356b54683773d536d286f00326a3d868495b",
    byte_size: 52_288,
    row_count: Some(2_480),
    catalog_date: None,
    format: "application/vnd.apache.parquet",
};

/// Exact station/band residual-covariance requirement used by the Institute profile.
pub const V2_FULL_RESOURCE: ResourceRequirement = ResourceRequirement {
    slot: ResourceSlot::StationBandResidualCovarianceTable,
    product: "v2_full",
    product_version: "v2_full",
    schema_version: "adam-observatory-bias-table/v1",
    sha256: "ed0083b041fe523e6cfbabd3bff80bb0688b3056982de48d7789e59fdf370bac",
    byte_size: 158_965,
    row_count: Some(1_126),
    catalog_date: Some("2026-06-23"),
    format: "application/vnd.apache.parquet",
};

/// Caller-owned concrete paths for semantic resource slots.
#[derive(Debug, Clone, Default)]
pub struct ResourceBindings {
    paths: BTreeMap<ResourceSlot, PathBuf>,
}

impl ResourceBindings {
    /// Construct empty bindings; profiles then reject any missing required slots.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind one semantic slot to a caller-owned local path.
    pub fn bind(mut self, slot: ResourceSlot, path: impl AsRef<Path>) -> Self {
        self.paths.insert(slot, path.as_ref().to_path_buf());
        self
    }

    /// Return the local path bound to `slot`, if present.
    pub fn get(&self, slot: ResourceSlot) -> Option<&Path> {
        self.paths.get(&slot).map(PathBuf::as_path)
    }

    /// Iterate bound slots in deterministic semantic-slot order.
    pub fn slots(&self) -> impl Iterator<Item = ResourceSlot> + '_ {
        self.paths.keys().copied()
    }
}

/// One aggregate resource-validation defect associated with a semantic slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceProblem {
    /// Slot whose binding or content is invalid.
    pub slot: ResourceSlot,
    /// Human-readable validation detail.
    pub reason: String,
}

/// Aggregate fail-closed validation error covering every supplied resource slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceValidationError {
    /// Deterministically ordered slot problems discovered before pipeline work.
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

/// Typed failure while constructing or validating a complete OD configuration.
/// Configuration defects are deliberately distinct from resource-slot defects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdConfigError {
    /// One or more invalid scientific configuration declarations.
    Configuration {
        /// Deterministically collected configuration defects.
        problems: Vec<String>,
    },
    /// One or more missing, unused, mismatched or malformed resources.
    Resources(ResourceValidationError),
}

impl OdConfigError {
    fn configuration(reason: impl Into<String>) -> Self {
        Self::Configuration {
            problems: vec![reason.into()],
        }
    }

    /// Return configuration defects, or an empty slice for resource failures.
    pub fn configuration_problems(&self) -> &[String] {
        match self {
            Self::Configuration { problems } => problems,
            Self::Resources(_) => &[],
        }
    }

    /// Return resource defects, or an empty slice for configuration failures.
    pub fn resource_problems(&self) -> &[ResourceProblem] {
        match self {
            Self::Configuration { .. } => &[],
            Self::Resources(error) => &error.problems,
        }
    }
}

impl From<ResourceValidationError> for OdConfigError {
    fn from(value: ResourceValidationError) -> Self {
        Self::Resources(value)
    }
}

impl fmt::Display for OdConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration { problems } => {
                writeln!(f, "OD configuration validation failed:")?;
                for problem in problems {
                    writeln!(f, "  {problem}")?;
                }
                Ok(())
            }
            Self::Resources(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for OdConfigError {}

/// Path-free identity of bytes validated against one resource requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedResourceIdentity {
    /// Requirement that selected and constrained these bytes.
    pub requirement: ResourceRequirement,
    /// Lowercase SHA-256 computed from the bound bytes.
    pub actual_sha256: String,
    /// File size computed from the bound bytes.
    pub actual_byte_size: u64,
}

impl ValidatedResourceIdentity {
    /// Return the canonical path-free identity persisted in provenance.
    pub fn canonical_value(&self) -> Value {
        json!({
            "requirement": self.requirement.canonical_value(),
            "actual_sha256": self.actual_sha256,
            "actual_byte_size": self.actual_byte_size,
        })
    }
}

/// How a station/band empirical residual covariance changes supplied covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualCovarianceOperation {
    /// Add empirical covariance to supplied covariance.
    Add,
    /// Replace supplied covariance with empirical covariance.
    Replace,
}

impl ResidualCovarianceOperation {
    /// Stable lowercase operation name used in provenance.
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
    /// Fill only missing angular uncertainty from a station/catalog lookup.
    FillMissingAstrometricUncertainty {
        /// Exact station/catalog uncertainty-table requirement to bind.
        resource: ResourceRequirement,
    },
    /// Floor supplied per-axis uncertainty at the VFC2017 station/catalog value.
    /// The supplied angular cross term is preserved.
    Vfc2017AstrometricUncertaintyFloor {
        /// Exact station/catalog uncertainty-table requirement to bind.
        resource: ResourceRequirement,
        /// Whether missing/non-finite supplied variances are filled by the floor.
        fill_missing: bool,
    },
    /// Replace supplied per-axis uncertainty with the VFC2017 station/catalog value.
    /// Both angular cross entries become exactly +0.0 because the table has no
    /// correlation.
    Vfc2017AstrometricUncertaintyReplace {
        /// Exact station/catalog uncertainty-table requirement to bind.
        resource: ResourceRequirement,
    },
    /// Apply EFCC18 catalog-bias correction except for explicitly excluded catalogs.
    Efcc18CatalogBiasCorrection {
        /// Exact EFCC18 catalog-bias-table requirement to bind.
        resource: ResourceRequirement,
        /// Astrometric catalog identifiers that must not receive the correction.
        exclude_astcats: Vec<String>,
    },
    /// Add or replace covariance from a sufficiently supported station/band lookup.
    StationBandResidualCovarianceAdjustment {
        /// Exact station/band residual-covariance-table requirement to bind.
        resource: ResourceRequirement,
        /// Whether empirical covariance is added to or replaces supplied covariance.
        operation: ResidualCovarianceOperation,
        /// Minimum residual count required before applying one lookup-table row.
        minimum_residual_count: usize,
    },
    /// Scale supplied covariance by `max(chi2_per_obs, 1)` from station/band statistics.
    PerformanceWeightedCovarianceScaling {
        /// Exact station/band residual-statistics table requirement to bind.
        resource: ResourceRequirement,
        /// Minimum residual count required before applying one lookup-table row.
        minimum_residual_count: usize,
    },
    /// Floor supplied per-axis uncertainty at measured station/band bias magnitude.
    MeasuredBiasAstrometricUncertaintyFloor {
        /// Exact station/band residual-statistics table requirement to bind.
        resource: ResourceRequirement,
        /// Minimum residual count required before applying one lookup-table row.
        minimum_residual_count: usize,
    },
    /// Deweight same-night observations to an effective independent-observation cap.
    NightlyBatchDeweighting {
        /// Effective independent-observation cap used to compute nightly deweighting.
        effective_observation_cap: usize,
    },
}

impl ObservationTransform {
    fn definition_value(&self) -> Value {
        match self {
            Self::FillMissingAstrometricUncertainty { .. } => {
                json!({"kind": "fill_missing_astrometric_uncertainty"})
            }
            Self::Vfc2017AstrometricUncertaintyFloor { fill_missing, .. } => json!({
                "kind": "vfc2017_astrometric_uncertainty_floor",
                "fill_missing": fill_missing,
            }),
            Self::Vfc2017AstrometricUncertaintyReplace { .. } => {
                json!({"kind": "vfc2017_astrometric_uncertainty_replace"})
            }
            Self::Efcc18CatalogBiasCorrection {
                exclude_astcats, ..
            } => json!({
                "kind": "efcc18_catalog_bias_correction",
                "exclude_astcats": exclude_astcats,
            }),
            Self::StationBandResidualCovarianceAdjustment {
                operation,
                minimum_residual_count,
                ..
            } => json!({
                "kind": "station_band_residual_covariance_adjustment",
                "operation": operation.as_str(),
                "minimum_residual_count": minimum_residual_count,
            }),
            Self::PerformanceWeightedCovarianceScaling {
                minimum_residual_count,
                ..
            } => json!({
                "kind": "performance_weighted_covariance_scaling",
                "minimum_residual_count": minimum_residual_count,
            }),
            Self::MeasuredBiasAstrometricUncertaintyFloor {
                minimum_residual_count,
                ..
            } => json!({
                "kind": "measured_bias_astrometric_uncertainty_floor",
                "minimum_residual_count": minimum_residual_count,
            }),
            Self::NightlyBatchDeweighting {
                effective_observation_cap,
            } => json!({
                "kind": "nightly_batch_deweighting",
                "effective_observation_cap": effective_observation_cap,
            }),
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::FillMissingAstrometricUncertainty { resource }
            | Self::Vfc2017AstrometricUncertaintyFloor { resource, .. }
            | Self::Vfc2017AstrometricUncertaintyReplace { resource } => {
                resource.validate()?;
                if resource.slot != ResourceSlot::StationCatalogUncertaintyTable {
                    return Err(
                        "VFC2017 uncertainty transforms require station_catalog_uncertainty_table"
                            .to_string(),
                    );
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
            }
            | Self::PerformanceWeightedCovarianceScaling {
                resource,
                minimum_residual_count,
            }
            | Self::MeasuredBiasAstrometricUncertaintyFloor {
                resource,
                minimum_residual_count,
            } => {
                resource.validate()?;
                if resource.slot != ResourceSlot::StationBandResidualCovarianceTable {
                    return Err("station/band covariance transforms require \
                         station_band_residual_covariance_table"
                        .to_string());
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
    Vfc2017AstrometricUncertaintyFloor {
        lookup: VeresSigmaLookup,
        fill_missing: bool,
        resource: ValidatedResourceIdentity,
    },
    Vfc2017AstrometricUncertaintyReplace {
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
    PerformanceWeightedCovarianceScaling {
        table: BiasTable,
        minimum_residual_count: usize,
        resource: ValidatedResourceIdentity,
    },
    MeasuredBiasAstrometricUncertaintyFloor {
        table: BiasTable,
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
            }
            | Self::PerformanceWeightedCovarianceScaling {
                minimum_residual_count,
                ..
            }
            | Self::MeasuredBiasAstrometricUncertaintyFloor {
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
            Self::Vfc2017AstrometricUncertaintyFloor { .. } => {
                "vfc2017_astrometric_uncertainty_floor"
            }
            Self::Vfc2017AstrometricUncertaintyReplace { .. } => {
                "vfc2017_astrometric_uncertainty_replace"
            }
            Self::Efcc18CatalogBiasCorrection { .. } => "efcc18_catalog_bias_correction",
            Self::StationBandResidualCovarianceAdjustment { .. } => {
                "station_band_residual_covariance_adjustment"
            }
            Self::PerformanceWeightedCovarianceScaling { .. } => {
                "performance_weighted_covariance_scaling"
            }
            Self::MeasuredBiasAstrometricUncertaintyFloor { .. } => {
                "measured_bias_astrometric_uncertainty_floor"
            }
            Self::NightlyBatchDeweighting { .. } => "nightly_batch_deweighting",
        }
    }

    #[doc(hidden)]
    pub(crate) fn apply(
        &self,
        observations: &mut OrbitDeterminationAstrometry,
    ) -> Result<bool, String> {
        self.validate()?;
        match self {
            Self::FillMissingAstrometricUncertainty { lookup, .. } => SigmaFillModel {
                lookup: lookup.clone(),
            }
            .apply(observations),
            Self::Vfc2017AstrometricUncertaintyFloor {
                lookup,
                fill_missing,
                ..
            } => VeresFloorModel {
                lookup: lookup.clone(),
                fill_missing: *fill_missing,
            }
            .apply(observations),
            Self::Vfc2017AstrometricUncertaintyReplace { lookup, .. } => VeresReplaceModel {
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
            Self::PerformanceWeightedCovarianceScaling {
                table,
                minimum_residual_count,
                ..
            } => PerformanceWeightedModel {
                table: table.clone(),
                min_resid_cov_n: *minimum_residual_count as f64,
            }
            .apply(observations),
            Self::MeasuredBiasAstrometricUncertaintyFloor {
                table,
                minimum_residual_count,
                ..
            } => SigmaFloorModel {
                table: table.clone(),
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
            Self::FillMissingAstrometricUncertainty { resource, .. }
            | Self::Vfc2017AstrometricUncertaintyReplace { resource, .. } => json!({
                "kind": self.kind(), "resource": resource.canonical_value(),
            }),
            Self::Vfc2017AstrometricUncertaintyFloor {
                fill_missing,
                resource,
                ..
            } => json!({
                "kind": self.kind(), "fill_missing": fill_missing,
                "resource": resource.canonical_value(),
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
            Self::PerformanceWeightedCovarianceScaling {
                minimum_residual_count,
                resource,
                ..
            }
            | Self::MeasuredBiasAstrometricUncertaintyFloor {
                minimum_residual_count,
                resource,
                ..
            } => json!({
                "kind": self.kind(),
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
    /// Select the deliberate no-transform pipeline recorded as `transforms=[]`.
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

    pub(crate) fn validate(&self) -> Result<(), String> {
        for transform in &self.transforms {
            transform.validate()?;
        }
        Ok(())
    }

    /// Return transform kind names in authoritative execution order.
    pub fn transform_kinds(&self) -> Vec<&'static str> {
        self.transforms
            .iter()
            .map(BoundObservationTransform::kind)
            .collect()
    }

    /// Whether this is the explicit no-transform pipeline.
    pub const fn is_as_supplied(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Execute exactly once in declaration order. Repeated variants are
    /// retained and executed; no sorting or deduplication occurs.
    pub(crate) fn apply(
        &self,
        observations: &mut OrbitDeterminationAstrometry,
    ) -> Result<bool, String> {
        let mut changed = false;
        for transform in &self.transforms {
            changed |= transform.apply(observations)?;
        }
        Ok(changed)
    }

    /// Return ordered, path-free transform provenance.
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
    ) -> Result<Self, OdConfigError> {
        if definitions.is_empty() {
            return Err(OdConfigError::configuration(
                "an empty transform list is ambiguous; use ObservationPipeline::as_supplied()",
            ));
        }
        let configuration_problems: Vec<String> = definitions
            .iter()
            .enumerate()
            .filter_map(|(index, definition)| {
                definition
                    .validate()
                    .err()
                    .map(|reason| format!("transform[{index}]: {reason}"))
            })
            .collect();
        if !configuration_problems.is_empty() {
            return Err(OdConfigError::Configuration {
                problems: configuration_problems,
            });
        }
        let requirements: Vec<_> = definitions
            .iter()
            .filter_map(|definition| match definition {
                ObservationTransform::FillMissingAstrometricUncertainty { resource }
                | ObservationTransform::Vfc2017AstrometricUncertaintyFloor { resource, .. }
                | ObservationTransform::Vfc2017AstrometricUncertaintyReplace { resource }
                | ObservationTransform::Efcc18CatalogBiasCorrection { resource, .. }
                | ObservationTransform::StationBandResidualCovarianceAdjustment {
                    resource, ..
                }
                | ObservationTransform::PerformanceWeightedCovarianceScaling { resource, .. }
                | ObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
                    resource, ..
                } => Some(*resource),
                ObservationTransform::NightlyBatchDeweighting { .. } => None,
            })
            .collect();
        let validated = validate_resources(&requirements, bindings)?;
        let mut transforms = Vec::with_capacity(definitions.len());
        let mut parse_problems = Vec::new();
        for definition in definitions {
            let result: Result<BoundObservationTransform, ResourceProblem> = match definition {
                ObservationTransform::FillMissingAstrometricUncertainty { resource } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    parse_sigma_parquet(&bound.bytes, resource)
                        .map(|lookup| {
                            BoundObservationTransform::FillMissingAstrometricUncertainty {
                                lookup,
                                resource: bound.identity.clone(),
                            }
                        })
                        .map_err(|reason| ResourceProblem {
                            slot: resource.slot,
                            reason,
                        })
                }
                ObservationTransform::Vfc2017AstrometricUncertaintyFloor {
                    resource,
                    fill_missing,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    parse_sigma_parquet(&bound.bytes, resource)
                        .map(|lookup| {
                            BoundObservationTransform::Vfc2017AstrometricUncertaintyFloor {
                                lookup,
                                fill_missing,
                                resource: bound.identity.clone(),
                            }
                        })
                        .map_err(|reason| ResourceProblem {
                            slot: resource.slot,
                            reason,
                        })
                }
                ObservationTransform::Vfc2017AstrometricUncertaintyReplace { resource } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    parse_sigma_parquet(&bound.bytes, resource)
                        .map(|lookup| {
                            BoundObservationTransform::Vfc2017AstrometricUncertaintyReplace {
                                lookup,
                                resource: bound.identity.clone(),
                            }
                        })
                        .map_err(|reason| ResourceProblem {
                            slot: resource.slot,
                            reason,
                        })
                }
                ObservationTransform::Efcc18CatalogBiasCorrection {
                    resource,
                    exclude_astcats,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    std::str::from_utf8(&bound.bytes)
                        .map_err(|error| ResourceProblem {
                            slot: resource.slot,
                            reason: format!("resource is not UTF-8: {error}"),
                        })
                        .and_then(|text| {
                            Efcc18BiasTable::parse_bias_dat(text).map_err(|reason| {
                                ResourceProblem {
                                    slot: resource.slot,
                                    reason,
                                }
                            })
                        })
                        .map(
                            |table| BoundObservationTransform::Efcc18CatalogBiasCorrection {
                                bias_table: Arc::new(table),
                                exclude_astcats,
                                resource: bound.identity.clone(),
                            },
                        )
                }
                ObservationTransform::StationBandResidualCovarianceAdjustment {
                    resource,
                    operation,
                    minimum_residual_count,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    parse_bias_parquet(&bound.bytes, resource)
                        .map(|table| {
                            BoundObservationTransform::StationBandResidualCovarianceAdjustment {
                                table,
                                operation,
                                minimum_residual_count,
                                resource: bound.identity.clone(),
                            }
                        })
                        .map_err(|reason| ResourceProblem {
                            slot: resource.slot,
                            reason,
                        })
                }
                ObservationTransform::PerformanceWeightedCovarianceScaling {
                    resource,
                    minimum_residual_count,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    parse_bias_parquet(&bound.bytes, resource)
                        .map(|table| {
                            BoundObservationTransform::PerformanceWeightedCovarianceScaling {
                                table,
                                minimum_residual_count,
                                resource: bound.identity.clone(),
                            }
                        })
                        .map_err(|reason| ResourceProblem {
                            slot: resource.slot,
                            reason,
                        })
                }
                ObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
                    resource,
                    minimum_residual_count,
                } => {
                    let bound = validated
                        .get(&resource.slot)
                        .expect("validated explicit resource");
                    parse_bias_parquet(&bound.bytes, resource)
                        .map(|table| {
                            BoundObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
                                table,
                                minimum_residual_count,
                                resource: bound.identity.clone(),
                            }
                        })
                        .map_err(|reason| ResourceProblem {
                            slot: resource.slot,
                            reason,
                        })
                }
                ObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap,
                } => Ok(BoundObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap,
                }),
            };
            match result {
                Ok(transform) => transforms.push(transform),
                Err(problem) => parse_problems.push(problem),
            }
        }
        if !parse_problems.is_empty() {
            return Err(ResourceValidationError {
                problems: parse_problems,
            }
            .into());
        }
        ObservationPipeline::ordered(transforms).map_err(OdConfigError::configuration)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TwoBodyDeclaration {
    propagation_max_iter: usize,
    propagation_tol: f64,
    lt_tol: f64,
    ephemeris_max_iter: usize,
    ephemeris_tol: f64,
    max_lt_iter: usize,
}

impl From<TwoBodyDeclaration> for TwoBodyModelConfig {
    fn from(value: TwoBodyDeclaration) -> Self {
        Self {
            propagation_max_iter: value.propagation_max_iter,
            propagation_tol: value.propagation_tol,
            lt_tol: value.lt_tol,
            ephemeris_max_iter: value.ephemeris_max_iter,
            ephemeris_tol: value.ephemeris_tol,
            max_lt_iter: value.max_lt_iter,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FitDeclaration {
    loss: String,
    f_scale: f64,
    jacobian: String,
    validate_covariance: bool,
    xtol: f64,
    ftol: f64,
    gtol: f64,
    max_iterations: usize,
    max_nfev: Option<usize>,
    two_body: TwoBodyDeclaration,
}

impl FitDeclaration {
    fn into_config(self) -> Result<WhitenedFitConfig, String> {
        let config = WhitenedFitConfig {
            loss: LossType::parse(&self.loss)?,
            f_scale: self.f_scale,
            jacobian: JacobianMethod::parse(&self.jacobian)?,
            validate_covariance: self.validate_covariance,
            xtol: self.xtol,
            ftol: self.ftol,
            gtol: self.gtol,
            max_iterations: self.max_iterations,
            max_nfev: self.max_nfev,
            two_body: self.two_body.into(),
        };
        config.validate()?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpanDeclaration {
    target_span_days: f64,
    min_span_days: f64,
    max_span_days: f64,
    dense_window_days: [f64; 2],
    edge_window_days: f64,
    max_triplets: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorrectionDeclaration {
    max_corrections: usize,
    max_converged: usize,
    growth: f64,
    min_stage_obs: usize,
    max_stages: usize,
    stage_loss: String,
    prefit_scope: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RangingDeclaration {
    enabled: bool,
    rho_min_au: f64,
    rho_max_au: f64,
    n_rho: usize,
    n_rho_dot: usize,
    max_corrections: usize,
    chunk_size: usize,
    max_arc_days: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IodDeclaration {
    min_obs: usize,
    min_arc_length: f64,
    contamination_percentage: f64,
    rchi2_threshold: f64,
    observation_selection_method: String,
    light_time: bool,
    mu: f64,
    speed_of_light: f64,
    max_triplets: Option<usize>,
    max_pair_visits: usize,
    max_raw_pair_visits: usize,
    max_middle_visits: usize,
    velocity_method: String,
    spans: SpanDeclaration,
    acceptance: String,
    correction: CorrectionDeclaration,
    ranging: RangingDeclaration,
}

impl IodDeclaration {
    fn into_config(self) -> Result<IodConfig, String> {
        let config = IodConfig {
            min_obs: self.min_obs,
            min_arc_length: self.min_arc_length,
            contamination_percentage: self.contamination_percentage,
            rchi2_threshold: self.rchi2_threshold,
            observation_selection_method: ObservationSelectionMethod::parse(
                &self.observation_selection_method,
            )?,
            light_time: self.light_time,
            mu: self.mu,
            speed_of_light: self.speed_of_light,
            max_triplets: self.max_triplets,
            max_pair_visits: self.max_pair_visits,
            max_raw_pair_visits: self.max_raw_pair_visits,
            max_middle_visits: self.max_middle_visits,
            velocity_method: VelocityMethod::parse(&self.velocity_method)?,
            spans: SpanSearchConfig {
                target_span_days: self.spans.target_span_days,
                min_span_days: self.spans.min_span_days,
                max_span_days: self.spans.max_span_days,
                dense_window_days: self.spans.dense_window_days,
                edge_window_days: self.spans.edge_window_days,
                max_triplets: self.spans.max_triplets,
            },
            acceptance: IodAcceptance::parse(&self.acceptance)?,
            correction: CorrectionConfig {
                max_corrections: self.correction.max_corrections,
                max_converged: self.correction.max_converged,
                growth: self.correction.growth,
                min_stage_obs: self.correction.min_stage_obs,
                max_stages: self.correction.max_stages,
                stage_loss: LossType::parse(&self.correction.stage_loss)?,
                prefit_scope: PrefitScope::parse(&self.correction.prefit_scope)?,
            },
            ranging: RangingConfig {
                enabled: self.ranging.enabled,
                rho_min_au: self.ranging.rho_min_au,
                rho_max_au: self.ranging.rho_max_au,
                n_rho: self.ranging.n_rho,
                n_rho_dot: self.ranging.n_rho_dot,
                max_corrections: self.ranging.max_corrections,
                chunk_size: self.ranging.chunk_size,
                max_arc_days: self.ranging.max_arc_days,
            },
        };
        config.validate()?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IodRunDeclaration {
    search_correction_ranging: IodDeclaration,
    candidate_correction_fit: FitDeclaration,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationDeclaration {
    covariance_dimension: usize,
    prediction_fit: FitDeclaration,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum RefinementDeclaration {
    WorstResidual {
        rchi2_threshold: f64,
        min_obs: usize,
        min_arc_length: f64,
        contamination_percentage: f64,
        fit: FitDeclaration,
    },
    Cmc2003 {
        chi2_reject: f64,
        chi2_recover: f64,
        chi2_frac: f64,
        max_iterations: usize,
        max_rejected_fraction: f64,
        apparition_gap_days: f64,
        psd_floor_frac: f64,
        min_kept: usize,
        fit: FitDeclaration,
    },
}

/// Complete declaration of deterministic IOD search, correction and candidate fitting.
#[derive(Debug, Clone)]
pub struct IodStageConfig {
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

fn d21_fit_config() -> WhitenedFitConfig {
    let mut config = WhitenedFitConfig::default();
    // Preserve the accepted d21 effective defaults exactly. These values were
    // supplied by EphemerisOptions::default() when no two-body keys existed.
    config.two_body.lt_tol = 1.0e-12;
    config.two_body.ephemeris_max_iter = 1_000;
    config.two_body.max_lt_iter = 10;
    config
}

impl IodStageConfig {
    /// Decode and validate a complete canonical IOD-stage JSON declaration.
    pub fn from_json(raw: &str) -> Result<Self, String> {
        let declaration: IodRunDeclaration = serde_json::from_str(raw)
            .map_err(|error| format!("invalid IOD run configuration: {error}"))?;
        Self::new(
            declaration.search_correction_ranging.into_config()?,
            declaration.candidate_correction_fit.into_config()?,
        )
    }

    /// Serialize the canonical IOD-stage declaration.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.canonical_value())
            .expect("IOD run configuration is serializable")
    }

    /// Construct a validated stage from explicit search and correction-fit settings.
    pub fn new(iod: IodConfig, correction_fit: WhitenedFitConfig) -> Result<Self, String> {
        let config = Self {
            iod,
            correction_fit,
        };
        config.validate()?;
        Ok(config)
    }

    /// Return the exact named Core-v1 IOD values; this is never an implicit default.
    pub fn core_v1() -> Self {
        Self::new(IodConfig::default(), d21_fit_config()).expect("Core-v1 IOD settings are valid")
    }

    /// Borrow the search, selection, acceptance, correction and ranging policy.
    pub const fn iod(&self) -> &IodConfig {
        &self.iod
    }

    /// Borrow the fit settings used to correct and score IOD candidates.
    pub const fn correction_fit(&self) -> &WhitenedFitConfig {
        &self.correction_fit
    }

    /// Return the path-free canonical JSON value used by complete configurations.
    pub fn canonical_value(&self) -> Value {
        json!({
            "search_correction_ranging": iod_config_value(&self.iod),
            "candidate_correction_fit": fit_config_value(&self.correction_fit),
        })
    }

    /// Validate every search and correction-fit setting.
    pub fn validate(&self) -> Result<(), String> {
        self.iod.validate()?;
        self.correction_fit.validate()?;
        Ok(())
    }
}

/// Complete standalone refinement-stage fit and outlier-treatment declaration.
#[derive(Debug, Clone, Copy)]
pub struct RefinementRunConfig {
    refinement: RefinementConfig,
}

impl RefinementRunConfig {
    /// Decode and validate a canonical refinement-stage JSON declaration.
    pub fn from_json(raw: &str) -> Result<Self, String> {
        let declaration: RefinementDeclaration = serde_json::from_str(raw)
            .map_err(|error| format!("invalid refinement run configuration: {error}"))?;
        let refinement = match declaration {
            RefinementDeclaration::WorstResidual {
                rchi2_threshold,
                min_obs,
                min_arc_length,
                contamination_percentage,
                fit,
            } => RefinementConfig::WorstResidual(IterativeFitConfig {
                rchi2_threshold,
                min_obs,
                min_arc_length,
                contamination_percentage,
                fit: fit.into_config()?,
            }),
            RefinementDeclaration::Cmc2003 {
                chi2_reject,
                chi2_recover,
                chi2_frac,
                max_iterations,
                max_rejected_fraction,
                apparition_gap_days,
                psd_floor_frac,
                min_kept,
                fit,
            } => RefinementConfig::Cmc2003(Cmc2003FitConfig {
                chi2_reject,
                chi2_recover,
                chi2_frac,
                max_iterations,
                max_rejected_fraction,
                apparition_gap_days,
                psd_floor_frac,
                min_kept,
                fit: fit.into_config()?,
            }),
        };
        Self::new(refinement)
    }

    /// Serialize the canonical refinement-stage declaration.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.canonical_value())
            .expect("refinement run configuration is serializable")
    }

    /// Construct a validated stage from one explicit refinement policy.
    pub fn new(refinement: RefinementConfig) -> Result<Self, String> {
        let config = Self { refinement };
        config.validate()?;
        Ok(config)
    }

    /// Return the exact named Core-v1 CMC2003 refinement values.
    pub fn core_v1() -> Self {
        let cmc = Cmc2003FitConfig {
            fit: d21_fit_config(),
            ..Cmc2003FitConfig::default()
        };
        Self::new(RefinementConfig::Cmc2003(cmc)).expect("Core-v1 refinement settings are valid")
    }

    /// Borrow the declared refinement policy.
    pub const fn refinement(&self) -> &RefinementConfig {
        &self.refinement
    }

    /// Return the path-free canonical JSON value used in provenance.
    pub fn canonical_value(&self) -> Value {
        refinement_config_value(&self.refinement)
    }

    /// Validate the complete fit and outlier-treatment policy.
    pub fn validate(&self) -> Result<(), String> {
        self.refinement.validate()
    }
}

/// Independent covariance and prediction-fit declaration for orbit evaluation.
#[derive(Debug, Clone, Copy)]
pub struct EvaluationConfig {
    covariance_dimension: usize,
    prediction_fit: WhitenedFitConfig,
}

impl EvaluationConfig {
    /// Decode and validate a canonical evaluation JSON declaration.
    pub fn from_json(raw: &str) -> Result<Self, String> {
        let declaration: EvaluationDeclaration = serde_json::from_str(raw)
            .map_err(|error| format!("invalid evaluation configuration: {error}"))?;
        Self::new(
            declaration.covariance_dimension,
            declaration.prediction_fit.into_config()?,
        )
    }

    /// Construct a validated evaluation stage with an explicit covariance dimension.
    pub fn new(
        covariance_dimension: usize,
        prediction_fit: WhitenedFitConfig,
    ) -> Result<Self, String> {
        let config = Self {
            covariance_dimension,
            prediction_fit,
        };
        config.validate()?;
        Ok(config)
    }

    /// Exact named Core-v1 evaluation values; never an implicit workflow default.
    pub fn core_v1() -> Self {
        Self::new(6, d21_fit_config()).expect("Core-v1 evaluation settings are valid")
    }

    /// Return the required coordinate covariance dimension (six for complete OD).
    pub const fn covariance_dimension(self) -> usize {
        self.covariance_dimension
    }

    /// Borrow the settings used to predict and whiten evaluation residuals.
    pub const fn prediction_fit(&self) -> &WhitenedFitConfig {
        &self.prediction_fit
    }

    /// Return the canonical path-free evaluation value.
    pub fn canonical_value(&self) -> Value {
        json!({
            "schema": "adam-core-evaluation-config/v1",
            "covariance_dimension": self.covariance_dimension,
            "prediction_fit": fit_config_value(&self.prediction_fit),
        })
    }

    /// Serialize the canonical evaluation declaration.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.canonical_value())
            .expect("evaluation configuration is serializable")
    }

    /// Validate the six-dimensional covariance contract and prediction settings.
    pub fn validate(&self) -> Result<(), String> {
        if self.covariance_dimension != 6 {
            return Err(format!(
                "complete OD evaluation requires covariance_dimension=6, got {}",
                self.covariance_dimension
            ));
        }
        self.prediction_fit.validate()
    }
}

/// Immutable name, version and definition digest of a built-in OD profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileIdentity {
    /// Stable profile name.
    pub name: &'static str,
    /// Profile-definition version.
    pub version: u32,
    /// Lowercase SHA-256 of the complete path-free profile definition.
    pub definition_sha256: &'static str,
}

fn profile_value(profile: Option<ProfileIdentity>) -> Value {
    profile.map_or(Value::Null, |profile| {
        json!({
            "name": profile.name,
            "version": profile.version,
            "definition_sha256": profile.definition_sha256,
        })
    })
}

fn configuration_sha256(mut value: Value) -> String {
    value
        .as_object_mut()
        .expect("configuration provenance is an object")
        .remove("profile");
    let canonical = serde_json::to_vec(&value).expect("OD configuration is serializable");
    format!("{:x}", Sha256::digest(canonical))
}

fn provenance_json(mut value: Value, fingerprint: String) -> String {
    value
        .as_object_mut()
        .expect("configuration provenance is an object")
        .insert(
            "configuration_sha256".to_string(),
            Value::String(fingerprint),
        );
    serde_json::to_string(&value).expect("OD configuration provenance is serializable")
}

/// Complete bound input for standalone or composed IOD.
#[derive(Debug, Clone)]
pub struct IodRunConfig {
    observation_pipeline: ObservationPipeline,
    stage: IodStageConfig,
    profile: Option<ProfileIdentity>,
}

impl IodRunConfig {
    /// Build an unprofiled IOD run from an already-bound pipeline and stage.
    pub fn explicit(
        observation_pipeline: ObservationPipeline,
        stage: IodStageConfig,
    ) -> Result<Self, OdConfigError> {
        let config = Self {
            observation_pipeline,
            stage,
            profile: None,
        };
        config.validate().map_err(OdConfigError::configuration)?;
        Ok(config)
    }

    fn from_parts(
        observation_pipeline: ObservationPipeline,
        stage: IodStageConfig,
        profile: Option<ProfileIdentity>,
    ) -> Self {
        Self {
            observation_pipeline,
            stage,
            profile,
        }
    }

    /// Revalidate the stage and every already-bound transform.
    pub fn validate(&self) -> Result<(), String> {
        self.stage.validate()?;
        for transform in self.observation_pipeline.bound_transforms() {
            transform.validate()?;
        }
        Ok(())
    }

    /// Borrow the exact ordered observation pipeline applied before IOD.
    pub const fn observation_pipeline(&self) -> &ObservationPipeline {
        &self.observation_pipeline
    }

    /// Borrow the complete IOD-stage declaration.
    pub const fn stage(&self) -> &IodStageConfig {
        &self.stage
    }

    /// Return the immutable built-in profile identity, if this is an exact projection.
    pub const fn profile_identity(&self) -> Option<ProfileIdentity> {
        self.profile
    }

    fn canonical_value_without_fingerprint(&self) -> Value {
        json!({
            "schema": "adam-core-iod-run-config/v1",
            "profile": profile_value(self.profile),
            "transforms": self.observation_pipeline.provenance_value(),
            "iod": self.stage.canonical_value(),
        })
    }

    /// Compute the path-free scientific configuration digest.
    pub fn configuration_sha256(&self) -> String {
        configuration_sha256(self.canonical_value_without_fingerprint())
    }

    /// Serialize complete canonical IOD-run provenance, including its digest.
    pub fn provenance_json(&self) -> String {
        provenance_json(
            self.canonical_value_without_fingerprint(),
            self.configuration_sha256(),
        )
    }
}

/// Complete Core-owned IOD-plus-refinement configuration. There is deliberately no `Default`.
#[derive(Debug, Clone)]
pub struct OdConfig {
    observation_pipeline: ObservationPipeline,
    iod: IodStageConfig,
    refinement: RefinementRunConfig,
    profile: Option<ProfileIdentity>,
}

impl OdConfig {
    /// Build an unprofiled full-OD config from an already-bound pipeline and stages.
    pub fn explicit(
        observation_pipeline: ObservationPipeline,
        iod: IodStageConfig,
        refinement: RefinementRunConfig,
    ) -> Result<Self, OdConfigError> {
        let config = Self {
            observation_pipeline,
            iod,
            refinement,
            profile: None,
        };
        config.validate().map_err(OdConfigError::configuration)?;
        Ok(config)
    }

    /// Revalidate both stages and every already-bound observation transform.
    pub fn validate(&self) -> Result<(), String> {
        self.iod.validate()?;
        self.refinement.validate()?;
        for transform in self.observation_pipeline.bound_transforms() {
            transform.validate()?;
        }
        Ok(())
    }

    /// Borrow the exact ordered pipeline shared by IOD and refinement.
    pub const fn observation_pipeline(&self) -> &ObservationPipeline {
        &self.observation_pipeline
    }

    /// Return the immutable profile identity retained only for an exact built-in definition.
    pub const fn profile_identity(&self) -> Option<ProfileIdentity> {
        self.profile
    }

    /// Borrow the full IOD-stage declaration.
    pub const fn iod_stage(&self) -> &IodStageConfig {
        &self.iod
    }

    /// Project pipeline and IOD state without reopening resources or losing profile identity.
    pub fn for_iod(&self) -> IodRunConfig {
        IodRunConfig::from_parts(
            self.observation_pipeline.clone(),
            self.iod.clone(),
            self.profile,
        )
    }

    /// Project the exact standalone refinement declaration used by `run_od`.
    pub const fn for_refinement(&self) -> RefinementRunConfig {
        self.refinement
    }

    /// Derive six-dimensional evaluation settings from the refinement fit declaration.
    pub fn for_evaluation(&self) -> EvaluationConfig {
        let fit = match *self.refinement.refinement() {
            RefinementConfig::WorstResidual(value) => value.fit,
            RefinementConfig::Cmc2003(value) => value.fit,
        };
        EvaluationConfig::new(6, fit).expect("validated refinement fit is valid for evaluation")
    }

    fn canonical_value_without_fingerprint(&self) -> Value {
        // The fixed six-dimensional covariance contract remains serialized to
        // keep accepted run_od configuration fingerprints stable. It is not a
        // configurable input to IOD or refinement.
        json!({
            "schema": "adam-core-od-config/v1",
            "profile": profile_value(self.profile),
            "transforms": self.observation_pipeline.provenance_value(),
            "iod": self.iod.canonical_value(),
            "refinement": self.refinement.canonical_value(),
            "evaluation": {"covariance_dimension": 6},
        })
    }

    /// SHA-256 of the complete, bound, path-free canonical configuration.
    pub fn configuration_sha256(&self) -> String {
        configuration_sha256(self.canonical_value_without_fingerprint())
    }

    /// Serialize complete canonical full-OD provenance, including its digest.
    pub fn provenance_json(&self) -> String {
        provenance_json(
            self.canonical_value_without_fingerprint(),
            self.configuration_sha256(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdProfileKind {
    Core,
    AsteroidInstitute,
}

/// Immutable built-in OD definition that becomes executable only after resource binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OdProfile {
    identity: ProfileIdentity,
    kind: OdProfileKind,
}

impl OdProfile {
    /// Return the immutable name, version and definition digest.
    pub const fn identity(self) -> ProfileIdentity {
        self.identity
    }

    /// Return the closed profile kind used to expand this definition.
    pub const fn kind(self) -> OdProfileKind {
        self.kind
    }
}

/// Public path-free profile. Selection is always explicit; it is not a
/// function-signature default. The ordered pipeline applies the exact EFCC18
/// correction and cap-four nightly deweighting. It requires finite positive
/// input astrometric variances and does not fill missing uncertainty.
pub const CORE_OD_V1: OdProfile = OdProfile {
    identity: ProfileIdentity {
        name: "CORE_OD_V1",
        version: 1,
        definition_sha256: "fb921274a3daf4bb09885a4247a3f2598b141e355883dc2e2815cfbd5153810b",
    },
    kind: OdProfileKind::Core,
};

/// Institute reference profile. Its ordered pipeline fills missing uncertainty
/// from the pinned station/catalog table, applies EFCC18, adds the pinned
/// station residual covariance for rows supported by at least 30 residuals,
/// then applies cap-four nightly deweighting. It shares Core V1 IOD,
/// refinement, and evaluation settings and never selects a propagator. Public
/// documentation exposes only semantic resource slots; deployment/package
/// details remain outside this crate.
pub const ASTEROID_INSTITUTE_OD_V1: OdProfile = OdProfile {
    identity: ProfileIdentity {
        name: "ASTEROID_INSTITUTE_OD_V1",
        version: 1,
        definition_sha256: "7cf341740994fc918f3d4186fdbe4db75e8860698c99d85407ab12dc448b60c9",
    },
    kind: OdProfileKind::AsteroidInstitute,
};

impl OdProfile {
    fn definitions(self) -> Vec<ObservationTransform> {
        let efcc = ObservationTransform::Efcc18CatalogBiasCorrection {
            resource: EFCC18_BIAS_RESOURCE,
            exclude_astcats: Vec::new(),
        };
        let nightly = ObservationTransform::NightlyBatchDeweighting {
            effective_observation_cap: 4,
        };
        match self.kind {
            OdProfileKind::Core => vec![efcc, nightly],
            OdProfileKind::AsteroidInstitute => vec![
                ObservationTransform::FillMissingAstrometricUncertainty {
                    resource: V2_SIGMA_FILL_RESOURCE,
                },
                efcc,
                ObservationTransform::StationBandResidualCovarianceAdjustment {
                    resource: V2_FULL_RESOURCE,
                    operation: ResidualCovarianceOperation::Add,
                    minimum_residual_count: 30,
                },
                nightly,
            ],
        }
    }

    /// Expand the complete path-free profile definition as canonical JSON.
    pub fn definition_value(self) -> Value {
        let pipeline = self
            .definitions()
            .iter()
            .map(ObservationTransform::definition_value)
            .collect::<Vec<_>>();
        json!({
            "schema": "adam-core-od-profile/v1",
            "name": self.identity.name,
            "version": self.identity.version,
            "pipeline": pipeline,
            "requirements": self.requirements().iter().map(|requirement| requirement.canonical_value()).collect::<Vec<_>>(),
            "iod": IodStageConfig::core_v1().canonical_value(),
            "refinement": RefinementRunConfig::core_v1().canonical_value(),
            "evaluation": {"covariance_dimension": 6},
        })
    }

    /// Verify that the compiled definition matches its immutable declared digest.
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

    /// Return every semantic resource requirement in definition order.
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

    /// Return the exact evaluation declaration associated with this profile.
    pub fn evaluation_config(self) -> EvaluationConfig {
        EvaluationConfig::core_v1()
    }

    /// Validate and parse every resource once, returning a complete profiled config.
    pub fn bind(self, bindings: &ResourceBindings) -> Result<OdConfig, OdConfigError> {
        self.verify_definition_fingerprint()
            .map_err(OdConfigError::configuration)?;
        let pipeline = ObservationPipeline::bind_explicit(self.definitions(), bindings)?;
        let config = OdConfig {
            observation_pipeline: pipeline,
            iod: IodStageConfig::core_v1(),
            refinement: RefinementRunConfig::core_v1(),
            profile: Some(self.identity),
        };
        config.validate().map_err(|reason| {
            OdConfigError::configuration(format!("profile definition is invalid: {reason}"))
        })?;
        Ok(config)
    }
}

#[derive(Debug, Clone)]
struct ValidatedResource {
    identity: ValidatedResourceIdentity,
    bytes: Vec<u8>,
}

fn validate_resources(
    requirements: &[ResourceRequirement],
    bindings: &ResourceBindings,
) -> Result<BTreeMap<ResourceSlot, ValidatedResource>, ResourceValidationError> {
    let mut problems = Vec::new();
    let mut required = BTreeMap::new();
    for &requirement in requirements {
        if let Some(previous) = required.insert(requirement.slot, requirement) {
            if previous != requirement {
                problems.push(ResourceProblem {
                    slot: requirement.slot,
                    reason: "conflicting requirements for the same semantic slot".to_string(),
                });
            }
        }
    }
    for slot in bindings.slots() {
        if !required.contains_key(&slot) {
            problems.push(ResourceProblem {
                slot,
                reason: "a path was bound but this configuration does not use the slot".to_string(),
            });
        }
    }

    let mut validated = BTreeMap::new();
    for &requirement in required.values() {
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
    VeresSigmaLookup::new(&rows, None)
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
        let problems = error.resource_problems();
        assert_eq!(problems.len(), 3);
        assert_eq!(
            problems[0].slot,
            ResourceSlot::StationCatalogUncertaintyTable
        );
        assert_eq!(problems[1].slot, ResourceSlot::Efcc18CatalogBiasTable);
        assert_eq!(
            problems[2].slot,
            ResourceSlot::StationBandResidualCovarianceTable
        );
    }

    #[test]
    #[ignore = "local acceptance check against separately packaged reference resources"]
    fn separately_packaged_reference_resources_parse() {
        let jpl = std::env::var("ADAM_TEST_EFCC18_BIAS_DAT").unwrap();
        let sigma = std::env::var("ADAM_TEST_SIGMA_FILL_PARQUET").unwrap();
        let full = std::env::var("ADAM_TEST_RESIDUAL_COVARIANCE_PARQUET").unwrap();
        let sigma_binding = ResourceBindings::new()
            .bind(ResourceSlot::StationCatalogUncertaintyTable, sigma.clone());
        let bindings = ResourceBindings::new()
            .bind(ResourceSlot::Efcc18CatalogBiasTable, jpl)
            .bind(ResourceSlot::StationCatalogUncertaintyTable, sigma)
            .bind(ResourceSlot::StationBandResidualCovarianceTable, full);
        let config = ASTEROID_INSTITUTE_OD_V1.bind(&bindings).unwrap();
        assert_eq!(config.observation_pipeline.transform_kinds().len(), 4);

        let replace = ObservationPipeline::bind_explicit(
            vec![ObservationTransform::Vfc2017AstrometricUncertaintyReplace {
                resource: V2_SIGMA_FILL_RESOURCE,
            }],
            &sigma_binding,
        )
        .unwrap();
        let reported_variance = (1.0_f64 / 3600.0).powi(2);
        let reported_covariance = 0.9 * reported_variance;
        let mut covariance = vec![f64::NAN; 36];
        covariance[7] = reported_variance;
        covariance[14] = reported_variance;
        covariance[8] = reported_covariance;
        covariance[13] = reported_covariance;
        let mut observations = OrbitDeterminationAstrometry {
            lon: vec![10.0],
            lat: vec![0.0],
            covariance,
            obs_code: vec!["006".to_string()],
            band: vec![None],
            astcat: vec![Some("UCAC4".to_string())],
            mjd_utc: vec![60_000.0],
            jd_tdb: vec![2_460_000.5],
        };
        assert!(replace.apply(&mut observations).unwrap());
        assert_eq!(observations.covariance[8].to_bits(), 0.0_f64.to_bits());
        assert_eq!(observations.covariance[13].to_bits(), 0.0_f64.to_bits());
        let determinant = observations.covariance[7] * observations.covariance[14]
            - observations.covariance[8] * observations.covariance[13];
        assert!(determinant.is_finite() && determinant > 0.0);
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
        let problems = error.resource_problems();
        let slots: std::collections::BTreeSet<_> =
            problems.iter().map(|problem| problem.slot).collect();
        assert_eq!(slots.len(), 3);
        assert!(problems
            .iter()
            .all(|problem| problem.reason.contains("mismatch")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repeated_transform_declarations_preserve_provenance_order() {
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

    #[test]
    fn optional_transform_declarations_are_path_free_and_ordered() {
        let definitions = [
            ObservationTransform::Vfc2017AstrometricUncertaintyFloor {
                resource: V2_SIGMA_FILL_RESOURCE,
                fill_missing: false,
            },
            ObservationTransform::Vfc2017AstrometricUncertaintyFloor {
                resource: V2_SIGMA_FILL_RESOURCE,
                fill_missing: true,
            },
            ObservationTransform::Vfc2017AstrometricUncertaintyReplace {
                resource: V2_SIGMA_FILL_RESOURCE,
            },
            ObservationTransform::PerformanceWeightedCovarianceScaling {
                resource: V2_FULL_RESOURCE,
                minimum_residual_count: 31,
            },
            ObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
                resource: V2_FULL_RESOURCE,
                minimum_residual_count: 47,
            },
        ];
        let values: Vec<Value> = definitions
            .iter()
            .map(ObservationTransform::definition_value)
            .collect();
        assert_eq!(
            values,
            vec![
                json!({"kind": "vfc2017_astrometric_uncertainty_floor", "fill_missing": false}),
                json!({"kind": "vfc2017_astrometric_uncertainty_floor", "fill_missing": true}),
                json!({"kind": "vfc2017_astrometric_uncertainty_replace"}),
                json!({
                    "kind": "performance_weighted_covariance_scaling",
                    "minimum_residual_count": 31
                }),
                json!({
                    "kind": "measured_bias_astrometric_uncertainty_floor",
                    "minimum_residual_count": 47
                }),
            ]
        );
        assert!(!Value::Array(values).to_string().contains("\"path\""));
    }

    #[test]
    fn optional_bound_transforms_execute_the_retained_scientific_models() {
        let identity = |requirement: ResourceRequirement| ValidatedResourceIdentity {
            requirement,
            actual_sha256: requirement.sha256.to_string(),
            actual_byte_size: requirement.byte_size,
        };
        let lookup = VeresSigmaLookup::new(
            &[VeresSigmaRow {
                obs_code: None,
                astcat: None,
                sigma_ra_arcsec: 2.0,
                sigma_dec_arcsec: 3.0,
            }],
            None,
        )
        .unwrap();
        let table = BiasTable::new(
            vec!["500".to_string()],
            vec![None],
            vec![BiasTableRow {
                bias_ra_arcsec: 2.0,
                bias_dec_arcsec: 3.0,
                resid_var_ra: 0.0,
                resid_var_dec: 0.0,
                resid_cov_ra_dec: 0.0,
                resid_cov_n: 50.0,
                chi2_per_obs: 4.0,
            }],
        )
        .unwrap();
        let input = || {
            let mut covariance = vec![f64::NAN; 36];
            covariance[7] = 1.0e-8;
            covariance[14] = 1.0e-8;
            covariance[8] = 1.0e-9;
            covariance[13] = 1.0e-9;
            OrbitDeterminationAstrometry {
                lon: vec![10.0],
                lat: vec![0.0],
                covariance,
                obs_code: vec!["500".to_string()],
                band: vec![None],
                astcat: vec![None],
                mjd_utc: vec![60_000.0],
                jd_tdb: vec![2_460_000.5],
            }
        };

        let floor = BoundObservationTransform::Vfc2017AstrometricUncertaintyFloor {
            lookup: lookup.clone(),
            fill_missing: false,
            resource: identity(V2_SIGMA_FILL_RESOURCE),
        };
        let replace = BoundObservationTransform::Vfc2017AstrometricUncertaintyReplace {
            lookup: lookup.clone(),
            resource: identity(V2_SIGMA_FILL_RESOURCE),
        };
        let provenance = vec![floor.provenance_value(), replace.provenance_value()];
        assert_eq!(provenance[0]["fill_missing"], json!(false));
        assert_eq!(
            provenance[1]["resource"]["actual_sha256"],
            json!(V2_SIGMA_FILL_RESOURCE.sha256)
        );
        assert!(!Value::Array(provenance).to_string().contains("\"path\""));

        let mut floored = input();
        assert!(floor.apply(&mut floored).unwrap());
        assert!(floored.covariance[7] > 1.0e-8);
        assert!(floored.covariance[14] > 1.0e-8);
        assert_eq!(floored.covariance[8].to_bits(), 1.0e-9_f64.to_bits());
        assert_eq!(floored.covariance[13].to_bits(), 1.0e-9_f64.to_bits());

        let mut replaced = input();
        replaced.covariance[8] = 9.0e-9;
        replaced.covariance[13] = 9.0e-9;
        let input_determinant = replaced.covariance[7] * replaced.covariance[14]
            - replaced.covariance[8] * replaced.covariance[13];
        assert!(input_determinant.is_finite() && input_determinant > 0.0);
        let replace_pipeline = ObservationPipeline::ordered(vec![replace]).unwrap();
        assert!(replace_pipeline.apply(&mut replaced).unwrap());
        assert!(replaced.covariance[7].is_finite() && replaced.covariance[7] > 0.0);
        assert!(replaced.covariance[14].is_finite() && replaced.covariance[14] > 0.0);
        assert_eq!(replaced.covariance[8].to_bits(), 0.0_f64.to_bits());
        assert_eq!(replaced.covariance[13].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            replaced.covariance[8].to_bits(),
            replaced.covariance[13].to_bits()
        );
        let determinant = replaced.covariance[7] * replaced.covariance[14]
            - replaced.covariance[8] * replaced.covariance[13];
        assert!(determinant.is_finite() && determinant > 0.0);

        let mut missing_without_fill = input();
        missing_without_fill.covariance[7] = f64::NAN;
        BoundObservationTransform::Vfc2017AstrometricUncertaintyFloor {
            lookup: lookup.clone(),
            fill_missing: false,
            resource: identity(V2_SIGMA_FILL_RESOURCE),
        }
        .apply(&mut missing_without_fill)
        .unwrap();
        assert!(missing_without_fill.covariance[7].is_nan());
        let mut missing_with_fill = input();
        missing_with_fill.covariance[7] = f64::NAN;
        BoundObservationTransform::Vfc2017AstrometricUncertaintyFloor {
            lookup,
            fill_missing: true,
            resource: identity(V2_SIGMA_FILL_RESOURCE),
        }
        .apply(&mut missing_with_fill)
        .unwrap();
        assert!(missing_with_fill.covariance[7].is_finite());
        assert_eq!(
            missing_with_fill.covariance[8].to_bits(),
            1.0e-9_f64.to_bits()
        );

        let mut observations = input();
        BoundObservationTransform::PerformanceWeightedCovarianceScaling {
            table: table.clone(),
            minimum_residual_count: 30,
            resource: identity(V2_FULL_RESOURCE),
        }
        .apply(&mut observations)
        .unwrap();
        assert_eq!(observations.covariance[7].to_bits(), 4.0e-8_f64.to_bits());
        assert_eq!(observations.covariance[8].to_bits(), 4.0e-9_f64.to_bits());

        let mut observations = input();
        BoundObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
            table,
            minimum_residual_count: 30,
            resource: identity(V2_FULL_RESOURCE),
        }
        .apply(&mut observations)
        .unwrap();
        assert!(observations.covariance[7] > 1.0e-8);
        assert!(observations.covariance[14] > 1.0e-8);
        assert_eq!(observations.covariance[8].to_bits(), 1.0e-9_f64.to_bits());
    }

    #[test]
    fn pipeline_executes_non_commuting_transforms_in_declaration_order() {
        let fill = || {
            let lookup = VeresSigmaLookup::new(
                &[VeresSigmaRow {
                    obs_code: None,
                    astcat: None,
                    sigma_ra_arcsec: 1.0,
                    sigma_dec_arcsec: 1.0,
                }],
                None,
            )
            .unwrap();
            BoundObservationTransform::FillMissingAstrometricUncertainty {
                lookup,
                resource: ValidatedResourceIdentity {
                    requirement: V2_SIGMA_FILL_RESOURCE,
                    actual_sha256: V2_SIGMA_FILL_RESOURCE.sha256.to_string(),
                    actual_byte_size: V2_SIGMA_FILL_RESOURCE.byte_size,
                },
            }
        };
        let deweight = || BoundObservationTransform::NightlyBatchDeweighting {
            effective_observation_cap: 1,
        };
        let input = OrbitDeterminationAstrometry {
            lon: vec![10.0; 3],
            lat: vec![0.0; 3],
            covariance: vec![f64::NAN; 3 * 36],
            obs_code: vec!["500".to_string(); 3],
            band: vec![None; 3],
            astcat: vec![None; 3],
            mjd_utc: vec![60_000.3, 60_000.4, 60_000.5],
            jd_tdb: vec![2_460_000.8, 2_460_000.9, 2_460_001.0],
        };
        let mut fill_then_deweight = input.clone();
        ObservationPipeline::ordered(vec![fill(), deweight()])
            .unwrap()
            .apply(&mut fill_then_deweight)
            .unwrap();
        let mut deweight_then_fill = input;
        ObservationPipeline::ordered(vec![deweight(), fill()])
            .unwrap()
            .apply(&mut deweight_then_fill)
            .unwrap();

        let filled_variance = (1.0_f64 / 3600.0).powi(2);
        assert_eq!(
            fill_then_deweight.covariance[7].to_bits(),
            (3.0 * filled_variance).to_bits()
        );
        assert_eq!(
            deweight_then_fill.covariance[7].to_bits(),
            filled_variance.to_bits()
        );
        assert_ne!(
            fill_then_deweight.covariance[7].to_bits(),
            deweight_then_fill.covariance[7].to_bits()
        );
    }

    #[test]
    fn core_v1_preserves_d21_effective_two_body_constants_bitwise() {
        let iod = IodStageConfig::core_v1();
        let refinement = RefinementRunConfig::core_v1();
        let refinement_fit = match refinement.refinement() {
            RefinementConfig::WorstResidual(value) => value.fit,
            RefinementConfig::Cmc2003(value) => value.fit,
        };
        for fit in [*iod.correction_fit(), refinement_fit] {
            assert_eq!(fit.two_body.lt_tol.to_bits(), 1.0e-12_f64.to_bits());
            assert_eq!(fit.two_body.ephemeris_max_iter, 1_000);
            assert_eq!(fit.two_body.max_lt_iter, 10);
        }
    }

    #[test]
    fn stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields() {
        let iod = IodStageConfig::core_v1();
        let raw = iod.canonical_json();
        let round_trip = IodStageConfig::from_json(&raw).unwrap();
        assert_eq!(round_trip.canonical_json(), raw);
        let mut value: Value = serde_json::from_str(&raw).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("bogus".to_string(), json!(1));
        assert!(IodStageConfig::from_json(&value.to_string()).is_err());
        let mut value: Value = serde_json::from_str(&raw).unwrap();
        value
            .get_mut("search_correction_ranging")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("min_obs");
        assert!(IodStageConfig::from_json(&value.to_string()).is_err());

        let refinement = RefinementRunConfig::core_v1();
        let raw = refinement.canonical_json();
        assert_eq!(
            RefinementRunConfig::from_json(&raw)
                .unwrap()
                .canonical_json(),
            raw
        );
    }

    #[test]
    fn stage_json_preserves_known_non_roundtripping_floats_bitwise() {
        let mut iod: Value =
            serde_json::from_str(&IodStageConfig::core_v1().canonical_json()).unwrap();
        iod["search_correction_ranging"]["ranging"]["rho_min_au"] =
            json!(1.8135371537546276e-10_f64);
        let iod = IodStageConfig::from_json(&iod.to_string()).unwrap();
        let iod_value: Value = serde_json::from_str(&iod.canonical_json()).unwrap();
        assert_eq!(
            iod_value["search_correction_ranging"]["ranging"]["rho_min_au"]
                .as_f64()
                .unwrap()
                .to_bits(),
            1.8135371537546276e-10_f64.to_bits()
        );

        let mut refinement: Value =
            serde_json::from_str(&RefinementRunConfig::core_v1().canonical_json()).unwrap();
        refinement["psd_floor_frac"] = json!(1e-30_f64);
        let refinement = RefinementRunConfig::from_json(&refinement.to_string()).unwrap();
        let refinement_value: Value = serde_json::from_str(&refinement.canonical_json()).unwrap();
        assert_eq!(
            refinement_value["psd_floor_frac"]
                .as_f64()
                .unwrap()
                .to_bits(),
            1e-30_f64.to_bits()
        );
    }

    #[test]
    fn invalid_transform_is_configuration_error_not_resource_error() {
        let error = ObservationPipeline::bind_explicit(
            vec![ObservationTransform::NightlyBatchDeweighting {
                effective_observation_cap: 0,
            }],
            &ResourceBindings::new(),
        )
        .unwrap_err();
        assert!(!error.configuration_problems().is_empty());
        assert!(error.resource_problems().is_empty());
    }

    #[test]
    fn unused_binding_is_a_slot_correct_resource_error() {
        let bindings = ResourceBindings::new().bind(
            ResourceSlot::Efcc18CatalogBiasTable,
            "/path/that/must/not/be-read",
        );
        let error = ObservationPipeline::bind_explicit(
            vec![ObservationTransform::NightlyBatchDeweighting {
                effective_observation_cap: 4,
            }],
            &bindings,
        )
        .unwrap_err();
        assert_eq!(error.resource_problems().len(), 1);
        assert_eq!(
            error.resource_problems()[0].slot,
            ResourceSlot::Efcc18CatalogBiasTable
        );
        assert!(error.resource_problems()[0].reason.contains("does not use"));
    }

    #[test]
    fn complete_config_fingerprint_is_deterministic_and_path_free() {
        let first = OdConfig::explicit(
            ObservationPipeline::as_supplied(),
            IodStageConfig::core_v1(),
            RefinementRunConfig::core_v1(),
        )
        .unwrap();
        let second = first.clone();
        assert_eq!(first.configuration_sha256(), second.configuration_sha256());
        let provenance = first.provenance_json();
        assert!(!provenance.contains("/Users/"));
        assert!(!provenance.contains("/tmp/"));
        assert!(provenance.contains(&first.configuration_sha256()));
    }
}
