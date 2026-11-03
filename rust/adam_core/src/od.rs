//! Canonical, batch-native orbit-determination workflows.
//!
//! This module owns grouping, preparation, scientific-failure continuation,
//! output disposition, fingerprints, and provenance for both standalone Rust
//! callers and the Python bindings.  Predictor-level numerical kernels remain
//! available under [`crate::od::algorithms`], but are not alternate batch APIs.
//!
//! # Example
//!
//! The four workflows are generic only over the propagation-only origin
//! translation service; their scientific inputs and outputs remain concrete
//! typed Rust values.
//!
//! ```no_run
//! use adam_core::{evaluate_orbits, refine, run_iod, run_od};
//! use adam_core::coords::translation::OriginTranslationProvider;
//!
//! fn bind_workflows<T: OriginTranslationProvider>() {
//!     let _ = run_iod::<T>;
//!     let _ = run_od::<T>;
//!     let _ = refine::<T>;
//!     let _ = evaluate_orbits::<T>;
//! }
//! ```

use adam_core_rs_coords::od::algorithms::{
    prepare_observations, refine_with_predictor, run_iod_with_predictor, PreparedObservations,
};
use adam_core_rs_coords::propagation::adapter::{refine_with, SphericalPredictor};
use adam_core_rs_coords::propagation::{
    evaluate_orbit, AstrometrySnapshot, FitEvaluation, GroupScientificFailureCode, IodResult,
    PropagationError, RefinedOutput,
};
use adam_core_rs_coords::translation::OriginTranslationProvider;
use adam_core_rs_coords::types::time::TimeScaleProvider;
use adam_core_rs_coords::{
    AssociationBatch, CoordinateBatch, CoordinateValues, CovarianceBatch, ObserverBatch,
    OrbitBatch, SchemaError, SourceCatalogBatch,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fmt;

/// Predictor-level numerical building blocks used by the canonical workflows.
pub mod algorithms {
    pub use adam_core_rs_coords::od::algorithms::{
        prepare_observations, refine_with_predictor, run_iod_with_predictor,
        run_od_with_predictors, ConfiguredRunOdOutput, PreparedObservations,
    };
}

/// Immutable complete OD profiles and their resource declarations.
pub mod profiles {
    pub use adam_core_rs_coords::od::profiles::*;
}

pub use adam_core_rs_coords::od::{
    plan_associations, plan_orbit_members, AssociationGroup, EvaluationConfig, IodRunConfig,
    ObservationPipeline, ObservationTransform, OdConfig, OdConfigError, OrbitMemberGroup,
    RefinementRunConfig, ResidualCovarianceOperation, ResourceBindings, ResourceProblem,
    ResourceRequirement, ResourceSlot, ResourceValidationError, ValidatedResourceIdentity,
};

/// Stable classification for invocation-fatal batch errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdBatchErrorCode {
    /// A decoded typed batch is internally inconsistent.
    InvalidInput,
    /// The explicit stage configuration is invalid.
    InvalidConfiguration,
    /// The Core source/build identity is unavailable or malformed.
    InvalidBuildIdentity,
    /// The prediction service identity or fingerprint is incomplete.
    InvalidProviderIdentity,
    /// Persisted orbit/member provenance is missing, malformed, or inconsistent.
    InvalidProvenance,
    /// A propagation, resource, protocol, or other invocation-fatal service error.
    Propagation,
}

impl OdBatchErrorCode {
    /// Stable snake-case representation used by foreign-language bindings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::InvalidConfiguration => "invalid_configuration",
            Self::InvalidBuildIdentity => "invalid_build_identity",
            Self::InvalidProviderIdentity => "invalid_provider_identity",
            Self::InvalidProvenance => "invalid_provenance",
            Self::Propagation => "propagation",
        }
    }
}

/// Invocation-fatal error from a canonical OD batch workflow.
#[derive(Debug)]
pub struct OdBatchError {
    /// Stable machine-readable classification.
    pub code: OdBatchErrorCode,
    /// Human-readable detail suitable for logs and Python exceptions.
    pub message: String,
    /// Typed propagation error, when the failure originated below the batch layer.
    pub source: Option<PropagationError>,
}

impl OdBatchError {
    fn new(code: OdBatchErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    fn propagation(source: PropagationError) -> Self {
        Self {
            code: OdBatchErrorCode::Propagation,
            message: source.to_string(),
            source: Some(source),
        }
    }
}

impl fmt::Display for OdBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for OdBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|error| error as &(dyn std::error::Error + 'static))
    }
}

impl From<PropagationError> for OdBatchError {
    fn from(value: PropagationError) -> Self {
        Self::propagation(value)
    }
}

impl From<SchemaError> for OdBatchError {
    fn from(value: SchemaError) -> Self {
        Self::propagation(value.into())
    }
}

/// Exact Core build identity recorded in every fitted-orbit provenance record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreBuildIdentity {
    /// adam-core semantic version.
    pub version: String,
    /// Exact lowercase 40-hex Git commit.
    pub commit: String,
}

impl CoreBuildIdentity {
    /// Construct and fail closed on an inexact source identity.
    pub fn new(
        version: impl Into<String>,
        commit: impl Into<String>,
    ) -> Result<Self, OdBatchError> {
        let identity = Self {
            version: version.into(),
            commit: commit.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Validate the exact version/commit pair.
    pub fn validate(&self) -> Result<(), OdBatchError> {
        if self.version.trim().is_empty() {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidBuildIdentity,
                "canonical OD requires a non-empty adam-core version",
            ));
        }
        if !is_git_commit(&self.commit) {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidBuildIdentity,
                "canonical OD requires an exact 40-character lowercase hexadecimal build commit",
            ));
        }
        Ok(())
    }
}

/// Complete, content-pinned identity of one prediction service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderMetadata {
    pub name: String,
    pub version: String,
    pub abi_version: u32,
    pub identity_validated_by_capsule: bool,
    pub capabilities: u64,
    pub known_capabilities: u64,
    pub integration_time_scale: String,
    pub two_body_dynamics: bool,
    pub configuration_fingerprint: String,
    pub data_fingerprint: String,
}

impl ProviderMetadata {
    /// Validate the complete provider identity before any group is executed.
    pub fn validate(&self) -> Result<(), OdBatchError> {
        for (field, value) in [
            ("name", self.name.as_str()),
            ("version", self.version.as_str()),
            (
                "integration_time_scale",
                self.integration_time_scale.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidProviderIdentity,
                    format!("prediction service has no {field}"),
                ));
            }
        }
        for (field, value) in [
            (
                "configuration_fingerprint",
                self.configuration_fingerprint.as_str(),
            ),
            ("data_fingerprint", self.data_fingerprint.as_str()),
        ] {
            if !is_sha256(value) {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidProviderIdentity,
                    format!("prediction service {field} is not a lowercase SHA-256"),
                ));
            }
        }
        if self.capabilities & !self.known_capabilities != 0 {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProviderIdentity,
                "prediction service advertises unknown capabilities",
            ));
        }
        Ok(())
    }

    /// Canonical provider representation persisted by the Python facade.
    pub fn canonical_json(&self) -> String {
        json!({
            "name": self.name,
            "version": self.version,
            "abi_version": self.abi_version,
            "identity_validated_by_capsule": self.identity_validated_by_capsule,
            "capabilities": self.capabilities,
            "known_capabilities": self.known_capabilities,
            "integration_time_scale": self.integration_time_scale,
            "two_body_dynamics": self.two_body_dynamics,
            "configuration_fingerprint": self.configuration_fingerprint,
            "data_fingerprint": self.data_fingerprint,
        })
        .to_string()
    }

    /// Human-readable backend identity used in provenance.
    pub fn backend_label(&self) -> String {
        format!("{} {}", self.name, self.version)
    }
}

/// One prediction stage and its complete identity.
pub struct PredictionService<'a> {
    /// Predictor implementation. It is propagation-only; no fit callback exists.
    pub predictor: &'a dyn SphericalPredictor,
    /// Fail-closed provider identity.
    pub metadata: ProviderMetadata,
}

/// Shared non-prediction services and build identity for a batch invocation.
pub struct OdExecutionContext<'a, T: OriginTranslationProvider> {
    pub build: &'a CoreBuildIdentity,
    pub time_provider: &'a dyn TimeScaleProvider,
    pub translation_provider: &'a T,
}

/// Typed observation/source input shared by all four workflows.
#[derive(Debug, Clone)]
pub struct OdObservationBatch {
    pub source: SourceCatalogBatch,
    pub observed: CoordinateBatch,
    pub observers: ObserverBatch,
}

/// Typed association input to [`run_iod`] and [`run_od`].
#[derive(Debug, Clone)]
pub struct AssociatedObservations {
    pub observations: OdObservationBatch,
    pub associations: AssociationBatch,
}

/// Typed nullable astrometry columns persisted for explicit fitted members.
#[derive(Debug, Clone, PartialEq)]
pub struct AstrometryColumns {
    pub lon: Vec<Option<f64>>,
    pub lat: Vec<Option<f64>>,
    pub sigma_lon: Vec<Option<f64>>,
    pub sigma_lat: Vec<Option<f64>>,
    pub variance_lon: Vec<Option<f64>>,
    pub variance_lat: Vec<Option<f64>>,
    pub cov_lonlat: Vec<Option<f64>>,
}

impl AstrometryColumns {
    fn lengths(&self) -> [(&'static str, usize); 7] {
        [
            ("lon", self.lon.len()),
            ("lat", self.lat.len()),
            ("sigma_lon", self.sigma_lon.len()),
            ("sigma_lat", self.sigma_lat.len()),
            ("variance_lon", self.variance_lon.len()),
            ("variance_lat", self.variance_lat.len()),
            ("cov_lonlat", self.cov_lonlat.len()),
        ]
    }
}

/// Explicit fitted-orbit membership input to [`refine`] and [`evaluate_orbits`].
#[derive(Debug, Clone)]
pub struct FittedOrbitMembersBatch {
    pub orbit_id: Vec<String>,
    pub observation_id: Vec<String>,
    pub outlier: Vec<Option<bool>>,
    pub input_fingerprint: Vec<Option<String>>,
    pub membership_fingerprint: Vec<Option<String>>,
    pub original_astrometry: AstrometryColumns,
    pub used_astrometry: AstrometryColumns,
}

impl FittedOrbitMembersBatch {
    /// Number of explicit membership rows.
    pub fn len(&self) -> usize {
        self.orbit_id.len()
    }

    /// Whether no memberships were supplied.
    pub fn is_empty(&self) -> bool {
        self.orbit_id.is_empty()
    }

    fn validate_lengths(&self, require_outlier: bool) -> Result<(), OdBatchError> {
        let expected = self.len();
        for (name, actual) in [
            ("obs_id", self.observation_id.len()),
            ("input_fingerprint", self.input_fingerprint.len()),
            ("membership_fingerprint", self.membership_fingerprint.len()),
        ]
        .into_iter()
        .chain(self.original_astrometry.lengths())
        .chain(self.used_astrometry.lengths())
        {
            if actual != expected {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidInput,
                    format!(
                        "FittedOrbitMembers {name} length {actual} does not match {expected} rows"
                    ),
                ));
            }
        }
        if require_outlier && self.outlier.len() != expected {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                format!(
                    "FittedOrbitMembers outlier length {} does not match {expected} rows",
                    self.outlier.len()
                ),
            ));
        }
        Ok(())
    }
}

/// Typed orbit/member input to [`refine`] and [`evaluate_orbits`].
#[derive(Debug, Clone)]
pub struct MemberedOrbits {
    pub orbits: OrbitBatch,
    pub observations: OdObservationBatch,
    pub members: FittedOrbitMembersBatch,
    /// One persisted, validated provenance value per input orbit.
    pub provenance: Vec<Option<OdProvenance>>,
}

/// Canonical workflow route recorded in provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdRoute {
    RunIod,
    RunOd,
    Refine,
    EvaluateOrbits,
}

impl OdRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RunIod => "run_iod",
            Self::RunOd => "run_od",
            Self::Refine => "refine",
            Self::EvaluateOrbits => "evaluate_orbits",
        }
    }
}

/// Validated path-free canonical configuration identity.
#[derive(Debug, Clone)]
pub struct CanonicalConfiguration {
    value: Value,
    science_sha256: String,
    complete_sha256: String,
}

impl CanonicalConfiguration {
    /// Construct from a complete OD configuration.
    pub fn from_od(config: &OdConfig) -> Result<Self, OdBatchError> {
        config.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        Self::from_value(
            serde_json::from_str(&config.provenance_json()).expect("OdConfig JSON is valid"),
        )
    }

    /// Construct from a standalone refinement-stage configuration.
    pub fn from_refinement(config: &RefinementRunConfig) -> Result<Self, OdBatchError> {
        config.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        Self::from_value(config.canonical_value())
    }

    /// Validate a canonical configuration value decoded by a foreign-language binding.
    pub fn from_value(mut value: Value) -> Result<Self, OdBatchError> {
        let complete = serde_json::to_vec(&value).map_err(|error| {
            OdBatchError::new(
                OdBatchErrorCode::InvalidConfiguration,
                format!("configuration is not serializable: {error}"),
            )
        })?;
        let complete_sha256 = sha256(&complete);
        let declared = value
            .get("configuration_sha256")
            .and_then(Value::as_str)
            .map(str::to_string);
        let science_sha256 = if let Some(declared) = declared {
            if !is_sha256(&declared) {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidConfiguration,
                    "configuration_sha256 is not a lowercase SHA-256",
                ));
            }
            let object = value.as_object_mut().ok_or_else(|| {
                OdBatchError::new(
                    OdBatchErrorCode::InvalidConfiguration,
                    "configuration provenance is not an object",
                )
            })?;
            object.remove("configuration_sha256");
            object.remove("profile");
            let actual = sha256(&serde_json::to_vec(&value).map_err(|error| {
                OdBatchError::new(
                    OdBatchErrorCode::InvalidConfiguration,
                    format!("configuration is not serializable: {error}"),
                )
            })?);
            if actual != declared {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidConfiguration,
                    "configuration_sha256 does not match canonical configuration",
                ));
            }
            // Restore the original, complete provenance representation.
            let mut complete_value: Value = serde_json::from_slice(&complete)
                .expect("previously serialized configuration remains valid");
            value = std::mem::take(&mut complete_value);
            declared
        } else {
            complete_sha256.clone()
        };
        Ok(Self {
            value,
            science_sha256,
            complete_sha256,
        })
    }

    pub fn science_sha256(&self) -> &str {
        &self.science_sha256
    }

    pub fn complete_sha256(&self) -> &str {
        &self.complete_sha256
    }

    /// Canonical JSON used only at serialization boundaries.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.value).expect("validated configuration is serializable")
    }

    fn refinement_value(&self) -> Option<&Value> {
        self.value.get("refinement")
    }
}

/// Typed IOD diagnostic provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IodDiagnostics {
    pub triplets_tried: usize,
    pub candidates_evaluated: usize,
    pub candidates_failed: usize,
    pub corrections_run: usize,
    pub search_cap_exhausted: bool,
}

impl IodDiagnostics {
    pub fn canonical_json(&self) -> String {
        json!({
            "triplets_tried": self.triplets_tried,
            "candidates_evaluated": self.candidates_evaluated,
            "candidates_failed": self.candidates_failed,
            "corrections_run": self.corrections_run,
            "search_cap_exhausted": self.search_cap_exhausted,
        })
        .to_string()
    }
}

/// Typed refinement diagnostic provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementDiagnostics {
    pub passes: usize,
    pub n_rejected: usize,
    pub n_recovered: usize,
    pub flags: Vec<String>,
}

impl RefinementDiagnostics {
    pub fn canonical_json(&self) -> String {
        json!({
            "passes": self.passes,
            "n_rejected": self.n_rejected,
            "n_recovered": self.n_recovered,
            "flags": self.flags,
        })
        .to_string()
    }
}

/// Typed evaluation diagnostic provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationDiagnostics {
    pub num_obs: usize,
    pub arc_length: f64,
    pub chi2: f64,
    pub reduced_chi2: f64,
}

impl EvaluationDiagnostics {
    pub fn canonical_json(&self) -> String {
        json!({
            "num_obs": self.num_obs,
            "arc_length": self.arc_length,
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
        })
        .to_string()
    }
}

/// Complete Rust-owned fitted-orbit provenance.
#[derive(Debug, Clone)]
pub struct OdProvenance {
    pub adam_core: CoreBuildIdentity,
    pub input_fingerprint: String,
    pub state_fingerprint: Option<String>,
    pub provenance_fingerprint: String,
    pub route: OdRoute,
    pub iod_provider: Option<ProviderMetadata>,
    pub refinement_provider: Option<ProviderMetadata>,
    pub evaluation_provider: Option<ProviderMetadata>,
    pub iod_diagnostics: Option<IodDiagnostics>,
    pub refinement_diagnostics: Option<RefinementDiagnostics>,
    pub evaluation_diagnostics: Option<EvaluationDiagnostics>,
    pub configuration: CanonicalConfiguration,
    pub iod_ambiguous: Option<bool>,
    pub iod_alternatives: Option<usize>,
    pub num_rejected: Option<usize>,
    pub rejected_fraction: Option<f64>,
}

impl OdProvenance {
    pub fn iod_backend(&self) -> Option<String> {
        self.iod_provider
            .as_ref()
            .map(ProviderMetadata::backend_label)
    }

    pub fn refinement_backend(&self) -> Option<String> {
        self.refinement_provider
            .as_ref()
            .map(ProviderMetadata::backend_label)
    }

    pub fn evaluation_backend(&self) -> Option<String> {
        self.evaluation_provider
            .as_ref()
            .map(ProviderMetadata::backend_label)
    }

    /// Revalidate all identity fields after a persistence round trip.
    pub fn validate(&self) -> Result<(), OdBatchError> {
        self.adam_core.validate()?;
        if !is_sha256(&self.input_fingerprint)
            || self
                .state_fingerprint
                .as_deref()
                .is_some_and(|value| !is_sha256(value))
            || !is_sha256(&self.provenance_fingerprint)
        {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                "orbit provenance contains an invalid SHA-256 fingerprint",
            ));
        }
        if self.state_fingerprint.is_none() {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                "fitted-orbit provenance has no state fingerprint",
            ));
        }
        for provider in [
            self.iod_provider.as_ref(),
            self.refinement_provider.as_ref(),
            self.evaluation_provider.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            provider.validate()?;
        }
        let iod = self.iod_provider.is_some() && self.iod_diagnostics.is_some();
        let refinement =
            self.refinement_provider.is_some() && self.refinement_diagnostics.is_some();
        let evaluation =
            self.evaluation_provider.is_some() && self.evaluation_diagnostics.is_some();
        let complete = match self.route {
            OdRoute::RunIod => iod && !refinement && !evaluation,
            OdRoute::RunOd => iod && refinement && !evaluation,
            OdRoute::Refine => !iod && refinement && !evaluation,
            OdRoute::EvaluateOrbits => !iod && !refinement && evaluation,
        };
        let full_configuration = self.configuration.value.get("iod").is_some()
            && self.configuration.value.get("refinement").is_some()
            && self.configuration.value.get("evaluation").is_some();
        let configuration_matches_route = match self.route {
            OdRoute::Refine => {
                !full_configuration && self.configuration.value.get("method").is_some()
            }
            OdRoute::RunIod | OdRoute::RunOd | OdRoute::EvaluateOrbits => full_configuration,
        };
        let no_half_stage = self.iod_provider.is_some() == self.iod_diagnostics.is_some()
            && self.refinement_provider.is_some() == self.refinement_diagnostics.is_some()
            && self.evaluation_provider.is_some() == self.evaluation_diagnostics.is_some();
        let iod_counts =
            self.iod_ambiguous.is_some() == iod && self.iod_alternatives.is_some() == iod;
        let rejection_counts = self.num_rejected.is_some() == refinement
            && self.rejected_fraction.is_some() == refinement;
        let finite_counts = self
            .rejected_fraction
            .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            && self.evaluation_diagnostics.as_ref().is_none_or(|value| {
                value.arc_length.is_finite()
                    && value.chi2.is_finite()
                    && value.reduced_chi2.is_finite()
            });
        if !complete
            || !configuration_matches_route
            || !no_half_stage
            || !iod_counts
            || !rejection_counts
            || !finite_counts
        {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                "orbit provenance workflow route is inconsistent with its consistency digest",
            ));
        }
        let expected = provenance_consistency_digest(self);
        if expected != self.provenance_fingerprint {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                "orbit provenance does not match its consistency digest",
            ));
        }
        Ok(())
    }
}

/// Stable group-local scientific/fitting failure classifications.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdGroupFailureCode {
    NoSolution,
    NotConverged,
    Scientific(GroupScientificFailureCode),
}

impl OdGroupFailureCode {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NoSolution => "no_solution",
            Self::NotConverged => "not_converged",
            Self::Scientific(code) => code.as_str(),
        }
    }
}

/// Typed per-group failure. These values continue the surrounding batch.
#[derive(Debug, Clone)]
pub struct OrbitDeterminationFailure {
    pub group_id: String,
    pub orbit_id: Option<String>,
    pub input_orbit_id: Option<String>,
    pub stage: &'static str,
    pub code: OdGroupFailureCode,
    pub message: String,
    pub num_obs: usize,
    pub arc_length: Option<f64>,
    pub triplets_tried: Option<usize>,
    pub candidates_evaluated: Option<usize>,
    pub candidates_failed: Option<usize>,
    pub corrections_run: Option<usize>,
    pub passes: Option<usize>,
    pub n_rejected: Option<usize>,
    pub n_recovered: Option<usize>,
    pub status_code: Option<i64>,
    pub best_state_retained: bool,
}

/// Rust-owned fitted-orbit record for one successful group disposition.
#[derive(Debug, Clone)]
pub struct FittedOrbitRecord {
    pub group_id: String,
    pub orbit_id: String,
    pub input_orbit_index: Option<usize>,
    pub iod: Option<IodResult>,
    pub refined: Option<RefinedOutput>,
    pub evaluation: Option<FitEvaluation>,
    pub provenance: OdProvenance,
}

/// Rust-owned explicit fitted-orbit membership record.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedOrbitMemberRecord {
    pub orbit_id: String,
    pub observation_id: String,
    pub source_index: usize,
    pub input_member_index: Option<usize>,
    pub input_fingerprint: String,
    pub membership_fingerprint: String,
    pub original: [f64; 7],
    pub used: [f64; 7],
    pub outlier: bool,
}

/// One deterministic group disposition in a canonical batch result.
#[derive(Debug, Clone)]
pub struct OdGroupResult {
    pub group_id: String,
    pub orbit_id: String,
    pub row_indices: Vec<usize>,
    pub input_orbit_index: Option<usize>,
    pub member_indices: Vec<usize>,
    pub prepared: PreparedObservations,
    pub orbit: Option<FittedOrbitRecord>,
    pub members: Vec<FittedOrbitMemberRecord>,
    pub failure: Option<OrbitDeterminationFailure>,
    /// IOD diagnostics are retained even when no fitted orbit exists.
    pub iod: Option<IodResult>,
}

/// Provider identities used by a canonical batch invocation.
#[derive(Debug, Clone, Default)]
pub struct OdStageProviders {
    pub iod: Option<ProviderMetadata>,
    pub refinement: Option<ProviderMetadata>,
    pub evaluation: Option<ProviderMetadata>,
}

/// Exact three-concept batch product: fitted orbits, explicit memberships, and
/// typed group failures, retained in deterministic group order.
#[derive(Debug, Clone, Default)]
pub struct OdBatchResult {
    pub groups: Vec<OdGroupResult>,
    pub providers: OdStageProviders,
}

impl OdBatchResult {
    /// Fitted-orbit records in deterministic group order.
    pub fn fitted_orbits(&self) -> impl Iterator<Item = &FittedOrbitRecord> {
        self.groups.iter().filter_map(|group| group.orbit.as_ref())
    }

    /// Explicit membership records in deterministic group/source-row order.
    pub fn fitted_orbit_members(&self) -> impl Iterator<Item = &FittedOrbitMemberRecord> {
        self.groups.iter().flat_map(|group| group.members.iter())
    }

    /// Typed group failures in deterministic group order.
    pub fn failures(&self) -> impl Iterator<Item = &OrbitDeterminationFailure> {
        self.groups
            .iter()
            .filter_map(|group| group.failure.as_ref())
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

fn is_git_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn fingerprint_string(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn fingerprint_optional_string(digest: &mut Sha256, value: Option<&str>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        fingerprint_string(digest, value);
    }
}

fn fingerprint_optional_f64(digest: &mut Sha256, value: Option<f64>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(value.to_bits().to_le_bytes());
    }
}

fn validate_problem(input: &OdObservationBatch) -> Result<(), OdBatchError> {
    let source = &input.source;
    source
        .validate()
        .map_err(|error| OdBatchError::new(OdBatchErrorCode::InvalidInput, error.to_string()))?;
    input
        .observed
        .validate()
        .map_err(|error| OdBatchError::new(OdBatchErrorCode::InvalidInput, error.to_string()))?;
    input
        .observers
        .validate()
        .map_err(|error| OdBatchError::new(OdBatchErrorCode::InvalidInput, error.to_string()))?;
    if source.len() != input.observed.len() || source.len() != input.observers.len() {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            format!(
                "SourceCatalog, observed coordinates, and observers must have equal length ({} != {} != {})",
                source.len(),
                input.observed.len(),
                input.observers.len()
            ),
        ));
    }
    for (row, (expected, actual)) in source
        .observatory_code
        .iter()
        .zip(input.observers.code.iter())
        .enumerate()
    {
        if expected != &actual.0 {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                format!(
                    "observer code mismatch at row {row}: SourceCatalog {expected:?}, observers {:?}",
                    actual.0
                ),
            ));
        }
    }
    let observed_times = input.observed.times.as_ref().ok_or_else(|| {
        OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "orbit-determination observed coordinates require time",
        )
    })?;
    let observer_times = input.observers.coordinates.times.as_ref().ok_or_else(|| {
        OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "orbit-determination observer coordinates require time",
        )
    })?;
    for (label, scale) in [
        ("observed", observed_times.scale),
        ("observer", observer_times.scale),
    ] {
        if scale != source.time.scale {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                format!(
                    "SourceCatalog/{label} time-scale mismatch: {} != {}",
                    source.time.scale.as_str(),
                    scale.as_str()
                ),
            ));
        }
    }
    for row in 0..source.len() {
        for (label, epoch) in [
            ("observed", observed_times.epochs[row]),
            ("observer", observer_times.epochs[row]),
        ] {
            if epoch.days != source.time.days[row] || epoch.nanos != source.time.nanos[row] {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidInput,
                    format!("SourceCatalog/{label} epoch mismatch at row {row}"),
                ));
            }
        }
    }
    Ok(())
}

fn member_input_fingerprint(
    source: &SourceCatalogBatch,
    observers: &ObserverBatch,
    source_row: usize,
    original: &AstrometrySnapshot,
    used: &AstrometrySnapshot,
    snapshot_row: usize,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-member-input/v1\0");
    fingerprint_string(&mut digest, &source.id[source_row]);
    fingerprint_optional_string(&mut digest, source.exposure_id[source_row].as_deref());
    fingerprint_string(&mut digest, source.time.scale.as_str());
    digest.update(source.time.days[source_row].to_le_bytes());
    digest.update(source.time.nanos[source_row].to_le_bytes());
    for value in [source.ra[source_row], source.dec[source_row]] {
        digest.update(value.to_bits().to_le_bytes());
    }
    for value in [
        source.ra_sigma[source_row],
        source.dec_sigma[source_row],
        source.radec_corr[source_row],
        source.mag[source_row],
        source.mag_sigma[source_row],
        source.fwhm[source_row],
        source.a[source_row],
        source.a_sigma[source_row],
        source.b[source_row],
        source.b_sigma[source_row],
        source.pa[source_row],
        source.pa_sigma[source_row],
        source.exposure_duration[source_row],
        source.exposure_seeing[source_row],
        source.exposure_depth_5sigma[source_row],
    ] {
        fingerprint_optional_f64(&mut digest, value);
    }
    fingerprint_string(&mut digest, &source.observatory_code[source_row]);
    fingerprint_optional_string(&mut digest, source.filter[source_row].as_deref());
    let exposure_start_valid = source
        .exposure_start_time
        .validity
        .as_ref()
        .is_none_or(|validity| validity[source_row]);
    digest.update([u8::from(exposure_start_valid)]);
    if exposure_start_valid {
        fingerprint_string(&mut digest, source.exposure_start_time.scale.as_str());
        digest.update(source.exposure_start_time.days[source_row].to_le_bytes());
        digest.update(source.exposure_start_time.nanos[source_row].to_le_bytes());
    }
    fingerprint_optional_string(&mut digest, source.object_id[source_row].as_deref());
    fingerprint_string(&mut digest, &source.catalog_id[source_row]);
    fingerprint_optional_string(&mut digest, source.astcat[source_row].as_deref());
    fingerprint_optional_string(&mut digest, source.photcat[source_row].as_deref());
    fingerprint_string(&mut digest, &observers.code[source_row].0);
    fingerprint_string(&mut digest, observers.coordinates.frame.as_str());
    fingerprint_string(
        &mut digest,
        &observers.coordinates.origins.origins[source_row].code(),
    );
    for value in observers.coordinates.values.raw_values()[source_row] {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update([u8::from(observers.coordinates.times.is_some())]);
    if let Some(times) = &observers.coordinates.times {
        fingerprint_string(&mut digest, times.scale.as_str());
        digest.update(times.epochs[source_row].days.to_le_bytes());
        digest.update(times.epochs[source_row].nanos.to_le_bytes());
    }
    digest.update([u8::from(observers.coordinates.covariance.is_some())]);
    if let Some(covariance) = &observers.coordinates.covariance {
        digest.update((covariance.row_dimension(source_row) as u64).to_le_bytes());
        digest.update([u8::from(covariance.is_row_valid(source_row))]);
        for value in covariance.row_values(source_row) {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    for value in [
        original.lon[snapshot_row],
        original.lat[snapshot_row],
        original.sigma_lon[snapshot_row],
        original.sigma_lat[snapshot_row],
        original.variance_lon[snapshot_row],
        original.variance_lat[snapshot_row],
        original.cov_lonlat[snapshot_row],
        used.lon[snapshot_row],
        used.lat[snapshot_row],
        used.sigma_lon[snapshot_row],
        used.sigma_lat[snapshot_row],
        used.variance_lon[snapshot_row],
        used.variance_lat[snapshot_row],
        used.cov_lonlat[snapshot_row],
    ] {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn member_input_fingerprints(
    source: &SourceCatalogBatch,
    observers: &ObserverBatch,
    rows: &[usize],
    original: &AstrometrySnapshot,
    used: &AstrometrySnapshot,
) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(snapshot_row, &source_row)| {
            member_input_fingerprint(source, observers, source_row, original, used, snapshot_row)
        })
        .collect()
}

fn membership_fingerprint(
    group_id: &str,
    orbit_id: &str,
    observation_id: &str,
    member_fingerprint: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-membership-pair/v1\0");
    for value in [group_id, orbit_id, observation_id, member_fingerprint] {
        fingerprint_string(&mut digest, value);
    }
    format!("{:x}", digest.finalize())
}

fn orbit_state_fingerprint(
    orbits: &OrbitBatch,
    row: usize,
    state_override: Option<&[f64; 6]>,
    covariance_override: Option<&[f64; 36]>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-orbit-state/v1\0");
    fingerprint_string(&mut digest, orbits.coordinates.frame.as_str());
    fingerprint_string(&mut digest, &orbits.coordinates.origins.origins[row].code());
    for value in state_override
        .copied()
        .unwrap_or(orbits.coordinates.values.raw_values()[row])
    {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update([u8::from(orbits.coordinates.times.is_some())]);
    if let Some(times) = &orbits.coordinates.times {
        fingerprint_string(&mut digest, times.scale.as_str());
        digest.update(times.epochs[row].days.to_le_bytes());
        digest.update(times.epochs[row].nanos.to_le_bytes());
    }
    let covariance = covariance_override.map(<[f64; 36]>::as_slice).or_else(|| {
        orbits
            .coordinates
            .covariance
            .as_ref()
            .map(|values| values.row_values(row))
    });
    digest.update([u8::from(covariance.is_some())]);
    if let Some(covariance) = covariance {
        digest.update((covariance.len() as u64).to_le_bytes());
        for &value in covariance {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    digest.update([u8::from(orbits.physical_parameters.is_some())]);
    if let Some(parameters) = &orbits.physical_parameters {
        for value in [
            parameters.h_v[row],
            parameters.h_v_sigma[row],
            parameters.g[row],
            parameters.g_sigma[row],
            parameters.sigma_eff[row],
            parameters.chi2_red[row],
        ] {
            fingerprint_optional_f64(&mut digest, value);
        }
    }
    digest.update([u8::from(orbits.non_gravitational_parameters.is_some())]);
    if let Some(parameters) = &orbits.non_gravitational_parameters {
        fingerprint_optional_string(&mut digest, parameters.source[row].as_deref());
        let values = parameters.row(row);
        for value in [
            values.a1, values.a2, values.a3, values.aln, values.nk, values.nm, values.nn, values.r0,
        ] {
            fingerprint_optional_f64(&mut digest, value);
        }
    }
    format!("{:x}", digest.finalize())
}

#[allow(clippy::too_many_arguments)]
fn group_input_fingerprint(
    group_id: &str,
    source: &SourceCatalogBatch,
    observers: &ObserverBatch,
    rows: &[usize],
    original: &AstrometrySnapshot,
    used: &AstrometrySnapshot,
    input_orbit: Option<(&OrbitBatch, usize)>,
    member_outlier: Option<&[bool]>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-input/v1\0");
    fingerprint_string(&mut digest, group_id);
    if let Some((orbits, row)) = input_orbit {
        digest.update([1]);
        fingerprint_string(&mut digest, &orbits.orbit_id[row].0);
        fingerprint_optional_string(
            &mut digest,
            orbits.object_id[row].as_ref().map(|value| value.0.as_str()),
        );
        fingerprint_string(
            &mut digest,
            &orbit_state_fingerprint(orbits, row, None, None),
        );
    } else {
        digest.update([0]);
    }
    digest.update([u8::from(member_outlier.is_some())]);
    if let Some(member_outlier) = member_outlier {
        digest.update((member_outlier.len() as u64).to_le_bytes());
        for value in member_outlier {
            digest.update([u8::from(*value)]);
        }
    }
    for fingerprint in member_input_fingerprints(source, observers, rows, original, used) {
        fingerprint_string(&mut digest, &fingerprint);
    }
    format!("{:x}", digest.finalize())
}

fn canonical_orbit_id(group_id: &str, config_sha256: &str, input_fingerprint: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-orbit/v1\0");
    for value in [group_id, config_sha256, input_fingerprint] {
        fingerprint_string(&mut digest, value);
    }
    format!("{:x}", digest.finalize())
}

fn provenance_consistency_digest(provenance: &OdProvenance) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-provenance-consistency/v3\0");
    fingerprint_string(&mut digest, provenance.route.as_str());
    fingerprint_string(&mut digest, &provenance.adam_core.version);
    fingerprint_string(&mut digest, &provenance.adam_core.commit);
    fingerprint_string(&mut digest, provenance.configuration.science_sha256());
    fingerprint_string(&mut digest, provenance.configuration.complete_sha256());
    fingerprint_string(&mut digest, &provenance.input_fingerprint);
    fingerprint_optional_string(&mut digest, provenance.state_fingerprint.as_deref());
    for value in [
        provenance
            .iod_provider
            .as_ref()
            .map(ProviderMetadata::canonical_json),
        provenance
            .refinement_provider
            .as_ref()
            .map(ProviderMetadata::canonical_json),
        provenance
            .evaluation_provider
            .as_ref()
            .map(ProviderMetadata::canonical_json),
        provenance
            .iod_diagnostics
            .as_ref()
            .map(IodDiagnostics::canonical_json),
        provenance
            .refinement_diagnostics
            .as_ref()
            .map(RefinementDiagnostics::canonical_json),
        provenance
            .evaluation_diagnostics
            .as_ref()
            .map(EvaluationDiagnostics::canonical_json),
    ] {
        fingerprint_optional_string(&mut digest, value.as_deref());
    }
    match provenance.iod_ambiguous {
        Some(value) => digest.update([1, u8::from(value)]),
        None => digest.update([0]),
    }
    for value in [provenance.iod_alternatives, provenance.num_rejected] {
        match value {
            Some(value) => {
                digest.update([1]);
                digest.update((value as u64).to_le_bytes());
            }
            None => digest.update([0]),
        }
    }
    fingerprint_optional_f64(&mut digest, provenance.rejected_fraction);
    format!("{:x}", digest.finalize())
}

fn group_scientific_code(error: &PropagationError) -> Option<GroupScientificFailureCode> {
    match error {
        PropagationError::GroupScientific { code, .. } => Some(*code),
        _ => None,
    }
}

fn iod_failure(group_id: &str, result: &IodResult, n: usize) -> Option<OrbitDeterminationFailure> {
    let output = result.to_output(n);
    (!output.found).then(|| OrbitDeterminationFailure {
        group_id: group_id.to_string(),
        orbit_id: None,
        input_orbit_id: None,
        stage: "iod",
        code: OdGroupFailureCode::NoSolution,
        message: output
            .not_found_reason
            .unwrap_or_else(|| "IOD found no canonical state".to_string()),
        num_obs: n,
        arc_length: None,
        triplets_tried: Some(output.triplets_tried),
        candidates_evaluated: Some(output.candidates_evaluated),
        candidates_failed: Some(output.candidates_failed),
        corrections_run: Some(result.corrections_run),
        passes: None,
        n_rejected: None,
        n_recovered: None,
        status_code: None,
        best_state_retained: false,
    })
}

fn scientific_failure(
    group_id: &str,
    orbit_id: Option<&str>,
    input_orbit_id: Option<&str>,
    stage: &'static str,
    error: &PropagationError,
    n: usize,
) -> OrbitDeterminationFailure {
    OrbitDeterminationFailure {
        group_id: group_id.to_string(),
        orbit_id: orbit_id.map(str::to_string),
        input_orbit_id: input_orbit_id.map(str::to_string),
        stage,
        code: OdGroupFailureCode::Scientific(
            group_scientific_code(error).expect("scientific failure was classified"),
        ),
        message: error.to_string(),
        num_obs: n,
        arc_length: None,
        triplets_tried: None,
        candidates_evaluated: None,
        candidates_failed: None,
        corrections_run: None,
        passes: None,
        n_rejected: None,
        n_recovered: None,
        status_code: None,
        best_state_retained: false,
    }
}

fn refinement_failure(
    group_id: &str,
    orbit_id: &str,
    input_orbit_id: Option<&str>,
    refined: &RefinedOutput,
) -> Option<OrbitDeterminationFailure> {
    (!refined.fit.converged).then(|| OrbitDeterminationFailure {
        group_id: group_id.to_string(),
        orbit_id: Some(orbit_id.to_string()),
        input_orbit_id: input_orbit_id.map(str::to_string),
        stage: "refinement",
        code: OdGroupFailureCode::NotConverged,
        message: format!(
            "refinement did not converge (status_code={})",
            refined.fit.status_code
        ),
        num_obs: refined.fit.evaluation.num_obs,
        arc_length: Some(refined.fit.evaluation.arc_length),
        triplets_tried: None,
        candidates_evaluated: None,
        candidates_failed: None,
        corrections_run: None,
        passes: Some(refined.passes),
        n_rejected: Some(refined.n_rejected),
        n_recovered: Some(refined.n_recovered),
        status_code: Some(refined.fit.status_code),
        best_state_retained: true,
    })
}

fn output_state_fingerprint(
    iod: Option<&IodResult>,
    refined: Option<&RefinedOutput>,
    input_orbit: Option<(&OrbitBatch, usize)>,
) -> Result<Option<String>, OdBatchError> {
    let override_state = refined.map(|value| &value.fit.state);
    let override_covariance = refined.map(|value| &value.fit.covariance);
    if let Some((orbits, row)) = input_orbit {
        return Ok(Some(orbit_state_fingerprint(
            orbits,
            row,
            override_state,
            override_covariance,
        )));
    }
    let Some(iod) = iod else {
        return Ok(None);
    };
    let Some(seed) = iod.seed_orbit()? else {
        return Ok(None);
    };
    Ok(Some(orbit_state_fingerprint(
        &seed,
        0,
        override_state,
        override_covariance,
    )))
}

#[allow(clippy::too_many_arguments)]
fn make_provenance(
    build: &CoreBuildIdentity,
    route: OdRoute,
    input_fingerprint: String,
    config: CanonicalConfiguration,
    iod_provider: Option<ProviderMetadata>,
    refinement_provider: Option<ProviderMetadata>,
    evaluation_provider: Option<ProviderMetadata>,
    iod: Option<&IodResult>,
    refined: Option<&RefinedOutput>,
    evaluation: Option<&FitEvaluation>,
    state_fingerprint: Option<String>,
) -> OdProvenance {
    let iod_diagnostics = iod.map(|result| IodDiagnostics {
        triplets_tried: result.triplets_tried,
        candidates_evaluated: result.candidates_evaluated,
        candidates_failed: result.candidates_failed,
        corrections_run: result.corrections_run,
        search_cap_exhausted: result.search_cap_exhausted,
    });
    let refinement_diagnostics = refined.map(|result| RefinementDiagnostics {
        passes: result.passes,
        n_rejected: result.n_rejected,
        n_recovered: result.n_recovered,
        flags: result.flags.clone(),
    });
    let evaluation_diagnostics = evaluation.map(|result| EvaluationDiagnostics {
        num_obs: result.num_obs,
        arc_length: result.arc_length,
        chi2: result.orbit_chi2,
        reduced_chi2: result.reduced_chi2,
    });
    let iod_ambiguous = iod.map(IodResult::ambiguous);
    let iod_alternatives = iod.map(|result| result.ambiguous_with.len());
    let mut provenance = OdProvenance {
        adam_core: build.clone(),
        input_fingerprint,
        state_fingerprint,
        provenance_fingerprint: String::new(),
        route,
        iod_provider,
        refinement_provider,
        evaluation_provider,
        iod_diagnostics,
        refinement_diagnostics,
        evaluation_diagnostics,
        configuration: config,
        iod_ambiguous,
        iod_alternatives,
        num_rejected: refined.map(|result| result.n_rejected),
        rejected_fraction: refined.map(|result| {
            if result.fit.evaluation.chi2.is_empty() {
                0.0
            } else {
                result.n_rejected as f64 / result.fit.evaluation.chi2.len() as f64
            }
        }),
    };
    provenance.provenance_fingerprint = provenance_consistency_digest(&provenance);
    provenance
}

fn prepare_association_groups(
    groups: &[AssociationGroup],
    input: &OdObservationBatch,
    config: &OdConfig,
    provider: &dyn TimeScaleProvider,
) -> Result<Vec<(PreparedObservations, ObserverBatch)>, OdBatchError> {
    groups
        .iter()
        .map(|group| {
            let observed = input
                .observed
                .take(&group.source_indices)
                .map_err(PropagationError::from)?;
            let observers = input
                .observers
                .take(&group.source_indices)
                .map_err(PropagationError::from)?;
            let obs_code = group
                .source_indices
                .iter()
                .map(|&row| input.source.observatory_code[row].clone())
                .collect::<Vec<_>>();
            let band = group
                .source_indices
                .iter()
                .map(|&row| input.source.filter[row].clone())
                .collect::<Vec<_>>();
            let astcat = group
                .source_indices
                .iter()
                .map(|&row| input.source.astcat[row].clone())
                .collect::<Vec<_>>();
            let prepared =
                prepare_observations(&observed, &obs_code, &band, &astcat, config, provider)?;
            Ok((prepared, observers))
        })
        .collect()
}

fn member_records(
    group_id: &str,
    orbit_id: &str,
    rows: &[usize],
    member_indices: &[usize],
    prepared: &PreparedObservations,
    input: &OdObservationBatch,
    outlier: &[bool],
) -> Vec<FittedOrbitMemberRecord> {
    let fingerprints = member_input_fingerprints(
        &input.source,
        &input.observers,
        rows,
        &prepared.original,
        &prepared.used,
    );
    rows.iter()
        .enumerate()
        .map(|(row, &source_index)| {
            let fingerprint = fingerprints[row].clone();
            FittedOrbitMemberRecord {
                orbit_id: orbit_id.to_string(),
                observation_id: input.source.id[source_index].clone(),
                source_index,
                input_member_index: member_indices.get(row).copied(),
                membership_fingerprint: membership_fingerprint(
                    group_id,
                    orbit_id,
                    &input.source.id[source_index],
                    &fingerprint,
                ),
                input_fingerprint: fingerprint,
                original: [
                    prepared.original.lon[row],
                    prepared.original.lat[row],
                    prepared.original.sigma_lon[row],
                    prepared.original.sigma_lat[row],
                    prepared.original.variance_lon[row],
                    prepared.original.variance_lat[row],
                    prepared.original.cov_lonlat[row],
                ],
                used: [
                    prepared.used.lon[row],
                    prepared.used.lat[row],
                    prepared.used.sigma_lon[row],
                    prepared.used.sigma_lat[row],
                    prepared.used.variance_lon[row],
                    prepared.used.variance_lat[row],
                    prepared.used.cov_lonlat[row],
                ],
                outlier: outlier.get(row).copied().unwrap_or(false),
            }
        })
        .collect()
}

/// Validated and transformed association groups ready for IOD execution.
///
/// Fields remain private so callers cannot bypass canonical planning or mutate
/// prepared observations between validation and execution.
#[doc(hidden)]
pub struct PreparedAssociatedBatch<'a> {
    input: &'a AssociatedObservations,
    config: &'a OdConfig,
    build: CoreBuildIdentity,
    groups: Vec<AssociationGroup>,
    prepared_groups: Vec<(PreparedObservations, ObserverBatch)>,
}

/// Validate, plan, and transform associated observations before a prediction
/// provider is acquired. `None` is the canonical empty-batch result.
#[doc(hidden)]
pub fn prepare_associated_batch<'a, T: OriginTranslationProvider>(
    input: &'a AssociatedObservations,
    config: &'a OdConfig,
    context: &OdExecutionContext<'_, T>,
) -> Result<Option<PreparedAssociatedBatch<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    context.build.validate()?;
    let groups = plan_associations(&input.observations.source.id, &input.associations)
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidInput, message))?;
    if groups.is_empty() {
        return Ok(None);
    }
    validate_problem(&input.observations)?;
    let prepared_groups =
        prepare_association_groups(&groups, &input.observations, config, context.time_provider)?;
    Ok(Some(PreparedAssociatedBatch {
        input,
        config,
        build: context.build.clone(),
        groups,
        prepared_groups,
    }))
}

/// Canonical batch IOD over deterministic first-appearance association groups.
pub fn run_iod<T: OriginTranslationProvider>(
    input: &AssociatedObservations,
    config: &OdConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let Some(prepared) = prepare_associated_batch(input, config, context)? else {
        return Ok(OdBatchResult::default());
    };
    run_iod_from_prepared(prepared, service, context)
}

/// Execute IOD from a Rust-owned validated preparation without rerunning the
/// observation pipeline.
#[doc(hidden)]
pub fn run_iod_from_prepared<T: OriginTranslationProvider>(
    prepared: PreparedAssociatedBatch<'_>,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    context.build.validate()?;
    if prepared.build != *context.build {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidBuildIdentity,
            "OD preparation and execution Core identities differ",
        ));
    }
    service.metadata.validate()?;
    run_iod_prepared(
        prepared.input,
        prepared.config,
        service,
        context,
        prepared.groups,
        prepared.prepared_groups,
    )
}

fn run_iod_prepared<T: OriginTranslationProvider>(
    input: &AssociatedObservations,
    config: &OdConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
    groups: Vec<AssociationGroup>,
    prepared_groups: Vec<(PreparedObservations, ObserverBatch)>,
) -> Result<OdBatchResult, OdBatchError> {
    let configuration = CanonicalConfiguration::from_od(config)?;
    let config_sha = config.configuration_sha256();
    let mut output = OdBatchResult {
        groups: Vec::with_capacity(groups.len()),
        providers: OdStageProviders {
            iod: Some(service.metadata.clone()),
            ..OdStageProviders::default()
        },
    };
    for (group, (prepared, observers)) in groups.into_iter().zip(prepared_groups) {
        let input_fingerprint = group_input_fingerprint(
            &group.group_id,
            &input.observations.source,
            &input.observations.observers,
            &group.source_indices,
            &prepared.original,
            &prepared.used,
            None,
            None,
        );
        let orbit_id = canonical_orbit_id(&group.group_id, &config_sha, &input_fingerprint);
        let result = run_iod_with_predictor(
            service.predictor,
            &prepared,
            &observers,
            config,
            context.time_provider,
            context.translation_provider,
        );
        match result {
            Ok(iod) => {
                let failure = iod_failure(&group.group_id, &iod, group.source_indices.len());
                let state_fingerprint = output_state_fingerprint(Some(&iod), None, None)?;
                let provenance = make_provenance(
                    context.build,
                    OdRoute::RunIod,
                    input_fingerprint,
                    configuration.clone(),
                    Some(service.metadata.clone()),
                    None,
                    None,
                    Some(&iod),
                    None,
                    None,
                    state_fingerprint,
                );
                let orbit = iod.accepted.is_some().then(|| FittedOrbitRecord {
                    group_id: group.group_id.clone(),
                    orbit_id: orbit_id.clone(),
                    input_orbit_index: None,
                    iod: Some(iod.clone()),
                    refined: None,
                    evaluation: None,
                    provenance,
                });
                let members = orbit.as_ref().map_or_else(Vec::new, |_| {
                    let outlier = iod.to_output(group.source_indices.len()).outlier;
                    member_records(
                        &group.group_id,
                        &orbit_id,
                        &group.source_indices,
                        &[],
                        &prepared,
                        &input.observations,
                        &outlier,
                    )
                });
                output.groups.push(OdGroupResult {
                    group_id: group.group_id,
                    orbit_id,
                    row_indices: group.source_indices,
                    input_orbit_index: None,
                    member_indices: Vec::new(),
                    prepared,
                    orbit,
                    members,
                    failure,
                    iod: Some(iod),
                });
            }
            Err(error) if group_scientific_code(&error).is_some() => {
                let failure = scientific_failure(
                    &group.group_id,
                    None,
                    None,
                    "iod",
                    &error,
                    group.source_indices.len(),
                );
                output.groups.push(OdGroupResult {
                    group_id: group.group_id,
                    orbit_id,
                    row_indices: group.source_indices,
                    input_orbit_index: None,
                    member_indices: Vec::new(),
                    prepared,
                    orbit: None,
                    members: Vec::new(),
                    failure: Some(failure),
                    iod: None,
                });
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(output)
}

/// Canonical exact composition: prepare once, IOD, then refinement over the
/// same prepared observations without rerunning transforms.
pub fn run_od<T: OriginTranslationProvider>(
    input: &AssociatedObservations,
    config: &OdConfig,
    iod_service: &PredictionService<'_>,
    refinement_service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let Some(prepared) = prepare_associated_batch(input, config, context)? else {
        return Ok(OdBatchResult::default());
    };
    run_od_from_prepared(prepared, iod_service, refinement_service, context)
}

/// Execute exact IOD-plus-refinement composition from one Rust-owned prepared
/// association batch.
#[doc(hidden)]
pub fn run_od_from_prepared<T: OriginTranslationProvider>(
    prepared: PreparedAssociatedBatch<'_>,
    iod_service: &PredictionService<'_>,
    refinement_service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let PreparedAssociatedBatch {
        input,
        config,
        build,
        groups,
        prepared_groups,
    } = prepared;
    context.build.validate()?;
    if build != *context.build {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidBuildIdentity,
            "OD preparation and execution Core identities differ",
        ));
    }
    iod_service.metadata.validate()?;
    refinement_service.metadata.validate()?;
    let configuration = CanonicalConfiguration::from_od(config)?;
    let refinement_observers = prepared_groups
        .iter()
        .map(|(_, observers)| observers.clone())
        .collect::<Vec<_>>();
    let mut output =
        run_iod_prepared(input, config, iod_service, context, groups, prepared_groups)?;
    output.providers.refinement = Some(refinement_service.metadata.clone());

    for (group, observers) in output.groups.iter_mut().zip(refinement_observers) {
        if group.orbit.is_none() {
            continue;
        }
        let iod = group
            .iod
            .as_ref()
            .expect("successful IOD group retains diagnostics")
            .clone();
        let seed = iod
            .seed_orbit()?
            .expect("accepted IOD result has a seed orbit");
        match refine_with_predictor(
            refinement_service.predictor,
            &seed,
            &group.prepared,
            &observers,
            config,
            context.time_provider,
            context.translation_provider,
        ) {
            Ok(refined) => {
                let failure = refinement_failure(
                    &group.group_id,
                    &group.orbit_id,
                    Some(&group.orbit_id),
                    &refined,
                );
                let input_fingerprint = group
                    .orbit
                    .as_ref()
                    .expect("successful IOD group retains an orbit")
                    .provenance
                    .input_fingerprint
                    .clone();
                let state_fingerprint = output_state_fingerprint(Some(&iod), Some(&refined), None)?;
                let provenance = make_provenance(
                    context.build,
                    OdRoute::RunOd,
                    input_fingerprint,
                    configuration.clone(),
                    Some(iod_service.metadata.clone()),
                    Some(refinement_service.metadata.clone()),
                    None,
                    Some(&iod),
                    Some(&refined),
                    None,
                    state_fingerprint,
                );
                let outlier = refined.fit.evaluation.outlier.clone();
                group.members = member_records(
                    &group.group_id,
                    &group.orbit_id,
                    &group.row_indices,
                    &[],
                    &group.prepared,
                    &input.observations,
                    &outlier,
                );
                group.orbit = Some(FittedOrbitRecord {
                    group_id: group.group_id.clone(),
                    orbit_id: group.orbit_id.clone(),
                    input_orbit_index: None,
                    iod: Some(iod),
                    refined: Some(refined),
                    evaluation: None,
                    provenance,
                });
                group.failure = failure;
            }
            Err(error) if group_scientific_code(&error).is_some() => {
                group.failure = Some(scientific_failure(
                    &group.group_id,
                    None,
                    Some(&group.orbit_id),
                    "refinement",
                    &error,
                    group.row_indices.len(),
                ));
                group.orbit = None;
                group.members.clear();
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(output)
}

fn orbit_group_id<'a>(orbits: &'a OrbitBatch, row: usize, fallback: &'a str) -> &'a str {
    orbits.object_id[row]
        .as_ref()
        .map(|value| value.0.as_str())
        .unwrap_or(fallback)
}

fn prepared_from_member_snapshots(
    observed: CoordinateBatch,
    member_indices: &[usize],
    members: &FittedOrbitMembersBatch,
) -> Result<PreparedObservations, OdBatchError> {
    let required = member_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |value| value + 1);
    for (name, len) in members
        .original_astrometry
        .lengths()
        .into_iter()
        .chain(members.used_astrometry.lengths())
    {
        if len < required {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                format!("FittedOrbitMembers {name} has inconsistent length"),
            ));
        }
    }
    let spherical = observed.values.spherical().ok_or_else(|| {
        OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "OD observations must be spherical",
        )
    })?;
    let source_lon = spherical.iter().map(|value| value[1]).collect::<Vec<_>>();
    let source_lat = spherical.iter().map(|value| value[2]).collect::<Vec<_>>();
    let source_covariance = observed.covariance.as_ref().ok_or_else(|| {
        OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "refinement observations require covariance",
        )
    })?;
    if source_covariance.dimension != 6 {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "refinement requires six-dimensional spherical observation covariance",
        ));
    }
    let source_original = AstrometrySnapshot::from_columns(
        &source_lon,
        &source_lat,
        &source_covariance.values_row_major,
    );
    let mut values = observed.values.raw_values().to_vec();
    let mut covariance = source_covariance.values_row_major.clone();
    let mut original = AstrometrySnapshot {
        lon: Vec::with_capacity(member_indices.len()),
        lat: Vec::with_capacity(member_indices.len()),
        sigma_lon: Vec::with_capacity(member_indices.len()),
        sigma_lat: Vec::with_capacity(member_indices.len()),
        variance_lon: Vec::with_capacity(member_indices.len()),
        variance_lat: Vec::with_capacity(member_indices.len()),
        cov_lonlat: Vec::with_capacity(member_indices.len()),
    };
    let mut lon = Vec::with_capacity(member_indices.len());
    let mut lat = Vec::with_capacity(member_indices.len());
    let mut models_changed = false;
    let equal = |left: f64, right: f64| {
        left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan())
    };
    for (row, &member) in member_indices.iter().enumerate() {
        let required_value = |column: &[Option<f64>], label: &str| {
            column[member].ok_or_else(|| {
                OdBatchError::new(
                    OdBatchErrorCode::InvalidInput,
                    format!("refinement/evaluation requires {label} for every declared member"),
                )
            })
        };
        let persisted_original = [
            required_value(&members.original_astrometry.lon, "original_astrometry")?,
            required_value(&members.original_astrometry.lat, "original_astrometry")?,
            required_value(
                &members.original_astrometry.sigma_lon,
                "original_astrometry",
            )?,
            required_value(
                &members.original_astrometry.sigma_lat,
                "original_astrometry",
            )?,
            required_value(
                &members.original_astrometry.variance_lon,
                "original_astrometry",
            )?,
            required_value(
                &members.original_astrometry.variance_lat,
                "original_astrometry",
            )?,
            required_value(
                &members.original_astrometry.cov_lonlat,
                "original_astrometry",
            )?,
        ];
        let source_values = [
            source_original.lon[row],
            source_original.lat[row],
            source_original.sigma_lon[row],
            source_original.sigma_lat[row],
            source_original.variance_lon[row],
            source_original.variance_lat[row],
            source_original.cov_lonlat[row],
        ];
        if persisted_original
            .iter()
            .zip(source_values)
            .any(|(&persisted, source)| !equal(persisted, source))
        {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!(
                    "observation source row {row} does not match persisted original_astrometry member row {member}"
                ),
            ));
        }
        original.lon.push(persisted_original[0]);
        original.lat.push(persisted_original[1]);
        original.sigma_lon.push(persisted_original[2]);
        original.sigma_lat.push(persisted_original[3]);
        original.variance_lon.push(persisted_original[4]);
        original.variance_lat.push(persisted_original[5]);
        original.cov_lonlat.push(persisted_original[6]);
        let used = [
            required_value(&members.used_astrometry.lon, "used_astrometry")?,
            required_value(&members.used_astrometry.lat, "used_astrometry")?,
            required_value(&members.used_astrometry.sigma_lon, "used_astrometry")?,
            required_value(&members.used_astrometry.sigma_lat, "used_astrometry")?,
            required_value(&members.used_astrometry.variance_lon, "used_astrometry")?,
            required_value(&members.used_astrometry.variance_lat, "used_astrometry")?,
            members.used_astrometry.cov_lonlat[member].unwrap_or(f64::NAN),
        ];
        if !used[0].is_finite()
            || !(0.0..360.0).contains(&used[0])
            || !used[1].is_finite()
            || !(-90.0..=90.0).contains(&used[1])
            || !used[2..6]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            || !equal(used[2], used[4].sqrt())
            || !equal(used[3], used[5].sqrt())
            || used[6].is_infinite()
            || (used[6].is_finite() && used[4] * used[5] - used[6].powi(2) <= 0.0)
        {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!("member used_astrometry row {member} is invalid"),
            ));
        }
        models_changed |= persisted_original
            .iter()
            .zip(used)
            .any(|(&original, used)| !equal(original, used));
        values[row][1] = used[0];
        values[row][2] = used[1];
        let block = &mut covariance[row * 36..(row + 1) * 36];
        block[7] = used[4];
        block[14] = used[5];
        block[8] = used[6];
        block[13] = used[6];
        lon.push(used[0]);
        lat.push(used[1]);
    }
    let used = AstrometrySnapshot::from_columns(&lon, &lat, &covariance);
    let mut covariance_batch = CovarianceBatch::new(
        member_indices.len(),
        source_covariance.dimension,
        covariance,
        source_covariance.units.clone(),
    )?;
    if let Some(validity) = &source_covariance.row_validity {
        covariance_batch = covariance_batch.with_row_validity(validity.clone())?;
    }
    let coordinates = CoordinateBatch::new(
        CoordinateValues::Spherical(values),
        observed.frame,
        observed.origins,
        observed.times,
        Some(covariance_batch),
    )?;
    Ok(PreparedObservations {
        original,
        used,
        models_changed,
        observations: coordinates,
    })
}

fn prepare_member_groups(
    groups: &[OrbitMemberGroup],
    input: &MemberedOrbits,
) -> Result<Vec<(PreparedObservations, ObserverBatch)>, OdBatchError> {
    groups
        .iter()
        .map(|group| {
            let observed = input
                .observations
                .observed
                .take(&group.source_indices)
                .map_err(PropagationError::from)?;
            let observers = input
                .observations
                .observers
                .take(&group.source_indices)
                .map_err(PropagationError::from)?;
            let prepared = prepared_from_member_snapshots(
                observed,
                &group.member_indices,
                &input.members,
            )?;
            let actual = member_input_fingerprints(
                &input.observations.source,
                &input.observations.observers,
                &group.source_indices,
                &prepared.original,
                &prepared.used,
            );
            let group_id = orbit_group_id(&input.orbits, group.orbit_index, &group.orbit_id);
            for ((&source_row, &member_row), fingerprint) in group
                .source_indices
                .iter()
                .zip(group.member_indices.iter())
                .zip(actual.iter())
            {
                let persisted = input.members.input_fingerprint[member_row]
                    .as_deref()
                    .filter(|value| is_sha256(value))
                    .ok_or_else(|| {
                        OdBatchError::new(
                            OdBatchErrorCode::InvalidProvenance,
                            format!("member input fingerprint is missing or invalid at row {member_row}"),
                        )
                    })?;
                if persisted != fingerprint {
                    return Err(OdBatchError::new(
                        OdBatchErrorCode::InvalidProvenance,
                        format!("current input does not match member fingerprint at row {member_row}"),
                    ));
                }
                let membership = membership_fingerprint(
                    group_id,
                    &group.orbit_id,
                    &input.observations.source.id[source_row],
                    fingerprint,
                );
                if input.members.membership_fingerprint[member_row].as_deref() != Some(&membership) {
                    return Err(OdBatchError::new(
                        OdBatchErrorCode::InvalidProvenance,
                        format!("current membership does not match fingerprint at row {member_row}"),
                    ));
                }
            }
            Ok((prepared, observers))
        })
        .collect()
}

fn validate_member_input(
    input: &MemberedOrbits,
    require_outlier: bool,
    context: &CoreBuildIdentity,
) -> Result<Vec<OrbitMemberGroup>, OdBatchError> {
    context.validate()?;
    input.members.validate_lengths(require_outlier)?;
    input
        .orbits
        .validate()
        .map_err(|error| OdBatchError::new(OdBatchErrorCode::InvalidInput, error.to_string()))?;
    if input.provenance.len() != input.orbits.len() {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidProvenance,
            format!(
                "orbit provenance length {} does not match {} orbit rows",
                input.provenance.len(),
                input.orbits.len()
            ),
        ));
    }
    for (row, provenance) in input.provenance.iter().enumerate() {
        let provenance = provenance.as_ref().ok_or_else(|| {
            OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!("canonical orbit row {row} has no Rust provenance"),
            )
        })?;
        provenance.validate()?;
        if provenance.adam_core != *context {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!("orbit provenance at row {row} has incompatible Core identity"),
            ));
        }
        let actual = orbit_state_fingerprint(&input.orbits, row, None, None);
        if provenance.state_fingerprint.as_deref() != Some(&actual) {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!("orbit row {row} does not match its state fingerprint"),
            ));
        }
    }
    plan_orbit_members(
        &input.orbits,
        &input.observations.source.id,
        &input.members.orbit_id,
        &input.members.observation_id,
    )
    .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidInput, message))
}

/// Validated member groups with persisted astrometry restored for execution.
#[doc(hidden)]
pub struct PreparedMemberBatch<'a> {
    input: &'a MemberedOrbits,
    build: CoreBuildIdentity,
    groups: Vec<OrbitMemberGroup>,
    prepared_groups: Vec<(PreparedObservations, ObserverBatch)>,
}

/// Validate refinement inputs and seed provenance before acquiring a provider.
#[doc(hidden)]
pub fn prepare_refinement_batch<'a, T: OriginTranslationProvider>(
    input: &'a MemberedOrbits,
    config: &RefinementRunConfig,
    context: &OdExecutionContext<'_, T>,
) -> Result<Option<PreparedMemberBatch<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    let groups = validate_member_input(input, false, context.build)?;
    if groups.is_empty() {
        return Ok(None);
    }
    validate_problem(&input.observations)?;
    let prepared_groups = prepare_member_groups(&groups, input)?;
    for (group, (prepared, _)) in groups.iter().zip(&prepared_groups) {
        let group_id = orbit_group_id(&input.orbits, group.orbit_index, &group.orbit_id);
        let seed_provenance = input.provenance[group.orbit_index]
            .as_ref()
            .expect("validated provenance");
        if seed_provenance.route == OdRoute::RunIod {
            if seed_provenance.configuration.refinement_value() != Some(&config.canonical_value()) {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidProvenance,
                    format!(
                        "seed IOD refinement configuration at orbit row {} does not match",
                        group.orbit_index
                    ),
                ));
            }
            let actual = group_input_fingerprint(
                group_id,
                &input.observations.source,
                &input.observations.observers,
                &group.source_indices,
                &prepared.original,
                &prepared.used,
                None,
                None,
            );
            if actual != seed_provenance.input_fingerprint {
                return Err(OdBatchError::new(
                    OdBatchErrorCode::InvalidProvenance,
                    format!(
                        "current observations for orbit row {} do not match the canonical IOD input fingerprint",
                        group.orbit_index
                    ),
                ));
            }
        }
    }
    Ok(Some(PreparedMemberBatch {
        input,
        build: context.build.clone(),
        groups,
        prepared_groups,
    }))
}

/// Validate evaluation inputs before acquiring a provider.
#[doc(hidden)]
pub fn prepare_evaluation_batch<'a, T: OriginTranslationProvider>(
    input: &'a MemberedOrbits,
    config: &OdConfig,
    context: &OdExecutionContext<'_, T>,
) -> Result<Option<PreparedMemberBatch<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    let groups = validate_member_input(input, true, context.build)?;
    if groups.is_empty() {
        return Ok(None);
    }
    validate_problem(&input.observations)?;
    if let Some(row) = input.members.outlier.iter().position(Option::is_none) {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            format!("FittedOrbitMembers outlier is null at row {row}"),
        ));
    }
    let prepared_groups = prepare_member_groups(&groups, input)?;
    Ok(Some(PreparedMemberBatch {
        input,
        build: context.build.clone(),
        groups,
        prepared_groups,
    }))
}

/// Refine explicit orbit/member groups using persisted prepared astrometry.
pub fn refine<T: OriginTranslationProvider>(
    input: &MemberedOrbits,
    config: &RefinementRunConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let Some(prepared) = prepare_refinement_batch(input, config, context)? else {
        return Ok(OdBatchResult::default());
    };
    refine_from_prepared(prepared, config, service, context)
}

/// Execute refinement from a Rust-owned validated member preparation.
#[doc(hidden)]
pub fn refine_from_prepared<T: OriginTranslationProvider>(
    prepared: PreparedMemberBatch<'_>,
    config: &RefinementRunConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let PreparedMemberBatch {
        input,
        build,
        groups,
        prepared_groups,
    } = prepared;
    context.build.validate()?;
    if build != *context.build {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidBuildIdentity,
            "OD preparation and execution Core identities differ",
        ));
    }
    service.metadata.validate()?;
    let configuration = CanonicalConfiguration::from_refinement(config)?;
    let refinement = *config.refinement();
    let mut output = OdBatchResult {
        groups: Vec::with_capacity(groups.len()),
        providers: OdStageProviders {
            refinement: Some(service.metadata.clone()),
            ..OdStageProviders::default()
        },
    };
    for (group, (prepared, observers)) in groups.into_iter().zip(prepared_groups) {
        let group_id =
            orbit_group_id(&input.orbits, group.orbit_index, &group.orbit_id).to_string();
        let seed = input.orbits.take(&[group.orbit_index])?;
        let seed_provenance = input.provenance[group.orbit_index]
            .as_ref()
            .expect("validated provenance");
        let refined = refine_with(
            service.predictor,
            &seed,
            &prepared.observations,
            &observers,
            &refinement,
            context.time_provider,
            context.translation_provider,
        );
        match refined {
            Ok(refined) => {
                let failure =
                    refinement_failure(&group_id, &group.orbit_id, Some(&group.orbit_id), &refined);
                let effective_route = if seed_provenance.route == OdRoute::RunIod {
                    OdRoute::RunOd
                } else {
                    OdRoute::Refine
                };
                let effective_configuration = if effective_route == OdRoute::RunOd {
                    seed_provenance.configuration.clone()
                } else {
                    configuration.clone()
                };
                let input_fingerprint = if effective_route == OdRoute::RunOd {
                    seed_provenance.input_fingerprint.clone()
                } else {
                    group_input_fingerprint(
                        &group_id,
                        &input.observations.source,
                        &input.observations.observers,
                        &group.source_indices,
                        &prepared.original,
                        &prepared.used,
                        Some((&input.orbits, group.orbit_index)),
                        None,
                    )
                };
                let iod_provider = (effective_route == OdRoute::RunOd)
                    .then(|| seed_provenance.iod_provider.clone())
                    .flatten();
                let state_fingerprint = output_state_fingerprint(
                    None,
                    Some(&refined),
                    Some((&input.orbits, group.orbit_index)),
                )?;
                let mut provenance = make_provenance(
                    context.build,
                    effective_route,
                    input_fingerprint,
                    effective_configuration,
                    iod_provider,
                    Some(service.metadata.clone()),
                    None,
                    None,
                    Some(&refined),
                    None,
                    state_fingerprint,
                );
                if effective_route == OdRoute::RunOd {
                    provenance.iod_diagnostics = seed_provenance.iod_diagnostics.clone();
                    provenance.iod_ambiguous = seed_provenance.iod_ambiguous;
                    provenance.iod_alternatives = seed_provenance.iod_alternatives;
                    provenance.provenance_fingerprint = provenance_consistency_digest(&provenance);
                }
                let outlier = refined.fit.evaluation.outlier.clone();
                let members = member_records(
                    &group_id,
                    &group.orbit_id,
                    &group.source_indices,
                    &group.member_indices,
                    &prepared,
                    &input.observations,
                    &outlier,
                );
                output.groups.push(OdGroupResult {
                    group_id: group_id.clone(),
                    orbit_id: group.orbit_id.clone(),
                    row_indices: group.source_indices,
                    input_orbit_index: Some(group.orbit_index),
                    member_indices: group.member_indices,
                    prepared,
                    orbit: Some(FittedOrbitRecord {
                        group_id,
                        orbit_id: group.orbit_id,
                        input_orbit_index: Some(group.orbit_index),
                        iod: None,
                        refined: Some(refined.clone()),
                        evaluation: None,
                        provenance,
                    }),
                    members,
                    failure,
                    iod: None,
                });
            }
            Err(error) if group_scientific_code(&error).is_some() => {
                let failure = scientific_failure(
                    &group_id,
                    None,
                    Some(&group.orbit_id),
                    "refinement",
                    &error,
                    group.source_indices.len(),
                );
                output.groups.push(OdGroupResult {
                    group_id,
                    orbit_id: group.orbit_id,
                    row_indices: group.source_indices,
                    input_orbit_index: Some(group.orbit_index),
                    member_indices: group.member_indices,
                    prepared,
                    orbit: None,
                    members: Vec::new(),
                    failure: Some(failure),
                    iod: None,
                });
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(output)
}

/// Evaluate explicit orbit/member groups without changing their states.
pub fn evaluate_orbits<T: OriginTranslationProvider>(
    input: &MemberedOrbits,
    config: &OdConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let Some(prepared) = prepare_evaluation_batch(input, config, context)? else {
        return Ok(OdBatchResult::default());
    };
    evaluate_orbits_from_prepared(prepared, config, service, context)
}

/// Execute evaluation from a Rust-owned validated member preparation.
#[doc(hidden)]
pub fn evaluate_orbits_from_prepared<T: OriginTranslationProvider>(
    prepared: PreparedMemberBatch<'_>,
    config: &OdConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_, T>,
) -> Result<OdBatchResult, OdBatchError> {
    let PreparedMemberBatch {
        input,
        build,
        groups,
        prepared_groups,
    } = prepared;
    context.build.validate()?;
    if build != *context.build {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidBuildIdentity,
            "OD preparation and execution Core identities differ",
        ));
    }
    service.metadata.validate()?;
    let configuration = CanonicalConfiguration::from_od(config)?;
    let parameters = config.for_evaluation().covariance_dimension() as i64;
    let mut output = OdBatchResult {
        groups: Vec::with_capacity(groups.len()),
        providers: OdStageProviders {
            evaluation: Some(service.metadata.clone()),
            ..OdStageProviders::default()
        },
    };
    for (group, (prepared, observers)) in groups.into_iter().zip(prepared_groups) {
        let group_id =
            orbit_group_id(&input.orbits, group.orbit_index, &group.orbit_id).to_string();
        let orbit = input.orbits.take(&[group.orbit_index])?;
        let state = orbit
            .coordinates
            .values
            .cartesian()
            .expect("validated orbit coordinates are Cartesian")[0];
        let ignore = group
            .member_indices
            .iter()
            .map(|&row| input.members.outlier[row].unwrap_or(false))
            .collect::<Vec<_>>();
        match evaluate_orbit(
            service.predictor,
            state,
            &orbit,
            &prepared.observations,
            &observers,
            &ignore,
            parameters,
        ) {
            Ok(evaluation) => {
                let input_fingerprint = group_input_fingerprint(
                    &group_id,
                    &input.observations.source,
                    &input.observations.observers,
                    &group.source_indices,
                    &prepared.original,
                    &prepared.used,
                    Some((&input.orbits, group.orbit_index)),
                    Some(&ignore),
                );
                let state_fingerprint = Some(orbit_state_fingerprint(
                    &input.orbits,
                    group.orbit_index,
                    None,
                    None,
                ));
                let provenance = make_provenance(
                    context.build,
                    OdRoute::EvaluateOrbits,
                    input_fingerprint,
                    configuration.clone(),
                    None,
                    None,
                    Some(service.metadata.clone()),
                    None,
                    None,
                    Some(&evaluation),
                    state_fingerprint,
                );
                let members = member_records(
                    &group_id,
                    &group.orbit_id,
                    &group.source_indices,
                    &group.member_indices,
                    &prepared,
                    &input.observations,
                    &ignore,
                );
                output.groups.push(OdGroupResult {
                    group_id: group_id.clone(),
                    orbit_id: group.orbit_id.clone(),
                    row_indices: group.source_indices,
                    input_orbit_index: Some(group.orbit_index),
                    member_indices: group.member_indices,
                    prepared,
                    orbit: Some(FittedOrbitRecord {
                        group_id,
                        orbit_id: group.orbit_id,
                        input_orbit_index: Some(group.orbit_index),
                        iod: None,
                        refined: None,
                        evaluation: Some(evaluation),
                        provenance,
                    }),
                    members,
                    failure: None,
                    iod: None,
                });
            }
            Err(error) if group_scientific_code(&error).is_some() => {
                let failure = scientific_failure(
                    &group_id,
                    None,
                    Some(&group.orbit_id),
                    "evaluation",
                    &error,
                    group.source_indices.len(),
                );
                output.groups.push(OdGroupResult {
                    group_id,
                    orbit_id: group.orbit_id,
                    row_indices: group.source_indices,
                    input_orbit_index: Some(group.orbit_index),
                    member_indices: group.member_indices,
                    prepared,
                    orbit: None,
                    members: Vec::new(),
                    failure: Some(failure),
                    iod: None,
                });
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str) -> ProviderMetadata {
        ProviderMetadata {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            abi_version: 2,
            identity_validated_by_capsule: true,
            capabilities: 0,
            known_capabilities: 0,
            integration_time_scale: "tdb".to_string(),
            two_body_dynamics: false,
            configuration_fingerprint: "0".repeat(64),
            data_fingerprint: "1".repeat(64),
        }
    }

    fn run_od_provenance() -> OdProvenance {
        let mut provenance = OdProvenance {
            adam_core: CoreBuildIdentity {
                version: "0.7.0".to_string(),
                commit: "a".repeat(40),
            },
            input_fingerprint: "2".repeat(64),
            state_fingerprint: Some("3".repeat(64)),
            provenance_fingerprint: String::new(),
            route: OdRoute::RunOd,
            iod_provider: Some(provider("iod")),
            refinement_provider: Some(provider("refinement")),
            evaluation_provider: None,
            iod_diagnostics: Some(IodDiagnostics {
                triplets_tried: 3,
                candidates_evaluated: 2,
                candidates_failed: 1,
                corrections_run: 1,
                search_cap_exhausted: false,
            }),
            refinement_diagnostics: Some(RefinementDiagnostics {
                passes: 2,
                n_rejected: 1,
                n_recovered: 0,
                flags: vec!["not_converged".to_string()],
            }),
            evaluation_diagnostics: None,
            configuration: CanonicalConfiguration::from_value(json!({
                "iod": {},
                "refinement": {},
                "evaluation": {}
            }))
            .expect("test configuration is canonical"),
            iod_ambiguous: Some(false),
            iod_alternatives: Some(0),
            num_rejected: Some(1),
            rejected_fraction: Some(0.25),
        };
        provenance.provenance_fingerprint = provenance_consistency_digest(&provenance);
        provenance
    }

    #[test]
    fn provenance_fingerprint_covers_every_provider() {
        let mut provenance = run_od_provenance();
        provenance.validate().expect("complete provenance is valid");
        provenance
            .refinement_provider
            .as_mut()
            .expect("refinement provider")
            .version = "1.0.1".to_string();
        assert_eq!(
            provenance.validate().expect_err("tampering must fail").code,
            OdBatchErrorCode::InvalidProvenance
        );
    }

    #[test]
    fn route_validation_rejects_half_present_stage_provenance() {
        let mut provenance = run_od_provenance();
        provenance.refinement_diagnostics = None;
        provenance.provenance_fingerprint = provenance_consistency_digest(&provenance);
        assert_eq!(
            provenance
                .validate()
                .expect_err("incomplete stage identity must fail")
                .code,
            OdBatchErrorCode::InvalidProvenance
        );
    }
}
