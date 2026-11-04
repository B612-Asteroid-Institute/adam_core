//! Canonical, batch-native orbit-determination workflows.
//!
//! This module owns grouping, preparation, scientific-failure continuation,
//! output disposition, fingerprints, and provenance for both standalone Rust
//! callers and the Python bindings.  Predictor-level numerical kernels remain
//! available under [`crate::od::algorithms`], but are not alternate batch APIs.
//!
//! # Example
//!
//! Each workflow accepts an [`crate::od::OdPropagator`] whose identity is
//! derived from the actual two-body implementation or validated
//! propagation-only ABI-v2 capsule. Static validation and one-time preparation
//! remain private. Fitted `orbit_id`
//! values are mutable relational labels; immutable association identity remains
//! in each record's `group_id` / orbit `object_id`. Callers may convert a batch
//! with [`crate::od::OdBatchResult::to_membered_orbits`], edit that value's public orbit
//! and member label columns, and pass it to [`refine`] or [`evaluate_orbits`],
//! where relational integrity is validated.
//!
//! ```no_run
//! use adam_core::{run_iod, od::{AssociatedObservations, IodRunConfig, OdPropagator}};
//!
//! fn run(input: &AssociatedObservations, config: &IodRunConfig) {
//!     let provider = OdPropagator::two_body();
//!     let result = run_iod(input, config, &provider);
//!     # let _ = result;
//! }
//! ```

use adam_core_rs_coords::od::algorithms::{
    prepare_observations, refine_with_predictor, run_iod_with_predictor, PreparedObservations,
};
use adam_core_rs_coords::propagation::adapter::{
    refine_with, PropagatorPredictor, SphericalPredictor,
};
use adam_core_rs_coords::propagation::{
    evaluate_orbit, two_body_reference, AstrometrySnapshot, CovariancePropagation,
    EphemerisOptions, EpochPolicy, ExternalPropagator, FitEvaluation, GroupScientificFailureCode,
    IodResult, PropagationError, PropagationOptions, Propagator, RefinedOutput, TwoBodyPropagator,
    TwoBodyPropagatorConfig, EXTERNAL_PROPAGATOR_ABI_VERSION,
    EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL, EXTERNAL_PROPAGATOR_CAP_STM,
    EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE,
};
use adam_core_rs_coords::translation::OriginTranslationProvider;
use adam_core_rs_coords::types::time::TimeScaleProvider;
use adam_core_rs_coords::{
    AssociationBatch, CoordinateBatch, CoordinateRepresentation, CoordinateValues, CovarianceBatch,
    CovarianceUnits, DataFrame, ObjectId, ObserverBatch, OrbitBatch, OrbitId, OriginArray,
    OriginId, SchemaError, SourceCatalogBatch, TimeArray, TimeScale, Validity,
};
use adam_core_rs_spice::global_backend;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::sync::LazyLock;

/// Predictor-level numerical building blocks used by the canonical workflows.
pub mod algorithms {
    use super::*;

    pub use adam_core_rs_coords::od::algorithms::{
        prepare_observations, refine_with_predictor, run_iod_with_predictor,
        run_od_with_predictors, ConfiguredRunOdOutput, PreparedObservations,
    };

    /// Canonical ephemeris options and actual two-body backend derived from one
    /// authoritative typed fit configuration.
    pub struct CanonicalStageSettings {
        pub options: EphemerisOptions,
        pub two_body: TwoBodyPropagator,
    }

    pub fn canonical_stage_settings(
        config: WhitenedFitConfig,
    ) -> Result<CanonicalStageSettings, OdBatchError> {
        config.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        let two_body = config.two_body;
        let options = EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            lt_tol: two_body.lt_tol,
            max_iter: two_body.ephemeris_max_iter,
            tol: two_body.ephemeris_tol,
            stellar_aberration: false,
            max_lt_iter: two_body.max_lt_iter,
            output_time_scale: TimeScale::Utc,
            include_aberrated_coordinates: false,
            ..EphemerisOptions::default()
        };
        let two_body = TwoBodyPropagator::new(TwoBodyPropagatorConfig {
            max_iter: two_body.propagation_max_iter,
            tol: two_body.propagation_tol,
        })?;
        Ok(CanonicalStageSettings { options, two_body })
    }

    /// Canonical ERFA-backed time conversion shared by Core and PyO3 routes.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ErfaTimeProvider;

    impl TimeScaleProvider for ErfaTimeProvider {
        fn rescale(
            &self,
            times: &TimeArray,
            new_scale: TimeScale,
        ) -> Result<TimeArray, SchemaError> {
            times.rescale(new_scale)
        }
    }

    /// Canonical process-global SPICE origin translation shared by Core and
    /// PyO3 routes.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct GlobalSpiceTranslation;

    impl OriginTranslationProvider for GlobalSpiceTranslation {
        fn origin_translation_vectors(
            &self,
            origins: &OriginArray,
            target_origin: &OriginId,
            frame: DataFrame,
            times: &TimeArray,
        ) -> Result<Vec<[f64; 6]>, SchemaError> {
            let backend = global_backend().lock().map_err(|_| {
                SchemaError::InvalidRecordBatch("SPICE backend lock is poisoned".to_string())
            })?;
            OriginTranslationProvider::origin_translation_vectors(
                &*backend,
                origins,
                target_origin,
                frame,
                times,
            )
        }
    }

    /// Derive provenance metadata from the actual configured two-body backend.
    pub fn two_body_provider_metadata(propagator: &TwoBodyPropagator) -> ProviderMetadata {
        let identity = propagator.backend();
        let mut capabilities = EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE;
        if propagator.supports_stm() {
            capabilities |= EXTERNAL_PROPAGATOR_CAP_STM;
        }
        if propagator.supports_non_gravitational_parameters() {
            capabilities |= EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL;
        }
        ProviderMetadata {
            name: identity.name,
            version: identity.version,
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            identity_validated_by_capsule: false,
            capabilities,
            known_capabilities: capabilities,
            integration_time_scale: propagator.integration_time_scale().as_str().to_string(),
            two_body_dynamics: identity.two_body_dynamics,
            configuration_fingerprint: two_body_reference::configuration_fingerprint(
                propagator.config(),
            ),
            data_fingerprint: two_body_reference::DATA_FINGERPRINT.to_string(),
        }
    }

    /// Derive and validate provenance metadata from an actual ABI-v2 backend.
    pub fn external_provider_metadata(
        propagator: &ExternalPropagator,
    ) -> Result<ProviderMetadata, OdBatchError> {
        OdPropagator::external(propagator).external_metadata()
    }
}

use algorithms::{
    canonical_stage_settings, two_body_provider_metadata, ErfaTimeProvider, GlobalSpiceTranslation,
};

/// Immutable complete OD profiles and their resource declarations.
pub mod profiles {
    pub use adam_core_rs_coords::od::profiles::*;
}

pub use adam_core_rs_coords::propagation::IodConfig;
pub use adam_core_rs_coords::{
    IterativeFitConfig, ProfileIdentity, RefinementConfig, WhitenedFitConfig,
};

use adam_core_rs_coords::od::{
    plan_associations, plan_orbit_members, AssociationGroup, OrbitMemberGroup,
};
pub use adam_core_rs_coords::od::{
    EvaluationConfig, IodRunConfig, IodStageConfig, ObservationPipeline, ObservationTransform,
    OdConfig, OdConfigError, RefinementRunConfig, ResidualCovarianceOperation, ResourceBindings,
    ResourceProblem, ResourceRequirement, ResourceSlot, ResourceValidationError,
    ValidatedResourceIdentity,
};

/// Stable classification for invocation-fatal batch errors.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdBatchErrorCode {
    /// A decoded typed batch is internally inconsistent.
    InvalidInput,
    /// The explicit stage configuration is invalid.
    InvalidConfiguration,
    /// The prediction service identity or fingerprint is incomplete.
    InvalidProviderIdentity,
    /// Persisted orbit/member provenance is missing, malformed, or inconsistent.
    InvalidProvenance,
    /// Invalid propagation request, schema, or numerical service error.
    Propagation,
    /// Fatal propagation backend operation or coverage failure.
    Backend,
    /// Fatal propagation ABI or protocol failure.
    Protocol,
    /// Runtime service setup failure, such as thread-pool construction.
    Runtime,
}

impl OdBatchErrorCode {
    /// Stable snake-case representation used by foreign-language bindings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::InvalidConfiguration => "invalid_configuration",
            Self::InvalidProviderIdentity => "invalid_provider_identity",
            Self::InvalidProvenance => "invalid_provenance",
            Self::Propagation => "propagation",
            Self::Backend => "backend",
            Self::Protocol => "protocol",
            Self::Runtime => "runtime",
        }
    }
}

impl fmt::Display for OdBatchErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for OdBatchErrorCode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "invalid_input" => Ok(Self::InvalidInput),
            "invalid_configuration" => Ok(Self::InvalidConfiguration),
            "invalid_provider_identity" => Ok(Self::InvalidProviderIdentity),
            "invalid_provenance" => Ok(Self::InvalidProvenance),
            "propagation" => Ok(Self::Propagation),
            "backend" => Ok(Self::Backend),
            "protocol" => Ok(Self::Protocol),
            "runtime" => Ok(Self::Runtime),
            _ => Err(format!("unknown OD batch error code {value:?}")),
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

    /// Construct an error reported while lazily resolving an actual provider.
    pub fn invalid_provider_identity(message: impl Into<String>) -> Self {
        Self::new(OdBatchErrorCode::InvalidProviderIdentity, message)
    }

    fn propagation(source: PropagationError) -> Self {
        let code = match &source {
            PropagationError::Backend(_) | PropagationError::RowFailure(_) => {
                OdBatchErrorCode::Backend
            }
            PropagationError::BackendProtocol(_) => OdBatchErrorCode::Protocol,
            PropagationError::ThreadPool(_) => OdBatchErrorCode::Runtime,
            _ => OdBatchErrorCode::Propagation,
        };
        Self {
            code,
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

/// Canonical propagation backend for public OD workflows.
///
/// Stage options and provider provenance are derived inside Rust from the
/// typed workflow configuration and this actual backend. Arbitrary predictors
/// remain available only through [`algorithms`].
pub struct OdPropagator<'a> {
    backend: OdPropagatorBackend<'a>,
}

enum OdPropagatorBackend<'a> {
    TwoBody,
    External(&'a ExternalPropagator),
}

impl OdPropagator<'static> {
    /// Use adam-core's built-in two-body propagator. Each stage receives the
    /// exact tolerances and iteration limits declared by its typed config.
    pub const fn two_body() -> Self {
        Self {
            backend: OdPropagatorBackend::TwoBody,
        }
    }
}

impl<'a> OdPropagator<'a> {
    /// Select a propagation-only ABI backend without resolving its identity.
    /// Identity and ABI-v2 provenance validation happen only after static
    /// preparation establishes that the batch is non-empty.
    pub const fn external(propagator: &'a ExternalPropagator) -> Self {
        Self {
            backend: OdPropagatorBackend::External(propagator),
        }
    }

    fn external_metadata(&self) -> Result<ProviderMetadata, OdBatchError> {
        let OdPropagatorBackend::External(propagator) = self.backend else {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProviderIdentity,
                "external provider metadata requested for the two-body backend",
            ));
        };
        if propagator.configuration_fingerprint().is_none()
            || propagator.data_fingerprint().is_none()
        {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProviderIdentity,
                "native OD requires the ABI v2 provenance metadata suffix (configuration_fingerprint and data_fingerprint); this provider is propagation-compatible but not provenance-complete",
            ));
        }
        let identity = propagator.identity();
        let metadata = ProviderMetadata {
            name: identity.name.clone(),
            version: identity.version.clone(),
            abi_version: propagator.abi_version(),
            identity_validated_by_capsule: true,
            capabilities: propagator.capabilities(),
            known_capabilities: propagator.known_capabilities(),
            integration_time_scale: propagator.integration_scale().as_str().to_string(),
            two_body_dynamics: identity.two_body_dynamics,
            configuration_fingerprint: propagator
                .configuration_fingerprint()
                .expect("metadata presence checked above")
                .to_string(),
            data_fingerprint: propagator
                .data_fingerprint()
                .expect("metadata presence checked above")
                .to_string(),
        };
        metadata.validate()?;
        Ok(metadata)
    }
}

/// Internal lazy provider-acquisition seam used by binding adapters and tests.
trait OdPropagatorSource {
    /// Run one already-prepared operation with the source's actual backend.
    fn with_propagator<R>(
        &self,
        operation: impl for<'provider> FnOnce(
            &'provider OdPropagator<'provider>,
        ) -> Result<R, OdBatchError>,
    ) -> Result<R, OdBatchError>;
}

impl OdPropagatorSource for OdPropagator<'_> {
    fn with_propagator<R>(
        &self,
        operation: impl for<'provider> FnOnce(
            &'provider OdPropagator<'provider>,
        ) -> Result<R, OdBatchError>,
    ) -> Result<R, OdBatchError> {
        operation(self)
    }
}

struct PredictionService<'a> {
    predictor: &'a dyn SphericalPredictor,
    metadata: ProviderMetadata,
}

struct OdExecutionContext<'a> {
    time_provider: &'a ErfaTimeProvider,
    translation_provider: &'a GlobalSpiceTranslation,
}

fn with_stage_predictor<R>(
    provider: &OdPropagator<'_>,
    config: WhitenedFitConfig,
    driver: impl FnOnce(&PredictionService<'_>, &OdExecutionContext<'_>) -> Result<R, OdBatchError>,
) -> Result<R, OdBatchError> {
    let settings = canonical_stage_settings(config)?;
    let time = ErfaTimeProvider;
    let translation = GlobalSpiceTranslation;
    let context = OdExecutionContext {
        time_provider: &time,
        translation_provider: &translation,
    };
    match &provider.backend {
        OdPropagatorBackend::TwoBody => {
            let predictor = PropagatorPredictor {
                propagator: &settings.two_body,
                options: &settings.options,
                provider: &time,
                translation_provider: &translation,
            };
            let service = PredictionService {
                predictor: &predictor,
                metadata: two_body_provider_metadata(&settings.two_body),
            };
            driver(&service, &context)
        }
        OdPropagatorBackend::External(propagator) => {
            let predictor = PropagatorPredictor {
                propagator: *propagator,
                options: &settings.options,
                provider: &time,
                translation_provider: &translation,
            };
            let service = PredictionService {
                predictor: &predictor,
                metadata: provider.external_metadata()?,
            };
            driver(&service, &context)
        }
    }
}

fn with_stage_predictors<R>(
    provider: &OdPropagator<'_>,
    iod_config: WhitenedFitConfig,
    refinement_config: WhitenedFitConfig,
    driver: impl FnOnce(
        &PredictionService<'_>,
        &PredictionService<'_>,
        &OdExecutionContext<'_>,
    ) -> Result<R, OdBatchError>,
) -> Result<R, OdBatchError> {
    let iod = canonical_stage_settings(iod_config)?;
    let refinement = canonical_stage_settings(refinement_config)?;
    let time = ErfaTimeProvider;
    let translation = GlobalSpiceTranslation;
    let context = OdExecutionContext {
        time_provider: &time,
        translation_provider: &translation,
    };
    match &provider.backend {
        OdPropagatorBackend::TwoBody => {
            let iod_predictor = PropagatorPredictor {
                propagator: &iod.two_body,
                options: &iod.options,
                provider: &time,
                translation_provider: &translation,
            };
            let refinement_predictor = PropagatorPredictor {
                propagator: &refinement.two_body,
                options: &refinement.options,
                provider: &time,
                translation_provider: &translation,
            };
            let iod_service = PredictionService {
                predictor: &iod_predictor,
                metadata: two_body_provider_metadata(&iod.two_body),
            };
            let refinement_service = PredictionService {
                predictor: &refinement_predictor,
                metadata: two_body_provider_metadata(&refinement.two_body),
            };
            driver(&iod_service, &refinement_service, &context)
        }
        OdPropagatorBackend::External(propagator) => {
            let iod_predictor = PropagatorPredictor {
                propagator: *propagator,
                options: &iod.options,
                provider: &time,
                translation_provider: &translation,
            };
            let refinement_predictor = PropagatorPredictor {
                propagator: *propagator,
                options: &refinement.options,
                provider: &time,
                translation_provider: &translation,
            };
            let metadata = provider.external_metadata()?;
            let iod_service = PredictionService {
                predictor: &iod_predictor,
                metadata: metadata.clone(),
            };
            let refinement_service = PredictionService {
                predictor: &refinement_predictor,
                metadata,
            };
            driver(&iod_service, &refinement_service, &context)
        }
    }
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
#[derive(Debug, Clone, Default, PartialEq)]
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
#[non_exhaustive]
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

impl fmt::Display for OdRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for OdRoute {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "run_iod" => Ok(Self::RunIod),
            "run_od" => Ok(Self::RunOd),
            "refine" => Ok(Self::Refine),
            "evaluate_orbits" => Ok(Self::EvaluateOrbits),
            _ => Err(format!("unknown OD route {value:?}")),
        }
    }
}

/// Validated path-free canonical configuration identity.
#[derive(Debug, Clone)]
pub struct CanonicalConfiguration {
    value: Value,
    science_sha256: String,
    complete_sha256: String,
    declared_science_sha256: bool,
}

fn invalid_profile_claim(message: impl Into<String>) -> OdBatchError {
    OdBatchError::new(OdBatchErrorCode::InvalidProvenance, message)
}

#[derive(Debug)]
struct ValidatedProfileDefinition {
    profile: profiles::OdProfile,
    identity: Value,
    transforms: Value,
    iod: Value,
    refinement: Value,
    evaluation: Value,
}

fn profile_transforms(profile: profiles::OdProfile, definition: &Value) -> Value {
    let mut transforms = definition
        .get("pipeline")
        .and_then(Value::as_array)
        .expect("known profile pipeline is an array")
        .clone();
    for transform in &mut transforms {
        let kind = transform
            .get("kind")
            .and_then(Value::as_str)
            .expect("known profile transform has a kind");
        let slot = match kind {
            "fill_missing_astrometric_uncertainty" => {
                Some(ResourceSlot::StationCatalogUncertaintyTable)
            }
            "efcc18_catalog_bias_correction" => Some(ResourceSlot::Efcc18CatalogBiasTable),
            "station_band_residual_covariance_adjustment" => {
                Some(ResourceSlot::StationBandResidualCovarianceTable)
            }
            "nightly_batch_deweighting" => None,
            _ => unreachable!("unknown transform in a verified profile"),
        };
        if let Some(slot) = slot {
            let requirement = profile
                .requirements()
                .iter()
                .find(|requirement| requirement.slot == slot)
                .expect("known profile declares every transform resource");
            transform
                .as_object_mut()
                .expect("known profile transform is an object")
                .insert(
                    "resource".to_string(),
                    json!({
                        "requirement": requirement.canonical_value(),
                        "actual_sha256": requirement.sha256,
                        "actual_byte_size": requirement.byte_size,
                    }),
                );
        }
    }
    Value::Array(transforms)
}

fn build_validated_profile(
    profile: profiles::OdProfile,
) -> Result<ValidatedProfileDefinition, String> {
    profile.verify_definition_fingerprint()?;
    let definition = profile.definition_value();
    let identity = profile.identity();
    Ok(ValidatedProfileDefinition {
        profile,
        identity: json!({
            "name": identity.name,
            "version": identity.version,
            "definition_sha256": identity.definition_sha256,
        }),
        transforms: profile_transforms(profile, &definition),
        iod: definition
            .get("iod")
            .expect("known profile declares IOD")
            .clone(),
        refinement: definition
            .get("refinement")
            .expect("known profile declares refinement")
            .clone(),
        evaluation: definition
            .get("evaluation")
            .expect("known profile declares evaluation")
            .clone(),
    })
}

static CORE_OD_V1_DEFINITION: LazyLock<Result<ValidatedProfileDefinition, String>> =
    LazyLock::new(|| build_validated_profile(profiles::CORE_OD_V1));
static ASTEROID_INSTITUTE_OD_V1_DEFINITION: LazyLock<Result<ValidatedProfileDefinition, String>> =
    LazyLock::new(|| build_validated_profile(profiles::ASTEROID_INSTITUTE_OD_V1));

fn cached_profile_definition(
    name: &str,
) -> Result<&'static ValidatedProfileDefinition, OdBatchError> {
    let cached = match name {
        "CORE_OD_V1" => &*CORE_OD_V1_DEFINITION,
        "ASTEROID_INSTITUTE_OD_V1" => &*ASTEROID_INSTITUTE_OD_V1_DEFINITION,
        _ => {
            return Err(invalid_profile_claim(format!(
                "unknown OD profile identity {name:?}"
            )))
        }
    };
    cached
        .as_ref()
        .map_err(|message| invalid_profile_claim(message.clone()))
}

fn profile_from_claim(
    value: &Value,
) -> Result<Option<&'static ValidatedProfileDefinition>, OdBatchError> {
    let Some(claim) = value.get("profile") else {
        return Ok(None);
    };
    if claim.is_null() {
        return Ok(None);
    }
    let name = claim
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_profile_claim("profile identity name is missing or invalid"))?;
    let profile = cached_profile_definition(name)?;
    if claim != &profile.identity {
        return Err(invalid_profile_claim(format!(
            "profile identity for {name} does not match its immutable definition"
        )));
    }
    Ok(Some(profile))
}

fn validate_profile_claim(
    value: &Value,
) -> Result<Option<&'static ValidatedProfileDefinition>, OdBatchError> {
    let Some(profile) = profile_from_claim(value)? else {
        return Ok(None);
    };
    let schema = value.get("schema").and_then(Value::as_str);
    if !matches!(
        schema,
        Some("adam-core-iod-run-config/v1" | "adam-core-od-config/v1")
    ) {
        return Err(invalid_profile_claim(
            "profile identity is attached to an unsupported configuration schema",
        ));
    }
    if value.get("transforms") != Some(&profile.transforms)
        || value.get("iod") != Some(&profile.iod)
    {
        return Err(invalid_profile_claim(format!(
            "configuration does not match the immutable {} profile pipeline and IOD definition",
            profile.profile.identity().name
        )));
    }
    if schema == Some("adam-core-od-config/v1")
        && (value.get("refinement") != Some(&profile.refinement)
            || value.get("evaluation") != Some(&profile.evaluation))
    {
        return Err(invalid_profile_claim(format!(
            "configuration does not match the immutable {} profile refinement and evaluation definition",
            profile.profile.identity().name
        )));
    }
    Ok(Some(profile))
}

impl CanonicalConfiguration {
    /// Construct from a complete bound IOD-stage configuration.
    pub fn from_iod(config: &IodRunConfig) -> Result<Self, OdBatchError> {
        config.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        Self::from_value(
            serde_json::from_str(&config.provenance_json()).expect("IodRunConfig JSON is valid"),
        )
    }

    /// Construct from a complete OD configuration.
    pub fn from_od(config: &OdConfig) -> Result<Self, OdBatchError> {
        config.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        Self::from_value(
            serde_json::from_str(&config.provenance_json()).expect("OdConfig JSON is valid"),
        )
    }

    /// Construct from a self-contained evaluation configuration.
    pub fn from_evaluation(config: &EvaluationConfig) -> Result<Self, OdBatchError> {
        config.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        Self::from_value(config.canonical_value())
    }

    /// Compose the exact canonical `run_od` configuration from an IOD seed and
    /// the explicit refinement configuration.
    pub fn compose_od(iod: &Self, refinement: &RefinementRunConfig) -> Result<Self, OdBatchError> {
        refinement.validate().map_err(|message| {
            OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message)
        })?;
        let mut value = iod.value.clone();
        if value.get("schema").and_then(Value::as_str) != Some("adam-core-iod-run-config/v1") {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                "run_iod seed does not contain a canonical IOD-stage configuration",
            ));
        }
        let claimed_profile = validate_profile_claim(&value)?;
        let refinement_value = refinement.canonical_value();
        let retain_profile =
            claimed_profile.is_some_and(|profile| profile.refinement == refinement_value);
        let object = value.as_object_mut().ok_or_else(|| {
            OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                "run_iod seed configuration is not an object",
            )
        })?;
        object.remove("configuration_sha256");
        if claimed_profile.is_some() && !retain_profile {
            object.insert("profile".to_string(), Value::Null);
        }
        object.insert(
            "schema".to_string(),
            Value::String("adam-core-od-config/v1".to_string()),
        );
        object.insert("refinement".to_string(), refinement_value);
        object.insert("evaluation".to_string(), json!({"covariance_dimension": 6}));
        let mut science = value.clone();
        science
            .as_object_mut()
            .expect("configuration is an object")
            .remove("profile");
        let fingerprint =
            sha256(&serde_json::to_vec(&science).expect("configuration is serializable"));
        value
            .as_object_mut()
            .expect("configuration is an object")
            .insert(
                "configuration_sha256".to_string(),
                Value::String(fingerprint),
            );
        Self::from_value(value)
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
        validate_profile_claim(&value)?;
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
        let declared_science_sha256 = declared.is_some();
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
            declared_science_sha256,
        })
    }

    pub fn science_sha256(&self) -> &str {
        &self.science_sha256
    }

    pub fn complete_sha256(&self) -> &str {
        &self.complete_sha256
    }

    /// The declared science digest carried by complete OD configurations.
    pub fn declared_science_sha256(&self) -> Option<&str> {
        self.declared_science_sha256
            .then_some(self.science_sha256.as_str())
    }

    /// Canonical JSON used only at serialization boundaries.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.value).expect("validated configuration is serializable")
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
            OdRoute::RunIod => {
                self.configuration
                    .value
                    .get("schema")
                    .and_then(Value::as_str)
                    == Some("adam-core-iod-run-config/v1")
                    && self.configuration.value.get("iod").is_some()
                    && self.configuration.value.get("transforms").is_some()
                    && !full_configuration
            }
            OdRoute::RunOd => full_configuration,
            OdRoute::Refine => {
                !full_configuration && self.configuration.value.get("method").is_some()
            }
            OdRoute::EvaluateOrbits => {
                self.configuration
                    .value
                    .get("schema")
                    .and_then(Value::as_str)
                    == Some("adam-core-evaluation-config/v1")
                    && self.configuration.value.get("prediction_fit").is_some()
                    && !full_configuration
            }
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

/// Stable canonical workflow stages used by typed group failures.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdStage {
    Iod,
    Refinement,
    Evaluation,
}

impl OdStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Iod => "iod",
            Self::Refinement => "refinement",
            Self::Evaluation => "evaluation",
        }
    }
}

impl fmt::Display for OdStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for OdStage {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "iod" => Ok(Self::Iod),
            "refinement" => Ok(Self::Refinement),
            "evaluation" => Ok(Self::Evaluation),
            _ => Err(format!("unknown OD stage {value:?}")),
        }
    }
}

/// Stable group-local scientific/fitting failure classifications.
#[non_exhaustive]
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

impl fmt::Display for OdGroupFailureCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for OdGroupFailureCode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "no_solution" => Ok(Self::NoSolution),
            "not_converged" => Ok(Self::NotConverged),
            "rank_deficient" => Ok(Self::Scientific(GroupScientificFailureCode::RankDeficient)),
            "no_included_observations" => Ok(Self::Scientific(
                GroupScientificFailureCode::NoIncludedObservations,
            )),
            "no_fit_pass" => Ok(Self::Scientific(GroupScientificFailureCode::NoFitPass)),
            _ => Err(format!("unknown OD group failure code {value:?}")),
        }
    }
}

/// Typed per-group failure. These values continue the surrounding batch.
#[derive(Debug, Clone)]
pub struct OrbitDeterminationFailure {
    pub group_id: String,
    pub orbit_id: Option<String>,
    pub input_orbit_id: Option<String>,
    pub stage: OdStage,
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
    /// Materialized one-row fitted/evaluated orbit, including covariance when fitted.
    pub orbit: OrbitBatch,
    pub group_id: String,
    pub input_orbit_index: Option<usize>,
    pub iod: Option<IodResult>,
    pub refined: Option<RefinedOutput>,
    pub evaluation: Option<FitEvaluation>,
    pub provenance: OdProvenance,
}

/// Named astrometric values persisted before and after the observation pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AstrometryRecord {
    pub lon: f64,
    pub lat: f64,
    pub sigma_lon: f64,
    pub sigma_lat: f64,
    pub variance_lon: f64,
    pub variance_lat: f64,
    pub cov_lonlat: f64,
}

impl AstrometryRecord {
    fn from_snapshot(snapshot: &AstrometrySnapshot, row: usize) -> Self {
        Self {
            lon: snapshot.lon[row],
            lat: snapshot.lat[row],
            sigma_lon: snapshot.sigma_lon[row],
            sigma_lat: snapshot.sigma_lat[row],
            variance_lon: snapshot.variance_lon[row],
            variance_lat: snapshot.variance_lat[row],
            cov_lonlat: snapshot.cov_lonlat[row],
        }
    }
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
    pub original: AstrometryRecord,
    pub used: AstrometryRecord,
    pub outlier: bool,
}

#[derive(Debug, Clone)]
struct WorkingGroupResult {
    group_id: String,
    orbit_id: String,
    row_indices: Vec<usize>,
    input_orbit_index: Option<usize>,
    member_indices: Vec<usize>,
    prepared: PreparedObservations,
    orbit: Option<FittedOrbitRecord>,
    members: Vec<FittedOrbitMemberRecord>,
    failure: Option<OrbitDeterminationFailure>,
    iod: Option<IodResult>,
}

/// Internal planning/diagnostic metadata for one deterministic disposition.
#[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
#[derive(Debug, Clone)]
struct OdGroupResult {
    group_id: String,
    orbit_id: String,
    row_indices: Vec<usize>,
    input_orbit_index: Option<usize>,
    member_indices: Vec<usize>,
    prepared: PreparedObservations,
    orbit_index: Option<usize>,
    member_start: usize,
    member_end: usize,
    failure_index: Option<usize>,
    /// IOD diagnostics are retained even when no fitted orbit exists.
    iod: Option<IodResult>,
}

#[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
struct GroupView<'a> {
    group: &'a OdGroupResult,
    orbit: Option<&'a FittedOrbitRecord>,
    members: &'a [FittedOrbitMemberRecord],
    failure: Option<&'a OrbitDeterminationFailure>,
}

/// Binding-only borrowed view of one disposition and its associated products.
///
/// The view is created only from one borrowed [`OdBatchResult`], so its orbit,
/// membership, failure, and diagnostic references cannot be paired with a
/// different batch.
#[cfg(feature = "python-bindings")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct PythonGroupView<'a>(GroupView<'a>);

#[cfg(feature = "python-bindings")]
impl<'a> PythonGroupView<'a> {
    pub fn group_id(self) -> &'a str {
        &self.0.group.group_id
    }

    pub fn orbit_id(self) -> &'a str {
        &self.0.group.orbit_id
    }

    pub fn row_indices(self) -> &'a [usize] {
        &self.0.group.row_indices
    }

    pub fn input_orbit_index(self) -> Option<usize> {
        self.0.group.input_orbit_index
    }

    pub fn member_indices(self) -> &'a [usize] {
        &self.0.group.member_indices
    }

    pub fn prepared(self) -> &'a PreparedObservations {
        &self.0.group.prepared
    }

    pub fn iod(self) -> Option<&'a IodResult> {
        self.0.group.iod.as_ref()
    }

    pub fn orbit(self) -> Option<&'a FittedOrbitRecord> {
        self.0.orbit
    }

    pub fn members(self) -> &'a [FittedOrbitMemberRecord] {
        self.0.members
    }

    pub fn failure(self) -> Option<&'a OrbitDeterminationFailure> {
        self.0.failure
    }
}

/// Provider identities used by a canonical batch invocation.
#[derive(Debug, Clone, Default)]
pub struct OdStageProviders {
    pub iod: Option<ProviderMetadata>,
    pub refinement: Option<ProviderMetadata>,
    pub evaluation: Option<ProviderMetadata>,
}

#[derive(Debug, Clone, Default)]
struct WorkingBatchResult {
    groups: Vec<WorkingGroupResult>,
    providers: OdStageProviders,
}

impl WorkingBatchResult {
    fn finish(self) -> OdBatchResult {
        let mut fitted_orbits = Vec::new();
        let mut fitted_orbit_members = Vec::new();
        let mut failures = Vec::new();
        let mut groups = Vec::with_capacity(self.groups.len());
        for group in self.groups {
            let orbit_id = group
                .orbit
                .as_ref()
                .and_then(|record| record.orbit.orbit_id.first())
                .map(|value| value.0.clone())
                .unwrap_or(group.orbit_id);
            let orbit_index = group.orbit.map(|orbit| {
                let index = fitted_orbits.len();
                fitted_orbits.push(orbit);
                index
            });
            let member_start = fitted_orbit_members.len();
            fitted_orbit_members.extend(group.members);
            let member_end = fitted_orbit_members.len();
            let failure_index = group.failure.map(|failure| {
                let index = failures.len();
                failures.push(failure);
                index
            });
            groups.push(OdGroupResult {
                group_id: group.group_id,
                orbit_id,
                row_indices: group.row_indices,
                input_orbit_index: group.input_orbit_index,
                member_indices: group.member_indices,
                prepared: group.prepared,
                orbit_index,
                member_start,
                member_end,
                failure_index,
                iod: group.iod,
            });
        }
        OdBatchResult {
            fitted_orbits,
            fitted_orbit_members,
            failures,
            groups,
            providers: self.providers,
        }
    }
}

/// Exact three-concept batch product retained in deterministic group order.
///
/// Product storage is private so the accessorial group ranges cannot be
/// invalidated after construction. Consumers receive immutable slices or may
/// move all three products out together with [`Self::into_products`]. Edit
/// relational orbit labels only after conversion with [`Self::to_membered_orbits`].
///
/// ```compile_fail
/// use adam_core::od::OdBatchResult;
/// fn cannot_invalidate_products(result: &mut OdBatchResult) {
///     result.fitted_orbits.clear();
/// }
/// ```
///
/// Raw group handles are deliberately absent from the ordinary public API, so
/// a handle from one batch cannot be paired with another batch.
///
/// ```compile_fail
/// use adam_core::od::OdBatchResult;
/// fn cannot_obtain_raw_group_handles(result: &OdBatchResult) {
///     let _ = result.groups();
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct OdBatchResult {
    fitted_orbits: Vec<FittedOrbitRecord>,
    fitted_orbit_members: Vec<FittedOrbitMemberRecord>,
    failures: Vec<OrbitDeterminationFailure>,
    groups: Vec<OdGroupResult>,
    providers: OdStageProviders,
}

fn append_fitted_orbit(output: &mut OrbitBatch, next: &OrbitBatch) -> Result<(), OdBatchError> {
    output.validate()?;
    next.validate()?;
    if next.len() != 1 {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "canonical fitted-orbit record must contain exactly one row",
        ));
    }
    if output.coordinates.frame != next.coordinates.frame
        || output.coordinates.representation() != next.coordinates.representation()
    {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "fitted orbit rows have incompatible coordinate representations",
        ));
    }

    output.orbit_id.extend(next.orbit_id.iter().cloned());
    output.object_id.extend(next.object_id.iter().cloned());
    match (&mut output.coordinates.values, &next.coordinates.values) {
        (CoordinateValues::Cartesian(left), CoordinateValues::Cartesian(right))
        | (CoordinateValues::Spherical(left), CoordinateValues::Spherical(right))
        | (CoordinateValues::Keplerian(left), CoordinateValues::Keplerian(right))
        | (CoordinateValues::Cometary(left), CoordinateValues::Cometary(right))
        | (CoordinateValues::Geodetic(left), CoordinateValues::Geodetic(right)) => {
            left.extend(right.iter().copied());
        }
        _ => unreachable!("representation equality checked above"),
    }
    output
        .coordinates
        .origins
        .origins
        .extend(next.coordinates.origins.origins.iter().cloned());
    match (&mut output.coordinates.times, &next.coordinates.times) {
        (Some(left), Some(right)) if left.scale == right.scale => {
            left.epochs.extend(right.epochs.iter().copied());
        }
        (None, None) => {}
        _ => {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                "fitted orbit rows have incompatible epochs",
            ));
        }
    }
    match (
        &mut output.coordinates.covariance,
        &next.coordinates.covariance,
    ) {
        (Some(left), Some(right))
            if left.dimension == right.dimension && left.units == right.units =>
        {
            let mut validity = (0..left.rows)
                .map(|row| left.is_row_valid(row))
                .collect::<Vec<_>>();
            validity.extend((0..right.rows).map(|row| right.is_row_valid(row)));
            left.values_row_major
                .extend_from_slice(&right.values_row_major);
            left.rows += right.rows;
            left.row_validity =
                (!validity.iter().all(|value| *value)).then(|| Validity::from_bools(&validity));
        }
        (None, None) => {}
        _ => {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                "fitted orbit rows have incompatible covariance layouts",
            ));
        }
    }
    match (&mut output.physical_parameters, &next.physical_parameters) {
        (Some(left), Some(right)) => {
            left.h_v.extend(right.h_v.iter().copied());
            left.h_v_sigma.extend(right.h_v_sigma.iter().copied());
            left.g.extend(right.g.iter().copied());
            left.g_sigma.extend(right.g_sigma.iter().copied());
            left.sigma_eff.extend(right.sigma_eff.iter().copied());
            left.chi2_red.extend(right.chi2_red.iter().copied());
        }
        (None, None) => {}
        _ => {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                "fitted orbit rows have incompatible physical-parameter layouts",
            ));
        }
    }
    match (
        &mut output.non_gravitational_parameters,
        &next.non_gravitational_parameters,
    ) {
        (Some(left), Some(right)) => {
            left.source.extend(right.source.iter().cloned());
            left.a1.extend(right.a1.iter().copied());
            left.a2.extend(right.a2.iter().copied());
            left.a3.extend(right.a3.iter().copied());
            left.aln.extend(right.aln.iter().copied());
            left.nk.extend(right.nk.iter().copied());
            left.nm.extend(right.nm.iter().copied());
            left.nn.extend(right.nn.iter().copied());
            left.r0.extend(right.r0.iter().copied());
        }
        (None, None) => {}
        _ => {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                "fitted orbit rows have incompatible non-gravitational-parameter layouts",
            ));
        }
    }
    output.validate()?;
    Ok(())
}

fn member_batch_from_records<'a>(
    members: impl IntoIterator<Item = &'a FittedOrbitMemberRecord>,
) -> FittedOrbitMembersBatch {
    let members = members.into_iter().collect::<Vec<_>>();
    let astrometry = |used: bool| AstrometryColumns {
        lon: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.lon
                } else {
                    member.original.lon
                })
            })
            .collect(),
        lat: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.lat
                } else {
                    member.original.lat
                })
            })
            .collect(),
        sigma_lon: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.sigma_lon
                } else {
                    member.original.sigma_lon
                })
            })
            .collect(),
        sigma_lat: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.sigma_lat
                } else {
                    member.original.sigma_lat
                })
            })
            .collect(),
        variance_lon: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.variance_lon
                } else {
                    member.original.variance_lon
                })
            })
            .collect(),
        variance_lat: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.variance_lat
                } else {
                    member.original.variance_lat
                })
            })
            .collect(),
        cov_lonlat: members
            .iter()
            .map(|member| {
                Some(if used {
                    member.used.cov_lonlat
                } else {
                    member.original.cov_lonlat
                })
            })
            .collect(),
    };
    FittedOrbitMembersBatch {
        orbit_id: members
            .iter()
            .map(|member| member.orbit_id.clone())
            .collect(),
        observation_id: members
            .iter()
            .map(|member| member.observation_id.clone())
            .collect(),
        outlier: members.iter().map(|member| Some(member.outlier)).collect(),
        input_fingerprint: members
            .iter()
            .map(|member| Some(member.input_fingerprint.clone()))
            .collect(),
        membership_fingerprint: members
            .iter()
            .map(|member| Some(member.membership_fingerprint.clone()))
            .collect(),
        original_astrometry: astrometry(false),
        used_astrometry: astrometry(true),
    }
}

impl OdBatchResult {
    fn group_views(&self) -> impl ExactSizeIterator<Item = GroupView<'_>> + '_ {
        self.groups.iter().map(|group| GroupView {
            group,
            orbit: group.orbit_index.map(|index| &self.fitted_orbits[index]),
            members: &self.fitted_orbit_members[group.member_start..group.member_end],
            failure: group.failure_index.map(|index| &self.failures[index]),
        })
    }

    /// Borrow complete per-group dispositions for Python result marshalling.
    ///
    /// This feature-internal bridge returns self-contained borrowed views; it
    /// never accepts a standalone group handle alongside a separate batch.
    #[cfg(feature = "python-bindings")]
    #[doc(hidden)]
    pub fn python_group_views(&self) -> impl ExactSizeIterator<Item = PythonGroupView<'_>> + '_ {
        self.group_views().map(PythonGroupView)
    }

    pub fn providers(&self) -> &OdStageProviders {
        &self.providers
    }

    /// Materialized fitted-orbit products in deterministic successful-group order.
    pub fn fitted_orbits(&self) -> &[FittedOrbitRecord] {
        &self.fitted_orbits
    }

    /// Materialized membership products in deterministic group/source order.
    pub fn fitted_orbit_members(&self) -> &[FittedOrbitMemberRecord] {
        &self.fitted_orbit_members
    }

    /// Materialized typed scientific failures in deterministic group order.
    pub fn failures(&self) -> &[OrbitDeterminationFailure] {
        &self.failures
    }

    /// Consume the batch and move out its three primary products losslessly.
    pub fn into_products(
        self,
    ) -> (
        Vec<FittedOrbitRecord>,
        Vec<FittedOrbitMemberRecord>,
        Vec<OrbitDeterminationFailure>,
    ) {
        (self.fitted_orbits, self.fitted_orbit_members, self.failures)
    }

    pub fn to_membered_orbits(
        &self,
        observations: &OdObservationBatch,
    ) -> Result<Option<MemberedOrbits>, OdBatchError> {
        let Some(first) = self.fitted_orbits.first() else {
            return Ok(None);
        };
        let mut orbits = first.orbit.clone();
        for fitted in self.fitted_orbits.iter().skip(1) {
            append_fitted_orbit(&mut orbits, &fitted.orbit)?;
        }
        let output = MemberedOrbits {
            orbits,
            provenance: self
                .fitted_orbits
                .iter()
                .map(|fitted| Some(fitted.provenance.clone()))
                .collect(),
            observations: observations.clone(),
            members: member_batch_from_records(&self.fitted_orbit_members),
        };
        validate_member_input(&output, true)?;
        Ok(Some(output))
    }

    pub fn to_membered_groups(
        &self,
        observations: &OdObservationBatch,
    ) -> Result<Vec<MemberedOrbits>, OdBatchError> {
        let mut output = Vec::with_capacity(self.fitted_orbits.len());
        for group in self.group_views() {
            let Some(orbit) = group.orbit else {
                continue;
            };
            let membered = MemberedOrbits {
                orbits: orbit.orbit.clone(),
                provenance: vec![Some(orbit.provenance.clone())],
                observations: observations.clone(),
                members: member_batch_from_records(group.members),
            };
            validate_member_input(&membered, true)?;
            output.push(membered);
        }
        Ok(output)
    }

    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(
            self.groups.is_empty(),
            self.fitted_orbits.is_empty() && self.failures.is_empty()
        );
        self.fitted_orbits.is_empty() && self.failures.is_empty()
    }
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

fn fingerprint_string(digest: &mut impl sha2::digest::Update, value: &str) {
    digest.update(&(value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn fingerprint_optional_string(digest: &mut impl sha2::digest::Update, value: Option<&str>) {
    digest.update(&[u8::from(value.is_some())]);
    if let Some(value) = value {
        fingerprint_string(digest, value);
    }
}

fn fingerprint_optional_f64(digest: &mut impl sha2::digest::Update, value: Option<f64>) {
    digest.update(&[u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(&value.to_bits().to_le_bytes());
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
    observation_id: &str,
    member_fingerprint: &str,
) -> String {
    let mut digest = Sha256::new();
    // v2 deliberately excludes the mutable relational orbit_id label. The
    // immutable association group, observation identity, and prepared member
    // content remain covered.
    digest.update(b"adam-core-od-membership-pair/v2\0");
    for value in [group_id, observation_id, member_fingerprint] {
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
    // Preserve the established IOD hypothesis fingerprint. Orbit-consuming
    // routes use v2, whose scientific identity excludes mutable orbit labels.
    digest.update(if input_orbit.is_some() {
        b"adam-core-od-input/v2\0".as_slice()
    } else {
        b"adam-core-od-input/v1\0".as_slice()
    });
    fingerprint_string(&mut digest, group_id);
    if let Some((orbits, row)) = input_orbit {
        digest.update([1]);
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

fn default_orbit_id(group_id: &str, config_sha256: &str, input_fingerprint: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-orbit/v1\0");
    for value in [group_id, config_sha256, input_fingerprint] {
        fingerprint_string(&mut digest, value);
    }
    format!("{:x}", digest.finalize())
}

fn write_provenance_consistency_payload(
    provenance: &OdProvenance,
    digest: &mut impl sha2::digest::Update,
) {
    digest.update(b"adam-core-od-provenance-consistency/v3\0");
    fingerprint_string(digest, provenance.route.as_str());
    fingerprint_string(digest, provenance.configuration.science_sha256());
    fingerprint_string(digest, provenance.configuration.complete_sha256());
    fingerprint_string(digest, &provenance.input_fingerprint);
    fingerprint_optional_string(digest, provenance.state_fingerprint.as_deref());
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
        fingerprint_optional_string(digest, value.as_deref());
    }
    match provenance.iod_ambiguous {
        Some(value) => digest.update(&[1, u8::from(value)]),
        None => digest.update(&[0]),
    }
    for value in [provenance.iod_alternatives, provenance.num_rejected] {
        match value {
            Some(value) => {
                digest.update(&[1]);
                digest.update(&(value as u64).to_le_bytes());
            }
            None => digest.update(&[0]),
        }
    }
    fingerprint_optional_f64(digest, provenance.rejected_fraction);
}

fn provenance_consistency_digest(provenance: &OdProvenance) -> String {
    let mut digest = Sha256::new();
    write_provenance_consistency_payload(provenance, &mut digest);
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
        stage: OdStage::Iod,
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
    stage: OdStage,
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
        stage: OdStage::Refinement,
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
    pipeline: &ObservationPipeline,
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
                prepare_observations(&observed, &obs_code, &band, &astcat, pipeline, provider)?;
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
                    &input.source.id[source_index],
                    &fingerprint,
                ),
                input_fingerprint: fingerprint,
                original: AstrometryRecord::from_snapshot(&prepared.original, row),
                used: AstrometryRecord::from_snapshot(&prepared.used, row),
                outlier: outlier.get(row).copied().unwrap_or(false),
            }
        })
        .collect()
}

/// Result of static batch preparation before a propagation backend is acquired.
enum BatchPreparation<P> {
    /// The canonical input contains no groups and requires no provider.
    Empty,
    /// Static validation and one-time observation preparation succeeded.
    Ready(P),
}

struct PreparedAssociatedBatch<'a, C> {
    input: &'a AssociatedObservations,
    config: C,
    groups: Vec<AssociationGroup>,
    prepared_groups: Vec<(PreparedObservations, ObserverBatch)>,
}

fn prepare_associated_batch<'a, C: Clone>(
    input: &'a AssociatedObservations,
    config: &C,
    pipeline: &ObservationPipeline,
) -> Result<Option<PreparedAssociatedBatch<'a, C>>, OdBatchError> {
    let groups = plan_associations(&input.observations.source.id, &input.associations)
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidInput, message))?;
    if groups.is_empty() {
        return Ok(None);
    }
    validate_problem(&input.observations)?;
    let prepared_groups =
        prepare_association_groups(&groups, &input.observations, pipeline, &ErfaTimeProvider)?;
    Ok(Some(PreparedAssociatedBatch {
        input,
        config: config.clone(),
        groups,
        prepared_groups,
    }))
}

/// Validated, transformed IOD input whose configuration cannot be swapped.
struct PreparedIod<'a> {
    inner: PreparedAssociatedBatch<'a, IodRunConfig>,
}

/// Perform every provider-independent IOD check and transform exactly once.
fn prepare_iod<'a>(
    input: &'a AssociatedObservations,
    config: &IodRunConfig,
) -> Result<BatchPreparation<PreparedIod<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    Ok(
        match prepare_associated_batch(input, config, config.observation_pipeline())? {
            Some(inner) => BatchPreparation::Ready(PreparedIod { inner }),
            None => BatchPreparation::Empty,
        },
    )
}

impl PreparedIod<'_> {
    /// Execute the prepared IOD batch with metadata derived from `provider`.
    fn execute(self, provider: &OdPropagator<'_>) -> Result<OdBatchResult, OdBatchError> {
        let fit = *self.inner.config.stage().correction_fit();
        with_stage_predictor(provider, fit, |service, context| {
            service.metadata.validate()?;
            run_iod_prepared(
                self.inner.input,
                &self.inner.config,
                service,
                context,
                self.inner.groups,
                self.inner.prepared_groups,
            )
            .map(WorkingBatchResult::finish)
        })
    }
}

fn run_iod_with_source<S: OdPropagatorSource + ?Sized>(
    input: &AssociatedObservations,
    config: &IodRunConfig,
    source: &S,
) -> Result<OdBatchResult, OdBatchError> {
    match prepare_iod(input, config)? {
        BatchPreparation::Empty => Ok(OdBatchResult::default()),
        BatchPreparation::Ready(prepared) => {
            source.with_propagator(|provider| prepared.execute(provider))
        }
    }
}

/// Run canonical batch initial orbit determination (IOD).
///
/// `input` contains typed source/coordinate/observer rows plus explicit
/// association pairs. `config` is a self-contained [`IodRunConfig`]: its
/// [`ObservationPipeline`], verified resources, complete [`IodStageConfig`],
/// and optional immutable profile identity are already bound. `provider` is an
/// explicit propagation-only [`OdPropagator`]; it never owns IOD policy.
///
/// # Ordering and products
///
/// Association groups follow first appearance, successful members follow
/// source-row order, and a detection may belong to multiple groups. The result
/// owns exactly three primary products: [`OdBatchResult::fitted_orbits`],
/// [`OdBatchResult::fitted_orbit_members`], and
/// [`OdBatchResult::failures`]. The immutable slices preserve that order;
/// [`OdBatchResult::into_products`] moves all three out together.
///
/// Empty input returns an empty `OdBatchResult` without resolving an external
/// provider. Safely isolated no-solution outcomes become typed failure records
/// and later groups continue. Invalid IDs/memberships, configuration/resources,
/// provenance, or provider/protocol behavior are invocation-fatal
/// [`OdBatchError`] values and produce no partial result. Static preparation
/// completes before provider acquisition.
///
/// Every emitted orbit carries complete path-free configuration, ordered input,
/// state, provider, diagnostics, and consistency provenance. `orbit_id` remains
/// a mutable relational label; immutable association identity is held by the
/// group/orbit object identity and member fingerprints.
///
/// # Example
///
/// ```no_run
/// use adam_core::{
///     od::{AssociatedObservations, IodRunConfig, IodStageConfig,
///          ObservationPipeline, OdPropagator},
///     run_iod,
/// };
///
/// # fn workflow(input: &AssociatedObservations) -> Result<(), Box<dyn std::error::Error>> {
/// let config = IodRunConfig::explicit(
///     ObservationPipeline::as_supplied(),
///     IodStageConfig::core_v1(),
/// )?;
/// let provider = OdPropagator::two_body();
/// let result = run_iod(input, &config, &provider)?;
/// let (_orbits, _members, _failures) = result.into_products();
/// # Ok(())
/// # }
/// ```
pub fn run_iod(
    input: &AssociatedObservations,
    config: &IodRunConfig,
    provider: &OdPropagator<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    run_iod_with_source(input, config, provider)
}

fn run_iod_prepared(
    input: &AssociatedObservations,
    config: &IodRunConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_>,
    groups: Vec<AssociationGroup>,
    prepared_groups: Vec<(PreparedObservations, ObserverBatch)>,
) -> Result<WorkingBatchResult, OdBatchError> {
    let configuration = CanonicalConfiguration::from_iod(config)?;
    let config_sha = config.configuration_sha256();
    let mut output = WorkingBatchResult {
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
        let orbit_id = default_orbit_id(&group.group_id, &config_sha, &input_fingerprint);
        let result = run_iod_with_predictor(
            service.predictor,
            &prepared,
            &observers,
            config.stage(),
            context.time_provider,
            context.translation_provider,
        );
        match result {
            Ok(iod) => {
                let failure = iod_failure(&group.group_id, &iod, group.source_indices.len());
                let state_fingerprint = output_state_fingerprint(Some(&iod), None, None)?;
                let provenance = make_provenance(
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
                let orbit = if iod.accepted.is_some() {
                    Some(FittedOrbitRecord {
                        orbit: materialized_iod_orbit(&iod, &group.group_id, &orbit_id)?,
                        group_id: group.group_id.clone(),
                        input_orbit_index: None,
                        iod: Some(iod.clone()),
                        refined: None,
                        evaluation: None,
                        provenance,
                    })
                } else {
                    None
                };
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
                output.groups.push(WorkingGroupResult {
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
                    OdStage::Iod,
                    &error,
                    group.source_indices.len(),
                );
                output.groups.push(WorkingGroupResult {
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

/// Validated, transformed exact IOD-plus-refinement input.
struct PreparedOd<'a> {
    inner: PreparedAssociatedBatch<'a, OdConfig>,
}

/// Perform every provider-independent `run_od` check and transform exactly once.
fn prepare_od<'a>(
    input: &'a AssociatedObservations,
    config: &OdConfig,
) -> Result<BatchPreparation<PreparedOd<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    Ok(
        match prepare_associated_batch(input, config, config.observation_pipeline())? {
            Some(inner) => BatchPreparation::Ready(PreparedOd { inner }),
            None => BatchPreparation::Empty,
        },
    )
}

impl PreparedOd<'_> {
    /// Execute the prepared exact composition with one actual backend.
    fn execute(self, provider: &OdPropagator<'_>) -> Result<OdBatchResult, OdBatchError> {
        let iod_fit = *self.inner.config.iod_stage().correction_fit();
        let refinement = *self.inner.config.for_refinement().refinement();
        let refinement_fit = match refinement {
            adam_core_rs_coords::RefinementConfig::WorstResidual(value) => value.fit,
            adam_core_rs_coords::RefinementConfig::Cmc2003(value) => value.fit,
        };
        with_stage_predictors(
            provider,
            iod_fit,
            refinement_fit,
            |iod_service, refinement_service, context| {
                run_od_prepared(self.inner, iod_service, refinement_service, context)
            },
        )
    }
}

fn run_od_with_source<S: OdPropagatorSource + ?Sized>(
    input: &AssociatedObservations,
    config: &OdConfig,
    source: &S,
) -> Result<OdBatchResult, OdBatchError> {
    match prepare_od(input, config)? {
        BatchPreparation::Empty => Ok(OdBatchResult::default()),
        BatchPreparation::Ready(prepared) => {
            source.with_propagator(|provider| prepared.execute(provider))
        }
    }
}

/// Run canonical full OD: one preparation, IOD, then refinement.
///
/// `input` is an [`AssociatedObservations`] batch, `config` is the complete
/// bound [`OdConfig`], and `provider` is the explicit propagation-only backend.
/// Profiles and semantic resource paths must be bound before this call; Rust
/// validates their exact identities and stores only path-free provenance.
///
/// # Composition and ordering
///
/// Rust validates all memberships, applies the configured observation pipeline
/// once, runs IOD, and refines every accepted seed on those exact prepared rows.
/// This is the authoritative `run_od == run_iod + refine` composition: IOD
/// failures are carried forward, successful seeds are refined, and standalone
/// refinement consumes persisted used astrometry rather than rerunning the
/// transforms. Matching direct and composed routes have equal scientific
/// products and canonical OD provenance. Emitted `orbit_id` values are mutable
/// relational labels; immutable association identity remains in orbit/group
/// object identity and member fingerprints.
///
/// Groups follow first association appearance; members follow source order;
/// overlapping hypotheses are retained. The result exposes immutable fitted
/// orbit, fitted-member, and typed-failure slices, or all three owned vectors
/// through [`OdBatchResult::into_products`]. Empty input produces three empty
/// products without provider resolution.
///
/// Invalid static contracts (configuration, resources, IDs, mappings, or
/// covariance dimension) and invocation-fatal provider/protocol failures return
/// [`OdBatchError`] without partial products. Isolated IOD/refinement scientific
/// failures continue by group. A failed refinement emits a state only when Rust
/// retained an actual native best state. Complete configuration/input/state/
/// provider/diagnostic provenance is attached to each emitted orbit.
///
/// # Example
///
/// ```no_run
/// use adam_core::od::{
///     self, AssociatedObservations, IodStageConfig, ObservationPipeline,
///     OdConfig, OdPropagator, RefinementRunConfig,
/// };
///
/// # fn workflow(input: &AssociatedObservations) -> Result<(), Box<dyn std::error::Error>> {
/// let config = OdConfig::explicit(
///     ObservationPipeline::as_supplied(),
///     IodStageConfig::core_v1(),
///     RefinementRunConfig::core_v1(),
/// )?;
/// let provider = OdPropagator::two_body();
/// let result = od::run_od(input, &config, &provider)?;
/// let (_orbits, _members, _failures) = result.into_products();
/// # Ok(())
/// # }
/// ```
pub fn run_od(
    input: &AssociatedObservations,
    config: &OdConfig,
    provider: &OdPropagator<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    run_od_with_source(input, config, provider)
}

fn run_od_prepared(
    prepared: PreparedAssociatedBatch<'_, OdConfig>,
    iod_service: &PredictionService<'_>,
    refinement_service: &PredictionService<'_>,
    context: &OdExecutionContext<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    let PreparedAssociatedBatch {
        input,
        config,
        groups,
        prepared_groups,
    } = prepared;
    iod_service.metadata.validate()?;
    refinement_service.metadata.validate()?;
    let configuration = CanonicalConfiguration::from_od(&config)?;
    let iod_config = config.for_iod();
    let refinement_observers = prepared_groups
        .iter()
        .map(|(_, observers)| observers.clone())
        .collect::<Vec<_>>();
    let mut output = run_iod_prepared(
        input,
        &iod_config,
        iod_service,
        context,
        groups,
        prepared_groups,
    )?;
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
            &config.for_refinement(),
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
                    orbit: materialized_fit_orbit(
                        seed.clone(),
                        &group.group_id,
                        &group.orbit_id,
                        refined.fit.state,
                        refined.fit.covariance,
                    )?,
                    group_id: group.group_id.clone(),
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
                    OdStage::Refinement,
                    &error,
                    group.row_indices.len(),
                ));
                group.orbit = None;
                group.members.clear();
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(output.finish())
}

fn materialized_iod_orbit(
    iod: &IodResult,
    group_id: &str,
    orbit_id: &str,
) -> Result<OrbitBatch, OdBatchError> {
    let mut orbit = iod.seed_orbit()?.ok_or_else(|| {
        OdBatchError::new(
            OdBatchErrorCode::InvalidInput,
            "accepted IOD diagnostics do not contain an orbit",
        )
    })?;
    orbit.orbit_id = vec![OrbitId(orbit_id.to_string())];
    orbit.object_id = vec![Some(ObjectId(group_id.to_string()))];
    orbit.validate()?;
    Ok(orbit)
}

fn materialized_fit_orbit(
    mut orbit: OrbitBatch,
    group_id: &str,
    orbit_id: &str,
    state: [f64; 6],
    covariance: [f64; 36],
) -> Result<OrbitBatch, OdBatchError> {
    orbit.orbit_id = vec![OrbitId(orbit_id.to_string())];
    orbit.object_id = vec![Some(ObjectId(group_id.to_string()))];
    orbit.coordinates.values = CoordinateValues::Cartesian(vec![state]);
    orbit.coordinates.covariance = Some(CovarianceBatch::new(
        1,
        6,
        covariance.to_vec(),
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )?);
    orbit.validate()?;
    Ok(orbit)
}

fn orbit_group_id(orbits: &OrbitBatch, row: usize) -> &str {
    orbits.object_id[row]
        .as_ref()
        .map(|value| value.0.as_str())
        .expect("validated canonical orbit object_id")
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
            let group_id = orbit_group_id(&input.orbits, group.orbit_index);
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

fn validate_iod_seed_provenance(row: usize, provenance: &OdProvenance) -> Result<(), OdBatchError> {
    if provenance.route != OdRoute::RunIod {
        return Ok(());
    }
    if provenance
        .configuration
        .value
        .get("schema")
        .and_then(Value::as_str)
        != Some("adam-core-iod-run-config/v1")
    {
        return Err(OdBatchError::new(
            OdBatchErrorCode::InvalidProvenance,
            format!("seed IOD configuration at orbit row {row} is not a canonical IOD-stage configuration"),
        ));
    }
    provenance
        .configuration
        .declared_science_sha256()
        .ok_or_else(|| {
            OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!(
                    "seed IOD configuration at orbit row {row} has no valid configuration_sha256"
                ),
            )
        })?;
    Ok(())
}

fn validate_member_input(
    input: &MemberedOrbits,
    require_outlier: bool,
) -> Result<Vec<OrbitMemberGroup>, OdBatchError> {
    input.members.validate_lengths(require_outlier)?;
    input
        .orbits
        .validate()
        .map_err(|error| OdBatchError::new(OdBatchErrorCode::InvalidInput, error.to_string()))?;
    let mut group_ids = HashSet::with_capacity(input.orbits.len());
    for (row, object_id) in input.orbits.object_id.iter().enumerate() {
        let group_id = object_id
            .as_ref()
            .map(|value| value.0.as_str())
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                OdBatchError::new(
                    OdBatchErrorCode::InvalidProvenance,
                    format!(
                        "canonical orbit row {row} requires its immutable association group object_id"
                    ),
                )
            })?;
        if !group_ids.insert(group_id) {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                format!("association group object_id {group_id:?} is not unique"),
            ));
        }
    }
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
        let actual = orbit_state_fingerprint(&input.orbits, row, None, None);
        if provenance.state_fingerprint.as_deref() != Some(&actual) {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidProvenance,
                format!("orbit row {row} does not match its state fingerprint"),
            ));
        }
        validate_iod_seed_provenance(row, provenance)?;
    }
    plan_orbit_members(
        &input.orbits,
        &input.observations.source.id,
        &input.members.orbit_id,
        &input.members.observation_id,
    )
    .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidInput, message))
}

struct PreparedMemberBatch<'a> {
    input: &'a MemberedOrbits,
    groups: Vec<OrbitMemberGroup>,
    prepared_groups: Vec<(PreparedObservations, ObserverBatch)>,
}

fn prepare_member_batch<'a>(
    input: &'a MemberedOrbits,
    require_outlier: bool,
) -> Result<Option<PreparedMemberBatch<'a>>, OdBatchError> {
    let groups = validate_member_input(input, require_outlier)?;
    if groups.is_empty() {
        return Ok(None);
    }
    validate_problem(&input.observations)?;
    if require_outlier {
        if let Some(row) = input.members.outlier.iter().position(Option::is_none) {
            return Err(OdBatchError::new(
                OdBatchErrorCode::InvalidInput,
                format!("FittedOrbitMembers outlier is null at row {row}"),
            ));
        }
    }
    let prepared_groups = prepare_member_groups(&groups, input)?;
    Ok(Some(PreparedMemberBatch {
        input,
        groups,
        prepared_groups,
    }))
}

/// Validated refinement input that owns the exact checked configuration.
struct PreparedRefinement<'a> {
    inner: PreparedMemberBatch<'a>,
    config: RefinementRunConfig,
}

/// Validate persisted members, seed identity, and refinement configuration
/// before acquiring a propagation backend.
fn prepare_refinement<'a>(
    input: &'a MemberedOrbits,
    config: &RefinementRunConfig,
) -> Result<BatchPreparation<PreparedRefinement<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    let Some(inner) = prepare_member_batch(input, false)? else {
        return Ok(BatchPreparation::Empty);
    };
    for (group, (prepared, _)) in inner.groups.iter().zip(&inner.prepared_groups) {
        let group_id = orbit_group_id(&input.orbits, group.orbit_index);
        let seed_provenance = input.provenance[group.orbit_index]
            .as_ref()
            .expect("validated provenance");
        if seed_provenance.route == OdRoute::RunIod {
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
    Ok(BatchPreparation::Ready(PreparedRefinement {
        inner,
        config: *config,
    }))
}

impl PreparedRefinement<'_> {
    /// Execute using stage settings and identity derived from `provider`.
    fn execute(self, provider: &OdPropagator<'_>) -> Result<OdBatchResult, OdBatchError> {
        let refinement = *self.config.refinement();
        let fit = match refinement {
            adam_core_rs_coords::RefinementConfig::WorstResidual(value) => value.fit,
            adam_core_rs_coords::RefinementConfig::Cmc2003(value) => value.fit,
        };
        with_stage_predictor(provider, fit, |service, context| {
            refine_prepared(self.inner, &self.config, service, context)
        })
    }
}

fn refine_with_source<S: OdPropagatorSource + ?Sized>(
    input: &MemberedOrbits,
    config: &RefinementRunConfig,
    source: &S,
) -> Result<OdBatchResult, OdBatchError> {
    match prepare_refinement(input, config)? {
        BatchPreparation::Empty => Ok(OdBatchResult::default()),
        BatchPreparation::Ready(prepared) => {
            source.with_propagator(|provider| prepared.execute(provider))
        }
    }
}

/// Refine explicit orbit/member groups using their persisted prepared handoff.
///
/// `input` is a [`MemberedOrbits`] value containing exact states, explicit
/// members, original/used astrometry, source and membership fingerprints, and
/// one validated provenance value per orbit. `config` is an independent,
/// self-contained [`RefinementRunConfig`]. `provider` supplies propagation only.
/// The observation pipeline is not reapplied.
///
/// Groups follow the first appearance of each orbit label in the member table;
/// rows inside a group return to source-observation order. Empty input returns
/// three empty products without provider resolution. Isolated fit failures
/// continue and appear in [`OdBatchResult::failures`]; `input_orbit_id` identifies
/// a seed when no output state exists. Invalid or incomplete memberships,
/// changed source/astrometry, missing or inconsistent provenance, covariance
/// errors, and fatal provider/protocol failures return an invocation-wide
/// [`OdBatchError`] with no partial result. Static membership, source-identity,
/// covariance, and provenance checks complete before provider resolution.
///
/// `orbit_id` is mutable relational data. Relabel only a safe owned
/// [`MemberedOrbits`] value and update its orbit and member labels together;
/// immutable scientific and membership fingerprints deliberately exclude the
/// presentation label. Returned orbits carry the composed or standalone
/// refinement configuration plus input/state/provider/diagnostic provenance.
///
/// # Example
///
/// ```no_run
/// use adam_core::{
///     od::{MemberedOrbits, OdPropagator, RefinementRunConfig},
///     refine,
/// };
///
/// # fn workflow(input: &MemberedOrbits) -> Result<(), Box<dyn std::error::Error>> {
/// let config = RefinementRunConfig::core_v1();
/// let provider = OdPropagator::two_body();
/// let result = refine(input, &config, &provider)?;
/// let fitted: &[_] = result.fitted_orbits();
/// let members: &[_] = result.fitted_orbit_members();
/// let failures: &[_] = result.failures();
/// # let _ = (fitted, members, failures);
/// # Ok(())
/// # }
/// ```
pub fn refine(
    input: &MemberedOrbits,
    config: &RefinementRunConfig,
    provider: &OdPropagator<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    refine_with_source(input, config, provider)
}

fn refine_prepared(
    prepared: PreparedMemberBatch<'_>,
    config: &RefinementRunConfig,
    service: &PredictionService<'_>,
    context: &OdExecutionContext<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    let PreparedMemberBatch {
        input,
        groups,
        prepared_groups,
    } = prepared;
    service.metadata.validate()?;
    let configuration = CanonicalConfiguration::from_refinement(config)?;
    let refinement = *config.refinement();
    let mut output = WorkingBatchResult {
        groups: Vec::with_capacity(groups.len()),
        providers: OdStageProviders {
            refinement: Some(service.metadata.clone()),
            ..OdStageProviders::default()
        },
    };
    for (group, (prepared, observers)) in groups.into_iter().zip(prepared_groups) {
        let group_id = orbit_group_id(&input.orbits, group.orbit_index).to_string();
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
                    CanonicalConfiguration::compose_od(&seed_provenance.configuration, config)?
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
                output.groups.push(WorkingGroupResult {
                    group_id: group_id.clone(),
                    orbit_id: group.orbit_id.clone(),
                    row_indices: group.source_indices,
                    input_orbit_index: Some(group.orbit_index),
                    member_indices: group.member_indices,
                    prepared,
                    orbit: Some(FittedOrbitRecord {
                        orbit: materialized_fit_orbit(
                            seed,
                            &group_id,
                            &group.orbit_id,
                            refined.fit.state,
                            refined.fit.covariance,
                        )?,
                        group_id,
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
                    OdStage::Refinement,
                    &error,
                    group.source_indices.len(),
                );
                output.groups.push(WorkingGroupResult {
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
    Ok(output.finish())
}

/// Validated evaluation input that owns the exact checked configuration.
struct PreparedEvaluation<'a> {
    inner: PreparedMemberBatch<'a>,
    config: EvaluationConfig,
}

/// Validate persisted members and evaluation configuration before acquiring a
/// propagation backend.
fn prepare_evaluation<'a>(
    input: &'a MemberedOrbits,
    config: &EvaluationConfig,
) -> Result<BatchPreparation<PreparedEvaluation<'a>>, OdBatchError> {
    config
        .validate()
        .map_err(|message| OdBatchError::new(OdBatchErrorCode::InvalidConfiguration, message))?;
    Ok(match prepare_member_batch(input, true)? {
        Some(inner) => BatchPreparation::Ready(PreparedEvaluation {
            inner,
            config: *config,
        }),
        None => BatchPreparation::Empty,
    })
}

impl PreparedEvaluation<'_> {
    /// Execute using stage settings and identity derived from `provider`.
    fn execute(self, provider: &OdPropagator<'_>) -> Result<OdBatchResult, OdBatchError> {
        let fit = *self.config.prediction_fit();
        with_stage_predictor(provider, fit, |service, _context| {
            evaluate_prepared(self.inner, &self.config, service)
        })
    }
}

fn evaluate_orbits_with_source<S: OdPropagatorSource + ?Sized>(
    input: &MemberedOrbits,
    config: &EvaluationConfig,
    source: &S,
) -> Result<OdBatchResult, OdBatchError> {
    match prepare_evaluation(input, config)? {
        BatchPreparation::Empty => Ok(OdBatchResult::default()),
        BatchPreparation::Ready(prepared) => {
            source.with_propagator(|provider| prepared.execute(provider))
        }
    }
}

/// Evaluate explicit orbit/member groups without changing their states.
///
/// `input` is a canonical [`MemberedOrbits`] handoff; a validated explicit
/// subset of its members is permitted. `config` is the independent
/// [`EvaluationConfig`] containing the mandatory covariance dimension and exact
/// prediction-fit settings. It is intentionally not an `OdConfig`. `provider`
/// supplies propagation only.
///
/// Groups follow first member appearance and rows follow source order. Empty
/// input returns three empty products without provider resolution. Scientific
/// evaluation failures continue per orbit. Invalid covariance, memberships,
/// source/state/fingerprint/provenance identity, or fatal provider/protocol
/// behavior returns an invocation-wide [`OdBatchError`] and no partial result.
///
/// Evaluation does not refit, transform, or relabel states. Successful records
/// preserve the input state while adding evaluation statistics and complete
/// evaluation-route configuration/provider/diagnostic provenance; prior-stage
/// backend fields are not copied as if they had executed again. `orbit_id`
/// remains mutable relational data, but evaluation does not change it; callers
/// relabel an owned [`MemberedOrbits`] orbit row and its member foreign keys
/// together.
///
/// # Example
///
/// ```no_run
/// use adam_core::od::{self, EvaluationConfig, MemberedOrbits, OdPropagator};
///
/// # fn workflow(input: &MemberedOrbits) -> Result<(), Box<dyn std::error::Error>> {
/// let config = EvaluationConfig::core_v1();
/// let provider = OdPropagator::two_body();
/// let result = od::evaluate_orbits(input, &config, &provider)?;
/// let (_orbits, _members, _failures) = result.into_products();
/// # Ok(())
/// # }
/// ```
pub fn evaluate_orbits(
    input: &MemberedOrbits,
    config: &EvaluationConfig,
    provider: &OdPropagator<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    evaluate_orbits_with_source(input, config, provider)
}

fn evaluate_prepared(
    prepared: PreparedMemberBatch<'_>,
    config: &EvaluationConfig,
    service: &PredictionService<'_>,
) -> Result<OdBatchResult, OdBatchError> {
    let PreparedMemberBatch {
        input,
        groups,
        prepared_groups,
    } = prepared;
    service.metadata.validate()?;
    let configuration = CanonicalConfiguration::from_evaluation(config)?;
    let parameters = config.covariance_dimension() as i64;
    let mut output = WorkingBatchResult {
        groups: Vec::with_capacity(groups.len()),
        providers: OdStageProviders {
            evaluation: Some(service.metadata.clone()),
            ..OdStageProviders::default()
        },
    };
    for (group, (prepared, observers)) in groups.into_iter().zip(prepared_groups) {
        let group_id = orbit_group_id(&input.orbits, group.orbit_index).to_string();
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
                output.groups.push(WorkingGroupResult {
                    group_id: group_id.clone(),
                    orbit_id: group.orbit_id.clone(),
                    row_indices: group.source_indices,
                    input_orbit_index: Some(group.orbit_index),
                    member_indices: group.member_indices,
                    prepared,
                    orbit: Some(FittedOrbitRecord {
                        orbit,
                        group_id,
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
                    OdStage::Evaluation,
                    &error,
                    group.source_indices.len(),
                );
                output.groups.push(WorkingGroupResult {
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
    Ok(output.finish())
}

/// Feature-gated deferred acquisition bridge used only by `adam_core_py`.
///
/// It is excluded from the ordinary Rust API and delegates every scientific
/// operation to the same private preparation/execution implementation as the
/// four primary workflows.
#[cfg(feature = "python-bindings")]
#[doc(hidden)]
pub mod python_bindings {
    use super::*;

    pub trait ProviderSource {
        fn with_propagator<R>(
            &self,
            operation: impl for<'provider> FnOnce(
                &'provider OdPropagator<'provider>,
            ) -> Result<R, OdBatchError>,
        ) -> Result<R, OdBatchError>;
    }

    struct Adapter<'a, S: ?Sized>(&'a S);

    impl<S: ProviderSource + ?Sized> super::OdPropagatorSource for Adapter<'_, S> {
        fn with_propagator<R>(
            &self,
            operation: impl for<'provider> FnOnce(
                &'provider OdPropagator<'provider>,
            ) -> Result<R, OdBatchError>,
        ) -> Result<R, OdBatchError> {
            self.0.with_propagator(operation)
        }
    }

    pub fn run_iod<S: ProviderSource + ?Sized>(
        input: &AssociatedObservations,
        config: &IodRunConfig,
        source: &S,
    ) -> Result<OdBatchResult, OdBatchError> {
        super::run_iod_with_source(input, config, &Adapter(source))
    }

    pub fn run_od<S: ProviderSource + ?Sized>(
        input: &AssociatedObservations,
        config: &OdConfig,
        source: &S,
    ) -> Result<OdBatchResult, OdBatchError> {
        super::run_od_with_source(input, config, &Adapter(source))
    }

    pub fn refine<S: ProviderSource + ?Sized>(
        input: &MemberedOrbits,
        config: &RefinementRunConfig,
        source: &S,
    ) -> Result<OdBatchResult, OdBatchError> {
        super::refine_with_source(input, config, &Adapter(source))
    }

    pub fn evaluate_orbits<S: ProviderSource + ?Sized>(
        input: &MemberedOrbits,
        config: &EvaluationConfig,
        source: &S,
    ) -> Result<OdBatchResult, OdBatchError> {
        super::evaluate_orbits_with_source(input, config, &Adapter(source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_core_rs_coords::propagation::generate_ephemeris_barycentric;
    use adam_core_rs_coords::{Epoch, ObservatoryCode, TimeColumn};
    use adam_core_rs_spice::{SpkWriter, Type9Segment};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static SPICE_FIXTURE_LOCK: Mutex<()> = Mutex::new(());

    struct CountingSource {
        calls: AtomicUsize,
    }

    impl CountingSource {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl OdPropagatorSource for CountingSource {
        fn with_propagator<R>(
            &self,
            operation: impl for<'provider> FnOnce(
                &'provider OdPropagator<'provider>,
            ) -> Result<R, OdBatchError>,
        ) -> Result<R, OdBatchError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            operation(&OdPropagator::two_body())
        }
    }

    struct FurnishedKernel(PathBuf);

    impl Drop for FurnishedKernel {
        fn drop(&mut self) {
            if let Ok(mut backend) = global_backend().lock() {
                backend.unload(&self.0);
            }
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn explicit_config() -> OdConfig {
        OdConfig::explicit(
            ObservationPipeline::as_supplied(),
            IodStageConfig::new(
                IodConfig {
                    max_triplets: Some(3),
                    rchi2_threshold: 50.0,
                    ..IodConfig::default()
                },
                WhitenedFitConfig::default(),
            )
            .expect("IOD config"),
            RefinementRunConfig::core_v1(),
        )
        .expect("explicit OD config")
    }

    fn observation_batch(rows: usize) -> OdObservationBatch {
        let epochs = (0..rows)
            .map(|row| Epoch::new(60_002 + row as i64, 10_000_000_000))
            .collect::<Vec<_>>();
        let times = TimeArray::new(TimeScale::Tdb, epochs.clone()).expect("times");
        let mut covariance = vec![f64::NAN; rows * 36];
        for row in 0..rows {
            covariance[row * 36 + 7] = 1.0e-8;
            covariance[row * 36 + 14] = 1.0e-8;
        }
        let observed = CoordinateBatch::spherical(
            (0..rows)
                .map(|row| {
                    [
                        f64::NAN,
                        10.0 + row as f64,
                        5.0,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                    ]
                })
                .collect(),
            DataFrame::Equatorial,
            OriginArray::repeat(OriginId::from_code("SUN"), rows),
            Some(times.clone()),
            Some(
                CovarianceBatch::new(
                    rows,
                    6,
                    covariance,
                    CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
                )
                .expect("observation covariance"),
            ),
        )
        .expect("observations");
        let codes = (0..rows)
            .map(|row| ObservatoryCode(format!("T{row:02}")))
            .collect::<Vec<_>>();
        let observers = ObserverBatch::new(
            codes.clone(),
            CoordinateBatch::cartesian(
                vec![[0.0; 6]; rows],
                DataFrame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), rows),
                Some(times),
                None,
            )
            .expect("observer coordinates"),
        )
        .expect("observers");
        let time = TimeColumn {
            scale: TimeScale::Tdb,
            days: epochs.iter().map(|epoch| epoch.days).collect(),
            nanos: epochs.iter().map(|epoch| epoch.nanos).collect(),
            validity: None,
        };
        OdObservationBatch {
            source: SourceCatalogBatch {
                id: (0..rows).map(|row| format!("obs-{row}")).collect(),
                exposure_id: vec![None; rows],
                time: time.clone(),
                ra: (0..rows).map(|row| 10.0 + row as f64).collect(),
                dec: vec![5.0; rows],
                ra_sigma: vec![Some(1.0e-4); rows],
                dec_sigma: vec![Some(1.0e-4); rows],
                radec_corr: vec![Some(0.0); rows],
                mag: vec![None; rows],
                mag_sigma: vec![None; rows],
                fwhm: vec![None; rows],
                a: vec![None; rows],
                a_sigma: vec![None; rows],
                b: vec![None; rows],
                b_sigma: vec![None; rows],
                pa: vec![None; rows],
                pa_sigma: vec![None; rows],
                observatory_code: codes.into_iter().map(|code| code.0).collect(),
                filter: vec![None; rows],
                exposure_start_time: time,
                exposure_duration: vec![None; rows],
                exposure_seeing: vec![None; rows],
                exposure_depth_5sigma: vec![None; rows],
                object_id: vec![None; rows],
                catalog_id: vec!["test".to_string(); rows],
                astcat: vec![None; rows],
                photcat: vec![None; rows],
            },
            observed,
            observers,
        }
    }

    struct IdentityTimeProvider;

    impl TimeScaleProvider for IdentityTimeProvider {
        fn rescale(
            &self,
            times: &TimeArray,
            new_scale: TimeScale,
        ) -> Result<TimeArray, SchemaError> {
            TimeArray::new(new_scale, times.epochs.clone())
        }
    }

    struct ZeroTranslationProvider;

    impl OriginTranslationProvider for ZeroTranslationProvider {
        fn origin_translation_vectors(
            &self,
            origins: &OriginArray,
            _target_origin: &OriginId,
            _frame: DataFrame,
            _times: &TimeArray,
        ) -> Result<Vec<[f64; 6]>, SchemaError> {
            Ok(vec![[0.0; 6]; origins.len()])
        }
    }

    fn zero_sun_spk() -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("adam-core-od-zero-sun-{}.bsp", std::process::id()));
        let start_et = (59_000.0 - 51_544.5) * 86_400.0;
        let end_et = (61_000.0 - 51_544.5) * 86_400.0;
        let mut writer = SpkWriter::new_spk("adam-core-od-test");
        writer
            .add_type9(Type9Segment {
                target: 10,
                center: 0,
                frame_id: 1,
                start_et,
                end_et,
                segment_id: "ZERO SUN".to_string(),
                degree: 1,
                states: vec![0.0; 12],
                epochs: vec![start_et, end_et],
            })
            .expect("zero-Sun SPK segment");
        writer.write(&path).expect("zero-Sun SPK");
        global_backend()
            .lock()
            .expect("SPICE backend")
            .furnsh(&path)
            .expect("furnish zero-Sun SPK");
        path
    }

    fn scientific_observation_batch() -> OdObservationBatch {
        const ROWS: usize = 12;
        const NANOS_PER_DAY: f64 = 86_400_000_000_000.0;
        let offsets = (0..ROWS)
            .map(|row| 0.3 + 3.0 * (row / 3) as f64 + 0.2 * (row % 3) as f64)
            .collect::<Vec<_>>();
        let times = TimeArray::new(
            TimeScale::Tdb,
            offsets
                .iter()
                .map(|offset| Epoch::new(60_002, (offset * NANOS_PER_DAY) as i64))
                .collect(),
        )
        .expect("scientific times");
        let observers = ObserverBatch::new(
            (0..ROWS)
                .map(|row| ObservatoryCode(format!("T{row:02}")))
                .collect(),
            CoordinateBatch::cartesian(
                offsets
                    .iter()
                    .map(|offset| {
                        let theta = 0.35 + 0.01725 * offset;
                        [
                            theta.cos(),
                            theta.sin(),
                            0.0,
                            -0.01725 * theta.sin(),
                            0.01725 * theta.cos(),
                            0.0,
                        ]
                    })
                    .collect(),
                DataFrame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), ROWS),
                Some(times.clone()),
                None,
            )
            .expect("scientific observer coordinates"),
        )
        .expect("scientific observers");
        let truth = one_row_orbit("truth", Some("truth-group"));
        let mut truth = truth;
        truth.coordinates.values =
            CoordinateValues::Cartesian(vec![[1.2, 0.1, 0.05, -0.002, 0.016, 0.001]]);
        truth.coordinates.times =
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).expect("truth epoch"));
        let predicted = generate_ephemeris_barycentric(
            &TwoBodyPropagator::default(),
            &truth,
            &observers,
            &EphemerisOptions {
                propagation: PropagationOptions {
                    chunk_size: None,
                    thread_limit: None,
                    epoch_policy: EpochPolicy::CrossProduct,
                    covariance: CovariancePropagation::None,
                },
                output_time_scale: TimeScale::Tdb,
                ..EphemerisOptions::default()
            },
            &IdentityTimeProvider,
            &ZeroTranslationProvider,
        )
        .expect("scientific observations")
        .ephemeris
        .coordinates;
        let mut values = predicted.values.raw_values().to_vec();
        for value in &mut values {
            value[0] = f64::NAN;
            value[3] = f64::NAN;
            value[4] = f64::NAN;
            value[5] = f64::NAN;
        }
        let sigma = 0.5 / 3600.0;
        let mut covariance = vec![f64::NAN; ROWS * 36];
        for row in 0..ROWS {
            covariance[row * 36 + 7] = sigma * sigma;
            covariance[row * 36 + 14] = sigma * sigma;
        }
        let observed = CoordinateBatch::new(
            CoordinateValues::Spherical(values.clone()),
            predicted.frame,
            predicted.origins,
            predicted.times.clone(),
            Some(
                CovarianceBatch::new(
                    ROWS,
                    6,
                    covariance,
                    CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
                )
                .expect("scientific covariance"),
            ),
        )
        .expect("scientific observed coordinates");
        let epochs = predicted.times.expect("predicted times").epochs;
        let time = TimeColumn {
            scale: TimeScale::Tdb,
            days: epochs.iter().map(|epoch| epoch.days).collect(),
            nanos: epochs.iter().map(|epoch| epoch.nanos).collect(),
            validity: None,
        };
        OdObservationBatch {
            source: SourceCatalogBatch {
                id: (0..ROWS).map(|row| format!("science-{row}")).collect(),
                exposure_id: vec![None; ROWS],
                time: time.clone(),
                ra: values.iter().map(|value| value[1]).collect(),
                dec: values.iter().map(|value| value[2]).collect(),
                ra_sigma: vec![Some(sigma); ROWS],
                dec_sigma: vec![Some(sigma); ROWS],
                radec_corr: vec![Some(0.0); ROWS],
                mag: vec![None; ROWS],
                mag_sigma: vec![None; ROWS],
                fwhm: vec![None; ROWS],
                a: vec![None; ROWS],
                a_sigma: vec![None; ROWS],
                b: vec![None; ROWS],
                b_sigma: vec![None; ROWS],
                pa: vec![None; ROWS],
                pa_sigma: vec![None; ROWS],
                observatory_code: observers.code.iter().map(|code| code.0.clone()).collect(),
                filter: vec![None; ROWS],
                exposure_start_time: time,
                exposure_duration: vec![None; ROWS],
                exposure_seeing: vec![None; ROWS],
                exposure_depth_5sigma: vec![None; ROWS],
                object_id: vec![None; ROWS],
                catalog_id: vec!["test".to_string(); ROWS],
                astcat: vec![None; ROWS],
                photcat: vec![None; ROWS],
            },
            observed,
            observers,
        }
    }

    fn empty_membered_orbits() -> MemberedOrbits {
        MemberedOrbits {
            orbits: OrbitBatch::new(
                Vec::new(),
                Vec::new(),
                CoordinateBatch::cartesian(
                    Vec::new(),
                    DataFrame::Ecliptic,
                    OriginArray::new(Vec::new()),
                    None,
                    None,
                )
                .expect("empty coordinates"),
            )
            .expect("empty orbits"),
            provenance: Vec::new(),
            observations: observation_batch(0),
            members: FittedOrbitMembersBatch {
                orbit_id: Vec::new(),
                observation_id: Vec::new(),
                outlier: Vec::new(),
                input_fingerprint: Vec::new(),
                membership_fingerprint: Vec::new(),
                original_astrometry: AstrometryColumns::default(),
                used_astrometry: AstrometryColumns::default(),
            },
        }
    }

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

    fn profile_iod_configuration(profile: profiles::OdProfile) -> CanonicalConfiguration {
        let identity = profile.identity();
        let mut value = json!({
            "schema": "adam-core-iod-run-config/v1",
            "profile": {
                "name": identity.name,
                "version": identity.version,
                "definition_sha256": identity.definition_sha256,
            },
            "transforms": cached_profile_definition(profile.identity().name)
                .expect("known profile definition")
                .transforms
                .clone(),
            "iod": IodStageConfig::core_v1().canonical_value(),
        });
        let mut science = value.clone();
        science
            .as_object_mut()
            .expect("profile configuration object")
            .remove("profile");
        let fingerprint =
            sha256(&serde_json::to_vec(&science).expect("profile configuration is serializable"));
        value
            .as_object_mut()
            .expect("profile configuration object")
            .insert(
                "configuration_sha256".to_string(),
                Value::String(fingerprint),
            );
        CanonicalConfiguration::from_value(value).expect("known profile IOD configuration")
    }

    fn run_od_provenance() -> OdProvenance {
        let mut provenance = OdProvenance {
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

    fn one_row_orbit(orbit_id: &str, object_id: Option<&str>) -> OrbitBatch {
        OrbitBatch::new(
            vec![OrbitId(orbit_id.to_string())],
            vec![object_id.map(|value| ObjectId(value.to_string()))],
            CoordinateBatch::cartesian(
                vec![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                DataFrame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), 1),
                None,
                None,
            )
            .expect("one-row Cartesian coordinates"),
        )
        .expect("one-row orbit")
    }

    #[test]
    fn built_in_profile_validation_is_cached_as_immutable_state() {
        for profile in [profiles::CORE_OD_V1, profiles::ASTEROID_INSTITUTE_OD_V1] {
            let first = cached_profile_definition(profile.identity().name)
                .expect("known profile validates");
            let second = cached_profile_definition(profile.identity().name)
                .expect("known profile remains valid");
            assert!(std::ptr::eq(first, second));
            assert_eq!(first.identity["name"], profile.identity().name);
            assert_eq!(
                first.refinement,
                RefinementRunConfig::core_v1().canonical_value()
            );
            assert_eq!(first.evaluation, json!({"covariance_dimension": 6}));
        }
    }

    #[test]
    fn profile_composition_retains_only_an_exact_immutable_profile_definition() {
        let mut custom_value = RefinementRunConfig::core_v1().canonical_value();
        custom_value["max_iterations"] = json!(7);
        let custom = RefinementRunConfig::from_json(&custom_value.to_string())
            .expect("custom refinement configuration");

        for profile in [profiles::CORE_OD_V1, profiles::ASTEROID_INSTITUTE_OD_V1] {
            let iod = profile_iod_configuration(profile);
            let matching =
                CanonicalConfiguration::compose_od(&iod, &RefinementRunConfig::core_v1())
                    .expect("matching profile composition");
            let matching_value: Value =
                serde_json::from_str(&matching.canonical_json()).expect("matching JSON");
            assert_eq!(matching_value["profile"]["name"], profile.identity().name);

            let cross_config = CanonicalConfiguration::compose_od(&iod, &custom)
                .expect("cross-configuration composition");
            let cross_value: Value =
                serde_json::from_str(&cross_config.canonical_json()).expect("cross JSON");
            assert!(cross_value["profile"].is_null());
            assert_eq!(cross_value["refinement"], custom.canonical_value());

            for mutation in ["unknown", "digest", "pipeline"] {
                let mut forged: Value =
                    serde_json::from_str(&iod.canonical_json()).expect("IOD JSON");
                match mutation {
                    "unknown" => forged["profile"]["name"] = json!("UNKNOWN_OD_V1"),
                    "digest" => forged["profile"]["definition_sha256"] = json!("f".repeat(64)),
                    "pipeline" => forged["transforms"] = json!([]),
                    _ => unreachable!(),
                }
                let error =
                    CanonicalConfiguration::from_value(forged).expect_err("forged profile claim");
                assert_eq!(error.code, OdBatchErrorCode::InvalidProvenance);
                assert!(error.message.contains("profile"));
            }
        }
    }

    #[test]
    fn default_batch_products_and_group_views_are_consistent() {
        let result = OdBatchResult::default();
        assert!(result.is_empty());
        assert!(result.groups.is_empty());
        assert!(result.fitted_orbits().is_empty());
        assert!(result.fitted_orbit_members().is_empty());
        assert!(result.failures().is_empty());
        let (orbits, members, failures) = result.into_products();
        assert!(orbits.is_empty() && members.is_empty() && failures.is_empty());
    }

    #[test]
    fn public_workflows_execute_empty_batches_without_provider_resolution() {
        let config = explicit_config();
        let associated = AssociatedObservations {
            observations: observation_batch(0),
            associations: AssociationBatch {
                detection_id: Vec::new(),
                object_id: Vec::new(),
            },
        };
        let provider = OdPropagator::two_body();
        assert!(run_iod(&associated, &config.for_iod(), &provider)
            .expect("empty IOD")
            .is_empty());
        assert!(run_od(&associated, &config, &provider)
            .expect("empty OD")
            .is_empty());
        let membered = empty_membered_orbits();
        assert!(refine(&membered, &config.for_refinement(), &provider)
            .expect("empty refinement")
            .is_empty());
        assert!(
            evaluate_orbits(&membered, &EvaluationConfig::core_v1(), &provider)
                .expect("empty evaluation")
                .is_empty()
        );
    }

    #[test]
    fn public_iod_and_od_preserve_overlap_order_and_continue_scientific_failures() {
        let observations = observation_batch(3);
        let input = AssociatedObservations {
            associations: AssociationBatch {
                detection_id: vec![
                    "obs-2".to_string(),
                    "obs-0".to_string(),
                    "obs-1".to_string(),
                    "obs-2".to_string(),
                    "obs-1".to_string(),
                ],
                object_id: vec![
                    Some("first".to_string()),
                    Some("first".to_string()),
                    Some("first".to_string()),
                    Some("second".to_string()),
                    Some("second".to_string()),
                ],
            },
            observations,
        };
        let config = explicit_config();
        let provider = OdPropagator::two_body();
        let iod = run_iod(&input, &config.for_iod(), &provider).expect("scientific IOD failures");
        assert_eq!(
            iod.groups
                .iter()
                .map(|group| group.group_id.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(iod.groups[0].row_indices, [0, 1, 2]);
        assert_eq!(iod.groups[1].row_indices, [1, 2]);
        assert!(iod.group_views().all(|group| {
            group.failure.is_some_and(|failure| {
                failure.stage == OdStage::Iod && failure.code == OdGroupFailureCode::NoSolution
            })
        }));
        let composed = run_od(&input, &config, &provider).expect("composed scientific failures");
        assert_eq!(composed.groups.len(), iod.groups.len());
        for (iod_group, composed_group) in iod.group_views().zip(composed.group_views()) {
            assert_eq!(composed_group.group.group_id, iod_group.group.group_id);
            assert_eq!(
                composed_group.group.row_indices,
                iod_group.group.row_indices
            );
            assert_eq!(
                composed_group.failure.map(|failure| &failure.code),
                iod_group.failure.map(|failure| &failure.code)
            );
        }
    }

    #[test]
    fn direct_multi_orbit_label_edits_preserve_science_and_fail_closed() {
        let _fixture_lock = SPICE_FIXTURE_LOCK.lock().expect("SPICE fixture lock");
        let _kernel = FurnishedKernel(zero_sun_spk());
        let observations = scientific_observation_batch();
        let associations = AssociationBatch {
            detection_id: observations
                .source
                .id
                .iter()
                .chain(&observations.source.id)
                .cloned()
                .collect(),
            object_id: std::iter::repeat_n(
                Some("multi-one".to_string()),
                observations.source.len(),
            )
            .chain(std::iter::repeat_n(
                Some("multi-two".to_string()),
                observations.source.len(),
            ))
            .collect(),
        };
        let input = AssociatedObservations {
            observations: observations.clone(),
            associations,
        };
        let config = explicit_config();
        let provider = OdPropagator::two_body();
        let iod = run_iod(&input, &config.for_iod(), &provider).expect("two-group scientific IOD");
        assert_eq!(iod.fitted_orbits().len(), 2);
        assert!(iod.failures().is_empty());
        let (owned_orbits, owned_members, owned_failures) = iod.clone().into_products();
        assert_eq!(owned_orbits.len(), 2);
        assert_eq!(owned_members.len(), iod.fitted_orbit_members().len());
        assert!(owned_failures.is_empty());
        let membered_groups = iod
            .to_membered_groups(&observations)
            .expect("safe per-group conversion");
        assert_eq!(membered_groups.len(), 2);
        for membered in &membered_groups {
            assert_eq!(membered.orbits.len(), 1);
            let orbit_id = &membered.orbits.orbit_id[0].0;
            assert!(membered
                .members
                .orbit_id
                .iter()
                .all(|member_id| member_id == orbit_id));
        }
        for group in iod.group_views() {
            assert_eq!(
                group
                    .orbit
                    .and_then(|record| record.orbit.orbit_id.first())
                    .map(|label| label.0.as_str()),
                Some(group.group.orbit_id.as_str())
            );
        }
        #[cfg(feature = "python-bindings")]
        for group in iod.python_group_views() {
            let orbit = group.orbit().expect("successful group has an orbit");
            assert_eq!(orbit.group_id, group.group_id());
            assert_eq!(
                orbit.orbit.orbit_id[0].0,
                group.orbit_id(),
                "borrowed group view selected another group's orbit"
            );
            assert!(group
                .members()
                .iter()
                .all(|member| member.orbit_id == group.orbit_id()));
            assert!(group.failure().is_none());
        }

        let external_table = two_body_reference::new(TwoBodyPropagator::default());
        // SAFETY: the reference ABI table and its owned context remain alive
        // for every external workflow invocation below.
        let external_backend = unsafe { ExternalPropagator::from_raw(external_table.table()) }
            .expect("reference external backend");
        let external_provider = OdPropagator::external(&external_backend);
        let external_iod =
            run_iod(&input, &config.for_iod(), &external_provider).expect("external IOD");
        assert!(
            external_iod
                .providers()
                .iod
                .as_ref()
                .expect("external metadata")
                .identity_validated_by_capsule
        );
        let external_fused =
            run_od(&input, &config, &external_provider).expect("external fused OD");
        let external_input = external_iod
            .to_membered_orbits(&observations)
            .expect("external conversion")
            .expect("external fitted orbits");
        let external_composed = refine(
            &external_input,
            &config.for_refinement(),
            &external_provider,
        )
        .expect("external composed OD");
        for (fused, composed) in external_fused
            .group_views()
            .zip(external_composed.group_views())
        {
            assert_eq!(
                fused.orbit.expect("external fused orbit").orbit,
                composed.orbit.expect("external composed orbit").orbit
            );
            assert_eq!(fused.members.len(), composed.members.len());
        }

        let original_input = iod
            .to_membered_orbits(&observations)
            .expect("batch conversion")
            .expect("fitted orbits");
        assert_eq!(original_input.orbits.len(), 2);
        let original_ids = original_input
            .orbits
            .orbit_id
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>();
        let original_membership = original_input.members.membership_fingerprint.clone();
        let original_provenance = original_input
            .provenance
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .expect("provenance")
                    .provenance_fingerprint
                    .clone()
            })
            .collect::<Vec<_>>();

        let mut relabelled = original_input.clone();
        relabelled.orbits.orbit_id[0].0 = "user-visible-label".to_string();
        for member_id in &mut relabelled.members.orbit_id {
            if member_id == &original_ids[0] {
                *member_id = "user-visible-label".to_string();
            }
        }
        assert_eq!(relabelled.orbits.orbit_id[1].0, original_ids[1]);
        assert_eq!(
            relabelled.members.membership_fingerprint,
            original_membership
        );
        assert_eq!(
            relabelled
                .provenance
                .iter()
                .map(|value| {
                    value
                        .as_ref()
                        .expect("provenance")
                        .provenance_fingerprint
                        .clone()
                })
                .collect::<Vec<_>>(),
            original_provenance
        );

        let source = CountingSource::new();
        let refined = refine_with_source(&relabelled, &config.for_refinement(), &source)
            .expect("refine directly relabelled seed");
        assert_eq!(source.calls(), 1);
        assert_eq!(refined.groups[0].group_id, "multi-one");
        assert_eq!(refined.groups[0].orbit_id, "user-visible-label");
        assert_eq!(refined.groups[1].group_id, "multi-two");
        assert_eq!(refined.groups[1].orbit_id, original_ids[1]);
        let evaluated = evaluate_orbits(&relabelled, &EvaluationConfig::core_v1(), &provider)
            .expect("evaluate directly relabelled seed");
        assert_eq!(evaluated.groups[0].orbit_id, "user-visible-label");
        assert_eq!(evaluated.groups[1].orbit_id, original_ids[1]);

        let fused = run_od(&input, &config, &provider).expect("fused OD");
        let composed =
            refine(&original_input, &config.for_refinement(), &provider).expect("composed OD");
        for (fused_group, composed_group) in fused.group_views().zip(composed.group_views()) {
            let fused_orbit = fused_group.orbit.expect("fused orbit");
            let composed_orbit = composed_group.orbit.expect("composed orbit");
            assert_eq!(fused_orbit.orbit, composed_orbit.orbit);
            assert_eq!(
                fused_orbit.provenance.provenance_fingerprint,
                composed_orbit.provenance.provenance_fingerprint
            );
            assert_eq!(fused_group.members.len(), composed_group.members.len());
        }

        if let Some(bias) = std::env::var_os("ADAM_CORE_EFCC18_BIAS") {
            let bindings = ResourceBindings::new()
                .bind(ResourceSlot::Efcc18CatalogBiasTable, PathBuf::from(bias));
            let profile = profiles::CORE_OD_V1
                .bind(&bindings)
                .expect("CORE_OD_V1 binding");
            let profile_iod = run_iod(&input, &profile.for_iod(), &provider).expect("profile IOD");
            let profile_fused = run_od(&input, &profile, &provider).expect("profile fused OD");
            let profile_input = profile_iod
                .to_membered_orbits(&observations)
                .expect("profile conversion")
                .expect("profile fitted orbits");
            let profile_composed = refine(&profile_input, &profile.for_refinement(), &provider)
                .expect("profile composed OD");
            for (fused, composed) in profile_fused
                .group_views()
                .zip(profile_composed.group_views())
            {
                assert_eq!(
                    fused.orbit.map(|value| &value.orbit),
                    composed.orbit.map(|value| &value.orbit)
                );
                assert_eq!(fused.members.len(), composed.members.len());
            }
        }

        for invalid in [
            {
                let mut value = original_input.clone();
                value.members.orbit_id[0] = "dangling".to_string();
                value
            },
            {
                let mut value = original_input.clone();
                value.orbits.orbit_id[0].0 = "   ".to_string();
                value
            },
            {
                let mut value = original_input.clone();
                value.orbits.orbit_id[1].0 = value.orbits.orbit_id[0].0.clone();
                value
            },
        ] {
            let source = CountingSource::new();
            assert_eq!(
                refine_with_source(&invalid, &config.for_refinement(), &source)
                    .expect_err("invalid relational labels")
                    .code,
                OdBatchErrorCode::InvalidInput
            );
            assert_eq!(source.calls(), 0, "provider resolved before validation");
        }
    }

    #[test]
    fn stable_enums_round_trip_through_their_public_strings() {
        for code in [
            OdBatchErrorCode::InvalidInput,
            OdBatchErrorCode::InvalidConfiguration,
            OdBatchErrorCode::InvalidProviderIdentity,
            OdBatchErrorCode::InvalidProvenance,
            OdBatchErrorCode::Propagation,
            OdBatchErrorCode::Backend,
            OdBatchErrorCode::Protocol,
            OdBatchErrorCode::Runtime,
        ] {
            assert_eq!(code.as_str().parse::<OdBatchErrorCode>(), Ok(code));
            assert_eq!(code.to_string(), code.as_str());
        }
        for route in [
            OdRoute::RunIod,
            OdRoute::RunOd,
            OdRoute::Refine,
            OdRoute::EvaluateOrbits,
        ] {
            assert_eq!(route.as_str().parse::<OdRoute>(), Ok(route));
            assert_eq!(route.to_string(), route.as_str());
        }
        for stage in [OdStage::Iod, OdStage::Refinement, OdStage::Evaluation] {
            assert_eq!(stage.as_str().parse::<OdStage>(), Ok(stage));
            assert_eq!(stage.to_string(), stage.as_str());
        }
        for code in [
            OdGroupFailureCode::NoSolution,
            OdGroupFailureCode::NotConverged,
            OdGroupFailureCode::Scientific(GroupScientificFailureCode::RankDeficient),
            OdGroupFailureCode::Scientific(GroupScientificFailureCode::NoIncludedObservations),
            OdGroupFailureCode::Scientific(GroupScientificFailureCode::NoFitPass),
        ] {
            assert_eq!(
                code.as_str().parse::<OdGroupFailureCode>(),
                Ok(code.clone())
            );
            assert_eq!(code.to_string(), code.as_str());
        }
    }

    #[test]
    fn iod_seed_provenance_requires_declared_science_identity() {
        let configuration = explicit_config().for_iod();
        let mut provenance = run_od_provenance();
        provenance.route = OdRoute::RunIod;
        provenance.configuration =
            CanonicalConfiguration::from_iod(&configuration).expect("declared configuration");
        validate_iod_seed_provenance(0, &provenance).expect("declared science identity is valid");

        let mut value: Value =
            serde_json::from_str(&configuration.provenance_json()).expect("configuration JSON");
        value
            .as_object_mut()
            .expect("configuration object")
            .remove("configuration_sha256");
        provenance.configuration = CanonicalConfiguration::from_value(value)
            .expect("configuration without declared digest");
        assert_eq!(
            validate_iod_seed_provenance(0, &provenance)
                .expect_err("science digest remains mandatory")
                .code,
            OdBatchErrorCode::InvalidProvenance
        );
    }

    #[test]
    fn materialized_fit_orbit_carries_state_and_covariance() {
        let seed = one_row_orbit("seed", Some("seed-object"));
        let state = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let covariance = std::array::from_fn(|index| index as f64);
        let fitted = materialized_fit_orbit(seed, "group", "fit", state, covariance)
            .expect("materialized fit");
        assert_eq!(fitted.orbit_id[0].0, "fit");
        assert_eq!(
            fitted.object_id[0].as_ref().map(|value| value.0.as_str()),
            Some("group")
        );
        assert_eq!(
            fitted.coordinates.values.cartesian(),
            Some([state].as_slice())
        );
        let stored = fitted.coordinates.covariance.expect("fit covariance");
        assert_eq!(stored.dimension, 6);
        assert_eq!(stored.values_row_major, covariance);
    }

    #[test]
    fn successful_group_converts_losslessly_to_refine_and_evaluate_input() {
        let observations = observation_batch(1);
        let config = explicit_config();
        let prepared = prepare_observations(
            &observations.observed,
            &observations.source.observatory_code,
            &observations.source.filter,
            &observations.source.astcat,
            config.observation_pipeline(),
            &ErfaTimeProvider,
        )
        .expect("prepared observations");
        let orbit = materialized_fit_orbit(
            one_row_orbit("seed", Some("group")),
            "group",
            "fit",
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0; 36],
        )
        .expect("materialized fit");
        let mut provenance = run_od_provenance();
        provenance.state_fingerprint = Some(orbit_state_fingerprint(&orbit, 0, None, None));
        provenance.provenance_fingerprint = provenance_consistency_digest(&provenance);
        let original = AstrometryRecord::from_snapshot(&prepared.original, 0);
        let used = AstrometryRecord::from_snapshot(&prepared.used, 0);
        let result = WorkingBatchResult {
            groups: vec![WorkingGroupResult {
                group_id: "group".to_string(),
                orbit_id: "fit".to_string(),
                row_indices: vec![0],
                input_orbit_index: None,
                member_indices: Vec::new(),
                prepared,
                orbit: Some(FittedOrbitRecord {
                    orbit: orbit.clone(),
                    group_id: "group".to_string(),
                    input_orbit_index: None,
                    iod: None,
                    refined: None,
                    evaluation: None,
                    provenance,
                }),
                members: vec![FittedOrbitMemberRecord {
                    orbit_id: "fit".to_string(),
                    observation_id: "obs-0".to_string(),
                    source_index: 0,
                    input_member_index: None,
                    input_fingerprint: "4".repeat(64),
                    membership_fingerprint: "5".repeat(64),
                    original,
                    used,
                    outlier: false,
                }],
                failure: None,
                iod: None,
            }],
            providers: OdStageProviders::default(),
        }
        .finish();
        let converted = result
            .to_membered_orbits(&observations)
            .expect("conversion")
            .expect("successful group");
        assert_eq!(converted.orbits, orbit);
        assert_eq!(converted.provenance.len(), 1);
        assert_eq!(converted.members.observation_id, ["obs-0"]);
        assert_eq!(
            converted.members.original_astrometry.lon,
            [Some(original.lon)]
        );
        assert_eq!(converted.members.used_astrometry.lon, [Some(used.lon)]);
    }

    #[test]
    fn canonical_two_body_identity_and_options_are_derived_from_typed_config() {
        let mut config = WhitenedFitConfig::default();
        config.two_body.propagation_max_iter += 3;
        config.two_body.propagation_tol *= 2.0;
        config.two_body.lt_tol *= 3.0;
        config.two_body.ephemeris_max_iter += 4;
        config.two_body.ephemeris_tol *= 5.0;
        config.two_body.max_lt_iter += 6;
        let settings = canonical_stage_settings(config).expect("settings");
        assert_eq!(
            settings.two_body.config().max_iter,
            config.two_body.propagation_max_iter
        );
        assert_eq!(
            settings.two_body.config().tol,
            config.two_body.propagation_tol
        );
        assert_eq!(settings.options.lt_tol, config.two_body.lt_tol);
        assert_eq!(
            settings.options.max_iter,
            config.two_body.ephemeris_max_iter
        );
        assert_eq!(settings.options.tol, config.two_body.ephemeris_tol);
        assert_eq!(settings.options.max_lt_iter, config.two_body.max_lt_iter);
        assert!(!settings.options.stellar_aberration);
        assert_eq!(settings.options.output_time_scale, TimeScale::Utc);
        let expected = two_body_provider_metadata(&settings.two_body);
        with_stage_predictor(&OdPropagator::two_body(), config, |service, _context| {
            assert_eq!(service.metadata, expected);
            Ok(())
        })
        .expect("canonical provider binding");
    }

    #[test]
    fn fatal_service_errors_keep_backend_protocol_and_runtime_codes() {
        assert_eq!(
            OdBatchError::from(PropagationError::Backend("coverage".to_string())).code,
            OdBatchErrorCode::Backend
        );
        assert_eq!(
            OdBatchError::from(PropagationError::BackendProtocol("abi".to_string())).code,
            OdBatchErrorCode::Protocol
        );
        assert_eq!(
            OdBatchError::from(PropagationError::ThreadPool("pool".to_string())).code,
            OdBatchErrorCode::Runtime
        );
    }

    #[test]
    fn provenance_consistency_payload_excludes_compiled_commit() {
        #[derive(Default)]
        struct RecordedPayload(Vec<u8>);

        impl sha2::digest::Update for RecordedPayload {
            fn update(&mut self, data: &[u8]) {
                self.0.extend_from_slice(data);
            }
        }

        let provenance = run_od_provenance();
        let mut payload = RecordedPayload::default();
        write_provenance_consistency_payload(&provenance, &mut payload);
        let commit = crate::build_identity().commit.as_bytes();
        assert!(!commit.is_empty());
        assert!(!payload.0.windows(commit.len()).any(|value| value == commit));
        assert_eq!(
            provenance.provenance_fingerprint,
            format!("{:x}", Sha256::digest(&payload.0))
        );
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
