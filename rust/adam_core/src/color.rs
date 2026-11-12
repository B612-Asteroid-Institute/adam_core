//! High-level Rust-native asteroid color estimation.
//!
//! [`estimate_colors`] is the primary public API used by both standalone Rust
//! consumers and the thin Python IPC adapter.  It owns resource initialization,
//! grouping, orbit preparation, geometry, fitting, failure isolation, and typed
//! result assembly.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::{Mutex, RwLockReadGuard, TryLockError};

use adam_core_rs_coords::{
    bandpass_data, bundled_bandpass_data, estimate_color_object, origin_mu_au3_day2,
    propagate_2body_along_arc, validate_color_config, ColorComposition, ColorEstimatorConfig,
    ColorFitBatch, ColorFitRow, ColorFitStatus, ColorObservation, ColorPhaseModel, ColorProjection,
    DataFrame, OrbitBatch, OriginId, SchemaError, SourceCatalogBatch, TimeArray, TimeScale,
    NANOS_PER_DAY,
};
use adam_core_rs_kernel_data::{KernelDataError, Resolver as KernelDataResolver};
use adam_core_rs_spice::{
    global_backend, try_initialize_global_kernels, AdamCoreSpiceBackend,
    GlobalKernelInitializationError, SpiceBackendError,
};
use thiserror::Error;

#[derive(Debug, Clone)]
enum ColorBandpassSource {
    Bundled,
    Directory(PathBuf),
}

/// Owned runtime resources required by [`estimate_colors`].
///
/// Bandpass resources are validated for every call. Observer-geometry kernels
/// and MPC obscodes initialize lazily only when an object reaches observer-state
/// preparation. Kernel paths accumulate and are never unloaded by this API; the
/// first successfully loaded nonempty obscodes table remains authoritative; the
/// embedded bundle is parsed once; and explicit directories cache by path. Once
/// acquired, the SPICE read guard is held through the rest of the estimate, so
/// callers must complete resource mutation before concurrent science.
#[derive(Clone)]
pub struct ColorEstimationResources {
    bandpass_source: ColorBandpassSource,
    observer_geometry_kernel_paths: Vec<PathBuf>,
    mpc_obscodes_json: String,
}

impl fmt::Debug for ColorEstimationResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColorEstimationResources")
            .field("bandpass_source", &self.bandpass_source)
            .field(
                "observer_geometry_kernel_paths",
                &self.observer_geometry_kernel_paths,
            )
            .field("mpc_obscodes_json_bytes", &self.mpc_obscodes_json.len())
            .finish()
    }
}

impl ColorEstimationResources {
    /// Construct resources backed by an explicit bandpass-data directory.
    pub fn new(
        bandpass_data_dir: PathBuf,
        observer_geometry_kernel_paths: Vec<PathBuf>,
        mpc_obscodes_json: String,
    ) -> Self {
        Self {
            bandpass_source: ColorBandpassSource::Directory(bandpass_data_dir),
            observer_geometry_kernel_paths,
            mpc_obscodes_json,
        }
    }

    /// Construct resources backed by the embedded canonical bandpass tables.
    pub fn bundled(
        observer_geometry_kernel_paths: Vec<PathBuf>,
        mpc_obscodes_json: String,
    ) -> Self {
        Self {
            bandpass_source: ColorBandpassSource::Bundled,
            observer_geometry_kernel_paths,
            mpc_obscodes_json,
        }
    }

    /// Resolve embedded bandpasses, observer-geometry kernels, and MPC obscodes.
    ///
    /// Kernels and obscodes follow [`KernelDataResolver`]'s documented override,
    /// installed-Python, cache, and checksummed-download chain.
    pub fn resolve(resolver: &KernelDataResolver) -> Result<Self, KernelDataError> {
        Ok(Self::bundled(
            resolver.observer_geometry_kernel_paths()?,
            resolver.obscodes_json()?,
        ))
    }
}

/// Typed failure from the complete public color-estimation workflow.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ColorEstimationError {
    /// Scientific options, resource paths, or input coordinate semantics are invalid.
    #[error("{0}")]
    InvalidConfiguration(String),
    /// A typed input or time column violates its schema contract.
    #[error("{context}: {source}")]
    Schema {
        /// Operation that was validating or converting the schema.
        context: &'static str,
        /// Underlying typed schema error.
        #[source]
        source: SchemaError,
    },
    /// SPICE resource loading or observer-state preparation failed.
    #[error(transparent)]
    Spice(#[from] SpiceBackendError),
    /// A genuine resource mutation raced active scientific readers.
    #[error(
        "adam-core SPICE backend is busy; kernel mutation is allowed only during initialization"
    )]
    BackendBusy,
    /// The process-global SPICE backend read/write lock was poisoned.
    #[error("adam-core SPICE backend lock is poisoned")]
    BackendPoisoned,
}

impl ColorEstimationError {
    fn schema(context: &'static str, source: SchemaError) -> Self {
        Self::Schema { context, source }
    }

    fn configuration(context: &'static str, source: SchemaError) -> Self {
        match source {
            SchemaError::InvalidRecordBatch(message) => Self::InvalidConfiguration(message),
            source => Self::Schema { context, source },
        }
    }
}

static COLOR_RESOURCE_INITIALIZATION: Mutex<()> = Mutex::new(());

/// Estimate canonical-filter absolute magnitudes and colors per object.
///
/// This is the high-level public Rust counterpart of
/// `adam_core.photometry.estimate_colors`. The semantic arguments have the same
/// order and meaning. The final `resources` argument supplies the installed
/// data that Python resolves implicitly from its environment. Standalone users
/// can obtain those resources with [`ColorEstimationResources::resolve`] and
/// [`crate::kernel_data::Resolver::from_env`]; see `examples/color_estimation.rs`.
///
/// # Process-global resource policy
///
/// Embedded bandpasses are parsed once from compiled bytes; explicit directory
/// bandpasses are cached by path. Batch/configuration validation, empty-orbit
/// isolation, and per-object failures before observer geometry do not initialize
/// SPICE. The first object that reaches observer-state preparation initializes
/// the DE440, EOP predict/historical/high-precision, and ITRF93 frame kernels;
/// UTC-to-TDB conversion is ERFA-owned, so no leapseconds/TLS kernel is needed.
/// Kernel paths accumulate in load order and are not unloaded. The first
/// successfully loaded nonempty MPC obscodes table is retained, so later JSON
/// arguments are ignored. Once initialized, the estimate holds the global SPICE
/// read guard until every remaining object is complete. Identical concurrent
/// first calls serialize initialization and then proceed together; adding
/// different resources while any science call holds a read guard returns
/// [`ColorEstimationError::BackendBusy`]. Initialize the full observer-geometry
/// resource set before concurrent science. A later generic `setup_SPICE()` may
/// add TLS, so it must also complete before concurrent readers. Until the
/// separately tracked ERFA first-use race is fixed, applications must perform
/// one UTC time rescale on a single thread before concurrent calls that contain
/// UTC times.
///
/// Per-object failure strings are Rust-native scientific messages and never
/// contain Python exception-class prefixes.
///
/// # Example
///
/// ```no_run
/// use adam_core::coords::{OrbitBatch, SourceCatalogBatch};
/// use adam_core::kernel_data::Resolver;
/// use adam_core::{estimate_colors, ColorEstimationResources, ColorPhaseModel};
/// # fn load_inputs() -> (SourceCatalogBatch, OrbitBatch) { todo!() }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (observations, orbits) = load_inputs();
/// let resources = ColorEstimationResources::resolve(&Resolver::from_env())?;
/// let fits = estimate_colors(
///     &observations,
///     &orbits,
///     ColorPhaseModel::Hg12Star,
///     true,
///     None,
///     false,
///     None,
///     &resources,
/// )?;
/// for row in &fits.rows {
///     println!("{}: {}", row.object_id, row.status);
/// }
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn estimate_colors(
    observations: &SourceCatalogBatch,
    orbits: &OrbitBatch,
    phi_type: ColorPhaseModel,
    force_g_bounds: bool,
    color_term_composition: Option<ColorComposition>,
    strict_band_mapping: bool,
    filter_projection: Option<BTreeMap<String, String>>,
    resources: &ColorEstimationResources,
) -> Result<ColorFitBatch, ColorEstimationError> {
    observations
        .validate()
        .map_err(|error| ColorEstimationError::schema("invalid SourceCatalog", error))?;
    orbits
        .validate()
        .map_err(|error| ColorEstimationError::schema("invalid Orbits", error))?;

    let projection = match (filter_projection, color_term_composition) {
        (None, None) => None,
        (None, Some(_)) => {
            return Err(ColorEstimationError::InvalidConfiguration(
                "composition is accepted only with an explicit source-to-target projection"
                    .to_string(),
            ))
        }
        (Some(_), None) => {
            return Err(ColorEstimationError::InvalidConfiguration(
                "projection requires exactly one composition: template_id or C/S mix".to_string(),
            ))
        }
        (Some(source_to_target), Some(composition)) => Some(ColorProjection {
            source_to_target,
            composition,
        }),
    };
    let config = ColorEstimatorConfig {
        phase_model: phi_type,
        force_phase_bounds: force_g_bounds,
        strict_band_mapping,
        projection,
    };
    let bandpasses = match &resources.bandpass_source {
        ColorBandpassSource::Bundled => bundled_bandpass_data(),
        ColorBandpassSource::Directory(data_dir) => bandpass_data(data_dir),
    }
    .map_err(|error| ColorEstimationError::configuration("failed to load bandpass data", error))?;
    validate_color_config(bandpasses.as_ref(), &config).map_err(|error| {
        ColorEstimationError::configuration("invalid color configuration", error)
    })?;

    let mut object_order = Vec::new();
    let mut source_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (row, object_id) in observations.object_id.iter().enumerate() {
        let Some(object_id) = object_id else {
            continue;
        };
        if !source_groups.contains_key(object_id) {
            object_order.push(object_id.clone());
        }
        source_groups
            .entry(object_id.clone())
            .or_default()
            .push(row);
    }
    let mut orbit_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (row, object_id) in orbits.object_id.iter().enumerate() {
        if let Some(object_id) = object_id {
            orbit_groups
                .entry(object_id.0.clone())
                .or_default()
                .push(row);
        }
    }

    if orbits.is_empty() {
        return Ok(ColorFitBatch::new(
            object_order
                .into_iter()
                .map(|object_id| {
                    ColorFitRow::terminal(
                        object_id.clone(),
                        source_groups[&object_id].len(),
                        ColorFitStatus::NoOrbit,
                        "No orbit given for this object",
                    )
                })
                .collect(),
        ));
    }

    let source_times = TimeArray::from_parts(
        observations.time.scale,
        observations.time.days.clone(),
        observations.time.nanos.clone(),
    )
    .and_then(|times| times.rescale(TimeScale::Tdb))
    .map_err(|error| {
        ColorEstimationError::schema("cannot convert observation times to TDB", error)
    })?;

    if orbits.coordinates.frame != DataFrame::Ecliptic {
        return Err(ColorEstimationError::InvalidConfiguration(format!(
            "color estimation requires ecliptic orbit coordinates; got {}",
            orbits.coordinates.frame.as_str()
        )));
    }
    if orbits.coordinates.values.cartesian().is_none() {
        return Err(ColorEstimationError::InvalidConfiguration(
            "color estimation requires Cartesian orbits".to_string(),
        ));
    }
    let orbit_times = orbits
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| {
            ColorEstimationError::InvalidConfiguration(
                "color estimation requires orbit epochs".to_string(),
            )
        })?
        .rescale(TimeScale::Tdb)
        .map_err(|error| {
            ColorEstimationError::schema("cannot convert orbit epochs to TDB", error)
        })?;

    let mut output = Vec::with_capacity(object_order.len());
    let mut backend: Option<RwLockReadGuard<'static, AdamCoreSpiceBackend>> = None;
    for object_id in object_order {
        let rows = &source_groups[&object_id];
        let orbit_rows = orbit_groups
            .get(&object_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if orbit_rows.is_empty() {
            output.push(ColorFitRow::terminal(
                object_id,
                rows.len(),
                ColorFitStatus::NoOrbit,
                "No orbit given for this object",
            ));
            continue;
        }
        if orbit_rows.len() != 1 {
            output.push(ColorFitRow::terminal(
                object_id,
                rows.len(),
                ColorFitStatus::MultipleOrbits,
                format!("Expected exactly one orbit, got {}", orbit_rows.len()),
            ));
            continue;
        }

        let orbit_row = orbit_rows[0];
        if orbits
            .non_gravitational_parameters
            .as_ref()
            .is_some_and(|parameters| parameters.row(orbit_row).is_active())
        {
            output.push(ColorFitRow::terminal(
                object_id,
                rows.len(),
                ColorFitStatus::Failed,
                "two-body color estimation does not support active non-gravitational parameters",
            ));
            continue;
        }

        let propagated =
            match propagate_color_object(&source_times, orbits, &orbit_times, orbit_row, rows) {
                Ok(propagated) => propagated,
                Err(error) => {
                    output.push(ColorFitRow::terminal(
                        object_id,
                        rows.len(),
                        ColorFitStatus::Failed,
                        isolated_error_message(error),
                    ));
                    continue;
                }
            };

        if backend.is_none() {
            backend = Some(initialize_resources(resources)?);
        }
        let backend_ref = backend
            .as_deref()
            .expect("SPICE backend initialized before observer-state preparation");
        match prepare_color_observations(
            observations,
            &source_times,
            rows,
            &propagated,
            backend_ref,
        ) {
            Ok(prepared) => output.push(estimate_color_object(
                bandpasses.as_ref(),
                &object_id,
                &prepared,
                &config,
            )),
            Err(error) => output.push(ColorFitRow::terminal(
                object_id,
                rows.len(),
                ColorFitStatus::Failed,
                isolated_error_message(error),
            )),
        }
    }

    Ok(ColorFitBatch::new(output))
}

fn isolated_error_message(error: ColorEstimationError) -> String {
    match error {
        ColorEstimationError::Spice(SpiceBackendError::InvalidObserverSite { message, .. }) => {
            message
        }
        other => other.to_string(),
    }
}

fn initialize_resources(
    resources: &ColorEstimationResources,
) -> Result<RwLockReadGuard<'static, AdamCoreSpiceBackend>, ColorEstimationError> {
    if resources.observer_geometry_kernel_paths.is_empty() {
        return Err(ColorEstimationError::InvalidConfiguration(
            "color estimation requires observer-geometry SPICE kernel paths".to_string(),
        ));
    }
    // Serialize only the initialization phase. Identical concurrent first
    // calls wait here, observe the resources installed by the winner, and then
    // acquire independent read guards. Genuine mutation against an older
    // active scientific reader still fails fast through `try_write`.
    let _initialization = COLOR_RESOURCE_INITIALIZATION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    try_initialize_global_kernels(&resources.observer_geometry_kernel_paths).map_err(|error| {
        match error {
            GlobalKernelInitializationError::Busy => ColorEstimationError::BackendBusy,
            GlobalKernelInitializationError::Poisoned => ColorEstimationError::BackendPoisoned,
            GlobalKernelInitializationError::Backend(error) => ColorEstimationError::Spice(error),
        }
    })?;

    let needs_obscodes = global_backend()
        .read()
        .map_err(|_| ColorEstimationError::BackendPoisoned)?
        .mpc_obscodes_loaded()
        == 0;
    if needs_obscodes {
        if resources.mpc_obscodes_json.is_empty() {
            return Err(ColorEstimationError::InvalidConfiguration(
                "color estimation requires MPC observatory data".to_string(),
            ));
        }
        let mut backend = match global_backend().try_write() {
            Ok(backend) => backend,
            Err(TryLockError::WouldBlock) => return Err(ColorEstimationError::BackendBusy),
            Err(TryLockError::Poisoned(_)) => return Err(ColorEstimationError::BackendPoisoned),
        };
        if backend.mpc_obscodes_loaded() == 0 {
            backend.load_mpc_obscodes(&resources.mpc_obscodes_json)?;
        }
    }

    global_backend()
        .read()
        .map_err(|_| ColorEstimationError::BackendPoisoned)
}

fn propagate_color_object(
    source_times_tdb: &TimeArray,
    orbits: &OrbitBatch,
    orbit_times_tdb: &TimeArray,
    orbit_row: usize,
    source_rows: &[usize],
) -> Result<Vec<[f64; 6]>, ColorEstimationError> {
    let states = orbits
        .coordinates
        .values
        .cartesian()
        .expect("Cartesian representation validated once per estimate_colors call");
    let origin = &orbits.coordinates.origins.origins[orbit_row];
    let heliocentric = matches!(origin, OriginId::Named(code) if code == "SUN")
        || matches!(origin, OriginId::Naif(10));
    if !heliocentric {
        return Err(ColorEstimationError::InvalidConfiguration(
            "color estimation requires heliocentric (SUN-origin) orbit coordinates".to_string(),
        ));
    }
    let mu = origin_mu_au3_day2(origin).map_err(|error| {
        ColorEstimationError::InvalidConfiguration(format!("invalid orbit origin: {error}"))
    })?;
    let epoch = orbit_times_tdb.epochs[orbit_row];
    let dts = source_rows
        .iter()
        .map(|&row| {
            let target = source_times_tdb.epochs[row];
            (target.days - epoch.days) as f64
                + (target.nanos - epoch.nanos) as f64 / NANOS_PER_DAY as f64
        })
        .collect::<Vec<_>>();
    let propagated = propagate_2body_along_arc(states[orbit_row], &dts, mu, 1000, 1.0e-14);
    if propagated
        .iter()
        .flat_map(|state| state.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ColorEstimationError::InvalidConfiguration(
            "two-body propagation produced non-finite state".to_string(),
        ));
    }
    Ok(propagated)
}

fn prepare_color_observations(
    source: &SourceCatalogBatch,
    source_times_tdb: &TimeArray,
    source_rows: &[usize],
    propagated: &[[f64; 6]],
    backend: &AdamCoreSpiceBackend,
) -> Result<Vec<ColorObservation>, ColorEstimationError> {
    let mut unique_codes = Vec::<String>::new();
    let mut code_slots = HashMap::<String, usize>::new();
    let mut slots = Vec::with_capacity(source_rows.len());
    for &row in source_rows {
        let code = &source.observatory_code[row];
        let slot = if let Some(slot) = code_slots.get(code) {
            *slot
        } else {
            let slot = unique_codes.len();
            unique_codes.push(code.clone());
            code_slots.insert(code.clone(), slot);
            slot
        };
        slots.push(slot);
    }
    let observer_times = TimeArray::new(
        TimeScale::Tdb,
        source_rows
            .iter()
            .map(|&row| source_times_tdb.epochs[row])
            .collect(),
    )
    .map_err(|error| ColorEstimationError::schema("invalid observer times", error))?;
    let observer_flat = backend.observer_states_from_codes(
        &unique_codes,
        &slots,
        &observer_times,
        DataFrame::Ecliptic,
        &OriginId::from_code("SUN"),
    )?;

    Ok(source_rows
        .iter()
        .enumerate()
        .map(|(local, &row)| ColorObservation {
            magnitude: source.mag[row],
            magnitude_sigma: source.mag_sigma[row],
            observatory_code: source.observatory_code[row].clone(),
            reported_filter: source.filter[row].clone(),
            object_position: [
                propagated[local][0],
                propagated[local][1],
                propagated[local][2],
            ],
            observer_position: [
                observer_flat[6 * local],
                observer_flat[6 * local + 1],
                observer_flat[6 * local + 2],
            ],
        })
        .collect())
}
