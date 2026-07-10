//! adam-core SPICE backend built directly on the public `spicekit` crate.
//!
//! This crate owns adam-core's kernel registration and multi-kernel
//! dispatch semantics. Python bindings in `adam_core_py` are intentionally
//! thin wrappers over this Rust API so the same backend can be used by a
//! future standalone `adam-core-rs` consumer without crossing Python.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use adam_core_rs_coords::types::{
    CoordinateBatch, Epoch, Frame, OriginArray, OriginId, SchemaError, SchemaResult, TimeArray,
    TimeScale,
};
use adam_core_rs_coords::{
    calculate_moid_batch, rotate_cartesian_time_varying_flat6, transform_values_flat6,
    transform_with_covariance_flat6, Frame as KernelFrame, OriginTranslationProvider,
    Representation,
};
use spicekit::frame::{
    apply_sxform, invert_sxform, j2000_to_eclipj2000, pck_euler_rotation_and_derivative,
    sxform_from_rotation,
};
use spicekit::{
    bodc2n as spicekit_bodc2n, bodn2c as spicekit_bodn2c, parse_text_kernel, BodyBinding,
    FrameAssociation, LeapSecondsKernel, NaifIdError, TextKernelContent, TextKernelError,
};
use thiserror::Error;

pub use spicekit::spk_writer::{
    SpkWriter, SpkWriterError, Type3Record, Type3Segment, Type9Segment,
};
pub use spicekit::{NaifFrame, PckError, PckFile, SpkError, SpkFile};

use rayon::prelude::*;

const ITRF93_FRAME_CODE: i32 = 3000;
/// Below this epoch count the batch kernel evaluators stay serial: the
/// per-call Rayon spawn tax exceeds the Chebyshev evaluation cost for small
/// batches (mirrors the coords-crate kernel policy).
const RAYON_SERIAL_THRESHOLD_EPOCHS: usize = 1024;
pub const KM_PER_AU: f64 = 149_597_870.700;
pub const SECONDS_PER_DAY: f64 = 86_400.0;
pub const EARTH_NAIF_ID: i32 = 399;
pub const EARTH_EQUATORIAL_RADIUS_AU: f64 = 6_378.136_3 / KM_PER_AU;
pub const EARTH_ROTATION_RAD_PER_DAY: f64 = 2.0 * std::f64::consts::PI / 0.997_269_675_925_926;

#[derive(Debug, Error)]
pub enum SpiceBackendError {
    #[error("io error reading kernel {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("cannot open SPK {path}: {source}")]
    SpkOpen {
        path: String,
        #[source]
        source: SpkError,
    },
    #[error("cannot open PCK {path}: {source}")]
    PckOpen {
        path: String,
        #[source]
        source: PckError,
    },
    #[error("cannot parse text kernel {path}: {source}")]
    TextKernel {
        path: String,
        #[source]
        source: TextKernelError,
    },
    #[error("text kernel {path} contains no supported content (no body bindings, no leapseconds, no Earth->ITRF93 frame association)")]
    UnsupportedTextKernel { path: String },
    #[error("{0}")]
    NotCovered(String),
    #[error("unsupported NAIF frame: {0}")]
    UnsupportedFrame(String),
    #[error("invalid observer site {code}: {message}")]
    InvalidObserverSite { code: String, message: String },
    #[error("cannot parse MPC observatory codes: {0}")]
    ObsCodes(String),
    #[error(transparent)]
    DataModel(#[from] SchemaError),
    #[error(transparent)]
    NaifId(#[from] NaifIdError),
}

impl SpiceBackendError {
    pub fn is_not_covered(&self) -> bool {
        matches!(self, SpiceBackendError::NotCovered(_))
    }
}

enum Kernel {
    Spk {
        path: String,
        reader: SpkFile,
    },
    Pck {
        path: String,
        reader: PckFile,
    },
    Text {
        path: String,
        content: TextKernelContent,
    },
}

impl Kernel {
    fn path(&self) -> &str {
        match self {
            Kernel::Spk { path, .. } | Kernel::Pck { path, .. } | Kernel::Text { path, .. } => path,
        }
    }
}

/// Process-local SPICE backend state.
///
/// Kernel precedence matches CSPICE's effective behavior for adam-core's
/// use cases: later-loaded SPK/PCK files are tried first, and text-kernel
/// body bindings are replayed in load order so the most recent binding
/// wins.
pub struct AdamCoreSpiceBackend {
    kernels: Vec<Kernel>,
    custom_names: HashMap<String, i32>,
    obscodes: HashMap<String, GroundObserverSite>,
}

/// Process-global SPICE backend.
///
/// Kernel state (loaded SPK/PCK/text readers, custom name bindings) is a
/// single Rust-owned source of truth for the whole process. Python's
/// `spice_backend.py` / `PyAdamCoreSpiceBackend` and every Rust-native
/// consumer (coordinate transforms, ephemeris, observers, OD) share this one
/// backend, so kernel-load bookkeeping can never desync from what is actually
/// loaded (the root cause behind personal-cmy.23). The backend holds only
/// memory-mapped/parsed readers, which are inherited read-only across a Ray
/// fork, so no per-PID rebuild is required.
static GLOBAL_BACKEND: OnceLock<Mutex<AdamCoreSpiceBackend>> = OnceLock::new();

/// Return the process-global SPICE backend, initializing it empty on first use.
pub fn global_backend() -> &'static Mutex<AdamCoreSpiceBackend> {
    GLOBAL_BACKEND.get_or_init(|| Mutex::new(AdamCoreSpiceBackend::new()))
}

/// Earth-fixed observer described by MPC observatory parallax coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundObserverSite {
    pub code: String,
    pub longitude_deg: f64,
    pub cos_phi: f64,
    pub sin_phi: f64,
}

impl GroundObserverSite {
    pub fn new(
        code: impl Into<String>,
        longitude_deg: f64,
        cos_phi: f64,
        sin_phi: f64,
    ) -> Result<Self, SpiceBackendError> {
        let site = Self {
            code: code.into(),
            longitude_deg,
            cos_phi,
            sin_phi,
        };
        site.validate()?;
        Ok(site)
    }

    pub fn geocenter() -> Self {
        Self {
            code: "500".to_string(),
            longitude_deg: 0.0,
            cos_phi: 0.0,
            sin_phi: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), SpiceBackendError> {
        if self.code.trim().is_empty() {
            return Err(SpiceBackendError::InvalidObserverSite {
                code: self.code.clone(),
                message: "code must not be empty".to_string(),
            });
        }
        if self.longitude_deg.is_finite() && self.cos_phi.is_finite() && self.sin_phi.is_finite() {
            return Ok(());
        }
        Err(SpiceBackendError::InvalidObserverSite {
            code: self.code.clone(),
            message: "longitude, cos_phi, and sin_phi must be finite".to_string(),
        })
    }

    fn itrf93_state(&self) -> [f64; 6] {
        let (sin_lon, cos_lon) = self.longitude_deg.to_radians().sin_cos();
        let unit = [cos_lon * self.cos_phi, sin_lon * self.cos_phi, self.sin_phi];
        let position = scale3(&unit, EARTH_EQUATORIAL_RADIUS_AU);
        let rotation_direction = [unit[1], -unit[0], 0.0];
        let velocity = scale3(
            &rotation_direction,
            -EARTH_ROTATION_RAD_PER_DAY * EARTH_EQUATORIAL_RADIUS_AU,
        );
        [
            position[0],
            position[1],
            position[2],
            velocity[0],
            velocity[1],
            velocity[2],
        ]
    }
}

impl Default for AdamCoreSpiceBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of the native [`AdamCoreSpiceBackend::transform_coordinates`]
/// orchestrator: transformed state values (`n * ncols`, `ncols` = 13 for
/// Keplerian output else 6) and, when the input carried covariance, the
/// propagated covariance (`n * 36`, row-major 6x6 per row).
#[derive(Debug, Clone)]
pub struct TransformOutput {
    pub values: Vec<f64>,
    pub ncols: usize,
    pub covariance: Option<Vec<f64>>,
}

impl AdamCoreSpiceBackend {
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            custom_names: HashMap::new(),
            obscodes: HashMap::new(),
        }
    }

    /// Parse and cache MPC observatory parallax coefficients (the JSON shipped
    /// by the Python `mpc_obscodes` package). Space-based codes with non-finite
    /// geodetics are skipped. Returns the number of ground sites loaded. The
    /// obscodes table lives in the process-global backend so both the
    /// `Observers.from_codes` fast path and the coordinate-transform
    /// orchestrator resolve observatory origins from one shared source.
    pub fn load_mpc_obscodes(&mut self, json: &str) -> Result<usize, SpiceBackendError> {
        self.obscodes = parse_mpc_obscodes(json)?;
        Ok(self.obscodes.len())
    }

    /// Number of MPC observatory ground sites currently loaded.
    pub fn mpc_obscodes_loaded(&self) -> usize {
        self.obscodes.len()
    }

    /// Look up a loaded MPC observatory ground site by code (used by the
    /// transform orchestrator to resolve observatory-code origins).
    pub fn ground_observer_site(&self, code: &str) -> Option<&GroundObserverSite> {
        self.obscodes.get(code)
    }

    /// Single-crossing `Observers.from_codes` state generation: per-row
    /// dictionary-encoded observatory codes (`unique_codes` + `code_indices`
    /// dictionary slots) and `times` -> ground-observer states (flat `N*6`) in
    /// `frame` relative to `origin`. Rows are grouped by code so each distinct
    /// site makes one batched `ground_observer_state` call.
    pub fn observer_states_from_codes(
        &self,
        unique_codes: &[String],
        code_indices: &[usize],
        times: &TimeArray,
        frame: Frame,
        origin: &OriginId,
    ) -> Result<Vec<f64>, SpiceBackendError> {
        if code_indices.len() != times.len() {
            return Err(SchemaError::LengthMismatch {
                field: "observer_states.code_indices".to_string(),
                expected: times.len(),
                actual: code_indices.len(),
            }
            .into());
        }
        if self.obscodes.is_empty() {
            return Err(SpiceBackendError::ObsCodes(
                "MPC observatory codes are not loaded; call load_mpc_obscodes first".to_string(),
            ));
        }
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); unique_codes.len()];
        for (row, &slot) in code_indices.iter().enumerate() {
            groups
                .get_mut(slot)
                .ok_or_else(|| {
                    SpiceBackendError::ObsCodes(
                        "code_indices reference a missing dictionary slot".to_string(),
                    )
                })?
                .push(row);
        }
        let mut out = vec![f64::NAN; code_indices.len() * 6];
        for (slot, rows) in groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let code = &unique_codes[slot];
            let site = self.obscodes.get(code.as_str()).ok_or_else(|| {
                SpiceBackendError::InvalidObserverSite {
                    code: code.clone(),
                    message: "unknown or space-based observatory code".to_string(),
                }
            })?;
            let epochs: Vec<Epoch> = rows.iter().map(|&row| times.epochs[row]).collect();
            let group_times = TimeArray::new(times.scale, epochs)?;
            let coordinates = self.ground_observer_state(site, &group_times, frame, origin)?;
            let values = coordinates.values.cartesian().ok_or_else(|| {
                SpiceBackendError::NotCovered(
                    "ground observer states were not Cartesian".to_string(),
                )
            })?;
            for (group_row, &row) in rows.iter().enumerate() {
                out[row * 6..row * 6 + 6].copy_from_slice(&values[group_row]);
            }
        }
        Ok(out)
    }

    /// Paths of every kernel currently loaded, in load order. The `kernels`
    /// Vec is the registry: `furnsh` is idempotent by path and `unload`
    /// removes by path, so this is the single source of truth for "what is
    /// loaded" (superseding the former Python-side registered-kernels set).
    pub fn registered_kernels(&self) -> Vec<String> {
        self.kernels.iter().map(|k| k.path().to_string()).collect()
    }

    /// Whether `path` is currently loaded.
    pub fn is_registered(&self, path: &str) -> bool {
        self.kernels.iter().any(|k| k.path() == path)
    }

    /// Unload every kernel and clear all custom name bindings. Used for test
    /// isolation (a Rust-level reset of the process-global backend) and any
    /// caller that needs a clean kernel pool.
    pub fn clear(&mut self) {
        self.kernels.clear();
        self.custom_names.clear();
        self.obscodes.clear();
    }

    /// Native `transform_coordinates`: the full composition -- representation
    /// change, origin translation via SPICE (perturber or observatory), and
    /// constant-frame rotation, with covariance forward-AD when covariance is
    /// supplied -- in one call, using this backend for the origin shift. This
    /// is the Rust-native equivalent of the legacy public `transform_coordinates`
    /// (same name per the surface-parity requirement).
    ///
    /// `target_origin = None` preserves each row's input origin (no shift).
    /// Origin-translation vectors are resolved in the INPUT frame and applied
    /// before frame rotation. Returns `Ok(None)` for combinations this native
    /// path does not yet cover (time-varying ITRF93 frames), signalling the
    /// caller to fall back to the legacy composition.
    #[allow(clippy::too_many_arguments)]
    pub fn transform_coordinates(
        &self,
        coords_flat: &[f64],
        covariance_flat: Option<&[f64]>,
        rep_in: Representation,
        rep_out: Representation,
        frame_in: Frame,
        frame_out: Frame,
        origins: &OriginArray,
        target_origin: Option<&OriginId>,
        times: &TimeArray,
        t0: &[f64],
        mu: &[f64],
        a: f64,
        f: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Option<TransformOutput>, SpiceBackendError> {
        // The transform kernels use the crate-level coords `Frame` (the SPICE
        // backend and this method use the schema `types::Frame`) and only
        // support constant ecliptic<->equatorial rotation. Time-varying ITRF93
        // and unspecified frames are not handled by this native path yet and
        // fall back to the legacy composition (`Ok(None)`).
        // Unspecified frames are not transformable natively.
        if frame_in == Frame::Unspecified || frame_out == Frame::Unspecified {
            return Ok(None);
        }
        let to_kernel_frame = |frame: Frame| match frame {
            Frame::Ecliptic => KernelFrame::Ecliptic,
            Frame::Equatorial => KernelFrame::Equatorial,
            Frame::Itrf93 => KernelFrame::Itrf93,
            Frame::Unspecified => unreachable!("unspecified frames handled above"),
        };

        // Origin-translation vectors (resolved in the INPUT frame), applied as
        // a constant offset before frame rotation. Skip when there is no origin
        // change.
        let translation_flat: Option<Vec<f64>> = match target_origin {
            Some(target) if origins.origins.iter().any(|origin| origin != target) => {
                let vectors = self.origin_translation_vectors(origins, target, frame_in, times)?;
                Some(vectors.into_iter().flatten().collect())
            }
            _ => None,
        };

        // An ITRF93 frame CHANGE needs a time-varying (per-epoch) rotation that
        // the constant-frame kernel does not provide; same-frame ITRF93 (no
        // change) and ecliptic<->equatorial go through the constant path below.
        let itrf93_frame_change =
            frame_in != frame_out && (frame_in == Frame::Itrf93 || frame_out == Frame::Itrf93);
        if itrf93_frame_change {
            // Only Cartesian input is supported for the time-varying rotation
            // (matches the legacy _rust_transform_supports contract).
            if rep_in != Representation::Cartesian {
                return Ok(None);
            }
            // Origin shift in the input frame first (covariance-invariant),
            // then the time-varying rotation of state + covariance.
            let mut cart = coords_flat.to_vec();
            if let Some(shift) = translation_flat.as_deref() {
                for (value, delta) in cart.iter_mut().zip(shift.iter()) {
                    *value += delta;
                }
            }
            let (rotated, rotated_cov) =
                self.rotate_time_varying(&cart, covariance_flat, frame_in, frame_out, times)?;
            if rep_out == Representation::Cartesian {
                return Ok(Some(TransformOutput {
                    values: rotated,
                    ncols: 6,
                    covariance: rotated_cov,
                }));
            }
            // Representation conversion in the (now target) frame: identity
            // frame rotation with covariance forward-AD.
            let ktarget = to_kernel_frame(frame_out);
            return Ok(Some(match rotated_cov.as_deref() {
                Some(cov) => {
                    let (values, cov_out) = transform_with_covariance_flat6(
                        &rotated,
                        cov,
                        Representation::Cartesian,
                        rep_out,
                        ktarget,
                        ktarget,
                        t0,
                        mu,
                        a,
                        f,
                        max_iter,
                        tol,
                        None,
                    );
                    TransformOutput {
                        values,
                        ncols: 6,
                        covariance: Some(cov_out),
                    }
                }
                None => {
                    let (values, ncols) = transform_values_flat6(
                        &rotated,
                        Representation::Cartesian,
                        rep_out,
                        ktarget,
                        ktarget,
                        t0,
                        mu,
                        a,
                        f,
                        max_iter,
                        tol,
                        None,
                    )
                    .map_err(SpiceBackendError::NotCovered)?;
                    TransformOutput {
                        values,
                        ncols,
                        covariance: None,
                    }
                }
            }));
        }

        // Constant-frame path: same-frame (any frame, no rotation) or
        // ecliptic<->equatorial constant rotation, plus representation change
        // and covariance forward-AD, in one kernel call.
        let kframe_in = to_kernel_frame(frame_in);
        let kframe_out = to_kernel_frame(frame_out);
        if let Some(cov) = covariance_flat {
            let (values, cov_out) = transform_with_covariance_flat6(
                coords_flat,
                cov,
                rep_in,
                rep_out,
                kframe_in,
                kframe_out,
                t0,
                mu,
                a,
                f,
                max_iter,
                tol,
                translation_flat.as_deref(),
            );
            Ok(Some(TransformOutput {
                values,
                ncols: 6,
                covariance: Some(cov_out),
            }))
        } else {
            let (values, ncols) = transform_values_flat6(
                coords_flat,
                rep_in,
                rep_out,
                kframe_in,
                kframe_out,
                t0,
                mu,
                a,
                f,
                max_iter,
                tol,
                translation_flat.as_deref(),
            )
            .map_err(SpiceBackendError::NotCovered)?;
            Ok(Some(TransformOutput {
                values,
                ncols,
                covariance: None,
            }))
        }
    }

    pub fn furnsh<P: AsRef<Path>>(&mut self, path: P) -> Result<(), SpiceBackendError> {
        let path = path.as_ref();
        let path_s = path.display().to_string();
        if self.kernels.iter().any(|k| k.path() == path_s) {
            return Ok(());
        }

        match peek_daf_idword(path)? {
            Some(idword) if idword.starts_with(b"DAF/SPK") => {
                let reader = SpkFile::open(path).map_err(|source| SpiceBackendError::SpkOpen {
                    path: path_s.clone(),
                    source,
                })?;
                self.kernels.push(Kernel::Spk {
                    path: path_s,
                    reader,
                });
            }
            Some(idword) if idword.starts_with(b"DAF/PCK") => {
                let reader = PckFile::open(path).map_err(|source| SpiceBackendError::PckOpen {
                    path: path_s.clone(),
                    source,
                })?;
                self.kernels.push(Kernel::Pck {
                    path: path_s,
                    reader,
                });
            }
            _ => {
                let content =
                    parse_text_kernel(path).map_err(|source| SpiceBackendError::TextKernel {
                        path: path_s.clone(),
                        source,
                    })?;
                if content.is_empty() {
                    return Err(SpiceBackendError::UnsupportedTextKernel { path: path_s });
                }
                let has_bindings = !content.bindings.is_empty();
                self.kernels.push(Kernel::Text {
                    path: path_s,
                    content,
                });
                if has_bindings {
                    self.rebuild_name_index();
                }
            }
        }
        Ok(())
    }

    pub fn unload<P: AsRef<Path>>(&mut self, path: P) {
        let path_s = path.as_ref().display().to_string();
        let mut had_text = false;
        self.kernels.retain(|k| {
            if k.path() != path_s {
                return true;
            }
            had_text |= matches!(k, Kernel::Text { .. });
            false
        });
        if had_text {
            self.rebuild_name_index();
        }
    }

    pub fn spkez(
        &self,
        target: i32,
        et: f64,
        frame: &str,
        observer: i32,
    ) -> Result<[f64; 6], SpiceBackendError> {
        let states = self.spkez_batch(target, observer, frame, &[et])?;
        Ok([
            states[0], states[1], states[2], states[3], states[4], states[5],
        ])
    }

    pub fn spkez_batch(
        &self,
        target: i32,
        observer: i32,
        frame: &str,
        ets: &[f64],
    ) -> Result<Vec<f64>, SpiceBackendError> {
        self.try_spk_state_batch(target, observer, frame, ets)
            .ok_or_else(|| {
                SpiceBackendError::NotCovered(format!(
                    "rust SPK readers do not cover target={target} observer={observer} frame={frame} across {} epochs",
                    ets.len()
                ))
            })
    }

    pub fn pxform_batch(
        &self,
        frame_from: &str,
        frame_to: &str,
        ets: &[f64],
    ) -> Result<Vec<f64>, SpiceBackendError> {
        self.try_pxform_batch(frame_from, frame_to, ets)
            .ok_or_else(|| {
                SpiceBackendError::NotCovered(format!(
                    "rust PCK readers do not cover pxform({frame_from},{frame_to})"
                ))
            })
    }

    pub fn sxform_batch(
        &self,
        frame_from: &str,
        frame_to: &str,
        ets: &[f64],
    ) -> Result<Vec<f64>, SpiceBackendError> {
        self.try_sxform_batch(frame_from, frame_to, ets)
            .ok_or_else(|| {
                SpiceBackendError::NotCovered(format!(
                    "rust PCK readers do not cover sxform({frame_from},{frame_to})"
                ))
            })
    }

    pub fn bodn2c(&self, name: &str) -> Result<i32, SpiceBackendError> {
        let key = normalize_name(name);
        if let Some(&code) = self.custom_names.get(&key) {
            return Ok(code);
        }
        spicekit_bodn2c(name).map_err(SpiceBackendError::NaifId)
    }

    pub fn bodc2n(&self, code: i32) -> Result<&'static str, SpiceBackendError> {
        spicekit_bodc2n(code).map_err(SpiceBackendError::NaifId)
    }

    pub fn resolve_origin_id(&self, origin: &OriginId) -> Result<i32, SpiceBackendError> {
        match origin {
            OriginId::SolarSystemBarycenter => Ok(0),
            OriginId::Naif(id) => Ok(*id),
            OriginId::Named(name) if name == "SOLAR_SYSTEM_BARYCENTER" => Ok(0),
            OriginId::Named(name) => self.bodn2c(name),
        }
    }

    pub fn frame_pxform_matrices(
        &self,
        frame_from: Frame,
        frame_to: Frame,
        times: &TimeArray,
    ) -> Result<Vec<[[f64; 3]; 3]>, SpiceBackendError> {
        let ets = et_seconds(times)?;
        if let Some(rotation) = static_frame_rotation(frame_from, frame_to) {
            return Ok(vec![rotation; times.len()]);
        }
        let flat = self.pxform_batch(
            spice_frame_name(frame_from)?,
            spice_frame_name(frame_to)?,
            &ets,
        )?;
        Ok(unflatten3(&flat))
    }

    pub fn frame_sxform_matrices(
        &self,
        frame_from: Frame,
        frame_to: Frame,
        times: &TimeArray,
    ) -> Result<Vec<[[f64; 6]; 6]>, SpiceBackendError> {
        let ets = et_seconds(times)?;
        if let Some(rotation) = static_frame_rotation(frame_from, frame_to) {
            return Ok(vec![sxform_from_static_rotation(rotation); times.len()]);
        }
        let flat = self.sxform_batch(
            spice_frame_name(frame_from)?,
            spice_frame_name(frame_to)?,
            &ets,
        )?;
        Ok(unflatten6(&flat))
    }

    /// Time-varying (per-epoch) rotation of Cartesian states and optional
    /// covariance between `frame_in` and `frame_out`. Mirrors the Python
    /// `apply_time_varying_rotation`: fetch the per-epoch 6x6 sxform matrices
    /// (km, km/s), unit-convert each to AU, AU/day, then apply per-row
    /// `M @ state` and `M @ Sigma @ M^T`. Used for ITRF93<->inertial frame
    /// changes (ecliptic<->equatorial reduces to the shared static rotation).
    pub fn rotate_time_varying(
        &self,
        states_flat: &[f64],
        covariance_flat: Option<&[f64]>,
        frame_in: Frame,
        frame_out: Frame,
        times: &TimeArray,
    ) -> Result<(Vec<f64>, Option<Vec<f64>>), SpiceBackendError> {
        if !states_flat.len().is_multiple_of(6) {
            return Err(SpiceBackendError::NotCovered(
                "states length must be a multiple of 6".to_string(),
            ));
        }
        let n = states_flat.len() / 6;
        let matrices = self.frame_sxform_matrices(frame_in, frame_out, times)?;
        if matrices.len() != n {
            return Err(SpiceBackendError::NotCovered(
                "sxform matrix count must match state rows".to_string(),
            ));
        }
        // sxform is km / km-s; states are AU / AU-day. With the diagonal unit
        // matrix U = diag(KM_PER_AU x3, KM_PER_AU/SECONDS_PER_DAY x3), the
        // AU-frame rotation is inv(U) @ M @ U, i.e. per element
        // M_aud[i][j] = M_km[i][j] * u[j] / u[i].
        let c_pos = KM_PER_AU;
        let c_vel = KM_PER_AU / SECONDS_PER_DAY;
        let u = [c_pos, c_pos, c_pos, c_vel, c_vel, c_vel];
        let mut matrices_flat = Vec::with_capacity(n * 36);
        for m in &matrices {
            for i in 0..6 {
                for j in 0..6 {
                    matrices_flat.push(m[i][j] * u[j] / u[i]);
                }
            }
        }
        // One matrix per row, so the per-row index is the identity mapping.
        let time_index: Vec<usize> = (0..n).collect();
        let cov = covariance_flat.unwrap_or(&[]);
        let (rotated, rotated_cov) =
            rotate_cartesian_time_varying_flat6(states_flat, cov, &time_index, &matrices_flat)
                .map_err(|err| SpiceBackendError::NotCovered(err.to_string()))?;
        let rotated_cov = covariance_flat.map(|_| rotated_cov);
        Ok((rotated, rotated_cov))
    }

    pub fn state_vectors(
        &self,
        target: &OriginId,
        origin: &OriginId,
        frame: Frame,
        times: &TimeArray,
    ) -> Result<Vec<[f64; 6]>, SpiceBackendError> {
        let target_code = self.resolve_origin_id(target)?;
        let origin_code = self.resolve_origin_id(origin)?;
        if target_code == origin_code {
            return Ok(vec![[0.0; 6]; times.len()]);
        }

        let ets = et_seconds(times)?;
        let states = self.spkez_batch(target_code, origin_code, spice_frame_name(frame)?, &ets)?;
        Ok(km_kms_to_au_day(&states))
    }

    pub fn state_batch(
        &self,
        target: &OriginId,
        origin: &OriginId,
        frame: Frame,
        times: &TimeArray,
    ) -> Result<CoordinateBatch, SpiceBackendError> {
        let states = self.state_vectors(target, origin, frame, times)?;
        CoordinateBatch::cartesian(
            states,
            frame,
            OriginArray::repeat(origin.clone(), times.len()),
            Some(times.clone()),
            None,
        )
        .map_err(SpiceBackendError::from)
    }

    /// Single-crossing perturber-MOID orchestrator: for each perturber, fetch
    /// its per-epoch state relative to `origin` (spkez) and run the batched
    /// Rust MOID kernel against the `primary` Cartesian orbits, all in Rust.
    /// Returns `(moids, dt_mins)`, each laid out perturber-major then
    /// orbit-minor (`p * n + i`), matching the Python veneer's row assembly.
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_perturber_moids(
        &self,
        primary_flat: &[f64],
        mus: &[f64],
        times: &TimeArray,
        perturbers: &[OriginId],
        frame: Frame,
        origin: &OriginId,
        max_iter: usize,
        xtol: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), SpiceBackendError> {
        if !primary_flat.len().is_multiple_of(6) {
            return Err(SpiceBackendError::NotCovered(
                "primary states must be a multiple of 6".to_string(),
            ));
        }
        let n = primary_flat.len() / 6;
        if mus.len() != n {
            return Err(SpiceBackendError::NotCovered(
                "mus length must match primary orbit rows".to_string(),
            ));
        }
        if times.len() != n {
            return Err(SpiceBackendError::NotCovered(
                "times length must match primary orbit rows".to_string(),
            ));
        }
        let mut moids = Vec::with_capacity(perturbers.len() * n);
        let mut dt_mins = Vec::with_capacity(perturbers.len() * n);
        for perturber in perturbers {
            let secondary = self.state_vectors(perturber, origin, frame, times)?;
            let secondary_flat: Vec<f64> = secondary.iter().flatten().copied().collect();
            let (m, dt) = calculate_moid_batch(primary_flat, &secondary_flat, mus, max_iter, xtol);
            moids.extend(m);
            dt_mins.extend(dt);
        }
        Ok((moids, dt_mins))
    }

    pub fn origin_translation_vectors(
        &self,
        origins: &OriginArray,
        target_origin: &OriginId,
        frame: Frame,
        times: &TimeArray,
    ) -> Result<Vec<[f64; 6]>, SpiceBackendError> {
        if origins.len() != times.len() {
            return Err(SchemaError::LengthMismatch {
                field: "origin_translation.origins".to_string(),
                expected: times.len(),
                actual: origins.len(),
            }
            .into());
        }

        let target_code = self.resolve_origin_id(target_origin)?;
        let all_ets = et_seconds(times)?;
        let mut out = vec![[0.0; 6]; times.len()];
        for (source, indices) in unique_origin_indices(origins) {
            // Perturber origins (NAIF bodies / OriginCodes) resolve to a NAIF
            // id and translate via spkez; MPC observatory-code origins do not
            // resolve as bodies and instead translate via the loaded obscodes
            // ground-observer state. This mirrors the Python
            // _resolve_origin_translation_vectors dispatch (perturber first,
            // then observatory) so the transform orchestrator can shift either
            // kind of origin natively.
            match self.resolve_origin_id(&source) {
                Ok(source_code) => {
                    if source_code == target_code {
                        continue;
                    }
                    let ets = indices.iter().map(|&i| all_ets[i]).collect::<Vec<_>>();
                    let flat =
                        self.spkez_batch(source_code, target_code, spice_frame_name(frame)?, &ets)?;
                    let states = km_kms_to_au_day(&flat);
                    for (state, &row) in states.iter().zip(indices.iter()) {
                        out[row] = *state;
                    }
                }
                Err(_) => {
                    let code = source.code();
                    let site = self.obscodes.get(&code).ok_or_else(|| {
                        SpiceBackendError::InvalidObserverSite {
                            code: code.clone(),
                            message: "unknown origin: not a NAIF body or loaded observatory code"
                                .to_string(),
                        }
                    })?;
                    let epochs: Vec<Epoch> = indices.iter().map(|&i| times.epochs[i]).collect();
                    let group_times = TimeArray::new(times.scale, epochs)?;
                    let coords =
                        self.ground_observer_state(site, &group_times, frame, target_origin)?;
                    let values = coords.values.cartesian().ok_or_else(|| {
                        SpiceBackendError::NotCovered(
                            "observer states were not Cartesian".to_string(),
                        )
                    })?;
                    for (state, &row) in values.iter().zip(indices.iter()) {
                        out[row] = *state;
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn coordinate_origin_translation_vectors(
        &self,
        coordinates: &CoordinateBatch,
        target_origin: &OriginId,
    ) -> Result<Vec<[f64; 6]>, SpiceBackendError> {
        let times = coordinates
            .times
            .as_ref()
            .ok_or_else(|| SchemaError::MissingRequiredField("coordinates.time".to_string()))?;
        self.origin_translation_vectors(
            &coordinates.origins,
            target_origin,
            coordinates.frame,
            times,
        )
    }

    pub fn ground_observer_state(
        &self,
        site: &GroundObserverSite,
        times: &TimeArray,
        frame: Frame,
        origin: &OriginId,
    ) -> Result<CoordinateBatch, SpiceBackendError> {
        site.validate()?;
        spice_frame_name(frame)?;

        let topocentric = site.itrf93_state();

        // Deduplicate epochs so the expensive SPICE lookups (Earth ephemeris
        // via spkez, ITRF93->frame rotation via pxform) run once per DISTINCT
        // epoch and scatter back to rows. Mirrors the legacy
        // get_mpc_observer_state vectorization (np.unique + inverse indices):
        // many rows sharing exposure epochs (surveys / ephemeris grids) no
        // longer repeat identical per-row SPICE calls, which is what dominates
        // this SPICE-bound path.
        let (unique_times, inverse) = dedup_epochs(times)?;
        // When every epoch is distinct the dedup collapses nothing and the
        // scatter is the identity, so we can use the per-epoch states directly
        // and avoid an extra O(n) copy pass -- keeping the common
        // per-observation (all-unique) case regression-free.
        let all_unique = unique_times.len() == times.len();

        let mut states = if frame == Frame::Itrf93 {
            vec![topocentric; times.len()]
        } else {
            let earth = self.state_vectors(&earth_origin(), origin, frame, &unique_times)?;
            let rotations = self.frame_pxform_matrices(Frame::Itrf93, frame, &unique_times)?;
            let unique_states: Vec<[f64; 6]> = earth
                .iter()
                .zip(rotations.iter())
                .map(|(earth_state, rotation)| {
                    let position =
                        rotate3(rotation, &[topocentric[0], topocentric[1], topocentric[2]]);
                    let velocity =
                        rotate3(rotation, &[topocentric[3], topocentric[4], topocentric[5]]);
                    let mut state = *earth_state;
                    add_position_velocity(&mut state, &position, &velocity);
                    state
                })
                .collect();
            if all_unique {
                unique_states
            } else {
                inverse.iter().map(|&slot| unique_states[slot]).collect()
            }
        };

        if frame == Frame::Itrf93 && self.resolve_origin_id(origin)? != EARTH_NAIF_ID {
            let unique_earth = self.state_vectors(&earth_origin(), origin, frame, &unique_times)?;
            for (state, &slot) in states.iter_mut().zip(inverse.iter()) {
                add_state(state, &unique_earth[slot]);
            }
        }

        CoordinateBatch::cartesian(
            states,
            frame,
            OriginArray::repeat(origin.clone(), times.len()),
            Some(times.clone()),
            None,
        )
        .map_err(SpiceBackendError::from)
    }

    fn rebuild_name_index(&mut self) {
        self.custom_names.clear();
        for k in &self.kernels {
            if let Kernel::Text { content, .. } = k {
                for binding in &content.bindings {
                    self.custom_names
                        .insert(normalize_name(&binding.name), binding.code);
                }
            }
        }
    }

    /// Most recently loaded structured `KPL/LSK` leapseconds-kernel content,
    /// or `None` if no LSK has been loaded. adam-core's `Timestamp.rescale()`
    /// uses ERFA for time-scale conversions, so this accessor is exposed for
    /// downstream introspection rather than being consumed internally.
    pub fn leapseconds(&self) -> Option<&LeapSecondsKernel> {
        self.kernels.iter().rev().find_map(|k| match k {
            Kernel::Text { content, .. } => content.leapseconds.as_ref(),
            _ => None,
        })
    }

    /// All Earth->ITRF93-style body-fixed frame associations parsed from
    /// loaded text kernels, in load order.
    pub fn frame_associations(&self) -> Vec<&FrameAssociation> {
        self.kernels
            .iter()
            .filter_map(|k| match k {
                Kernel::Text { content, .. } => Some(&content.frame_associations),
                _ => None,
            })
            .flat_map(|v| v.iter())
            .collect()
    }

    fn try_spk_state_batch(
        &self,
        target: i32,
        center: i32,
        frame: &str,
        ets: &[f64],
    ) -> Option<Vec<f64>> {
        let readers: Vec<&SpkFile> = self
            .kernels
            .iter()
            .rev()
            .filter_map(|k| match k {
                Kernel::Spk { reader, .. } => Some(reader),
                _ => None,
            })
            .collect();
        if readers.is_empty() {
            return None;
        }

        if matches!(frame, "J2000" | "ECLIPJ2000") {
            let out_frame = if frame == "J2000" {
                NaifFrame::J2000
            } else {
                NaifFrame::EclipJ2000
            };
            for reader in readers {
                let evaluate = |&et: &f64| {
                    if frame == "J2000" {
                        reader.state(target, center, et).map_err(|_| ())
                    } else {
                        reader
                            .state_in_frame(target, center, et, out_frame)
                            .map_err(|_| ())
                    }
                };
                // Chebyshev evaluation is read-only per epoch; parallelize
                // large batches (multi-thread gate policy) and stay serial
                // below the spawn-tax threshold.
                let states: Result<Vec<_>, ()> = if ets.len() >= RAYON_SERIAL_THRESHOLD_EPOCHS {
                    ets.par_iter().map(evaluate).collect()
                } else {
                    ets.iter().map(evaluate).collect()
                };
                if let Ok(states) = states {
                    let mut out = Vec::with_capacity(ets.len() * 6);
                    for state in states {
                        out.extend_from_slice(&state);
                    }
                    return Some(out);
                }
            }
            return None;
        }

        if frame == "ITRF93" {
            let sxform = self.try_sxform_batch("J2000", "ITRF93", ets)?;
            for reader in readers {
                let mut out = Vec::with_capacity(ets.len() * 6);
                let mut covered = true;
                for (i, &et) in ets.iter().enumerate() {
                    match reader.state(target, center, et) {
                        Ok(s) => {
                            let m = slice_to_mat6(&sxform[i * 36..(i + 1) * 36]);
                            let rotated = apply_sxform(&m, &s);
                            out.extend_from_slice(&rotated);
                        }
                        Err(_) => {
                            covered = false;
                            break;
                        }
                    }
                }
                if covered {
                    return Some(out);
                }
            }
        }

        None
    }

    fn try_pxform_batch(&self, frame_from: &str, frame_to: &str, ets: &[f64]) -> Option<Vec<f64>> {
        if frame_from != "ITRF93" && frame_to != "ITRF93" {
            return None;
        }
        let sx = self.try_sxform_batch(frame_from, frame_to, ets)?;
        let mut out = Vec::with_capacity(ets.len() * 9);
        for i in 0..ets.len() {
            let base = i * 36;
            for r in 0..3 {
                for c in 0..3 {
                    out.push(sx[base + r * 6 + c]);
                }
            }
        }
        Some(out)
    }

    fn try_sxform_batch(&self, frame_from: &str, frame_to: &str, ets: &[f64]) -> Option<Vec<f64>> {
        if frame_from != "ITRF93" && frame_to != "ITRF93" {
            return None;
        }
        for reader in self.kernels.iter().rev().filter_map(|k| match k {
            Kernel::Pck { reader, .. } => Some(reader),
            _ => None,
        }) {
            let evaluate =
                |&et: &f64| pck_sxform_matrix(reader, frame_from, frame_to, et).map_err(|_| ());
            let matrices: Result<Vec<_>, ()> = if ets.len() >= RAYON_SERIAL_THRESHOLD_EPOCHS {
                ets.par_iter().map(evaluate).collect()
            } else {
                ets.iter().map(evaluate).collect()
            };
            if let Ok(matrices) = matrices {
                let mut out = Vec::with_capacity(ets.len() * 36);
                for m in matrices {
                    out.extend(m.iter().flat_map(|row| row.iter()).copied());
                }
                return Some(out);
            }
        }
        None
    }
}

impl OriginTranslationProvider for AdamCoreSpiceBackend {
    fn origin_translation_vectors(
        &self,
        origins: &OriginArray,
        target_origin: &OriginId,
        frame: Frame,
        times: &TimeArray,
    ) -> SchemaResult<Vec<[f64; 6]>> {
        // Method-call syntax resolves to the inherent method (inherent methods
        // take precedence over trait methods), so this delegates to the SPICE
        // implementation and maps its error into the coords SchemaError surface.
        self.origin_translation_vectors(origins, target_origin, frame, times)
            .map_err(|err| SchemaError::InvalidRecordBatch(err.to_string()))
    }
}

/// Parse the MPC extended observatory-codes JSON (as shipped by the Python
/// `mpc_obscodes` package: an object keyed by observatory code whose values are
/// `{ "Longitude": f64, "cos": f64, "sin": f64, "Name": str }`) into
/// ground-observer sites. Space-based codes with null/non-finite geodetic
/// fields are skipped. This is the Rust-native source of topocentric parallax
/// coefficients, replacing the Python `Observers`/`get_observer_state` lookup;
/// the resulting [`GroundObserverSite`] feeds [`AdamCoreSpiceBackend::ground_observer_state`].
pub fn parse_mpc_obscodes(
    json: &str,
) -> Result<HashMap<String, GroundObserverSite>, SpiceBackendError> {
    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_str(json).map_err(|err| SpiceBackendError::ObsCodes(err.to_string()))?;
    let mut sites = HashMap::with_capacity(raw.len());
    for (code, entry) in raw {
        let longitude = entry.get("Longitude").and_then(serde_json::Value::as_f64);
        let cos_phi = entry.get("cos").and_then(serde_json::Value::as_f64);
        let sin_phi = entry.get("sin").and_then(serde_json::Value::as_f64);
        let (Some(longitude), Some(cos_phi), Some(sin_phi)) = (longitude, cos_phi, sin_phi) else {
            continue;
        };
        if !(longitude.is_finite() && cos_phi.is_finite() && sin_phi.is_finite()) {
            continue;
        }
        let site = GroundObserverSite::new(code.clone(), longitude, cos_phi, sin_phi)?;
        sites.insert(code, site);
    }
    Ok(sites)
}

pub fn builtin_bodn2c(name: &str) -> Result<i32, SpiceBackendError> {
    spicekit_bodn2c(name).map_err(SpiceBackendError::NaifId)
}

pub fn builtin_bodc2n(code: i32) -> Result<&'static str, SpiceBackendError> {
    spicekit_bodc2n(code).map_err(SpiceBackendError::NaifId)
}

pub fn parse_text_kernel_bindings<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<BodyBinding>, SpiceBackendError> {
    let path_ref = path.as_ref();
    parse_text_kernel(path_ref)
        .map(|content| content.bindings)
        .map_err(|source| SpiceBackendError::TextKernel {
            path: path_ref.display().to_string(),
            source,
        })
}

pub fn parse_text_kernel_content<P: AsRef<Path>>(
    path: P,
) -> Result<TextKernelContent, SpiceBackendError> {
    let path_ref = path.as_ref();
    parse_text_kernel(path_ref).map_err(|source| SpiceBackendError::TextKernel {
        path: path_ref.display().to_string(),
        source,
    })
}

fn peek_daf_idword(path: &Path) -> Result<Option<[u8; 8]>, SpiceBackendError> {
    let mut file = File::open(path).map_err(|source| SpiceBackendError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut idword = [0u8; 8];
    match file.read_exact(&mut idword) {
        Ok(()) => {
            if idword.starts_with(b"DAF/") {
                Ok(Some(idword))
            } else {
                Ok(None)
            }
        }
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(source) => Err(SpiceBackendError::Io {
            path: path.display().to_string(),
            source,
        }),
    }
}

pub fn pck_sxform_matrix(
    reader: &PckFile,
    frame_from: &str,
    frame_to: &str,
    et: f64,
) -> Result<[[f64; 6]; 6], SpiceBackendError> {
    let (inertial_name, body_name, body_is_to) =
        match (is_inertial(frame_from), is_inertial(frame_to)) {
            (true, false) => (frame_from, frame_to, true),
            (false, true) => (frame_to, frame_from, false),
            (true, true) | (false, false) => {
                return Err(SpiceBackendError::UnsupportedFrame(format!(
                    "{frame_from}->{frame_to}"
                )));
            }
        };

    let body_frame = body_frame_code(body_name)?;
    let (ref_frame, euler) =
        reader
            .euler_state_with_ref(body_frame, et)
            .map_err(|err| match err {
                PckError::NoCoverage { .. } => SpiceBackendError::NotCovered(format!("{err}")),
                other => SpiceBackendError::PckOpen {
                    path: "<loaded PCK>".to_string(),
                    source: other,
                },
            })?;
    let (r, dr) = pck_euler_rotation_and_derivative(
        euler[0], euler[1], euler[2], euler[3], euler[4], euler[5],
    );
    let ref_to_body = sxform_from_rotation(&r, &dr);
    let target_to_ref = static_inter_inertial(inertial_name, ref_frame)?;
    let target_to_ref_6x6 = sxform_from_rotation(&target_to_ref, &[[0.0; 3]; 3]);
    let target_to_body = matmul6(&ref_to_body, &target_to_ref_6x6);

    Ok(if body_is_to {
        target_to_body
    } else {
        invert_sxform(&target_to_body)
    })
}

fn body_frame_code(name: &str) -> Result<i32, SpiceBackendError> {
    match name {
        "ITRF93" => Ok(ITRF93_FRAME_CODE),
        _ => Err(SpiceBackendError::UnsupportedFrame(name.to_string())),
    }
}

fn is_inertial(name: &str) -> bool {
    matches!(name, "J2000" | "ECLIPJ2000")
}

fn static_inter_inertial(
    target_name: &str,
    ref_id: i32,
) -> Result<[[f64; 3]; 3], SpiceBackendError> {
    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    match (target_name, ref_id) {
        ("J2000", 1) | ("ECLIPJ2000", 17) => Ok(IDENTITY),
        ("J2000", 17) => Ok(j2000_to_eclipj2000()),
        ("ECLIPJ2000", 1) => {
            let s = j2000_to_eclipj2000();
            Ok([
                [s[0][0], s[1][0], s[2][0]],
                [s[0][1], s[1][1], s[2][1]],
                [s[0][2], s[1][2], s[2][2]],
            ])
        }
        _ => Err(SpiceBackendError::UnsupportedFrame(format!(
            "target={target_name}, ref_id={ref_id}"
        ))),
    }
}

fn matmul6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut acc = 0.0;
            for (k, b_row) in b.iter().enumerate() {
                acc += a[i][k] * b_row[j];
            }
            c[i][j] = acc;
        }
    }
    c
}

fn slice_to_mat6(values: &[f64]) -> [[f64; 6]; 6] {
    let mut m = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = values[i * 6 + j];
        }
    }
    m
}

fn normalize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_space = true;
    for ch in name.chars() {
        if ch.is_whitespace() || ch == '_' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            for up in ch.to_uppercase() {
                out.push(up);
            }
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

pub fn spice_frame_name(frame: Frame) -> Result<&'static str, SpiceBackendError> {
    match frame {
        Frame::Equatorial => Ok("J2000"),
        Frame::Ecliptic => Ok("ECLIPJ2000"),
        Frame::Itrf93 => Ok("ITRF93"),
        Frame::Unspecified => Err(SpiceBackendError::UnsupportedFrame(
            "unspecified".to_string(),
        )),
    }
}

pub fn et_seconds(times: &TimeArray) -> Result<Vec<f64>, SpiceBackendError> {
    Ok(times.rescale(TimeScale::Tdb)?.tdb_et_seconds()?)
}

fn earth_origin() -> OriginId {
    OriginId::Naif(EARTH_NAIF_ID)
}

/// Deduplicate a time array into its distinct epochs (first-appearance order)
/// plus a per-row inverse index mapping each original row to its slot in the
/// unique set. Lets SPICE-bound per-epoch work (spkez, pxform) run once per
/// distinct epoch and scatter to rows -- the Rust analogue of the legacy
/// `get_mpc_observer_state` `np.unique(..., return_inverse=True)` vectorization.
fn dedup_epochs(times: &TimeArray) -> Result<(TimeArray, Vec<usize>), SpiceBackendError> {
    let epochs = &times.epochs;
    let n = epochs.len();
    let key = |epoch: &Epoch| (epoch.days, epoch.nanos);
    let mut unique: Vec<Epoch> = Vec::new();
    let mut inverse: Vec<usize> = Vec::with_capacity(n);
    // Ephemeris epoch arrays are usually sorted, so a consecutive-run dedup
    // needs no hash map -- and for the common all-distinct case it is a cheap
    // linear pass rather than n hash insertions. Fall back to a hash map only
    // for unsorted input (non-adjacent duplicates).
    if epochs.windows(2).all(|w| key(&w[0]) <= key(&w[1])) {
        for epoch in epochs {
            if unique.last().map(&key) != Some(key(epoch)) {
                unique.push(*epoch);
            }
            inverse.push(unique.len() - 1);
        }
    } else {
        let mut slot_by_key: std::collections::HashMap<(i64, i64), usize> =
            std::collections::HashMap::with_capacity(n);
        for epoch in epochs {
            let slot = *slot_by_key.entry(key(epoch)).or_insert_with(|| {
                unique.push(*epoch);
                unique.len() - 1
            });
            inverse.push(slot);
        }
    }
    let unique_times = TimeArray::new(times.scale, unique)?;
    Ok((unique_times, inverse))
}

fn static_frame_rotation(frame_from: Frame, frame_to: Frame) -> Option<[[f64; 3]; 3]> {
    if frame_from == Frame::Unspecified || frame_to == Frame::Unspecified {
        return None;
    }
    if frame_from == frame_to {
        return Some(identity3());
    }
    match (frame_from, frame_to) {
        (Frame::Equatorial, Frame::Ecliptic) => Some(j2000_to_eclipj2000()),
        (Frame::Ecliptic, Frame::Equatorial) => Some(transpose3(&j2000_to_eclipj2000())),
        _ => None,
    }
}

fn unique_origin_indices(origins: &OriginArray) -> Vec<(OriginId, Vec<usize>)> {
    let mut unique: Vec<(OriginId, Vec<usize>)> = Vec::new();
    for (row, origin) in origins.origins.iter().cloned().enumerate() {
        if let Some((_existing, indices)) = unique
            .iter_mut()
            .find(|(existing, _indices)| *existing == origin)
        {
            indices.push(row);
        } else {
            unique.push((origin, vec![row]));
        }
    }
    unique
}

fn km_kms_to_au_day(flat: &[f64]) -> Vec<[f64; 6]> {
    flat.chunks_exact(6)
        .map(|chunk| {
            [
                chunk[0] / KM_PER_AU,
                chunk[1] / KM_PER_AU,
                chunk[2] / KM_PER_AU,
                chunk[3] / KM_PER_AU * SECONDS_PER_DAY,
                chunk[4] / KM_PER_AU * SECONDS_PER_DAY,
                chunk[5] / KM_PER_AU * SECONDS_PER_DAY,
            ]
        })
        .collect()
}

fn unflatten3(flat: &[f64]) -> Vec<[[f64; 3]; 3]> {
    flat.chunks_exact(9)
        .map(|chunk| {
            [
                [chunk[0], chunk[1], chunk[2]],
                [chunk[3], chunk[4], chunk[5]],
                [chunk[6], chunk[7], chunk[8]],
            ]
        })
        .collect()
}

fn unflatten6(flat: &[f64]) -> Vec<[[f64; 6]; 6]> {
    flat.chunks_exact(36)
        .map(|chunk| {
            let mut matrix = [[0.0; 6]; 6];
            for row in 0..6 {
                for col in 0..6 {
                    matrix[row][col] = chunk[row * 6 + col];
                }
            }
            matrix
        })
        .collect()
}

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn transpose3(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn sxform_from_static_rotation(rotation: [[f64; 3]; 3]) -> [[f64; 6]; 6] {
    let mut matrix = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            matrix[row][col] = rotation[row][col];
            matrix[row + 3][col + 3] = rotation[row][col];
        }
    }
    matrix
}

fn rotate3(rotation: &[[f64; 3]; 3], vector: &[f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
        rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
        rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
    ]
}

fn scale3(vector: &[f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn add_position_velocity(state: &mut [f64; 6], position: &[f64; 3], velocity: &[f64; 3]) {
    state[0] += position[0];
    state[1] += position[1];
    state[2] += position[2];
    state[3] += velocity[0];
    state[4] += velocity[1];
    state[5] += velocity[2];
}

fn add_state(state: &mut [f64; 6], delta: &[f64; 6]) {
    for i in 0..6 {
        state[i] += delta[i];
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use adam_core_rs_coords::types::{CoordinateValues, Epoch};
    use tempfile::{tempdir, NamedTempFile};

    use super::*;

    #[test]
    fn builtin_body_lookup_works() {
        let backend = AdamCoreSpiceBackend::new();
        assert_eq!(backend.bodn2c("EARTH").unwrap(), 399);
        assert_eq!(backend.bodn2c("EARTH_MOON_BARYCENTER").unwrap(), 3);
        assert_eq!(backend.bodc2n(399).unwrap(), "EARTH");
    }

    #[test]
    fn transform_coordinates_no_origin_change_matches_kernels() {
        use adam_core_rs_coords::types::{Frame as SchemaFrame, OriginArray, OriginId, TimeArray};
        use adam_core_rs_coords::{
            transform_values_flat6, transform_with_covariance_flat6, Frame as KernelFrame,
            Representation,
        };

        let backend = AdamCoreSpiceBackend::new();
        let coords = vec![
            1.0, 0.5, -0.3, 0.01, 0.005, -0.003, -2.1, 0.8, 0.02, -0.004, 0.012, 0.0008,
        ];
        let t0 = vec![60000.0, 60000.0];
        let mu = vec![2.95912208284120e-04_f64; 2];
        let times = TimeArray::new(
            TimeScale::Tdb,
            vec![Epoch::new(60000, 0), Epoch::new(60000, 0)],
        )
        .unwrap();
        let origins = OriginArray::repeat(OriginId::from_code("SUN"), 2);

        // No origin change (target None): values match transform_values_flat6.
        let out = backend
            .transform_coordinates(
                &coords,
                None,
                Representation::Cartesian,
                Representation::Spherical,
                SchemaFrame::Ecliptic,
                SchemaFrame::Equatorial,
                &origins,
                None,
                &times,
                &t0,
                &mu,
                0.0,
                0.0,
                100,
                1e-15,
            )
            .unwrap()
            .unwrap();
        let (expected, ncols) = transform_values_flat6(
            &coords,
            Representation::Cartesian,
            Representation::Spherical,
            KernelFrame::Ecliptic,
            KernelFrame::Equatorial,
            &t0,
            &mu,
            0.0,
            0.0,
            100,
            1e-15,
            None,
        )
        .unwrap();
        assert_eq!(out.ncols, ncols);
        assert_eq!(out.values, expected);
        assert!(out.covariance.is_none());

        // With covariance: matches transform_with_covariance_flat6.
        let cov = vec![1e-18_f64; 2 * 36];
        let out_cov = backend
            .transform_coordinates(
                &coords,
                Some(&cov),
                Representation::Cartesian,
                Representation::Cartesian,
                SchemaFrame::Ecliptic,
                SchemaFrame::Equatorial,
                &origins,
                None,
                &times,
                &t0,
                &mu,
                0.0,
                0.0,
                100,
                1e-15,
            )
            .unwrap()
            .unwrap();
        let (exp_vals, exp_cov) = transform_with_covariance_flat6(
            &coords,
            &cov,
            Representation::Cartesian,
            Representation::Cartesian,
            KernelFrame::Ecliptic,
            KernelFrame::Equatorial,
            &t0,
            &mu,
            0.0,
            0.0,
            100,
            1e-15,
            None,
        );
        assert_eq!(out_cov.values, exp_vals);
        assert_eq!(out_cov.covariance.unwrap(), exp_cov);

        // Time-varying ITRF93 is native, but an empty backend has no PCK
        // coverage for the requested sxform and must fail explicitly.
        let itrf_error = backend
            .transform_coordinates(
                &coords,
                None,
                Representation::Cartesian,
                Representation::Cartesian,
                SchemaFrame::Ecliptic,
                SchemaFrame::Itrf93,
                &origins,
                None,
                &times,
                &t0,
                &mu,
                0.0,
                0.0,
                100,
                1e-15,
            )
            .unwrap_err();
        assert!(matches!(itrf_error, SpiceBackendError::NotCovered(_)));
    }

    #[test]
    fn parse_mpc_obscodes_extracts_ground_sites_and_skips_space_based() {
        let json = r#"{
            "500": {"Longitude": 0.0, "cos": 0.0, "sin": 0.0, "Name": "Geocentric"},
            "X05": {"Longitude": 243.14012, "cos": 0.845303, "sin": 0.533213, "Name": "ZTF"},
            "C49": {"Longitude": null, "cos": null, "sin": null, "Name": "Space-based"}
        }"#;
        let sites = parse_mpc_obscodes(json).unwrap();
        assert_eq!(sites.len(), 2);
        assert!(!sites.contains_key("C49"));
        let x05 = sites.get("X05").expect("X05 site");
        assert!((x05.longitude_deg - 243.14012).abs() < 1.0e-9);
        assert!((x05.cos_phi - 0.845303).abs() < 1.0e-9);
        assert!((x05.sin_phi - 0.533213).abs() < 1.0e-9);
    }

    const SAMPLE_LSK: &str = concat!(
        "\\begindata\n",
        "DELTET/DELTA_T_A = 32.184\n",
        "DELTET/K = 1.657D-3\n",
        "DELTET/EB = 1.671D-2\n",
        "DELTET/M = ( 6.239996D0 1.99096871D-7 )\n",
        "DELTET/DELTA_AT = ( 10, @1972-JAN-1, 11, @1972-JUL-1 )\n",
        "\\begintext\n",
    );

    #[test]
    fn furnsh_retains_leapseconds_kernel_content() {
        let mut lsk_file = NamedTempFile::new().unwrap();
        write!(lsk_file, "{SAMPLE_LSK}").unwrap();
        lsk_file.flush().unwrap();

        let mut backend = AdamCoreSpiceBackend::new();
        backend.furnsh(lsk_file.path()).unwrap();

        let lsk = backend
            .leapseconds()
            .expect("LSK content should be retained, not classified as ignored");
        assert!((lsk.delta_t_a - 32.184).abs() < 1e-12);
        assert_eq!(lsk.delta_at.len(), 2);
        assert_eq!(lsk.delta_at[0].leap_seconds, 10);
        assert_eq!(lsk.delta_at[1].leap_seconds, 11);
    }

    #[test]
    fn furnsh_rejects_text_kernel_with_no_supported_content() {
        let mut empty = NamedTempFile::new().unwrap();
        write!(
            empty,
            "\\begintext\nThis text kernel has only commentary.\n\\begindata\n\\begintext\n"
        )
        .unwrap();
        empty.flush().unwrap();

        let mut backend = AdamCoreSpiceBackend::new();
        let err = backend.furnsh(empty.path()).expect_err(
            "empty text kernels must fail loudly so unsupported content cannot be silently dropped",
        );
        assert!(matches!(
            err,
            SpiceBackendError::UnsupportedTextKernel { .. }
        ));
    }

    #[test]
    fn text_kernel_bindings_are_last_loaded_wins() {
        let mut first = NamedTempFile::new().unwrap();
        write!(
            first,
            "\\begindata\nNAIF_BODY_NAME += ( 'OVERRIDE_ME' )\nNAIF_BODY_CODE += ( -1 )\n\\begintext\n"
        )
        .unwrap();
        first.flush().unwrap();

        let mut second = NamedTempFile::new().unwrap();
        write!(
            second,
            "\\begindata\nNAIF_BODY_NAME += ( 'OVERRIDE_ME' )\nNAIF_BODY_CODE += ( -2 )\n\\begintext\n"
        )
        .unwrap();
        second.flush().unwrap();

        let mut backend = AdamCoreSpiceBackend::new();
        backend.furnsh(first.path()).unwrap();
        assert_eq!(backend.bodn2c("OVERRIDE_ME").unwrap(), -1);
        backend.furnsh(second.path()).unwrap();
        assert_eq!(backend.bodn2c("OVERRIDE_ME").unwrap(), -2);
        backend.unload(second.path());
        assert_eq!(backend.bodn2c("OVERRIDE_ME").unwrap(), -1);
    }

    fn tdb_times() -> TimeArray {
        TimeArray::from_parts(
            TimeScale::Tdb,
            vec![51_544, 51_544],
            vec![43_200_000_000_000, 43_300_000_000_000],
        )
        .unwrap()
    }

    fn write_test_spk() -> (tempfile::TempDir, String) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bsp");
        let epochs = et_seconds(&tdb_times()).unwrap();
        let start_et = *epochs.first().unwrap();
        let end_et = *epochs.last().unwrap();
        let mut writer = SpkWriter::new_spk("adam-core-rs-spice-service-test");
        writer
            .add_type9(Type9Segment {
                target: 42,
                center: 0,
                frame_id: 1,
                start_et,
                end_et,
                segment_id: "TEST BODY".to_string(),
                degree: 1,
                states: vec![
                    KM_PER_AU,
                    2.0 * KM_PER_AU,
                    3.0 * KM_PER_AU,
                    KM_PER_AU / SECONDS_PER_DAY,
                    2.0 * KM_PER_AU / SECONDS_PER_DAY,
                    3.0 * KM_PER_AU / SECONDS_PER_DAY,
                    2.0 * KM_PER_AU,
                    4.0 * KM_PER_AU,
                    6.0 * KM_PER_AU,
                    2.0 * KM_PER_AU / SECONDS_PER_DAY,
                    4.0 * KM_PER_AU / SECONDS_PER_DAY,
                    6.0 * KM_PER_AU / SECONDS_PER_DAY,
                ],
                epochs,
            })
            .unwrap();
        writer.write(&path).unwrap();
        (dir, path.display().to_string())
    }

    #[test]
    fn typed_frame_service_handles_static_inertial_rotation() {
        let backend = AdamCoreSpiceBackend::new();
        let times = TimeArray::from_parts(TimeScale::Utc, vec![60_000], vec![0]).unwrap();

        let px = backend
            .frame_pxform_matrices(Frame::Equatorial, Frame::Ecliptic, &times)
            .unwrap();
        assert_eq!(px.len(), 1);
        assert_eq!(px[0], j2000_to_eclipj2000());

        let sx = backend
            .frame_sxform_matrices(Frame::Ecliptic, Frame::Ecliptic, &times)
            .unwrap();
        assert_eq!(sx[0][0][0], 1.0);
        assert_eq!(sx[0][3][3], 1.0);
    }

    #[test]
    fn state_batch_returns_coordinate_batch_in_adam_core_units() {
        let (_dir, path) = write_test_spk();
        let mut backend = AdamCoreSpiceBackend::new();
        backend.furnsh(path).unwrap();
        let times = tdb_times();

        let batch = backend
            .state_batch(
                &OriginId::Naif(42),
                &OriginId::SolarSystemBarycenter,
                Frame::Equatorial,
                &times,
            )
            .unwrap();

        assert_eq!(batch.frame, Frame::Equatorial);
        assert_eq!(
            batch.origins.origins,
            vec![OriginId::SolarSystemBarycenter; 2]
        );
        let values = match batch.values {
            CoordinateValues::Cartesian(values) => values,
            _ => panic!("state_batch must return Cartesian coordinates"),
        };
        assert_eq!(values[0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(values[1], [2.0, 4.0, 6.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn origin_translation_vectors_group_sources_and_preserve_row_order() {
        let (_dir, path) = write_test_spk();
        let mut backend = AdamCoreSpiceBackend::new();
        backend.furnsh(path).unwrap();
        let times = tdb_times();
        let origins = OriginArray::new(vec![OriginId::SolarSystemBarycenter, OriginId::Naif(42)]);

        let vectors = backend
            .origin_translation_vectors(
                &origins,
                &OriginId::SolarSystemBarycenter,
                Frame::Equatorial,
                &times,
            )
            .unwrap();

        assert_eq!(vectors[0], [0.0; 6]);
        assert_eq!(vectors[1], [2.0, 4.0, 6.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn ground_observer_itrf93_state_matches_adam_core_formula() {
        let backend = AdamCoreSpiceBackend::new();
        let site = GroundObserverSite::new("TEST", 90.0, 1.0, 0.0).unwrap();
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0); 2]).unwrap();

        let batch = backend
            .ground_observer_state(&site, &times, Frame::Itrf93, &earth_origin())
            .unwrap();
        let values = match batch.values {
            CoordinateValues::Cartesian(values) => values,
            _ => panic!("observer states must be Cartesian"),
        };

        assert!(values.iter().all(|row| *row == values[0]));
        assert!(values[0][0].abs() < 1.0e-18);
        assert!((values[0][1] - EARTH_EQUATORIAL_RADIUS_AU).abs() < 1.0e-18);
        assert!(
            (values[0][3] + EARTH_ROTATION_RAD_PER_DAY * EARTH_EQUATORIAL_RADIUS_AU).abs()
                < 1.0e-18
        );
        assert!(values[0][4].abs() < 1.0e-18);
    }
}
