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

use adam_core_rs_coords::types::{
    CoordinateBatch, Frame, OriginArray, OriginId, SchemaError, SchemaResult, TimeArray, TimeScale,
};
use adam_core_rs_coords::OriginTranslationProvider;
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

const ITRF93_FRAME_CODE: i32 = 3000;
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

impl AdamCoreSpiceBackend {
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            custom_names: HashMap::new(),
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
            let source_code = self.resolve_origin_id(&source)?;
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
        let mut states = if frame == Frame::Itrf93 {
            vec![topocentric; times.len()]
        } else {
            let mut states = self.state_vectors(&earth_origin(), origin, frame, times)?;
            let rotations = self.frame_pxform_matrices(Frame::Itrf93, frame, times)?;
            for (state, rotation) in states.iter_mut().zip(rotations.iter()) {
                let position = rotate3(rotation, &[topocentric[0], topocentric[1], topocentric[2]]);
                let velocity = rotate3(rotation, &[topocentric[3], topocentric[4], topocentric[5]]);
                add_position_velocity(state, &position, &velocity);
            }
            states
        };

        if frame == Frame::Itrf93 && self.resolve_origin_id(origin)? != EARTH_NAIF_ID {
            let earth_state = self.state_vectors(&earth_origin(), origin, frame, times)?;
            for (state, earth) in states.iter_mut().zip(earth_state.iter()) {
                add_state(state, earth);
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
            for reader in readers {
                let out_frame = if frame == "J2000" {
                    NaifFrame::J2000
                } else {
                    NaifFrame::EclipJ2000
                };
                let mut out = Vec::with_capacity(ets.len() * 6);
                let mut covered = true;
                for &et in ets {
                    let state = if frame == "J2000" {
                        reader.state(target, center, et)
                    } else {
                        reader.state_in_frame(target, center, et, out_frame)
                    };
                    match state {
                        Ok(s) => out.extend_from_slice(&s),
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
            let mut out = Vec::with_capacity(ets.len() * 36);
            let mut covered = true;
            for &et in ets {
                match pck_sxform_matrix(reader, frame_from, frame_to, et) {
                    Ok(m) => out.extend(m.iter().flat_map(|row| row.iter()).copied()),
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
