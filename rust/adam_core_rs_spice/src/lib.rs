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

use spicekit::frame::{
    apply_sxform, invert_sxform, j2000_to_eclipj2000, pck_euler_rotation_and_derivative,
    sxform_from_rotation,
};
use spicekit::{
    bodc2n as spicekit_bodc2n, bodn2c as spicekit_bodn2c, parse_body_bindings, BodyBinding,
    NaifIdError, TextKernelError,
};
use thiserror::Error;

pub use spicekit::spk_writer::{
    SpkWriter, SpkWriterError, Type3Record, Type3Segment, Type9Segment,
};
pub use spicekit::{NaifFrame, PckError, PckFile, SpkError, SpkFile};

const ITRF93_FRAME_CODE: i32 = 3000;

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
    #[error("{0}")]
    NotCovered(String),
    #[error("unsupported NAIF frame: {0}")]
    UnsupportedFrame(String),
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
        bindings: Vec<BodyBinding>,
    },
    Ignored {
        path: String,
    },
}

impl Kernel {
    fn path(&self) -> &str {
        match self {
            Kernel::Spk { path, .. }
            | Kernel::Pck { path, .. }
            | Kernel::Text { path, .. }
            | Kernel::Ignored { path } => path,
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
                let bindings =
                    parse_body_bindings(path).map_err(|source| SpiceBackendError::TextKernel {
                        path: path_s.clone(),
                        source,
                    })?;
                if bindings.is_empty() {
                    self.kernels.push(Kernel::Ignored { path: path_s });
                } else {
                    self.kernels.push(Kernel::Text {
                        path: path_s,
                        bindings,
                    });
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

    fn rebuild_name_index(&mut self) {
        self.custom_names.clear();
        for k in &self.kernels {
            if let Kernel::Text { bindings, .. } = k {
                for binding in bindings {
                    self.custom_names
                        .insert(normalize_name(&binding.name), binding.code);
                }
            }
        }
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
    parse_body_bindings(path_ref).map_err(|source| SpiceBackendError::TextKernel {
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn builtin_body_lookup_works() {
        let backend = AdamCoreSpiceBackend::new();
        assert_eq!(backend.bodn2c("EARTH").unwrap(), 399);
        assert_eq!(backend.bodn2c("EARTH_MOON_BARYCENTER").unwrap(), 3);
        assert_eq!(backend.bodc2n(399).unwrap(), "EARTH");
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
}
