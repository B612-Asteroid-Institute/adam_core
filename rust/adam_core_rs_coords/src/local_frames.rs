//! Local orbital frames (SANA orbit-relative reference frame registry names)
//! for covariances: RSW (aliases RTN, RIC), TNW and VNC, each `_INERTIAL`
//! (pure rotation) or `_ROTATING` (velocity rows carry the two-body frame
//! rate). Jacobians come from forward-mode autodiff of the frame axes along
//! the two-body motion; the covariance product `J C J^T` is accumulated in
//! double-double arithmetic and rounded once.
//!
//! Interface fixed on 2026-10-08. Bodies are filled by the kernel task; the
//! OEM renderer calls these functions for its COV_REF_FRAME blocks.

use crate::types::{SchemaError, SchemaResult};

/// The three axis families of the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalFrameFamily {
    /// x along position, z along the orbital angular momentum h.
    Rsw,
    /// x along velocity, z along h.
    Tnw,
    /// x along velocity, y along h.
    Vnc,
}

/// A local orbital frame: family plus whether velocity rows carry the frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalFrame {
    pub family: LocalFrameFamily,
    pub rotating: bool,
}

impl LocalFrame {
    /// Parse a registry name (`VNC_ROTATING`, `TNW_INERTIAL`, ...), an alias
    /// (`RTN`, `RIC` mean RSW) or a bare family name (meaning `_INERTIAL`),
    /// case-insensitively and ignoring surrounding whitespace.
    pub fn parse(name: &str) -> SchemaResult<Self> {
        let _ = name;
        Err(SchemaError::InvalidRecordBatch(
            "LocalFrame::parse is not implemented yet".to_string(),
        ))
    }

    /// The canonical registry name, e.g. `VNC_ROTATING`.
    pub fn canonical_name(&self) -> &'static str {
        match (self.family, self.rotating) {
            (LocalFrameFamily::Rsw, true) => "RSW_ROTATING",
            (LocalFrameFamily::Rsw, false) => "RSW_INERTIAL",
            (LocalFrameFamily::Tnw, true) => "TNW_ROTATING",
            (LocalFrameFamily::Tnw, false) => "TNW_INERTIAL",
            (LocalFrameFamily::Vnc, true) => "VNC_ROTATING",
            (LocalFrameFamily::Vnc, false) => "VNC_INERTIAL",
        }
    }
}

/// Jacobians from inertial position and velocity to `frame`, one 6x6 per
/// state, returned row-major as `N * 36` values. `values` is `N * 6`
/// (x, y, z, vx, vy, vz in AU and AU/day), `mu` has one entry per state
/// (AU^3/day^2) and is only read for `_ROTATING` frames.
pub fn local_frame_jacobians(
    values: &[f64],
    mu: &[f64],
    frame: LocalFrame,
) -> SchemaResult<Vec<f64>> {
    let _ = (values, mu, frame);
    Err(SchemaError::InvalidRecordBatch(
        "local_frame_jacobians is not implemented yet".to_string(),
    ))
}

/// `J C J^T` for every state, `covariances` and the result row-major `N * 36`
/// (AU, AU/day units). The input covariance is symmetrised exactly, the
/// product is accumulated in double-double arithmetic and rounded once, and
/// the result is symmetric to the bit. Rows whose covariance is all NaN stay NaN.
pub fn local_frame_covariances(
    values: &[f64],
    covariances: &[f64],
    mu: &[f64],
    frame: LocalFrame,
) -> SchemaResult<Vec<f64>> {
    let _ = (values, covariances, mu, frame);
    Err(SchemaError::InvalidRecordBatch(
        "local_frame_covariances is not implemented yet".to_string(),
    ))
}
