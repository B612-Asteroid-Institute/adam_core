//! Rust-native frame rotation on the canonical typed batches (W1 keystone,
//! bead personal-cmy.13). Demonstrates a real workflow executed entirely on
//! `OrbitBatch`/`CoordinateBatch` across a single Arrow-bridge crossing, using
//! the same rotation kernels as `transform_coordinates`.

use crate::types::{
    CoordinateBatch, CoordinateRepresentation, CovarianceBatch, CovarianceUnits, Frame, OrbitBatch,
    SchemaError, SchemaResult,
};
use crate::{rotate_cartesian_frame_flat6, transform_with_covariance_flat6, Representation};

pub(crate) fn chunk6(flat: &[f64]) -> Vec<[f64; 6]> {
    flat.chunks_exact(6)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5]])
        .collect()
}

/// Map the typed-batch `Frame` onto the rotation-kernel `Frame`, which has no
/// `Unspecified` variant (rotation needs a concrete source/target frame).
fn kernel_frame(frame: Frame) -> SchemaResult<crate::Frame> {
    match frame {
        Frame::Ecliptic => Ok(crate::Frame::Ecliptic),
        Frame::Equatorial => Ok(crate::Frame::Equatorial),
        Frame::Itrf93 => Ok(crate::Frame::Itrf93),
        Frame::Unspecified => Err(SchemaError::InvalidRecordBatch(
            "cannot rotate coordinates with an unspecified frame".to_string(),
        )),
    }
}

impl CoordinateBatch {
    /// Rotate Cartesian coordinates (and covariance, when present) into the
    /// `target` frame. No-op when already in `target`. Uses the same kernels as
    /// `transform_coordinates`, so results match the legacy NumPy path.
    pub fn rotate_frame(&self, target: Frame) -> SchemaResult<CoordinateBatch> {
        self.validate()?;
        if self.frame == target {
            return Ok(self.clone());
        }
        let values = self.values.cartesian().ok_or_else(|| {
            SchemaError::InvalidRecordBatch(
                "rotate_frame requires Cartesian coordinates".to_string(),
            )
        })?;
        let n = values.len();
        let flat: Vec<f64> = values.iter().flat_map(|row| row.iter().copied()).collect();
        let from = kernel_frame(self.frame)?;
        let to = kernel_frame(target)?;

        match self.covariance.as_ref() {
            None => {
                let rotated = rotate_cartesian_frame_flat6(&flat, from, to)
                    .map_err(|err| SchemaError::InvalidRecordBatch(err.to_string()))?;
                CoordinateBatch::cartesian(
                    chunk6(&rotated),
                    target,
                    self.origins.clone(),
                    self.times.clone(),
                    None,
                )
            }
            Some(covariance) => {
                if covariance.dimension != 6 {
                    return Err(SchemaError::InvalidCovarianceShape {
                        rows: covariance.rows,
                        dimension: covariance.dimension,
                        values: covariance.values_row_major.len(),
                    });
                }
                let zeros = vec![0.0_f64; n];
                let (coords_out, cov_out) = transform_with_covariance_flat6(
                    &flat,
                    &covariance.values_row_major,
                    Representation::Cartesian,
                    Representation::Cartesian,
                    from,
                    to,
                    &zeros,
                    &zeros,
                    0.0,
                    0.0,
                    100,
                    1e-15,
                    None,
                );
                let new_covariance = CovarianceBatch::new(
                    n,
                    6,
                    cov_out,
                    CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
                )?;
                let new_covariance = match covariance.row_validity.clone() {
                    Some(validity) => new_covariance.with_row_validity(validity)?,
                    None => new_covariance,
                };
                CoordinateBatch::cartesian(
                    chunk6(&coords_out),
                    target,
                    self.origins.clone(),
                    self.times.clone(),
                    Some(new_covariance),
                )
            }
        }
    }
}

impl OrbitBatch {
    /// Rotate the orbit coordinates into the `target` frame, preserving orbit
    /// ids and physical parameters.
    pub fn rotate_frame(&self, target: Frame) -> SchemaResult<OrbitBatch> {
        let coordinates = self.coordinates.rotate_frame(target)?;
        let rotated = OrbitBatch::new(self.orbit_id.clone(), self.object_id.clone(), coordinates)?;
        match self.physical_parameters.clone() {
            Some(physical_parameters) => rotated.with_physical_parameters(physical_parameters),
            None => Ok(rotated),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OriginArray, OriginId, TimeArray, TimeScale};

    #[test]
    fn rotate_frame_is_noop_when_target_matches() {
        let coords = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            Some(TimeArray::from_parts(TimeScale::Tdb, vec![60000], vec![0]).unwrap()),
            None,
        )
        .unwrap();
        let rotated = coords.rotate_frame(Frame::Ecliptic).unwrap();
        assert_eq!(rotated, coords);
    }

    #[test]
    fn rotate_frame_round_trips_ecliptic_equatorial() {
        let coords = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let equatorial = coords.rotate_frame(Frame::Equatorial).unwrap();
        assert_eq!(equatorial.frame, Frame::Equatorial);
        let back = equatorial.rotate_frame(Frame::Ecliptic).unwrap();
        let original = coords.values.cartesian().unwrap()[0];
        let restored = back.values.cartesian().unwrap()[0];
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}
