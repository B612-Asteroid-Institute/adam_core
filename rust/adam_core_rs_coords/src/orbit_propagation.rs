//! Rust-native 2-body propagation over the canonical typed batches (W1 keystone,
//! bead personal-cmy.13). Demonstrates a propagation workflow executed entirely
//! on `OrbitBatch` across a single Arrow-bridge crossing, using the same Kepler
//! kernel (`propagate_2body_flat6`) as the public `propagate_2body` path.

use crate::coordinate_frame::chunk6;
use crate::types::{
    origin_mu_au3_day2, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, Epoch, OrbitBatch, SchemaError, SchemaResult, TimeArray, TimeScale,
    NANOS_PER_DAY,
};
use crate::{propagate_2body_flat6, propagate_2body_with_covariance_flat6};

impl OrbitBatch {
    /// Propagate every orbit from its own epoch to the shared `target` epoch with
    /// 2-body Keplerian dynamics. When covariance is present it is transported via
    /// the state-transition matrix (`Sigma_out = J Sigma_in J^T`), matching
    /// `propagate_2body`. Orbit epochs must already be in `scale`, the dynamics
    /// time scale (typically TDB).
    pub fn propagate_2body_to(
        &self,
        target: Epoch,
        scale: TimeScale,
        max_iter: usize,
        tol: f64,
    ) -> SchemaResult<OrbitBatch> {
        let times = self.coordinates.times.as_ref().ok_or_else(|| {
            SchemaError::InvalidRecordBatch("propagation requires orbit epochs".to_string())
        })?;
        if times.scale != scale {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "orbit epoch scale {:?} must match the target scale {:?}",
                times.scale, scale
            )));
        }
        let values = self.coordinates.values.cartesian().ok_or_else(|| {
            SchemaError::InvalidRecordBatch(
                "propagation requires Cartesian coordinates".to_string(),
            )
        })?;
        let n = values.len();
        let flat: Vec<f64> = values.iter().flat_map(|row| row.iter().copied()).collect();
        let dts: Vec<f64> = times
            .epochs
            .iter()
            .map(|epoch| {
                (target.days - epoch.days) as f64
                    + (target.nanos - epoch.nanos) as f64 / NANOS_PER_DAY as f64
            })
            .collect();
        let mus = self
            .coordinates
            .origins
            .origins
            .iter()
            .map(origin_mu_au3_day2)
            .collect::<SchemaResult<Vec<f64>>>()?;
        let target_times = TimeArray::new(scale, vec![target; n])?;
        let coordinates = match self.coordinates.covariance.as_ref() {
            None => {
                let states = propagate_2body_flat6(&flat, &dts, &mus, max_iter, tol);
                CoordinateBatch::cartesian(
                    chunk6(&states),
                    self.coordinates.frame,
                    self.coordinates.origins.clone(),
                    Some(target_times),
                    None,
                )?
            }
            Some(covariance) => {
                if covariance.dimension != 6 {
                    return Err(SchemaError::InvalidCovarianceShape {
                        rows: covariance.rows,
                        dimension: covariance.dimension,
                        values: covariance.values_row_major.len(),
                    });
                }
                let (states, covariances) = propagate_2body_with_covariance_flat6(
                    &flat,
                    &covariance.values_row_major,
                    &dts,
                    &mus,
                    max_iter,
                    tol,
                );
                let propagated_covariance = CovarianceBatch::new(
                    n,
                    6,
                    covariances,
                    CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
                )?;
                let propagated_covariance = match covariance.row_validity.clone() {
                    Some(validity) => propagated_covariance.with_row_validity(validity)?,
                    None => propagated_covariance,
                };
                CoordinateBatch::cartesian(
                    chunk6(&states),
                    self.coordinates.frame,
                    self.coordinates.origins.clone(),
                    Some(target_times),
                    Some(propagated_covariance),
                )?
            }
        };
        let propagated_orbits =
            OrbitBatch::new(self.orbit_id.clone(), self.object_id.clone(), coordinates)?;
        match self.physical_parameters.clone() {
            Some(physical_parameters) => {
                propagated_orbits.with_physical_parameters(physical_parameters)
            }
            None => Ok(propagated_orbits),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Frame, OrbitId, OriginArray, OriginId};

    fn sample() -> OrbitBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 0.0, 0.0, 0.0, 0.017_202, 0.0]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::from_parts(TimeScale::Tdb, vec![60000], vec![0]).unwrap()),
            None,
        )
        .unwrap();
        OrbitBatch::new(vec![OrbitId("o".to_string())], vec![None], coordinates).unwrap()
    }

    #[test]
    fn propagate_to_same_epoch_is_identity() {
        let orbits = sample();
        let out = orbits
            .propagate_2body_to(Epoch::new(60000, 0), TimeScale::Tdb, 100, 1e-14)
            .unwrap();
        let original = orbits.coordinates.values.cartesian().unwrap()[0];
        let propagated = out.coordinates.values.cartesian().unwrap()[0];
        for (a, b) in original.iter().zip(propagated.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn propagate_rejects_scale_mismatch() {
        let orbits = sample();
        let err = orbits
            .propagate_2body_to(Epoch::new(60010, 0), TimeScale::Utc, 100, 1e-14)
            .unwrap_err();
        assert!(matches!(err, SchemaError::InvalidRecordBatch(_)));
    }
}
