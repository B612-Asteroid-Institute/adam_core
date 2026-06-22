//! Backend-generic origin-translation provider for the typed pipeline.
//!
//! This mirrors the `TimeScaleProvider` pattern: `adam_core_rs_coords` stays
//! backend-generic, and a SPICE-backed implementation (in `adam_core_rs_spice`)
//! supplies the per-row body-state translation vectors. The typed
//! propagation/ephemeris pipeline can then translate orbit/observer coordinates
//! to a common origin natively, which is the Rust-native replacement for the
//! Python `transform_coordinates` / `get_perturber_state` glue.
//!
//! Frame rotation already lives in Rust (`rotate_*_row` kernels); only the
//! origin shift needs ephemeris body states, so that is what this trait owns.

use crate::types::{
    CoordinateBatch, Frame, OriginArray, OriginId, SchemaError, SchemaResult, TimeArray,
};

/// Supplies per-row origin-translation state vectors `[x, y, z, vx, vy, vz]`
/// (AU, AU/day) in `frame`, taking coordinates from each row's current origin
/// to `target_origin`. The returned vector is ADDED to a state expressed
/// relative to the current origin to re-express it relative to `target_origin`.
///
/// This matches the convention of
/// `adam_core_rs_spice::AdamCoreSpiceBackend::origin_translation_vectors` and of
/// the Python `transform_coordinates` origin-shift path.
pub trait OriginTranslationProvider {
    fn origin_translation_vectors(
        &self,
        origins: &OriginArray,
        target_origin: &OriginId,
        frame: Frame,
        times: &TimeArray,
    ) -> SchemaResult<Vec<[f64; 6]>>;
}

/// Translate a Cartesian `CoordinateBatch` from its per-row origins to a single
/// `target_origin`, using `provider` for the body-state translation vectors.
///
/// The origin shift is covariance-invariant (it adds a deterministic state), so
/// the input covariance is preserved unchanged. Frame and times are preserved;
/// the output origins are all `target_origin`.
pub fn translate_coordinates_to_origin<P>(
    coordinates: &CoordinateBatch,
    target_origin: &OriginId,
    provider: &P,
) -> SchemaResult<CoordinateBatch>
where
    P: OriginTranslationProvider + ?Sized,
{
    let states = coordinates.values.cartesian().ok_or_else(|| {
        SchemaError::InvalidRecordBatch(
            "origin translation requires Cartesian coordinates".to_string(),
        )
    })?;
    let times = coordinates
        .times
        .as_ref()
        .ok_or_else(|| SchemaError::MissingRequiredField("coordinates.time".to_string()))?;

    let vectors = provider.origin_translation_vectors(
        &coordinates.origins,
        target_origin,
        coordinates.frame,
        times,
    )?;
    if vectors.len() != states.len() {
        return Err(SchemaError::LengthMismatch {
            field: "origin_translation.vectors".to_string(),
            expected: states.len(),
            actual: vectors.len(),
        });
    }

    let translated = states
        .iter()
        .zip(vectors.iter())
        .map(|(state, vector)| {
            let mut out = *state;
            for index in 0..6 {
                out[index] += vector[index];
            }
            out
        })
        .collect::<Vec<_>>();

    CoordinateBatch::cartesian(
        translated,
        coordinates.frame,
        OriginArray::repeat(target_origin.clone(), states.len()),
        Some(times.clone()),
        coordinates.covariance.clone(),
    )
}

/// Rotate a Cartesian `CoordinateBatch` between the ecliptic and equatorial
/// frames using the canonical adam-core obliquity kernels.
///
/// Frame rotation is not covariance-invariant, and this helper does not rotate
/// covariance, so it requires `covariance == None` for an actual frame change
/// (a no-op same-frame call preserves covariance). Only ecliptic<->equatorial
/// is supported here; time-varying (ITRF93) rotation is out of scope.
pub fn rotate_coordinates_to_frame(
    coordinates: &CoordinateBatch,
    target_frame: Frame,
) -> SchemaResult<CoordinateBatch> {
    if coordinates.frame == target_frame {
        return Ok(coordinates.clone());
    }
    if coordinates.covariance.is_some() {
        return Err(SchemaError::InvalidRecordBatch(
            "frame rotation of coordinates with covariance is not supported here".to_string(),
        ));
    }
    let states = coordinates.values.cartesian().ok_or_else(|| {
        SchemaError::InvalidRecordBatch("frame rotation requires Cartesian coordinates".to_string())
    })?;
    let rotated: Vec<[f64; 6]> = match (coordinates.frame, target_frame) {
        (Frame::Equatorial, Frame::Ecliptic) => states
            .iter()
            .map(crate::rotate_equatorial_to_ecliptic_row)
            .collect(),
        (Frame::Ecliptic, Frame::Equatorial) => states
            .iter()
            .map(crate::rotate_ecliptic_to_equatorial_row)
            .collect(),
        _ => {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "unsupported frame rotation from {} to {}",
                coordinates.frame.as_str(),
                target_frame.as_str()
            )))
        }
    };
    CoordinateBatch::cartesian(
        rotated,
        target_frame,
        coordinates.origins.clone(),
        coordinates.times.clone(),
        None,
    )
}

/// Normalize a Cartesian `CoordinateBatch` to a single `target_origin` and
/// `target_frame`: rotate to the target frame first (so the origin-translation
/// vectors are evaluated in that frame), then shift the origin via `provider`.
pub fn normalize_coordinates_to<P>(
    coordinates: &CoordinateBatch,
    target_origin: &OriginId,
    target_frame: Frame,
    provider: &P,
) -> SchemaResult<CoordinateBatch>
where
    P: OriginTranslationProvider + ?Sized,
{
    let rotated = rotate_coordinates_to_frame(coordinates, target_frame)?;
    translate_coordinates_to_origin(&rotated, target_origin, provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CoordinateRepresentation, CovarianceBatch, CovarianceUnits, Epoch, TimeScale,
    };

    struct StubProvider {
        vectors: Vec<[f64; 6]>,
    }

    impl OriginTranslationProvider for StubProvider {
        fn origin_translation_vectors(
            &self,
            origins: &OriginArray,
            _target_origin: &OriginId,
            _frame: Frame,
            times: &TimeArray,
        ) -> SchemaResult<Vec<[f64; 6]>> {
            assert_eq!(origins.len(), times.len());
            Ok(self.vectors.clone())
        }
    }

    fn close(a: [f64; 6], b: [f64; 6]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1.0e-12)
    }

    fn sample_batch() -> (CoordinateBatch, Vec<f64>) {
        let states = vec![
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
        ];
        let cov_values: Vec<f64> = (0..72).map(|v| v as f64 * 1.0e-12).collect();
        let covariance = CovarianceBatch::new(
            2,
            6,
            cov_values.clone(),
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let times = TimeArray::new(
            TimeScale::Tdb,
            vec![Epoch::new(60_000, 0), Epoch::new(60_001, 0)],
        )
        .unwrap();
        let coords = CoordinateBatch::cartesian(
            states,
            Frame::Equatorial,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            Some(times),
            Some(covariance),
        )
        .unwrap();
        (coords, cov_values)
    }

    #[test]
    fn translate_to_origin_shifts_states_and_preserves_covariance() {
        let (coords, cov_values) = sample_batch();
        let provider = StubProvider {
            vectors: vec![
                [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
                [100.0, 200.0, 300.0, 4.0, 5.0, 6.0],
            ],
        };
        let out = translate_coordinates_to_origin(&coords, &OriginId::Naif(10), &provider).unwrap();
        let out_states = out.values.cartesian().unwrap();
        assert!(close(out_states[0], [11.0, 22.0, 33.0, 1.1, 2.2, 3.3]));
        assert!(close(out_states[1], [104.0, 205.0, 306.0, 4.4, 5.5, 6.6]));
        assert!(out
            .origins
            .origins
            .iter()
            .all(|origin| matches!(origin, OriginId::Naif(10))));
        assert_eq!(out.frame, Frame::Equatorial);
        assert_eq!(
            out.covariance.as_ref().unwrap().values_row_major,
            cov_values
        );
    }

    #[test]
    fn rotate_to_frame_roundtrips_through_obliquity() {
        let states = vec![[1.0, 0.5, -0.3, 0.01, 0.005, -0.003]];
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap();
        let ecliptic = CoordinateBatch::cartesian(
            states.clone(),
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(times),
            None,
        )
        .unwrap();
        let equatorial = rotate_coordinates_to_frame(&ecliptic, Frame::Equatorial).unwrap();
        assert_eq!(equatorial.frame, Frame::Equatorial);
        let back = rotate_coordinates_to_frame(&equatorial, Frame::Ecliptic).unwrap();
        assert!(close(
            back.values.cartesian().unwrap()[0],
            ecliptic.values.cartesian().unwrap()[0]
        ));
    }

    #[test]
    fn rotate_to_frame_rejects_covariance_change() {
        let (coords, _) = sample_batch();
        // coords is equatorial with covariance; rotating to ecliptic must error.
        let err = rotate_coordinates_to_frame(&coords, Frame::Ecliptic).unwrap_err();
        assert!(matches!(err, SchemaError::InvalidRecordBatch(_)));
    }

    #[test]
    fn normalize_rotates_then_translates_origin() {
        let states = vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]];
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap();
        let ecliptic = CoordinateBatch::cartesian(
            states,
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            Some(times),
            None,
        )
        .unwrap();
        let provider = StubProvider {
            vectors: vec![[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]],
        };
        let out = normalize_coordinates_to(
            &ecliptic,
            &OriginId::Named("SUN".to_string()),
            Frame::Ecliptic,
            &provider,
        )
        .unwrap();
        assert_eq!(out.frame, Frame::Ecliptic);
        assert!(matches!(
            out.origins.origins[0],
            OriginId::Named(ref code) if code == "SUN"
        ));
        assert!(close(
            out.values.cartesian().unwrap()[0],
            [11.0, 22.0, 33.0, 1.1, 2.2, 3.3]
        ));
    }

    #[test]
    fn translate_to_origin_requires_times() {
        let coords = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let provider = StubProvider {
            vectors: vec![[0.0; 6]],
        };
        let err =
            translate_coordinates_to_origin(&coords, &OriginId::Naif(10), &provider).unwrap_err();
        assert!(matches!(err, SchemaError::MissingRequiredField(_)));
    }
}
