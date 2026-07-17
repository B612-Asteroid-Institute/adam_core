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
    CoordinateBatch, CovarianceBatch, Epoch, Frame, OriginArray, OriginId, SchemaError,
    SchemaResult, TimeArray,
};
use std::collections::HashMap;

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

/// Resolve origin translations once per unique `(origin, epoch)` pair and
/// expand them back to caller row order.
pub fn deduplicated_origin_translation_vectors<P>(
    provider: &P,
    origins: &OriginArray,
    target_origin: &OriginId,
    frame: Frame,
    times: &TimeArray,
) -> SchemaResult<Vec<[f64; 6]>>
where
    P: OriginTranslationProvider + ?Sized,
{
    if origins.len() != times.len() {
        return Err(SchemaError::LengthMismatch {
            field: "origin_translation.times".to_string(),
            expected: origins.len(),
            actual: times.len(),
        });
    }
    let mut unique_indices = HashMap::<(OriginId, Epoch), usize>::new();
    let mut unique_origins = Vec::new();
    let mut unique_epochs = Vec::new();
    let mut row_to_unique = Vec::with_capacity(times.len());
    for (origin, epoch) in origins.origins.iter().zip(times.epochs.iter()) {
        let key = (origin.clone(), *epoch);
        let index = match unique_indices.get(&key) {
            Some(index) => *index,
            None => {
                let index = unique_origins.len();
                unique_indices.insert(key, index);
                unique_origins.push(origin.clone());
                unique_epochs.push(*epoch);
                index
            }
        };
        row_to_unique.push(index);
    }
    let unique_times = TimeArray::new(times.scale, unique_epochs)?;
    let unique_vectors = provider.origin_translation_vectors(
        &OriginArray::new(unique_origins),
        target_origin,
        frame,
        &unique_times,
    )?;
    if unique_vectors.len() != unique_indices.len() {
        return Err(SchemaError::LengthMismatch {
            field: "origin_translation.unique_vectors".to_string(),
            expected: unique_indices.len(),
            actual: unique_vectors.len(),
        });
    }
    Ok(row_to_unique
        .into_iter()
        .map(|index| unique_vectors[index])
        .collect())
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

    let vectors = deduplicated_origin_translation_vectors(
        provider,
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
/// Covariance is rotated by the same constant 6x6 block rotation as the state.
/// Only ecliptic<->equatorial is supported here; time-varying (ITRF93) rotation
/// is out of scope.
pub fn rotate_coordinates_to_frame(
    coordinates: &CoordinateBatch,
    target_frame: Frame,
) -> SchemaResult<CoordinateBatch> {
    if coordinates.frame == target_frame {
        return Ok(coordinates.clone());
    }
    let states = coordinates.values.cartesian().ok_or_else(|| {
        SchemaError::InvalidRecordBatch("frame rotation requires Cartesian coordinates".to_string())
    })?;
    let rotation: fn(&[f64; 6]) -> [f64; 6] = match (coordinates.frame, target_frame) {
        (Frame::Equatorial, Frame::Ecliptic) => crate::rotate_equatorial_to_ecliptic_row,
        (Frame::Ecliptic, Frame::Equatorial) => crate::rotate_ecliptic_to_equatorial_row,
        _ => {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "unsupported frame rotation from {} to {}",
                coordinates.frame.as_str(),
                target_frame.as_str()
            )))
        }
    };
    let rotated = states.iter().map(rotation).collect::<Vec<_>>();
    let covariance = coordinates
        .covariance
        .as_ref()
        .map(|covariance| rotate_covariance(covariance, rotation))
        .transpose()?;
    CoordinateBatch::cartesian(
        rotated,
        target_frame,
        coordinates.origins.clone(),
        coordinates.times.clone(),
        covariance,
    )
}

fn rotate_covariance(
    covariance: &CovarianceBatch,
    rotation: fn(&[f64; 6]) -> [f64; 6],
) -> SchemaResult<CovarianceBatch> {
    let mut jacobian = [[0.0_f64; 6]; 6];
    for column in 0..6 {
        let mut basis = [0.0_f64; 6];
        basis[column] = 1.0;
        let rotated = rotation(&basis);
        for row in 0..6 {
            jacobian[row][column] = rotated[row];
        }
    }
    let mut values = Vec::with_capacity(covariance.rows * 36);
    for row in 0..covariance.rows {
        let input = covariance.row_values(row);
        let mut intermediate = [[0.0_f64; 6]; 6];
        for (output_row, intermediate_row) in intermediate.iter_mut().enumerate() {
            for (column, value) in intermediate_row.iter_mut().enumerate() {
                for inner in 0..6 {
                    *value += jacobian[output_row][inner] * input[inner * 6 + column];
                }
            }
        }
        for intermediate_row in &intermediate {
            for jacobian_row in &jacobian {
                let value = intermediate_row
                    .iter()
                    .zip(jacobian_row.iter())
                    .map(|(left, right)| left * right)
                    .sum();
                values.push(value);
            }
        }
    }
    let rotated = CovarianceBatch::new(
        covariance.rows,
        covariance.dimension,
        values,
        covariance.units.clone(),
    )?;
    Ok(match &covariance.row_validity {
        Some(validity) => rotated.with_row_validity(validity.clone())?,
        None => rotated,
    })
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
    fn rotate_to_frame_roundtrips_covariance() {
        let (coords, _) = sample_batch();
        let ecliptic = rotate_coordinates_to_frame(&coords, Frame::Ecliptic).unwrap();
        let back = rotate_coordinates_to_frame(&ecliptic, Frame::Equatorial).unwrap();
        let expected = &coords.covariance.as_ref().unwrap().values_row_major;
        let actual = &back.covariance.as_ref().unwrap().values_row_major;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-24);
        }
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
