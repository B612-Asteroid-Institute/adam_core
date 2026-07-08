//! Rust-native covariance variant sampling and collapse helpers.
//!
//! These functions mirror the public Python `VariantOrbits.create` / `collapse`
//! semantics at the typed batch layer so propagator adapters can keep covariance
//! expansion, propagation, and collapse inside one Rust boundary.

use crate::propagation::{
    EphemerisResult, PropagationError, PropagationResult, PropagationResultValue,
};
use crate::types::SchemaResult;
use crate::{
    CoordinateBatch, CoordinateRepresentation, CoordinateValues, CovarianceBatch, CovarianceUnits,
    EphemerisBatch, OrbitBatch, OrbitVariantBatch, OriginArray, SchemaError, TimeArray, Validity,
    VariantId,
};
use std::collections::HashMap;

const DIM: usize = 6;
const SIGMA_POINT_COUNT: usize = 2 * DIM + 1;
const JACOBI_MAX_SWEEPS: usize = 100;
const JACOBI_TOLERANCE: f64 = 1.0e-18;
const PSD_TOLERANCE: f64 = 1.0e-12;
const MONTE_CARLO_PSD_TOLERANCE: f64 = 1.0e-15;
const AUTO_RECONSTRUCTION_TOLERANCE: f64 = 1.0e-12;
const DEFAULT_RANDOM_SEED: u64 = 0x4d59_5df4_d0f3_3173;
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;

#[derive(Debug, Clone, PartialEq)]
pub struct OrbitVariantSamples {
    pub variants: OrbitVariantBatch,
    /// `source_orbit_indices[variant_row] = original orbit row index`.
    pub source_orbit_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitVariantSamplingMethod {
    Auto,
    SigmaPoint,
    MonteCarlo,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CoordinateSample {
    values: [f64; DIM],
    weight: f64,
    weight_cov: f64,
}

/// Create orbit variants from per-row coordinate covariance using public
/// `VariantOrbits.create` sampling semantics.
pub fn create_sampled_orbit_variants(
    orbits: &OrbitBatch,
    method: OrbitVariantSamplingMethod,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<OrbitVariantSamples> {
    match method {
        OrbitVariantSamplingMethod::Auto => {
            validate_sigma_point_parameters(alpha, beta, kappa)?;
            validate_monte_carlo_parameters(num_samples)?;
            create_orbit_variants_from_sampler(
                orbits,
                num_samples.max(SIGMA_POINT_COUNT),
                |row_index, mean, covariance| {
                    auto_samples(
                        row_index,
                        mean,
                        covariance,
                        num_samples,
                        seed,
                        alpha,
                        beta,
                        kappa,
                    )
                },
            )
        }
        OrbitVariantSamplingMethod::SigmaPoint => {
            create_sigma_point_orbit_variants(orbits, alpha, beta, kappa)
        }
        OrbitVariantSamplingMethod::MonteCarlo => {
            create_monte_carlo_orbit_variants(orbits, num_samples, seed)
        }
    }
}

/// Create sigma-point orbit variants from per-row coordinate covariance.
///
/// Matches the Python `sample_covariance_sigma_points` ordering and weights:
/// mean row first, then `mean + sqrt((D+lambda)Σ)[i]`, then `mean - ...[i]`.
pub fn create_sigma_point_orbit_variants(
    orbits: &OrbitBatch,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<OrbitVariantSamples> {
    validate_sigma_point_parameters(alpha, beta, kappa)?;
    create_orbit_variants_from_sampler(orbits, SIGMA_POINT_COUNT, |_row_index, mean, covariance| {
        sigma_point_samples(mean, covariance, alpha, beta, kappa)
    })
}

/// Create Monte Carlo orbit variants from per-row coordinate covariance.
///
/// The same explicit seed is re-used for each coordinate row, matching the
/// Python loop which calls `sample_covariance_random(..., seed=seed)` once per
/// orbit row.
pub fn create_monte_carlo_orbit_variants(
    orbits: &OrbitBatch,
    num_samples: usize,
    seed: Option<u64>,
) -> SchemaResult<OrbitVariantSamples> {
    validate_monte_carlo_parameters(num_samples)?;
    create_orbit_variants_from_sampler(orbits, num_samples, |row_index, mean, covariance| {
        monte_carlo_samples(mean, covariance, num_samples, seed_for_row(seed, row_index))
    })
}

fn create_orbit_variants_from_sampler<F>(
    orbits: &OrbitBatch,
    sample_capacity_per_orbit: usize,
    mut sampler: F,
) -> SchemaResult<OrbitVariantSamples>
where
    F: FnMut(usize, &[f64; DIM], &[f64]) -> SchemaResult<Vec<CoordinateSample>>,
{
    orbits.validate()?;
    let means = coordinate_rows(&orbits.coordinates.values);
    let covariances =
        orbits.coordinates.covariance.as_ref().ok_or_else(|| {
            SchemaError::MissingRequiredField("coordinates.covariance".to_string())
        })?;
    if covariances.dimension != DIM {
        return Err(SchemaError::InvalidCovarianceShape {
            rows: covariances.rows,
            dimension: covariances.dimension,
            values: covariances.values_row_major.len(),
        });
    }
    let times = orbits
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| SchemaError::MissingRequiredField("coordinates.time".to_string()))?;

    let expected_capacity = orbits.len() * sample_capacity_per_orbit;
    let mut sample_rows = Vec::with_capacity(expected_capacity);
    let mut source_orbit_indices = Vec::with_capacity(expected_capacity);
    let mut orbit_ids = Vec::with_capacity(expected_capacity);
    let mut object_ids = Vec::with_capacity(expected_capacity);
    let mut variant_ids = Vec::with_capacity(expected_capacity);
    let mut weights = Vec::with_capacity(expected_capacity);
    let mut weights_cov = Vec::with_capacity(expected_capacity);
    let mut origins = Vec::with_capacity(expected_capacity);
    let mut epochs = Vec::with_capacity(expected_capacity);

    for (row_index, mean) in means.iter().enumerate() {
        validate_sampling_row(mean, covariances.row_values(row_index))?;
        let samples = sampler(row_index, mean, covariances.row_values(row_index))?;
        for sample in samples {
            append_variant_row(
                row_index,
                sample.values,
                sample.weight,
                sample.weight_cov,
                orbits,
                times,
                &mut sample_rows,
                &mut source_orbit_indices,
                &mut orbit_ids,
                &mut object_ids,
                &mut variant_ids,
                &mut weights,
                &mut weights_cov,
                &mut origins,
                &mut epochs,
            );
        }
    }

    let coordinates = CoordinateBatch::new(
        coordinate_values_with_representation(orbits.coordinates.representation(), sample_rows),
        orbits.coordinates.frame,
        OriginArray::new(origins),
        Some(TimeArray::new(times.scale, epochs)?),
        None,
    )?;
    let variants = OrbitVariantBatch::new(
        orbit_ids,
        object_ids,
        variant_ids,
        weights,
        weights_cov,
        coordinates,
    )?;
    // Carry per-source-orbit physical parameters onto the sampled variants
    // (bead personal-cmy.13.2) so boundaries no longer reattach them.
    let variants = match orbits.physical_parameters.as_ref() {
        Some(physical_parameters) => {
            variants.with_physical_parameters(physical_parameters.take(&source_orbit_indices))?
        }
        None => variants,
    };
    Ok(OrbitVariantSamples {
        variants,
        source_orbit_indices,
    })
}

fn sigma_point_samples(
    mean: &[f64; DIM],
    covariance: &[f64],
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<Vec<CoordinateSample>> {
    let denom = alpha * alpha * (DIM as f64 + kappa);
    if !denom.is_finite() || denom <= 0.0 {
        return Err(SchemaError::InvalidRecordBatch(
            "sigma-point alpha^2 * (dimension + kappa) must be positive".to_string(),
        ));
    }
    let lambda = denom - DIM as f64;
    let w0 = lambda / denom;
    let w0_cov = w0 + (1.0 - alpha * alpha + beta);
    let wi = 1.0 / (2.0 * denom);
    let root = symmetric_square_root_scaled(covariance, denom)?;

    let mut samples = Vec::with_capacity(SIGMA_POINT_COUNT);
    samples.push(CoordinateSample {
        values: *mean,
        weight: w0,
        weight_cov: w0_cov,
    });
    for offset in root.iter().take(DIM) {
        samples.push(CoordinateSample {
            values: add_rows(mean, offset),
            weight: wi,
            weight_cov: wi,
        });
    }
    for offset in root.iter().take(DIM) {
        samples.push(CoordinateSample {
            values: sub_rows(mean, offset),
            weight: wi,
            weight_cov: wi,
        });
    }
    Ok(samples)
}

fn monte_carlo_samples(
    mean: &[f64; DIM],
    covariance: &[f64],
    num_samples: usize,
    seed: u64,
) -> SchemaResult<Vec<CoordinateSample>> {
    let root =
        symmetric_square_root_scaled_with_tolerance(covariance, 1.0, MONTE_CARLO_PSD_TOLERANCE)?;
    let mut rng = SplitMix64Normal::new(seed);
    let weight = 1.0 / num_samples as f64;
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let mut z = [0.0; DIM];
        for value in z.iter_mut().take(DIM) {
            *value = rng.standard_normal();
        }
        let mut values = *mean;
        for dim in 0..DIM {
            let mut offset = 0.0;
            for k in 0..DIM {
                offset += z[k] * root[k][dim];
            }
            values[dim] += offset;
        }
        samples.push(CoordinateSample {
            values,
            weight,
            weight_cov: weight,
        });
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn auto_samples(
    row_index: usize,
    mean: &[f64; DIM],
    covariance: &[f64],
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<Vec<CoordinateSample>> {
    let samples = sigma_point_samples(mean, covariance, alpha, beta, kappa)?;
    if sigma_points_reconstruct_input(mean, covariance, &samples) {
        return Ok(samples);
    }
    // Intentional deviation from legacy Python (decision 2026-07-03): the
    // user-supplied seed now threads into the Monte Carlo fallback so
    // auto-mode is reproducible given a seed. Legacy auto-mode always drew
    // an unseeded scipy sample here; exact scipy RNG parity is not required.
    monte_carlo_samples(mean, covariance, num_samples, seed_for_row(seed, row_index))
}

fn sigma_points_reconstruct_input(
    mean: &[f64; DIM],
    covariance: &[f64],
    samples: &[CoordinateSample],
) -> bool {
    let sample_values = samples
        .iter()
        .flat_map(|sample| sample.values.iter().copied())
        .collect::<Vec<_>>();
    let weights = samples
        .iter()
        .map(|sample| sample.weight)
        .collect::<Vec<_>>();
    let weights_cov = samples
        .iter()
        .map(|sample| sample.weight_cov)
        .collect::<Vec<_>>();
    let reconstructed_mean =
        crate::weighted_mean_flat(&sample_values, &weights, samples.len(), DIM);
    let reconstructed_covariance = crate::weighted_covariance_flat(
        &reconstructed_mean,
        &sample_values,
        &weights_cov,
        samples.len(),
        DIM,
    );
    mean.iter()
        .zip(reconstructed_mean.iter())
        .all(|(expected, actual)| (actual - expected).abs() < AUTO_RECONSTRUCTION_TOLERANCE)
        && covariance
            .iter()
            .zip(reconstructed_covariance.iter())
            .all(|(expected, actual)| (actual - expected).abs() < AUTO_RECONSTRUCTION_TOLERANCE)
}

/// Collapse propagated covariance variants into nominal-orbit covariance rows.
///
/// The nominal propagated state remains the mean, matching Python
/// `VariantOrbits.collapse(propagated_nominal)` semantics.
/// Collapse a variant ephemeris into per-(orbit, observer-epoch) covariance on
/// the nominal ephemeris. Mirrors the public Python `VariantEphemeris.collapse`:
/// a weighted covariance over the variant topocentric-spherical coordinates
/// (and the aberrated Cartesian coordinates when present), with the mean taken
/// from the nominal ephemeris.
///
/// Ephemeris output is orbit-major, observer-minor (`output_row =
/// orbit * observer_rows + observer`), so a variant row
/// `v * observer_rows + obs` contributes to nominal row
/// `source_orbit_indices[v] * observer_rows + obs`, and
/// `variant_weights_cov[v]` is variant `v`'s covariance weight. The nominal
/// ephemeris otherwise passes through unchanged (states, magnitudes,
/// light-time, validity, diagnostics), so the boundary/contract is identical to
/// the no-covariance path.
pub fn collapse_variant_ephemeris(
    nominal: &EphemerisResult,
    variant: &EphemerisResult,
    source_orbit_indices: &[usize],
    variant_weights_cov: &[f64],
    observer_rows: usize,
) -> PropagationResultValue<EphemerisResult> {
    let nominal_batch = &nominal.ephemeris;
    let variant_batch = &variant.ephemeris;

    let output_rows = nominal_batch.coordinates.len();
    let n_variants = source_orbit_indices.len();
    if observer_rows == 0 {
        return Err(PropagationError::InvalidRequest(
            "variant ephemeris collapse requires observer_rows > 0".to_string(),
        ));
    }
    if variant_weights_cov.len() != n_variants {
        return Err(PropagationError::InvalidRequest(
            "variant_weights_cov length must match source_orbit_indices".to_string(),
        ));
    }
    if variant_batch.coordinates.len() != n_variants * observer_rows {
        return Err(PropagationError::InvalidRequest(
            "variant ephemeris rows must equal n_variants * observer_rows".to_string(),
        ));
    }
    if !output_rows.is_multiple_of(observer_rows) {
        return Err(PropagationError::InvalidRequest(
            "nominal ephemeris rows must be a multiple of observer_rows".to_string(),
        ));
    }

    // Weighted covariance of one coordinate set (topocentric spherical or
    // aberrated Cartesian) using the nominal value as the mean, gathering the
    // variant samples by orbit-major/observer-minor index arithmetic.
    let collapse_values = |nominal_values: &[[f64; 6]], variant_values: &[[f64; 6]]| -> Vec<f64> {
        let mut samples_by_row = vec![Vec::<f64>::new(); output_rows];
        let mut weights_by_row = vec![Vec::<f64>::new(); output_rows];
        for (variant_index, &source) in source_orbit_indices.iter().enumerate() {
            let weight = variant_weights_cov[variant_index];
            for observer in 0..observer_rows {
                let nominal_row = source * observer_rows + observer;
                let variant_row = variant_index * observer_rows + observer;
                if nominal_row < output_rows && variant_row < variant_values.len() {
                    samples_by_row[nominal_row].extend_from_slice(&variant_values[variant_row]);
                    weights_by_row[nominal_row].push(weight);
                }
            }
        }
        let mut covariance = Vec::with_capacity(output_rows * DIM * DIM);
        for row in 0..output_rows {
            let count = weights_by_row[row].len();
            let valid = count > 0
                && samples_by_row[row].len() == count * DIM
                && nominal_batch.validity.is_valid(row);
            if valid {
                let cov = crate::weighted_covariance_flat(
                    &nominal_values[row],
                    &samples_by_row[row],
                    &weights_by_row[row],
                    count,
                    DIM,
                );
                if cov.iter().all(|value| value.is_finite()) {
                    covariance.extend_from_slice(&cov);
                    continue;
                }
            }
            covariance.extend(std::iter::repeat_n(f64::NAN, DIM * DIM));
        }
        covariance
    };

    // Topocentric spherical covariance.
    let nominal_spherical = nominal_batch
        .coordinates
        .values
        .spherical()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "variant ephemeris collapse requires spherical nominal coordinates".to_string(),
            )
        })?;
    let variant_spherical = variant_batch
        .coordinates
        .values
        .spherical()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "variant ephemeris collapse requires spherical variant coordinates".to_string(),
            )
        })?;
    let spherical_covariance = CovarianceBatch::new(
        output_rows,
        DIM,
        collapse_values(nominal_spherical, variant_spherical),
        CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
    )?;
    let collapsed_coordinates = CoordinateBatch::spherical(
        nominal_spherical.to_vec(),
        nominal_batch.coordinates.frame,
        nominal_batch.coordinates.origins.clone(),
        nominal_batch.coordinates.times.clone(),
        Some(spherical_covariance),
    )?;

    // Aberrated Cartesian covariance, when both sides carry aberrated states.
    let collapsed_aberrated = match (
        nominal_batch.aberrated_coordinates.as_ref(),
        variant_batch.aberrated_coordinates.as_ref(),
    ) {
        (Some(nominal_aberrated), Some(variant_aberrated)) => {
            let nominal_values = nominal_aberrated.values.cartesian().ok_or_else(|| {
                PropagationError::InvalidRequest(
                    "variant ephemeris collapse requires Cartesian aberrated nominal coordinates"
                        .to_string(),
                )
            })?;
            let variant_values = variant_aberrated.values.cartesian().ok_or_else(|| {
                PropagationError::InvalidRequest(
                    "variant ephemeris collapse requires Cartesian aberrated variant coordinates"
                        .to_string(),
                )
            })?;
            let aberrated_covariance = CovarianceBatch::new(
                output_rows,
                DIM,
                collapse_values(nominal_values, variant_values),
                CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
            )?;
            Some(CoordinateBatch::cartesian(
                nominal_values.to_vec(),
                nominal_aberrated.frame,
                nominal_aberrated.origins.clone(),
                nominal_aberrated.times.clone(),
                Some(aberrated_covariance),
            )?)
        }
        _ => nominal_batch.aberrated_coordinates.clone(),
    };

    let ephemeris = EphemerisBatch::new(
        nominal_batch.orbit_id.clone(),
        nominal_batch.object_id.clone(),
        collapsed_coordinates,
        nominal_batch.predicted_magnitude_v.clone(),
        nominal_batch.alpha_deg.clone(),
        nominal_batch.light_time_days.clone(),
        collapsed_aberrated,
        nominal_batch.validity.clone(),
    )?;
    Ok(EphemerisResult {
        ephemeris,
        diagnostics: nominal.diagnostics.clone(),
    })
}

pub fn collapse_propagated_variants_to_orbits(
    nominal: &PropagationResult,
    propagated_variants: &PropagationResult,
    source_orbit_indices: &[usize],
) -> PropagationResultValue<PropagationResult> {
    if nominal.variants.is_some() {
        return Err(PropagationError::InvalidRequest(
            "cannot collapse covariance variants into nominal VariantOrbits".to_string(),
        ));
    }
    let nominal_states = nominal
        .orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "variant covariance collapse currently requires Cartesian nominal output"
                    .to_string(),
            )
        })?;
    let variants = propagated_variants.variants.as_ref().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "propagated covariance collapse requires propagated VariantOrbits".to_string(),
        )
    })?;
    let variant_states = variants.coordinates.values.cartesian().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "variant covariance collapse currently requires Cartesian variant output".to_string(),
        )
    })?;
    if source_orbit_indices.len() != variants.coordinates.len() / nominal.times.len().max(1) {
        // This heuristic catches accidental use of unrelated source metadata while still
        // allowing the precise per-row checks below to own the protocol errors.
        if source_orbit_indices.len() < variants.coordinates.len() / nominal.times.len().max(1) {
            return Err(PropagationError::InvalidRequest(
                "source_orbit_indices is shorter than the propagated variant input rows"
                    .to_string(),
            ));
        }
    }

    let mut nominal_row_by_key = HashMap::with_capacity(nominal.diagnostics.convergence.len());
    for row in &nominal.diagnostics.convergence {
        nominal_row_by_key.insert(
            (row.input_orbit_index, row.input_time_index),
            row.output_row,
        );
    }

    let output_rows = nominal_states.len();
    let mut samples_by_row = vec![Vec::<f64>::new(); output_rows];
    let mut weights_by_row = vec![Vec::<f64>::new(); output_rows];
    let mut collapse_validity = vec![true; output_rows];

    for convergence in &propagated_variants.diagnostics.convergence {
        if convergence.input_orbit_index >= source_orbit_indices.len() {
            return Err(PropagationError::InvalidRequest(format!(
                "variant input index {} is outside source index table length {}",
                convergence.input_orbit_index,
                source_orbit_indices.len()
            )));
        }
        let source_orbit_index = source_orbit_indices[convergence.input_orbit_index];
        let nominal_row = *nominal_row_by_key
            .get(&(source_orbit_index, convergence.input_time_index))
            .ok_or_else(|| {
                PropagationError::InvalidRequest(format!(
                    "missing nominal output row for source orbit {source_orbit_index} time index {}",
                    convergence.input_time_index
                ))
            })?;
        if !propagated_variants
            .validity
            .is_valid(convergence.output_row)
        {
            collapse_validity[nominal_row] = false;
        }
        let weight = variants.weights_cov[convergence.output_row].ok_or_else(|| {
            PropagationError::InvalidRequest(
                "propagated covariance variant is missing weights_cov".to_string(),
            )
        })?;
        samples_by_row[nominal_row].extend_from_slice(&variant_states[convergence.output_row]);
        weights_by_row[nominal_row].push(weight);
    }

    let mut covariance_values = Vec::with_capacity(output_rows * DIM * DIM);
    for output_row in 0..output_rows {
        let sample_count = weights_by_row[output_row].len();
        if sample_count == 0 || samples_by_row[output_row].len() != sample_count * DIM {
            return Err(PropagationError::InvalidRequest(format!(
                "missing covariance samples for nominal output row {output_row}"
            )));
        }
        if collapse_validity[output_row] && nominal.validity.is_valid(output_row) {
            let covariance = crate::weighted_covariance_flat(
                &nominal_states[output_row],
                &samples_by_row[output_row],
                &weights_by_row[output_row],
                sample_count,
                DIM,
            );
            if covariance.iter().all(|value| value.is_finite()) {
                covariance_values.extend_from_slice(&covariance);
            } else {
                collapse_validity[output_row] = false;
                covariance_values.extend(std::iter::repeat_n(f64::NAN, DIM * DIM));
            }
        } else {
            collapse_validity[output_row] = false;
            covariance_values.extend(std::iter::repeat_n(f64::NAN, DIM * DIM));
        }
    }

    let covariance = CovarianceBatch::new(
        output_rows,
        DIM,
        covariance_values,
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )?
    .with_row_validity(Validity::from_bools(&collapse_validity))?;
    let coordinates = CoordinateBatch::cartesian(
        nominal_states.to_vec(),
        nominal.orbits.coordinates.frame,
        nominal.orbits.coordinates.origins.clone(),
        nominal.orbits.coordinates.times.clone(),
        Some(covariance),
    )?;
    let orbits = OrbitBatch::new(
        nominal.orbits.orbit_id.clone(),
        nominal.orbits.object_id.clone(),
        coordinates,
    )?;
    Ok(PropagationResult {
        orbits,
        variants: None,
        times: nominal.times.clone(),
        validity: nominal.validity.clone(),
        diagnostics: nominal.diagnostics.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn append_variant_row(
    source_index: usize,
    sample: [f64; DIM],
    weight: f64,
    weight_cov: f64,
    orbits: &OrbitBatch,
    times: &TimeArray,
    sample_rows: &mut Vec<[f64; DIM]>,
    source_orbit_indices: &mut Vec<usize>,
    orbit_ids: &mut Vec<crate::OrbitId>,
    object_ids: &mut Vec<Option<crate::ObjectId>>,
    variant_ids: &mut Vec<Option<VariantId>>,
    weights: &mut Vec<Option<f64>>,
    weights_cov: &mut Vec<Option<f64>>,
    origins: &mut Vec<crate::OriginId>,
    epochs: &mut Vec<crate::Epoch>,
) {
    let variant_index = sample_rows.len();
    sample_rows.push(sample);
    source_orbit_indices.push(source_index);
    orbit_ids.push(orbits.orbit_id[source_index].clone());
    object_ids.push(orbits.object_id[source_index].clone());
    variant_ids.push(Some(VariantId(variant_index.to_string())));
    weights.push(Some(weight));
    weights_cov.push(Some(weight_cov));
    origins.push(orbits.coordinates.origins.origins[source_index].clone());
    epochs.push(times.epochs[source_index]);
}

fn validate_sigma_point_parameters(alpha: f64, beta: f64, kappa: f64) -> SchemaResult<()> {
    if !alpha.is_finite() || !beta.is_finite() || !kappa.is_finite() {
        return Err(SchemaError::InvalidRecordBatch(
            "sigma-point covariance parameters must be finite".to_string(),
        ));
    }
    if alpha <= 0.0 {
        return Err(SchemaError::InvalidRecordBatch(
            "sigma-point alpha must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_monte_carlo_parameters(num_samples: usize) -> SchemaResult<()> {
    if num_samples == 0 {
        return Err(SchemaError::InvalidRecordBatch(
            "Monte Carlo covariance sampling requires num_samples > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_sampling_row(mean: &[f64; DIM], covariance: &[f64]) -> SchemaResult<()> {
    if mean.iter().any(|value| !value.is_finite()) {
        return Err(SchemaError::InvalidRecordBatch(
            "Cannot sample coordinate covariances when some coordinate dimensions are undefined."
                .to_string(),
        ));
    }
    if covariance.iter().any(|value| value.is_nan()) {
        return Err(SchemaError::InvalidRecordBatch(
            "Cannot sample coordinate covariances when some covariance elements are undefined."
                .to_string(),
        ));
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(SchemaError::InvalidRecordBatch(
            "Cannot sample coordinate covariances when covariance elements are non-finite."
                .to_string(),
        ));
    }
    Ok(())
}

fn coordinate_rows(values: &CoordinateValues) -> &[[f64; DIM]] {
    match values {
        CoordinateValues::Cartesian(rows)
        | CoordinateValues::Spherical(rows)
        | CoordinateValues::Keplerian(rows)
        | CoordinateValues::Cometary(rows)
        | CoordinateValues::Geodetic(rows) => rows,
    }
}

fn coordinate_values_with_representation(
    representation: CoordinateRepresentation,
    rows: Vec<[f64; DIM]>,
) -> CoordinateValues {
    match representation {
        CoordinateRepresentation::Cartesian => CoordinateValues::Cartesian(rows),
        CoordinateRepresentation::Spherical => CoordinateValues::Spherical(rows),
        CoordinateRepresentation::Keplerian => CoordinateValues::Keplerian(rows),
        CoordinateRepresentation::Cometary => CoordinateValues::Cometary(rows),
        CoordinateRepresentation::Geodetic => CoordinateValues::Geodetic(rows),
    }
}

fn add_rows(left: &[f64; DIM], right: &[f64; DIM]) -> [f64; DIM] {
    let mut out = [0.0; DIM];
    for index in 0..DIM {
        out[index] = left[index] + right[index];
    }
    out
}

fn sub_rows(left: &[f64; DIM], right: &[f64; DIM]) -> [f64; DIM] {
    let mut out = [0.0; DIM];
    for index in 0..DIM {
        out[index] = left[index] - right[index];
    }
    out
}

fn symmetric_square_root_scaled(
    values_row_major: &[f64],
    scale: f64,
) -> SchemaResult<[[f64; DIM]; DIM]> {
    symmetric_square_root_scaled_with_tolerance(values_row_major, scale, PSD_TOLERANCE)
}

fn symmetric_square_root_scaled_with_tolerance(
    values_row_major: &[f64],
    scale: f64,
    psd_tolerance: f64,
) -> SchemaResult<[[f64; DIM]; DIM]> {
    if values_row_major.len() != DIM * DIM {
        return Err(SchemaError::InvalidCovarianceShape {
            rows: 1,
            dimension: DIM,
            values: values_row_major.len(),
        });
    }
    let mut a = [[0.0; DIM]; DIM];
    for row in 0..DIM {
        for col in 0..DIM {
            let left = values_row_major[row * DIM + col];
            let right = values_row_major[col * DIM + row];
            a[row][col] = 0.5 * scale * (left + right);
        }
    }
    let mut vectors = [[0.0; DIM]; DIM];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    for _ in 0..JACOBI_MAX_SWEEPS {
        let (p, q, max_off_diag) = max_off_diagonal(&a);
        if max_off_diag <= JACOBI_TOLERANCE {
            break;
        }
        rotate_jacobi(&mut a, &mut vectors, p, q);
    }

    let mut roots = [0.0; DIM];
    for index in 0..DIM {
        let eigenvalue = a[index][index];
        if eigenvalue < -psd_tolerance {
            return Err(SchemaError::InvalidRecordBatch(
                "Covariance matrix is not positive semidefinite for covariance sampling."
                    .to_string(),
            ));
        }
        roots[index] = eigenvalue.max(0.0).sqrt();
    }

    let mut sqrt = [[0.0; DIM]; DIM];
    for row in 0..DIM {
        for col in 0..DIM {
            let mut value = 0.0;
            for k in 0..DIM {
                value += vectors[row][k] * roots[k] * vectors[col][k];
            }
            sqrt[row][col] = value;
        }
    }
    Ok(sqrt)
}

fn max_off_diagonal(a: &[[f64; DIM]; DIM]) -> (usize, usize, f64) {
    let mut p = 0;
    let mut q = 1;
    let mut max_value = a[p][q].abs();
    for (row, values) in a.iter().enumerate().take(DIM) {
        for (col, value) in values.iter().enumerate().take(DIM).skip(row + 1) {
            let abs_value = value.abs();
            if abs_value > max_value {
                p = row;
                q = col;
                max_value = abs_value;
            }
        }
    }
    (p, q, max_value)
}

fn rotate_jacobi(a: &mut [[f64; DIM]; DIM], vectors: &mut [[f64; DIM]; DIM], p: usize, q: usize) {
    let app = a[p][p];
    let aqq = a[q][q];
    let apq = a[p][q];
    if apq == 0.0 {
        return;
    }
    let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
    let c = theta.cos();
    let s = theta.sin();

    // Jacobi rotation updates rows p/q and column k of the same matrix, so
    // direct index access is clearer than an iterator here.
    #[allow(clippy::needless_range_loop)]
    for k in 0..DIM {
        if k != p && k != q {
            let akp = a[k][p];
            let akq = a[k][q];
            let new_kp = c * akp - s * akq;
            let new_kq = s * akp + c * akq;
            a[k][p] = new_kp;
            a[p][k] = new_kp;
            a[k][q] = new_kq;
            a[q][k] = new_kq;
        }
    }
    a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    for row in vectors.iter_mut().take(DIM) {
        let vip = row[p];
        let viq = row[q];
        row[p] = c * vip - s * viq;
        row[q] = s * vip + c * viq;
    }
}

fn seed_for_row(seed: Option<u64>, row_index: usize) -> u64 {
    if let Some(seed) = seed {
        return seed;
    }
    DEFAULT_RANDOM_SEED ^ ((row_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

#[derive(Debug, Clone)]
struct SplitMix64Normal {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64Normal {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_open_unit(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / TWO_POW_53
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let u1 = self.next_open_unit();
        let u2 = self.next_open_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some(radius * theta.sin());
        radius * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::{
        EpochOrder, PropagationConvergence, PropagationConvergenceStatus, PropagationDiagnostics,
    };
    use crate::types::Frame;
    use crate::{Epoch, OrbitId, OriginId, TimeScale, Validity};

    fn sample_orbits() -> (OrbitBatch, [f64; DIM], Vec<f64>) {
        let state = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let mut covariance = vec![0.0; DIM * DIM];
        for i in 0..DIM {
            covariance[i * DIM + i] = 1.0e-6 * (i as f64 + 1.0);
        }
        covariance[1] = 2.0e-7;
        covariance[DIM] = 2.0e-7;
        let covariance_batch = CovarianceBatch::new(
            1,
            DIM,
            covariance.clone(),
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            Some(covariance_batch),
        )
        .unwrap();
        let orbits =
            OrbitBatch::new(vec![OrbitId("o1".to_string())], vec![None], coordinates).unwrap();
        (orbits, state, covariance)
    }

    fn assert_reconstructs_covariance(
        samples: &OrbitVariantSamples,
        mean: &[f64; DIM],
        covariance: &[f64],
    ) {
        let variant_values = coordinate_rows(&samples.variants.coordinates.values)
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let weights = samples
            .variants
            .weights_cov
            .iter()
            .map(|value| value.unwrap())
            .collect::<Vec<_>>();
        let reconstructed = crate::weighted_covariance_flat(
            mean,
            &variant_values,
            &weights,
            samples.variants.len(),
            DIM,
        );
        for (actual, expected) in reconstructed.iter().zip(covariance.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-18,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn sigma_point_variants_reconstruct_input_covariance() {
        let (orbits, state, covariance) = sample_orbits();
        let samples = create_sigma_point_orbit_variants(&orbits, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(samples.variants.len(), SIGMA_POINT_COUNT);
        assert_eq!(samples.source_orbit_indices, vec![0; SIGMA_POINT_COUNT]);
        assert_reconstructs_covariance(&samples, &state, &covariance);
    }

    #[test]
    fn sigma_point_variants_reconstruct_tiny_public_scale_covariance() {
        let state = [1.05, 0.0, 0.0, 0.0, 0.016787, 0.0];
        let sigmas = [1.0e-9, 2.0e-9, 3.0e-9, 1.0e-10, 2.0e-10, 3.0e-10];
        let mut covariance = vec![0.0; DIM * DIM];
        for i in 0..DIM {
            covariance[i * DIM + i] = sigmas[i] * sigmas[i];
        }
        let covariance_batch = CovarianceBatch::new(
            1,
            DIM,
            covariance.clone(),
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            Some(covariance_batch),
        )
        .unwrap();
        let orbits =
            OrbitBatch::new(vec![OrbitId("o1".to_string())], vec![None], coordinates).unwrap();
        let samples = create_sigma_point_orbit_variants(&orbits, 1.0, 0.0, 0.0).unwrap();
        assert_reconstructs_covariance(&samples, &state, &covariance);
    }

    #[test]
    fn auto_variants_use_sigma_points_for_well_conditioned_covariance() {
        let (orbits, state, covariance) = sample_orbits();
        let samples = create_sampled_orbit_variants(
            &orbits,
            OrbitVariantSamplingMethod::Auto,
            64,
            Some(7),
            1.0,
            0.0,
            0.0,
        )
        .unwrap();
        assert_eq!(samples.variants.len(), SIGMA_POINT_COUNT);
        assert_reconstructs_covariance(&samples, &state, &covariance);
    }

    #[test]
    fn monte_carlo_variants_use_requested_sample_count_and_seed() {
        let (orbits, _state, _covariance) = sample_orbits();
        let left = create_monte_carlo_orbit_variants(&orbits, 8, Some(42)).unwrap();
        let right = create_monte_carlo_orbit_variants(&orbits, 8, Some(42)).unwrap();
        assert_eq!(left.variants.len(), 8);
        assert_eq!(left.source_orbit_indices, vec![0; 8]);
        assert!(left
            .variants
            .weights
            .iter()
            .all(|weight| (weight.unwrap() - 0.125).abs() < 1.0e-15));
        assert_eq!(
            coordinate_rows(&left.variants.coordinates.values),
            coordinate_rows(&right.variants.coordinates.values)
        );
    }

    #[test]
    fn propagated_variants_collapse_reconstructs_covariance() {
        let (orbits, _state, covariance) = sample_orbits();
        let samples = create_sigma_point_orbit_variants(&orbits, 1.0, 0.0, 0.0).unwrap();
        let target_times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_001, 0)]).unwrap();
        let nominal = PropagationResult {
            orbits: orbits.clone(),
            variants: None,
            times: target_times.clone(),
            validity: Validity::all_valid(1),
            diagnostics: PropagationDiagnostics {
                convergence: vec![PropagationConvergence {
                    output_row: 0,
                    input_orbit_index: 0,
                    input_time_index: 0,
                    status: PropagationConvergenceStatus::Converged,
                    backend: None,
                    iterations: None,
                    failure_code: None,
                    message: None,
                }],
                epoch_order: EpochOrder::from_times(&target_times),
            },
        };
        let variant_convergence = (0..samples.variants.len())
            .map(|index| PropagationConvergence {
                output_row: index,
                input_orbit_index: index,
                input_time_index: 0,
                status: PropagationConvergenceStatus::Converged,
                backend: None,
                iterations: None,
                failure_code: None,
                message: None,
            })
            .collect::<Vec<_>>();
        let propagated_variants = PropagationResult {
            orbits: samples.variants.to_orbit_batch().unwrap(),
            variants: Some(samples.variants.clone()),
            times: target_times.clone(),
            validity: Validity::all_valid(samples.variants.len()),
            diagnostics: PropagationDiagnostics {
                convergence: variant_convergence,
                epoch_order: EpochOrder::from_times(&target_times),
            },
        };
        let collapsed = collapse_propagated_variants_to_orbits(
            &nominal,
            &propagated_variants,
            &samples.source_orbit_indices,
        )
        .unwrap();
        let collapsed_covariance = collapsed
            .orbits
            .coordinates
            .covariance
            .as_ref()
            .unwrap()
            .row_values(0);
        for (actual, expected) in collapsed_covariance.iter().zip(covariance.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-18,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn propagated_variants_collapse_uses_output_row_weights_for_multiple_times() {
        let (orbits, state, covariance) = sample_orbits();
        let samples = create_sigma_point_orbit_variants(&orbits, 1.0, 0.0, 0.0).unwrap();
        let target_times = TimeArray::new(
            TimeScale::Tdb,
            vec![Epoch::new(60_001, 0), Epoch::new(60_002, 0)],
        )
        .unwrap();
        let nominal_coordinates = CoordinateBatch::cartesian(
            vec![state, state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            Some(target_times.clone()),
            None,
        )
        .unwrap();
        let nominal_orbits = OrbitBatch::new(
            vec![OrbitId("o1".to_string()), OrbitId("o1".to_string())],
            vec![None, None],
            nominal_coordinates,
        )
        .unwrap();
        let nominal = PropagationResult {
            orbits: nominal_orbits,
            variants: None,
            times: target_times.clone(),
            validity: Validity::all_valid(2),
            diagnostics: PropagationDiagnostics {
                convergence: (0..2)
                    .map(|time_index| PropagationConvergence {
                        output_row: time_index,
                        input_orbit_index: 0,
                        input_time_index: time_index,
                        status: PropagationConvergenceStatus::Converged,
                        backend: None,
                        iterations: None,
                        failure_code: None,
                        message: None,
                    })
                    .collect(),
                epoch_order: EpochOrder::from_times(&target_times),
            },
        };

        let sample_rows = coordinate_rows(&samples.variants.coordinates.values);
        let mut output_rows = Vec::with_capacity(samples.variants.len() * 2);
        let mut orbit_ids = Vec::with_capacity(samples.variants.len() * 2);
        let mut object_ids = Vec::with_capacity(samples.variants.len() * 2);
        let mut variant_ids = Vec::with_capacity(samples.variants.len() * 2);
        let mut weights = Vec::with_capacity(samples.variants.len() * 2);
        let mut weights_cov = Vec::with_capacity(samples.variants.len() * 2);
        let mut origins = Vec::with_capacity(samples.variants.len() * 2);
        let mut epochs = Vec::with_capacity(samples.variants.len() * 2);
        let mut convergence = Vec::with_capacity(samples.variants.len() * 2);
        for (sample_index, sample) in sample_rows.iter().enumerate() {
            for time_index in 0..2 {
                let output_row = output_rows.len();
                output_rows.push(*sample);
                orbit_ids.push(samples.variants.orbit_id[sample_index].clone());
                object_ids.push(samples.variants.object_id[sample_index].clone());
                variant_ids.push(samples.variants.variant_id[sample_index].clone());
                weights.push(samples.variants.weights[sample_index]);
                weights_cov.push(samples.variants.weights_cov[sample_index]);
                origins.push(OriginId::SolarSystemBarycenter);
                epochs.push(target_times.epochs[time_index]);
                convergence.push(PropagationConvergence {
                    output_row,
                    input_orbit_index: sample_index,
                    input_time_index: time_index,
                    status: PropagationConvergenceStatus::Converged,
                    backend: None,
                    iterations: None,
                    failure_code: None,
                    message: None,
                });
            }
        }
        let variant_coordinates = CoordinateBatch::cartesian(
            output_rows,
            Frame::Ecliptic,
            OriginArray::new(origins),
            Some(TimeArray::new(TimeScale::Tdb, epochs).unwrap()),
            None,
        )
        .unwrap();
        let output_variants = OrbitVariantBatch::new(
            orbit_ids,
            object_ids,
            variant_ids,
            weights,
            weights_cov,
            variant_coordinates,
        )
        .unwrap();
        let propagated_variants = PropagationResult {
            orbits: output_variants.to_orbit_batch().unwrap(),
            variants: Some(output_variants),
            times: target_times.clone(),
            validity: Validity::all_valid(samples.variants.len() * 2),
            diagnostics: PropagationDiagnostics {
                convergence,
                epoch_order: EpochOrder::from_times(&target_times),
            },
        };

        let collapsed = collapse_propagated_variants_to_orbits(
            &nominal,
            &propagated_variants,
            &samples.source_orbit_indices,
        )
        .unwrap();
        let collapsed_covariance = collapsed.orbits.coordinates.covariance.as_ref().unwrap();
        for output_row in 0..2 {
            for (actual, expected) in collapsed_covariance
                .row_values(output_row)
                .iter()
                .zip(covariance.iter())
            {
                assert!(
                    (actual - expected).abs() < 1.0e-18,
                    "{actual} != {expected}"
                );
            }
        }
    }
}
