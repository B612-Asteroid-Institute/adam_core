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
    EphemerisBatch, NonGravitationalParametersBatch, OrbitBatch, OrbitVariantBatch, OriginArray,
    SchemaError, TimeArray, Validity, VariantId,
};
use std::collections::HashMap;

const DIM: usize = 6;
const SIGMA_POINT_COUNT: usize = 2 * DIM + 1;
/// Jacobi rotation budget as 30 sweeps over every upper-triangle pair.
/// Classical max-element Jacobi converges quadratically, so well-posed
/// covariances need far fewer rotations; this only bounds pathological inputs.
const JACOBI_MAX_SWEEPS: usize = 30;
/// Scale-aware Jacobi convergence threshold, relative to the largest absolute
/// entry of the symmetrized, scaled matrix. An absolute threshold is not
/// meaningful across covariance scales: public orbit covariances near 1e-17
/// "converged" against the previous absolute 1e-18 cutoff while material
/// off-diagonal structure remained, and the resulting sigma-point clouds
/// failed to reconstruct the input covariance by ~0.86% (bead personal-yv7s).
const JACOBI_RELATIVE_TOLERANCE: f64 = f64::EPSILON;
/// Sigma-point PSD floor relative to the largest absolute eigenvalue. Legacy
/// Python had no PSD validation on the sigma-point path (scipy `sqrtm`
/// silently returned complex roots), so this scale-aware floor keeps the
/// established Rust fail-closed behavior while staying meaningful for both
/// order-one and public-scale (~1e-17) covariances. For order-one matrices it
/// matches the previous absolute 1e-12 floor exactly.
const SIGMA_POINT_PSD_RELATIVE_TOLERANCE: f64 = 1.0e-12;
/// Absolute Monte Carlo eigenvalue floor mirroring the legacy public
/// `sample_covariance_random(semidef_tol=1e-15)` acceptance contract.
const MONTE_CARLO_PSD_TOLERANCE: f64 = 1.0e-15;
/// Absolute auto-mode reconstruction thresholds mirroring the legacy Python
/// `create_coordinate_variants` sigma-point-versus-Monte-Carlo decision
/// (`diff >= 1e-12` per entry). This is intentionally a method-selection
/// contract, not an accuracy guarantee; accuracy is owned by the scale-aware
/// symmetric square root below.
const AUTO_RECONSTRUCTION_TOLERANCE: f64 = 1.0e-12;
const DEFAULT_RANDOM_SEED: u64 = 0x4d59_5df4_d0f3_3173;
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;

#[derive(Debug, Clone, PartialEq)]
pub struct OrbitVariantSamples {
    pub variants: OrbitVariantBatch,
    /// `source_orbit_indices[variant_row] = original orbit row index`.
    pub source_orbit_indices: Vec<usize>,
    /// Widest sampled solved-state dimension in the batch.
    pub covariance_dimension: usize,
    /// Solved-state dimension for each original orbit row (6 or 9).
    pub source_covariance_dimensions: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitVariantSamplingMethod {
    Auto,
    SigmaPoint,
    MonteCarlo,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CovarianceSample<T> {
    values: T,
    weight: f64,
    weight_cov: f64,
}

type CoordinateSample<const N: usize> = CovarianceSample<[f64; N]>;

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
    create_sampled_orbit_variants_with_nongrav(
        orbits,
        method,
        num_samples,
        seed,
        alpha,
        beta,
        kappa,
        true,
    )
}

/// Create orbit variants while optionally excluding the solved A1/A2/A3
/// dimensions. Excluding them reduces every extended covariance row to its
/// leading coordinate block and removes the parameter values before sampling.
#[allow(clippy::too_many_arguments)]
pub fn create_sampled_orbit_variants_with_nongrav(
    orbits: &OrbitBatch,
    method: OrbitVariantSamplingMethod,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
    include_nongrav: bool,
) -> SchemaResult<OrbitVariantSamples> {
    let stripped = if include_nongrav {
        None
    } else {
        Some(without_non_gravitational_solution(orbits)?)
    };
    let sampling_orbits = stripped.as_ref().unwrap_or(orbits);
    match method {
        OrbitVariantSamplingMethod::Auto => {
            validate_sigma_point_parameters(alpha, beta, kappa)?;
            validate_monte_carlo_parameters(num_samples)?;
            dispatch_orbit_variant_sampler(
                sampling_orbits,
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
            create_sigma_point_orbit_variants(sampling_orbits, alpha, beta, kappa)
        }
        OrbitVariantSamplingMethod::MonteCarlo => {
            create_monte_carlo_orbit_variants(sampling_orbits, num_samples, seed)
        }
    }
}

fn without_non_gravitational_solution(orbits: &OrbitBatch) -> SchemaResult<OrbitBatch> {
    let mut stripped = orbits.clone();
    stripped.non_gravitational_parameters = None;
    let Some(covariance) = stripped.coordinates.covariance.as_ref() else {
        return Ok(stripped);
    };
    if covariance.dimension == DIM {
        return Ok(stripped);
    }
    if covariance.dimension != 9 {
        return Err(SchemaError::InvalidCovarianceShape {
            rows: covariance.rows,
            dimension: covariance.dimension,
            values: covariance.values_row_major.len(),
        });
    }
    let mut values = Vec::with_capacity(covariance.rows * DIM * DIM);
    for row in 0..covariance.rows {
        let source = covariance.row_values(row);
        for coordinate_row in 0..DIM {
            values.extend_from_slice(
                &source[coordinate_row * covariance.dimension
                    ..coordinate_row * covariance.dimension + DIM],
            );
        }
    }
    let coordinate_covariance =
        CovarianceBatch::new(covariance.rows, DIM, values, covariance.units.clone())?;
    stripped.coordinates.covariance = Some(match covariance.row_validity.clone() {
        Some(validity) => coordinate_covariance.with_row_validity(validity)?,
        None => coordinate_covariance,
    });
    Ok(stripped)
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
    dispatch_orbit_variant_sampler(
        orbits,
        19,
        |_row_index, mean, covariance| sigma_point_samples(mean, covariance, alpha, beta, kappa),
        |_row_index, mean, covariance| sigma_point_samples(mean, covariance, alpha, beta, kappa),
    )
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
    dispatch_orbit_variant_sampler(
        orbits,
        num_samples,
        |row_index, mean, covariance| {
            monte_carlo_samples(mean, covariance, num_samples, seed_for_row(seed, row_index))
        },
        |row_index, mean, covariance| {
            monte_carlo_samples(mean, covariance, num_samples, seed_for_row(seed, row_index))
        },
    )
}

fn dispatch_orbit_variant_sampler<F6, F9>(
    orbits: &OrbitBatch,
    sample_capacity_per_orbit: usize,
    sampler6: F6,
    sampler9: F9,
) -> SchemaResult<OrbitVariantSamples>
where
    F6: FnMut(usize, &[f64; 6], &[f64]) -> SchemaResult<Vec<CoordinateSample<6>>>,
    F9: FnMut(usize, &[f64; 9], &[f64]) -> SchemaResult<Vec<CoordinateSample<9>>>,
{
    let covariance =
        orbits.coordinates.covariance.as_ref().ok_or_else(|| {
            SchemaError::MissingRequiredField("coordinates.covariance".to_string())
        })?;
    let source_covariance_dimensions = covariance_row_dimensions(covariance)?;
    let has_six = source_covariance_dimensions.contains(&6);
    let has_nine = source_covariance_dimensions.contains(&9);
    let row_order = if has_six && has_nine {
        source_covariance_dimensions
            .iter()
            .enumerate()
            .filter_map(|(row, &dimension)| (dimension == 9).then_some(row))
            .chain(
                source_covariance_dimensions
                    .iter()
                    .enumerate()
                    .filter_map(|(row, &dimension)| (dimension == 6).then_some(row)),
            )
            .collect::<Vec<_>>()
    } else {
        (0..orbits.len()).collect()
    };
    create_orbit_variants_from_samplers(
        orbits,
        sample_capacity_per_orbit,
        &source_covariance_dimensions,
        &row_order,
        sampler6,
        sampler9,
    )
}

fn covariance_row_dimensions(covariance: &CovarianceBatch) -> SchemaResult<Vec<usize>> {
    match covariance.dimension {
        6 => Ok(vec![6; covariance.rows]),
        9 => Ok((0..covariance.rows)
            .map(|row| covariance.row_dimension(row))
            .collect()),
        dimension => Err(SchemaError::InvalidCovarianceShape {
            rows: covariance.rows,
            dimension,
            values: covariance.values_row_major.len(),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn create_orbit_variants_from_samplers<F6, F9>(
    orbits: &OrbitBatch,
    sample_capacity_per_orbit: usize,
    source_covariance_dimensions: &[usize],
    row_order: &[usize],
    mut sampler6: F6,
    mut sampler9: F9,
) -> SchemaResult<OrbitVariantSamples>
where
    F6: FnMut(usize, &[f64; 6], &[f64]) -> SchemaResult<Vec<CoordinateSample<6>>>,
    F9: FnMut(usize, &[f64; 9], &[f64]) -> SchemaResult<Vec<CoordinateSample<9>>>,
{
    orbits.validate()?;
    let coordinate_means = coordinate_rows(&orbits.coordinates.values);
    let covariances =
        orbits.coordinates.covariance.as_ref().ok_or_else(|| {
            SchemaError::MissingRequiredField("coordinates.covariance".to_string())
        })?;
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
    let mut sampled_nongrav = Vec::with_capacity(expected_capacity);

    for &row_index in row_order {
        let coordinate_mean = coordinate_means[row_index];
        match source_covariance_dimensions[row_index] {
            6 => {
                let covariance = if covariances.dimension == 6 {
                    covariances.row_values(row_index).to_vec()
                } else {
                    let source = covariances.row_values(row_index);
                    let mut values = Vec::with_capacity(36);
                    for coordinate_row in 0..6 {
                        values
                            .extend_from_slice(&source[coordinate_row * 9..coordinate_row * 9 + 6]);
                    }
                    values
                };
                validate_sampling_row(&coordinate_mean, &covariance)?;
                for sample in sampler6(row_index, &coordinate_mean, &covariance)? {
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
                    sampled_nongrav.push(None);
                }
            }
            9 => {
                let mut mean = [0.0_f64; 9];
                mean[..DIM].copy_from_slice(&coordinate_mean);
                if let Some(parameters) = &orbits.non_gravitational_parameters {
                    mean[6] = parameters.a1[row_index].unwrap_or(0.0);
                    mean[7] = parameters.a2[row_index].unwrap_or(0.0);
                    mean[8] = parameters.a3[row_index].unwrap_or(0.0);
                }
                let covariance = covariances.row_values(row_index);
                validate_sampling_row(&mean, covariance)?;
                for sample in sampler9(row_index, &mean, covariance)? {
                    append_variant_row(
                        row_index,
                        sample.values[..DIM]
                            .try_into()
                            .expect("sample contains six coordinate values"),
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
                    sampled_nongrav.push(Some([
                        sample.values[6],
                        sample.values[7],
                        sample.values[8],
                    ]));
                }
            }
            dimension => {
                return Err(SchemaError::InvalidCovarianceShape {
                    rows: covariances.rows,
                    dimension,
                    values: covariances.values_row_major.len(),
                });
            }
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
    let variants = match orbits.physical_parameters.as_ref() {
        Some(physical_parameters) => {
            variants.with_physical_parameters(physical_parameters.take(&source_orbit_indices))?
        }
        None => variants,
    };
    let variants = if orbits.non_gravitational_parameters.is_some()
        || sampled_nongrav.iter().any(Option::is_some)
    {
        let mut parameters = orbits
            .non_gravitational_parameters
            .as_ref()
            .map(|parameters| parameters.take(&source_orbit_indices))
            .unwrap_or_else(|| null_non_gravitational_parameters(variants.len()));
        for (row, sample) in sampled_nongrav.into_iter().enumerate() {
            if let Some([a1, a2, a3]) = sample {
                parameters.a1[row] = Some(a1);
                parameters.a2[row] = Some(a2);
                parameters.a3[row] = Some(a3);
            }
        }
        variants.with_non_gravitational_parameters(parameters)?
    } else {
        variants
    };
    Ok(OrbitVariantSamples {
        variants,
        source_orbit_indices,
        covariance_dimension: source_covariance_dimensions
            .iter()
            .copied()
            .max()
            .unwrap_or(DIM),
        source_covariance_dimensions: source_covariance_dimensions.to_vec(),
    })
}

fn null_non_gravitational_parameters(rows: usize) -> NonGravitationalParametersBatch {
    NonGravitationalParametersBatch {
        source: vec![None; rows],
        a1: vec![None; rows],
        a2: vec![None; rows],
        a3: vec![None; rows],
        aln: vec![None; rows],
        nk: vec![None; rows],
        nm: vec![None; rows],
        nn: vec![None; rows],
        r0: vec![None; rows],
    }
}

/// Flat per-row covariance sampling with public `create_coordinate_variants`
/// semantics: NaN covariance/mean rows are rejected with the legacy error
/// messages and per-row samples are concatenated with source-row indices.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn sample_coordinate_covariances_flat(
    means_flat: &[f64],
    covariances_flat: &[f64],
    dimension: usize,
    method: OrbitVariantSamplingMethod,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>)> {
    match dimension {
        6 => sample_coordinate_covariances_dimension::<6>(
            means_flat,
            covariances_flat,
            method,
            num_samples,
            seed,
            alpha,
            beta,
            kappa,
        ),
        9 => sample_coordinate_covariances_dimension::<9>(
            means_flat,
            covariances_flat,
            method,
            num_samples,
            seed,
            alpha,
            beta,
            kappa,
        ),
        _ => Err(SchemaError::InvalidRecordBatch(format!(
            "coordinate covariance sampling supports dimensions 6 or 9, got {dimension}"
        ))),
    }
}

/// Public single-distribution sigma-point sampler matching
/// `sample_covariance_sigma_points` output ordering and weights.
pub fn sample_covariance_sigma_points_flat(
    mean: &[f64],
    covariance: &[f64],
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_sigma_point_parameters(alpha, beta, kappa)?;
    match mean.len() {
        6 => sample_sigma_points_dimension::<6>(mean, covariance, alpha, beta, kappa),
        9 => sample_sigma_points_dimension::<9>(mean, covariance, alpha, beta, kappa),
        dimension => Err(SchemaError::InvalidRecordBatch(format!(
            "covariance sampling supports dimensions 6 or 9, got {dimension}"
        ))),
    }
}

/// Public single-distribution Monte Carlo sampler (Rust-native RNG;
/// statistically equivalent to, but not bit-identical with, the legacy scipy
/// sampler per decision 2026-07-03).
pub fn sample_covariance_random_flat(
    mean: &[f64],
    covariance: &[f64],
    num_samples: usize,
    seed: Option<u64>,
) -> SchemaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_monte_carlo_parameters(num_samples)?;
    match mean.len() {
        6 => sample_random_dimension::<6>(mean, covariance, num_samples, seed),
        9 => sample_random_dimension::<9>(mean, covariance, num_samples, seed),
        dimension => Err(SchemaError::InvalidRecordBatch(format!(
            "covariance sampling supports dimensions 6 or 9, got {dimension}"
        ))),
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sample_coordinate_covariances_dimension<const N: usize>(
    means_flat: &[f64],
    covariances_flat: &[f64],
    method: OrbitVariantSamplingMethod,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>)> {
    let rows = means_flat.len() / N;
    if !means_flat.len().is_multiple_of(N) || covariances_flat.len() != rows * N * N {
        return Err(SchemaError::InvalidRecordBatch(format!(
            "means must have shape (N, {N}) and covariances (N, {N}, {N})"
        )));
    }
    match method {
        OrbitVariantSamplingMethod::Auto => {
            validate_sigma_point_parameters(alpha, beta, kappa)?;
            validate_monte_carlo_parameters(num_samples)?;
        }
        OrbitVariantSamplingMethod::SigmaPoint => {
            validate_sigma_point_parameters(alpha, beta, kappa)?
        }
        OrbitVariantSamplingMethod::MonteCarlo => validate_monte_carlo_parameters(num_samples)?,
    }
    let mut samples_out = Vec::new();
    let mut weights = Vec::new();
    let mut weights_cov = Vec::new();
    let mut source_rows = Vec::new();
    for row in 0..rows {
        let covariance = &covariances_flat[row * N * N..(row + 1) * N * N];
        if covariance.iter().any(|value| value.is_nan()) {
            return Err(SchemaError::InvalidRecordBatch(
                "Cannot sample coordinate covariances when some covariance elements are undefined."
                    .to_string(),
            ));
        }
        let mean_slice = &means_flat[row * N..(row + 1) * N];
        if mean_slice.iter().any(|value| value.is_nan()) {
            return Err(SchemaError::InvalidRecordBatch(
                "Cannot sample coordinate covariances when some coordinate dimensions are undefined."
                    .to_string(),
            ));
        }
        let mut mean = [0.0_f64; N];
        mean.copy_from_slice(mean_slice);
        validate_sampling_row(&mean, covariance)?;
        let row_samples = match method {
            OrbitVariantSamplingMethod::SigmaPoint => {
                sigma_point_samples(&mean, covariance, alpha, beta, kappa)?
            }
            OrbitVariantSamplingMethod::MonteCarlo => {
                monte_carlo_samples(&mean, covariance, num_samples, seed_for_row(seed, row))?
            }
            OrbitVariantSamplingMethod::Auto => auto_samples(
                row,
                &mean,
                covariance,
                num_samples,
                seed,
                alpha,
                beta,
                kappa,
            )?,
        };
        for sample in row_samples {
            samples_out.extend_from_slice(&sample.values);
            weights.push(sample.weight);
            weights_cov.push(sample.weight_cov);
            source_rows.push(row as i64);
        }
    }
    Ok((samples_out, weights, weights_cov, source_rows))
}

fn sample_sigma_points_dimension<const N: usize>(
    mean: &[f64],
    covariance: &[f64],
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mean: [f64; N] = mean
        .try_into()
        .map_err(|_| SchemaError::InvalidRecordBatch(format!("mean must have shape ({N},)")))?;
    validate_sampling_row(&mean, covariance)?;
    Ok(split_samples(sigma_point_samples(
        &mean, covariance, alpha, beta, kappa,
    )?))
}

fn sample_random_dimension<const N: usize>(
    mean: &[f64],
    covariance: &[f64],
    num_samples: usize,
    seed: Option<u64>,
) -> SchemaResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mean: [f64; N] = mean
        .try_into()
        .map_err(|_| SchemaError::InvalidRecordBatch(format!("mean must have shape ({N},)")))?;
    validate_sampling_row(&mean, covariance)?;
    Ok(split_samples(monte_carlo_samples(
        &mean,
        covariance,
        num_samples,
        seed_for_row(seed, 0),
    )?))
}

fn split_samples<const N: usize>(
    samples: Vec<CoordinateSample<N>>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut values = Vec::with_capacity(samples.len() * N);
    let mut weights = Vec::with_capacity(samples.len());
    let mut weights_cov = Vec::with_capacity(samples.len());
    for sample in samples {
        values.extend_from_slice(&sample.values);
        weights.push(sample.weight);
        weights_cov.push(sample.weight_cov);
    }
    (values, weights, weights_cov)
}

fn sigma_point_samples<const N: usize>(
    mean: &[f64; N],
    covariance: &[f64],
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<Vec<CoordinateSample<N>>> {
    let denom = alpha * alpha * (N as f64 + kappa);
    if !denom.is_finite() || denom <= 0.0 {
        return Err(SchemaError::InvalidRecordBatch(
            "sigma-point alpha^2 * (dimension + kappa) must be positive".to_string(),
        ));
    }
    let lambda = denom - N as f64;
    let w0 = lambda / denom;
    let w0_cov = w0 + (1.0 - alpha * alpha + beta);
    let wi = 1.0 / (2.0 * denom);
    if !w0.is_finite() || !w0_cov.is_finite() || !wi.is_finite() {
        return Err(SchemaError::InvalidRecordBatch(
            "sigma-point covariance parameters must produce finite weights".to_string(),
        ));
    }
    let root = symmetric_square_root_scaled::<N>(covariance, denom)?;

    let mut samples = Vec::with_capacity(2 * N + 1);
    samples.push(CoordinateSample::<N> {
        values: *mean,
        weight: w0,
        weight_cov: w0_cov,
    });
    for offset in root.iter().take(N) {
        samples.push(CoordinateSample::<N> {
            values: add_rows(mean, offset),
            weight: wi,
            weight_cov: wi,
        });
    }
    for offset in root.iter().take(N) {
        samples.push(CoordinateSample::<N> {
            values: sub_rows(mean, offset),
            weight: wi,
            weight_cov: wi,
        });
    }
    if samples
        .iter()
        .flat_map(|sample| sample.values)
        .any(|value| !value.is_finite())
    {
        return Err(SchemaError::InvalidRecordBatch(
            "Sigma-point covariance sampling produced non-finite samples.".to_string(),
        ));
    }
    Ok(samples)
}

fn monte_carlo_samples<const N: usize>(
    mean: &[f64; N],
    covariance: &[f64],
    num_samples: usize,
    seed: u64,
) -> SchemaResult<Vec<CoordinateSample<N>>> {
    let root = symmetric_square_root_scaled_with_tolerance::<N>(
        covariance,
        1.0,
        PsdTolerance::Absolute(MONTE_CARLO_PSD_TOLERANCE),
    )?;
    let mut rng = SplitMix64Normal::new(seed);
    let weight = 1.0 / num_samples as f64;
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let mut z = [0.0; N];
        for value in z.iter_mut().take(N) {
            *value = rng.standard_normal();
        }
        let mut values = *mean;
        for dim in 0..N {
            let mut offset = 0.0;
            for k in 0..N {
                offset += z[k] * root[k][dim];
            }
            values[dim] += offset;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SchemaError::InvalidRecordBatch(
                "Monte Carlo covariance sampling produced non-finite samples.".to_string(),
            ));
        }
        samples.push(CoordinateSample::<N> {
            values,
            weight,
            weight_cov: weight,
        });
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn auto_samples<const N: usize>(
    row_index: usize,
    mean: &[f64; N],
    covariance: &[f64],
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> SchemaResult<Vec<CoordinateSample<N>>> {
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

fn sigma_points_reconstruct_input<const N: usize>(
    mean: &[f64; N],
    covariance: &[f64],
    samples: &[CoordinateSample<N>],
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
    let reconstructed_mean = crate::weighted_mean_flat(&sample_values, &weights, samples.len(), N);
    let reconstructed_covariance = crate::weighted_covariance_flat(
        &reconstructed_mean,
        &sample_values,
        &weights_cov,
        samples.len(),
        N,
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
    source_covariance_dimensions: &[usize],
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

    let source_orbit_rows = nominal
        .diagnostics
        .convergence
        .iter()
        .map(|diagnostic| diagnostic.input_orbit_index + 1)
        .max()
        .unwrap_or(0);
    if source_covariance_dimensions.len() != source_orbit_rows
        || source_covariance_dimensions
            .iter()
            .any(|dimension| !matches!(dimension, 6 | 9))
    {
        return Err(PropagationError::InvalidRequest(
            "source covariance dimensions must contain one 6 or 9 value per nominal orbit row"
                .to_string(),
        ));
    }
    let widest_dimension = source_covariance_dimensions
        .iter()
        .copied()
        .max()
        .unwrap_or(DIM);
    let nongrav_variants = variants.non_gravitational_parameters.as_ref();
    let nominal_nongrav = nominal.orbits.non_gravitational_parameters.as_ref();
    if widest_dimension == 9 && nominal_nongrav.is_none() {
        return Err(PropagationError::InvalidRequest(
            "9D covariance collapse requires nominal non-gravitational parameters".to_string(),
        ));
    }
    let output_rows = nominal_states.len();
    let mut dimensions_by_output_row = vec![DIM; output_rows];
    for diagnostic in &nominal.diagnostics.convergence {
        dimensions_by_output_row[diagnostic.output_row] =
            source_covariance_dimensions[diagnostic.input_orbit_index];
    }
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
        if dimensions_by_output_row[nominal_row] == 9 {
            let parameters = nongrav_variants.ok_or_else(|| {
                PropagationError::InvalidRequest(
                    "9D covariance collapse requires variant non-gravitational parameters"
                        .to_string(),
                )
            })?;
            samples_by_row[nominal_row].extend([
                parameters.a1[convergence.output_row].unwrap_or(0.0),
                parameters.a2[convergence.output_row].unwrap_or(0.0),
                parameters.a3[convergence.output_row].unwrap_or(0.0),
            ]);
        }
        weights_by_row[nominal_row].push(weight);
    }

    let mut covariance_values = vec![f64::NAN; output_rows * widest_dimension * widest_dimension];
    for output_row in 0..output_rows {
        let dimension = dimensions_by_output_row[output_row];
        let sample_count = weights_by_row[output_row].len();
        if sample_count == 0 || samples_by_row[output_row].len() != sample_count * dimension {
            return Err(PropagationError::InvalidRequest(format!(
                "missing covariance samples for nominal output row {output_row}"
            )));
        }
        if collapse_validity[output_row] && nominal.validity.is_valid(output_row) {
            let mut nominal_mean = nominal_states[output_row].to_vec();
            if dimension == 9 {
                let parameters = nominal_nongrav.expect("validated 9D nominal parameters");
                nominal_mean.extend([
                    parameters.a1[output_row].unwrap_or(0.0),
                    parameters.a2[output_row].unwrap_or(0.0),
                    parameters.a3[output_row].unwrap_or(0.0),
                ]);
            }
            let covariance = crate::weighted_covariance_flat(
                &nominal_mean,
                &samples_by_row[output_row],
                &weights_by_row[output_row],
                sample_count,
                dimension,
            );
            if covariance.iter().all(|value| value.is_finite()) {
                let row_start = output_row * widest_dimension * widest_dimension;
                for covariance_row in 0..dimension {
                    let source_start = covariance_row * dimension;
                    let target_start = row_start + covariance_row * widest_dimension;
                    covariance_values[target_start..target_start + dimension]
                        .copy_from_slice(&covariance[source_start..source_start + dimension]);
                }
            } else {
                collapse_validity[output_row] = false;
            }
        } else {
            collapse_validity[output_row] = false;
        }
    }

    let covariance = CovarianceBatch::new(
        output_rows,
        widest_dimension,
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
    let orbits = match nominal.orbits.physical_parameters.clone() {
        Some(parameters) => orbits.with_physical_parameters(parameters)?,
        None => orbits,
    };
    let orbits = match nominal.orbits.non_gravitational_parameters.clone() {
        Some(parameters) => orbits.with_non_gravitational_parameters(parameters)?,
        None => orbits,
    };
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

fn validate_sampling_row<const N: usize>(mean: &[f64; N], covariance: &[f64]) -> SchemaResult<()> {
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

fn add_rows<const N: usize>(left: &[f64; N], right: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for index in 0..N {
        out[index] = left[index] + right[index];
    }
    out
}

fn sub_rows<const N: usize>(left: &[f64; N], right: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for index in 0..N {
        out[index] = left[index] - right[index];
    }
    out
}

/// Eigenvalue floor semantics for PSD classification in covariance sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PsdTolerance {
    /// Fixed eigenvalue floor, preserving the legacy public absolute
    /// `semidef_tol` contract on the Monte Carlo path.
    Absolute(f64),
    /// Floor relative to the largest absolute eigenvalue, so classification
    /// stays meaningful across covariance scales (bead personal-yv7s).
    RelativeToLargestEigenvalue(f64),
}

fn symmetric_square_root_scaled<const N: usize>(
    values_row_major: &[f64],
    scale: f64,
) -> SchemaResult<[[f64; N]; N]> {
    symmetric_square_root_scaled_with_tolerance(
        values_row_major,
        scale,
        PsdTolerance::RelativeToLargestEigenvalue(SIGMA_POINT_PSD_RELATIVE_TOLERANCE),
    )
}

fn symmetric_square_root_scaled_with_tolerance<const N: usize>(
    values_row_major: &[f64],
    scale: f64,
    psd_tolerance: PsdTolerance,
) -> SchemaResult<[[f64; N]; N]> {
    if values_row_major.len() != N * N {
        return Err(SchemaError::InvalidCovarianceShape {
            rows: 1,
            dimension: N,
            values: values_row_major.len(),
        });
    }
    let mut a = [[0.0; N]; N];
    for row in 0..N {
        for col in 0..N {
            let left = values_row_major[row * N + col];
            let right = values_row_major[col * N + row];
            a[row][col] = 0.5 * scale * (left + right);
        }
    }
    // Fail closed on nonfinite input (NaN or infinity, including overflow
    // introduced by `scale`) before iterating: the Jacobi loop would
    // otherwise silently degrade instead of erroring.
    let mut max_abs = 0.0_f64;
    for row in &a {
        for value in row {
            // Check each value before reduction: `f64::max` deliberately
            // ignores a single NaN operand and would otherwise mask it.
            if !value.is_finite() {
                return Err(SchemaError::InvalidRecordBatch(
                    "Covariance matrix must contain finite values for covariance sampling."
                        .to_string(),
                ));
            }
            max_abs = max_abs.max(value.abs());
        }
    }

    let mut vectors = [[0.0; N]; N];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    // Scale-aware convergence: iterate until the largest off-diagonal is at
    // machine precision relative to the matrix magnitude, so the square root
    // reconstructs the input covariance to ~n*eps regardless of scale. A zero
    // matrix converges immediately (0 <= 0). If the rotation budget is
    // exhausted first, fail closed rather than return an inaccurate root.
    let threshold = JACOBI_RELATIVE_TOLERANCE * max_abs;
    let mut rotations = 0;
    loop {
        let (p, q, max_off_diag) = max_off_diagonal(&a);
        if max_off_diag <= threshold {
            break;
        }
        if rotations == JACOBI_MAX_SWEEPS * N * (N - 1) / 2 {
            return Err(SchemaError::InvalidRecordBatch(
                "Covariance eigendecomposition did not converge for covariance sampling."
                    .to_string(),
            ));
        }
        rotate_jacobi(&mut a, &mut vectors, p, q);
        rotations += 1;
    }

    let mut eigen_scale = 0.0_f64;
    for (index, row) in a.iter().enumerate() {
        eigen_scale = eigen_scale.max(row[index].abs());
    }
    let psd_floor = match psd_tolerance {
        PsdTolerance::Absolute(tolerance) => tolerance,
        PsdTolerance::RelativeToLargestEigenvalue(tolerance) => tolerance * eigen_scale,
    };
    let mut roots = [0.0; N];
    for index in 0..N {
        let eigenvalue = a[index][index];
        if eigenvalue < -psd_floor {
            return Err(SchemaError::InvalidRecordBatch(
                "Covariance matrix is not positive semidefinite for covariance sampling."
                    .to_string(),
            ));
        }
        roots[index] = eigenvalue.max(0.0).sqrt();
    }

    let mut sqrt = [[0.0; N]; N];
    for row in 0..N {
        for col in 0..N {
            let mut value = 0.0;
            for k in 0..N {
                value += vectors[row][k] * roots[k] * vectors[col][k];
            }
            sqrt[row][col] = value;
        }
    }
    for row in &sqrt {
        for value in row {
            if !value.is_finite() {
                return Err(SchemaError::InvalidRecordBatch(
                    "Covariance square root is not finite for covariance sampling.".to_string(),
                ));
            }
        }
    }
    Ok(sqrt)
}

fn max_off_diagonal<const N: usize>(a: &[[f64; N]; N]) -> (usize, usize, f64) {
    let mut p = 0;
    let mut q = 1;
    let mut max_value = a[p][q].abs();
    for (row, values) in a.iter().enumerate().take(N) {
        for (col, value) in values.iter().enumerate().take(N).skip(row + 1) {
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

fn rotate_jacobi<const N: usize>(
    a: &mut [[f64; N]; N],
    vectors: &mut [[f64; N]; N],
    p: usize,
    q: usize,
) {
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
    for k in 0..N {
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

    for row in vectors.iter_mut().take(N) {
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

    /// Frozen release-blocker fixture (bead personal-yv7s): the immutable
    /// public Apophis Cartesian orbit at MJD 59215 TDB whose small correlated
    /// covariance (eigenvalues ~1.91e-23..1.918e-17, condition ~1.0e6) the
    /// 0.5.6rc1 sampler failed to reconstruct (relative Frobenius error
    /// ~8.64e-3 versus ~3.64e-9 for legacy scipy `sqrtm`).
    fn apophis_public_scale_fixture() -> ([f64; DIM], Vec<f64>) {
        let mean = [
            -0.4098530841678254,
            0.9621648472038595,
            -0.06096604136465475,
            -0.015075524121655987,
            -0.004107955457204797,
            -0.00013931296759505984,
        ];
        let covariance = vec![
            1.605921382259025e-17,
            6.698340830041905e-18,
            -8.737065006307528e-19,
            -4.5472668722760005e-20,
            1.7763932740552372e-19,
            4.6331011863059437e-20,
            6.698340830042149e-18,
            4.812399377784115e-18,
            2.057968530102772e-18,
            -2.8962987106026045e-21,
            5.580793146667982e-20,
            -9.588370494872105e-21,
            -8.737065006307472e-19,
            2.057968530102776e-18,
            3.1385957942465762e-18,
            2.32327631807789e-20,
            -3.1089620646766e-20,
            -4.6360248829710874e-20,
            -4.5472668722761714e-20,
            -2.8962987106023167e-21,
            2.32327631807789e-20,
            4.84280807790098e-22,
            -7.4990334668123335e-22,
            -7.452802940544717e-22,
            1.77639327405528e-19,
            5.580793146667778e-20,
            -3.10896206467661e-20,
            -7.499033466812239e-22,
            2.2362011488275277e-21,
            1.0249870065302596e-21,
            4.6331011863059196e-20,
            -9.588370494872127e-21,
            -4.636024882971086e-20,
            -7.452802940544722e-22,
            1.0249870065302542e-21,
            1.9365744621176817e-21,
        ];
        (mean, covariance)
    }

    /// Relative Frobenius error between the weighted covariance reconstructed
    /// from a sample cloud and the (symmetrized) input covariance, mirroring
    /// the personal-yv7s evidence computation: weighted mean first, then
    /// weighted covariance about that mean.
    fn relative_frobenius_reconstruction_error(
        covariance: &[f64],
        values_flat: &[f64],
        weights: &[f64],
        weights_cov: &[f64],
    ) -> f64 {
        let n = weights.len();
        let reconstructed_mean = crate::weighted_mean_flat(values_flat, weights, n, DIM);
        let reconstructed =
            crate::weighted_covariance_flat(&reconstructed_mean, values_flat, weights_cov, n, DIM);
        let mut squared_error = 0.0_f64;
        let mut squared_norm = 0.0_f64;
        for row in 0..DIM {
            for col in 0..DIM {
                let expected = 0.5 * (covariance[row * DIM + col] + covariance[col * DIM + row]);
                let delta = reconstructed[row * DIM + col] - expected;
                squared_error += delta * delta;
                squared_norm += expected * expected;
            }
        }
        squared_error.sqrt() / squared_norm.sqrt()
    }

    #[test]
    fn sigma_point_cloud_reconstructs_small_correlated_public_scale_covariance() {
        let (mean, covariance) = apophis_public_scale_fixture();
        let (values, weights, weights_cov) =
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(values.len(), SIGMA_POINT_COUNT * DIM);
        let error =
            relative_frobenius_reconstruction_error(&covariance, &values, &weights, &weights_cov);
        assert!(
            error <= 1.0e-8,
            "sigma-point cloud reconstructs the public-scale covariance with relative \
             Frobenius error {error}, above the 1e-8 release gate (personal-yv7s)"
        );
    }

    #[test]
    fn sigma_point_variants_reconstruct_small_correlated_public_scale_covariance() {
        let (mean, covariance) = apophis_public_scale_fixture();
        let covariance_batch = CovarianceBatch::new(
            1,
            DIM,
            covariance.clone(),
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![mean],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(59_215, 0)]).unwrap()),
            Some(covariance_batch),
        )
        .unwrap();
        let orbits = OrbitBatch::new(
            vec![OrbitId("Apophis".to_string())],
            vec![None],
            coordinates,
        )
        .unwrap();
        let samples = create_sigma_point_orbit_variants(&orbits, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(samples.variants.len(), SIGMA_POINT_COUNT);
        let values = coordinate_rows(&samples.variants.coordinates.values)
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let weights = samples
            .variants
            .weights
            .iter()
            .map(|value| value.unwrap())
            .collect::<Vec<_>>();
        let weights_cov = samples
            .variants
            .weights_cov
            .iter()
            .map(|value| value.unwrap())
            .collect::<Vec<_>>();
        let error =
            relative_frobenius_reconstruction_error(&covariance, &values, &weights, &weights_cov);
        assert!(
            error <= 1.0e-8,
            "variant batch reconstructs the public-scale covariance with relative \
             Frobenius error {error}, above the 1e-8 release gate (personal-yv7s)"
        );
    }

    #[test]
    fn sigma_point_cloud_matches_frozen_legacy_scipy_cloud() {
        // Frozen legacy cloud from the personal-yv7s evidence
        // (full-history-parity/variant-samples/a0.json, SHA-256
        // d01649e5765c062997b921402d1e305eeaf34dd36f4141a3696cefb18550284c over
        // contiguous little-endian float64 coordinate bytes). The symmetric
        // PSD square root is unique, so a converged Jacobi root must land on
        // the same geometry as scipy `sqrtm`; the 0.5.6rc1 sampler deviated by
        // up to 7.12e-11 here.
        let expected: [[f64; DIM]; SIGMA_POINT_COUNT] = [
            [
                -0.4098530841678254,
                0.9621648472038595,
                -0.06096604136465475,
                -0.015075524121655987,
                -0.004107955457204797,
                -0.00013931296759505984,
            ],
            [
                -0.4098530749482144,
                0.9621648504438642,
                -0.060966042283045994,
                -0.015075524150197164,
                -0.004107955348857329,
                -0.0001393129468913005,
            ],
            [
                -0.4098530809278207,
                0.9621648509690769,
                -0.06096603931533489,
                -0.015075524116373654,
                -0.004107955445630225,
                -0.00013931296210079034,
            ],
            [
                -0.40985308508621665,
                0.9621648492531794,
                -0.06096603765227371,
                -0.015075524094707572,
                -0.004107955486503837,
                -0.00013931303874827007,
            ],
            [
                -0.40985308419636657,
                0.9621648472091419,
                -0.060966041337706334,
                -0.015075524090545002,
                -0.004107955464668054,
                -0.0001393129852973872,
            ],
            [
                -0.40985308405947796,
                0.962164847215434,
                -0.06096604139395379,
                -0.015075524129119245,
                -0.004107955439440535,
                -0.0001393129498648764,
            ],
            [
                -0.4098530841471216,
                0.9621648472093538,
                -0.06096604143580796,
                -0.015075524139358315,
                -0.004107955439474614,
                -0.00013931289363491517,
            ],
            [
                -0.4098530933874364,
                0.9621648439638548,
                -0.060966040446263504,
                -0.015075524093114811,
                -0.0041079555655522655,
                -0.0001393129882988192,
            ],
            [
                -0.4098530874078301,
                0.9621648434386422,
                -0.06096604341397461,
                -0.01507552412693832,
                -0.004107955468779369,
                -0.00013931297308932935,
            ],
            [
                -0.40985308324943415,
                0.9621648451545397,
                -0.06096604507703579,
                -0.015075524148604403,
                -0.004107955427905758,
                -0.00013931289644184962,
            ],
            [
                -0.40985308413928423,
                0.9621648471985772,
                -0.060966041391603165,
                -0.015075524152766973,
                -0.004107955449741541,
                -0.0001393129498927325,
            ],
            [
                -0.40985308427617284,
                0.962164847192285,
                -0.06096604133535571,
                -0.01507552411419273,
                -0.00410795547496906,
                -0.0001393129853252433,
            ],
            [
                -0.40985308418852917,
                0.9621648471983653,
                -0.06096604129350154,
                -0.01507552410395366,
                -0.004107955474934981,
                -0.00013931304155520452,
            ],
        ];
        let (mean, covariance) = apophis_public_scale_fixture();
        let (values, _weights, _weights_cov) =
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap();
        for (row, expected_row) in expected.iter().enumerate() {
            for (col, expected_value) in expected_row.iter().enumerate() {
                let actual = values[row * DIM + col];
                assert!(
                    (actual - expected_value).abs() <= 1.0e-12,
                    "sigma point [{row}][{col}] = {actual} deviates from the frozen legacy \
                     scipy cloud value {expected_value} by more than 1e-12"
                );
            }
        }
    }

    #[test]
    fn sigma_point_sampling_rejects_materially_non_psd_public_scale_covariance() {
        // -1e-19 against 1e-17 eigenvalues is materially indefinite (1e-2
        // relative) even though it passed the previous absolute 1e-12 floor.
        let mean = [0.0_f64; DIM];
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM - 1 {
            covariance[index * DIM + index] = 1.0e-17;
        }
        covariance[(DIM - 1) * DIM + (DIM - 1)] = -1.0e-19;
        let error =
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap_err();
        match error {
            SchemaError::InvalidRecordBatch(message) => assert!(
                message.contains("not positive semidefinite"),
                "unexpected message: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn sigma_point_sampling_clamps_float_noise_negative_eigenvalues() {
        // Legacy-equivalent acceptance: negatives within the scale-aware floor
        // (-1e-13 against unit-scale eigenvalues) are clamped to zero, exactly
        // as the previous absolute 1e-12 floor accepted them.
        let mean = [0.0_f64; DIM];
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM - 1 {
            covariance[index * DIM + index] = 1.0;
        }
        covariance[(DIM - 1) * DIM + (DIM - 1)] = -1.0e-13;
        let (values, weights, _weights_cov) =
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(weights.len(), SIGMA_POINT_COUNT);
        assert!(values.iter().all(|value| value.is_finite()));
        // The clamped dimension contributes zero offsets.
        let clamped_offset = values[SIGMA_POINT_COUNT * DIM - 1];
        assert_eq!(clamped_offset, 0.0);
    }

    #[test]
    fn covariance_sampling_fails_closed_for_nonfinite_covariance() {
        let mean = [0.0_f64; DIM];
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM {
            covariance[index * DIM + index] = 1.0;
        }
        covariance[1] = f64::INFINITY;
        covariance[DIM] = f64::INFINITY;
        let error =
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap_err();
        match error {
            SchemaError::InvalidRecordBatch(message) => assert!(
                message.contains("covariance elements are non-finite"),
                "unexpected message: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn square_root_boundary_fails_closed_for_nonfinite_covariance() {
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM {
            covariance[index * DIM + index] = 1.0;
        }
        covariance[0] = f64::NAN;
        let error = symmetric_square_root_scaled::<DIM>(&covariance, 1.0).unwrap_err();
        match error {
            SchemaError::InvalidRecordBatch(message) => assert!(
                message.contains("must contain finite values"),
                "unexpected message: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn direct_covariance_sampling_fails_closed_for_nan_covariance() {
        let mean = [0.0_f64; DIM];
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM {
            covariance[index * DIM + index] = 1.0;
        }
        covariance[0] = f64::NAN;
        let error =
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap_err();
        match error {
            SchemaError::InvalidRecordBatch(message) => assert!(
                message.contains("covariance elements are undefined"),
                "unexpected message: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn direct_covariance_samplers_fail_closed_for_nonfinite_mean() {
        let mut mean = [0.0_f64; DIM];
        mean[0] = f64::INFINITY;
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM {
            covariance[index * DIM + index] = 1.0;
        }
        for error in [
            sample_covariance_sigma_points_flat(&mean, &covariance, 1.0, 0.0, 0.0).unwrap_err(),
            sample_covariance_random_flat(&mean, &covariance, 16, Some(1)).unwrap_err(),
        ] {
            match error {
                SchemaError::InvalidRecordBatch(message) => assert!(
                    message.contains("coordinate dimensions are undefined"),
                    "unexpected message: {message}"
                ),
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn sigma_point_sampling_rejects_nonfinite_computed_weights() {
        let mean = [0.0_f64; DIM];
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM {
            covariance[index * DIM + index] = 1.0;
        }
        // All inputs are finite and denom remains positive/finite, but the
        // reciprocal sigma-point weight overflows. Fail before emitting an
        // infinite-weight cloud.
        let error = sample_covariance_sigma_points_flat(&mean, &covariance, 1.0e-155, 0.0, 0.0)
            .unwrap_err();
        match error {
            SchemaError::InvalidRecordBatch(message) => assert!(
                message.contains("finite weights"),
                "unexpected message: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn monte_carlo_sampling_keeps_legacy_absolute_psd_acceptance() {
        // The Monte Carlo path preserves the legacy public
        // `sample_covariance_random(semidef_tol=1e-15)` absolute contract:
        // -1e-16 is accepted (and clamped), -1e-14 is rejected.
        let mean = [0.0_f64; DIM];
        let mut covariance = vec![0.0_f64; DIM * DIM];
        for index in 0..DIM - 1 {
            covariance[index * DIM + index] = 1.0e-17;
        }
        covariance[(DIM - 1) * DIM + (DIM - 1)] = -1.0e-16;
        sample_covariance_random_flat(&mean, &covariance, 16, Some(1)).unwrap();
        covariance[(DIM - 1) * DIM + (DIM - 1)] = -1.0e-14;
        let error = sample_covariance_random_flat(&mean, &covariance, 16, Some(1)).unwrap_err();
        match error {
            SchemaError::InvalidRecordBatch(message) => assert!(
                message.contains("not positive semidefinite"),
                "unexpected message: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
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
            &samples.source_covariance_dimensions,
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
            &samples.source_covariance_dimensions,
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
