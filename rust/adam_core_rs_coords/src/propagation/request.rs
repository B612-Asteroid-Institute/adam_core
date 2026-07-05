use super::{PropagationError, PropagationResultValue};
use crate::{
    CoordinateBatch, ObjectId, OrbitBatch, OrbitId, OrbitVariantBatch, PhysicalParametersBatch,
    TimeArray, VariantId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochPolicy {
    /// Each orbit propagates from its own `coordinates.times` row to every
    /// requested epoch. Output rows are orbit-major and preserve caller epoch
    /// order within each orbit.
    CrossProduct,
    /// Requires `len(orbits) == len(times)`; output has one row per orbit.
    Pairwise,
    /// Reserved for a future compact per-orbit epoch subset representation.
    PerOrbit { indices: Box<[u32]> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovariancePropagation {
    None,
    Linearized,
    Monte { samples: usize, seed: u64 },
    SigmaPoint { alpha: f64, beta: f64, kappa: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropagationOptions {
    pub chunk_size: Option<usize>,
    pub thread_limit: Option<usize>,
    pub epoch_policy: EpochPolicy,
    pub covariance: CovariancePropagation,
}

impl Default for PropagationOptions {
    fn default() -> Self {
        Self {
            chunk_size: None,
            thread_limit: None,
            epoch_policy: EpochPolicy::CrossProduct,
            covariance: CovariancePropagation::Linearized,
        }
    }
}

impl PropagationOptions {
    pub fn validate(&self) -> PropagationResultValue<()> {
        if self.chunk_size == Some(0) {
            return Err(PropagationError::InvalidRequest(
                "chunk_size must be positive when provided".to_string(),
            ));
        }
        if self.thread_limit == Some(0) {
            return Err(PropagationError::InvalidRequest(
                "thread_limit must be positive when provided".to_string(),
            ));
        }
        match self.covariance {
            CovariancePropagation::Monte { samples: 0, .. } => {
                return Err(PropagationError::InvalidRequest(
                    "Monte covariance propagation requires samples > 0".to_string(),
                ));
            }
            CovariancePropagation::SigmaPoint { alpha, beta, kappa } => {
                if !alpha.is_finite() || !beta.is_finite() || !kappa.is_finite() {
                    return Err(PropagationError::InvalidRequest(
                        "sigma-point covariance parameters must be finite".to_string(),
                    ));
                }
                if alpha <= 0.0 {
                    return Err(PropagationError::InvalidRequest(
                        "sigma-point alpha must be positive".to_string(),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Chronological permutation metadata for requested epochs.
///
/// `sorted_to_input[sorted_position] = caller_epoch_index` and
/// `input_to_sorted[caller_epoch_index] = sorted_position`. Propagators may use
/// this to process epochs chronologically while still returning caller-order
/// rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochOrder {
    pub sorted_to_input: Vec<usize>,
    pub input_to_sorted: Vec<usize>,
    pub is_chronological: bool,
}

impl EpochOrder {
    pub fn from_times(times: &TimeArray) -> Self {
        let mut sorted_to_input = (0..times.len()).collect::<Vec<_>>();
        sorted_to_input.sort_by(|&left, &right| {
            times.epochs[left]
                .mjd()
                .partial_cmp(&times.epochs[right].mjd())
                .expect("epoch MJD values are finite")
        });
        let mut input_to_sorted = vec![0; times.len()];
        for (sorted_position, &input_index) in sorted_to_input.iter().enumerate() {
            input_to_sorted[input_index] = sorted_position;
        }
        let is_chronological = sorted_to_input.iter().copied().eq(0..times.len());
        Self {
            sorted_to_input,
            input_to_sorted,
            is_chronological,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropagationInput<'a> {
    Orbits(&'a OrbitBatch),
    Variants(&'a OrbitVariantBatch),
}

impl<'a> PropagationInput<'a> {
    pub fn validate(&self) -> PropagationResultValue<()> {
        match self {
            Self::Orbits(orbits) => orbits.validate()?,
            Self::Variants(variants) => variants.validate()?,
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.coordinates().len()
    }

    pub fn is_empty(&self) -> bool {
        self.coordinates().is_empty()
    }

    pub fn orbit_id(&self) -> &'a [OrbitId] {
        match self {
            Self::Orbits(orbits) => &orbits.orbit_id,
            Self::Variants(variants) => &variants.orbit_id,
        }
    }

    pub fn object_id(&self) -> &'a [Option<ObjectId>] {
        match self {
            Self::Orbits(orbits) => &orbits.object_id,
            Self::Variants(variants) => &variants.object_id,
        }
    }

    pub fn variant_id(&self) -> Option<&'a [Option<VariantId>]> {
        match self {
            Self::Orbits(_) => None,
            Self::Variants(variants) => Some(&variants.variant_id),
        }
    }

    pub fn weights(&self) -> Option<&'a [Option<f64>]> {
        match self {
            Self::Orbits(_) => None,
            Self::Variants(variants) => Some(&variants.weights),
        }
    }

    pub fn weights_cov(&self) -> Option<&'a [Option<f64>]> {
        match self {
            Self::Orbits(_) => None,
            Self::Variants(variants) => Some(&variants.weights_cov),
        }
    }

    pub fn physical_parameters(&self) -> Option<&'a PhysicalParametersBatch> {
        match self {
            Self::Orbits(orbits) => orbits.physical_parameters.as_ref(),
            Self::Variants(variants) => variants.physical_parameters.as_ref(),
        }
    }

    pub fn coordinates(&self) -> &'a CoordinateBatch {
        match self {
            Self::Orbits(orbits) => &orbits.coordinates,
            Self::Variants(variants) => &variants.coordinates,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropagationRequest<'a> {
    pub input: PropagationInput<'a>,
    pub times: &'a TimeArray,
    pub options: PropagationOptions,
    pub epoch_order: EpochOrder,
}

impl<'a> PropagationRequest<'a> {
    pub fn new(
        orbits: &'a OrbitBatch,
        times: &'a TimeArray,
        options: PropagationOptions,
    ) -> PropagationResultValue<Self> {
        Self::new_input(PropagationInput::Orbits(orbits), times, options)
    }

    pub fn new_variants(
        variants: &'a OrbitVariantBatch,
        times: &'a TimeArray,
        options: PropagationOptions,
    ) -> PropagationResultValue<Self> {
        Self::new_input(PropagationInput::Variants(variants), times, options)
    }

    pub fn new_input(
        input: PropagationInput<'a>,
        times: &'a TimeArray,
        options: PropagationOptions,
    ) -> PropagationResultValue<Self> {
        input.validate()?;
        times.validate()?;
        options.validate()?;
        if input.coordinates().times.is_none() {
            return Err(PropagationError::MissingOrbitTimes);
        }
        match &options.epoch_policy {
            EpochPolicy::CrossProduct => {}
            EpochPolicy::Pairwise => {
                if input.len() != times.len() {
                    return Err(PropagationError::InvalidRequest(format!(
                        "pairwise propagation requires len(orbits) == len(times); got {} and {}",
                        input.len(),
                        times.len()
                    )));
                }
            }
            EpochPolicy::PerOrbit { .. } => {
                return Err(PropagationError::InvalidRequest(
                    "PerOrbit epoch policy is reserved for a future compact representation"
                        .to_string(),
                ));
            }
        }
        Ok(Self {
            input,
            times,
            options,
            epoch_order: EpochOrder::from_times(times),
        })
    }

    pub fn output_len(&self) -> usize {
        match self.options.epoch_policy {
            EpochPolicy::CrossProduct => self.input.len() * self.times.len(),
            EpochPolicy::Pairwise => self.input.len(),
            EpochPolicy::PerOrbit { .. } => 0,
        }
    }
}
