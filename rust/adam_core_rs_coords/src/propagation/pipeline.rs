use super::diagnostics::{
    PropagationConvergence, PropagationConvergenceStatus, PropagationDiagnostics,
    PropagationFailureCode,
};
use super::request::{CovariancePropagation, EpochPolicy, PropagationInput, PropagationRequest};
use super::{PropagationError, PropagationResultValue};
use crate::{
    CoordinateBatch, CovarianceBatch, CovarianceUnits, Epoch, ObjectId, OrbitBatch, OrbitId,
    OrbitVariantBatch, OriginArray, OriginId, TimeArray, Validity, VariantId,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PropagationResult {
    pub orbits: OrbitBatch,
    pub variants: Option<OrbitVariantBatch>,
    pub times: TimeArray,
    pub validity: Validity,
    pub diagnostics: PropagationDiagnostics,
}

#[derive(Debug, Clone, Copy)]
pub struct OrbitRow<'a> {
    pub index: usize,
    pub orbit_id: &'a OrbitId,
    pub object_id: Option<&'a ObjectId>,
    pub variant_id: Option<&'a VariantId>,
    pub weight: Option<f64>,
    pub weight_cov: Option<f64>,
    pub state: [f64; 6],
    pub origin: &'a OriginId,
    pub mu: f64,
    pub time: Epoch,
    pub covariance: Option<&'a [f64]>,
    pub covariance_valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct OrbitBlock {
    pub(super) orbit_index: usize,
    pub(super) time_indices: Vec<usize>,
    pub(super) states: Vec<[f64; 6]>,
    pub(super) covariance: Option<Vec<[f64; 36]>>,
    pub(super) covariance_validity: Option<Vec<bool>>,
    pub(super) validity: Vec<bool>,
    pub(super) messages: Vec<Option<String>>,
    pub(super) backend: Option<String>,
    pub(super) iterations: Vec<Option<usize>>,
    pub(super) failure_codes: Vec<Option<PropagationFailureCode>>,
}

pub(super) fn normalized_chunk_size(chunk_size: Option<usize>, rows: usize) -> usize {
    chunk_size.unwrap_or(1).max(1).min(rows.max(1))
}

pub(super) fn time_indices_for_policy(
    policy: &EpochPolicy,
    orbit_index: usize,
    times_len: usize,
) -> Vec<usize> {
    match policy {
        EpochPolicy::CrossProduct => (0..times_len).collect(),
        EpochPolicy::Pairwise => vec![orbit_index],
        EpochPolicy::PerOrbit { .. } => Vec::new(),
    }
}

pub(super) fn orbit_row<'a>(
    input: PropagationInput<'a>,
    states: &'a [[f64; 6]],
    orbit_times: &'a TimeArray,
    covariance: Option<&'a CovarianceBatch>,
    mus: &[f64],
    orbit_index: usize,
    include_covariance: bool,
) -> OrbitRow<'a> {
    let covariance_values = if include_covariance {
        covariance.map(|covariance| covariance.row_values(orbit_index))
    } else {
        None
    };
    let covariance_valid = covariance
        .as_ref()
        .is_none_or(|covariance| covariance.is_row_valid(orbit_index));
    OrbitRow {
        index: orbit_index,
        orbit_id: &input.orbit_id()[orbit_index],
        object_id: input.object_id()[orbit_index].as_ref(),
        variant_id: input
            .variant_id()
            .and_then(|variant_id| variant_id[orbit_index].as_ref()),
        weight: input.weights().and_then(|weights| weights[orbit_index]),
        weight_cov: input
            .weights_cov()
            .and_then(|weights_cov| weights_cov[orbit_index]),
        state: states[orbit_index],
        origin: &input.coordinates().origins.origins[orbit_index],
        mu: mus[orbit_index],
        time: orbit_times.epochs[orbit_index],
        covariance: covariance_values,
        covariance_valid,
    }
}

pub(super) fn assemble_result(
    request: &PropagationRequest<'_>,
    target_times: &TimeArray,
    blocks: Vec<OrbitBlock>,
) -> PropagationResultValue<PropagationResult> {
    let output_rows = blocks.iter().map(|block| block.states.len()).sum::<usize>();
    let mut orbit_ids = Vec::with_capacity(output_rows);
    let mut object_ids = Vec::with_capacity(output_rows);
    let mut states = Vec::with_capacity(output_rows);
    let mut origins = Vec::with_capacity(output_rows);
    let mut epochs = Vec::with_capacity(output_rows);
    let mut validity = Vec::with_capacity(output_rows);
    let mut diagnostics = Vec::with_capacity(output_rows);
    let mut variant_ids = request
        .input
        .variant_id()
        .map(|_| Vec::with_capacity(output_rows));
    let mut weights = request
        .input
        .weights()
        .map(|_| Vec::with_capacity(output_rows));
    let mut weights_cov = request
        .input
        .weights_cov()
        .map(|_| Vec::with_capacity(output_rows));

    let input_coordinates = request.input.coordinates();
    let input_variant_ids = request.input.variant_id();
    let input_weights = request.input.weights();
    let input_weights_cov = request.input.weights_cov();
    let input_physical_parameters = request.input.physical_parameters();
    let mut source_orbit_indices = Vec::with_capacity(output_rows);

    let output_has_covariance = request.options.covariance == CovariancePropagation::Linearized
        && input_coordinates.covariance.is_some();
    let mut covariance_values = if output_has_covariance {
        Some(Vec::with_capacity(output_rows * 36))
    } else {
        None
    };
    let mut covariance_validity = if output_has_covariance {
        Some(Vec::with_capacity(output_rows))
    } else {
        None
    };

    for block in blocks {
        for row_offset in 0..block.states.len() {
            let output_row = states.len();
            let time_index = block.time_indices[row_offset];
            let row_valid = block.validity[row_offset];
            orbit_ids.push(request.input.orbit_id()[block.orbit_index].clone());
            object_ids.push(request.input.object_id()[block.orbit_index].clone());
            if let Some(values) = &mut variant_ids {
                values.push(
                    input_variant_ids.expect("variant ids are present")[block.orbit_index].clone(),
                );
            }
            if let Some(values) = &mut weights {
                values.push(input_weights.expect("variant weights are present")[block.orbit_index]);
            }
            if let Some(values) = &mut weights_cov {
                values.push(
                    input_weights_cov.expect("variant covariance weights are present")
                        [block.orbit_index],
                );
            }
            states.push(block.states[row_offset]);
            origins.push(input_coordinates.origins.origins[block.orbit_index].clone());
            epochs.push(target_times.epochs[time_index]);
            source_orbit_indices.push(block.orbit_index);
            validity.push(row_valid);
            diagnostics.push(PropagationConvergence {
                output_row,
                input_orbit_index: block.orbit_index,
                input_time_index: time_index,
                status: if row_valid {
                    PropagationConvergenceStatus::Converged
                } else {
                    PropagationConvergenceStatus::Failed
                },
                backend: block.backend.clone(),
                iterations: block.iterations[row_offset],
                failure_code: block.failure_codes[row_offset],
                message: block.messages[row_offset].clone(),
            });
            if let Some(values) = &mut covariance_values {
                let covariance_rows = block.covariance.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "missing covariance rows in propagator output".to_string(),
                    )
                })?;
                values.extend_from_slice(&covariance_rows[row_offset]);
            }
            if let Some(validity_values) = &mut covariance_validity {
                let row_validity = block.covariance_validity.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "missing covariance validity in propagator output".to_string(),
                    )
                })?;
                validity_values.push(row_validity[row_offset]);
            }
        }
    }

    let times = TimeArray::new(target_times.scale, epochs)?;
    let covariance = build_output_covariance(
        input_coordinates.covariance.as_ref(),
        covariance_values,
        covariance_validity,
        output_rows,
    )?;
    let coordinates = CoordinateBatch::cartesian(
        states,
        input_coordinates.frame,
        OriginArray::new(origins),
        Some(times.clone()),
        covariance,
    )?;
    let variants = match (variant_ids, weights, weights_cov) {
        (Some(variant_ids), Some(weights), Some(weights_cov)) => {
            let variants = OrbitVariantBatch::new(
                orbit_ids.clone(),
                object_ids.clone(),
                variant_ids,
                weights,
                weights_cov,
                coordinates.clone(),
            )?;
            Some(match input_physical_parameters {
                Some(physical_parameters) => variants
                    .with_physical_parameters(physical_parameters.take(&source_orbit_indices))?,
                None => variants,
            })
        }
        (None, None, None) => None,
        _ => {
            return Err(PropagationError::InvalidRequest(
                "incomplete variant metadata in propagation output".to_string(),
            ));
        }
    };
    let orbits = OrbitBatch::new(orbit_ids, object_ids, coordinates)?;
    let orbits = match input_physical_parameters {
        Some(physical_parameters) => {
            orbits.with_physical_parameters(physical_parameters.take(&source_orbit_indices))?
        }
        None => orbits,
    };
    Ok(PropagationResult {
        orbits,
        variants,
        times,
        validity: Validity::from_bools(&validity),
        diagnostics: PropagationDiagnostics {
            convergence: diagnostics,
            epoch_order: request.epoch_order.clone(),
        },
    })
}

fn build_output_covariance(
    input: Option<&CovarianceBatch>,
    values: Option<Vec<f64>>,
    row_validity: Option<Vec<bool>>,
    rows: usize,
) -> PropagationResultValue<Option<CovarianceBatch>> {
    let Some(input_covariance) = input else {
        return Ok(None);
    };
    let Some(values) = values else {
        return Ok(None);
    };
    let units = match &input_covariance.units {
        CovarianceUnits::Coordinate(representation) => CovarianceUnits::Coordinate(*representation),
        CovarianceUnits::ObservationAngular2D => CovarianceUnits::ObservationAngular2D,
        CovarianceUnits::Photometry1D => CovarianceUnits::Photometry1D,
        CovarianceUnits::Custom(units) => CovarianceUnits::Custom(units.clone()),
    };
    let covariance = CovarianceBatch::new(rows, 6, values, units)?;
    match row_validity {
        Some(validity) if validity.iter().any(|value| !*value) => Ok(Some(
            covariance.with_row_validity(Validity::from_bools(&validity))?,
        )),
        _ => Ok(Some(covariance)),
    }
}
