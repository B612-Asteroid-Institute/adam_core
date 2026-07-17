//! Rust-owned SPK product fitting and orchestration.

use crate::{
    et_seconds, AdamCoreSpiceBackend, SpiceBackendError, SpkWriter, SpkWriterError, Type3Record,
    Type3Segment, Type9Segment, SPK_SUMMARIES_PER_RECORD,
};
use adam_core_rs_coords::{DataFrame, OrbitBatch, OriginId, Representation};
use faer::{linalg::solvers::SolveLstsq, Mat};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use thiserror::Error;

const KM_PER_AU: f64 = 149_597_870.7;
const SECONDS_PER_DAY: f64 = 86_400.0;
const CHEBY_DEGREE: usize = 15;

#[derive(Debug, Error)]
pub enum SpkProductError {
    #[error(transparent)]
    Backend(#[from] SpiceBackendError),
    #[error(transparent)]
    Writer(#[from] SpkWriterError),
    #[error("SPK product requires Cartesian coordinates")]
    NonCartesian,
    #[error("SPK product requires coordinate times")]
    MissingTimes,
    #[error("invalid kernel type: {0}")]
    InvalidKernelType(String),
    #[error("Chebyshev least-squares fit failed: {0}")]
    LeastSquares(String),
    #[error("Chebyshev fit window contains no samples")]
    EmptyFitWindow,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpkKernelType {
    Type3,
    Type9,
}

impl SpkKernelType {
    pub fn parse(value: &str) -> Result<Self, SpkProductError> {
        match value {
            "w03" => Ok(Self::Type3),
            "w09" => Ok(Self::Type9),
            other => Err(SpkProductError::InvalidKernelType(other.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpkProductOptions {
    pub target_id_start: i32,
    pub window_seconds: f64,
    pub kernel_type: SpkKernelType,
}

#[derive(Debug, Clone)]
pub struct ChebyshevFit {
    /// Component-major `(6, degree + 1)` coefficients.
    pub coefficients: Vec<f64>,
    pub mid_time: f64,
    pub half_interval: f64,
}

/// NumPy `lstsq`-equivalent SVD fit over selected ET samples. Input states are
/// adam-core units (AU, AU/day); coefficients are SPK units (km, km/s).
pub fn fit_chebyshev(
    states: &[[f64; 6]],
    et_times: &[f64],
    window_start: f64,
    window_end: f64,
    degree: usize,
    mid_time: Option<f64>,
    half_interval: Option<f64>,
) -> Result<ChebyshevFit, SpkProductError> {
    let selected = et_times
        .iter()
        .enumerate()
        .filter_map(|(index, &et)| (et >= window_start && et <= window_end).then_some(index))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(SpkProductError::EmptyFitWindow);
    }
    let mid_time = mid_time.unwrap_or((window_end + window_start) / 2.0);
    let half_interval = half_interval.unwrap_or((window_end - window_start) / 2.0);
    let ncoef = degree + 1;
    let mut design = Mat::<f64>::zeros(selected.len(), ncoef);
    let mut rhs = Mat::<f64>::zeros(selected.len(), 6);
    for (row, &index) in selected.iter().enumerate() {
        let scaled = (et_times[index] - mid_time) / half_interval;
        design[(row, 0)] = 1.0;
        if ncoef > 1 {
            design[(row, 1)] = scaled;
        }
        for col in 2..ncoef {
            design[(row, col)] = 2.0 * scaled * design[(row, col - 1)] - design[(row, col - 2)];
        }
        for component in 0..3 {
            rhs[(row, component)] = states[index][component] * KM_PER_AU;
        }
        for component in 3..6 {
            rhs[(row, component)] = states[index][component] * (KM_PER_AU / SECONDS_PER_DAY);
        }
    }
    // NumPy `lstsq` permits underdetermined windows (legacy Type 3 defaults
    // commonly fit degree 15 from 11 daily samples). Form the SVD
    // pseudoinverse explicitly; faer's `solve_lstsq` intentionally asserts
    // rows >= columns.
    let solution = if design.nrows() >= design.ncols() {
        let svd = design
            .thin_svd()
            .map_err(|err| SpkProductError::LeastSquares(format!("{err:?}")))?;
        svd.solve_lstsq(&rhs)
    } else {
        // Minimum-norm solution for a full-row-rank underdetermined system:
        // x = A^T (A A^T)^+ b. The square Gram SVD avoids faer's current
        // rectangular-m<n SVD assertion while preserving NumPy semantics.
        let gram = &design * design.transpose();
        let svd = gram
            .thin_svd()
            .map_err(|err| SpkProductError::LeastSquares(format!("{err:?}")))?;
        let y = svd.solve_lstsq(&rhs);
        design.transpose() * &y
    };
    let mut coefficients = vec![0.0; 6 * ncoef];
    for component in 0..6 {
        for coefficient in 0..ncoef {
            coefficients[component * ncoef + coefficient] = solution[(coefficient, component)];
        }
    }
    Ok(ChebyshevFit {
        coefficients,
        mid_time,
        half_interval,
    })
}

fn segment_id(orbit_id: &str, start_et: f64, end_et: f64) -> String {
    let value = format!(
        "{}_{}_{}",
        orbit_id,
        adam_core_rs_coords::openspace::py_float_repr(start_et),
        adam_core_rs_coords::openspace::py_float_repr(end_et)
    );
    value.chars().take(40).collect()
}

pub fn type9_segment(
    backend: &AdamCoreSpiceBackend,
    orbit_id: &str,
    states: &[[f64; 6]],
    times: &adam_core_rs_coords::TimeArray,
    origin: &OriginId,
    frame: DataFrame,
    target_id: i32,
) -> Result<Type9Segment, SpkProductError> {
    let epochs = et_seconds(times)?;
    let mut flat = Vec::with_capacity(states.len() * 6);
    for state in states {
        flat.extend(state[..3].iter().map(|value| value * KM_PER_AU));
        flat.extend(
            state[3..]
                .iter()
                .map(|value| value * KM_PER_AU / SECONDS_PER_DAY),
        );
    }
    let start_et = epochs[0];
    let end_et = *epochs.last().expect("non-empty epochs");
    Ok(Type9Segment {
        target: target_id,
        center: backend.resolve_origin_id(origin)?,
        frame_id: match frame {
            DataFrame::Equatorial => 1,
            DataFrame::Ecliptic => 17,
            _ => 1,
        },
        start_et,
        end_et,
        segment_id: segment_id(orbit_id, start_et, end_et),
        degree: 15,
        states: flat,
        epochs,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn type3_segment(
    backend: &AdamCoreSpiceBackend,
    orbit_id: &str,
    states: &[[f64; 6]],
    times: &adam_core_rs_coords::TimeArray,
    origin: &OriginId,
    frame: DataFrame,
    target_id: i32,
    start_et: f64,
    end_et: f64,
    window_seconds: f64,
    degree: usize,
) -> Result<Type3Segment, SpkProductError> {
    let epochs = et_seconds(times)?;
    let num_windows = ((end_et - start_et) / window_seconds).ceil() as usize;
    let mut records = Vec::with_capacity(num_windows);
    let ncoef = degree + 1;
    for window in 0..num_windows {
        let window_start = start_et + window as f64 * window_seconds;
        let window_end = (window_start + window_seconds).min(end_et);
        let record_mid = start_et + (window as f64 + 0.5) * window_seconds;
        let record_half = window_seconds / 2.0;
        let fit = fit_chebyshev(
            states,
            &epochs,
            window_start,
            window_end,
            degree,
            Some(record_mid),
            Some(record_half),
        )?;
        let component =
            |index: usize| fit.coefficients[index * ncoef..(index + 1) * ncoef].to_vec();
        records.push(Type3Record {
            mid: record_mid,
            radius: record_half,
            x: component(0),
            y: component(1),
            z: component(2),
            vx: component(3),
            vy: component(4),
            vz: component(5),
        });
    }
    Ok(Type3Segment {
        target: target_id,
        center: backend.resolve_origin_id(origin)?,
        frame_id: match frame {
            DataFrame::Equatorial => 1,
            DataFrame::Ecliptic => 17,
            _ => 1,
        },
        start_et,
        end_et,
        segment_id: segment_id(orbit_id, start_et, end_et),
        init: start_et,
        intlen: window_seconds,
        records,
    })
}

const DAF_RECORD_BYTES: usize = 1024;
const DAF_DOUBLES_PER_RECORD: usize = 128;
const DAF_SUMMARY_BYTES: usize = 40;

/// Merge independently serialized one-summary-record SPK chunks into one DAF
/// summary/name chain. Segment payload addresses are relocated to the data
/// area after all summary/name pairs. This extends Spicekit 0.2.2's writer
/// beyond its per-record 25-summary limit while retaining its segment encoders.
pub fn write_spk_writers_atomic(
    writers: &[SpkWriter],
    output: &Path,
) -> Result<(), SpkProductError> {
    if writers.len() == 1 {
        writers[0].write(output)?;
        return Ok(());
    }
    let chunks = writers
        .iter()
        .map(SpkWriter::to_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let header_records = 1 + 2 * chunks.len();
    let first_data_address = header_records * DAF_DOUBLES_PER_RECORD + 1;
    let mut summaries = Vec::with_capacity(chunks.len());
    let mut names = Vec::with_capacity(chunks.len());
    let mut payloads = Vec::with_capacity(chunks.len());
    let mut payload_doubles = 0_usize;

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let mut summary = chunk[DAF_RECORD_BYTES..2 * DAF_RECORD_BYTES].to_vec();
        let name = chunk[2 * DAF_RECORD_BYTES..3 * DAF_RECORD_BYTES].to_vec();
        let nsum = f64::from_le_bytes(summary[16..24].try_into().expect("NSUM bytes")) as usize;
        let mut old_max_end = 384_i32;
        let new_base = first_data_address as i32 + payload_doubles as i32;
        let delta = new_base - 385;
        for segment in 0..nsum {
            let integer_offset = 24 + segment * DAF_SUMMARY_BYTES + 16;
            let start_offset = integer_offset + 16;
            let end_offset = integer_offset + 20;
            let old_start = i32::from_le_bytes(
                summary[start_offset..start_offset + 4]
                    .try_into()
                    .expect("start address bytes"),
            );
            let old_end = i32::from_le_bytes(
                summary[end_offset..end_offset + 4]
                    .try_into()
                    .expect("end address bytes"),
            );
            old_max_end = old_max_end.max(old_end);
            summary[start_offset..start_offset + 4]
                .copy_from_slice(&(old_start + delta).to_le_bytes());
            summary[end_offset..end_offset + 4].copy_from_slice(&(old_end + delta).to_le_bytes());
        }
        let payload_len = (old_max_end - 384).max(0) as usize;
        let payload_start = 3 * DAF_RECORD_BYTES;
        let payload_end = payload_start + payload_len * 8;
        payloads.push(chunk[payload_start..payload_end].to_vec());
        payload_doubles += payload_len;

        let summary_record = 2 + 2 * chunk_index;
        let next = if chunk_index + 1 < chunks.len() {
            (summary_record + 2) as f64
        } else {
            0.0
        };
        let previous = if chunk_index == 0 {
            0.0
        } else {
            (summary_record - 2) as f64
        };
        summary[0..8].copy_from_slice(&next.to_le_bytes());
        summary[8..16].copy_from_slice(&previous.to_le_bytes());
        summaries.push(summary);
        names.push(name);
    }

    let data_records = payload_doubles.div_ceil(DAF_DOUBLES_PER_RECORD);
    let mut output_bytes = vec![0_u8; (header_records + data_records) * DAF_RECORD_BYTES];
    output_bytes[..DAF_RECORD_BYTES].copy_from_slice(&chunks[0][..DAF_RECORD_BYTES]);
    output_bytes[76..80].copy_from_slice(&2_u32.to_le_bytes());
    output_bytes[80..84].copy_from_slice(&((2 + 2 * (chunks.len() - 1)) as u32).to_le_bytes());
    output_bytes[84..88]
        .copy_from_slice(&((first_data_address + payload_doubles) as u32).to_le_bytes());
    for index in 0..chunks.len() {
        let summary_offset = (1 + 2 * index) * DAF_RECORD_BYTES;
        output_bytes[summary_offset..summary_offset + DAF_RECORD_BYTES]
            .copy_from_slice(&summaries[index]);
        let name_offset = summary_offset + DAF_RECORD_BYTES;
        output_bytes[name_offset..name_offset + DAF_RECORD_BYTES].copy_from_slice(&names[index]);
    }
    let mut cursor = header_records * DAF_RECORD_BYTES;
    for payload in payloads {
        output_bytes[cursor..cursor + payload.len()].copy_from_slice(&payload);
        cursor += payload.len();
    }

    let temporary = output.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&output_bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(temporary, output)?;
    Ok(())
}

pub fn write_orbits_spk(
    backend: &AdamCoreSpiceBackend,
    orbits: &OrbitBatch,
    output: &Path,
    options: &SpkProductOptions,
) -> Result<Vec<(String, i32)>, SpkProductError> {
    let input_states = orbits
        .coordinates
        .values
        .cartesian()
        .ok_or(SpkProductError::NonCartesian)?;
    let times = orbits
        .coordinates
        .times
        .as_ref()
        .ok_or(SpkProductError::MissingTimes)?;
    let flat = input_states.iter().flatten().copied().collect::<Vec<_>>();
    let zeros = vec![0.0; input_states.len()];
    let transformed = backend
        .transform_coordinates(
            &flat,
            None,
            Representation::Cartesian,
            Representation::Cartesian,
            orbits.coordinates.frame,
            DataFrame::Equatorial,
            &orbits.coordinates.origins,
            Some(&OriginId::from_code("SOLAR_SYSTEM_BARYCENTER")),
            times,
            &zeros,
            &zeros,
            0.0,
            0.0,
            100,
            1e-15,
        )?
        .ok_or(SpkProductError::NonCartesian)?;
    let states = transformed
        .values
        .chunks_exact(6)
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect::<Vec<_>>();
    let all_et = et_seconds(times)?;

    let mut order = Vec::<String>::new();
    let mut groups = HashMap::<String, Vec<usize>>::new();
    for (index, orbit_id) in orbits.orbit_id.iter().enumerate() {
        if !groups.contains_key(&orbit_id.0) {
            order.push(orbit_id.0.clone());
        }
        groups.entry(orbit_id.0.clone()).or_default().push(index);
    }

    let mut writers = vec![SpkWriter::new_spk("adam-core")];
    let mut mappings = Vec::with_capacity(order.len());
    for (group_number, orbit_id) in order.into_iter().enumerate() {
        if group_number > 0 && group_number.is_multiple_of(SPK_SUMMARIES_PER_RECORD) {
            writers.push(SpkWriter::new_spk("adam-core"));
        }
        let writer = writers.last_mut().expect("at least one SPK writer");
        let mut indices = groups.remove(&orbit_id).expect("known orbit group");
        indices.sort_by(|&left, &right| all_et[left].total_cmp(&all_et[right]));
        let group_states = indices
            .iter()
            .map(|&index| states[index])
            .collect::<Vec<_>>();
        let group_epochs = indices
            .iter()
            .map(|&index| times.epochs[index])
            .collect::<Vec<_>>();
        let group_times = adam_core_rs_coords::TimeArray::new(times.scale, group_epochs)
            .map_err(SpiceBackendError::from)?;
        let target_id = options.target_id_start + group_number as i32;
        let start_et = all_et[indices[0]];
        let end_et = all_et[*indices.last().expect("non-empty group")];
        match options.kernel_type {
            SpkKernelType::Type3 => writer.add_type3(type3_segment(
                backend,
                &orbit_id,
                &group_states,
                &group_times,
                &OriginId::from_code("SOLAR_SYSTEM_BARYCENTER"),
                DataFrame::Equatorial,
                target_id,
                start_et,
                end_et,
                options.window_seconds,
                CHEBY_DEGREE,
            )?)?,
            SpkKernelType::Type9 => writer.add_type9(type9_segment(
                backend,
                &orbit_id,
                &group_states,
                &group_times,
                &OriginId::from_code("SOLAR_SYSTEM_BARYCENTER"),
                DataFrame::Equatorial,
                target_id,
            )?)?,
        }
        mappings.push((orbit_id, target_id));
    }
    write_spk_writers_atomic(&writers, output)?;
    Ok(mappings)
}
