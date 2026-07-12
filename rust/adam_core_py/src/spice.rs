use std::collections::HashMap;
use std::hint::black_box;
use std::sync::MutexGuard;
use std::time::Instant;

use adam_core_rs_coords::{
    CoordinateBatch, DataFrame, IntoNestedRecordBatch, ObservatoryCode, ObserverBatch,
    OrbitBatch as DataOrbitBatch, OriginArray, OriginId, TimeArray, TimeScale,
    TryFromNestedRecordBatch,
};
use adam_core_rs_spice::{
    builtin_bodc2n, builtin_bodn2c, global_backend, parse_text_kernel_bindings, pck_sxform_matrix,
    spk_product::{
        fit_chebyshev, type3_segment, type9_segment, write_orbits_spk, write_spk_writers_atomic,
        SpkKernelType, SpkProductOptions,
    },
    AdamCoreSpiceBackend, NaifFrame, PckError, PckFile, SpiceBackendError, SpkError, SpkFile,
    SpkWriter, SpkWriterError, Type3Record, Type3Segment, Type9Segment, SPK_SUMMARIES_PER_RECORD,
};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{Array, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn spice_err_to_py(err: SpiceBackendError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

fn observer_states_err_to_py(err: SpiceBackendError) -> PyErr {
    match err {
        SpiceBackendError::InvalidObserverSite { message, .. }
            if message.contains("is not a valid MPC observatory code") =>
        {
            PyValueError::new_err(message)
        }
        other => spice_err_to_py(other),
    }
}

fn spk_product_err_to_py(err: adam_core_rs_spice::spk_product::SpkProductError) -> PyErr {
    match err {
        adam_core_rs_spice::spk_product::SpkProductError::InvalidKernelType(value) => {
            PyValueError::new_err(format!("Invalid kernel type: {value}"))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Extract required int64 `days`/`nanos` columns from a pyarrow RecordBatch.
fn read_days_nanos(record_batch: &RecordBatch) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let column = |name: &str| -> PyResult<Vec<i64>> {
        let array = record_batch
            .column_by_name(name)
            .ok_or_else(|| PyValueError::new_err(format!("input batch missing '{name}' column")))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyValueError::new_err(format!("'{name}' column must be int64")))?;
        Ok((0..record_batch.num_rows())
            .map(|row| array.value(row))
            .collect())
    };
    Ok((column("days")?, column("nanos")?))
}

/// Map the adam-core frame string to the typed frame plus the SPICE frame
/// name, with the exact legacy error message.
fn adam_frame(frame: &str) -> PyResult<(DataFrame, &'static str)> {
    match frame {
        "ecliptic" => Ok((DataFrame::Ecliptic, "ECLIPJ2000")),
        "equatorial" => Ok((DataFrame::Equatorial, "J2000")),
        "itrf93" => Ok((DataFrame::Itrf93, "ITRF93")),
        _ => Err(PyValueError::new_err(
            "frame should be one of {'equatorial', 'ecliptic', 'itrf93'}",
        )),
    }
}

/// Assemble the fused perturber-state output batch: flat `N * 6` AU states
/// plus the caller's original-scale times and repeated origin code.
fn perturber_states_record_batch(
    flat: &[f64],
    frame: DataFrame,
    origin_code: &str,
    times: TimeArray,
) -> PyResult<RecordBatch> {
    let states: Vec<[f64; 6]> = flat
        .chunks_exact(6)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
        .collect();
    let rows = states.len();
    let coordinates = CoordinateBatch::cartesian(
        states,
        frame,
        OriginArray::repeat(OriginId::from_code(origin_code), rows),
        Some(times),
        None,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    coordinates
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn observer_positions_from_exposures(
    codes: &[String],
    days: &[i64],
    nanos: &[i64],
    duration: &[f64],
    time_scale: &str,
) -> PyResult<Vec<f64>> {
    let rows = codes.len();
    if days.len() != rows || nanos.len() != rows || duration.len() != rows {
        return Err(PyValueError::new_err(
            "observer exposure inputs must have equal lengths",
        ));
    }
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(Schema::new(vec![
            Field::new("code", DataType::LargeUtf8, false),
            Field::new("days", DataType::Int64, false),
            Field::new("nanos", DataType::Int64, false),
            Field::new("duration", DataType::Float64, false),
        ])),
        vec![
            std::sync::Arc::new(LargeStringArray::from(codes.to_vec())),
            std::sync::Arc::new(Int64Array::from(days.to_vec())),
            std::sync::Arc::new(Int64Array::from(nanos.to_vec())),
            std::sync::Arc::new(Float64Array::from(duration.to_vec())),
        ],
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let scale =
        TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let observer_batch = PyAdamCoreSpiceBackend.observer_states_from_codes_record_batch(
        &batch,
        scale,
        DataFrame::Ecliptic,
        OriginId::from_code("SUN"),
    )?;
    let observers = ObserverBatch::try_from_nested_record_batch(&observer_batch)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let states = observers
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| PyValueError::new_err("observer states must be Cartesian"))?;
    Ok(states
        .iter()
        .flat_map(|state| state[..3].iter().copied())
        .collect())
}

fn lock_err_to_py() -> PyErr {
    PyRuntimeError::new_err("adam-core SPICE backend lock is poisoned")
}

/// Thin Python veneer over the process-global Rust SPICE backend
/// (`adam_core_rs_spice::global_backend`). Kernel state is owned entirely in
/// Rust and shared by every consumer, so multiple `AdamCoreSpiceBackend()`
/// handles (e.g. after a Python `get_backend()` rebuild) all see the same
/// loaded kernels and MPC obscodes; there is no per-instance state to desync
/// (personal-cmy.23). Kernel state, custom name bindings, and the MPC
/// observatory table all live in the one global backend.
#[pyclass(name = "AdamCoreSpiceBackend")]
pub struct PyAdamCoreSpiceBackend;

impl PyAdamCoreSpiceBackend {
    fn lock(&self) -> PyResult<MutexGuard<'static, AdamCoreSpiceBackend>> {
        global_backend().lock().map_err(|_| lock_err_to_py())
    }

    /// Pure-Rust implementation behind the Arrow-native observers crossing.
    /// Both the public PyO3 method and the native-Rust benchmark hook call this
    /// function; no Python/PyO3 operation occurs inside it.
    pub(crate) fn observer_states_from_codes_record_batch(
        &self,
        record_batch: &RecordBatch,
        scale: TimeScale,
        frame: DataFrame,
        origin: OriginId,
    ) -> PyResult<RecordBatch> {
        let rows = record_batch.num_rows();
        let code_array = record_batch
            .column_by_name("code")
            .ok_or_else(|| PyValueError::new_err("input batch missing 'code' column"))?
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| PyValueError::new_err("'code' column must be large_utf8"))?;
        let days_array = record_batch
            .column_by_name("days")
            .ok_or_else(|| PyValueError::new_err("input batch missing 'days' column"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyValueError::new_err("'days' column must be int64"))?;
        let nanos_array = record_batch
            .column_by_name("nanos")
            .ok_or_else(|| PyValueError::new_err("input batch missing 'nanos' column"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyValueError::new_err("'nanos' column must be int64"))?;

        // Group rows by observatory code in Rust (replaces the former Python
        // pc.dictionary_encode boundary), preserving row order.
        let mut unique_codes: Vec<String> = Vec::new();
        let mut slot_of: HashMap<&str, usize> = HashMap::new();
        let mut slots: Vec<usize> = Vec::with_capacity(rows);
        let mut codes: Vec<ObservatoryCode> = Vec::with_capacity(rows);
        for row in 0..rows {
            if code_array.is_null(row) {
                return Err(PyValueError::new_err("observer code must not be null"));
            }
            let code = code_array.value(row);
            let slot = *slot_of.entry(code).or_insert_with(|| {
                unique_codes.push(code.to_string());
                unique_codes.len() - 1
            });
            slots.push(slot);
            codes.push(ObservatoryCode(code.to_string()));
        }

        let days: Vec<i64> = (0..rows).map(|row| days_array.value(row)).collect();
        let nanos: Vec<i64> = (0..rows).map(|row| nanos_array.value(row)).collect();
        // Exposure requests provide duration in seconds; midpoint epoch
        // construction is fused into this crossing and preserves Arrow's
        // half-to-even rounding used by Timestamp.add_seconds.
        let mut output_times = TimeArray::from_parts(scale, days, nanos)
            .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?;
        if let Some(duration) = record_batch.column_by_name("duration") {
            let duration = duration
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| PyValueError::new_err("'duration' column must be float64"))?;
            let delta = (0..rows)
                .map(|row| (duration.value(row) * 500_000_000.0).round_ties_even() as i64)
                .collect::<Vec<_>>();
            output_times = output_times
                .add_nanos_checked(&delta, true)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        // Output coordinates keep the caller's input time scale; SPICE lookups
        // use a TDB rescale internally.
        let tdb_times = output_times
            .clone()
            .rescale(TimeScale::Tdb)
            .map_err(|err| {
                PyValueError::new_err(format!("cannot rescale observer times to TDB: {err}"))
            })?;

        let flat = self
            .lock()?
            .observer_states_from_codes(&unique_codes, &slots, &tdb_times, frame, &origin)
            .map_err(observer_states_err_to_py)?;
        let states: Vec<[f64; 6]> = flat
            .chunks_exact(6)
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
            .collect();
        let coordinates = CoordinateBatch::cartesian(
            states,
            frame,
            OriginArray::repeat(origin, rows),
            Some(output_times),
            None,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let observers = ObserverBatch::new(codes, coordinates)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        observers
            .into_nested_record_batch()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

#[pymethods]
impl PyAdamCoreSpiceBackend {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Paths of every kernel currently loaded in the process-global backend.
    fn registered_kernels(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.registered_kernels())
    }

    /// Whether `path` is currently loaded in the process-global backend.
    fn is_registered(&self, path: &str) -> PyResult<bool> {
        Ok(self.lock()?.is_registered(path))
    }

    /// Unload every kernel and clear custom name bindings from the
    /// process-global backend. Primarily for test isolation.
    fn clear(&self) -> PyResult<()> {
        self.lock()?.clear();
        Ok(())
    }

    /// Parse and cache MPC observatory parallax coefficients (the JSON
    /// shipped by the Python `mpc_obscodes` package). Space-based codes with
    /// non-finite geodetics are skipped. Returns the number of ground sites.
    fn load_mpc_obscodes(&self, json: &str) -> PyResult<usize> {
        self.lock()?
            .load_mpc_obscodes(json)
            .map_err(spice_err_to_py)
    }

    fn mpc_obscodes_loaded(&self) -> PyResult<usize> {
        Ok(self.lock()?.mpc_obscodes_loaded())
    }

    /// Single-crossing ``Observers.from_codes`` state generation:
    /// dictionary-encoded per-row MPC observatory codes (``unique_codes`` +
    /// per-row ``code_indices``) and numpy epoch buffers -> ground-observer
    /// states in ``frame`` relative to ``origin_code`` (heliocentric
    /// ecliptic on the public path). Epochs rescale to TDB through the ERFA
    /// time service. Unknown or space-based codes error (the Python boundary
    /// falls back to the legacy per-code assembly). The dictionary-encoded
    /// boundary keeps large-row crossings at numpy-buffer cost instead of
    /// per-row Python string/int conversion cost.
    #[allow(clippy::too_many_arguments)]
    fn observer_states_from_codes<'py>(
        &self,
        py: Python<'py>,
        unique_codes: Vec<String>,
        code_indices: PyReadonlyArray1<'py, i64>,
        time_scale: &str,
        time_days: PyReadonlyArray1<'py, i64>,
        time_nanos: PyReadonlyArray1<'py, i64>,
        frame: &str,
        origin_code: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let code_indices = code_indices.as_slice()?;
        let time_days = time_days.as_slice()?;
        let time_nanos = time_nanos.as_slice()?;
        if code_indices.len() != time_days.len() || code_indices.len() != time_nanos.len() {
            return Err(PyValueError::new_err(
                "codes and times must share one length",
            ));
        }
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let times = TimeArray::from_parts(scale, time_days.to_vec(), time_nanos.to_vec())
            .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?
            .rescale(TimeScale::Tdb)
            .map_err(|err| {
                PyValueError::new_err(format!("cannot rescale observer times to TDB: {err}"))
            })?;
        let frame = match frame {
            "ecliptic" => DataFrame::Ecliptic,
            "equatorial" => DataFrame::Equatorial,
            "itrf93" => DataFrame::Itrf93,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported observer frame: {other}"
                )))
            }
        };
        let origin = OriginId::from_code(origin_code);
        let slots = code_indices
            .iter()
            .map(|&slot| {
                usize::try_from(slot).map_err(|_| {
                    PyValueError::new_err("code_indices must be non-negative dictionary slots")
                })
            })
            .collect::<PyResult<Vec<usize>>>()?;
        let out = self
            .lock()?
            .observer_states_from_codes(&unique_codes, &slots, &times, frame, &origin)
            .map_err(spice_err_to_py)?;
        let arr = ndarray::Array2::from_shape_vec((code_indices.len(), 6), out)
            .map_err(|err| PyValueError::new_err(format!("failed to shape states: {err}")))?;
        Ok(arr.into_pyarray(py))
    }

    /// Arrow-native ``Observers.from_codes``: a single Arrow C Data Interface
    /// crossing. Accepts a pyarrow ``RecordBatch`` with ``code`` (large_utf8),
    /// ``days`` (int64) and ``nanos`` (int64) columns plus the input
    /// ``time_scale``, and returns one pyarrow ``RecordBatch`` for the nested
    /// ``Observers`` quivr table (code + heliocentric-ecliptic Cartesian
    /// coordinates). Grouping by observatory code and epoch dedup happen in
    /// Rust -- no numpy buffers, no Python ``dictionary_encode``, and no
    /// Python-side table rebuild. Unknown / space-based codes error so the
    /// Python boundary can fall back to the legacy per-code assembly.
    fn observer_states_from_codes_arrow<'py>(
        &self,
        py: Python<'py>,
        batch: &Bound<'py, PyAny>,
        time_scale: &str,
    ) -> PyResult<PyObject> {
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.observer_states_from_codes_record_batch(
            &record_batch,
            scale,
            DataFrame::Ecliptic,
            OriginId::from_code("SUN"),
        )?
        .to_pyarrow(py)
    }

    /// Benchmark the underlying observers implementation directly in Rust.
    ///
    /// PyArrow -> Rust conversion and this outer PyO3 invocation happen once,
    /// before timing. Every returned sample is measured by ``Instant`` around
    /// a direct Rust call to ``observer_states_from_codes_record_batch``; no
    /// Python/PyO3 boundary is inside the timed interval.
    fn benchmark_observer_states_from_codes_arrow_rust(
        &self,
        batch: &Bound<'_, PyAny>,
        time_scale: &str,
        reps: usize,
        warmup: usize,
        trials: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        if reps == 0 || trials == 0 {
            return Err(PyValueError::new_err("reps and trials must be >= 1"));
        }
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut sample_trials = Vec::with_capacity(trials);
        for _ in 0..trials {
            for _ in 0..warmup {
                black_box(self.observer_states_from_codes_record_batch(
                    &record_batch,
                    scale,
                    DataFrame::Ecliptic,
                    OriginId::from_code("SUN"),
                )?);
            }
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let started = Instant::now();
                let output = self.observer_states_from_codes_record_batch(
                    &record_batch,
                    scale,
                    DataFrame::Ecliptic,
                    OriginId::from_code("SUN"),
                )?;
                black_box(&output);
                samples.push(started.elapsed().as_secs_f64());
            }
            sample_trials.push(samples);
        }
        Ok(sample_trials)
    }

    fn observer_states_from_exposures_arrow<'py>(
        &self,
        py: Python<'py>,
        batch: &Bound<'py, PyAny>,
        time_scale: &str,
        frame: &str,
        origin_code: &str,
    ) -> PyResult<PyObject> {
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let frame = match frame {
            "ecliptic" => DataFrame::Ecliptic,
            "equatorial" => DataFrame::Equatorial,
            "itrf93" => DataFrame::Itrf93,
            other => return Err(PyValueError::new_err(format!("Unknown frame: {other}"))),
        };
        self.observer_states_from_codes_record_batch(
            &record_batch,
            scale,
            frame,
            OriginId::from_code(origin_code),
        )?
        .to_pyarrow(py)
    }

    #[pyo3(signature = (batch, time_scale, frame, origin_code, reps, trials, warmup_reps=1))]
    #[allow(clippy::too_many_arguments)]
    fn benchmark_observer_states_from_exposures_arrow_rust(
        &self,
        batch: &Bound<'_, PyAny>,
        time_scale: &str,
        frame: &str,
        origin_code: &str,
        reps: usize,
        trials: usize,
        warmup_reps: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        if reps == 0 || trials == 0 {
            return Err(PyValueError::new_err("reps and trials must be >= 1"));
        }
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let frame = match frame {
            "ecliptic" => DataFrame::Ecliptic,
            "equatorial" => DataFrame::Equatorial,
            "itrf93" => DataFrame::Itrf93,
            other => return Err(PyValueError::new_err(format!("Unknown frame: {other}"))),
        };
        let origin = OriginId::from_code(origin_code);
        let mut trials_out = Vec::with_capacity(trials);
        for _ in 0..trials {
            for _ in 0..warmup_reps {
                black_box(self.observer_states_from_codes_record_batch(
                    &record_batch,
                    scale,
                    frame,
                    origin.clone(),
                )?);
            }
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let started = Instant::now();
                black_box(self.observer_states_from_codes_record_batch(
                    &record_batch,
                    scale,
                    frame,
                    origin.clone(),
                )?);
                samples.push(started.elapsed().as_secs_f64());
            }
            trials_out.push(samples);
        }
        Ok(trials_out)
    }

    /// Fused legacy `get_perturber_state` crossing: days/nanos RecordBatch
    /// plus scale in, one nested `CartesianCoordinates` RecordBatch out.
    /// Rust owns the TDB rescale, epoch dedup, bounded forward/reverse state
    /// cache, batched SPK lookup with legacy ET arithmetic, legacy
    /// divide-by-unit conversion, and table assembly.
    #[pyo3(signature = (batch, time_scale, target_id, origin_id, origin_code, frame, cache_maxsize))]
    fn perturber_states_arrow<'py>(
        &self,
        py: Python<'py>,
        batch: &Bound<'py, PyAny>,
        time_scale: &str,
        target_id: i32,
        origin_id: i32,
        origin_code: &str,
        frame: &str,
        cache_maxsize: usize,
    ) -> PyResult<PyObject> {
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let (days, nanos) = read_days_nanos(&record_batch)?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let (frame_enum, frame_spice) = adam_frame(frame)?;
        let times = TimeArray::from_parts(scale, days, nanos)
            .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?;
        let tdb = times
            .clone()
            .rescale(TimeScale::Tdb)
            .map_err(|err| PyValueError::new_err(format!("cannot rescale times to TDB: {err}")))?;
        let tdb_days: Vec<i64> = tdb.epochs.iter().map(|epoch| epoch.days).collect();
        let tdb_nanos: Vec<i64> = tdb.epochs.iter().map(|epoch| epoch.nanos).collect();
        let flat = self
            .lock()?
            .perturber_states_cached(
                target_id,
                origin_id,
                frame_spice,
                &tdb_days,
                &tdb_nanos,
                cache_maxsize,
            )
            .map_err(spice_err_to_py)?;
        perturber_states_record_batch(&flat, frame_enum, origin_code, times)?.to_pyarrow(py)
    }

    #[pyo3(signature = (batch, time_scale, target_id, origin_id, origin_code, frame, cache_maxsize, reps, trials, warmup_reps=1))]
    #[allow(clippy::too_many_arguments)]
    fn benchmark_perturber_states_arrow_rust(
        &self,
        batch: &Bound<'_, PyAny>,
        time_scale: &str,
        target_id: i32,
        origin_id: i32,
        origin_code: &str,
        frame: &str,
        cache_maxsize: usize,
        reps: usize,
        trials: usize,
        warmup_reps: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        if reps == 0 || trials == 0 {
            return Err(PyValueError::new_err("reps and trials must be >= 1"));
        }
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let (days, nanos) = read_days_nanos(&record_batch)?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let (frame_enum, frame_spice) = adam_frame(frame)?;
        let times = TimeArray::from_parts(scale, days, nanos)
            .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?;
        let tdb = times
            .clone()
            .rescale(TimeScale::Tdb)
            .map_err(|err| PyValueError::new_err(format!("cannot rescale times to TDB: {err}")))?;
        let tdb_days: Vec<i64> = tdb.epochs.iter().map(|epoch| epoch.days).collect();
        let tdb_nanos: Vec<i64> = tdb.epochs.iter().map(|epoch| epoch.nanos).collect();
        let run_once = || -> PyResult<RecordBatch> {
            // Semantic result cache cleared before every sample (cache policy).
            let mut backend = self.lock()?;
            backend.clear_spkez_state_cache();
            let flat = backend
                .perturber_states_cached(
                    target_id,
                    origin_id,
                    frame_spice,
                    &tdb_days,
                    &tdb_nanos,
                    cache_maxsize,
                )
                .map_err(spice_err_to_py)?;
            drop(backend);
            perturber_states_record_batch(&flat, frame_enum, origin_code, times.clone())
        };
        let mut sample_trials = Vec::with_capacity(trials);
        for _ in 0..trials {
            for _ in 0..warmup_reps {
                black_box(run_once()?);
            }
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let started = Instant::now();
                black_box(run_once()?);
                samples.push(started.elapsed().as_secs_f64());
            }
            sample_trials.push(samples);
        }
        Ok(sample_trials)
    }

    /// Fused legacy `get_spice_body_state` crossing: the shared
    /// `state_batch` core (legacy ET arithmetic and `/KM *S_P_DAY`
    /// conversion order) plus nested table assembly.
    fn spice_body_states_arrow<'py>(
        &self,
        py: Python<'py>,
        batch: &Bound<'py, PyAny>,
        time_scale: &str,
        body_id: i32,
        origin_code: &str,
        frame: &str,
    ) -> PyResult<PyObject> {
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let (days, nanos) = read_days_nanos(&record_batch)?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let (frame_enum, _frame_spice) = adam_frame(frame)?;
        let times = TimeArray::from_parts(scale, days, nanos)
            .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?;
        let coordinates = self
            .lock()?
            .state_batch(
                &OriginId::Naif(body_id),
                &OriginId::from_code(origin_code),
                frame_enum,
                &times,
            )
            .map_err(spice_err_to_py)?;
        coordinates
            .into_nested_record_batch()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .to_pyarrow(py)
    }

    #[pyo3(signature = (batch, time_scale, body_id, origin_code, frame, reps, trials, warmup_reps=1))]
    #[allow(clippy::too_many_arguments)]
    fn benchmark_spice_body_states_arrow_rust(
        &self,
        batch: &Bound<'_, PyAny>,
        time_scale: &str,
        body_id: i32,
        origin_code: &str,
        frame: &str,
        reps: usize,
        trials: usize,
        warmup_reps: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        if reps == 0 || trials == 0 {
            return Err(PyValueError::new_err("reps and trials must be >= 1"));
        }
        let record_batch = RecordBatch::from_pyarrow_bound(batch)
            .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
        let (days, nanos) = read_days_nanos(&record_batch)?;
        let scale =
            TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let (frame_enum, _frame_spice) = adam_frame(frame)?;
        let times = TimeArray::from_parts(scale, days, nanos)
            .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?;
        let target = OriginId::Naif(body_id);
        let origin = OriginId::from_code(origin_code);
        let run_once = || -> PyResult<RecordBatch> {
            let coordinates = self
                .lock()?
                .state_batch(&target, &origin, frame_enum, &times)
                .map_err(spice_err_to_py)?;
            coordinates
                .into_nested_record_batch()
                .map_err(|err| PyValueError::new_err(err.to_string()))
        };
        let mut sample_trials = Vec::with_capacity(trials);
        for _ in 0..trials {
            for _ in 0..warmup_reps {
                black_box(run_once()?);
            }
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let started = Instant::now();
                black_box(run_once()?);
                samples.push(started.elapsed().as_secs_f64());
            }
            sample_trials.push(samples);
        }
        Ok(sample_trials)
    }

    fn clear_spkez_state_cache(&self) -> PyResult<()> {
        self.lock()?.clear_spkez_state_cache();
        Ok(())
    }

    fn spkez_state_cache_len(&self) -> PyResult<usize> {
        Ok(self.lock()?.spkez_state_cache_len())
    }

    fn furnsh(&self, path: &str) -> PyResult<()> {
        self.lock()?.furnsh(path).map_err(spice_err_to_py)
    }

    fn unload(&self, path: &str) -> PyResult<()> {
        self.lock()?.unload(path);
        Ok(())
    }

    fn spkez(
        &self,
        target: i32,
        et: f64,
        frame: &str,
        observer: i32,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let state = self
            .lock()?
            .spkez(target, et, frame, observer)
            .map_err(spice_err_to_py)?;
        Ok((state[0], state[1], state[2], state[3], state[4], state[5]))
    }

    fn spkez_batch<'py>(
        &self,
        py: Python<'py>,
        target: i32,
        observer: i32,
        frame: &str,
        ets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ets = ets.as_slice()?;
        let out = self
            .lock()?
            .spkez_batch(target, observer, frame, ets)
            .map_err(spice_err_to_py)?;
        let arr = ndarray::Array2::from_shape_vec((ets.len(), 6), out)
            .map_err(|e| PyValueError::new_err(format!("spkez_batch shape error: {e}")))?;
        Ok(arr.into_pyarray(py))
    }

    fn pxform_batch<'py>(
        &self,
        py: Python<'py>,
        frame_from: &str,
        frame_to: &str,
        ets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let ets = ets.as_slice()?;
        let out = self
            .lock()?
            .pxform_batch(frame_from, frame_to, ets)
            .map_err(spice_err_to_py)?;
        let arr = ndarray::Array3::from_shape_vec((ets.len(), 3, 3), out)
            .map_err(|e| PyValueError::new_err(format!("pxform_batch shape error: {e}")))?;
        Ok(arr.into_pyarray(py))
    }

    fn sxform_batch<'py>(
        &self,
        py: Python<'py>,
        frame_from: &str,
        frame_to: &str,
        ets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let ets = ets.as_slice()?;
        let out = self
            .lock()?
            .sxform_batch(frame_from, frame_to, ets)
            .map_err(spice_err_to_py)?;
        let arr = ndarray::Array3::from_shape_vec((ets.len(), 6, 6), out)
            .map_err(|e| PyValueError::new_err(format!("sxform_batch shape error: {e}")))?;
        Ok(arr.into_pyarray(py))
    }

    fn bodn2c(&self, name: &str) -> PyResult<i32> {
        self.lock()?.bodn2c(name).map_err(spice_err_to_py)
    }
}

fn spk_err_to_py(err: SpkError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

fn parse_naif_frame(name: &str) -> PyResult<NaifFrame> {
    match name {
        "J2000" => Ok(NaifFrame::J2000),
        "ECLIPJ2000" => Ok(NaifFrame::EclipJ2000),
        _ => Err(PyValueError::new_err(format!(
            "unsupported NAIF frame: {name} (supported: J2000, ECLIPJ2000)"
        ))),
    }
}

#[pyclass]
struct NaifSpk {
    inner: SpkFile,
}

#[pymethods]
impl NaifSpk {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = SpkFile::open(path).map_err(spk_err_to_py)?;
        Ok(NaifSpk { inner })
    }

    fn state(&self, target: i32, center: i32, et: f64) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let s = self
            .inner
            .state(target, center, et)
            .map_err(spk_err_to_py)?;
        Ok((s[0], s[1], s[2], s[3], s[4], s[5]))
    }

    fn state_batch<'py>(
        &self,
        py: Python<'py>,
        target: i32,
        center: i32,
        ets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ets = ets.as_slice()?;
        let mut out = ndarray::Array2::<f64>::zeros((ets.len(), 6));
        for (i, &et) in ets.iter().enumerate() {
            let s = self
                .inner
                .state(target, center, et)
                .map_err(spk_err_to_py)?;
            for k in 0..6 {
                out[[i, k]] = s[k];
            }
        }
        Ok(out.into_pyarray(py))
    }

    fn state_batch_in_frame<'py>(
        &self,
        py: Python<'py>,
        target: i32,
        center: i32,
        ets: PyReadonlyArray1<'py, f64>,
        frame: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out_frame = parse_naif_frame(frame)?;
        let ets = ets.as_slice()?;
        let mut out = ndarray::Array2::<f64>::zeros((ets.len(), 6));
        for (i, &et) in ets.iter().enumerate() {
            let s = self
                .inner
                .state_in_frame(target, center, et, out_frame)
                .map_err(spk_err_to_py)?;
            for k in 0..6 {
                out[[i, k]] = s[k];
            }
        }
        Ok(out.into_pyarray(py))
    }
}

fn pck_err_to_py(err: PckError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

#[pyclass]
struct NaifPck {
    inner: PckFile,
}

#[pymethods]
impl NaifPck {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = PckFile::open(path).map_err(pck_err_to_py)?;
        Ok(NaifPck { inner })
    }

    fn euler_state(&self, body_frame: i32, et: f64) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let s = self
            .inner
            .euler_state(body_frame, et)
            .map_err(pck_err_to_py)?;
        Ok((s[0], s[1], s[2], s[3], s[4], s[5]))
    }

    fn pxform_batch<'py>(
        &self,
        py: Python<'py>,
        frame_from: &str,
        frame_to: &str,
        ets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let ets = ets.as_slice()?;
        let mut out = ndarray::Array3::<f64>::zeros((ets.len(), 3, 3));
        for (i, &et) in ets.iter().enumerate() {
            let m = pck_sxform_matrix(&self.inner, frame_from, frame_to, et)
                .map_err(spice_err_to_py)?;
            for r in 0..3 {
                for c in 0..3 {
                    out[[i, r, c]] = m[r][c];
                }
            }
        }
        Ok(out.into_pyarray(py))
    }

    fn sxform_batch<'py>(
        &self,
        py: Python<'py>,
        frame_from: &str,
        frame_to: &str,
        ets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let ets = ets.as_slice()?;
        let mut out = ndarray::Array3::<f64>::zeros((ets.len(), 6, 6));
        for (i, &et) in ets.iter().enumerate() {
            let m = pck_sxform_matrix(&self.inner, frame_from, frame_to, et)
                .map_err(spice_err_to_py)?;
            for r in 0..6 {
                for c in 0..6 {
                    out[[i, r, c]] = m[r][c];
                }
            }
        }
        Ok(out.into_pyarray(py))
    }
}

#[pyfunction]
#[pyo3(signature = (coordinates_ipc, window_start, window_end, degree, mid_time=None, half_interval=None))]
fn spk_fit_chebyshev<'py>(
    py: Python<'py>,
    coordinates_ipc: &Bound<'_, PyBytes>,
    window_start: f64,
    window_end: f64,
    degree: usize,
    mid_time: Option<f64>,
    half_interval: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, f64)> {
    let batch = crate::coordinates::read_orbit_ipc(coordinates_ipc.as_bytes())?;
    let coordinates = CoordinateBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode coordinates: {err}")))?;
    let states = coordinates
        .values
        .cartesian()
        .ok_or_else(|| PyValueError::new_err("fit_chebyshev requires Cartesian coordinates"))?;
    let times = coordinates
        .times
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("fit_chebyshev requires coordinate times"))?;
    let et = adam_core_rs_spice::et_seconds(times).map_err(spice_err_to_py)?;
    let fit = fit_chebyshev(
        states,
        &et,
        window_start,
        window_end,
        degree,
        mid_time,
        half_interval,
    )
    .map_err(spk_product_err_to_py)?;
    let coefficients = ndarray::Array2::from_shape_vec((6, degree + 1), fit.coefficients)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok((
        coefficients.into_pyarray(py),
        fit.mid_time,
        fit.half_interval,
    ))
}

fn decode_spk_orbits(bytes: &[u8]) -> PyResult<DataOrbitBatch> {
    let batch = crate::coordinates::read_orbit_ipc(bytes)?;
    DataOrbitBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode orbits: {err}")))
}

#[pyfunction]
#[pyo3(signature = (orbits_ipc, output_file, target_id_start=1_000_000, window_days=32.0, kernel_type="w03"))]
fn spk_write_orbits_product(
    orbits_ipc: &Bound<'_, PyBytes>,
    output_file: &str,
    target_id_start: i32,
    window_days: f64,
    kernel_type: &str,
) -> PyResult<Vec<(String, i32)>> {
    let orbits = decode_spk_orbits(orbits_ipc.as_bytes())?;
    let options = SpkProductOptions {
        target_id_start,
        window_seconds: window_days * 86_400.0,
        kernel_type: SpkKernelType::parse(kernel_type).map_err(spk_product_err_to_py)?,
    };
    let backend = global_backend()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
    write_orbits_spk(
        &backend,
        &orbits,
        std::path::Path::new(output_file),
        &options,
    )
    .map_err(spk_product_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (orbits_ipc, reps, trials, warmup_reps=1, *, output_file, target_id_start=1_000_000, window_days=32.0, kernel_type="w03"))]
#[allow(clippy::too_many_arguments)]
fn benchmark_spk_write_orbits_product(
    orbits_ipc: &Bound<'_, PyBytes>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    output_file: &str,
    target_id_start: i32,
    window_days: f64,
    kernel_type: &str,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let orbits = decode_spk_orbits(orbits_ipc.as_bytes())?;
    let options = SpkProductOptions {
        target_id_start,
        window_seconds: window_days * 86_400.0,
        kernel_type: SpkKernelType::parse(kernel_type).map_err(spk_product_err_to_py)?,
    };
    let run_once = || -> PyResult<()> {
        let mut backend = global_backend()
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
        backend.clear_spkez_state_cache();
        write_orbits_spk(
            &backend,
            &orbits,
            std::path::Path::new(output_file),
            &options,
        )
        .map_err(spk_product_err_to_py)?;
        Ok(())
    };
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once()?);
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run_once()?);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

fn spk_writer_err_to_py(err: SpkWriterError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

#[pyclass]
struct NaifSpkWriter {
    writers: Vec<SpkWriter>,
    segment_count: usize,
}

impl NaifSpkWriter {
    fn active_writer(&mut self) -> &mut SpkWriter {
        if self.segment_count > 0
            && self.segment_count.is_multiple_of(SPK_SUMMARIES_PER_RECORD)
            && self.writers.len() == self.segment_count / SPK_SUMMARIES_PER_RECORD
        {
            self.writers.push(SpkWriter::new_spk("adam-core"));
        }
        self.writers.last_mut().expect("at least one SPK writer")
    }

    fn mark_segment_added(&mut self) {
        self.segment_count += 1;
    }
}

#[pymethods]
impl NaifSpkWriter {
    #[new]
    #[pyo3(signature = (locifn = "adam-core"))]
    fn new(locifn: &str) -> Self {
        Self {
            writers: vec![SpkWriter::new_spk(locifn)],
            segment_count: 0,
        }
    }

    #[pyo3(signature = (target, center, frame_id, start_et, end_et, segment_id, init, intlen, records_coeffs))]
    fn add_type3<'py>(
        &mut self,
        target: i32,
        center: i32,
        frame_id: i32,
        start_et: f64,
        end_et: f64,
        segment_id: &str,
        init: f64,
        intlen: f64,
        records_coeffs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<()> {
        let arr = records_coeffs.as_array();
        let row_len = arr.ncols();
        if row_len < 2 + 6 {
            return Err(PyValueError::new_err(
                "records_coeffs must have >= 2+6 columns",
            ));
        }
        let coef_block = row_len - 2;
        if !coef_block.is_multiple_of(6) {
            return Err(PyValueError::new_err(
                "records_coeffs row length - 2 must be a multiple of 6",
            ));
        }
        let n_coef = coef_block / 6;
        let mut records = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mid = row[0];
            let radius = row[1];
            let slice = |start: usize| -> Vec<f64> {
                row.slice(ndarray::s![start..start + n_coef]).to_vec()
            };
            records.push(Type3Record {
                mid,
                radius,
                x: slice(2),
                y: slice(2 + n_coef),
                z: slice(2 + 2 * n_coef),
                vx: slice(2 + 3 * n_coef),
                vy: slice(2 + 4 * n_coef),
                vz: slice(2 + 5 * n_coef),
            });
        }
        self.active_writer()
            .add_type3(Type3Segment {
                target,
                center,
                frame_id,
                start_et,
                end_et,
                segment_id: segment_id.to_string(),
                init,
                intlen,
                records,
            })
            .map_err(spk_writer_err_to_py)?;
        self.mark_segment_added();
        Ok(())
    }

    #[pyo3(signature = (target, center, frame_id, start_et, end_et, segment_id, degree, states, epochs))]
    fn add_type9<'py>(
        &mut self,
        target: i32,
        center: i32,
        frame_id: i32,
        start_et: f64,
        end_et: f64,
        segment_id: &str,
        degree: i32,
        states: PyReadonlyArray2<'py, f64>,
        epochs: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let states_arr = states.as_array();
        if states_arr.ncols() != 6 {
            return Err(PyValueError::new_err("states must have shape (N, 6)"));
        }
        let epochs_slice = epochs.as_slice()?;
        if states_arr.nrows() != epochs_slice.len() {
            return Err(PyValueError::new_err(
                "states rows must match epochs length",
            ));
        }
        let mut flat = Vec::with_capacity(6 * epochs_slice.len());
        for row in states_arr.rows() {
            flat.extend_from_slice(row.as_slice().unwrap_or(&row.to_vec()));
        }
        self.active_writer()
            .add_type9(Type9Segment {
                target,
                center,
                frame_id,
                start_et,
                end_et,
                segment_id: segment_id.to_string(),
                degree,
                states: flat,
                epochs: epochs_slice.to_vec(),
            })
            .map_err(spk_writer_err_to_py)?;
        self.mark_segment_added();
        Ok(())
    }

    #[pyo3(signature = (orbits_ipc, target_id, start_time, end_time, window_seconds=86400.0, cheby_degree=15))]
    #[allow(clippy::too_many_arguments)]
    fn add_type3_orbits(
        &mut self,
        orbits_ipc: &Bound<'_, PyBytes>,
        target_id: i32,
        start_time: f64,
        end_time: f64,
        window_seconds: f64,
        cheby_degree: usize,
    ) -> PyResult<()> {
        let orbits = decode_spk_orbits(orbits_ipc.as_bytes())?;
        let states =
            orbits.coordinates.values.cartesian().ok_or_else(|| {
                PyValueError::new_err("SPK segment requires Cartesian coordinates")
            })?;
        let times = orbits
            .coordinates
            .times
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SPK segment requires coordinate times"))?;
        let orbit_id = orbits
            .orbit_id
            .first()
            .ok_or_else(|| PyValueError::new_err("SPK segment requires at least one orbit row"))?;
        let origin = orbits
            .coordinates
            .origins
            .origins
            .first()
            .ok_or_else(|| PyValueError::new_err("SPK segment requires an origin"))?;
        let backend = global_backend()
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
        let segment = type3_segment(
            &backend,
            &orbit_id.0,
            states,
            times,
            origin,
            orbits.coordinates.frame,
            target_id,
            start_time,
            end_time,
            window_seconds,
            cheby_degree,
        )
        .map_err(spk_product_err_to_py)?;
        self.active_writer()
            .add_type3(segment)
            .map_err(spk_writer_err_to_py)?;
        self.mark_segment_added();
        Ok(())
    }

    #[pyo3(signature = (orbits_ipc, target_id))]
    fn add_type9_orbits(
        &mut self,
        orbits_ipc: &Bound<'_, PyBytes>,
        target_id: i32,
    ) -> PyResult<()> {
        let orbits = decode_spk_orbits(orbits_ipc.as_bytes())?;
        let states =
            orbits.coordinates.values.cartesian().ok_or_else(|| {
                PyValueError::new_err("SPK segment requires Cartesian coordinates")
            })?;
        let times = orbits
            .coordinates
            .times
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SPK segment requires coordinate times"))?;
        let orbit_id = orbits
            .orbit_id
            .first()
            .ok_or_else(|| PyValueError::new_err("SPK segment requires at least one orbit row"))?;
        let origin = orbits
            .coordinates
            .origins
            .origins
            .first()
            .ok_or_else(|| PyValueError::new_err("SPK segment requires an origin"))?;
        let backend = global_backend()
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
        let segment = type9_segment(
            &backend,
            &orbit_id.0,
            states,
            times,
            origin,
            orbits.coordinates.frame,
            target_id,
        )
        .map_err(spk_product_err_to_py)?;
        self.active_writer()
            .add_type9(segment)
            .map_err(spk_writer_err_to_py)?;
        self.mark_segment_added();
        Ok(())
    }

    fn write(&self, path: &str) -> PyResult<()> {
        write_spk_writers_atomic(&self.writers, std::path::Path::new(path))
            .map_err(spk_product_err_to_py)
    }
}

pub(crate) fn mission_time_grid(
    start_days: &[i64],
    start_nanos: &[i64],
    start_scale: &str,
    end_days: &[i64],
    end_nanos: &[i64],
    end_scale: &str,
    step_size: f64,
) -> PyResult<TimeArray> {
    let start = TimeArray::from_parts(
        TimeScale::parse(start_scale).map_err(|err| PyValueError::new_err(err.to_string()))?,
        start_days.to_vec(),
        start_nanos.to_vec(),
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?
    .rescale(TimeScale::Tdb)
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let end = TimeArray::from_parts(
        TimeScale::parse(end_scale).map_err(|err| PyValueError::new_err(err.to_string()))?,
        end_days.to_vec(),
        end_nanos.to_vec(),
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?
    .rescale(TimeScale::Tdb)
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let start_mjd = *start
        .mjd_values()
        .first()
        .ok_or_else(|| PyValueError::new_err("start_time must contain at least one row"))?;
    let end_mjd = *end
        .mjd_values()
        .first()
        .ok_or_else(|| PyValueError::new_err("end_time must contain at least one row"))?;
    if step_size == 0.0 {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
    }
    if step_size.is_nan() {
        return Err(PyValueError::new_err("arange: cannot compute length"));
    }
    let advances =
        (step_size > 0.0 && start_mjd < end_mjd) || (step_size < 0.0 && start_mjd > end_mjd);
    let count = if !advances {
        0
    } else if step_size.is_infinite() {
        1
    } else {
        ((end_mjd - start_mjd) / step_size).ceil().max(0.0) as usize
    };
    let mjd = (0..count)
        .map(|row| start_mjd + row as f64 * step_size)
        .collect::<Vec<_>>();
    TimeArray::from_mjd(TimeScale::Tdb, &mjd).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn prepare_major_body_orbits(
    backend: &mut AdamCoreSpiceBackend,
    start_days: &[i64],
    start_nanos: &[i64],
    start_scale: &str,
    end_days: &[i64],
    end_nanos: &[i64],
    end_scale: &str,
    step_size: f64,
    target_id: i32,
    origin_id: i32,
    origin_code: &str,
) -> PyResult<RecordBatch> {
    let times = mission_time_grid(
        start_days,
        start_nanos,
        start_scale,
        end_days,
        end_nanos,
        end_scale,
        step_size,
    )?;
    let days = times
        .epochs
        .iter()
        .map(|epoch| epoch.days)
        .collect::<Vec<_>>();
    let nanos = times
        .epochs
        .iter()
        .map(|epoch| epoch.nanos)
        .collect::<Vec<_>>();
    let flat = backend
        .perturber_states_cached(target_id, origin_id, "ECLIPJ2000", &days, &nanos, 10_000)
        .map_err(spice_err_to_py)?;
    perturber_states_record_batch(&flat, DataFrame::Ecliptic, origin_code, times)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prepare_major_body_orbits_arrow<'py>(
    py: Python<'py>,
    start_days: PyReadonlyArray1<'py, i64>,
    start_nanos: PyReadonlyArray1<'py, i64>,
    start_scale: &str,
    end_days: PyReadonlyArray1<'py, i64>,
    end_nanos: PyReadonlyArray1<'py, i64>,
    end_scale: &str,
    step_size: f64,
    target_id: i32,
    origin_id: i32,
    origin_code: &str,
) -> PyResult<PyObject> {
    let mut backend = global_backend()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
    prepare_major_body_orbits(
        &mut backend,
        start_days.as_slice()?,
        start_nanos.as_slice()?,
        start_scale,
        end_days.as_slice()?,
        end_nanos.as_slice()?,
        end_scale,
        step_size,
        target_id,
        origin_id,
        origin_code,
    )?
    .to_pyarrow(py)
    .map_err(|err| PyRuntimeError::new_err(format!("failed to export prepared orbits: {err}")))
}

#[pyfunction]
#[pyo3(signature = (start_days, start_nanos, start_scale, end_days, end_nanos, end_scale, step_size, target_id, origin_id, origin_code, reps, trials, warmup_reps=1))]
#[allow(clippy::too_many_arguments)]
fn benchmark_prepare_major_body_orbits(
    start_days: PyReadonlyArray1<'_, i64>,
    start_nanos: PyReadonlyArray1<'_, i64>,
    start_scale: &str,
    end_days: PyReadonlyArray1<'_, i64>,
    end_nanos: PyReadonlyArray1<'_, i64>,
    end_scale: &str,
    step_size: f64,
    target_id: i32,
    origin_id: i32,
    origin_code: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let start_days = start_days.as_slice()?.to_vec();
    let start_nanos = start_nanos.as_slice()?.to_vec();
    let end_days = end_days.as_slice()?.to_vec();
    let end_nanos = end_nanos.as_slice()?.to_vec();
    let mut output = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            let mut backend = global_backend()
                .lock()
                .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
            backend.clear_spkez_state_cache();
            black_box(prepare_major_body_orbits(
                &mut backend,
                &start_days,
                &start_nanos,
                start_scale,
                &end_days,
                &end_nanos,
                end_scale,
                step_size,
                target_id,
                origin_id,
                origin_code,
            )?);
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mut backend = global_backend()
                .lock()
                .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
            backend.clear_spkez_state_cache();
            let started = Instant::now();
            black_box(prepare_major_body_orbits(
                &mut backend,
                &start_days,
                &start_nanos,
                start_scale,
                &end_days,
                &end_nanos,
                end_scale,
                step_size,
                target_id,
                origin_id,
                origin_code,
            )?);
            samples.push(started.elapsed().as_secs_f64());
        }
        output.push(samples);
    }
    Ok(output)
}

#[pyfunction]
fn naif_bodn2c(name: &str) -> PyResult<i32> {
    builtin_bodn2c(name).map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn naif_bodc2n(code: i32) -> PyResult<String> {
    builtin_bodc2n(code)
        .map(|s| s.to_string())
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn naif_parse_text_kernel_bindings(path: &str) -> PyResult<Vec<(String, i32)>> {
    parse_text_kernel_bindings(path)
        .map(|v| v.into_iter().map(|b| (b.name, b.code)).collect())
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn naif_spk_open(path: &str) -> PyResult<NaifSpk> {
    NaifSpk::new(path)
}

#[pyfunction]
fn naif_pck_open(path: &str) -> PyResult<NaifPck> {
    NaifPck::new(path)
}

#[pyfunction]
#[pyo3(signature = (locifn = "adam-core"))]
fn naif_spk_writer(locifn: &str) -> NaifSpkWriter {
    NaifSpkWriter::new(locifn)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdamCoreSpiceBackend>()?;
    m.add_class::<NaifSpk>()?;
    m.add_class::<NaifPck>()?;
    m.add_class::<NaifSpkWriter>()?;
    m.add_function(wrap_pyfunction!(naif_bodn2c, m)?)?;
    m.add_function(wrap_pyfunction!(naif_bodc2n, m)?)?;
    m.add_function(wrap_pyfunction!(naif_parse_text_kernel_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(naif_spk_open, m)?)?;
    m.add_function(wrap_pyfunction!(naif_pck_open, m)?)?;
    m.add_function(wrap_pyfunction!(naif_spk_writer, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_major_body_orbits_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_prepare_major_body_orbits, m)?)?;
    m.add_function(wrap_pyfunction!(spk_fit_chebyshev, m)?)?;
    m.add_function(wrap_pyfunction!(spk_write_orbits_product, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_spk_write_orbits_product, m)?)?;
    Ok(())
}
