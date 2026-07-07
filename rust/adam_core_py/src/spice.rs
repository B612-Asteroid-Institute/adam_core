use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use adam_core_rs_coords::{DataFrame, Epoch, OriginId, TimeArray, TimeScale};
use adam_core_rs_spice::{
    builtin_bodc2n, builtin_bodn2c, global_backend, parse_mpc_obscodes, parse_text_kernel_bindings,
    pck_sxform_matrix, AdamCoreSpiceBackend, GroundObserverSite, NaifFrame, PckError, PckFile,
    SpiceBackendError, SpkError, SpkFile, SpkWriter, SpkWriterError, Type3Record, Type3Segment,
    Type9Segment,
};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn spice_err_to_py(err: SpiceBackendError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

fn lock_err_to_py() -> PyErr {
    PyRuntimeError::new_err("adam-core SPICE backend lock is poisoned")
}

/// Thin Python veneer over the process-global Rust SPICE backend
/// (`adam_core_rs_spice::global_backend`). Kernel state is owned entirely in
/// Rust and shared by every consumer, so multiple `AdamCoreSpiceBackend()`
/// handles (e.g. after a Python `get_backend()` rebuild) all see the same
/// loaded kernels; there is no per-instance kernel state to desync
/// (personal-cmy.23). Only the MPC obscodes cache remains per-handle because
/// its lookups must not share the kernel lock during observer assembly.
#[pyclass(name = "AdamCoreSpiceBackend")]
pub struct PyAdamCoreSpiceBackend {
    obscodes: Mutex<HashMap<String, GroundObserverSite>>,
}

impl PyAdamCoreSpiceBackend {
    fn lock(&self) -> PyResult<MutexGuard<'static, AdamCoreSpiceBackend>> {
        global_backend().lock().map_err(|_| lock_err_to_py())
    }

    fn lock_obscodes(&self) -> PyResult<MutexGuard<'_, HashMap<String, GroundObserverSite>>> {
        self.obscodes.lock().map_err(|_| lock_err_to_py())
    }
}

#[pymethods]
impl PyAdamCoreSpiceBackend {
    #[new]
    fn new() -> Self {
        Self {
            obscodes: Mutex::new(HashMap::new()),
        }
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
        let sites = parse_mpc_obscodes(json).map_err(spice_err_to_py)?;
        let mut cache = self.lock_obscodes()?;
        *cache = sites;
        Ok(cache.len())
    }

    fn mpc_obscodes_loaded(&self) -> PyResult<usize> {
        Ok(self.lock_obscodes()?.len())
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
        let sites = self.lock_obscodes()?;
        if sites.is_empty() {
            return Err(PyValueError::new_err(
                "MPC observatory codes are not loaded; call load_mpc_obscodes first",
            ));
        }
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); unique_codes.len()];
        for (row, &slot) in code_indices.iter().enumerate() {
            let slot = usize::try_from(slot).map_err(|_| {
                PyValueError::new_err("code_indices must be non-negative dictionary slots")
            })?;
            groups
                .get_mut(slot)
                .ok_or_else(|| {
                    PyValueError::new_err("code_indices reference a missing dictionary slot")
                })?
                .push(row);
        }
        let backend = self.lock()?;
        let mut out = vec![f64::NAN; code_indices.len() * 6];
        for (slot, rows) in groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let code = &unique_codes[slot];
            let site = sites.get(code.as_str()).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Observatory code '{code}' is not a supported ground site \
                     (unknown or space-based)"
                ))
            })?;
            let epochs: Vec<Epoch> = rows.iter().map(|&row| times.epochs[row]).collect();
            let group_times = TimeArray::new(TimeScale::Tdb, epochs)
                .map_err(|err| PyValueError::new_err(format!("invalid group times: {err}")))?;
            let coordinates = backend
                .ground_observer_state(site, &group_times, frame, &origin)
                .map_err(spice_err_to_py)?;
            let values = coordinates.values.cartesian().ok_or_else(|| {
                PyRuntimeError::new_err("ground observer states were not Cartesian")
            })?;
            for (group_row, &row) in rows.iter().enumerate() {
                out[row * 6..row * 6 + 6].copy_from_slice(&values[group_row]);
            }
        }
        let arr = ndarray::Array2::from_shape_vec((code_indices.len(), 6), out)
            .map_err(|err| PyValueError::new_err(format!("failed to shape states: {err}")))?;
        Ok(arr.into_pyarray(py))
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

fn spk_writer_err_to_py(err: SpkWriterError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

#[pyclass]
struct NaifSpkWriter {
    inner: SpkWriter,
}

#[pymethods]
impl NaifSpkWriter {
    #[new]
    #[pyo3(signature = (locifn = "adam-core"))]
    fn new(locifn: &str) -> Self {
        Self {
            inner: SpkWriter::new_spk(locifn),
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
        self.inner
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
            .map_err(spk_writer_err_to_py)
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
        self.inner
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
            .map_err(spk_writer_err_to_py)
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(spk_writer_err_to_py)
    }
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
    Ok(())
}
