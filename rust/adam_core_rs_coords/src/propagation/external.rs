//! External native propagators (C ABI, version 1).
//!
//! A propagation backend compiled in another extension module (adam-assist's
//! ASSIST, for example) cannot share Rust trait objects with adam-core: the
//! two are separately compiled. It shares a C function instead. The backend
//! fills an [`ExternalPropagatorV1`] (identity plus a `propagate` function
//! pointer over flat arrays) and hands it to Python in a capsule named
//! [`EXTERNAL_PROPAGATOR_CAPSULE`]; adam-core wraps it as an
//! [`ExternalPropagator`], a [`Propagator`] like any other, so every
//! orbit-determination stage runs natively with the backend's dynamics: no
//! Python call and no GIL per prediction.
//!
//! The contract: `propagate` is called with a [`PropagateRequestV1`] whose
//! input arrays are valid for the call, and writes `rows * 6` states and
//! `rows` validity flags (`rows = n_orbits * n_times` orbit-major, or
//! `n_orbits` when `pairwise`). Output states are in the request's frame and
//! origin, at the target times. It returns 0 on success; any other value is
//! a request failure, with a NUL-terminated message written to `error`
//! (truncated to `error_len`). It must be callable from any thread, and the
//! context must stay alive as long as the capsule.

use std::ffi::{c_char, c_void, CStr, CString};

use super::pipeline::{assemble_result, OrbitBlock};
use super::{
    rescale_for_propagation, BackendIdentity, CovariancePropagation, EpochPolicy, PropagationError,
    PropagationFailureCode, PropagationRequest, PropagationResult, PropagationResultValue,
    Propagator, PropagatorShard, RowOutput,
};
use crate::types::time::TimeScaleProvider;
use crate::types::{Epoch, Frame, TimeScale};

/// Capsule name of an [`ExternalPropagatorV1`].
pub const EXTERNAL_PROPAGATOR_CAPSULE: &str = "adam_core.propagator.v1";
/// The ABI version these definitions describe.
pub const EXTERNAL_PROPAGATOR_ABI_VERSION: u32 = 1;

/// One propagation request across the C boundary. Strings are
/// NUL-terminated: the time scales and frame as adam-core spells them
/// (`"tdb"`, `"ecliptic"`, ...), the origin as its code (`"SUN"`).
#[repr(C)]
pub struct PropagateRequestV1 {
    pub n_orbits: usize,
    /// `n_orbits * 6` Cartesian states (au, au/day).
    pub states: *const f64,
    pub epoch_days: *const i64,
    pub epoch_nanos: *const i64,
    pub orbit_time_scale: *const c_char,
    pub frame: *const c_char,
    pub origin: *const c_char,
    pub n_times: usize,
    pub time_days: *const i64,
    pub time_nanos: *const i64,
    pub time_scale: *const c_char,
    /// 1: `n_orbits == n_times`, orbit `i` to time `i`; 0: every orbit to
    /// every time, orbit-major.
    pub pairwise: u8,
    pub out_states: *mut f64,
    pub out_valid: *mut u8,
    pub error: *mut c_char,
    pub error_len: usize,
}

/// An external propagator: identity and the `propagate` entry point.
#[repr(C)]
pub struct ExternalPropagatorV1 {
    pub abi_version: u32,
    /// Integration time scale (`"tdb"`).
    pub integration_time_scale: *const c_char,
    /// 1 when the backend integrates exactly the Kepler two-body problem.
    pub two_body_dynamics: u8,
    pub name: *const c_char,
    pub version: *const c_char,
    pub context: *const c_void,
    pub propagate:
        unsafe extern "C" fn(context: *const c_void, request: *const PropagateRequestV1) -> i32,
}

fn c_string(pointer: *const c_char, field: &str) -> PropagationResultValue<String> {
    if pointer.is_null() {
        return Err(PropagationError::BackendProtocol(format!(
            "external propagator: {field} is null"
        )));
    }
    // SAFETY: the ABI requires NUL-terminated strings valid for the
    // capsule's lifetime.
    Ok(unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned())
}

/// An [`ExternalPropagatorV1`] as a [`Propagator`]. State-only (no
/// covariance, no non-gravitational parameters); orbits must share one
/// origin.
pub struct ExternalPropagator {
    raw: *const ExternalPropagatorV1,
    identity: BackendIdentity,
    integration_scale: TimeScale,
}

// SAFETY: the ABI requires `propagate` to be callable from any thread and
// the context to outlive the capsule the caller keeps alive.
unsafe impl Send for ExternalPropagator {}
unsafe impl Sync for ExternalPropagator {}

impl ExternalPropagator {
    /// Wrap `raw`.
    ///
    /// # Safety
    /// `raw` must point to a valid [`ExternalPropagatorV1`] that, with its
    /// context, outlives the returned value.
    pub unsafe fn from_raw(raw: *const ExternalPropagatorV1) -> PropagationResultValue<Self> {
        if raw.is_null() {
            return Err(PropagationError::BackendProtocol(
                "external propagator: null pointer".to_string(),
            ));
        }
        let table = &*raw;
        if table.abi_version != EXTERNAL_PROPAGATOR_ABI_VERSION {
            return Err(PropagationError::BackendProtocol(format!(
                "external propagator ABI version {} (adam-core supports {})",
                table.abi_version, EXTERNAL_PROPAGATOR_ABI_VERSION
            )));
        }
        let integration_scale = TimeScale::parse(&c_string(
            table.integration_time_scale,
            "integration_time_scale",
        )?)
        .map_err(|err| PropagationError::BackendProtocol(err.to_string()))?;
        Ok(Self {
            raw,
            identity: BackendIdentity {
                name: c_string(table.name, "name")?,
                version: c_string(table.version, "version")?,
                two_body_dynamics: table.two_body_dynamics != 0,
            },
            integration_scale,
        })
    }
}

/// Never used: [`ExternalPropagator`] propagates whole requests.
pub struct ExternalShard;

impl PropagatorShard for ExternalShard {
    fn propagate_one(
        &mut self,
        _orbit: super::OrbitRow<'_>,
        _times: &[Epoch],
    ) -> PropagationResultValue<RowOutput> {
        Err(PropagationError::InvalidRequest(
            "external propagators propagate whole requests".to_string(),
        ))
    }
}

impl Propagator for ExternalPropagator {
    type Shard = ExternalShard;

    fn backend(&self) -> BackendIdentity {
        self.identity.clone()
    }

    fn integration_time_scale(&self) -> TimeScale {
        self.integration_scale
    }

    fn supports(&self, mode: CovariancePropagation) -> bool {
        mode == CovariancePropagation::None
    }

    fn create_shard(&self) -> Self::Shard {
        ExternalShard
    }

    fn propagate(
        &self,
        request: &PropagationRequest<'_>,
        provider: &dyn TimeScaleProvider,
    ) -> PropagationResultValue<PropagationResult> {
        request.input.validate()?;
        request.times.validate()?;
        request.options.validate()?;
        if !self.supports(request.options.covariance) {
            return Err(PropagationError::UnsupportedCovarianceMode(
                request.options.covariance,
            ));
        }
        if request.input.variant_id().is_some() {
            return Err(PropagationError::InvalidRequest(
                "external propagators take orbits, not variants".to_string(),
            ));
        }
        let coordinates = request.input.coordinates();
        let states = coordinates.values.cartesian().ok_or_else(|| {
            PropagationError::InvalidRequest(
                "external propagators require Cartesian orbit coordinates".to_string(),
            )
        })?;
        let n_orbits = states.len();
        let origins = &coordinates.origins.origins;
        let origin = match origins.first() {
            Some(first) if origins.iter().all(|o| o == first) => first.code(),
            Some(_) => {
                return Err(PropagationError::InvalidRequest(
                    "external propagators require one origin per request".to_string(),
                ))
            }
            None => "SUN".to_string(),
        };
        let input_times = coordinates
            .times
            .as_ref()
            .ok_or(PropagationError::MissingOrbitTimes)?;
        let orbit_times = rescale_for_propagation(
            input_times,
            self.integration_scale,
            provider,
            "orbit coordinate times",
        )?;
        let target_times = rescale_for_propagation(
            request.times,
            self.integration_scale,
            provider,
            "target times",
        )?;
        let pairwise = match request.options.epoch_policy {
            EpochPolicy::CrossProduct => false,
            EpochPolicy::Pairwise => true,
            EpochPolicy::PerOrbit { .. } => {
                return Err(PropagationError::InvalidRequest(
                    "external propagators support cross-product and pairwise epochs".to_string(),
                ))
            }
        };
        let n_times = target_times.len();
        let rows_per_orbit = if pairwise { 1 } else { n_times };
        let rows = n_orbits * rows_per_orbit;
        let flat: Vec<f64> = states.iter().flatten().copied().collect();
        let epoch_days: Vec<i64> = orbit_times.epochs.iter().map(|e| e.days).collect();
        let epoch_nanos: Vec<i64> = orbit_times.epochs.iter().map(|e| e.nanos).collect();
        let time_days: Vec<i64> = target_times.epochs.iter().map(|e| e.days).collect();
        let time_nanos: Vec<i64> = target_times.epochs.iter().map(|e| e.nanos).collect();
        let scale = CString::new(self.integration_scale.as_str()).expect("no NUL");
        let frame = CString::new(coordinates.frame.as_str()).expect("no NUL");
        let origin = CString::new(origin)
            .map_err(|err| PropagationError::InvalidRequest(format!("origin code: {err}")))?;
        let mut out_states = vec![f64::NAN; rows * 6];
        let mut out_valid = vec![0_u8; rows];
        let mut error = vec![0 as c_char; 1024];
        let c_request = PropagateRequestV1 {
            n_orbits,
            states: flat.as_ptr(),
            epoch_days: epoch_days.as_ptr(),
            epoch_nanos: epoch_nanos.as_ptr(),
            orbit_time_scale: scale.as_ptr(),
            frame: frame.as_ptr(),
            origin: origin.as_ptr(),
            n_times,
            time_days: time_days.as_ptr(),
            time_nanos: time_nanos.as_ptr(),
            time_scale: scale.as_ptr(),
            pairwise: u8::from(pairwise),
            out_states: out_states.as_mut_ptr(),
            out_valid: out_valid.as_mut_ptr(),
            error: error.as_mut_ptr(),
            error_len: error.len(),
        };
        // SAFETY: the table outlives `self`; every pointer in the request is
        // valid for the call and sized as the ABI states.
        let status = unsafe { ((*self.raw).propagate)((*self.raw).context, &c_request) };
        if status != 0 {
            *error.last_mut().expect("non-empty") = 0;
            // SAFETY: NUL-terminated by the line above.
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            return Err(PropagationError::Backend(format!(
                "{} {}: {message}",
                self.identity.name, self.identity.version
            )));
        }
        let blocks = (0..n_orbits)
            .map(|orbit| {
                let time_indices: Vec<usize> = if pairwise {
                    vec![orbit]
                } else {
                    (0..n_times).collect()
                };
                let base = orbit * rows_per_orbit;
                let states: Vec<[f64; 6]> = (0..rows_per_orbit)
                    .map(|k| std::array::from_fn(|axis| out_states[(base + k) * 6 + axis]))
                    .collect();
                let validity: Vec<bool> = (0..rows_per_orbit)
                    .map(|k| out_valid[base + k] != 0 && states[k].iter().all(|v| v.is_finite()))
                    .collect();
                let failure_codes = validity
                    .iter()
                    .map(|&ok| (!ok).then_some(PropagationFailureCode::IntegratorFailure))
                    .collect();
                OrbitBlock {
                    orbit_index: orbit,
                    time_indices,
                    states,
                    covariance: None,
                    covariance_validity: None,
                    messages: vec![None; rows_per_orbit],
                    backend: Some(self.identity.name.clone()),
                    iterations: vec![None; rows_per_orbit],
                    failure_codes,
                    validity,
                }
            })
            .collect();
        // Output rows carry the caller's target times (not the rescaled
        // copy), as every adam-core propagator returns them.
        assemble_result(request, request.times, blocks)
    }
}

/// Times reach [`OwnedExternalPropagator`]'s backend already in its
/// integration scale (the caller's [`ExternalPropagator`] rescales them), so
/// no conversion runs; a request that needs one is refused.
struct SameScale;

impl TimeScaleProvider for SameScale {
    fn rescale(
        &self,
        _times: &crate::TimeArray,
        new_scale: TimeScale,
    ) -> crate::types::SchemaResult<crate::TimeArray> {
        Err(crate::types::SchemaError::InvalidRecordBatch(format!(
            "external propagator received times outside {}",
            new_scale.as_str()
        )))
    }
}

/// The context behind an [`OwnedExternalPropagator`]'s table.
pub struct ExternalContext<P> {
    propagator: P,
    scale: CString,
    name: CString,
    version: CString,
}

/// Any [`Propagator`] behind the external ABI: the table (first field, so a
/// pointer to this value is a pointer to the table) and the context it
/// points into. A plugin builds one from its propagator and hands it to
/// Python in a capsule named [`EXTERNAL_PROPAGATOR_CAPSULE`]; dropping it
/// (with the capsule) frees the context.
#[repr(C)]
pub struct OwnedExternalPropagator<P: Propagator> {
    table: ExternalPropagatorV1,
    context: Box<ExternalContext<P>>,
}

// SAFETY: the table points into `context`, owned here; `P` is Sync.
unsafe impl<P: Propagator + Send> Send for OwnedExternalPropagator<P> {}

impl<P: Propagator + Send + 'static> OwnedExternalPropagator<P> {
    pub fn new(propagator: P, name: &str, version: &str) -> Self {
        let identity = propagator.backend();
        let context = Box::new(ExternalContext {
            scale: CString::new(propagator.integration_time_scale().as_str()).expect("no NUL"),
            name: CString::new(name).expect("no NUL in name"),
            version: CString::new(version).expect("no NUL in version"),
            propagator,
        });
        let table = ExternalPropagatorV1 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            integration_time_scale: context.scale.as_ptr(),
            two_body_dynamics: u8::from(identity.two_body_dynamics),
            name: context.name.as_ptr(),
            version: context.version.as_ptr(),
            context: &*context as *const ExternalContext<P> as *const c_void,
            propagate: propagate_with::<P>,
        };
        Self { table, context }
    }

    pub fn table(&self) -> *const ExternalPropagatorV1 {
        &self.table
    }
}

fn write_error(request: &PropagateRequestV1, message: &str) {
    if request.error.is_null() || request.error_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let n = bytes.len().min(request.error_len - 1);
    // SAFETY: `error` has `error_len` writable bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, request.error, n);
        *request.error.add(n) = 0;
    }
}

unsafe extern "C" fn propagate_with<P: Propagator>(
    context: *const c_void,
    request: *const PropagateRequestV1,
) -> i32 {
    let context = &*(context as *const ExternalContext<P>);
    let request = &*request;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_request(&context.propagator, request)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => {
            write_error(request, &message);
            1
        }
        Err(_) => {
            write_error(request, "external propagator panicked");
            2
        }
    }
}

unsafe fn run_request<P: Propagator>(
    propagator: &P,
    request: &PropagateRequestV1,
) -> Result<(), String> {
    use crate::propagation::PropagationOptions;
    use crate::{CoordinateBatch, OrbitBatch, OrbitId, OriginArray, OriginId, TimeArray};
    let n = request.n_orbits;
    let m = request.n_times;
    let flat = std::slice::from_raw_parts(request.states, n * 6);
    let states: Vec<[f64; 6]> = flat
        .chunks_exact(6)
        .map(|c| std::array::from_fn(|i| c[i]))
        .collect();
    let text = |p: *const c_char| CStr::from_ptr(p).to_string_lossy().into_owned();
    let scale = |p: *const c_char| TimeScale::parse(&text(p)).map_err(|e| e.to_string());
    let epochs = |days: *const i64, nanos: *const i64, k: usize| -> Vec<Epoch> {
        let d = std::slice::from_raw_parts(days, k);
        let ns = std::slice::from_raw_parts(nanos, k);
        d.iter().zip(ns).map(|(&a, &b)| Epoch::new(a, b)).collect()
    };
    let frame = Frame::parse(&text(request.frame)).map_err(|e| e.to_string())?;
    let origin = OriginId::from_code(text(request.origin));
    let orbit_times = TimeArray::new(
        scale(request.orbit_time_scale)?,
        epochs(request.epoch_days, request.epoch_nanos, n),
    )
    .map_err(|e| e.to_string())?;
    let orbits = OrbitBatch::new(
        (0..n).map(|i| OrbitId(i.to_string())).collect(),
        vec![None; n],
        CoordinateBatch::cartesian(
            states,
            frame,
            OriginArray::repeat(origin, n),
            Some(orbit_times),
            None,
        )
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let times = TimeArray::new(
        scale(request.time_scale)?,
        epochs(request.time_days, request.time_nanos, m),
    )
    .map_err(|e| e.to_string())?;
    let options = PropagationOptions {
        chunk_size: None,
        thread_limit: None,
        epoch_policy: if request.pairwise != 0 {
            EpochPolicy::Pairwise
        } else {
            EpochPolicy::CrossProduct
        },
        covariance: CovariancePropagation::None,
    };
    let propagation =
        PropagationRequest::new(&orbits, &times, options).map_err(|e| e.to_string())?;
    let result = propagator
        .propagate(&propagation, &SameScale)
        .map_err(|e| e.to_string())?;
    let out = result
        .orbits
        .coordinates
        .values
        .cartesian()
        .ok_or("non-Cartesian output")?;
    let rows = if request.pairwise != 0 { n } else { n * m };
    if out.len() != rows {
        return Err(format!("expected {rows} rows, got {}", out.len()));
    }
    let out_states = std::slice::from_raw_parts_mut(request.out_states, rows * 6);
    let out_valid = std::slice::from_raw_parts_mut(request.out_valid, rows);
    for (row, state) in out.iter().enumerate() {
        out_states[row * 6..row * 6 + 6].copy_from_slice(state);
        out_valid[row] = u8::from(result.validity.is_valid(row));
    }
    Ok(())
}

/// adam-core's two-body propagator behind the external ABI: the reference
/// implementation (and the test double of the external route).
pub mod two_body_reference {
    use super::OwnedExternalPropagator;
    use crate::propagation::TwoBodyPropagator;

    pub fn new(propagator: TwoBodyPropagator) -> OwnedExternalPropagator<TwoBodyPropagator> {
        OwnedExternalPropagator::new(
            propagator,
            "adam_core.two_body.external",
            env!("CARGO_PKG_VERSION"),
        )
    }

    /// A process-lifetime table (tests).
    pub fn table(propagator: TwoBodyPropagator) -> &'static super::ExternalPropagatorV1 {
        let owned: &'static OwnedExternalPropagator<TwoBodyPropagator> =
            Box::leak(Box::new(new(propagator)));
        // SAFETY: `owned` is leaked, so its table lives for the process.
        unsafe { &*owned.table() }
    }
}
