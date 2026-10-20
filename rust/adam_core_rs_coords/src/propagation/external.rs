//! External native propagators (C ABI, version 2).
//!
//! A propagation backend compiled in another extension module (adam-assist's
//! ASSIST, for example) cannot share Rust trait objects with adam-core: the
//! two are separately compiled. It shares a C function instead. The backend
//! fills an [`ExternalPropagatorV2`] (identity plus a `propagate` function
//! pointer over flat arrays) and hands it to Python in a capsule named
//! [`EXTERNAL_PROPAGATOR_CAPSULE`]; adam-core wraps it as an
//! [`ExternalPropagator`], a [`Propagator`] like any other, so every
//! orbit-determination stage runs natively with the backend's dynamics: no
//! Python call and no GIL per prediction.
//!
//! The contract: `propagate` is called with a [`PropagateRequestV2`] whose
//! input arrays are valid for the call, and writes `rows * 6` states and
//! `rows` validity flags (`rows = n_orbits * n_times` orbit-major, or
//! `n_orbits` when `pairwise`). Output states are in the request's frame and
//! origin, at the target times. It returns 0 on success; any other value is
//! a request failure, with a NUL-terminated message written to `error`
//! (truncated to `error_len`). It must be callable from any thread, and the
//! context must stay alive as long as the capsule.

use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::{align_of, size_of};

use super::pipeline::{assemble_result, OrbitBlock};
use super::{
    rescale_for_propagation, BackendIdentity, CovariancePropagation, EpochPolicy, PropagationError,
    PropagationFailureCode, PropagationRequest, PropagationResult, PropagationResultValue,
    Propagator, PropagatorShard, RowOutput,
};
use crate::types::time::TimeScaleProvider;
use crate::types::{Epoch, Frame, TimeScale};

/// Capsule name of an [`ExternalPropagatorV2`].
pub const EXTERNAL_PROPAGATOR_CAPSULE: &str = "adam_core.propagator.v2";
/// The ABI version these definitions describe.
pub const EXTERNAL_PROPAGATOR_ABI_VERSION: u32 = 2;
/// The provider promises its callbacks may be called concurrently.
pub const EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE: u64 = 1 << 0;
/// The optional state-transition-matrix request is implemented.
pub const EXTERNAL_PROPAGATOR_CAP_STM: u64 = 1 << 1;
/// Per-orbit A1/A2/A3 and Marsden-law parameters are accepted by propagate.
pub const EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL: u64 = 1 << 2;
/// Capabilities understood by this ABI revision.
pub const EXTERNAL_PROPAGATOR_KNOWN_CAPABILITIES: u64 = EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE
    | EXTERNAL_PROPAGATOR_CAP_STM
    | EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL;
/// A1, A2, A3, ALN, NK, NM, NN, R0; NaN denotes a null field.
pub const EXTERNAL_NON_GRAVITATIONAL_WIDTH: usize = 8;
/// H_v, H_v_sigma, G, G_sigma, sigma_eff, chi2_red; NaN denotes null.
pub const EXTERNAL_PHYSICAL_PARAMETERS_WIDTH: usize = 6;
/// Reserved pointer-sized words in each v2 structure. They must be zero.
pub const EXTERNAL_PROPAGATOR_RESERVED_WORDS: usize = 4;
/// Callback completed the request; individual failed rows use `out_valid=0`.
pub const EXTERNAL_PROPAGATOR_STATUS_OK: i32 = 0;
/// Provider rejected or failed the request and wrote an error message.
pub const EXTERNAL_PROPAGATOR_STATUS_ERROR: i32 = 1;
/// Provider caught a panic/exception before it crossed the ABI.
pub const EXTERNAL_PROPAGATOR_STATUS_PANIC: i32 = 2;
/// A null context/request or another boundary-level contract violation.
pub const EXTERNAL_PROPAGATOR_STATUS_INVALID: i32 = 3;

/// ABI value for a successful propagation row.
pub const EXTERNAL_PROPAGATION_FAILURE_NONE: u8 = 0;
/// Stable ABI values mirroring [`PropagationFailureCode`].
pub const EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_INPUT_STATE: u8 = 1;
pub const EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_OUTPUT_STATE: u8 = 2;
pub const EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_COVARIANCE: u8 = 3;
pub const EXTERNAL_PROPAGATION_FAILURE_SOLVER_ZERO_DERIVATIVE: u8 = 4;
pub const EXTERNAL_PROPAGATION_FAILURE_SOLVER_MAX_ITERATIONS: u8 = 5;
pub const EXTERNAL_PROPAGATION_FAILURE_INTEGRATOR: u8 = 6;

/// One propagation request across the C boundary. Strings are
/// NUL-terminated: the time scales and frame as adam-core spells them
/// (`"tdb"`, `"ecliptic"`, ...), the origin as its code (`"SUN"`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PropagateRequestV2 {
    pub abi_version: u32,
    /// Bytes supplied by the caller. Providers require the frozen v2 prefix
    /// and ignore a longer append-only suffix.
    pub struct_size: usize,
    pub n_orbits: usize,
    /// `n_orbits * 6` Cartesian states (au, au/day).
    pub states: *const f64,
    /// Null, or `n_orbits * EXTERNAL_PHYSICAL_PARAMETERS_WIDTH` values in
    /// H_v/H_v_sigma/G/G_sigma/sigma_eff/chi2_red order. NaN is null.
    pub physical_parameters: *const f64,
    /// Null, or `n_orbits * EXTERNAL_NON_GRAVITATIONAL_WIDTH` values in
    /// A1/A2/A3/ALN/NK/NM/NN/R0 order. NaN represents a null optional field.
    pub non_gravitational_parameters: *const f64,
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
    /// Optional caller-owned `rows` bytes. Providers write 0 for valid rows
    /// and one stable `EXTERNAL_PROPAGATION_FAILURE_*` value for invalid rows.
    pub out_failure_codes: *mut u8,
    pub error: *mut c_char,
    pub error_len: usize,
    /// Reserved forever. Every advertised complete word must be zero.
    pub reserved: [usize; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
}

/// One state-plus-STM request. Non-gravitational parameters are held fixed
/// while differentiating the six Cartesian state components.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PropagateStmRequestV2 {
    pub abi_version: u32,
    pub struct_size: usize,
    pub state: *const f64,
    pub non_gravitational_parameters: *const f64,
    pub epoch_days: i64,
    pub epoch_nanos: i64,
    pub n_times: usize,
    pub time_days: *const i64,
    pub time_nanos: *const i64,
    pub out_states: *mut f64,
    pub out_stm: *mut f64,
    pub out_valid: *mut u8,
    pub error: *mut c_char,
    pub error_len: usize,
    /// Reserved forever. Every advertised complete word must be zero.
    pub reserved: [usize; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
}

/// An external propagator: identity and the `propagate` entry point.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExternalPropagatorV2 {
    pub abi_version: u32,
    /// Bytes supplied by the provider. Consumers require the frozen v2
    /// prefix and ignore a longer append-only suffix.
    pub struct_size: usize,
    /// Bitwise `EXTERNAL_PROPAGATOR_CAP_*` declarations.
    pub capabilities: u64,
    /// Integration time scale (`"tdb"`).
    pub integration_time_scale: *const c_char,
    /// 1 when the backend integrates exactly the Kepler two-body problem.
    pub two_body_dynamics: u8,
    pub name: *const c_char,
    pub version: *const c_char,
    pub context: *const c_void,
    pub propagate: Option<PropagateV2>,
    /// Optional (null when absent): one heliocentric ecliptic state at an
    /// epoch (TDB days, nanoseconds) propagated to `n_times` TDB times with
    /// state-transition matrices. Writes `n_times * 6` states, `n_times *
    /// 36` row-major STMs and `n_times` validity flags; returns 0 on
    /// success, else writes `error` like `propagate`.
    pub propagate_stm: Option<PropagateStmV2>,
    /// Reserved forever. Every advertised complete word must be zero.
    pub reserved: [usize; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
}

/// The two fields that may be read before any v2 structure size is known.
#[repr(C)]
#[derive(Clone, Copy)]
struct AbiHeaderV2 {
    abi_version: u32,
    struct_size: usize,
}

/// Required table fields. Reading this copy is sound once the header reports
/// at least `EXTERNAL_PROPAGATOR_V2_MIN_SIZE` bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExternalPropagatorRequiredV2 {
    abi_version: u32,
    struct_size: usize,
    capabilities: u64,
    integration_time_scale: *const c_char,
    two_body_dynamics: u8,
    name: *const c_char,
    version: *const c_char,
    context: *const c_void,
    propagate: Option<PropagateV2>,
}

/// Required table fields plus the capability-gated STM slot.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExternalPropagatorStmV2 {
    required: ExternalPropagatorRequiredV2,
    propagate_stm: Option<PropagateStmV2>,
}

/// Required propagation request fields, copied only after header validation.
#[repr(C)]
#[derive(Clone, Copy)]
struct PropagateRequestPrefixV2 {
    abi_version: u32,
    struct_size: usize,
    n_orbits: usize,
    states: *const f64,
    physical_parameters: *const f64,
    non_gravitational_parameters: *const f64,
    epoch_days: *const i64,
    epoch_nanos: *const i64,
    orbit_time_scale: *const c_char,
    frame: *const c_char,
    origin: *const c_char,
    n_times: usize,
    time_days: *const i64,
    time_nanos: *const i64,
    time_scale: *const c_char,
    pairwise: u8,
    out_states: *mut f64,
    out_valid: *mut u8,
    out_failure_codes: *mut u8,
    error: *mut c_char,
    error_len: usize,
}

/// Required STM request fields, copied only after header validation.
#[repr(C)]
#[derive(Clone, Copy)]
struct PropagateStmRequestPrefixV2 {
    abi_version: u32,
    struct_size: usize,
    state: *const f64,
    non_gravitational_parameters: *const f64,
    epoch_days: i64,
    epoch_nanos: i64,
    n_times: usize,
    time_days: *const i64,
    time_nanos: *const i64,
    out_states: *mut f64,
    out_stm: *mut f64,
    out_valid: *mut u8,
    error: *mut c_char,
    error_len: usize,
}

/// Required v2 table prefix, through the mandatory propagation callback.
/// The field-end expression is target-layout-aware and excludes any unknown
/// append-only suffix.
pub const EXTERNAL_PROPAGATOR_V2_MIN_SIZE: usize =
    std::mem::offset_of!(ExternalPropagatorV2, propagate) + size_of::<Option<PropagateV2>>();
/// Table prefix required when the STM capability is declared.
pub const EXTERNAL_PROPAGATOR_V2_STM_SIZE: usize =
    std::mem::offset_of!(ExternalPropagatorV2, propagate_stm) + size_of::<Option<PropagateStmV2>>();
/// Required v2 propagation request prefix. Reserved words are not required.
pub const PROPAGATE_REQUEST_V2_MIN_SIZE: usize =
    std::mem::offset_of!(PropagateRequestV2, error_len) + size_of::<usize>();
/// Required v2 STM request prefix. Reserved words are not required.
pub const PROPAGATE_STM_REQUEST_V2_MIN_SIZE: usize =
    std::mem::offset_of!(PropagateStmRequestV2, error_len) + size_of::<usize>();

// Every exact value-copy prefix must end at its last required field: consumers
// never read target-specific tail padding beyond the advertised byte count.
const _: () = {
    assert!(size_of::<ExternalPropagatorRequiredV2>() == EXTERNAL_PROPAGATOR_V2_MIN_SIZE);
    assert!(size_of::<ExternalPropagatorStmV2>() == EXTERNAL_PROPAGATOR_V2_STM_SIZE);
    assert!(size_of::<PropagateRequestPrefixV2>() == PROPAGATE_REQUEST_V2_MIN_SIZE);
    assert!(size_of::<PropagateStmRequestPrefixV2>() == PROPAGATE_STM_REQUEST_V2_MIN_SIZE);
    assert!(
        std::mem::offset_of!(ExternalPropagatorV2, reserved) == EXTERNAL_PROPAGATOR_V2_STM_SIZE
    );
    assert!(std::mem::offset_of!(PropagateRequestV2, reserved) == PROPAGATE_REQUEST_V2_MIN_SIZE);
    assert!(
        std::mem::offset_of!(PropagateStmRequestV2, reserved) == PROPAGATE_STM_REQUEST_V2_MIN_SIZE
    );
};

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(EXTERNAL_PROPAGATOR_V2_MIN_SIZE == 72);
    assert!(EXTERNAL_PROPAGATOR_V2_STM_SIZE == 80);
    assert!(PROPAGATE_REQUEST_V2_MIN_SIZE == 168);
    assert!(PROPAGATE_STM_REQUEST_V2_MIN_SIZE == 112);
};

/// Signature of [`ExternalPropagatorV2::propagate`].
pub type PropagateV2 =
    unsafe extern "C" fn(context: *const c_void, request: *const PropagateRequestV2) -> i32;
/// Signature of [`ExternalPropagatorV2::propagate_stm`].
pub type PropagateStmV2 =
    unsafe extern "C" fn(context: *const c_void, request: *const PropagateStmRequestV2) -> i32;

fn pointer_is_aligned<T>(pointer: *const T) -> bool {
    (pointer as usize).is_multiple_of(align_of::<T>())
}

unsafe fn read_header(pointer: *const c_void, label: &str) -> PropagationResultValue<AbiHeaderV2> {
    if pointer.is_null() {
        return Err(PropagationError::BackendProtocol(format!(
            "external propagator: null {label} pointer"
        )));
    }
    if !pointer_is_aligned(pointer.cast::<AbiHeaderV2>()) {
        return Err(PropagationError::BackendProtocol(format!(
            "external propagator: misaligned {label} pointer"
        )));
    }
    // SAFETY: the ABI's minimum allocation is its two-field header. Reading a
    // value copy creates no reference to a larger advertised structure.
    Ok(std::ptr::read(pointer.cast::<AbiHeaderV2>()))
}

unsafe fn validate_reserved_words(
    pointer: *const c_void,
    struct_size: usize,
    reserved_offset: usize,
    label: &str,
) -> PropagationResultValue<()> {
    let available = struct_size.saturating_sub(reserved_offset) / size_of::<usize>();
    for index in 0..available.min(EXTERNAL_PROPAGATOR_RESERVED_WORDS) {
        let offset = reserved_offset + index * size_of::<usize>();
        // Prefix and reserved offsets are pointer-aligned by the repr(C)
        // definitions, and the base alignment was validated before parsing.
        let value = std::ptr::read((pointer.cast::<u8>().add(offset)).cast::<usize>());
        if value != 0 {
            return Err(PropagationError::BackendProtocol(format!(
                "external propagator: {label} reserved word {index} must be zero"
            )));
        }
    }
    Ok(())
}

fn encode_optional_parameter(
    value: Option<f64>,
    batch: &str,
    field: &str,
    row: usize,
) -> PropagationResultValue<f64> {
    match value {
        Some(value) if !value.is_finite() => Err(PropagationError::InvalidRequest(format!(
            "external propagator {batch}.{field}[{row}] is present but non-finite"
        ))),
        Some(value) => Ok(value),
        None => Ok(f64::NAN),
    }
}

fn failure_code_to_abi(code: Option<PropagationFailureCode>) -> u8 {
    match code {
        None => EXTERNAL_PROPAGATION_FAILURE_NONE,
        Some(PropagationFailureCode::NonFiniteInputState) => {
            EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_INPUT_STATE
        }
        Some(PropagationFailureCode::NonFiniteOutputState) => {
            EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_OUTPUT_STATE
        }
        Some(PropagationFailureCode::NonFiniteCovariance) => {
            EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_COVARIANCE
        }
        Some(PropagationFailureCode::SolverZeroDerivative) => {
            EXTERNAL_PROPAGATION_FAILURE_SOLVER_ZERO_DERIVATIVE
        }
        Some(PropagationFailureCode::SolverMaxIterations) => {
            EXTERNAL_PROPAGATION_FAILURE_SOLVER_MAX_ITERATIONS
        }
        Some(PropagationFailureCode::IntegratorFailure) => EXTERNAL_PROPAGATION_FAILURE_INTEGRATOR,
    }
}

fn failure_code_from_abi(code: u8) -> PropagationResultValue<Option<PropagationFailureCode>> {
    match code {
        EXTERNAL_PROPAGATION_FAILURE_NONE => Ok(None),
        EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_INPUT_STATE => {
            Ok(Some(PropagationFailureCode::NonFiniteInputState))
        }
        EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_OUTPUT_STATE => {
            Ok(Some(PropagationFailureCode::NonFiniteOutputState))
        }
        EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_COVARIANCE => {
            Ok(Some(PropagationFailureCode::NonFiniteCovariance))
        }
        EXTERNAL_PROPAGATION_FAILURE_SOLVER_ZERO_DERIVATIVE => {
            Ok(Some(PropagationFailureCode::SolverZeroDerivative))
        }
        EXTERNAL_PROPAGATION_FAILURE_SOLVER_MAX_ITERATIONS => {
            Ok(Some(PropagationFailureCode::SolverMaxIterations))
        }
        EXTERNAL_PROPAGATION_FAILURE_INTEGRATOR => {
            Ok(Some(PropagationFailureCode::IntegratorFailure))
        }
        other => Err(PropagationError::BackendProtocol(format!(
            "external propagator returned unknown row failure code {other}"
        ))),
    }
}

fn decode_row_result(
    row: usize,
    valid_flag: u8,
    state_is_finite: bool,
    failure_code: u8,
) -> PropagationResultValue<(bool, Option<PropagationFailureCode>)> {
    let code = failure_code_from_abi(failure_code)?;
    match (valid_flag != 0, code) {
        (true, Some(code)) => Err(PropagationError::BackendProtocol(format!(
            "external propagator marked row {row} valid with failure code {code:?}"
        ))),
        (false, None) => Err(PropagationError::BackendProtocol(format!(
            "external propagator marked row {row} invalid without a failure code"
        ))),
        (false, Some(code)) => Ok((false, Some(code))),
        (true, None) if state_is_finite => Ok((true, None)),
        // A provider may only know that integration succeeded. Adam-core owns
        // the final finite-output validation and supplies its stable code.
        (true, None) => Ok((false, Some(PropagationFailureCode::NonFiniteOutputState))),
    }
}

fn callback_status_error(
    identity: &BackendIdentity,
    operation: &str,
    status: i32,
    error: &mut [c_char],
) -> PropagationError {
    if let Some(last) = error.last_mut() {
        *last = 0;
    }
    let message = if error.is_empty() {
        "provider supplied no error buffer".into()
    } else {
        // SAFETY: adam-core initializes the buffer to NUL and restores a
        // trailing NUL above even when a provider fills the whole buffer.
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    let prefix = format!("{} {} {operation}", identity.name, identity.version);
    match status {
        EXTERNAL_PROPAGATOR_STATUS_ERROR => {
            PropagationError::Backend(format!("{prefix} failed: {message}"))
        }
        EXTERNAL_PROPAGATOR_STATUS_PANIC => {
            PropagationError::Backend(format!("{prefix} provider panic: {message}"))
        }
        EXTERNAL_PROPAGATOR_STATUS_INVALID => PropagationError::BackendProtocol(format!(
            "{prefix} rejected the ABI request as invalid: {message}"
        )),
        other => PropagationError::BackendProtocol(format!(
            "{prefix} returned unknown status {other}: {message}"
        )),
    }
}

fn c_string(pointer: *const c_char, field: &str) -> PropagationResultValue<String> {
    if pointer.is_null() {
        return Err(PropagationError::BackendProtocol(format!(
            "external propagator: {field} is null"
        )));
    }
    // SAFETY: the ABI requires NUL-terminated strings valid for the
    // capsule's lifetime.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|err| {
            PropagationError::BackendProtocol(format!(
                "external propagator: {field} is not UTF-8: {err}"
            ))
        })
}

/// An [`ExternalPropagatorV2`] as a [`Propagator`]. Covariance propagation is
/// not part of the ABI; non-gravitational inputs are capability-gated and
/// orbits in one request must share an origin.
pub struct ExternalPropagator {
    context: *const c_void,
    propagate_callback: PropagateV2,
    stm_callback: Option<PropagateStmV2>,
    identity: BackendIdentity,
    integration_scale: TimeScale,
    capabilities: u64,
}

// SAFETY: the ABI requires every declared callback to be callable from any
// thread and the context to outlive the capsule the Python driver retains.
unsafe impl Send for ExternalPropagator {}
unsafe impl Sync for ExternalPropagator {}

impl ExternalPropagator {
    /// Wrap `raw`.
    ///
    /// # Safety
    /// `raw` must point to the readable prefix declared by `struct_size`; the
    /// table, strings, callbacks and context must outlive the returned value.
    pub unsafe fn from_raw(raw: *const ExternalPropagatorV2) -> PropagationResultValue<Self> {
        let header = read_header(raw.cast(), "table")?;
        if header.abi_version != EXTERNAL_PROPAGATOR_ABI_VERSION {
            return Err(PropagationError::BackendProtocol(format!(
                "external propagator ABI version {} (adam-core requires exactly {})",
                header.abi_version, EXTERNAL_PROPAGATOR_ABI_VERSION
            )));
        }
        if header.struct_size < EXTERNAL_PROPAGATOR_V2_MIN_SIZE {
            return Err(PropagationError::BackendProtocol(format!(
                "external propagator table has {} bytes; ABI v{} requires the {}-byte prefix",
                header.struct_size,
                EXTERNAL_PROPAGATOR_ABI_VERSION,
                EXTERNAL_PROPAGATOR_V2_MIN_SIZE
            )));
        }
        if !pointer_is_aligned(raw.cast::<ExternalPropagatorRequiredV2>()) {
            return Err(PropagationError::BackendProtocol(
                "external propagator: misaligned table prefix".to_string(),
            ));
        }
        // SAFETY: the validated size covers this exact repr(C) prefix. Copying
        // it does not create a reference to the larger public table type.
        let table = std::ptr::read(raw.cast::<ExternalPropagatorRequiredV2>());
        validate_reserved_words(
            raw.cast(),
            header.struct_size,
            std::mem::offset_of!(ExternalPropagatorV2, reserved),
            "table",
        )?;
        // Unknown optional capability bits are forward-compatible provider
        // advertisements. This consumer ignores them; every known capability
        // whose semantics are required below is still validated fail-closed.
        if table.context.is_null() {
            return Err(PropagationError::BackendProtocol(
                "external propagator context is null".to_string(),
            ));
        }
        if table.two_body_dynamics > 1 {
            return Err(PropagationError::BackendProtocol(format!(
                "external propagator two_body_dynamics must be 0 or 1, got {}",
                table.two_body_dynamics
            )));
        }
        if table.capabilities & EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE == 0 {
            return Err(PropagationError::BackendProtocol(
                "external propagator must declare thread-safe callbacks".to_string(),
            ));
        }
        let propagate_callback = table.propagate.ok_or_else(|| {
            PropagationError::BackendProtocol(
                "external propagator has no propagation callback".to_string(),
            )
        })?;
        let stm_callback = if table.capabilities & EXTERNAL_PROPAGATOR_CAP_STM != 0 {
            if header.struct_size < EXTERNAL_PROPAGATOR_V2_STM_SIZE {
                return Err(PropagationError::BackendProtocol(format!(
                    "external propagator declares STM support but its {}-byte table does not contain the STM slot ({} bytes required)",
                    header.struct_size, EXTERNAL_PROPAGATOR_V2_STM_SIZE
                )));
            }
            // SAFETY: the validated size covers this exact repr(C) prefix.
            let with_stm = std::ptr::read(raw.cast::<ExternalPropagatorStmV2>());
            with_stm
                .propagate_stm
                .ok_or_else(|| {
                    PropagationError::BackendProtocol(
                        "external propagator declares STM support without an STM callback"
                            .to_string(),
                    )
                })?
                .into()
        } else {
            None
        };
        let integration_scale = TimeScale::parse(&c_string(
            table.integration_time_scale,
            "integration_time_scale",
        )?)
        .map_err(|err| PropagationError::BackendProtocol(err.to_string()))?;
        let identity = BackendIdentity {
            name: c_string(table.name, "name")?,
            version: c_string(table.version, "version")?,
            two_body_dynamics: table.two_body_dynamics != 0,
        };
        if identity.name.is_empty() || identity.version.is_empty() {
            return Err(PropagationError::BackendProtocol(
                "external propagator identity must have non-empty name and version".to_string(),
            ));
        }
        Ok(Self {
            context: table.context,
            propagate_callback,
            stm_callback,
            identity,
            integration_scale,
            capabilities: table.capabilities,
        })
    }

    /// Identity read from the immutable external table.
    pub fn identity(&self) -> &BackendIdentity {
        &self.identity
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

    fn supports_stm(&self) -> bool {
        self.capabilities & EXTERNAL_PROPAGATOR_CAP_STM != 0
    }

    fn supports_non_gravitational_parameters(&self) -> bool {
        self.capabilities & EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL != 0
    }

    fn propagate_stm(
        &self,
        state: [f64; 6],
        non_gravitational_parameters: Option<crate::NonGravitationalParametersRow>,
        epoch_tdb: Epoch,
        times_tdb: &[Epoch],
    ) -> PropagationResultValue<Vec<super::StmRow>> {
        let Some(function) = self.stm_callback else {
            return Err(PropagationError::InvalidRequest(format!(
                "{} provides no state-transition matrices",
                self.identity.name
            )));
        };
        // Validate every present value before inactive rows are filtered and
        // before NaN is used as the provider-side null sentinel.
        let encoded_non_gravitational_values = non_gravitational_parameters
            .map(|parameters| {
                Ok::<_, PropagationError>([
                    encode_optional_parameter(
                        parameters.a1,
                        "non_gravitational_parameters",
                        "a1",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.a2,
                        "non_gravitational_parameters",
                        "a2",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.a3,
                        "non_gravitational_parameters",
                        "a3",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.aln,
                        "non_gravitational_parameters",
                        "aln",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.nk,
                        "non_gravitational_parameters",
                        "nk",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.nm,
                        "non_gravitational_parameters",
                        "nm",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.nn,
                        "non_gravitational_parameters",
                        "nn",
                        0,
                    )?,
                    encode_optional_parameter(
                        parameters.r0,
                        "non_gravitational_parameters",
                        "r0",
                        0,
                    )?,
                ])
            })
            .transpose()?;
        let non_gravitational_parameters =
            non_gravitational_parameters.filter(crate::NonGravitationalParametersRow::is_active);
        if non_gravitational_parameters.is_some() && !self.supports_non_gravitational_parameters() {
            return Err(PropagationError::InvalidRequest(format!(
                "{} does not accept non-gravitational parameters",
                self.identity.name
            )));
        }
        let non_gravitational_values =
            non_gravitational_parameters.and(encoded_non_gravitational_values);
        let n = times_tdb.len();
        let state_values = n.checked_mul(6).ok_or_else(|| {
            PropagationError::InvalidRequest("external STM state length overflows".to_string())
        })?;
        let stm_values = n.checked_mul(36).ok_or_else(|| {
            PropagationError::InvalidRequest("external STM matrix length overflows".to_string())
        })?;
        let days: Vec<i64> = times_tdb.iter().map(|e| e.days).collect();
        let nanos: Vec<i64> = times_tdb.iter().map(|e| e.nanos).collect();
        let mut states = vec![f64::NAN; state_values];
        let mut stms = vec![f64::NAN; stm_values];
        let mut valid = vec![0_u8; n];
        let mut error = vec![0 as c_char; 1024];
        let request = PropagateStmRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<PropagateStmRequestV2>(),
            state: state.as_ptr(),
            non_gravitational_parameters: non_gravitational_values
                .as_ref()
                .map_or(std::ptr::null(), |values| values.as_ptr()),
            epoch_days: epoch_tdb.days,
            epoch_nanos: epoch_tdb.nanos,
            n_times: n,
            time_days: days.as_ptr(),
            time_nanos: nanos.as_ptr(),
            out_states: states.as_mut_ptr(),
            out_stm: stms.as_mut_ptr(),
            out_valid: valid.as_mut_ptr(),
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        // SAFETY: every pointer is valid and sized as the ABI states.
        let status = unsafe { function(self.context, &request) };
        if status != EXTERNAL_PROPAGATOR_STATUS_OK {
            return Err(callback_status_error(
                &self.identity,
                "STM callback",
                status,
                &mut error,
            ));
        }
        Ok((0..n)
            .map(|i| super::StmRow {
                state: std::array::from_fn(|k| states[i * 6 + k]),
                stm: std::array::from_fn(|k| stms[i * 36 + k]),
                valid: valid[i] != 0
                    && states[i * 6..i * 6 + 6]
                        .iter()
                        .all(|value| value.is_finite())
                    && stms[i * 36..i * 36 + 36]
                        .iter()
                        .all(|value| value.is_finite()),
            })
            .collect())
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
        if pairwise && n_orbits != n_times {
            return Err(PropagationError::InvalidRequest(format!(
                "pairwise external propagation requires equal orbit/time lengths ({n_orbits} != {n_times})"
            )));
        }
        let rows_per_orbit = if pairwise { 1 } else { n_times };
        let rows = n_orbits.checked_mul(rows_per_orbit).ok_or_else(|| {
            PropagationError::InvalidRequest("external propagation row count overflows".to_string())
        })?;
        let state_values = rows.checked_mul(6).ok_or_else(|| {
            PropagationError::InvalidRequest(
                "external propagation state output length overflows".to_string(),
            )
        })?;
        let flat: Vec<f64> = states.iter().flatten().copied().collect();
        let physical_parameters = request
            .input
            .physical_parameters()
            .map(|parameters| {
                let mut values =
                    Vec::with_capacity(n_orbits.saturating_mul(EXTERNAL_PHYSICAL_PARAMETERS_WIDTH));
                for row in 0..n_orbits {
                    for (field, value) in [
                        ("h_v", parameters.h_v[row]),
                        ("h_v_sigma", parameters.h_v_sigma[row]),
                        ("g", parameters.g[row]),
                        ("g_sigma", parameters.g_sigma[row]),
                        ("sigma_eff", parameters.sigma_eff[row]),
                        ("chi2_red", parameters.chi2_red[row]),
                    ] {
                        values.push(encode_optional_parameter(
                            value,
                            "physical_parameters",
                            field,
                            row,
                        )?);
                    }
                }
                Ok::<_, PropagationError>(values)
            })
            .transpose()?;
        let non_gravitational_parameters = match request.input.non_gravitational_parameters() {
            None => None,
            Some(parameters) => {
                let mut values =
                    Vec::with_capacity(n_orbits.saturating_mul(EXTERNAL_NON_GRAVITATIONAL_WIDTH));
                for row in 0..n_orbits {
                    let parameters_row = parameters.row(row);
                    for (field, value) in [
                        ("a1", parameters_row.a1),
                        ("a2", parameters_row.a2),
                        ("a3", parameters_row.a3),
                        ("aln", parameters_row.aln),
                        ("nk", parameters_row.nk),
                        ("nm", parameters_row.nm),
                        ("nn", parameters_row.nn),
                        ("r0", parameters_row.r0),
                    ] {
                        values.push(encode_optional_parameter(
                            value,
                            "non_gravitational_parameters",
                            field,
                            row,
                        )?);
                    }
                }
                if parameters.has_values() {
                    if !self.supports_non_gravitational_parameters() {
                        return Err(PropagationError::InvalidRequest(format!(
                            "{} does not accept non-gravitational parameters",
                            self.identity.name
                        )));
                    }
                    Some(values)
                } else {
                    None
                }
            }
        };
        let epoch_days: Vec<i64> = orbit_times.epochs.iter().map(|e| e.days).collect();
        let epoch_nanos: Vec<i64> = orbit_times.epochs.iter().map(|e| e.nanos).collect();
        let time_days: Vec<i64> = target_times.epochs.iter().map(|e| e.days).collect();
        let time_nanos: Vec<i64> = target_times.epochs.iter().map(|e| e.nanos).collect();
        let scale = CString::new(self.integration_scale.as_str()).expect("no NUL");
        let frame = CString::new(coordinates.frame.as_str()).expect("no NUL");
        let origin = CString::new(origin)
            .map_err(|err| PropagationError::InvalidRequest(format!("origin code: {err}")))?;
        let mut out_states = vec![f64::NAN; state_values];
        let mut out_valid = vec![0_u8; rows];
        let mut out_failure_codes = vec![u8::MAX; rows];
        let mut error = vec![0 as c_char; 1024];
        let c_request = PropagateRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<PropagateRequestV2>(),
            n_orbits,
            states: flat.as_ptr(),
            physical_parameters: physical_parameters
                .as_ref()
                .map_or(std::ptr::null(), |values| values.as_ptr()),
            non_gravitational_parameters: non_gravitational_parameters
                .as_ref()
                .map_or(std::ptr::null(), |values| values.as_ptr()),
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
            out_failure_codes: out_failure_codes.as_mut_ptr(),
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        // SAFETY: the table outlives `self`; every pointer in the request is
        // valid for the call and sized as the ABI states.
        let status = unsafe { (self.propagate_callback)(self.context, &c_request) };
        if status != EXTERNAL_PROPAGATOR_STATUS_OK {
            return Err(callback_status_error(
                &self.identity,
                "propagation callback",
                status,
                &mut error,
            ));
        }
        let blocks = (0..n_orbits)
            .map(|orbit| -> PropagationResultValue<OrbitBlock> {
                let time_indices: Vec<usize> = if pairwise {
                    vec![orbit]
                } else {
                    (0..n_times).collect()
                };
                let base = orbit * rows_per_orbit;
                let states: Vec<[f64; 6]> = (0..rows_per_orbit)
                    .map(|k| std::array::from_fn(|axis| out_states[(base + k) * 6 + axis]))
                    .collect();
                let row_results: PropagationResultValue<Vec<_>> = (0..rows_per_orbit)
                    .map(|k| {
                        decode_row_result(
                            base + k,
                            out_valid[base + k],
                            states[k].iter().all(|value| value.is_finite()),
                            out_failure_codes[base + k],
                        )
                    })
                    .collect();
                let row_results = row_results?;
                let validity = row_results.iter().map(|(valid, _)| *valid).collect();
                let failure_codes = row_results.into_iter().map(|(_, code)| code).collect();
                Ok(OrbitBlock {
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
                })
            })
            .collect::<PropagationResultValue<Vec<_>>>()?;
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
    table: ExternalPropagatorV2,
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
        let mut capabilities = EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE;
        if context.propagator.supports_stm() {
            capabilities |= EXTERNAL_PROPAGATOR_CAP_STM;
        }
        if context.propagator.supports_non_gravitational_parameters() {
            capabilities |= EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL;
        }
        let table = ExternalPropagatorV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<ExternalPropagatorV2>(),
            capabilities,
            integration_time_scale: context.scale.as_ptr(),
            two_body_dynamics: u8::from(identity.two_body_dynamics),
            name: context.name.as_ptr(),
            version: context.version.as_ptr(),
            context: &*context as *const ExternalContext<P> as *const c_void,
            propagate: Some(propagate_with::<P>),
            propagate_stm: context
                .propagator
                .supports_stm()
                .then_some(propagate_stm_with::<P> as PropagateStmV2),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        Self { table, context }
    }

    pub fn table(&self) -> *const ExternalPropagatorV2 {
        &self.table
    }
}

fn write_error(request: &PropagateRequestPrefixV2, message: &str) {
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

unsafe fn parse_propagate_request_prefix(
    request: *const PropagateRequestV2,
) -> Result<PropagateRequestPrefixV2, (Option<Box<PropagateRequestPrefixV2>>, String)> {
    let header = read_header(request.cast(), "propagation request")
        .map_err(|error| (None, error.to_string()))?;
    // A caller using another ABI version has not promised that any v2 offset
    // after the header exists. In particular, never write through its error
    // pointer.
    if header.abi_version != EXTERNAL_PROPAGATOR_ABI_VERSION {
        return Err((
            None,
            format!(
                "propagation request ABI version {} (provider requires exactly {})",
                header.abi_version, EXTERNAL_PROPAGATOR_ABI_VERSION
            ),
        ));
    }
    if header.struct_size < PROPAGATE_REQUEST_V2_MIN_SIZE {
        return Err((
            None,
            format!(
                "propagation request has {} bytes; ABI v{} requires the {}-byte prefix",
                header.struct_size, EXTERNAL_PROPAGATOR_ABI_VERSION, PROPAGATE_REQUEST_V2_MIN_SIZE
            ),
        ));
    }
    if !pointer_is_aligned(request.cast::<PropagateRequestPrefixV2>()) {
        return Err((None, "misaligned propagation request prefix".to_string()));
    }
    // SAFETY: the validated byte count covers this exact prefix type.
    let prefix = std::ptr::read(request.cast::<PropagateRequestPrefixV2>());
    if let Err(error) = validate_reserved_words(
        request.cast(),
        header.struct_size,
        std::mem::offset_of!(PropagateRequestV2, reserved),
        "propagation request",
    ) {
        return Err((Some(Box::new(prefix)), error.to_string()));
    }
    Ok(prefix)
}

unsafe extern "C" fn propagate_with<P: Propagator>(
    context: *const c_void,
    request: *const PropagateRequestV2,
) -> i32 {
    if context.is_null() || request.is_null() {
        return EXTERNAL_PROPAGATOR_STATUS_INVALID;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let request = match parse_propagate_request_prefix(request) {
            Ok(request) => request,
            Err((reportable, message)) => {
                if let Some(request) = reportable {
                    write_error(&request, &message);
                }
                return EXTERNAL_PROPAGATOR_STATUS_INVALID;
            }
        };
        if !pointer_is_aligned(context.cast::<ExternalContext<P>>()) {
            write_error(&request, "misaligned external propagator context");
            return EXTERNAL_PROPAGATOR_STATUS_INVALID;
        }
        // SAFETY: this is the provider's own context pointer and its exact
        // alignment was validated before creating the reference.
        let context = &*(context.cast::<ExternalContext<P>>());
        match run_request(&context.propagator, &request) {
            Ok(()) => EXTERNAL_PROPAGATOR_STATUS_OK,
            Err(CallbackFailure::Invalid(message)) => {
                write_error(&request, &message);
                EXTERNAL_PROPAGATOR_STATUS_INVALID
            }
            Err(CallbackFailure::Backend(message)) => {
                write_error(&request, &message);
                EXTERNAL_PROPAGATOR_STATUS_ERROR
            }
        }
    }));
    match result {
        Ok(status) => status,
        Err(_) => {
            // Parsing is repeated without touching provider state. Only an
            // exact-v2 request with the complete prefix owns the error slot.
            match parse_propagate_request_prefix(request) {
                Ok(request) => write_error(&request, "external propagator panicked"),
                Err((Some(request), _)) => write_error(&request, "external propagator panicked"),
                Err((None, _)) => {}
            }
            EXTERNAL_PROPAGATOR_STATUS_PANIC
        }
    }
}

fn validate_non_aliasing(
    buffers: &[(usize, usize, usize, usize, &'static str)],
) -> Result<(), String> {
    let mut ranges = Vec::with_capacity(buffers.len());
    for &(start, elements, element_size, alignment, name) in buffers {
        if start == 0 || elements == 0 {
            continue;
        }
        if start % alignment != 0 {
            return Err(format!(
                "request buffer {name} is not aligned to {alignment} bytes"
            ));
        }
        let bytes = elements
            .checked_mul(element_size)
            .ok_or_else(|| format!("{name} byte length overflows usize"))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| format!("{name} address range overflows usize"))?;
        ranges.push((start, end, name));
    }
    for (index, &(left_start, left_end, left_name)) in ranges.iter().enumerate() {
        for &(right_start, right_end, right_name) in &ranges[index + 1..] {
            if left_start < right_end && right_start < left_end {
                return Err(format!(
                    "request buffers {left_name} and {right_name} overlap"
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
enum CallbackFailure {
    Invalid(String),
    Backend(String),
}

impl From<String> for CallbackFailure {
    fn from(message: String) -> Self {
        Self::Backend(message)
    }
}

unsafe fn write_stm_error(request: &PropagateStmRequestPrefixV2, message: &str) {
    if request.error.is_null() || request.error_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let n = bytes.len().min(request.error_len - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, request.error, n);
    *request.error.add(n) = 0;
}

unsafe fn run_stm_request<P: Propagator>(
    context: &ExternalContext<P>,
    request: &PropagateStmRequestPrefixV2,
) -> Result<(), CallbackFailure> {
    let state_values = 6usize;
    let time_values = request.n_times;
    let output_state_values = request
        .n_times
        .checked_mul(6)
        .ok_or_else(|| CallbackFailure::Invalid("STM state output length overflows".to_string()))?;
    let output_matrix_values = request.n_times.checked_mul(36).ok_or_else(|| {
        CallbackFailure::Invalid("STM matrix output length overflows".to_string())
    })?;
    let required_for_rows_are_null = request.n_times > 0
        && (request.time_days.is_null()
            || request.time_nanos.is_null()
            || request.out_states.is_null()
            || request.out_stm.is_null()
            || request.out_valid.is_null());
    if request.state.is_null() || required_for_rows_are_null {
        return Err(CallbackFailure::Invalid(
            "STM request has a null required pointer".to_string(),
        ));
    }
    validate_non_aliasing(&[
        (
            request.state as usize,
            state_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "state",
        ),
        (
            request.non_gravitational_parameters as usize,
            EXTERNAL_NON_GRAVITATIONAL_WIDTH,
            size_of::<f64>(),
            align_of::<f64>(),
            "non_gravitational_parameters",
        ),
        (
            request.time_days as usize,
            time_values,
            size_of::<i64>(),
            align_of::<i64>(),
            "time_days",
        ),
        (
            request.time_nanos as usize,
            time_values,
            size_of::<i64>(),
            align_of::<i64>(),
            "time_nanos",
        ),
        (
            request.out_states as usize,
            output_state_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "out_states",
        ),
        (
            request.out_stm as usize,
            output_matrix_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "out_stm",
        ),
        (
            request.out_valid as usize,
            request.n_times,
            size_of::<u8>(),
            align_of::<u8>(),
            "out_valid",
        ),
        (
            request.error as usize,
            request.error_len,
            size_of::<c_char>(),
            align_of::<c_char>(),
            "error",
        ),
    ])
    .map_err(CallbackFailure::Invalid)?;

    let state_values = std::slice::from_raw_parts(request.state, 6);
    let state: [f64; 6] = std::array::from_fn(|i| state_values[i]);
    let non_gravitational_parameters = if request.non_gravitational_parameters.is_null() {
        None
    } else {
        let values = std::slice::from_raw_parts(
            request.non_gravitational_parameters,
            EXTERNAL_NON_GRAVITATIONAL_WIDTH,
        );
        let field = |index: usize| (!values[index].is_nan()).then_some(values[index]);
        Some(crate::NonGravitationalParametersRow {
            a1: field(0),
            a2: field(1),
            a3: field(2),
            aln: field(3),
            nk: field(4),
            nm: field(5),
            nn: field(6),
            r0: field(7),
        })
        .filter(crate::NonGravitationalParametersRow::is_active)
    };
    let days = if request.n_times == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(request.time_days, request.n_times)
    };
    let nanos = if request.n_times == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(request.time_nanos, request.n_times)
    };
    let times: Vec<Epoch> = days
        .iter()
        .zip(nanos)
        .map(|(&days, &nanos)| Epoch::new(days, nanos))
        .collect();
    let rows = context
        .propagator
        .propagate_stm(
            state,
            non_gravitational_parameters,
            Epoch::new(request.epoch_days, request.epoch_nanos),
            &times,
        )
        .map_err(|err| CallbackFailure::Backend(err.to_string()))?;
    if rows.len() != request.n_times {
        return Err(CallbackFailure::Backend(format!(
            "expected {} STM rows, got {}",
            request.n_times,
            rows.len()
        )));
    }
    if request.n_times == 0 {
        return Ok(());
    }
    let states = std::slice::from_raw_parts_mut(request.out_states, output_state_values);
    let stms = std::slice::from_raw_parts_mut(request.out_stm, output_matrix_values);
    let valid = std::slice::from_raw_parts_mut(request.out_valid, request.n_times);
    for (i, row) in rows.iter().enumerate() {
        states[i * 6..i * 6 + 6].copy_from_slice(&row.state);
        stms[i * 36..i * 36 + 36].copy_from_slice(&row.stm);
        valid[i] = u8::from(row.valid);
    }
    Ok(())
}

unsafe fn parse_stm_request_prefix(
    request: *const PropagateStmRequestV2,
) -> Result<PropagateStmRequestPrefixV2, (Option<Box<PropagateStmRequestPrefixV2>>, String)> {
    let header =
        read_header(request.cast(), "STM request").map_err(|error| (None, error.to_string()))?;
    if header.abi_version != EXTERNAL_PROPAGATOR_ABI_VERSION {
        return Err((
            None,
            format!(
                "STM request ABI version {} (provider requires exactly {})",
                header.abi_version, EXTERNAL_PROPAGATOR_ABI_VERSION
            ),
        ));
    }
    if header.struct_size < PROPAGATE_STM_REQUEST_V2_MIN_SIZE {
        return Err((
            None,
            format!(
                "STM request has {} bytes; ABI v{} requires the {}-byte prefix",
                header.struct_size,
                EXTERNAL_PROPAGATOR_ABI_VERSION,
                PROPAGATE_STM_REQUEST_V2_MIN_SIZE
            ),
        ));
    }
    if !pointer_is_aligned(request.cast::<PropagateStmRequestPrefixV2>()) {
        return Err((None, "misaligned STM request prefix".to_string()));
    }
    // SAFETY: the validated byte count covers this exact prefix type.
    let prefix = std::ptr::read(request.cast::<PropagateStmRequestPrefixV2>());
    if let Err(error) = validate_reserved_words(
        request.cast(),
        header.struct_size,
        std::mem::offset_of!(PropagateStmRequestV2, reserved),
        "STM request",
    ) {
        return Err((Some(Box::new(prefix)), error.to_string()));
    }
    Ok(prefix)
}

unsafe extern "C" fn propagate_stm_with<P: Propagator>(
    context: *const c_void,
    request: *const PropagateStmRequestV2,
) -> i32 {
    if context.is_null() || request.is_null() {
        return EXTERNAL_PROPAGATOR_STATUS_INVALID;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let request = match parse_stm_request_prefix(request) {
            Ok(request) => request,
            Err((reportable, message)) => {
                if let Some(request) = reportable {
                    write_stm_error(&request, &message);
                }
                return EXTERNAL_PROPAGATOR_STATUS_INVALID;
            }
        };
        if !pointer_is_aligned(context.cast::<ExternalContext<P>>()) {
            write_stm_error(&request, "misaligned external propagator context");
            return EXTERNAL_PROPAGATOR_STATUS_INVALID;
        }
        // SAFETY: this is the provider's own context pointer and its exact
        // alignment was validated before creating the reference.
        let context = &*(context.cast::<ExternalContext<P>>());
        match run_stm_request(context, &request) {
            Ok(()) => EXTERNAL_PROPAGATOR_STATUS_OK,
            Err(CallbackFailure::Invalid(message)) => {
                write_stm_error(&request, &message);
                EXTERNAL_PROPAGATOR_STATUS_INVALID
            }
            Err(CallbackFailure::Backend(message)) => {
                write_stm_error(&request, &message);
                EXTERNAL_PROPAGATOR_STATUS_ERROR
            }
        }
    }));
    match result {
        Ok(status) => status,
        Err(_) => {
            match parse_stm_request_prefix(request) {
                Ok(request) => write_stm_error(&request, "external propagator panicked"),
                Err((Some(request), _)) => {
                    write_stm_error(&request, "external propagator panicked")
                }
                Err((None, _)) => {}
            }
            EXTERNAL_PROPAGATOR_STATUS_PANIC
        }
    }
}

/// How far a propagated row's epoch may be from the requested one (1 ms):
/// well above a backend's float-time quantization, far below any spacing of
/// distinct target times.
const EPOCH_MATCH_TOLERANCE_NS: f64 = 1e6;
const NANOS_PER_DAY: f64 = 86_400e9;

unsafe fn run_request<P: Propagator>(
    propagator: &P,
    request: &PropagateRequestPrefixV2,
) -> Result<(), CallbackFailure> {
    use crate::propagation::PropagationOptions;
    use crate::{
        CoordinateBatch, NonGravitationalParametersBatch, OrbitBatch, OrbitId, OriginArray,
        OriginId, PhysicalParametersBatch, TimeArray,
    };
    let n = request.n_orbits;
    let m = request.n_times;
    if request.pairwise > 1 {
        return Err(CallbackFailure::Invalid(format!(
            "pairwise must be 0 or 1, got {}",
            request.pairwise
        )));
    }
    let pairwise = request.pairwise == 1;
    if pairwise && n != m {
        return Err(CallbackFailure::Invalid(format!(
            "pairwise propagation requires n_orbits == n_times ({n} != {m})"
        )));
    }
    let checked_product = |left: usize, right: usize, field: &str| {
        left.checked_mul(right)
            .ok_or_else(|| CallbackFailure::Invalid(format!("{field} length overflows usize")))
    };
    let state_values = checked_product(n, 6, "states")?;
    let physical_values =
        checked_product(n, EXTERNAL_PHYSICAL_PARAMETERS_WIDTH, "physical_parameters")?;
    let non_gravitational_values = checked_product(
        n,
        EXTERNAL_NON_GRAVITATIONAL_WIDTH,
        "non_gravitational_parameters",
    )?;
    let rows = if pairwise {
        n
    } else {
        checked_product(n, m, "output rows")?
    };
    let output_values = checked_product(rows, 6, "out_states")?;
    let required = [
        (request.states.is_null() && state_values > 0, "states"),
        (request.epoch_days.is_null() && n > 0, "epoch_days"),
        (request.epoch_nanos.is_null() && n > 0, "epoch_nanos"),
        (request.time_days.is_null() && m > 0, "time_days"),
        (request.time_nanos.is_null() && m > 0, "time_nanos"),
        (request.out_states.is_null() && rows > 0, "out_states"),
        (request.out_valid.is_null() && rows > 0, "out_valid"),
        (request.orbit_time_scale.is_null(), "orbit_time_scale"),
        (request.time_scale.is_null(), "time_scale"),
        (request.frame.is_null(), "frame"),
        (request.origin.is_null(), "origin"),
    ];
    if let Some((_, field)) = required.into_iter().find(|(missing, _)| *missing) {
        return Err(CallbackFailure::Invalid(format!(
            "propagation request field {field} is null"
        )));
    }
    validate_non_aliasing(&[
        (
            request.states as usize,
            state_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "states",
        ),
        (
            request.physical_parameters as usize,
            physical_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "physical_parameters",
        ),
        (
            request.non_gravitational_parameters as usize,
            non_gravitational_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "non_gravitational_parameters",
        ),
        (
            request.epoch_days as usize,
            n,
            size_of::<i64>(),
            align_of::<i64>(),
            "epoch_days",
        ),
        (
            request.epoch_nanos as usize,
            n,
            size_of::<i64>(),
            align_of::<i64>(),
            "epoch_nanos",
        ),
        (
            request.time_days as usize,
            m,
            size_of::<i64>(),
            align_of::<i64>(),
            "time_days",
        ),
        (
            request.time_nanos as usize,
            m,
            size_of::<i64>(),
            align_of::<i64>(),
            "time_nanos",
        ),
        (
            request.out_states as usize,
            output_values,
            size_of::<f64>(),
            align_of::<f64>(),
            "out_states",
        ),
        (
            request.out_valid as usize,
            rows,
            size_of::<u8>(),
            align_of::<u8>(),
            "out_valid",
        ),
        (
            request.out_failure_codes as usize,
            rows,
            size_of::<u8>(),
            align_of::<u8>(),
            "out_failure_codes",
        ),
        (
            request.error as usize,
            request.error_len,
            size_of::<c_char>(),
            align_of::<c_char>(),
            "error",
        ),
    ])
    .map_err(CallbackFailure::Invalid)?;
    let flat: &[f64] = if state_values == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(request.states, state_values)
    };
    let states: Vec<[f64; 6]> = flat
        .chunks_exact(6)
        .map(|c| std::array::from_fn(|i| c[i]))
        .collect();
    let text = |p: *const c_char| CStr::from_ptr(p).to_string_lossy().into_owned();
    let scale = |p: *const c_char| {
        TimeScale::parse(&text(p)).map_err(|e| CallbackFailure::Invalid(e.to_string()))
    };
    let epochs = |days: *const i64, nanos: *const i64, k: usize| -> Vec<Epoch> {
        if k == 0 {
            return Vec::new();
        }
        let d = std::slice::from_raw_parts(days, k);
        let ns = std::slice::from_raw_parts(nanos, k);
        d.iter().zip(ns).map(|(&a, &b)| Epoch::new(a, b)).collect()
    };
    let frame =
        Frame::parse(&text(request.frame)).map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    let origin = OriginId::from_code(text(request.origin));
    let orbit_times = TimeArray::new(
        scale(request.orbit_time_scale)?,
        epochs(request.epoch_days, request.epoch_nanos, n),
    )
    .map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    // Zero-padded ids keep input order for backends that sort by id; the
    // rows are placed by id and epoch below whatever order they come in.
    let width = n.to_string().len();
    let mut orbits = OrbitBatch::new(
        (0..n).map(|i| OrbitId(format!("{i:0width$}"))).collect(),
        vec![None; n],
        CoordinateBatch::cartesian(
            states,
            frame,
            OriginArray::repeat(origin, n),
            Some(orbit_times),
            None,
        )
        .map_err(|e| CallbackFailure::Invalid(e.to_string()))?,
    )
    .map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    if n > 0 && !request.physical_parameters.is_null() {
        let values = std::slice::from_raw_parts(request.physical_parameters, physical_values);
        let field = |row: usize, axis: usize| {
            let value = values[row * EXTERNAL_PHYSICAL_PARAMETERS_WIDTH + axis];
            (!value.is_nan()).then_some(value)
        };
        let column = |axis: usize| (0..n).map(|row| field(row, axis)).collect();
        orbits = orbits
            .with_physical_parameters(PhysicalParametersBatch {
                h_v: column(0),
                h_v_sigma: column(1),
                g: column(2),
                g_sigma: column(3),
                sigma_eff: column(4),
                chi2_red: column(5),
            })
            .map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    }
    if n > 0 && !request.non_gravitational_parameters.is_null() {
        if !propagator.supports_non_gravitational_parameters() {
            return Err(CallbackFailure::Invalid(
                "provider received unsupported non-gravitational parameters".to_string(),
            ));
        }
        let values = std::slice::from_raw_parts(
            request.non_gravitational_parameters,
            non_gravitational_values,
        );
        let field = |row: usize, axis: usize| {
            let value = values[row * EXTERNAL_NON_GRAVITATIONAL_WIDTH + axis];
            (!value.is_nan()).then_some(value)
        };
        let column = |axis: usize| (0..n).map(|row| field(row, axis)).collect();
        orbits = orbits
            .with_non_gravitational_parameters(NonGravitationalParametersBatch {
                source: vec![None; n],
                a1: column(0),
                a2: column(1),
                a3: column(2),
                aln: column(3),
                nk: column(4),
                nm: column(5),
                nn: column(6),
                r0: column(7),
            })
            .map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    }
    let times = TimeArray::new(
        scale(request.time_scale)?,
        epochs(request.time_days, request.time_nanos, m),
    )
    .map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    let options = PropagationOptions {
        chunk_size: None,
        thread_limit: None,
        epoch_policy: if pairwise {
            EpochPolicy::Pairwise
        } else {
            EpochPolicy::CrossProduct
        },
        covariance: CovariancePropagation::None,
    };
    let propagation = PropagationRequest::new(&orbits, &times, options)
        .map_err(|e| CallbackFailure::Invalid(e.to_string()))?;
    let result = propagator
        .propagate(&propagation, &SameScale)
        .map_err(|e| CallbackFailure::Backend(e.to_string()))?;
    let out = result
        .orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| CallbackFailure::Backend("non-Cartesian output".to_string()))?;
    if out.len() != rows {
        return Err(CallbackFailure::Backend(format!(
            "expected {rows} rows, got {}",
            out.len()
        )));
    }
    // The ABI's row order is (orbit, target time); a propagator may return
    // its rows in another order (e.g. sorted by id), so each row is placed by
    // its orbit id and epoch rather than by position.
    let out_times =
        result.orbits.coordinates.times.as_ref().ok_or_else(|| {
            CallbackFailure::Backend("propagated rows carry no times".to_string())
        })?;
    let target_epochs = &times.epochs;
    let mut filled = vec![false; rows];
    let out_states: &mut [f64] = if output_values == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(request.out_states, output_values)
    };
    let out_valid: &mut [u8] = if rows == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(request.out_valid, rows)
    };
    let out_failure_codes: &mut [u8] = if rows == 0 || request.out_failure_codes.is_null() {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(request.out_failure_codes, rows)
    };
    for (row, state) in out.iter().enumerate() {
        let id = &result.orbits.orbit_id[row].0;
        let orbit = id
            .parse::<usize>()
            .ok()
            .filter(|&orbit| orbit < n)
            .ok_or_else(|| {
                CallbackFailure::Backend(format!(
                    "propagated row {row} has unknown orbit id {id:?}"
                ))
            })?;
        let epoch = out_times.epochs[row];
        // Backends may quantize their output times (ASSIST rebuilds them
        // from f64 Julian dates, tens of microseconds), so a row matches the
        // nearest unfilled requested epoch of its orbit within a millisecond.
        let offset = |slot: usize| {
            let target = target_epochs[if pairwise { slot } else { slot - orbit * m }];
            ((epoch.days - target.days) as f64 * NANOS_PER_DAY
                + (epoch.nanos - target.nanos) as f64)
                .abs()
        };
        let candidates: Vec<usize> = if pairwise {
            vec![orbit]
        } else {
            (0..m).map(|time| orbit * m + time).collect()
        };
        let slot = candidates
            .into_iter()
            .filter(|&slot| !filled[slot] && offset(slot) <= EPOCH_MATCH_TOLERANCE_NS)
            .min_by(|&a, &b| offset(a).total_cmp(&offset(b)).then(a.cmp(&b)))
            .ok_or_else(|| {
                format!("propagated row {row} (orbit {orbit}) does not match a requested epoch")
            })?;
        filled[slot] = true;
        out_states[slot * 6..slot * 6 + 6].copy_from_slice(state);
        let valid = result.validity.is_valid(row);
        out_valid[slot] = u8::from(valid);
        if !out_failure_codes.is_empty() {
            let code = result
                .diagnostics
                .convergence
                .get(row)
                .and_then(|diagnostic| diagnostic.failure_code)
                .or_else(|| (!valid).then_some(PropagationFailureCode::IntegratorFailure));
            out_failure_codes[slot] = failure_code_to_abi(code);
        }
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
    pub fn table(propagator: TwoBodyPropagator) -> &'static super::ExternalPropagatorV2 {
        let owned: &'static OwnedExternalPropagator<TwoBodyPropagator> =
            Box::leak(Box::new(new(propagator)));
        // SAFETY: `owned` is leaked, so its table lives for the process.
        unsafe { &*owned.table() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::{PropagationInput, PropagationOptions, TwoBodyPropagator};
    use crate::{
        CoordinateBatch, NonGravitationalParametersBatch, OrbitBatch, OrbitId, OriginArray,
        OriginId, PhysicalParametersBatch, TimeArray,
    };

    /// Two-body dynamics that return their rows in reverse order (as a
    /// backend sorting by id, or by anything else, may) with output times
    /// shifted by 30 microseconds (as ASSIST's float Julian-date times are):
    /// the ABI must still deliver rows in (orbit, time) order.
    struct Reversed(TwoBodyPropagator);

    impl Propagator for Reversed {
        type Shard = <TwoBodyPropagator as Propagator>::Shard;

        fn integration_time_scale(&self) -> TimeScale {
            self.0.integration_time_scale()
        }

        fn supports(&self, mode: CovariancePropagation) -> bool {
            self.0.supports(mode)
        }

        fn create_shard(&self) -> Self::Shard {
            self.0.create_shard()
        }

        fn propagate(
            &self,
            request: &PropagationRequest<'_>,
            provider: &dyn TimeScaleProvider,
        ) -> PropagationResultValue<crate::propagation::PropagationResult> {
            let result = self.0.propagate(request, provider)?;
            let rows: Vec<usize> = (0..result.orbits.orbit_id.len()).rev().collect();
            let mut coordinates =
                super::super::od::take_coordinate_batch(&result.orbits.coordinates, &rows)?;
            if let Some(times) = coordinates.times.as_mut() {
                for epoch in times.epochs.iter_mut() {
                    *epoch = Epoch::new(epoch.days, epoch.nanos + 30_000);
                }
            }
            let orbits = OrbitBatch {
                orbit_id: rows
                    .iter()
                    .map(|&r| result.orbits.orbit_id[r].clone())
                    .collect(),
                object_id: rows
                    .iter()
                    .map(|&r| result.orbits.object_id[r].clone())
                    .collect(),
                coordinates,
                physical_parameters: result
                    .orbits
                    .physical_parameters
                    .as_ref()
                    .map(|batch| batch.take(&rows)),
                non_gravitational_parameters: result
                    .orbits
                    .non_gravitational_parameters
                    .as_ref()
                    .map(|batch| batch.take(&rows)),
            };
            let validity: Vec<bool> = rows.iter().map(|&r| result.validity.is_valid(r)).collect();
            Ok(crate::propagation::PropagationResult {
                orbits,
                validity: crate::types::Validity::from_bools(&validity),
                ..result
            })
        }
    }

    /// Test provider that proves all non-gravitational fields crossed the ABI,
    /// then strips them before delegating to the two-body test dynamics.
    struct NonGravAware(TwoBodyPropagator);

    impl Propagator for NonGravAware {
        type Shard = <TwoBodyPropagator as Propagator>::Shard;

        fn integration_time_scale(&self) -> TimeScale {
            self.0.integration_time_scale()
        }

        fn supports_non_gravitational_parameters(&self) -> bool {
            true
        }

        fn supports_stm(&self) -> bool {
            true
        }

        fn propagate_stm(
            &self,
            state: [f64; 6],
            parameters: Option<crate::NonGravitationalParametersRow>,
            epoch_tdb: Epoch,
            times_tdb: &[Epoch],
        ) -> PropagationResultValue<Vec<crate::propagation::StmRow>> {
            let parameters = parameters.expect("fixed non-grav parameters must cross STM ABI");
            assert_eq!(parameters.a1, Some(1e-10));
            assert_eq!(parameters.a2, Some(-2e-11));
            assert_eq!(parameters.a3, Some(3e-12));
            assert_eq!(parameters.aln, Some(0.1112620426));
            assert_eq!(parameters.nk, Some(4.6142));
            assert_eq!(parameters.nm, Some(2.15));
            assert_eq!(parameters.nn, Some(5.093));
            assert_eq!(parameters.r0, Some(2.808));
            self.0.propagate_stm(state, None, epoch_tdb, times_tdb)
        }

        fn supports(&self, mode: CovariancePropagation) -> bool {
            self.0.supports(mode)
        }

        fn create_shard(&self) -> Self::Shard {
            self.0.create_shard()
        }

        fn propagate(
            &self,
            request: &PropagationRequest<'_>,
            provider: &dyn TimeScaleProvider,
        ) -> PropagationResultValue<crate::propagation::PropagationResult> {
            let PropagationInput::Orbits(input) = request.input else {
                return Err(PropagationError::InvalidRequest(
                    "test provider requires orbits".to_string(),
                ));
            };
            let parameters = input
                .non_gravitational_parameters
                .as_ref()
                .ok_or_else(|| PropagationError::InvalidRequest("missing non-grav".to_string()))?;
            assert_eq!(parameters.len(), 2);
            let first = parameters.row(0);
            assert_eq!(first.a1, Some(1e-10));
            assert_eq!(first.a2, Some(-2e-11));
            assert_eq!(first.a3, Some(3e-12));
            assert_eq!(first.aln, Some(0.1112620426));
            assert_eq!(first.nk, Some(4.6142));
            assert_eq!(first.nm, Some(2.15));
            assert_eq!(first.nn, Some(5.093));
            assert_eq!(first.r0, Some(2.808));
            let second = parameters.row(1);
            assert_eq!(second.a1, Some(2e-10));
            assert_eq!(second.a2, None);
            assert_eq!(second.a3, Some(-4e-12));
            assert_eq!(second.aln, None);
            assert_eq!(second.nk, Some(0.0));
            assert_eq!(second.nm, Some(2.0));
            assert_eq!(second.nn, Some(5.093));
            assert_eq!(second.r0, Some(1.0));
            let physical = input
                .physical_parameters
                .as_ref()
                .ok_or_else(|| PropagationError::InvalidRequest("missing physical".to_string()))?;
            assert_eq!(physical.h_v, vec![Some(12.1), Some(17.2)]);
            assert_eq!(physical.h_v_sigma, vec![Some(0.2), None]);
            assert_eq!(physical.g, vec![Some(0.15), Some(0.25)]);
            assert_eq!(physical.g_sigma, vec![None, Some(0.03)]);
            assert_eq!(physical.sigma_eff, vec![Some(0.7), Some(0.8)]);
            assert_eq!(physical.chi2_red, vec![Some(1.1), Some(2.2)]);
            let mut stripped = input.clone();
            stripped.non_gravitational_parameters = None;
            let delegated =
                PropagationRequest::new(&stripped, request.times, request.options.clone())?;
            self.0.propagate(&delegated, provider)
        }
    }

    #[derive(Clone, Copy)]
    enum StatusMode {
        Error,
        Panic,
    }

    struct StatusProvider {
        inner: TwoBodyPropagator,
        mode: StatusMode,
    }

    impl Propagator for StatusProvider {
        type Shard = <TwoBodyPropagator as Propagator>::Shard;

        fn integration_time_scale(&self) -> TimeScale {
            self.inner.integration_time_scale()
        }

        fn supports_stm(&self) -> bool {
            true
        }

        fn supports(&self, mode: CovariancePropagation) -> bool {
            self.inner.supports(mode)
        }

        fn create_shard(&self) -> Self::Shard {
            self.inner.create_shard()
        }

        fn propagate(
            &self,
            _request: &PropagationRequest<'_>,
            _provider: &dyn TimeScaleProvider,
        ) -> PropagationResultValue<crate::propagation::PropagationResult> {
            match self.mode {
                StatusMode::Error => Err(PropagationError::Backend("expected error".to_string())),
                StatusMode::Panic => panic!("expected panic"),
            }
        }

        fn propagate_stm(
            &self,
            _state: [f64; 6],
            _parameters: Option<crate::NonGravitationalParametersRow>,
            _epoch_tdb: Epoch,
            _times_tdb: &[Epoch],
        ) -> PropagationResultValue<Vec<crate::propagation::StmRow>> {
            match self.mode {
                StatusMode::Error => {
                    Err(PropagationError::Backend("expected STM error".to_string()))
                }
                StatusMode::Panic => panic!("expected STM panic"),
            }
        }
    }

    fn raw_propagation_status(mode: Option<StatusMode>, pairwise: u8) -> i32 {
        let states = [1.0, 0.0, 0.0, 0.0, 0.017, 0.0];
        let epoch_days = [60_000_i64];
        let epoch_nanos = [0_i64];
        let time_days = [60_001_i64];
        let time_nanos = [0_i64];
        let scale = CString::new("tdb").unwrap();
        let frame = CString::new("ecliptic").unwrap();
        let origin = CString::new("SUN").unwrap();
        let mut out_states = vec![0.0; 6];
        let mut out_valid = vec![0_u8; 1];
        let mut out_failure_codes = vec![u8::MAX; 1];
        let mut error = vec![0 as c_char; 128];
        let request = PropagateRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<PropagateRequestV2>(),
            n_orbits: 1,
            states: states.as_ptr(),
            physical_parameters: std::ptr::null(),
            non_gravitational_parameters: std::ptr::null(),
            epoch_days: epoch_days.as_ptr(),
            epoch_nanos: epoch_nanos.as_ptr(),
            orbit_time_scale: scale.as_ptr(),
            frame: frame.as_ptr(),
            origin: origin.as_ptr(),
            n_times: 1,
            time_days: time_days.as_ptr(),
            time_nanos: time_nanos.as_ptr(),
            time_scale: scale.as_ptr(),
            pairwise,
            out_states: out_states.as_mut_ptr(),
            out_valid: out_valid.as_mut_ptr(),
            out_failure_codes: out_failure_codes.as_mut_ptr(),
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        match mode {
            None => {
                let owned = two_body_reference::new(TwoBodyPropagator::default());
                unsafe { ((*owned.table()).propagate.unwrap())((*owned.table()).context, &request) }
            }
            Some(mode) => {
                let owned = OwnedExternalPropagator::new(
                    StatusProvider {
                        inner: TwoBodyPropagator::default(),
                        mode,
                    },
                    "status",
                    "0",
                );
                unsafe { ((*owned.table()).propagate.unwrap())((*owned.table()).context, &request) }
            }
        }
    }

    fn raw_stm_status(mode: StatusMode, abi_version: u32) -> i32 {
        let state = [1.0, 0.0, 0.0, 0.0, 0.017, 0.0];
        let time_days = [60_001_i64];
        let time_nanos = [0_i64];
        let mut out_states = vec![0.0; 6];
        let mut out_stm = vec![0.0; 36];
        let mut out_valid = vec![0_u8; 1];
        let mut error = vec![0 as c_char; 128];
        let request = PropagateStmRequestV2 {
            abi_version,
            struct_size: size_of::<PropagateStmRequestV2>(),
            state: state.as_ptr(),
            non_gravitational_parameters: std::ptr::null(),
            epoch_days: 60_000,
            epoch_nanos: 0,
            n_times: 1,
            time_days: time_days.as_ptr(),
            time_nanos: time_nanos.as_ptr(),
            out_states: out_states.as_mut_ptr(),
            out_stm: out_stm.as_mut_ptr(),
            out_valid: out_valid.as_mut_ptr(),
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        let owned = OwnedExternalPropagator::new(
            StatusProvider {
                inner: TwoBodyPropagator::default(),
                mode,
            },
            "status",
            "0",
        );
        unsafe { ((*owned.table()).propagate_stm.unwrap())((*owned.table()).context, &request) }
    }

    fn propagate_both(
        n: usize,
        times_days: &[i64],
        pairwise: bool,
    ) -> (Vec<[f64; 6]>, Vec<[f64; 6]>) {
        let base = [1.1, 0.2, 0.05, -0.003, 0.016, 0.0004];
        let states: Vec<[f64; 6]> = (0..n)
            .map(|i| {
                std::array::from_fn(|axis| base[axis] * (1.0 + 1e-3 * (i * (axis + 1)) as f64))
            })
            .collect();
        let orbits = OrbitBatch::new(
            (0..n).map(|i| OrbitId(format!("orbit-{i}"))).collect(),
            vec![None; n],
            CoordinateBatch::cartesian(
                states,
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), n),
                Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60000, 0); n]).unwrap()),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let times = TimeArray::new(
            TimeScale::Tdb,
            times_days.iter().map(|&d| Epoch::new(d, 0)).collect(),
        )
        .unwrap();
        let options = PropagationOptions {
            chunk_size: None,
            thread_limit: None,
            epoch_policy: if pairwise {
                EpochPolicy::Pairwise
            } else {
                EpochPolicy::CrossProduct
            },
            covariance: CovariancePropagation::None,
        };
        let request = PropagationRequest::new(&orbits, &times, options).unwrap();
        let owned: &'static OwnedExternalPropagator<Reversed> = Box::leak(Box::new(
            OwnedExternalPropagator::new(Reversed(TwoBodyPropagator::default()), "reversed", "0"),
        ));
        // SAFETY: `owned` is leaked, so its table lives for the process.
        let external = unsafe { ExternalPropagator::from_raw(owned.table()) }.unwrap();
        let rows = |result: crate::propagation::PropagationResult| {
            result
                .orbits
                .coordinates
                .values
                .cartesian()
                .unwrap()
                .to_vec()
        };
        (
            rows(external.propagate(&request, &SameScale).unwrap()),
            rows(
                TwoBodyPropagator::default()
                    .propagate(&request, &SameScale)
                    .unwrap(),
            ),
        )
    }

    #[test]
    fn rows_are_placed_by_orbit_and_time_whatever_order_the_backend_returns() {
        // 12 orbits (the central-difference batch; ids "10" and "11" sort
        // before "2" as text) across three target times.
        let (external, direct) = propagate_both(12, &[60010, 60030, 60050], false);
        assert_eq!(external.len(), 36);
        assert_eq!(external, direct);
        // A repeated target time: its two rows are interchangeable (equal to
        // rounding), so they are compared to 1e-12.
        let (external, direct) = propagate_both(12, &[60010, 60030, 60010], false);
        for (a, b) in external.iter().zip(&direct) {
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12));
        }
        let (external, direct) = propagate_both(
            12,
            &(0..12).map(|i| 60001 + 3 * i).collect::<Vec<_>>(),
            true,
        );
        assert_eq!(external, direct);
    }

    #[test]
    fn rejects_short_or_non_thread_safe_tables() {
        let mut owned = two_body_reference::new(TwoBodyPropagator::default());
        owned.table.struct_size = EXTERNAL_PROPAGATOR_V2_MIN_SIZE - 1;
        let error = unsafe { ExternalPropagator::from_raw(owned.table()) }
            .err()
            .expect("short table must fail");
        assert!(error.to_string().contains("requires the"));

        owned.table.struct_size = EXTERNAL_PROPAGATOR_V2_MIN_SIZE;
        owned.table.capabilities = EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE;
        unsafe { ExternalPropagator::from_raw(owned.table()) }
            .expect("the frozen mandatory prefix is sufficient without STM");

        owned.table.struct_size = size_of::<ExternalPropagatorV2>();
        owned.table.capabilities &= !EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE;
        let error = unsafe { ExternalPropagator::from_raw(owned.table()) }
            .err()
            .expect("non-thread-safe table must fail");
        assert!(error.to_string().contains("thread-safe"));

        owned.table.capabilities =
            EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE | EXTERNAL_PROPAGATOR_CAP_STM;
        owned.table.propagate_stm = None;
        let error = unsafe { ExternalPropagator::from_raw(owned.table()) }
            .err()
            .expect("declared STM capability needs its callback");
        assert!(error.to_string().contains("STM support"));

        owned.table.capabilities = EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE | (1_u64 << 63);
        let external = unsafe { ExternalPropagator::from_raw(owned.table()) }
            .expect("unknown optional capabilities must be ignored");
        assert_eq!(
            external.capabilities & EXTERNAL_PROPAGATOR_KNOWN_CAPABILITIES,
            EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE
        );
    }

    #[test]
    fn inactive_non_gravitational_rows_preserve_states_and_metadata_across_capsule() {
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_001, 0)]).unwrap();
        let owned = Box::leak(Box::new(two_body_reference::new(
            TwoBodyPropagator::default(),
        )));
        let external = unsafe { ExternalPropagator::from_raw(owned.table()) }.unwrap();

        for (source, a1) in [("source-only", None), ("zero-a1", Some(0.0))] {
            let parameters = NonGravitationalParametersBatch {
                source: vec![Some(source.to_string())],
                a1: vec![a1],
                a2: vec![None],
                a3: vec![None],
                aln: vec![Some(1.0)],
                nk: vec![Some(0.0)],
                nm: vec![Some(2.0)],
                nn: vec![Some(5.093)],
                r0: vec![Some(1.0)],
            };
            assert!(!parameters.has_values());
            let orbits = one_orbit_with_dynamics_inputs(None, Some(parameters.clone()));
            let request = one_day_request(&orbits, &times);
            let via_capsule = external.propagate(&request, &SameScale).unwrap();
            let in_tree = TwoBodyPropagator::default()
                .propagate(&request, &SameScale)
                .unwrap();
            assert_eq!(
                via_capsule.orbits.coordinates.values,
                in_tree.orbits.coordinates.values
            );
            assert_eq!(
                via_capsule.orbits.non_gravitational_parameters,
                Some(parameters.clone())
            );
            assert_eq!(
                in_tree.orbits.non_gravitational_parameters,
                Some(parameters)
            );
        }
    }

    #[test]
    fn non_gravitational_parameters_cross_the_external_boundary_or_fail_closed() {
        let parameters = NonGravitationalParametersBatch {
            source: vec![Some("first".to_string()), Some("second".to_string())],
            a1: vec![Some(1e-10), Some(2e-10)],
            a2: vec![Some(-2e-11), None],
            a3: vec![Some(3e-12), Some(-4e-12)],
            aln: vec![Some(0.1112620426), None],
            nk: vec![Some(4.6142), Some(0.0)],
            nm: vec![Some(2.15), Some(2.0)],
            nn: vec![Some(5.093), Some(5.093)],
            r0: vec![Some(2.808), Some(1.0)],
        };
        let parameter_row = parameters.row(0);
        let orbits = OrbitBatch::new(
            vec![OrbitId("first".to_string()), OrbitId("second".to_string())],
            vec![None, None],
            CoordinateBatch::cartesian(
                vec![
                    [1.1, 0.2, 0.05, -0.003, 0.016, 0.0004],
                    [1.3, -0.1, 0.02, 0.002, 0.014, -0.0002],
                ],
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), 2),
                Some(
                    TimeArray::new(
                        TimeScale::Tdb,
                        vec![Epoch::new(60000, 0), Epoch::new(60000, 0)],
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap(),
        )
        .unwrap()
        .with_physical_parameters(PhysicalParametersBatch {
            h_v: vec![Some(12.1), Some(17.2)],
            h_v_sigma: vec![Some(0.2), None],
            g: vec![Some(0.15), Some(0.25)],
            g_sigma: vec![None, Some(0.03)],
            sigma_eff: vec![Some(0.7), Some(0.8)],
            chi2_red: vec![Some(1.1), Some(2.2)],
        })
        .unwrap()
        .with_non_gravitational_parameters(parameters)
        .unwrap();
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60001, 0)]).unwrap();
        let request = PropagationRequest::new(
            &orbits,
            &times,
            PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
        )
        .unwrap();

        let aware = Box::leak(Box::new(OwnedExternalPropagator::new(
            NonGravAware(TwoBodyPropagator::default()),
            "nongrav-aware",
            "0",
        )));
        let external = unsafe { ExternalPropagator::from_raw(aware.table()) }.unwrap();
        let propagated = external.propagate(&request, &SameScale).unwrap();
        assert_eq!(
            propagated.orbits.physical_parameters,
            orbits.physical_parameters
        );
        assert_eq!(
            propagated.orbits.non_gravitational_parameters,
            orbits.non_gravitational_parameters
        );
        let first_orbit = OrbitBatch {
            orbit_id: vec![orbits.orbit_id[0].clone()],
            object_id: vec![orbits.object_id[0].clone()],
            coordinates: super::super::od::take_coordinate_batch(&orbits.coordinates, &[0])
                .unwrap(),
            physical_parameters: orbits
                .physical_parameters
                .as_ref()
                .map(|parameters| parameters.take(&[0])),
            non_gravitational_parameters: orbits
                .non_gravitational_parameters
                .as_ref()
                .map(|parameters| parameters.take(&[0])),
        };
        let geometry = super::super::od::OrbitGeometry::from_orbit(&first_orbit).unwrap();
        let candidates = super::super::od::candidate_orbit_batch(
            &[orbits.coordinates.values.cartesian().unwrap()[0]],
            &geometry,
        )
        .unwrap();
        assert_eq!(
            candidates.physical_parameters.as_ref().unwrap().h_v,
            vec![Some(12.1)]
        );
        assert_eq!(
            candidates
                .non_gravitational_parameters
                .as_ref()
                .unwrap()
                .row(0)
                .r0,
            Some(2.808)
        );
        external
            .propagate_stm(
                orbits.coordinates.values.cartesian().unwrap()[0],
                Some(parameter_row),
                Epoch::new(60000, 0),
                &[Epoch::new(60001, 0)],
            )
            .unwrap();

        let gravity_only = Box::leak(Box::new(two_body_reference::new(
            TwoBodyPropagator::default(),
        )));
        let external = unsafe { ExternalPropagator::from_raw(gravity_only.table()) }.unwrap();
        let error = external.propagate(&request, &SameScale).unwrap_err();
        assert!(error
            .to_string()
            .contains("does not accept non-gravitational"));

        // Null/zero acceleration with Marsden constants is dynamically
        // inactive, exactly like the in-tree two-body propagator, and does
        // not require the non-gravitational capability.
        let inactive = orbits
            .clone()
            .with_non_gravitational_parameters(NonGravitationalParametersBatch {
                source: vec![Some("inactive".to_string()); 2],
                a1: vec![Some(0.0), None],
                a2: vec![None, Some(0.0)],
                a3: vec![None, None],
                aln: vec![Some(1.0), None],
                nk: vec![Some(0.0), None],
                nm: vec![Some(2.0), None],
                nn: vec![Some(5.093), None],
                r0: vec![Some(1.0), None],
            })
            .unwrap();
        let inactive_request = PropagationRequest::new(
            &inactive,
            &times,
            PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
        )
        .unwrap();
        external.propagate(&inactive_request, &SameScale).unwrap();
    }

    #[test]
    fn safe_consumer_rejects_mismatched_pairwise_lengths() {
        let states = vec![
            [1.1, 0.2, 0.05, -0.003, 0.016, 0.0004],
            [1.3, -0.1, 0.02, 0.002, 0.014, -0.0002],
        ];
        let orbits = OrbitBatch::new(
            vec![OrbitId("first".to_string()), OrbitId("second".to_string())],
            vec![None, None],
            CoordinateBatch::cartesian(
                states,
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), 2),
                Some(
                    TimeArray::new(
                        TimeScale::Tdb,
                        vec![Epoch::new(60_000, 0), Epoch::new(60_000, 0)],
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_001, 0)]).unwrap();
        // Build the already-typed safe request directly to prove the
        // external consumer does not rely only on `PropagationRequest::new`.
        let request = PropagationRequest {
            input: PropagationInput::Orbits(&orbits),
            times: &times,
            options: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::Pairwise,
                covariance: CovariancePropagation::None,
            },
            epoch_order: super::super::EpochOrder::from_times(&times),
        };
        let owned = Box::leak(Box::new(two_body_reference::new(
            TwoBodyPropagator::default(),
        )));
        let external = unsafe { ExternalPropagator::from_raw(owned.table()) }.unwrap();
        let error = external.propagate(&request, &SameScale).unwrap_err();
        assert!(error.to_string().contains("equal orbit/time lengths"));
    }

    #[test]
    fn callback_statuses_distinguish_invalid_backend_and_panic() {
        assert_eq!(
            raw_propagation_status(None, 2),
            EXTERNAL_PROPAGATOR_STATUS_INVALID
        );
        assert_eq!(
            raw_propagation_status(Some(StatusMode::Error), 0),
            EXTERNAL_PROPAGATOR_STATUS_ERROR
        );
        assert_eq!(
            raw_propagation_status(Some(StatusMode::Panic), 0),
            EXTERNAL_PROPAGATOR_STATUS_PANIC
        );
        assert_eq!(
            raw_stm_status(StatusMode::Error, EXTERNAL_PROPAGATOR_ABI_VERSION - 1),
            EXTERNAL_PROPAGATOR_STATUS_INVALID
        );
        assert_eq!(
            raw_stm_status(StatusMode::Error, EXTERNAL_PROPAGATOR_ABI_VERSION),
            EXTERNAL_PROPAGATOR_STATUS_ERROR
        );
        assert_eq!(
            raw_stm_status(StatusMode::Panic, EXTERNAL_PROPAGATOR_ABI_VERSION),
            EXTERNAL_PROPAGATOR_STATUS_PANIC
        );
    }

    #[test]
    fn real_short_allocations_and_misaligned_prefixes_fail_before_wide_reads() {
        let header = Box::new(AbiHeaderV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<AbiHeaderV2>(),
        });
        let error = unsafe {
            ExternalPropagator::from_raw(
                (&*header as *const AbiHeaderV2).cast::<ExternalPropagatorV2>(),
            )
        }
        .err()
        .expect("a header-sized table allocation must be rejected safely");
        assert!(error.to_string().contains("requires the"));

        let owned = two_body_reference::new(TwoBodyPropagator::default());
        let propagate = owned.table.propagate.unwrap();
        let status = unsafe {
            propagate(
                owned.table.context,
                (&*header as *const AbiHeaderV2).cast::<PropagateRequestV2>(),
            )
        };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_INVALID);
        let stm = owned.table.propagate_stm.unwrap();
        let status = unsafe {
            stm(
                owned.table.context,
                (&*header as *const AbiHeaderV2).cast::<PropagateStmRequestV2>(),
            )
        };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_INVALID);

        let bytes = vec![0_u8; size_of::<AbiHeaderV2>() + align_of::<AbiHeaderV2>()];
        let misaligned = unsafe { bytes.as_ptr().add(1) };
        let error =
            unsafe { ExternalPropagator::from_raw(misaligned.cast::<ExternalPropagatorV2>()) }
                .err()
                .expect("a misaligned table header must fail before it is read");
        assert!(error.to_string().contains("misaligned"));
    }

    #[test]
    fn zero_length_misaligned_non_null_buffers_are_never_sliced() {
        let owned = two_body_reference::new(TwoBodyPropagator::default());
        let scale = CString::new("tdb").unwrap();
        let frame = CString::new("ecliptic").unwrap();
        let origin = CString::new("SUN").unwrap();
        let mut error = [0 as c_char; 256];
        let misaligned_const = std::ptr::dangling::<u8>();
        let misaligned_mut = std::ptr::dangling_mut::<u8>();
        let request = PropagateRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<PropagateRequestV2>(),
            n_orbits: 0,
            states: misaligned_const.cast(),
            physical_parameters: misaligned_const.cast(),
            non_gravitational_parameters: misaligned_const.cast(),
            epoch_days: misaligned_const.cast(),
            epoch_nanos: misaligned_const.cast(),
            orbit_time_scale: scale.as_ptr(),
            frame: frame.as_ptr(),
            origin: origin.as_ptr(),
            n_times: 0,
            time_days: misaligned_const.cast(),
            time_nanos: misaligned_const.cast(),
            time_scale: scale.as_ptr(),
            pairwise: 0,
            out_states: misaligned_mut.cast(),
            out_valid: misaligned_mut,
            out_failure_codes: misaligned_mut,
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        let status = unsafe { (owned.table.propagate.unwrap())(owned.table.context, &request) };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_OK, "{:?}", error);

        // An STM request always has one six-component input state, but all
        // target/output arrays are empty when n_times is zero and therefore
        // must not be turned into even zero-length slices at address 1.
        let state = [1.0, 0.0, 0.0, 0.0, 0.017, 0.0];
        let stm_request = PropagateStmRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION,
            struct_size: size_of::<PropagateStmRequestV2>(),
            state: state.as_ptr(),
            non_gravitational_parameters: std::ptr::null(),
            epoch_days: 60_000,
            epoch_nanos: 0,
            n_times: 0,
            time_days: misaligned_const.cast(),
            time_nanos: misaligned_const.cast(),
            out_states: misaligned_mut.cast(),
            out_stm: misaligned_mut.cast(),
            out_valid: misaligned_mut,
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        let status =
            unsafe { (owned.table.propagate_stm.unwrap())(owned.table.context, &stm_request) };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_OK, "{:?}", error);
    }

    #[test]
    fn version_mismatch_never_writes_v2_error_offsets_and_reserved_words_are_zero() {
        let owned = two_body_reference::new(TwoBodyPropagator::default());
        let mut error = [b'X' as c_char; 256];
        let mut request = PropagateRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION - 1,
            struct_size: size_of::<PropagateRequestV2>(),
            n_orbits: 0,
            states: std::ptr::null(),
            physical_parameters: std::ptr::null(),
            non_gravitational_parameters: std::ptr::null(),
            epoch_days: std::ptr::null(),
            epoch_nanos: std::ptr::null(),
            orbit_time_scale: std::ptr::null(),
            frame: std::ptr::null(),
            origin: std::ptr::null(),
            n_times: 0,
            time_days: std::ptr::null(),
            time_nanos: std::ptr::null(),
            time_scale: std::ptr::null(),
            pairwise: 0,
            out_states: std::ptr::null_mut(),
            out_valid: std::ptr::null_mut(),
            out_failure_codes: std::ptr::null_mut(),
            error: error.as_mut_ptr(),
            error_len: error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        let status = unsafe { (owned.table.propagate.unwrap())(owned.table.context, &request) };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_INVALID);
        assert!(error.iter().all(|&value| value == b'X' as c_char));

        let mut stm_error = [b'Y' as c_char; 32];
        let stm_request = PropagateStmRequestV2 {
            abi_version: EXTERNAL_PROPAGATOR_ABI_VERSION - 1,
            struct_size: size_of::<PropagateStmRequestV2>(),
            state: std::ptr::null(),
            non_gravitational_parameters: std::ptr::null(),
            epoch_days: 0,
            epoch_nanos: 0,
            n_times: 0,
            time_days: std::ptr::null(),
            time_nanos: std::ptr::null(),
            out_states: std::ptr::null_mut(),
            out_stm: std::ptr::null_mut(),
            out_valid: std::ptr::null_mut(),
            error: stm_error.as_mut_ptr(),
            error_len: stm_error.len(),
            reserved: [0; EXTERNAL_PROPAGATOR_RESERVED_WORDS],
        };
        let status =
            unsafe { (owned.table.propagate_stm.unwrap())(owned.table.context, &stm_request) };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_INVALID);
        assert!(stm_error.iter().all(|&value| value == b'Y' as c_char));

        request.abi_version = EXTERNAL_PROPAGATOR_ABI_VERSION;
        request.reserved[0] = 1;
        let status = unsafe { (owned.table.propagate.unwrap())(owned.table.context, &request) };
        assert_eq!(status, EXTERNAL_PROPAGATOR_STATUS_INVALID);
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        assert!(message.contains("reserved word 0"), "{message}");
    }

    #[test]
    fn row_failure_code_protocol_is_stable_and_non_finite_outputs_are_classified() {
        assert_eq!(
            decode_row_result(3, 1, false, EXTERNAL_PROPAGATION_FAILURE_NONE).unwrap(),
            (false, Some(PropagationFailureCode::NonFiniteOutputState))
        );
        assert_eq!(
            decode_row_result(
                3,
                0,
                true,
                EXTERNAL_PROPAGATION_FAILURE_SOLVER_MAX_ITERATIONS,
            )
            .unwrap(),
            (false, Some(PropagationFailureCode::SolverMaxIterations))
        );
        assert!(decode_row_result(
            3,
            1,
            true,
            EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_INPUT_STATE,
        )
        .unwrap_err()
        .to_string()
        .contains("valid with failure code"));
        assert!(
            decode_row_result(3, 0, true, EXTERNAL_PROPAGATION_FAILURE_NONE)
                .unwrap_err()
                .to_string()
                .contains("invalid without a failure code")
        );
        assert!(decode_row_result(3, 0, true, u8::MAX)
            .unwrap_err()
            .to_string()
            .contains("unknown row failure code"));
    }

    fn one_orbit_with_dynamics_inputs(
        physical: Option<PhysicalParametersBatch>,
        non_gravitational: Option<NonGravitationalParametersBatch>,
    ) -> OrbitBatch {
        let mut orbits = OrbitBatch::new(
            vec![OrbitId("finite".to_string())],
            vec![None],
            CoordinateBatch::cartesian(
                vec![[1.0, 0.0, 0.0, 0.0, 0.017, 0.0]],
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), 1),
                Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        if let Some(physical) = physical {
            orbits = orbits.with_physical_parameters(physical).unwrap();
        }
        if let Some(non_gravitational) = non_gravitational {
            orbits = orbits
                .with_non_gravitational_parameters(non_gravitational)
                .unwrap();
        }
        orbits
    }

    fn one_day_request<'a>(orbits: &'a OrbitBatch, times: &'a TimeArray) -> PropagationRequest<'a> {
        PropagationRequest::new(
            orbits,
            times,
            PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
        )
        .unwrap()
    }

    #[test]
    fn consumer_rejects_present_non_finite_dynamics_values_before_nan_null_encoding() {
        let owned = Box::leak(Box::new(two_body_reference::new(
            TwoBodyPropagator::default(),
        )));
        let external = unsafe { ExternalPropagator::from_raw(owned.table()) }.unwrap();
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_001, 0)]).unwrap();

        let physical = one_orbit_with_dynamics_inputs(
            Some(PhysicalParametersBatch {
                h_v: vec![Some(f64::INFINITY)],
                h_v_sigma: vec![None],
                g: vec![None],
                g_sigma: vec![None],
                sigma_eff: vec![None],
                chi2_red: vec![None],
            }),
            None,
        );
        let error = external
            .propagate(&one_day_request(&physical, &times), &SameScale)
            .unwrap_err();
        assert!(error.to_string().contains("physical_parameters.h_v[0]"));
        assert!(error.to_string().contains("present but non-finite"));

        let non_gravitational = one_orbit_with_dynamics_inputs(
            None,
            Some(NonGravitationalParametersBatch {
                source: vec![Some("invalid-a1".to_string())],
                a1: vec![Some(f64::NAN)],
                a2: vec![None],
                a3: vec![None],
                aln: vec![Some(1.0)],
                nk: vec![None],
                nm: vec![Some(2.0)],
                nn: vec![Some(5.093)],
                r0: vec![Some(1.0)],
            }),
        );
        let error = external
            .propagate(&one_day_request(&non_gravitational, &times), &SameScale)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("non_gravitational_parameters.a1[0]"));

        let error = external
            .propagate_stm(
                [1.0, 0.0, 0.0, 0.0, 0.017, 0.0],
                Some(
                    non_gravitational
                        .non_gravitational_parameters
                        .as_ref()
                        .unwrap()
                        .row(0),
                ),
                Epoch::new(60_000, 0),
                &[Epoch::new(60_001, 0)],
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("non_gravitational_parameters.a1[0]"));
    }

    #[test]
    fn row_failure_codes_survive_the_external_round_trip() {
        let orbits = OrbitBatch::new(
            vec![OrbitId("invalid".to_string())],
            vec![None],
            CoordinateBatch::cartesian(
                vec![[f64::NAN, 0.0, 0.0, 0.0, 0.01, 0.0]],
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::from_code("SUN"), 1),
                Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let times = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_001, 0)]).unwrap();
        let request = PropagationRequest::new(
            &orbits,
            &times,
            PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
        )
        .unwrap();
        let owned = Box::leak(Box::new(two_body_reference::new(
            TwoBodyPropagator::default(),
        )));
        let external = unsafe { ExternalPropagator::from_raw(owned.table()) }.unwrap();
        let result = external.propagate(&request, &SameScale).unwrap();
        assert_eq!(
            result.diagnostics.convergence[0].failure_code,
            Some(PropagationFailureCode::NonFiniteInputState)
        );
    }

    #[test]
    fn exact_abi_version_and_frozen_prefixes_are_enforced() {
        assert_eq!(
            PROPAGATE_REQUEST_V2_MIN_SIZE,
            std::mem::offset_of!(PropagateRequestV2, reserved)
        );
        assert_eq!(
            PROPAGATE_STM_REQUEST_V2_MIN_SIZE,
            std::mem::offset_of!(PropagateStmRequestV2, reserved)
        );
        assert_eq!(
            EXTERNAL_PROPAGATOR_V2_MIN_SIZE,
            std::mem::offset_of!(ExternalPropagatorV2, propagate_stm)
        );
        let mut owned = two_body_reference::new(TwoBodyPropagator::default());
        owned.table.abi_version = EXTERNAL_PROPAGATOR_ABI_VERSION - 1;
        let error = unsafe { ExternalPropagator::from_raw(owned.table()) }
            .err()
            .expect("wrong ABI version must fail");
        assert!(error.to_string().contains("requires exactly"));
        owned.table.abi_version = EXTERNAL_PROPAGATOR_ABI_VERSION;
        owned.table.struct_size = size_of::<ExternalPropagatorV2>() + 64;
        unsafe { ExternalPropagator::from_raw(owned.table()) }
            .expect("an unknown appended suffix is ignored");
        owned.table.struct_size = EXTERNAL_PROPAGATOR_V2_STM_SIZE - 1;
        let error = unsafe { ExternalPropagator::from_raw(owned.table()) }
            .err()
            .expect("short STM table must fail");
        assert!(error.to_string().contains("does not contain the STM slot"));
    }

    #[test]
    fn request_range_validation_rejects_overflow_and_error_aliasing() {
        let overflow = validate_non_aliasing(&[(usize::MAX - 7, 2, 8, 8, "states")])
            .expect_err("overflowing request range must fail closed");
        assert!(overflow.contains("overflows usize"), "{overflow}");

        let alignment = validate_non_aliasing(&[(0x1001, 1, 8, 8, "states")])
            .expect_err("misaligned typed buffers must fail before slice creation");
        assert!(alignment.contains("not aligned"), "{alignment}");

        let alias = validate_non_aliasing(&[
            (0x1000, 48, 1, 1, "out_states"),
            (0x1020, 64, 1, 1, "error"),
        ])
        .expect_err("the error buffer must not alias an output");
        assert!(alias.contains("out_states and error overlap"), "{alias}");
    }

    #[test]
    fn padded_ids_keep_input_order_under_text_sorting() {
        let width = 12_usize.to_string().len();
        let ids: Vec<String> = (0..12).map(|i| format!("{i:0width$}")).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(sorted, ids);
    }
}
