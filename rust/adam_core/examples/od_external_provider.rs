//! Canonical ownership boundary for a standalone ABI-v2 OD provider.
//!
//! This example intentionally does not fabricate an `ExternalPropagatorV2`
//! table. A real provider owns and initializes that table, every pointed-to
//! string, every callback, and the callback context in stable memory.

use adam_core::coords::propagation::{ExternalPropagator, ExternalPropagatorV2};
use adam_core::od::{AssociatedObservations, OdBatchError, OdBatchResult, OdConfig, OdPropagator};

/// Validate an owner-supplied ABI-v2 table and run the canonical OD workflow.
///
/// # Safety
///
/// `table` must be non-null, readable, properly aligned, and initialized for
/// its advertised `struct_size`. Its NUL-terminated identity and fingerprint
/// strings, callback functions, and callback context must remain alive and
/// immutable longer than `external` and the workflow call below. Callbacks must
/// obey every ABI-v2 buffer rule, support concurrent calls when advertised,
/// retain no request pointers, and never unwind across the C ABI. The owner
/// must not move any address-sensitive table or context storage while borrowed.
unsafe fn run_owned_provider(
    table: *const ExternalPropagatorV2,
    input: &AssociatedObservations,
    config: &OdConfig,
) -> Result<OdBatchResult, OdBatchError> {
    // SAFETY: this function's caller upholds the complete table, pointee,
    // callback, concurrency, and lifetime contract stated above.
    let external = unsafe { ExternalPropagator::from_raw(table) }?;
    let provider = OdPropagator::external(&external);
    adam_core::run_od(input, config, &provider)
}

fn main() {
    // Provider packages call `run_owned_provider` only while their stable
    // table/context owner is alive. Keeping the function item type-checked
    // proves the public ABI-v2 -> OdPropagator -> canonical workflow route.
    let _ = run_owned_provider;
}
