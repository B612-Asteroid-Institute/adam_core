//! Public Rust entrypoint for adam-core.
//!
//! Domain crates remain available independently for consumers that need a
//! smaller dependency graph. This umbrella crate provides stable module names
//! and re-exports the same reviewed implementations used by the Python wheel.

pub use adam_core_rs_autodiff as autodiff;
pub use adam_core_rs_coords as coords;
pub use adam_core_rs_kernel_data as kernel_data;
pub use adam_core_rs_orbit_determination as orbit_determination;
pub use adam_core_rs_spice as spice;

/// Compile-time identity of the `adam_core` crate for artifact diagnostics.
///
/// Scientific orbit and OD provenance deliberately does not store or validate
/// this value. Release tooling and benchmark evidence may report it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildIdentity {
    /// Compiled `adam_core` crate version.
    pub version: &'static str,
    /// Raw source identity embedded by the crate build.
    pub commit: &'static str,
}

/// Return the raw version and source identity embedded in this compiled crate.
pub const fn build_identity() -> BuildIdentity {
    BuildIdentity {
        version: env!("CARGO_PKG_VERSION"),
        commit: env!("ADAM_CORE_BUILD_COMMIT"),
    }
}

/// Canonical, batch-native orbit-determination workflows.
pub mod od;
pub use od::{evaluate_orbits, refine, run_iod, run_od};

pub use adam_core_rs_coords::{
    estimate_rotation_period, estimate_rotation_period_best_apparition,
    estimate_rotation_period_grouped, RotationPeriodConfig, RotationPeriodEstimate,
    RotationPeriodInput,
};
