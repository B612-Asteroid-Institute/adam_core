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

pub use adam_core_rs_coords::{
    estimate_rotation_period, estimate_rotation_period_best_apparition,
    estimate_rotation_period_grouped, RotationPeriodConfig, RotationPeriodEstimate,
    RotationPeriodInput,
};
