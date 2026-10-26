#![allow(
    clippy::useless_conversion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use pyo3::prelude::*;

mod bandpass_vendor;
mod coordinate_ops;
mod coordinates;
mod differential_correction;
mod dynamics;
mod efcc18;
mod horizons;
mod http;
mod native_benchmarks;
mod obs80;
mod od_config;
mod od_drivers;
mod od_ops;
mod od_support;
mod orbit_determination;
mod photometry;
mod query_clients;
mod rotation_observations;
mod rotation_period;
mod spice;
mod timestamp_ops;
mod trajectory;
mod variant_ephemeris;

#[pyfunction]
fn build_identity() -> (&'static str, &'static str) {
    (env!("CARGO_PKG_VERSION"), env!("ADAM_CORE_BUILD_COMMIT"))
}

/// Compile-time Cargo profile metadata for benchmark/release evidence.
#[pyfunction]
fn native_build_profile() -> (&'static str, &'static str, &'static str, bool) {
    (
        env!("ADAM_CORE_CARGO_PROFILE"),
        env!("ADAM_CORE_CARGO_OPT_LEVEL"),
        env!("ADAM_CORE_CARGO_TARGET"),
        cfg!(debug_assertions),
    )
}

#[pymodule]
fn _rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_identity, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_profile, m)?)?;
    bandpass_vendor::register(m)?;
    coordinate_ops::register(m)?;
    coordinates::register(m)?;
    differential_correction::register(m)?;
    dynamics::register(m)?;
    efcc18::register(m)?;
    horizons::register(m)?;
    native_benchmarks::register(m)?;
    obs80::register(m)?;
    od_config::register(m)?;
    od_drivers::register(m)?;
    od_ops::register(m)?;
    photometry::register(m)?;
    query_clients::register(m)?;
    rotation_observations::register(m)?;
    rotation_period::register(m)?;
    orbit_determination::register(m)?;
    spice::register(m)?;
    timestamp_ops::register(m)?;
    trajectory::register(m)?;
    variant_ephemeris::register(m)?;
    Ok(())
}
