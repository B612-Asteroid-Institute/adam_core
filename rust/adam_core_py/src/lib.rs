#![allow(
    clippy::useless_conversion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use pyo3::prelude::*;

mod bandpass_vendor;
mod coordinate_ops;
mod coordinates;
mod dynamics;
mod horizons;
mod http;
mod native_benchmarks;
mod od_ops;
mod orbit_determination;
mod photometry;
mod query_clients;
mod rotation_observations;
mod rotation_period;
mod spice;
mod timestamp_ops;
mod variant_ephemeris;

#[pymodule]
fn _rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bandpass_vendor::register(m)?;
    coordinate_ops::register(m)?;
    coordinates::register(m)?;
    dynamics::register(m)?;
    horizons::register(m)?;
    native_benchmarks::register(m)?;
    od_ops::register(m)?;
    photometry::register(m)?;
    query_clients::register(m)?;
    rotation_observations::register(m)?;
    rotation_period::register(m)?;
    orbit_determination::register(m)?;
    spice::register(m)?;
    timestamp_ops::register(m)?;
    variant_ephemeris::register(m)?;
    Ok(())
}
