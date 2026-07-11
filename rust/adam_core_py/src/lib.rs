#![allow(
    clippy::useless_conversion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use pyo3::prelude::*;

mod coordinate_ops;
mod coordinates;
mod dynamics;
mod native_benchmarks;
mod orbit_determination;
mod photometry;
mod spice;
mod timestamp_ops;
mod variant_ephemeris;

#[pymodule]
fn _rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    coordinate_ops::register(m)?;
    coordinates::register(m)?;
    dynamics::register(m)?;
    native_benchmarks::register(m)?;
    photometry::register(m)?;
    orbit_determination::register(m)?;
    spice::register(m)?;
    timestamp_ops::register(m)?;
    variant_ephemeris::register(m)?;
    Ok(())
}
