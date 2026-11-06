//! Stable PyO3 kernels for the public `adam_core.observations.efcc18` API.
//!
//! These are table parsing and arithmetic utilities, not executable OD model
//! callbacks. Configured OD binds and applies EFCC18 through the closed Rust
//! `ObservationTransform` enum.

use adam_core_rs_coords::healpix::ang2pix_lonlat;
use adam_core_rs_coords::{
    compute_efcc18_corrections, efcc18_catalog_columns, ra_dec_to_healpix,
    read_efcc18_bias_version, Efcc18BiasTable, EFCC18_N_CATALOGS, EFCC18_N_COMPONENTS,
    EFCC18_N_TILES,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn scalars(values: &PyReadonlyArray1<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    values
        .as_array()
        .as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| value_error(format!("{label} must be contiguous")))
}

fn efcc18_table(values: &PyReadonlyArray3<'_, f32>) -> PyResult<Efcc18BiasTable> {
    let view = values.as_array();
    let shape = view.shape();
    if shape != [EFCC18_N_TILES, EFCC18_N_CATALOGS, EFCC18_N_COMPONENTS] {
        return Err(value_error(format!(
            "bias_table has shape ({}, {}, {}), expected ({EFCC18_N_TILES}, {EFCC18_N_CATALOGS}, {EFCC18_N_COMPONENTS})",
            shape[0], shape[1], shape[2]
        )));
    }
    let flat = view
        .as_slice()
        .map(<[f32]>::to_vec)
        .ok_or_else(|| value_error("bias_table must be contiguous"))?;
    Efcc18BiasTable::from_values(flat).map_err(value_error)
}

#[pyfunction]
fn efcc18_ra_dec_to_healpix_numpy<'py>(
    py: Python<'py>,
    ra_deg: PyReadonlyArray1<'py, f64>,
    dec_deg: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ra = scalars(&ra_deg, "ra_deg")?;
    let dec = scalars(&dec_deg, "dec_deg")?;
    if ra.len() != dec.len() {
        return Err(value_error("ra_deg and dec_deg must have equal length"));
    }
    let tiles = ra
        .iter()
        .zip(&dec)
        .map(|(&ra, &dec)| ra_dec_to_healpix(ra, dec).map_err(value_error))
        .collect::<PyResult<Vec<i64>>>()?;
    Ok(tiles.into_pyarray(py))
}

#[pyfunction]
fn healpix_ang2pix_lonlat_numpy<'py>(
    py: Python<'py>,
    nside: i64,
    lon_deg: PyReadonlyArray1<'py, f64>,
    lat_deg: PyReadonlyArray1<'py, f64>,
    nest: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let lon = scalars(&lon_deg, "lon_deg")?;
    let lat = scalars(&lat_deg, "lat_deg")?;
    if lon.len() != lat.len() {
        return Err(value_error("lon_deg and lat_deg must have equal length"));
    }
    let pixels = lon
        .iter()
        .zip(&lat)
        .map(|(&lon, &lat)| ang2pix_lonlat(nside, lon, lat, nest).map_err(value_error))
        .collect::<PyResult<Vec<i64>>>()?;
    Ok(pixels.into_pyarray(py))
}

#[pyfunction]
fn efcc18_parse_bias_dat<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let table = py
        .allow_threads(|| Efcc18BiasTable::parse_bias_dat(text))
        .map_err(value_error)?;
    ndarray::Array3::from_shape_vec(
        (EFCC18_N_TILES, EFCC18_N_CATALOGS, EFCC18_N_COMPONENTS),
        table.values().to_vec(),
    )
    .map(|array| array.into_pyarray(py))
    .map_err(|error| value_error(error.to_string()))
}

#[pyfunction]
fn efcc18_read_bias_version(text: &str) -> Option<String> {
    read_efcc18_bias_version(text)
}

#[pyfunction]
fn efcc18_catalog_columns_numpy<'py>(
    py: Python<'py>,
    astcats: Vec<Option<String>>,
    exclude_astcats: Vec<String>,
) -> Bound<'py, PyArray1<i64>> {
    let astcats: Vec<Option<&str>> = astcats.iter().map(Option::as_deref).collect();
    let exclude: Vec<&str> = exclude_astcats.iter().map(String::as_str).collect();
    efcc18_catalog_columns(&astcats, &exclude).into_pyarray(py)
}

#[pyfunction]
fn efcc18_corrections_numpy<'py>(
    py: Python<'py>,
    ra_deg: PyReadonlyArray1<'py, f64>,
    dec_deg: PyReadonlyArray1<'py, f64>,
    astcats: Vec<Option<String>>,
    jd_tdb: PyReadonlyArray1<'py, f64>,
    bias_table: PyReadonlyArray3<'py, f32>,
    exclude_astcats: Vec<String>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ra = scalars(&ra_deg, "ra_deg")?;
    let dec = scalars(&dec_deg, "dec_deg")?;
    let jd = scalars(&jd_tdb, "jd_tdb")?;
    let table = efcc18_table(&bias_table)?;
    let astcats: Vec<Option<&str>> = astcats.iter().map(Option::as_deref).collect();
    let exclude: Vec<&str> = exclude_astcats.iter().map(String::as_str).collect();
    let corrections = compute_efcc18_corrections(&ra, &dec, &astcats, &jd, &table, &exclude)
        .map_err(value_error)?;
    let n = corrections.len();
    let flat: Vec<f64> = corrections.into_iter().flatten().collect();
    ndarray::Array2::from_shape_vec((n, 2), flat)
        .map(|array| array.into_pyarray(py))
        .map_err(|error| value_error(error.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(efcc18_ra_dec_to_healpix_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(healpix_ang2pix_lonlat_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_parse_bias_dat, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_read_bias_version, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_catalog_columns_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_corrections_numpy, m)?)?;
    Ok(())
}
