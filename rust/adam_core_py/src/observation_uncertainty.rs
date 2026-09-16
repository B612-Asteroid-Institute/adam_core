//! PyO3 surface for observation uncertainty models and star-catalog
//! debiasing: the bias-table interpreters, night-batch deweighting, the
//! EFCC18 table/tile/correction kernels and debias model, the VFC2017 sigma
//! table and interpreters, and the ADES angular-covariance kernel behind
//! `OrbitDeterminationObservations.from_ades`. Python keeps the quivr tables,
//! schema validation and column plumbing; each model's arithmetic runs here.

use adam_core_rs_coords::healpix::ang2pix_lonlat;
use adam_core_rs_coords::{
    ades_angular_covariance_flat, compute_efcc18_corrections, efcc18_catalog_columns,
    ra_dec_to_healpix, read_efcc18_bias_version, veres2017_sigma_table, BiasTable, BiasTableRow,
    Efcc18BiasTable, Efcc18DebiasModel, EmpiricalCovarianceMode, EmpiricalCovarianceModel,
    NightBatchDeweightingModel, ObservationUncertaintyModel, OrbitDeterminationAstrometry,
    PerformanceWeightedModel, SigmaFloorModel, VeresFloorModel, VeresReplaceModel,
    VeresSigmaLookup, VeresSigmaRow, EFCC18_N_CATALOGS, EFCC18_N_COMPONENTS, EFCC18_N_TILES,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

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

fn covariance_rows(values: &PyReadonlyArray3<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    let shape = view.shape();
    if shape[1] != 6 || shape[2] != 6 {
        return Err(value_error(format!("{label} must have shape (N, 6, 6)")));
    }
    view.as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| value_error(format!("{label} must be contiguous")))
}

fn covariance_array<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    ndarray::Array3::from_shape_vec((n, 6, 6), flat)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| value_error(err.to_string()))
}

/// Observation columns the models read: positions, covariance, station,
/// band, star catalog and epochs. Columns a model does not consult are
/// passed as empty defaults by the Python veneer.
#[allow(clippy::too_many_arguments)]
fn astrometry(
    lon: &[f64],
    lat: &[f64],
    covariance: Vec<f64>,
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
    astcat: Vec<Option<String>>,
    mjd_utc: Vec<f64>,
    jd_tdb: Vec<f64>,
) -> PyResult<OrbitDeterminationAstrometry> {
    let n = lat.len();
    let fill = |values: Vec<f64>| {
        if values.is_empty() {
            vec![f64::NAN; n]
        } else {
            values
        }
    };
    let observations = OrbitDeterminationAstrometry {
        lon: if lon.is_empty() {
            vec![f64::NAN; n]
        } else {
            lon.to_vec()
        },
        lat: lat.to_vec(),
        covariance,
        obs_code: if obs_code.is_empty() {
            vec![String::new(); n]
        } else {
            obs_code
        },
        band: if band.is_empty() { vec![None; n] } else { band },
        astcat: if astcat.is_empty() {
            vec![None; n]
        } else {
            astcat
        },
        mjd_utc: fill(mjd_utc),
        jd_tdb: fill(jd_tdb),
    };
    observations.validate().map_err(value_error)?;
    Ok(observations)
}

#[allow(clippy::too_many_arguments)]
fn bias_table(
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
    bias_ra_arcsec: PyReadonlyArray1<'_, f64>,
    bias_dec_arcsec: PyReadonlyArray1<'_, f64>,
    resid_var_ra: PyReadonlyArray1<'_, f64>,
    resid_var_dec: PyReadonlyArray1<'_, f64>,
    resid_cov_ra_dec: PyReadonlyArray1<'_, f64>,
    resid_cov_n: PyReadonlyArray1<'_, f64>,
    chi2_per_obs: PyReadonlyArray1<'_, f64>,
) -> PyResult<BiasTable> {
    let columns = [
        scalars(&bias_ra_arcsec, "bias_ra_arcsec")?,
        scalars(&bias_dec_arcsec, "bias_dec_arcsec")?,
        scalars(&resid_var_ra, "resid_var_ra")?,
        scalars(&resid_var_dec, "resid_var_dec")?,
        scalars(&resid_cov_ra_dec, "resid_cov_ra_dec")?,
        scalars(&resid_cov_n, "resid_cov_n")?,
        scalars(&chi2_per_obs, "chi2_per_obs")?,
    ];
    let rows = obs_code.len();
    if columns.iter().any(|column| column.len() != rows) || band.len() != rows {
        return Err(value_error("bias table columns must align"));
    }
    let rows: Vec<BiasTableRow> = (0..rows)
        .map(|row| BiasTableRow {
            bias_ra_arcsec: columns[0][row],
            bias_dec_arcsec: columns[1][row],
            resid_var_ra: columns[2][row],
            resid_var_dec: columns[3][row],
            resid_cov_ra_dec: columns[4][row],
            resid_cov_n: columns[5][row],
            chi2_per_obs: columns[6][row],
        })
        .collect();
    BiasTable::new(obs_code, band, rows).map_err(value_error)
}

type CovarianceUpdate<'py> = Option<Bound<'py, PyArray3<f64>>>;

/// Apply a bias-table interpreter (`empirical_covariance`,
/// `performance_weighted`, `sigma_floor`) to the observations' covariance
/// block. Returns the updated `(N, 6, 6)` covariance, or `None` when no entry
/// changed so the caller keeps its input table.
#[pyfunction]
#[pyo3(signature = (model, mode, min_resid_cov_n, table_obs_code, table_band, bias_ra_arcsec, bias_dec_arcsec, resid_var_ra, resid_var_dec, resid_cov_ra_dec, resid_cov_n, chi2_per_obs, lat, covariances, obs_code, band))]
#[allow(clippy::too_many_arguments)]
fn bias_table_model_apply_numpy<'py>(
    py: Python<'py>,
    model: &str,
    mode: &str,
    min_resid_cov_n: f64,
    table_obs_code: Vec<String>,
    table_band: Vec<Option<String>>,
    bias_ra_arcsec: PyReadonlyArray1<'py, f64>,
    bias_dec_arcsec: PyReadonlyArray1<'py, f64>,
    resid_var_ra: PyReadonlyArray1<'py, f64>,
    resid_var_dec: PyReadonlyArray1<'py, f64>,
    resid_cov_ra_dec: PyReadonlyArray1<'py, f64>,
    resid_cov_n: PyReadonlyArray1<'py, f64>,
    chi2_per_obs: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
) -> PyResult<CovarianceUpdate<'py>> {
    let table = bias_table(
        table_obs_code,
        table_band,
        bias_ra_arcsec,
        bias_dec_arcsec,
        resid_var_ra,
        resid_var_dec,
        resid_cov_ra_dec,
        resid_cov_n,
        chi2_per_obs,
    )?;
    let lat = scalars(&lat, "lat")?;
    let covariance = covariance_rows(&covariances, "covariances")?;
    let mut observations = astrometry(
        &[],
        &lat,
        covariance,
        obs_code,
        band,
        vec![],
        vec![],
        vec![],
    )?;
    let model: Box<dyn ObservationUncertaintyModel> = match model {
        "empirical_covariance" => Box::new(EmpiricalCovarianceModel {
            table,
            mode: EmpiricalCovarianceMode::parse(mode).map_err(value_error)?,
            min_resid_cov_n,
        }),
        "performance_weighted" => Box::new(PerformanceWeightedModel {
            table,
            min_resid_cov_n,
        }),
        "sigma_floor" => Box::new(SigmaFloorModel {
            table,
            min_resid_cov_n,
        }),
        _ => return Err(value_error(
            "model must be one of {'empirical_covariance', 'performance_weighted', 'sigma_floor'}",
        )),
    };
    let changed = model.apply(&mut observations).map_err(value_error)?;
    if !changed {
        return Ok(None);
    }
    let n = observations.len();
    Ok(Some(covariance_array(py, observations.covariance, n)?))
}

/// Night-batch deweighting (Veres, Farnocchia & Chesley 2017): scale the
/// RA/Dec covariance block of same-station, same-UTC-night batches larger
/// than `cap` by `N / cap`. `None` when every batch is within the cap.
#[pyfunction]
fn night_batch_deweighting_model_apply_numpy<'py>(
    py: Python<'py>,
    cap: usize,
    covariances: PyReadonlyArray3<'py, f64>,
    obs_code: Vec<String>,
    mjd_utc: PyReadonlyArray1<'py, f64>,
) -> PyResult<CovarianceUpdate<'py>> {
    let model = NightBatchDeweightingModel::new(cap).map_err(value_error)?;
    let mjd_utc = scalars(&mjd_utc, "mjd_utc")?;
    let covariance = covariance_rows(&covariances, "covariances")?;
    let lat = vec![f64::NAN; mjd_utc.len()];
    let mut observations = astrometry(
        &[],
        &lat,
        covariance,
        obs_code,
        vec![],
        vec![],
        mjd_utc,
        vec![],
    )?;
    let changed = model.apply(&mut observations).map_err(value_error)?;
    if !changed {
        return Ok(None);
    }
    let n = observations.len();
    Ok(Some(covariance_array(py, observations.covariance, n)?))
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

type PositionUpdate<'py> = Option<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>;

/// Subtract the EFCC18 star-catalog bias from observed positions (this model
/// modifies positions by design). Returns `(lon, lat)` in degrees, or `None`
/// when no covered observation moved.
#[pyfunction]
fn efcc18_debias_model_apply_numpy<'py>(
    py: Python<'py>,
    bias_table: PyReadonlyArray3<'py, f32>,
    exclude_astcats: Vec<String>,
    lon: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    astcat: Vec<Option<String>>,
    jd_tdb: PyReadonlyArray1<'py, f64>,
) -> PyResult<PositionUpdate<'py>> {
    let model = Efcc18DebiasModel {
        bias_table: Arc::new(efcc18_table(&bias_table)?),
        exclude_astcats,
    };
    let lon = scalars(&lon, "lon")?;
    let lat = scalars(&lat, "lat")?;
    let jd_tdb = scalars(&jd_tdb, "jd_tdb")?;
    let covariance = vec![f64::NAN; lat.len() * 36];
    let mut observations = astrometry(
        &lon,
        &lat,
        covariance,
        vec![],
        vec![],
        astcat,
        vec![],
        jd_tdb,
    )?;
    let changed = model.apply(&mut observations).map_err(value_error)?;
    if !changed {
        return Ok(None);
    }
    Ok(Some((
        observations.lon.into_pyarray(py),
        observations.lat.into_pyarray(py),
    )))
}

/// EFCC18 HEALPix tile index (`N_side = 64`, RING) for (RA, Dec) in degrees.
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

/// healpy `ang2pix(nside, lon, lat, nest=..., lonlat=True)` for arrays.
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

/// Parse the text of `bias.dat` into the `(49152, 26, 4)` float32 table.
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
    .map_err(|err| value_error(err.to_string()))
}

/// `BIAS_VERSION` tag from the `!` header of a `bias.dat` text, if present.
#[pyfunction]
fn efcc18_read_bias_version(text: &str) -> Option<String> {
    read_efcc18_bias_version(text)
}

/// EFCC18 column index per observation, or -1 where not covered.
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

/// Per-observation EFCC18 corrections `(N, 2)`: `(bias_ra_arcsec,
/// bias_dec_arcsec)` in the cos(dec)-corrected frame.
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
        .map_err(|err| value_error(err.to_string()))
}

type SigmaTableColumns = (Vec<Option<String>>, Vec<String>, Vec<f64>, Vec<f64>);

/// The bundled VFC2017-style sigma table as `(obs_code, astcat,
/// sigma_ra_arcsec, sigma_dec_arcsec)` columns.
#[pyfunction]
fn veres2017_sigma_table_columns() -> SigmaTableColumns {
    let rows = veres2017_sigma_table();
    (
        rows.iter().map(|row| row.obs_code.clone()).collect(),
        rows.iter().map(|row| row.astcat.clone()).collect(),
        rows.iter().map(|row| row.sigma_ra_arcsec).collect(),
        rows.iter().map(|row| row.sigma_dec_arcsec).collect(),
    )
}

fn veres_lookup(
    table_obs_code: Vec<Option<String>>,
    table_astcat: Vec<String>,
    sigma_ra_arcsec: &PyReadonlyArray1<'_, f64>,
    sigma_dec_arcsec: &PyReadonlyArray1<'_, f64>,
    fallback_sigma_arcsec: Option<f64>,
) -> PyResult<VeresSigmaLookup> {
    let sigma_ra = scalars(sigma_ra_arcsec, "sigma_ra_arcsec")?;
    let sigma_dec = scalars(sigma_dec_arcsec, "sigma_dec_arcsec")?;
    let rows = table_astcat.len();
    if table_obs_code.len() != rows || sigma_ra.len() != rows || sigma_dec.len() != rows {
        return Err(value_error("sigma table columns must align"));
    }
    let rows: Vec<VeresSigmaRow> = table_obs_code
        .into_iter()
        .zip(table_astcat)
        .zip(sigma_ra.into_iter().zip(sigma_dec))
        .map(
            |((obs_code, astcat), (sigma_ra_arcsec, sigma_dec_arcsec))| VeresSigmaRow {
                obs_code,
                astcat,
                sigma_ra_arcsec,
                sigma_dec_arcsec,
            },
        )
        .collect();
    VeresSigmaLookup::new(&rows, fallback_sigma_arcsec).map_err(value_error)
}

/// Resolve `(obs_code, astcat)` pairs through a VFC2017 sigma table:
/// `(sigma_ra_arcsec, sigma_dec_arcsec)` per observation, NaN where no sigma
/// is known (unknown catalog without a fallback).
#[pyfunction]
#[pyo3(signature = (table_obs_code, table_astcat, sigma_ra_arcsec, sigma_dec_arcsec, fallback_sigma_arcsec, obs_code, astcat))]
fn veres_sigma_lookup_numpy<'py>(
    py: Python<'py>,
    table_obs_code: Vec<Option<String>>,
    table_astcat: Vec<String>,
    sigma_ra_arcsec: PyReadonlyArray1<'py, f64>,
    sigma_dec_arcsec: PyReadonlyArray1<'py, f64>,
    fallback_sigma_arcsec: Option<f64>,
    obs_code: Vec<Option<String>>,
    astcat: Vec<Option<String>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let lookup = veres_lookup(
        table_obs_code,
        table_astcat,
        &sigma_ra_arcsec,
        &sigma_dec_arcsec,
        fallback_sigma_arcsec,
    )?;
    if obs_code.len() != astcat.len() {
        return Err(value_error("obs_code and astcat must have equal length"));
    }
    let n = obs_code.len();
    let mut flat = Vec::with_capacity(n * 2);
    for (code, catalog) in obs_code.iter().zip(&astcat) {
        match lookup.sigmas(code.as_deref(), catalog.as_deref()) {
            Some((ra, dec)) => flat.extend_from_slice(&[ra, dec]),
            None => flat.extend_from_slice(&[f64::NAN, f64::NAN]),
        }
    }
    ndarray::Array2::from_shape_vec((n, 2), flat)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| value_error(err.to_string()))
}

/// Apply a VFC2017 sigma interpreter (`floor` or `replace`) to the
/// observations' covariance block. `None` when no entry changed.
#[pyfunction]
#[pyo3(signature = (model, fill_missing, table_obs_code, table_astcat, sigma_ra_arcsec, sigma_dec_arcsec, fallback_sigma_arcsec, lat, covariances, obs_code, astcat))]
#[allow(clippy::too_many_arguments)]
fn veres_model_apply_numpy<'py>(
    py: Python<'py>,
    model: &str,
    fill_missing: bool,
    table_obs_code: Vec<Option<String>>,
    table_astcat: Vec<String>,
    sigma_ra_arcsec: PyReadonlyArray1<'py, f64>,
    sigma_dec_arcsec: PyReadonlyArray1<'py, f64>,
    fallback_sigma_arcsec: Option<f64>,
    lat: PyReadonlyArray1<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    obs_code: Vec<String>,
    astcat: Vec<Option<String>>,
) -> PyResult<CovarianceUpdate<'py>> {
    let lookup = veres_lookup(
        table_obs_code,
        table_astcat,
        &sigma_ra_arcsec,
        &sigma_dec_arcsec,
        fallback_sigma_arcsec,
    )?;
    let lat = scalars(&lat, "lat")?;
    let covariance = covariance_rows(&covariances, "covariances")?;
    let mut observations = astrometry(
        &[],
        &lat,
        covariance,
        obs_code,
        vec![],
        astcat,
        vec![],
        vec![],
    )?;
    let model: Box<dyn ObservationUncertaintyModel> = match model {
        "floor" => Box::new(VeresFloorModel {
            lookup,
            fill_missing,
        }),
        "replace" => Box::new(VeresReplaceModel { lookup }),
        _ => return Err(value_error("model must be one of {'floor', 'replace'}")),
    };
    let changed = model.apply(&mut observations).map_err(value_error)?;
    if !changed {
        return Ok(None);
    }
    let n = observations.len();
    Ok(Some(covariance_array(py, observations.covariance, n)?))
}

/// `(N, 6, 6)` spherical covariance from ADES `rmsRACosDec` / `rmsDec` /
/// `rmsCorr` (NaN for nulls), see `OrbitDeterminationObservations.from_ades`.
#[pyfunction]
fn ades_angular_covariance_numpy<'py>(
    py: Python<'py>,
    dec_deg: PyReadonlyArray1<'py, f64>,
    rms_ra_cos_dec: PyReadonlyArray1<'py, f64>,
    rms_dec: PyReadonlyArray1<'py, f64>,
    rms_corr: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let dec = scalars(&dec_deg, "dec")?;
    let flat = ades_angular_covariance_flat(
        &dec,
        &scalars(&rms_ra_cos_dec, "rmsRACosDec")?,
        &scalars(&rms_dec, "rmsDec")?,
        &scalars(&rms_corr, "rmsCorr")?,
    )
    .map_err(|err| value_error(err.to_string()))?;
    covariance_array(py, flat, dec.len())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bias_table_model_apply_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        night_batch_deweighting_model_apply_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(efcc18_debias_model_apply_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_ra_dec_to_healpix_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(healpix_ang2pix_lonlat_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_parse_bias_dat, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_read_bias_version, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_catalog_columns_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(efcc18_corrections_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(veres2017_sigma_table_columns, m)?)?;
    m.add_function(wrap_pyfunction!(veres_sigma_lookup_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(veres_model_apply_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ades_angular_covariance_numpy, m)?)?;
    Ok(())
}
