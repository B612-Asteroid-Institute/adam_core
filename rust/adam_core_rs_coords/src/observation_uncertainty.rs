//! Observation uncertainty models for orbit determination.
//!
//! Rust-canonical port of `adam_core.orbit_determination.observation_uncertainty`:
//! the [`ObservationUncertaintyModel`] interface plus the interpreters that
//! turn an observatory bias table into inflated observation covariances
//! ([`EmpiricalCovarianceModel`], [`PerformanceWeightedModel`],
//! [`SigmaFloorModel`]), the bias-table-free
//! [`NightBatchDeweightingModel`], the position-modifying
//! [`Efcc18DebiasModel`], and [`CompositeModel`]. The Python classes of the
//! same names are thin veneers over these kernels; quivr schema validation of
//! the bias table (`BIAS_TABLE_SCHEMA`) stays at the pyarrow boundary.
//!
//! Frames and units
//! ----------------
//! Bias-table angular quantities (biases, RMS values and residual variances /
//! covariances) are in arcseconds (arcsec² for variances) with the RA axis in
//! the cos(dec)-corrected frame (MPC/ADES `rmsRA` convention).
//! `SphericalCoordinates` covariances are in degrees² with lon (RA) NOT
//! cos(dec)-corrected, so a bias-table quantity converts onto the lon axis by
//! dividing by cos(dec) once per RA factor: variances by cos²(dec), the RA×Dec
//! cross-covariance by cos(dec) once, 1-sigma values by cos(dec) once (on top
//! of the arcsec → degree scaling).
//!
//! Position rules
//! --------------
//! Every bias-table model and the night-batch model modify ONLY the RA/Dec
//! covariance block (row-major 6×6 entries `[1,1]`, `[2,2]`, `[1,2]`, `[2,1]`);
//! observed positions pass through unchanged. `Efcc18DebiasModel` is the
//! deliberate exception: it SUBTRACTS the EFCC18 star-catalog bias from the
//! observed lon/lat and leaves the covariance untouched (standard astrometric
//! debiasing, applied by JPL, the MPC and OrbFit alike). Models operate on an
//! [`OrbitDeterminationAstrometry`] column view and report whether anything
//! changed so the Python veneer can hand back its input object untouched.

use crate::efcc18::{
    compute_efcc18_corrections, efcc18_debias_positions, is_efcc18_covered, Efcc18BiasTable,
};
use std::collections::HashMap;
use std::sync::Arc;

pub const ARCSEC_PER_DEG: f64 = 3600.0;
const ARCSEC_PER_DEG_SQ: f64 = ARCSEC_PER_DEG * ARCSEC_PER_DEG;

/// Row-major 6×6 covariance offsets of the (lon, lat) block.
const VAR_LON: usize = 7;
const VAR_LAT: usize = 14;
const COV_LON_LAT: usize = 8;
const COV_LAT_LON: usize = 13;

/// Column view of `OrbitDeterminationObservations` that uncertainty models
/// read or modify: positions (degrees), the `(N, 36)` row-major spherical
/// covariance (degrees²), station code, photometric band, star catalog, and
/// the observation epochs as UTC MJD (night batching) and TDB JD (EFCC18
/// proper-motion term).
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitDeterminationAstrometry {
    pub lon: Vec<f64>,
    pub lat: Vec<f64>,
    pub covariance: Vec<f64>,
    pub obs_code: Vec<String>,
    pub band: Vec<Option<String>>,
    pub astcat: Vec<Option<String>>,
    pub mjd_utc: Vec<f64>,
    pub jd_tdb: Vec<f64>,
}

impl OrbitDeterminationAstrometry {
    pub fn len(&self) -> usize {
        self.lon.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lon.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.len();
        if self.lat.len() != n
            || self.obs_code.len() != n
            || self.band.len() != n
            || self.astcat.len() != n
            || self.mjd_utc.len() != n
            || self.jd_tdb.len() != n
        {
            return Err("observation columns must have equal length".to_string());
        }
        if self.covariance.len() != n * 36 {
            return Err(format!(
                "covariance must have shape (N, 6, 6): got {} values for {n} observations",
                self.covariance.len()
            ));
        }
        Ok(())
    }

    fn cos_dec(&self, row: usize) -> f64 {
        self.lat[row].to_radians().cos()
    }

    fn astcats(&self) -> Vec<Option<&str>> {
        self.astcat.iter().map(Option::as_deref).collect()
    }
}

/// Models that transform observation uncertainties (or, for EFCC18,
/// positions) before orbit fitting. `apply` edits in place and returns
/// whether any value changed.
pub trait ObservationUncertaintyModel: Send + Sync {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String>;
}

/// No-op model: observations are returned unchanged (the naive baseline).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityModel;

impl ObservationUncertaintyModel for IdentityModel {
    fn apply(&self, _observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        Ok(false)
    }
}

/// Float-valued bias-table columns cached per row for fast lookup. Null
/// entries are carried as NaN, matching the Python interpreters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasTableRow {
    pub bias_ra_arcsec: f64,
    pub bias_dec_arcsec: f64,
    pub resid_var_ra: f64,
    pub resid_var_dec: f64,
    pub resid_cov_ra_dec: f64,
    pub resid_cov_n: f64,
    pub chi2_per_obs: f64,
}

/// Observatory bias table keyed by `(obs_code, band)`; a station+band lookup
/// falls back to the station rollup row (`band` null) when no band-specific
/// row exists.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasTable {
    rows: HashMap<(String, Option<String>), BiasTableRow>,
}

impl BiasTable {
    pub fn new(
        obs_code: Vec<String>,
        band: Vec<Option<String>>,
        rows: Vec<BiasTableRow>,
    ) -> Result<Self, String> {
        if obs_code.len() != band.len() || obs_code.len() != rows.len() {
            return Err("bias table columns must have equal length".to_string());
        }
        let mut table = HashMap::with_capacity(rows.len());
        for ((code, band), row) in obs_code.into_iter().zip(band).zip(rows) {
            let key = (code, band);
            if table.contains_key(&key) {
                return Err(format!(
                    "Bias table contains duplicate rows for (obs_code, band) = ({:?}, {:?})",
                    key.0, key.1
                ));
            }
            table.insert(key, row);
        }
        Ok(Self { rows: table })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn lookup(&self, code: &str, band: Option<&str>) -> Option<&BiasTableRow> {
        if let Some(band) = band {
            if let Some(row) = self.rows.get(&(code.to_string(), Some(band.to_string()))) {
                return Some(row);
            }
        }
        self.rows.get(&(code.to_string(), None))
    }
}

/// Updated `(var_lon, var_lat, cov_lonlat)` block for one observation, in
/// deg² with lon NOT cos(dec)-corrected; `None` passes the row through.
type BlockUpdate = Option<(f64, f64, f64)>;

fn apply_bias_table_model(
    table: &BiasTable,
    min_resid_cov_n: f64,
    observations: &mut OrbitDeterminationAstrometry,
    updated_block: impl Fn(&BiasTableRow, f64, f64, f64, f64) -> BlockUpdate,
) -> Result<bool, String> {
    observations.validate()?;
    let mut changed = false;
    for row in 0..observations.len() {
        let Some(bias) = table.lookup(
            &observations.obs_code[row],
            observations.band[row].as_deref(),
        ) else {
            continue;
        };
        if !bias.resid_cov_n.is_finite() || bias.resid_cov_n < min_resid_cov_n {
            continue;
        }
        let offset = row * 36;
        let block = &observations.covariance[offset..offset + 36];
        let Some((var_lon, var_lat, cov_lonlat)) = updated_block(
            bias,
            block[VAR_LON],
            block[VAR_LAT],
            block[COV_LON_LAT],
            observations.cos_dec(row),
        ) else {
            continue;
        };
        let block = &mut observations.covariance[offset..offset + 36];
        changed |= set_block(block, var_lon, var_lat, cov_lonlat);
    }
    Ok(changed)
}

/// Write the (lon, lat) block and report whether any entry changed (NaN
/// equal to NaN, as `np.array_equal(..., equal_nan=True)`).
fn set_block(block: &mut [f64], var_lon: f64, var_lat: f64, cov_lonlat: f64) -> bool {
    let mut changed = false;
    for (index, value) in [
        (VAR_LON, var_lon),
        (VAR_LAT, var_lat),
        (COV_LON_LAT, cov_lonlat),
        (COV_LAT_LON, cov_lonlat),
    ] {
        if !same_value(block[index], value) {
            changed = true;
        }
        block[index] = value;
    }
    changed
}

fn same_value(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// How [`EmpiricalCovarianceModel`] combines the measured residual covariance
/// with the reported one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalCovarianceMode {
    Add,
    Replace,
}

impl EmpiricalCovarianceMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "add" => Ok(Self::Add),
            "replace" => Ok(Self::Replace),
            _ => Err(format!("mode must be 'add' or 'replace', got {value:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Replace => "replace",
        }
    }
}

/// Inflate the RA/Dec covariance block with the station's measured 2×2
/// residual covariance (flagship, calibrated model):
///
/// ```text
/// C_used[ra, ra]   = C_base[ra, ra]   + resid_var_ra
/// C_used[dec, dec] = C_base[dec, dec] + resid_var_dec
/// C_used[ra, dec]  = C_base[ra, dec]  + resid_cov_ra_dec
/// ```
///
/// (in the table's cos(dec)-corrected arcsec² frame). `Replace` sets the block
/// to the measured covariance instead. A non-finite baseline cross-term is
/// treated as 0 when adding; non-finite baseline variances propagate as NaN.
/// Rows whose station lacks finite residual covariance values, or with
/// degenerate cos(dec), pass through unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalCovarianceModel {
    pub table: BiasTable,
    pub mode: EmpiricalCovarianceMode,
    pub min_resid_cov_n: f64,
}

impl ObservationUncertaintyModel for EmpiricalCovarianceModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        let mode = self.mode;
        apply_bias_table_model(
            &self.table,
            self.min_resid_cov_n,
            observations,
            |row, var_lon, var_lat, cov_lonlat, cos_dec| {
                if !(row.resid_var_ra.is_finite()
                    && row.resid_var_dec.is_finite()
                    && row.resid_cov_ra_dec.is_finite())
                {
                    return None;
                }
                if !cos_dec.is_finite() || cos_dec <= 0.0 {
                    return None;
                }
                // arcsec² (cos(dec)-corrected RA) -> deg² (lon not cos(dec)-corrected):
                // variances divide by cos²(dec) on the RA axis, the cross-term by
                // cos(dec) once, the Dec axis converts units only.
                let add_var_lon = row.resid_var_ra / (ARCSEC_PER_DEG_SQ * (cos_dec * cos_dec));
                let add_var_lat = row.resid_var_dec / ARCSEC_PER_DEG_SQ;
                let add_cov = row.resid_cov_ra_dec / (ARCSEC_PER_DEG_SQ * cos_dec);
                match mode {
                    EmpiricalCovarianceMode::Replace => Some((add_var_lon, add_var_lat, add_cov)),
                    EmpiricalCovarianceMode::Add => {
                        let base_cov = if cov_lonlat.is_finite() {
                            cov_lonlat
                        } else {
                            0.0
                        };
                        Some((
                            var_lon + add_var_lon,
                            var_lat + add_var_lat,
                            base_cov + add_cov,
                        ))
                    }
                }
            },
        )
    }
}

/// Scale each axis sigma by `sqrt(max(chi2_per_obs, 1))` for the observation's
/// station (aggressive model): the RA/Dec covariance block is multiplied by
/// `max(chi2_per_obs, 1)`. Frame-independent; stations with `chi2_per_obs <= 1`
/// or non-finite pass through unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceWeightedModel {
    pub table: BiasTable,
    pub min_resid_cov_n: f64,
}

impl ObservationUncertaintyModel for PerformanceWeightedModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        apply_bias_table_model(
            &self.table,
            self.min_resid_cov_n,
            observations,
            |row, var_lon, var_lat, cov_lonlat, _cos_dec| {
                let factor_sq = row.chi2_per_obs;
                if !factor_sq.is_finite() || factor_sq <= 1.0 {
                    return None;
                }
                Some((
                    var_lon * factor_sq,
                    var_lat * factor_sq,
                    cov_lonlat * factor_sq,
                ))
            },
        )
    }
}

/// Floor each axis sigma at the magnitude of the station's measured bias:
/// `sigma_used = max(sigma_baseline, |bias|)` per axis (reference model). The
/// RA floor is converted onto the lon axis by dividing by cos(dec); with a
/// degenerate cos(dec) the RA floor is skipped while the Dec floor still
/// applies. Axes with a non-finite baseline variance or bias pass through;
/// the cross-term is left unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct SigmaFloorModel {
    pub table: BiasTable,
    pub min_resid_cov_n: f64,
}

impl ObservationUncertaintyModel for SigmaFloorModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        apply_bias_table_model(
            &self.table,
            self.min_resid_cov_n,
            observations,
            |row, var_lon, var_lat, cov_lonlat, cos_dec| {
                let mut new_var_lon = var_lon;
                let mut new_var_lat = var_lat;
                if row.bias_ra_arcsec.is_finite()
                    && var_lon.is_finite()
                    && cos_dec.is_finite()
                    && cos_dec > 0.0
                {
                    let floor_lon = (row.bias_ra_arcsec.abs() / (ARCSEC_PER_DEG * cos_dec)).powi(2);
                    new_var_lon = var_lon.max(floor_lon);
                }
                if row.bias_dec_arcsec.is_finite() && var_lat.is_finite() {
                    let floor_lat = (row.bias_dec_arcsec.abs() / ARCSEC_PER_DEG).powi(2);
                    new_var_lat = var_lat.max(floor_lat);
                }
                if new_var_lon == var_lon && new_var_lat == var_lat {
                    return None;
                }
                Some((new_var_lon, new_var_lat, cov_lonlat))
            },
        )
    }
}

/// Deweight same-station, same-night batches following Veres, Farnocchia &
/// Chesley (2017): when a station contributes `N > cap` observations on one
/// night (`floor(UTC MJD)`), each of those observations has its RA/Dec
/// covariance block scaled by `N / cap`, capping the effective statistical
/// weight of the batch at ~`cap` independent observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NightBatchDeweightingModel {
    pub cap: usize,
}

impl NightBatchDeweightingModel {
    pub fn new(cap: usize) -> Result<Self, String> {
        if cap < 1 {
            return Err(format!("cap must be a positive integer, got {cap}"));
        }
        Ok(Self { cap })
    }

    /// Per-observation deweighting factor `max(N_batch / cap, 1)`.
    pub fn factors(&self, observations: &OrbitDeterminationAstrometry) -> Vec<f64> {
        let mut batch_sizes: HashMap<(&str, i64), usize> = HashMap::new();
        let nights: Vec<i64> = observations
            .mjd_utc
            .iter()
            .map(|mjd| mjd.floor() as i64)
            .collect();
        for (code, night) in observations.obs_code.iter().zip(&nights) {
            *batch_sizes.entry((code.as_str(), *night)).or_insert(0) += 1;
        }
        observations
            .obs_code
            .iter()
            .zip(&nights)
            .map(|(code, night)| {
                let size = batch_sizes[&(code.as_str(), *night)];
                (size as f64 / self.cap as f64).max(1.0)
            })
            .collect()
    }
}

impl Default for NightBatchDeweightingModel {
    fn default() -> Self {
        Self { cap: 4 }
    }
}

impl ObservationUncertaintyModel for NightBatchDeweightingModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        observations.validate()?;
        if observations.is_empty() {
            return Ok(false);
        }
        let factors = self.factors(observations);
        if factors.iter().all(|&factor| factor == 1.0) {
            return Ok(false);
        }
        for (row, factor) in factors.iter().enumerate() {
            let block = &mut observations.covariance[row * 36..(row + 1) * 36];
            for index in [VAR_LON, VAR_LAT, COV_LON_LAT, COV_LAT_LON] {
                block[index] *= factor;
            }
        }
        Ok(true)
    }
}

/// Subtract the EFCC18 star-catalog bias from observed positions.
///
/// THIS MODEL MODIFIES POSITIONS by design; the covariance block is left
/// exactly as supplied. Observations pass through unchanged when their
/// `astcat` is null, not tabulated, or listed in `exclude_astcats`, and when
/// the position is non-finite or at a pole (where the RA correction is
/// undefined). See [`crate::efcc18`].
#[derive(Debug, Clone)]
pub struct Efcc18DebiasModel {
    pub bias_table: Arc<Efcc18BiasTable>,
    pub exclude_astcats: Vec<String>,
}

impl ObservationUncertaintyModel for Efcc18DebiasModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        observations.validate()?;
        if observations.is_empty() {
            return Ok(false);
        }
        let exclude: Vec<&str> = self.exclude_astcats.iter().map(String::as_str).collect();
        let astcats = observations.astcats();
        let covered = is_efcc18_covered(&astcats, &exclude);
        if !covered.iter().any(|&flag| flag) {
            return Ok(false);
        }
        let corrections = compute_efcc18_corrections(
            &observations.lon,
            &observations.lat,
            &astcats,
            &observations.jd_tdb,
            &self.bias_table,
            &exclude,
        )?;
        match efcc18_debias_positions(&observations.lon, &observations.lat, &corrections, &covered)
        {
            Some((lon, lat)) => {
                observations.lon = lon;
                observations.lat = lat;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

/// Apply a sequence of models in order (left to right). Because the shipped
/// models act multiplicatively or additively on the covariance block, order
/// can matter.
pub struct CompositeModel {
    pub models: Vec<Box<dyn ObservationUncertaintyModel>>,
}

impl CompositeModel {
    pub fn new(models: Vec<Box<dyn ObservationUncertaintyModel>>) -> Result<Self, String> {
        if models.is_empty() {
            return Err("CompositeModel requires at least one model".to_string());
        }
        Ok(Self { models })
    }
}

impl ObservationUncertaintyModel for CompositeModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        let mut changed = false;
        for model in &self.models {
            changed |= model.apply(observations)?;
        }
        Ok(changed)
    }
}

/// Assert that a model application did not modify observed positions
/// (ids/times are the caller's responsibility). Opt-in helper for tests of
/// position-preserving models.
pub fn assert_positions_unchanged(
    before: &OrbitDeterminationAstrometry,
    after: &OrbitDeterminationAstrometry,
) -> Result<(), String> {
    if before.len() != after.len() {
        return Err(format!(
            "Number of observations changed: {} -> {}",
            before.len(),
            after.len()
        ));
    }
    for (axis, (a, b)) in [
        ("lon", (&before.lon, &after.lon)),
        ("lat", (&before.lat, &after.lat)),
    ] {
        if a.iter().zip(b.iter()).any(|(x, y)| !same_value(*x, *y)) {
            return Err(format!("Observed positions changed on axis '{axis}'"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(overrides: impl Fn(&mut BiasTableRow)) -> BiasTableRow {
        let mut row = BiasTableRow {
            bias_ra_arcsec: 0.0,
            bias_dec_arcsec: 0.0,
            resid_var_ra: 0.0,
            resid_var_dec: 0.0,
            resid_cov_ra_dec: 0.0,
            resid_cov_n: 100.0,
            chi2_per_obs: 1.0,
        };
        overrides(&mut row);
        row
    }

    fn table(entries: Vec<(&str, Option<&str>, BiasTableRow)>) -> BiasTable {
        BiasTable::new(
            entries
                .iter()
                .map(|(code, _, _)| code.to_string())
                .collect(),
            entries
                .iter()
                .map(|(_, band, _)| band.map(str::to_string))
                .collect(),
            entries.into_iter().map(|(_, _, row)| row).collect(),
        )
        .unwrap()
    }

    fn observations(
        codes: &[&str],
        lats: &[f64],
        sigma_lon: f64,
        sigma_lat: f64,
        cov_lonlat: Option<f64>,
        bands: Option<&[Option<&str>]>,
    ) -> OrbitDeterminationAstrometry {
        let n = codes.len();
        let mut covariance = vec![f64::NAN; n * 36];
        for block in covariance.chunks_exact_mut(36) {
            block[VAR_LON] = sigma_lon * sigma_lon;
            block[VAR_LAT] = sigma_lat * sigma_lat;
            if let Some(cross) = cov_lonlat {
                block[COV_LON_LAT] = cross;
                block[COV_LAT_LON] = cross;
            }
        }
        OrbitDeterminationAstrometry {
            lon: (0..n).map(|i| 10.0 + i as f64).collect(),
            lat: lats.to_vec(),
            covariance,
            obs_code: codes.iter().map(|code| code.to_string()).collect(),
            band: match bands {
                Some(bands) => bands.iter().map(|band| band.map(str::to_string)).collect(),
                None => vec![None; n],
            },
            astcat: vec![None; n],
            mjd_utc: (0..n).map(|i| 60000.0 + 0.01 * i as f64).collect(),
            jd_tdb: (0..n).map(|i| 2_460_000.5 + 0.01 * i as f64).collect(),
        }
    }

    fn block(observations: &OrbitDeterminationAstrometry, row: usize) -> (f64, f64, f64, f64) {
        let block = &observations.covariance[row * 36..(row + 1) * 36];
        (
            block[VAR_LON],
            block[VAR_LAT],
            block[COV_LON_LAT],
            block[COV_LAT_LON],
        )
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1e-300)
    }

    /// NaN-aware equality of two column views (`PartialEq` treats NaN != NaN).
    fn assert_same(a: &OrbitDeterminationAstrometry, b: &OrbitDeterminationAstrometry) {
        assert_eq!(a.lon, b.lon);
        assert_eq!(a.lat, b.lat);
        assert_eq!(a.covariance.len(), b.covariance.len());
        for (x, y) in a.covariance.iter().zip(b.covariance.iter()) {
            assert!(same_value(*x, *y), "{x} != {y}");
        }
        assert_eq!(a.obs_code, b.obs_code);
        assert_eq!(a.band, b.band);
        assert_eq!(a.astcat, b.astcat);
    }

    #[test]
    fn empirical_add_mode_at_equator_and_high_declination() {
        let bias = table(vec![(
            "500",
            None,
            row(|r| {
                r.resid_var_ra = 0.36;
                r.resid_var_dec = 0.25;
                r.resid_cov_ra_dec = 0.09;
            }),
        )]);
        let model = EmpiricalCovarianceModel {
            table: bias,
            mode: EmpiricalCovarianceMode::Add,
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[0.0], 1e-4, 2e-4, Some(1e-9), None);
        let before = obs.clone();
        assert!(model.apply(&mut obs).unwrap());
        let (var_lon, var_lat, cov, cov_t) = block(&obs, 0);
        assert!(close(var_lon, 1e-8 + 0.36 / ARCSEC_PER_DEG_SQ));
        assert!(close(var_lat, 4e-8 + 0.25 / ARCSEC_PER_DEG_SQ));
        assert!(close(cov, 1e-9 + 0.09 / ARCSEC_PER_DEG_SQ));
        assert_eq!(cov, cov_t);
        assert_positions_unchanged(&before, &obs).unwrap();

        let mut obs = observations(&["500"], &[60.0], 1e-4, 2e-4, Some(0.0), None);
        assert!(model.apply(&mut obs).unwrap());
        let cos_dec = 60.0_f64.to_radians().cos();
        let (var_lon, var_lat, cov, _) = block(&obs, 0);
        assert!(close(
            var_lon,
            1e-8 + 0.36 / (ARCSEC_PER_DEG_SQ * cos_dec * cos_dec)
        ));
        assert!(close(var_lat, 4e-8 + 0.25 / ARCSEC_PER_DEG_SQ));
        assert!(close(cov, 0.09 / (ARCSEC_PER_DEG_SQ * cos_dec)));
    }

    #[test]
    fn empirical_replace_mode_nan_cross_term_and_pass_through_rules() {
        let bias = table(vec![(
            "500",
            None,
            row(|r| {
                r.resid_var_ra = 0.36;
                r.resid_var_dec = 0.25;
                r.resid_cov_ra_dec = 0.09;
            }),
        )]);
        let replace = EmpiricalCovarianceModel {
            table: bias.clone(),
            mode: EmpiricalCovarianceMode::Replace,
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[0.0], 1e-4, 2e-4, Some(1e-9), None);
        assert!(replace.apply(&mut obs).unwrap());
        let (var_lon, var_lat, cov, _) = block(&obs, 0);
        assert!(close(var_lon, 0.36 / ARCSEC_PER_DEG_SQ));
        assert!(close(var_lat, 0.25 / ARCSEC_PER_DEG_SQ));
        assert!(close(cov, 0.09 / ARCSEC_PER_DEG_SQ));

        // NaN base cross-term is treated as zero when adding.
        let add = EmpiricalCovarianceModel {
            table: bias,
            mode: EmpiricalCovarianceMode::Add,
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[0.0], 1e-4, 2e-4, None, None);
        assert!(add.apply(&mut obs).unwrap());
        assert!(close(block(&obs, 0).2, 0.09 / ARCSEC_PER_DEG_SQ));

        // Absent station, low or null resid_cov_n: untouched.
        let sparse = table(vec![
            ("X05", None, row(|r| r.resid_var_ra = 1.0)),
            (
                "F51",
                None,
                row(|r| {
                    r.resid_var_ra = 1.0;
                    r.resid_cov_n = 5.0;
                }),
            ),
            (
                "W84",
                None,
                row(|r| {
                    r.resid_var_ra = 1.0;
                    r.resid_cov_n = f64::NAN;
                }),
            ),
        ]);
        let model = EmpiricalCovarianceModel {
            table: sparse.clone(),
            mode: EmpiricalCovarianceMode::Add,
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(
            &["500", "F51", "W84"],
            &[0.0, 30.0, 0.0],
            1e-4,
            2e-4,
            None,
            None,
        );
        let before = obs.clone();
        assert!(!model.apply(&mut obs).unwrap());
        assert_same(&obs, &before);
        // A lower threshold admits F51.
        let model = EmpiricalCovarianceModel {
            table: sparse,
            mode: EmpiricalCovarianceMode::Add,
            min_resid_cov_n: 5.0,
        };
        assert!(model.apply(&mut obs).unwrap());
        assert!(block(&obs, 1).0 > 1e-8);
        assert_eq!(block(&obs, 0).0, block(&before, 0).0);
        assert_eq!(block(&obs, 0).1, block(&before, 0).1);

        assert!(EmpiricalCovarianceMode::parse("subtract").is_err());
        assert!(BiasTable::new(
            vec!["500".into(), "500".into()],
            vec![None, None],
            vec![row(|_| {}), row(|_| {})]
        )
        .unwrap_err()
        .contains("duplicate"));
    }

    #[test]
    fn band_specific_row_is_preferred() {
        let bias = table(vec![
            ("F51", None, row(|r| r.resid_var_ra = 0.25)),
            ("F51", Some("g"), row(|r| r.resid_var_ra = 1.0)),
        ]);
        let model = EmpiricalCovarianceModel {
            table: bias,
            mode: EmpiricalCovarianceMode::Add,
            min_resid_cov_n: 30.0,
        };
        let bands: [Option<&str>; 3] = [Some("g"), Some("r"), None];
        let mut obs = observations(&["F51"; 3], &[0.0; 3], 1e-4, 2e-4, None, Some(&bands));
        assert!(model.apply(&mut obs).unwrap());
        assert!(close(block(&obs, 0).0, 1e-8 + 1.0 / ARCSEC_PER_DEG_SQ));
        assert!(close(block(&obs, 1).0, 1e-8 + 0.25 / ARCSEC_PER_DEG_SQ));
        assert!(close(block(&obs, 2).0, 1e-8 + 0.25 / ARCSEC_PER_DEG_SQ));
    }

    #[test]
    fn performance_weighted_scales_by_chi2_and_clamps_below_one() {
        let model = PerformanceWeightedModel {
            table: table(vec![("500", None, row(|r| r.chi2_per_obs = 4.0))]),
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[45.0], 1e-4, 2e-4, Some(1e-9), None);
        assert!(model.apply(&mut obs).unwrap());
        let (var_lon, var_lat, cov, _) = block(&obs, 0);
        assert!(close(var_lon, 4e-8) && close(var_lat, 1.6e-7) && close(cov, 4e-9));

        let clamp = PerformanceWeightedModel {
            table: table(vec![("500", None, row(|r| r.chi2_per_obs = 0.25))]),
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[45.0], 1e-4, 2e-4, None, None);
        let before = obs.clone();
        assert!(!clamp.apply(&mut obs).unwrap());
        assert_same(&obs, &before);
    }

    #[test]
    fn sigma_floor_floors_axes_independently_with_cos_dec_on_ra() {
        let model = SigmaFloorModel {
            table: table(vec![(
                "500",
                None,
                row(|r| {
                    r.bias_ra_arcsec = -2.0;
                    r.bias_dec_arcsec = 0.1;
                }),
            )]),
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[0.0], 1e-4, 2e-4, Some(1e-9), None);
        assert!(model.apply(&mut obs).unwrap());
        let (var_lon, var_lat, cov, _) = block(&obs, 0);
        assert!(close(var_lon, (2.0 / ARCSEC_PER_DEG).powi(2)));
        assert!(close(var_lat, 4e-8));
        assert!(close(cov, 1e-9));

        let mut obs = observations(&["500"], &[60.0], 1e-4, 2e-4, None, None);
        assert!(model.apply(&mut obs).unwrap());
        let cos_dec = 60.0_f64.to_radians().cos();
        assert!(close(
            block(&obs, 0).0,
            (2.0 / (ARCSEC_PER_DEG * cos_dec)).powi(2)
        ));

        let small = SigmaFloorModel {
            table: table(vec![(
                "500",
                None,
                row(|r| {
                    r.bias_ra_arcsec = 0.01;
                    r.bias_dec_arcsec = 0.01;
                }),
            )]),
            min_resid_cov_n: 30.0,
        };
        let mut obs = observations(&["500"], &[0.0], 1e-4, 2e-4, None, None);
        let before = obs.clone();
        assert!(!small.apply(&mut obs).unwrap());
        assert_same(&obs, &before);
    }

    #[test]
    fn night_batch_deweighting_is_per_station_and_night() {
        assert!(NightBatchDeweightingModel::new(0).is_err());
        let model = NightBatchDeweightingModel::new(4).unwrap();
        let mut obs = observations(&["F51"; 6], &[10.0; 6], 1e-4, 2e-4, Some(1e-9), None);
        let before = obs.clone();
        assert!(model.apply(&mut obs).unwrap());
        assert_positions_unchanged(&before, &obs).unwrap();
        for row in 0..6 {
            let (var_lon, var_lat, cov, cov_t) = block(&obs, row);
            assert!(close(var_lon, 1.5e-8) && close(var_lat, 6e-8));
            assert!(close(cov, 1.5e-9) && close(cov_t, 1.5e-9));
        }

        let mut obs = observations(&["F51"; 4], &[10.0; 4], 1e-4, 2e-4, None, None);
        let before = obs.clone();
        assert!(!model.apply(&mut obs).unwrap());
        assert_same(&obs, &before);

        let codes = [
            "F51", "F51", "F51", "F51", "F51", "500", "500", "F51", "F51", "F51", "F51", "F51",
        ];
        let mut obs = observations(&codes, &[0.0; 12], 1e-4, 2e-4, None, None);
        obs.mjd_utc = vec![
            60000.1, 60000.11, 60000.12, 60000.13, 60000.14, 60000.3, 60000.31, 60001.1, 60001.11,
            60001.12, 60001.13, 60001.14,
        ];
        let factors = model.factors(&obs);
        let expected = [
            1.25, 1.25, 1.25, 1.25, 1.25, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25, 1.25,
        ];
        assert_eq!(factors, expected);
    }

    #[test]
    fn composite_applies_in_sequence_and_identity_is_noop() {
        let performance = PerformanceWeightedModel {
            table: table(vec![("F51", None, row(|r| r.chi2_per_obs = 4.0))]),
            min_resid_cov_n: 30.0,
        };
        let composite = CompositeModel::new(vec![
            Box::new(performance),
            Box::new(NightBatchDeweightingModel::new(4).unwrap()),
        ])
        .unwrap();
        let mut obs = observations(&["F51"; 5], &[10.0; 5], 1e-4, 2e-4, None, None);
        assert!(composite.apply(&mut obs).unwrap());
        for row in 0..5 {
            let (var_lon, var_lat, _, _) = block(&obs, row);
            assert!(close(var_lon, 1e-8 * 5.0) && close(var_lat, 4e-8 * 5.0));
        }
        assert!(CompositeModel::new(vec![]).is_err());

        let identity =
            CompositeModel::new(vec![Box::new(IdentityModel), Box::new(IdentityModel)]).unwrap();
        let mut obs = observations(&["F51", "500"], &[10.0, -20.0], 1e-4, 2e-4, None, None);
        let before = obs.clone();
        assert!(!identity.apply(&mut obs).unwrap());
        assert_same(&obs, &before);
    }

    #[test]
    fn efcc18_model_moves_positions_and_keeps_covariance() {
        use crate::efcc18::{efcc18_catalog_column, ra_dec_to_healpix, EFCC18_N_TILES};
        let lons = [100.0, 50.0];
        let lats = [30.0, 80.0];
        let mut values = vec![0.0_f32; EFCC18_N_TILES * 26 * 4];
        let column = efcc18_catalog_column("UCAC4").unwrap();
        for (lon, lat) in lons.iter().zip(lats.iter()) {
            let tile = ra_dec_to_healpix(*lon, *lat).unwrap() as usize;
            let offset = (tile * 26 + column) * 4;
            values[offset..offset + 4].copy_from_slice(&[0.5, -0.3, 20.0, -10.0]);
        }
        let model = Efcc18DebiasModel {
            bias_table: Arc::new(Efcc18BiasTable::from_values(values).unwrap()),
            exclude_astcats: vec![],
        };
        let mut obs = observations(&["500", "500"], &lats, 1e-4, 2e-4, Some(1e-9), None);
        obs.lon = lons.to_vec();
        obs.astcat = vec![Some("UCAC4".into()), Some("UCAC4".into())];
        obs.jd_tdb = vec![2_451_545.0 + 10.0 * 365.25; 2];
        let before = obs.clone();
        assert!(model.apply(&mut obs).unwrap());
        for row in 0..2 {
            let cos_dec = lats[row].to_radians().cos();
            let expected_lon = lons[row] - 0.7 / (ARCSEC_PER_DEG * cos_dec);
            let expected_lat = lats[row] + 0.4 / ARCSEC_PER_DEG;
            assert!((obs.lon[row] - expected_lon).abs() < 1e-10);
            assert!((obs.lat[row] - expected_lat).abs() < 1e-10);
        }
        assert_eq!(obs.covariance.len(), before.covariance.len());
        for (a, b) in obs.covariance.iter().zip(before.covariance.iter()) {
            assert!(same_value(*a, *b));
        }
        assert!(assert_positions_unchanged(&before, &obs).is_err());

        // Uncovered / null / excluded catalogs pass through untouched.
        let mut obs = before.clone();
        obs.astcat = vec![Some("Gaia2".into()), None];
        assert!(!model.apply(&mut obs).unwrap());
        let excluded = Efcc18DebiasModel {
            bias_table: model.bias_table.clone(),
            exclude_astcats: vec!["UCAC4".into()],
        };
        let mut obs = before.clone();
        obs.astcat = vec![Some("UCAC4".into()); 2];
        assert!(!excluded.apply(&mut obs).unwrap());
        assert_same(&obs, &before);
    }
}
