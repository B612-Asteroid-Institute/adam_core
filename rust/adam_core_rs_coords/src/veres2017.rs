//! Veres, Farnocchia & Chesley (2017) per-observation astrometric uncertainties.
//!
//! Rust-canonical port of `adam_core.orbit_determination.veres2017`. Veres,
//! Farnocchia, Chesley & Chamberlin (2017, Icarus 296, 139; "VFC2017") derived
//! station- and catalog-dependent astrometric uncertainties for MPC
//! observations from the residual statistics of well-determined orbits.
//! Agencies use such tables as default weights for observations that report
//! no uncertainty, or as floors on the reported ones. This module ships the
//! per-(station, catalog) sigma table used by the Asteroid Institute's OD
//! experiments together with the interpreters that apply it
//! ([`VeresFloorModel`], [`VeresReplaceModel`]).
//!
//! Table provenance
//! ----------------
//! [`veres2017_sigma_table`] is the working table maintained in
//! `adam_orbit_det_eval` (`VERES2017_CATALOG_DEFAULTS` +
//! `VERES2017_STN_CATALOG_OVERRIDES`, fallback 0.75"), transcribed verbatim.
//! It is a catalog-level summary in the spirit of VFC2017's Table 1 plus a few
//! station-specific overrides; it is NOT a transcription of the paper's full
//! station table.
//!
//! Frames and units
//! ----------------
//! Sigmas are in arcseconds with the RA axis in the cos(dec)-corrected frame
//! (MPC/ADES `rmsRACosDec` convention). `SphericalCoordinates` covariances are
//! in degrees² with lon NOT cos(dec)-corrected, so the RA sigma is divided by
//! cos(dec) once (in addition to the arcsec -> degree scaling) before squaring.

use crate::observation_uncertainty::{
    ObservationUncertaintyModel, OrbitDeterminationAstrometry, ARCSEC_PER_DEG,
};
use std::collections::HashMap;

/// Global fallback when neither a (station, catalog) nor a catalog row exists.
pub const VERES2017_FALLBACK_SIGMA_ARCSEC: f64 = 0.75;

/// Per-catalog defaults `(astcat, sigma_ra_arcsec, sigma_dec_arcsec)`; keys
/// are MPC `astCat` codes as carried on `OrbitDeterminationObservations.astcat`.
pub const VERES2017_CATALOG_DEFAULTS: [(&str, f64, f64); 34] = [
    // Gaia family
    ("Gaia3E", 0.15, 0.15),
    ("Gaia3", 0.15, 0.15),
    ("Gaia2", 0.18, 0.18),
    ("Gaia1", 0.25, 0.25),
    // ATLAS family
    ("ATLAS2", 0.20, 0.20),
    ("ATLAS", 0.25, 0.25),
    // UCAC family
    ("UCAC5", 0.25, 0.25),
    ("SSTRC4", 0.25, 0.25),
    ("UCAC4", 0.30, 0.30),
    ("UCAC3", 0.30, 0.30),
    ("UCAC2", 0.40, 0.40),
    ("UCAC1", 0.50, 0.50),
    // 2MASS
    ("2MASS", 0.20, 0.20),
    // USNO catalogs
    ("USNOB1", 0.50, 0.50),
    ("USNOA2", 0.60, 0.60),
    ("USNOSA2", 0.60, 0.60),
    ("USNOA1", 0.80, 0.80),
    // GSC family
    ("GSC", 0.50, 0.50),
    ("GSC1.1", 0.50, 0.50),
    ("GSC1.2", 0.50, 0.50),
    ("GSC2.2", 0.40, 0.40),
    ("GSC2.3", 0.35, 0.35),
    ("GSCACT", 0.50, 0.50),
    // PPMXL / PPM
    ("PPMXL", 0.35, 0.35),
    ("PPM", 0.50, 0.50),
    // Other catalogs
    ("SDSS8", 0.20, 0.20),
    ("SDSS7", 0.20, 0.20),
    ("NOMAD", 0.40, 0.40),
    ("CMC14", 0.35, 0.35),
    ("CMC15", 0.30, 0.30),
    ("Tycho", 0.06, 0.06),
    ("AC", 0.80, 0.80),
    ("Yale", 1.00, 1.00),
    ("UNK", 1.00, 1.00),
];

/// Per-(station, catalog) overrides `(obs_code, astcat, sigma_ra_arcsec, sigma_dec_arcsec)`.
pub const VERES2017_STATION_CATALOG_OVERRIDES: [(&str, &str, f64, f64); 14] = [
    ("703", "Gaia2", 0.34, 0.34), // Catalina Sky Survey: wider PSF
    ("703", "UCAC4", 0.45, 0.45),
    ("703", "UCAC2", 0.55, 0.55),
    ("G96", "Gaia2", 0.25, 0.25), // Mt. Lemmon Survey
    ("G96", "UCAC4", 0.35, 0.35),
    ("704", "USNOA2", 0.60, 0.75), // Spacewatch: known Dec bias
    ("F51", "Gaia2", 0.15, 0.15),  // Pan-STARRS 1
    ("F51", "Gaia1", 0.18, 0.18),
    ("F51", "2MASS", 0.20, 0.20),
    ("F52", "Gaia3E", 0.15, 0.15), // Pan-STARRS 2
    ("F52", "Gaia1", 0.18, 0.18),
    ("T05", "Gaia2", 0.25, 0.25), // ATLAS Haleakala
    ("T08", "Gaia2", 0.25, 0.25), // ATLAS Mauna Loa
    ("W68", "Gaia2", 0.25, 0.25), // ATLAS Chile
];

/// One row of a Veres-style sigma table: a per-catalog default (`obs_code`
/// None) or a (station, catalog) override.
#[derive(Debug, Clone, PartialEq)]
pub struct VeresSigmaRow {
    pub obs_code: Option<String>,
    pub astcat: String,
    pub sigma_ra_arcsec: f64,
    pub sigma_dec_arcsec: f64,
}

/// The bundled sigma table (catalog defaults first, then overrides).
pub fn veres2017_sigma_table() -> Vec<VeresSigmaRow> {
    let mut rows = Vec::with_capacity(
        VERES2017_CATALOG_DEFAULTS.len() + VERES2017_STATION_CATALOG_OVERRIDES.len(),
    );
    for (astcat, sigma_ra, sigma_dec) in VERES2017_CATALOG_DEFAULTS {
        rows.push(VeresSigmaRow {
            obs_code: None,
            astcat: astcat.to_string(),
            sigma_ra_arcsec: sigma_ra,
            sigma_dec_arcsec: sigma_dec,
        });
    }
    for (code, astcat, sigma_ra, sigma_dec) in VERES2017_STATION_CATALOG_OVERRIDES {
        rows.push(VeresSigmaRow {
            obs_code: Some(code.to_string()),
            astcat: astcat.to_string(),
            sigma_ra_arcsec: sigma_ra,
            sigma_dec_arcsec: sigma_dec,
        });
    }
    rows
}

fn validate_sigma(name: &str, value: f64) -> Result<(), String> {
    if value.is_nan() || value <= 0.0 || value == f64::INFINITY {
        return Err(format!(
            "Sigma table column '{name}' must contain positive finite values"
        ));
    }
    Ok(())
}

/// Resolve (station, catalog) to `(sigma_ra_arcsec, sigma_dec_arcsec)`.
///
/// Lookup order: the (station, catalog) override row, then the catalog
/// default row (`obs_code` None), then `fallback_sigma_arcsec` for both axes;
/// `None` as fallback means "no sigma known" (the caller passes the
/// observation through).
#[derive(Debug, Clone, PartialEq)]
pub struct VeresSigmaLookup {
    by_station_catalog: HashMap<(String, String), (f64, f64)>,
    by_catalog: HashMap<String, (f64, f64)>,
    pub fallback_sigma_arcsec: Option<f64>,
}

impl VeresSigmaLookup {
    pub fn new(rows: &[VeresSigmaRow], fallback_sigma_arcsec: Option<f64>) -> Result<Self, String> {
        let mut by_station_catalog = HashMap::new();
        let mut by_catalog = HashMap::new();
        for row in rows {
            validate_sigma("sigma_ra_arcsec", row.sigma_ra_arcsec)?;
            validate_sigma("sigma_dec_arcsec", row.sigma_dec_arcsec)?;
            let sigmas = (row.sigma_ra_arcsec, row.sigma_dec_arcsec);
            match &row.obs_code {
                None => {
                    if by_catalog.insert(row.astcat.clone(), sigmas).is_some() {
                        return Err(format!(
                            "Duplicate catalog default row for {:?}",
                            row.astcat
                        ));
                    }
                }
                Some(code) => {
                    let key = (code.clone(), row.astcat.clone());
                    if by_station_catalog.contains_key(&key) {
                        return Err(format!("Duplicate override row for {key:?}"));
                    }
                    by_station_catalog.insert(key, sigmas);
                }
            }
        }
        Ok(Self {
            by_station_catalog,
            by_catalog,
            fallback_sigma_arcsec,
        })
    }

    /// The bundled table with the default fallback.
    pub fn bundled() -> Self {
        Self::new(
            &veres2017_sigma_table(),
            Some(VERES2017_FALLBACK_SIGMA_ARCSEC),
        )
        .expect("bundled sigma table is valid")
    }

    /// `(sigma_ra_arcsec, sigma_dec_arcsec)` for the observation, or `None`.
    pub fn sigmas(&self, obs_code: Option<&str>, astcat: Option<&str>) -> Option<(f64, f64)> {
        if let (Some(code), Some(astcat)) = (obs_code, astcat) {
            if let Some(sigmas) = self
                .by_station_catalog
                .get(&(code.to_string(), astcat.to_string()))
            {
                return Some(*sigmas);
            }
        }
        if let Some(astcat) = astcat {
            if let Some(sigmas) = self.by_catalog.get(astcat) {
                return Some(*sigmas);
            }
        }
        self.fallback_sigma_arcsec.map(|sigma| (sigma, sigma))
    }
}

const VAR_LON: usize = 7;
const VAR_LAT: usize = 14;
const COV_LON_LAT: usize = 8;
const COV_LAT_LON: usize = 13;

fn same_value(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Shared machinery for the VFC2017 station/catalog sigma interpreters:
/// resolve each observation's (station, catalog) to per-axis sigmas and let
/// `updated_variances` decide how they combine with the reported covariance
/// (`(var_lon, var_lat, zero_cross_term)` or `None` to pass the row through).
/// Positions are never modified.
fn apply_veres_model(
    lookup: &VeresSigmaLookup,
    observations: &mut OrbitDeterminationAstrometry,
    updated_variances: impl Fn(f64, f64, f64, f64) -> Option<(f64, f64, bool)>,
) -> Result<bool, String> {
    observations.validate()?;
    let mut changed = false;
    for row in 0..observations.len() {
        let Some((sigma_ra, sigma_dec)) = lookup.sigmas(
            Some(observations.obs_code[row].as_str()),
            observations.astcat[row].as_deref(),
        ) else {
            continue;
        };
        let cos_dec = observations.lat[row].to_radians().cos();
        if !cos_dec.is_finite() || cos_dec <= 0.0 {
            continue;
        }
        let veres_var_lon = (sigma_ra / (ARCSEC_PER_DEG * cos_dec)).powi(2);
        let veres_var_lat = (sigma_dec / ARCSEC_PER_DEG).powi(2);
        let block = &mut observations.covariance[row * 36..(row + 1) * 36];
        let Some((var_lon, var_lat, zero_cross_term)) =
            updated_variances(block[VAR_LON], block[VAR_LAT], veres_var_lon, veres_var_lat)
        else {
            continue;
        };
        for (index, value) in [(VAR_LON, var_lon), (VAR_LAT, var_lat)] {
            changed |= !same_value(block[index], value);
            block[index] = value;
        }
        if zero_cross_term {
            for index in [COV_LON_LAT, COV_LAT_LON] {
                changed |= !same_value(block[index], 0.0);
                block[index] = 0.0;
            }
        }
    }
    Ok(changed)
}

/// Floor each axis sigma at the VFC2017 station/catalog sigma:
/// `sigma_used = max(sigma_reported, sigma_VFC2017)` per axis (agency floor
/// practice). The RA/Dec cross-term is left unchanged. A non-finite reported
/// variance is left as is unless `fill_missing` is set, in which case it is
/// replaced by the VFC2017 variance.
#[derive(Debug, Clone, PartialEq)]
pub struct VeresFloorModel {
    pub lookup: VeresSigmaLookup,
    pub fill_missing: bool,
}

impl ObservationUncertaintyModel for VeresFloorModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        let fill_missing = self.fill_missing;
        apply_veres_model(
            &self.lookup,
            observations,
            |var_lon, var_lat, veres_var_lon, veres_var_lat| {
                let floored = |var: f64, floor: f64| -> f64 {
                    if !var.is_finite() {
                        if fill_missing {
                            floor
                        } else {
                            var
                        }
                    } else {
                        var.max(floor)
                    }
                };
                let new_lon = floored(var_lon, veres_var_lon);
                let new_lat = floored(var_lat, veres_var_lat);
                if same_value(new_lon, var_lon) && same_value(new_lat, var_lat) {
                    return None;
                }
                Some((new_lon, new_lat, false))
            },
        )
    }
}

/// Replace each axis sigma by the VFC2017 station/catalog sigma outright
/// (classic weighting-file practice, ignoring reported uncertainties). The
/// RA/Dec cross-term is set to zero since the table carries no correlation.
#[derive(Debug, Clone, PartialEq)]
pub struct VeresReplaceModel {
    pub lookup: VeresSigmaLookup,
}

impl ObservationUncertaintyModel for VeresReplaceModel {
    fn apply(&self, observations: &mut OrbitDeterminationAstrometry) -> Result<bool, String> {
        apply_veres_model(
            &self.lookup,
            observations,
            |_var_lon, _var_lat, veres_var_lon, veres_var_lat| {
                Some((veres_var_lon, veres_var_lat, true))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_lookup(fallback: Option<f64>) -> VeresSigmaLookup {
        let rows = vec![
            VeresSigmaRow {
                obs_code: None,
                astcat: "Gaia2".into(),
                sigma_ra_arcsec: 0.20,
                sigma_dec_arcsec: 0.20,
            },
            VeresSigmaRow {
                obs_code: None,
                astcat: "UCAC4".into(),
                sigma_ra_arcsec: 0.30,
                sigma_dec_arcsec: 0.40,
            },
            VeresSigmaRow {
                obs_code: Some("703".into()),
                astcat: "Gaia2".into(),
                sigma_ra_arcsec: 0.34,
                sigma_dec_arcsec: 0.34,
            },
        ];
        VeresSigmaLookup::new(&rows, fallback).unwrap()
    }

    fn observations(
        codes: &[&str],
        astcats: &[Option<&str>],
        lats: &[f64],
    ) -> OrbitDeterminationAstrometry {
        let n = codes.len();
        let mut covariance = vec![f64::NAN; n * 36];
        for block in covariance.chunks_exact_mut(36) {
            block[VAR_LON] = 1e-10;
            block[VAR_LAT] = 4e-10;
        }
        OrbitDeterminationAstrometry {
            lon: vec![10.0; n],
            lat: lats.to_vec(),
            covariance,
            obs_code: codes.iter().map(|c| c.to_string()).collect(),
            band: vec![None; n],
            astcat: astcats.iter().map(|a| a.map(str::to_string)).collect(),
            mjd_utc: vec![60000.0; n],
            jd_tdb: vec![2_460_000.5; n],
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1e-300)
    }

    #[test]
    fn bundled_table_is_valid_and_ordered() {
        let rows = veres2017_sigma_table();
        assert_eq!(rows.len(), 48);
        assert!(rows[..34].iter().all(|row| row.obs_code.is_none()));
        assert!(rows[34..].iter().all(|row| row.obs_code.is_some()));
        let lookup = VeresSigmaLookup::bundled();
        assert_eq!(
            lookup.sigmas(Some("F51"), Some("Gaia2")),
            Some((0.15, 0.15))
        );
        assert_eq!(
            lookup.sigmas(Some("500"), Some("Gaia2")),
            Some((0.18, 0.18))
        );
        assert_eq!(
            lookup.sigmas(Some("704"), Some("USNOA2")),
            Some((0.60, 0.75))
        );
        assert_eq!(
            lookup.sigmas(Some("500"), Some("NEWCAT")),
            Some((0.75, 0.75))
        );
        assert_eq!(lookup.sigmas(None, None), Some((0.75, 0.75)));
    }

    #[test]
    fn lookup_order_and_validation() {
        let lookup = small_lookup(Some(0.75));
        assert_eq!(
            lookup.sigmas(Some("703"), Some("Gaia2")),
            Some((0.34, 0.34))
        );
        assert_eq!(
            lookup.sigmas(Some("F51"), Some("Gaia2")),
            Some((0.20, 0.20))
        );
        assert_eq!(lookup.sigmas(None, Some("UCAC4")), Some((0.30, 0.40)));
        assert_eq!(
            lookup.sigmas(Some("703"), Some("PPMXL")),
            Some((0.75, 0.75))
        );
        assert_eq!(lookup.sigmas(Some("703"), None), Some((0.75, 0.75)));
        let strict = small_lookup(None);
        assert_eq!(strict.sigmas(Some("703"), Some("PPMXL")), None);
        assert_eq!(strict.sigmas(Some("703"), None), None);

        let bad = vec![VeresSigmaRow {
            obs_code: None,
            astcat: "X".into(),
            sigma_ra_arcsec: 0.0,
            sigma_dec_arcsec: 0.1,
        }];
        assert!(VeresSigmaLookup::new(&bad, None)
            .unwrap_err()
            .contains("positive finite"));
        let dup = vec![
            VeresSigmaRow {
                obs_code: None,
                astcat: "X".into(),
                sigma_ra_arcsec: 0.1,
                sigma_dec_arcsec: 0.1,
            },
            VeresSigmaRow {
                obs_code: None,
                astcat: "X".into(),
                sigma_ra_arcsec: 0.2,
                sigma_dec_arcsec: 0.2,
            },
        ];
        assert!(VeresSigmaLookup::new(&dup, None)
            .unwrap_err()
            .contains("Duplicate"));
    }

    #[test]
    fn floor_model_floors_per_axis_with_cos_dec_and_fill_missing() {
        let model = VeresFloorModel {
            lookup: small_lookup(Some(0.75)),
            fill_missing: false,
        };
        // Reported sigmas: lon 1e-5 deg = 0.036", lat 2e-5 deg = 0.072": both
        // below every table sigma, so both axes floor.
        let mut obs = observations(
            &["500", "703"],
            &[Some("UCAC4"), Some("Gaia2")],
            &[0.0, 60.0],
        );
        let before = obs.clone();
        assert!(model.apply(&mut obs).unwrap());
        let block0 = &obs.covariance[0..36];
        assert!(close(block0[VAR_LON], (0.30 / ARCSEC_PER_DEG).powi(2)));
        assert!(close(block0[VAR_LAT], (0.40 / ARCSEC_PER_DEG).powi(2)));
        assert!(block0[COV_LON_LAT].is_nan());
        let cos_dec = 60.0_f64.to_radians().cos();
        let block1 = &obs.covariance[36..72];
        assert!(close(
            block1[VAR_LON],
            (0.34 / (ARCSEC_PER_DEG * cos_dec)).powi(2)
        ));
        assert!(close(block1[VAR_LAT], (0.34 / ARCSEC_PER_DEG).powi(2)));
        assert_eq!(obs.lon, before.lon);
        assert_eq!(obs.lat, before.lat);

        // Reported sigma above the floor is kept.
        let mut obs = observations(&["500"], &[Some("Gaia2")], &[0.0]);
        obs.covariance[VAR_LON] = 1e-6;
        obs.covariance[VAR_LAT] = 1e-6;
        assert!(!model.apply(&mut obs).unwrap());

        // Missing reported variance: untouched unless fill_missing.
        let mut obs = observations(&["500"], &[Some("Gaia2")], &[0.0]);
        obs.covariance[VAR_LON] = f64::NAN;
        obs.covariance[VAR_LAT] = 1e-6;
        assert!(!model.apply(&mut obs).unwrap());
        assert!(obs.covariance[VAR_LON].is_nan());
        let filling = VeresFloorModel {
            lookup: small_lookup(Some(0.75)),
            fill_missing: true,
        };
        assert!(filling.apply(&mut obs).unwrap());
        assert!(close(
            obs.covariance[VAR_LON],
            (0.20 / ARCSEC_PER_DEG).powi(2)
        ));

        // No fallback: unknown catalogs pass through.
        let strict = VeresFloorModel {
            lookup: small_lookup(None),
            fill_missing: false,
        };
        let mut obs = observations(&["500"], &[Some("PPMXL")], &[0.0]);
        assert!(!strict.apply(&mut obs).unwrap());
    }

    #[test]
    fn replace_model_overwrites_and_zeroes_cross_term() {
        let model = VeresReplaceModel {
            lookup: small_lookup(Some(0.75)),
        };
        let mut obs = observations(&["500"], &[Some("UCAC4")], &[30.0]);
        obs.covariance[VAR_LON] = 1e-6;
        obs.covariance[COV_LON_LAT] = 1e-9;
        obs.covariance[COV_LAT_LON] = 1e-9;
        assert!(model.apply(&mut obs).unwrap());
        let cos_dec = 30.0_f64.to_radians().cos();
        assert!(close(
            obs.covariance[VAR_LON],
            (0.30 / (ARCSEC_PER_DEG * cos_dec)).powi(2)
        ));
        assert!(close(
            obs.covariance[VAR_LAT],
            (0.40 / ARCSEC_PER_DEG).powi(2)
        ));
        assert_eq!(obs.covariance[COV_LON_LAT], 0.0);
        assert_eq!(obs.covariance[COV_LAT_LON], 0.0);
    }
}
