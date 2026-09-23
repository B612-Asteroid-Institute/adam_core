//! EFCC18 star-catalog debiasing of astrometric observations.
//!
//! Rust-canonical port of `adam_core.observations.efcc18`: the per-observation
//! correction of Eggl, Farnocchia, Chamberlin & Chesley (2020), "Star catalog
//! position and proper motion corrections in asteroid astrometry II: The Gaia
//! era", Icarus 339:113596 (EFCC18). The published table `bias.dat` covers 26
//! star catalogs over a HEALPix tessellation of the sky (`N_side = 64`, 49152
//! tiles) in **RING** ordering: the k-th data row of `bias.dat` is HEALPix
//! ring pixel k, as listed in the archive's `tiles.dat`. Each (tile, catalog)
//! cell stores four numbers: the position correction in RA*cos(Dec) at J2000
//! [arcsec], the position correction in Dec at J2000 [arcsec], and the proper
//! motion corrections in RA*cos(Dec) and Dec [mas/yr], all with respect to
//! Gaia-DR2.
//!
//! To debias one observation reduced against catalog `c` in tile `k` at epoch
//! `t` (Julian years):
//!
//! ```text
//! bias_ra_arcsec  = dRA[k, c]  + (t - 2000) * pmRA[k, c]  / 1000   # cos(dec) frame
//! bias_dec_arcsec = dDec[k, c] + (t - 2000) * pmDec[k, c] / 1000
//! corrected_RA  = observed_RA  - bias_ra_arcsec  / (3600 cos(Dec))  # degrees
//! corrected_Dec = observed_Dec - bias_dec_arcsec / 3600             # degrees
//! ```
//!
//! The tile lookup reuses the crate's `healpix::ang2pix` port of healpy's
//! `T_Healpix_Base<int64>::ang2pix` with the ring scheme. An earlier Python
//! implementation read the table in the nested scheme, which put every
//! observation in an unrelated tile; `ring_order_matches_jpl_tiles_dat`
//! pins the row order against the tile centres published in `tiles.dat`.
//!
//! Locating, downloading and caching `bias.dat` (the JPL archive, the
//! `jpl-debias-2018` data package, the XDG cache) stay in the Python veneer;
//! this module owns the parse, the tile/catalog lookup and the arithmetic.

use crate::healpix::ang2pix;

/// Source archive published by JPL SSD (also linked from Eggl et al. 2020).
pub const EFCC18_ARCHIVE_URL: &str = "https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz";
/// SHA-256 of the `bias.dat` inside the 2023-03-01 archive.
pub const EFCC18_BIAS_DAT_SHA256: &str =
    "ef6a50830bb83d8b1161e7acb391eed988ea9708e9fbcccc0c834da6109476e3";
/// `BIAS_VERSION` tag in the header of that `bias.dat`.
pub const EFCC18_BIAS_VERSION: &str = "3.0 (September 21, 2018)";

/// HEALPix resolution used by EFCC18 and the resulting number of tiles.
pub const EFCC18_NSIDE: i64 = 64;
pub const EFCC18_N_TILES: usize = 12 * (EFCC18_NSIDE as usize) * (EFCC18_NSIDE as usize);
/// Single-character EFCC18 catalog codes, in column order of `bias.dat`
/// (header line "Catalogs in this file (MPC designation)").
pub const EFCC18_CATALOG_CODES: &str = "abcdegijlmnopqrtuvwLNQRSUY";
pub const EFCC18_N_CATALOGS: usize = 26;
/// Numbers stored per (tile, catalog) cell.
pub const EFCC18_N_COMPONENTS: usize = 4;

/// Map from MPC/ADES `astCat` code to the single-character EFCC18 code.
/// Catalogs that postdate EFCC18 (Gaia-DR2/EDR3/DR3, ATLAS, Pan-STARRS, ...)
/// are intentionally absent: they have no EFCC18 entry and pass through with
/// zero correction.
pub const MPC_ASTCAT_TO_EFCC18: [(&str, char); 26] = [
    ("USNOA1", 'a'),
    ("USNOSA1", 'b'),
    ("USNOA2", 'c'),
    ("USNOSA2", 'd'),
    ("UCAC1", 'e'),
    ("Tycho", 'g'), // Tycho-2
    ("GSC1.1", 'i'),
    ("GSC1.2", 'j'),
    ("ACT", 'l'),
    ("GSCACT", 'm'),
    ("SDSS8", 'n'), // SDSS-DR8
    ("USNOB1", 'o'),
    ("PPM", 'p'),
    ("UCAC4", 'q'),
    ("UCAC2", 'r'),
    ("PPMXL", 't'),
    ("UCAC3", 'u'),
    ("NOMAD", 'v'),
    ("CMC14", 'w'),
    ("2MASS", 'L'),
    ("SDSS7", 'N'),
    ("CMC15", 'Q'),
    ("SSTRC4", 'R'),
    ("URAT1", 'S'),
    ("Gaia1", 'U'), // Gaia-DR1
    ("UCAC5", 'Y'),
];

/// Catalogs tabulated in `bias.dat` that JPL SSD nevertheless does not debias
/// (per the file header), as MPC `astCat` codes.
pub const EFCC18_JPL_UNDEBIASED_ASTCATS: [&str; 4] = ["Gaia1", "ACT", "Tycho", "UCAC5"];

const JD_J2000: f64 = 2_451_545.0;
const DAYS_PER_JULIAN_YEAR: f64 = 365.25;
const POLE_COS_DEC_FLOOR: f64 = 1.0e-12;
const NUMBERS_PER_ROW: usize = EFCC18_N_CATALOGS * EFCC18_N_COMPONENTS;

/// Single-character EFCC18 code for an MPC `astCat` code, if tabulated.
pub fn efcc18_catalog_code(astcat: &str) -> Option<char> {
    MPC_ASTCAT_TO_EFCC18
        .iter()
        .find(|(name, _)| *name == astcat)
        .map(|(_, code)| *code)
}

/// `bias.dat` column index (0..26) for an MPC `astCat` code, if tabulated.
pub fn efcc18_catalog_column(astcat: &str) -> Option<usize> {
    let code = efcc18_catalog_code(astcat)?;
    EFCC18_CATALOG_CODES.chars().position(|c| c == code)
}

/// Map (RA, Dec) in degrees to the EFCC18 HEALPix tile index.
///
/// `N_side = 64` in RING ordering on the J2000 equatorial frame, i.e. the row
/// order of `bias.dat`. The colatitude/longitude conversion mirrors the
/// legacy `ra_dec_to_healpix` exactly (`theta = deg2rad(90 - dec)`,
/// `phi = deg2rad(mod(ra, 360))`).
pub fn ra_dec_to_healpix(ra_deg: f64, dec_deg: f64) -> Result<i64, String> {
    let theta = (90.0 - dec_deg).to_radians();
    let phi = ra_deg.rem_euclid(360.0).to_radians();
    ang2pix(EFCC18_NSIDE, theta, phi, false)
}

/// The parsed EFCC18 table: `(EFCC18_N_TILES, 26, 4)` float32 values in
/// tile-major order, exactly as `bias.dat` lists them (ring pixel order).
#[derive(Debug, Clone, PartialEq)]
pub struct Efcc18BiasTable {
    values: Vec<f32>,
}

impl Efcc18BiasTable {
    /// Wrap already-parsed values (e.g. a cached copy). The length must be
    /// `EFCC18_N_TILES * 26 * 4`.
    pub fn from_values(values: Vec<f32>) -> Result<Self, String> {
        let expected = EFCC18_N_TILES * NUMBERS_PER_ROW;
        if values.len() != expected {
            return Err(format!(
                "bias_table has {} values, expected {} ({}, {}, {})",
                values.len(),
                expected,
                EFCC18_N_TILES,
                EFCC18_N_CATALOGS,
                EFCC18_N_COMPONENTS
            ));
        }
        Ok(Self { values })
    }

    /// Parse the text of `bias.dat`: `!`-prefixed header/comment lines and
    /// blank lines are skipped, every remaining line carries the 104 numbers
    /// of one tile, and exactly `EFCC18_N_TILES` data lines are required.
    pub fn parse_bias_dat(text: &str) -> Result<Self, String> {
        let mut values = Vec::with_capacity(EFCC18_N_TILES * NUMBERS_PER_ROW);
        let mut rows = 0_usize;
        for (line_number, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('!') {
                continue;
            }
            let mut count = 0_usize;
            for token in trimmed.split_ascii_whitespace() {
                let value: f64 = token.parse().map_err(|_| {
                    format!(
                        "bias.dat line {}: could not parse {token:?} as a number",
                        line_number + 1
                    )
                })?;
                values.push(value as f32);
                count += 1;
            }
            if count != NUMBERS_PER_ROW {
                return Err(format!(
                    "Unexpected bias.dat layout: line {} has {count} values; expected \
                     ({EFCC18_N_TILES}, {NUMBERS_PER_ROW})",
                    line_number + 1
                ));
            }
            rows += 1;
        }
        if rows != EFCC18_N_TILES {
            return Err(format!(
                "Unexpected bias.dat layout ({rows}, {NUMBERS_PER_ROW}); expected \
                 ({EFCC18_N_TILES}, {NUMBERS_PER_ROW})"
            ));
        }
        Self::from_values(values)
    }

    /// Flat tile-major values `(EFCC18_N_TILES * 26 * 4)`.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// `(dRA_arcsec, dDec_arcsec, pmRA_mas_yr, pmDec_mas_yr)` for one tile
    /// and catalog column.
    pub fn cell(&self, tile: usize, column: usize) -> [f32; EFCC18_N_COMPONENTS] {
        let offset = (tile * EFCC18_N_CATALOGS + column) * EFCC18_N_COMPONENTS;
        [
            self.values[offset],
            self.values[offset + 1],
            self.values[offset + 2],
            self.values[offset + 3],
        ]
    }
}

/// `BIAS_VERSION` tag from the `!` header of a `bias.dat` text, if present.
pub fn read_efcc18_bias_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if !line.starts_with('!') {
            break;
        }
        if let Some((_, version)) = line.split_once("BIAS_VERSION=") {
            return Some(version.trim().to_string());
        }
    }
    None
}

/// EFCC18 column index per observation, or -1 where not covered (null
/// catalog, catalog not tabulated, or catalog listed in `exclude_astcats`).
pub fn efcc18_catalog_columns(astcats: &[Option<&str>], exclude_astcats: &[&str]) -> Vec<i64> {
    astcats
        .iter()
        .map(|astcat| match astcat {
            Some(name) if !exclude_astcats.contains(name) => efcc18_catalog_column(name)
                .map(|column| column as i64)
                .unwrap_or(-1),
            _ => -1,
        })
        .collect()
}

/// Which `astcats` receive an EFCC18 correction.
pub fn is_efcc18_covered(astcats: &[Option<&str>], exclude_astcats: &[&str]) -> Vec<bool> {
    efcc18_catalog_columns(astcats, exclude_astcats)
        .into_iter()
        .map(|column| column >= 0)
        .collect()
}

/// Per-observation EFCC18 corrections `(bias_ra_arcsec, bias_dec_arcsec)` in
/// the cos(dec)-corrected tangent-plane frame. Rows whose catalog is not
/// covered, or whose position or epoch is not finite, receive zero.
pub fn compute_efcc18_corrections(
    ra_deg: &[f64],
    dec_deg: &[f64],
    astcats: &[Option<&str>],
    jd_tdb: &[f64],
    table: &Efcc18BiasTable,
    exclude_astcats: &[&str],
) -> Result<Vec<[f64; 2]>, String> {
    let n = ra_deg.len();
    if dec_deg.len() != n || astcats.len() != n || jd_tdb.len() != n {
        return Err("ra_deg, dec_deg, astcats and jd_tdb must all have length N".to_string());
    }
    let columns = efcc18_catalog_columns(astcats, exclude_astcats);
    let mut corrections = vec![[0.0_f64; 2]; n];
    for row in 0..n {
        let column = columns[row];
        if column < 0
            || !ra_deg[row].is_finite()
            || !dec_deg[row].is_finite()
            || !jd_tdb[row].is_finite()
        {
            continue;
        }
        let tile = ra_dec_to_healpix(ra_deg[row], dec_deg[row])?;
        let cell = table.cell(tile as usize, column as usize);
        let years_since_j2000 = (jd_tdb[row] - JD_J2000) / DAYS_PER_JULIAN_YEAR;
        // bias.dat stores proper motions in mas/yr; /1000 converts to arcsec/yr.
        corrections[row] = [
            f64::from(cell[0]) + years_since_j2000 * f64::from(cell[2]) / 1000.0,
            f64::from(cell[1]) + years_since_j2000 * f64::from(cell[3]) / 1000.0,
        ];
    }
    Ok(corrections)
}

/// Subtract EFCC18 corrections from observed positions.
///
/// Rows are corrected when `covered` is set and the position and cos(dec)
/// are finite with |cos(dec)| above the pole floor; the RA result is wrapped
/// to `[0, 360)`. Returns `None` when no position changed (the caller then
/// keeps its input object), matching `EFCC18DebiasModel.apply`.
pub fn efcc18_debias_positions(
    lon_deg: &[f64],
    lat_deg: &[f64],
    corrections: &[[f64; 2]],
    covered: &[bool],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut new_lon = lon_deg.to_vec();
    let mut new_lat = lat_deg.to_vec();
    let mut any_valid = false;
    for row in 0..lon_deg.len() {
        let lon = lon_deg[row];
        let lat = lat_deg[row];
        let cos_dec = lat.to_radians().cos();
        let valid = covered[row]
            && lon.is_finite()
            && lat.is_finite()
            && cos_dec.is_finite()
            && cos_dec.abs() > POLE_COS_DEC_FLOOR;
        if !valid {
            continue;
        }
        any_valid = true;
        new_lon[row] = (lon - corrections[row][0] / (3600.0 * cos_dec)).rem_euclid(360.0);
        new_lat[row] = lat - corrections[row][1] / 3600.0;
    }
    if !any_valid {
        return None;
    }
    let unchanged = new_lon
        .iter()
        .zip(lon_deg)
        .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
        && new_lat
            .iter()
            .zip(lat_deg)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
    if unchanged {
        None
    } else {
        Some((new_lon, new_lat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::healpix::ang2pix_lonlat;

    // Tile centres from JPL's debias_2018.tgz `tiles.dat` (tile number, RA
    // [rad], Dec [rad]), whose rows the archive README says are "sorted in the
    // same way" as bias.dat. They pin the row order of bias.dat to HEALPix
    // RING ordering: the ring-scheme pixel centres reproduce all 49152 of
    // them, the nested scheme reproduces 1. Anchors span both polar caps and
    // the equatorial belt.
    #[allow(clippy::approx_constant)] // published tile centres, not pi fractions
    const JPL_TILES_DAT_ANCHORS: [(i64, f64, f64); 8] = [
        (0, 0.785398, 1.558038),
        (4, 0.392699, 1.545280),
        (100, 3.702591, 1.481462),
        (4096, 4.764749, 0.988506),
        (24576, 3.153864, 0.000000),
        (29999, 4.295146, -0.220533),
        (45055, 1.518436, -0.988506),
        (49151, 5.497787, -1.558038),
    ];

    fn synthetic_table() -> Efcc18BiasTable {
        // Deterministic (tile, catalog, component)-encoded table.
        let mut values = vec![0.0_f32; EFCC18_N_TILES * NUMBERS_PER_ROW];
        for tile in 0..EFCC18_N_TILES {
            for catalog in 0..EFCC18_N_CATALOGS {
                let offset = (tile * EFCC18_N_CATALOGS + catalog) * EFCC18_N_COMPONENTS;
                values[offset] = 0.001 * catalog as f32 + 0.0001 * (tile % 7) as f32;
                values[offset + 1] = 0.002 * catalog as f32 + 0.0001 * (tile % 5) as f32;
            }
        }
        Efcc18BiasTable::from_values(values).unwrap()
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(EFCC18_CATALOG_CODES.chars().count(), EFCC18_N_CATALOGS);
        assert_eq!(EFCC18_N_TILES, 49152);
        for (name, code) in MPC_ASTCAT_TO_EFCC18 {
            assert!(EFCC18_CATALOG_CODES.contains(code), "{name} -> {code}");
        }
        let mut codes: Vec<char> = MPC_ASTCAT_TO_EFCC18.iter().map(|(_, c)| *c).collect();
        codes.sort_unstable();
        let mut expected: Vec<char> = EFCC18_CATALOG_CODES.chars().collect();
        expected.sort_unstable();
        assert_eq!(codes, expected);
        for absent in ["Gaia2", "Gaia3", "Gaia3E", "ATLAS", "ATLAS2", "PS1_DR1"] {
            assert_eq!(efcc18_catalog_code(absent), None);
        }
        for astcat in EFCC18_JPL_UNDEBIASED_ASTCATS {
            assert!(efcc18_catalog_code(astcat).is_some());
        }
        assert_eq!(efcc18_catalog_column("UCAC4"), Some(13));
    }

    #[test]
    fn ring_order_matches_jpl_tiles_dat() {
        // The tile index used to read bias.dat must be the RING index: the
        // centre of tile k in JPL's tiles.dat maps to row k. (Under the
        // nested scheme tile 0, at RA 45 deg near the north pole here, would
        // sit near the equator.)
        let mut nested_hits = 0;
        for (tile, ra_rad, dec_rad) in JPL_TILES_DAT_ANCHORS {
            let ra = ra_rad.to_degrees();
            let dec = dec_rad.to_degrees();
            assert_eq!(ra_dec_to_healpix(ra, dec).unwrap(), tile, "tile {tile}");
            if ang2pix_lonlat(EFCC18_NSIDE, ra, dec, true).unwrap() == tile {
                nested_hits += 1;
            }
        }
        assert!(
            nested_hits <= 1,
            "nested scheme reproduced {nested_hits} anchors"
        );
    }

    #[test]
    fn ring_order_matches_full_jpl_tiles_dat_when_available() {
        // Full 49152-tile parity against the JPL archive's tiles.dat when a
        // local copy is named by ADAM_CORE_EFCC18_TILES_DAT (not bundled).
        let Ok(path) = std::env::var("ADAM_CORE_EFCC18_TILES_DAT") else {
            return;
        };
        let text = std::fs::read_to_string(&path).expect("tiles.dat readable");
        let mut checked = 0_usize;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('!') {
                continue;
            }
            let fields: Vec<&str> = trimmed.split_ascii_whitespace().collect();
            let tile: i64 = fields[0].parse().unwrap();
            let ra: f64 = fields[1].parse::<f64>().unwrap().to_degrees();
            let dec: f64 = fields[2].parse::<f64>().unwrap().to_degrees();
            assert_eq!(ra_dec_to_healpix(ra, dec).unwrap(), tile, "tile {tile}");
            checked += 1;
        }
        assert_eq!(checked, EFCC18_N_TILES);
    }

    #[test]
    fn ra_wrap_is_harmless() {
        assert_eq!(
            ra_dec_to_healpix(-10.0, 5.0).unwrap(),
            ra_dec_to_healpix(350.0, 5.0).unwrap()
        );
        assert_eq!(
            ra_dec_to_healpix(370.0, 5.0).unwrap(),
            ra_dec_to_healpix(10.0, 5.0).unwrap()
        );
        assert_ne!(
            ra_dec_to_healpix(0.0, 0.0).unwrap(),
            ra_dec_to_healpix(180.0, 0.0).unwrap()
        );
    }

    #[test]
    fn parse_bias_dat_round_trip_and_layout_errors() {
        let table = synthetic_table();
        let mut text = String::from(
            "! BIAS_VERSION= synthetic (unit test)\n! NSIDE= 64\n! NPIX= 49152\n!\n\n",
        );
        for tile in 0..EFCC18_N_TILES {
            let row: Vec<String> = (0..NUMBERS_PER_ROW)
                .map(|k| format!("{:.4}", table.values()[tile * NUMBERS_PER_ROW + k]))
                .collect();
            text.push_str(&row.join(" "));
            text.push('\n');
        }
        assert_eq!(
            read_efcc18_bias_version(&text).as_deref(),
            Some("synthetic (unit test)")
        );
        let parsed = Efcc18BiasTable::parse_bias_dat(&text).unwrap();
        for tile in [0, 10, 12345, EFCC18_N_TILES - 1] {
            for catalog in 0..EFCC18_N_CATALOGS {
                let expected = table.cell(tile, catalog);
                let actual = parsed.cell(tile, catalog);
                for k in 0..EFCC18_N_COMPONENTS {
                    assert!(
                        (expected[k] - actual[k]).abs() < 5e-5,
                        "{tile} {catalog} {k}"
                    );
                }
            }
        }
        // Known cell: tile 10, catalog 'g' (index 5).
        assert!((parsed.cell(10, 5)[0] - (0.001 * 5.0 + 0.0001 * 3.0)).abs() < 5e-5);

        let short = "! BIAS_VERSION= bad\n0.0 0.0\n";
        let err = Efcc18BiasTable::parse_bias_dat(short).unwrap_err();
        assert!(err.contains("Unexpected bias.dat layout"), "{err}");
        assert_eq!(read_efcc18_bias_version("0.0 0.0\n"), None);
        assert!(Efcc18BiasTable::from_values(vec![0.0; 3 * 26 * 4]).is_err());
    }

    #[test]
    fn corrections_read_the_right_cell_and_apply_proper_motion() {
        let table = synthetic_table();
        let ucac4 = efcc18_catalog_column("UCAC4").unwrap();
        let out = compute_efcc18_corrections(
            &[100.0],
            &[30.0],
            &[Some("UCAC4")],
            &[JD_J2000],
            &table,
            &[],
        )
        .unwrap();
        let tile = ra_dec_to_healpix(100.0, 30.0).unwrap() as usize;
        assert_eq!(out[0][0], f64::from(table.cell(tile, ucac4)[0]));
        assert_eq!(out[0][1], f64::from(table.cell(tile, ucac4)[1]));

        // Uncovered, unknown and excluded catalogs are zero.
        let out = compute_efcc18_corrections(
            &[10.0, 200.0, 40.0, 10.0],
            &[-5.0, 25.0, 0.0, 5.0],
            &[Some("Gaia2"), Some("Gaia3E"), None, Some("Tycho")],
            &[2_459_200.0, 2_460_000.0, 2_460_000.0, JD_J2000],
            &table,
            &EFCC18_JPL_UNDEBIASED_ASTCATS,
        )
        .unwrap();
        assert!(out.iter().all(|row| *row == [0.0, 0.0]));

        // Proper motion: 100 mas/yr over ten years is one arcsecond, linear in time.
        let mut values = vec![0.0_f32; EFCC18_N_TILES * NUMBERS_PER_ROW];
        let offset = ucac4 * EFCC18_N_COMPONENTS;
        values[offset + 2] = 100.0;
        values[offset + 3] = -50.0;
        let pm_table = Efcc18BiasTable::from_values(values).unwrap();
        let (ra0, dec0) = (
            JPL_TILES_DAT_ANCHORS[0].1.to_degrees(),
            JPL_TILES_DAT_ANCHORS[0].2.to_degrees(),
        );
        let ten = compute_efcc18_corrections(
            &[ra0],
            &[dec0],
            &[Some("UCAC4")],
            &[JD_J2000 + 10.0 * DAYS_PER_JULIAN_YEAR],
            &pm_table,
            &[],
        )
        .unwrap();
        assert!((ten[0][0] - 1.0).abs() < 1e-9 && (ten[0][1] + 0.5).abs() < 1e-9);
        let twenty = compute_efcc18_corrections(
            &[ra0],
            &[dec0],
            &[Some("UCAC4")],
            &[JD_J2000 + 20.0 * DAYS_PER_JULIAN_YEAR],
            &pm_table,
            &[],
        )
        .unwrap();
        assert!((twenty[0][0] - 2.0).abs() < 1e-9 && (twenty[0][1] + 1.0).abs() < 1e-9);

        assert!(compute_efcc18_corrections(
            &[1.0, 2.0],
            &[0.0],
            &[Some("UCAC4"); 2],
            &[0.0; 2],
            &table,
            &[]
        )
        .is_err());
    }

    #[test]
    fn debias_positions_wraps_and_skips_poles() {
        let corrections = vec![[0.7, -0.4]; 3];
        let covered = vec![true, true, true];
        let (lon, lat) = efcc18_debias_positions(
            &[1e-6, 359.99999, 100.0],
            &[10.0, 10.0, 90.0],
            &corrections,
            &covered,
        )
        .unwrap();
        assert!(lon[0] > 359.0 && lon[0] < 360.0);
        assert!((lat[0] - (10.0 + 0.4 / 3600.0)).abs() < 1e-12);
        assert!(lon[1] < 360.0);
        // Pole: RA correction undefined -> untouched.
        assert_eq!(lon[2], 100.0);
        assert_eq!(lat[2], 90.0);
        // Nothing covered, or zero corrections, leave the input alone.
        assert!(efcc18_debias_positions(&[1.0], &[2.0], &[[0.7, 0.1]], &[false]).is_none());
        assert!(efcc18_debias_positions(&[1.0], &[2.0], &[[0.0, 0.0]], &[true]).is_none());
        // Non-finite positions pass through.
        assert!(efcc18_debias_positions(&[f64::NAN], &[2.0], &[[0.7, 0.1]], &[true]).is_none());
    }
}
