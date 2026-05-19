//! Origin helpers for Rust-native data-model and propagation workflows.
//!
//! The gravitational-parameter table mirrors `adam_core.coordinates.origin`:
//! values are sourced from the same km^3/s^2 constants and converted to
//! au^3/day^2 with the project AU/day constants. Unknown origins fail loudly.

use super::time::SECONDS_PER_DAY;
use super::{OriginId, SchemaError, SchemaResult};

pub const KM_PER_AU: f64 = 149_597_870.700;

pub fn convert_mu_km3_s2_to_au3_day2(mu: f64) -> f64 {
    mu / KM_PER_AU.powi(3) * SECONDS_PER_DAY.powi(2)
}

pub fn origin_mu_au3_day2(origin: &OriginId) -> SchemaResult<f64> {
    match origin {
        OriginId::SolarSystemBarycenter => Ok(solar_system_barycenter_mu_au3_day2()),
        OriginId::Naif(code) => {
            let Some(name) = naif_origin_name(*code) else {
                return Err(SchemaError::UnsupportedOrigin(format!("NAIF:{code}")));
            };
            origin_code_mu_au3_day2(name)
                .ok_or_else(|| SchemaError::UnsupportedOrigin(name.to_string()))
        }
        OriginId::Named(code) => origin_code_mu_au3_day2(code)
            .ok_or_else(|| SchemaError::UnsupportedOrigin(code.clone())),
    }
}

pub fn origin_code_mu_au3_day2(code: &str) -> Option<f64> {
    match code {
        "SOLAR_SYSTEM_BARYCENTER" => Some(solar_system_barycenter_mu_au3_day2()),
        "MERCURY_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(22_032.080_486_418)),
        "VENUS_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(324_858.592_000)),
        "MARS_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(42_828.375_816)),
        "JUPITER_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(126_712_764.100_000)),
        "SATURN_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(37_940_584.841_800)),
        "URANUS_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(5_794_556.400_000)),
        "NEPTUNE_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(6_836_527.100_580)),
        "PLUTO_BARYCENTER" => Some(convert_mu_km3_s2_to_au3_day2(975.500_000)),
        "SUN" => Some(convert_mu_km3_s2_to_au3_day2(132_712_440_041.279_42)),
        "MERCURY" => Some(convert_mu_km3_s2_to_au3_day2(22_031.868_551)),
        "VENUS" => Some(convert_mu_km3_s2_to_au3_day2(324_858.592_000)),
        "EARTH" => Some(convert_mu_km3_s2_to_au3_day2(398_600.435_507)),
        "MOON" => Some(convert_mu_km3_s2_to_au3_day2(4_902.800_118)),
        _ => None,
    }
}

pub fn naif_origin_name(code: i32) -> Option<&'static str> {
    match code {
        0 => Some("SOLAR_SYSTEM_BARYCENTER"),
        1 => Some("MERCURY_BARYCENTER"),
        2 => Some("VENUS_BARYCENTER"),
        4 => Some("MARS_BARYCENTER"),
        5 => Some("JUPITER_BARYCENTER"),
        6 => Some("SATURN_BARYCENTER"),
        7 => Some("URANUS_BARYCENTER"),
        8 => Some("NEPTUNE_BARYCENTER"),
        9 => Some("PLUTO_BARYCENTER"),
        10 => Some("SUN"),
        199 => Some("MERCURY"),
        299 => Some("VENUS"),
        301 => Some("MOON"),
        399 => Some("EARTH"),
        _ => None,
    }
}

pub fn solar_system_barycenter_mu_au3_day2() -> f64 {
    [
        132_712_440_041.279_42,
        22_032.080_486_418,
        324_858.592_000,
        398_600.435_507,
        4_902.800_118,
        42_828.375_816,
        126_712_764.100_000,
        37_940_584.841_800,
        5_794_556.400_000,
        6_836_527.100_580,
        975.500_000,
    ]
    .into_iter()
    .map(convert_mu_km3_s2_to_au3_day2)
    .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_origin_mu_values_match_python_origin_mu_contract() {
        let expected = [
            ("SOLAR_SYSTEM_BARYCENTER", 0.000_296_309_274_608_136_2),
            ("SUN", 0.000_295_912_208_284_119_56),
            ("MERCURY_BARYCENTER", 4.912_547_450_564_196e-11),
            ("EARTH", 8.887_692_446_706_6e-10),
            ("MOON", 1.093_189_462_300_414_3e-11),
        ];
        for (code, expected_mu) in expected {
            let actual = origin_mu_au3_day2(&OriginId::Named(code.to_string())).unwrap();
            assert!((actual - expected_mu).abs() < 1.0e-24, "{code}: {actual}");
        }
    }

    #[test]
    fn named_sun_and_naif_sun_have_same_mu() {
        let named = origin_mu_au3_day2(&OriginId::Named("SUN".to_string())).unwrap();
        let naif = origin_mu_au3_day2(&OriginId::Naif(10)).unwrap();
        assert_eq!(named, naif);
    }

    #[test]
    fn solar_system_barycenter_includes_major_body_terms() {
        let sun = origin_code_mu_au3_day2("SUN").unwrap();
        let ssb = origin_mu_au3_day2(&OriginId::SolarSystemBarycenter).unwrap();
        assert!(ssb > sun);
    }

    #[test]
    fn unsupported_origin_fails_loudly() {
        let err =
            origin_mu_au3_day2(&OriginId::Named("EARTH_MOON_BARYCENTER".to_string())).unwrap_err();
        assert!(
            matches!(err, SchemaError::UnsupportedOrigin(origin) if origin == "EARTH_MOON_BARYCENTER")
        );
    }
}
