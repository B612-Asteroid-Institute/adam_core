//! Time helpers for the standalone Rust data-model prototype.
//!
//! RM-STANDALONE-004 keeps the first Rust-owned behavior intentionally narrow:
//! `TimeArray` stores canonical epoch batches, and Rust can perform the
//! already-established TDB -> ET arithmetic without crossing Python. Cross-scale
//! conversion remains behind the fixture-pinned ERFA/SOFA strategy until the FFI
//! service lands.

use super::{Epoch, SchemaError, SchemaResult, TimeArray, TimeScale, NANOS_PER_DAY};

pub const SECONDS_PER_DAY: f64 = 86_400.0;
pub const J2000_TDB_MJD: f64 = 51_544.5;

impl Epoch {
    pub fn mjd(self) -> f64 {
        self.days as f64 + self.nanos as f64 / NANOS_PER_DAY as f64
    }
}

impl TimeArray {
    pub fn mjd_values(&self) -> Vec<f64> {
        self.epochs.iter().map(|epoch| epoch.mjd()).collect()
    }

    pub fn tdb_et_seconds(&self) -> SchemaResult<Vec<f64>> {
        if self.scale != TimeScale::Tdb {
            return Err(SchemaError::InvalidTimeScale(format!(
                "ET conversion requires tdb scale; got {}",
                self.scale.as_str()
            )));
        }
        Ok(self
            .epochs
            .iter()
            .map(|epoch| (epoch.mjd() - J2000_TDB_MJD) * SECONDS_PER_DAY)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn i64_array(value: &Value) -> Vec<i64> {
        value
            .as_array()
            .expect("fixture field must be an array")
            .iter()
            .map(|item| item.as_i64().expect("fixture value must be i64"))
            .collect()
    }

    fn f64_array(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("fixture field must be an array")
            .iter()
            .map(|item| item.as_f64().expect("fixture value must be f64"))
            .collect()
    }

    #[test]
    fn tdb_et_seconds_rejects_non_tdb_scales() {
        let times = TimeArray::from_parts(TimeScale::Utc, vec![60_000], vec![0]).unwrap();
        let error = times.tdb_et_seconds().unwrap_err();
        assert!(error.to_string().contains("requires tdb scale"));
    }

    #[test]
    fn tdb_et_seconds_preserves_python_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../migration/artifacts/time_scale_rescale_fixture_2026-05-15.json"
        ))
        .unwrap();
        let tdb_cases = fixture
            .get("tdb_et_cases")
            .expect("fixture must contain tdb_et_cases");
        let tdb = tdb_cases
            .get("tdb")
            .expect("tdb_et_cases must contain tdb payload");
        let days = i64_array(tdb.get("days").expect("tdb payload must contain days"));
        let nanos = i64_array(tdb.get("nanos").expect("tdb payload must contain nanos"));
        let expected = f64_array(
            tdb_cases
                .get("et_seconds")
                .expect("tdb_et_cases must contain et_seconds"),
        );

        let times = TimeArray::from_parts(TimeScale::Tdb, days, nanos).unwrap();
        let actual = times.tdb_et_seconds().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let delta = (actual - expected).abs();
            assert!(
                delta <= 1.0e-7,
                "ET fixture mismatch: actual={actual}, expected={expected}, delta={delta}"
            );
        }
    }
}
