//! Time helpers for the standalone Rust data-model prototype.
//!
//! RM-STANDALONE-004 keeps the first Rust-owned behavior intentionally narrow:
//! `TimeArray` stores canonical epoch batches, and Rust can perform the
//! already-established TDB -> ET arithmetic without crossing Python.
//! RM-STANDALONE-004A adds ERFA-backed UTC <-> TAI conversion and ports the
//! existing project-local TAI/TT and TT/TDB policies behind the same fixture.

use super::{Epoch, SchemaError, SchemaResult, TimeArray, TimeScale, NANOS_PER_DAY};

pub const SECONDS_PER_DAY: f64 = 86_400.0;
pub const J2000_TDB_MJD: f64 = 51_544.5;
pub const TAI_TT_NANOS: i64 = 32_184_000_000;

const MJD_TO_JD_DAY_OFFSET: f64 = 2_400_000.0;
const MJD_TO_JD_FRACTION_OFFSET: f64 = 0.5;
const NANOS_PER_DAY_F64: f64 = NANOS_PER_DAY as f64;
const NANOS_PER_SECOND_F64: f64 = 1_000_000_000.0;

pub trait TimeScaleProvider {
    fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray>;
}

enum ErfaScaleConverter {
    UtcTai,
    TaiUtc,
}

impl ErfaScaleConverter {
    fn name(&self) -> &'static str {
        match self {
            Self::UtcTai => "eraUtctai",
            Self::TaiUtc => "eraTaiutc",
        }
    }

    fn convert(&self, day: f64, fraction: f64) -> SchemaResult<(f64, f64)> {
        let result = match self {
            Self::UtcTai => erfars::timescales::Utctai(day, fraction),
            Self::TaiUtc => erfars::timescales::Taiutc(day, fraction),
        };
        result
            .map(|((new_day, new_fraction), _warning)| (new_day, new_fraction))
            .map_err(|code| {
                SchemaError::InvalidRecordBatch(format!(
                    "{} failed with ERFA status {code}",
                    self.name()
                ))
            })
    }
}

impl Epoch {
    pub fn mjd(self) -> f64 {
        self.days as f64 + self.nanos as f64 / NANOS_PER_DAY_F64
    }
}

impl TimeArray {
    pub fn mjd_values(&self) -> Vec<f64> {
        self.epochs.iter().map(|epoch| epoch.mjd()).collect()
    }

    pub fn rescale(&self, new_scale: TimeScale) -> SchemaResult<Self> {
        if self.scale == new_scale {
            return Ok(self.clone());
        }
        reject_unsupported_rescale(self.scale, new_scale)?;

        let correction = self.rescale_correction_nanos(new_scale)?;
        self.add_nanos_with_scale(new_scale, &correction)
    }

    pub fn rescale_with_provider(
        &self,
        new_scale: TimeScale,
        provider: &dyn TimeScaleProvider,
    ) -> SchemaResult<Self> {
        if self.scale == new_scale {
            return Ok(self.clone());
        }
        if requires_time_scale_provider(self.scale, new_scale) {
            return provider.rescale(self, new_scale);
        }
        self.rescale(new_scale)
    }

    pub fn utc_to_tai_erfa(&self) -> SchemaResult<Self> {
        if self.scale != TimeScale::Utc {
            return Err(SchemaError::InvalidTimeScale(format!(
                "UTC to TAI conversion requires utc scale; got {}",
                self.scale.as_str()
            )));
        }
        let zeros = vec![0; self.len()];
        let correction = self.leap_seconds_correction(ErfaScaleConverter::UtcTai, &zeros)?;
        self.add_nanos_with_scale(TimeScale::Tai, &correction)
    }

    pub fn tai_to_utc_erfa(&self) -> SchemaResult<Self> {
        if self.scale != TimeScale::Tai {
            return Err(SchemaError::InvalidTimeScale(format!(
                "TAI to UTC conversion requires tai scale; got {}",
                self.scale.as_str()
            )));
        }
        let zeros = vec![0; self.len()];
        let correction = self.leap_seconds_correction(ErfaScaleConverter::TaiUtc, &zeros)?;
        self.add_nanos_with_scale(TimeScale::Utc, &correction)
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

    fn rescale_correction_nanos(&self, new_scale: TimeScale) -> SchemaResult<Vec<i64>> {
        let zeros = vec![0; self.len()];
        match self.scale {
            TimeScale::Tt => match new_scale {
                TimeScale::Tai => Ok(constant_correction(self.len(), -TAI_TT_NANOS)),
                TimeScale::Utc => {
                    let leap = self.leap_seconds_correction(ErfaScaleConverter::TaiUtc, &zeros)?;
                    Ok(add_constant(&leap, -TAI_TT_NANOS))
                }
                TimeScale::Tdb => Ok(self.tt_tdb_correction(true, &zeros)),
                _ => unsupported_rescale(self.scale, new_scale),
            },
            TimeScale::Utc => {
                let utc_to_tai =
                    self.leap_seconds_correction(ErfaScaleConverter::UtcTai, &zeros)?;
                match new_scale {
                    TimeScale::Tai => Ok(utc_to_tai),
                    TimeScale::Tt => Ok(add_constant(&utc_to_tai, TAI_TT_NANOS)),
                    TimeScale::Tdb => {
                        let tt_offset = add_constant(&utc_to_tai, TAI_TT_NANOS);
                        let tt_tdb = self.tt_tdb_correction(true, &tt_offset);
                        Ok(add_constant(
                            &add_corrections(&utc_to_tai, &tt_tdb),
                            TAI_TT_NANOS,
                        ))
                    }
                    _ => unsupported_rescale(self.scale, new_scale),
                }
            }
            TimeScale::Tai => match new_scale {
                TimeScale::Tt => Ok(constant_correction(self.len(), TAI_TT_NANOS)),
                TimeScale::Tdb => {
                    let tt_tdb = self
                        .tt_tdb_correction(true, &constant_correction(self.len(), TAI_TT_NANOS));
                    Ok(add_constant(&tt_tdb, TAI_TT_NANOS))
                }
                TimeScale::Utc => self.leap_seconds_correction(ErfaScaleConverter::TaiUtc, &zeros),
                _ => unsupported_rescale(self.scale, new_scale),
            },
            TimeScale::Tdb => match new_scale {
                TimeScale::Tt => Ok(self.tt_tdb_correction(false, &zeros)),
                TimeScale::Tai => {
                    let tdb_tt = self.tt_tdb_correction(false, &zeros);
                    Ok(add_constant(&tdb_tt, -TAI_TT_NANOS))
                }
                TimeScale::Utc => {
                    let mut correction =
                        add_constant(&self.tt_tdb_correction(false, &zeros), -TAI_TT_NANOS);
                    let leap =
                        self.leap_seconds_correction(ErfaScaleConverter::TaiUtc, &correction)?;
                    correction = add_corrections(&correction, &leap);
                    Ok(correction)
                }
                _ => unsupported_rescale(self.scale, new_scale),
            },
            TimeScale::Ut1 | TimeScale::Gps => unsupported_rescale(self.scale, new_scale),
        }
    }

    fn leap_seconds_correction(
        &self,
        converter: ErfaScaleConverter,
        correction: &[i64],
    ) -> SchemaResult<Vec<i64>> {
        if correction.len() != self.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.correction".to_string(),
                expected: self.len(),
                actual: correction.len(),
            });
        }

        self.epochs
            .iter()
            .zip(correction.iter().copied())
            .map(|(epoch, correction)| {
                let old_day = epoch.days as f64 + MJD_TO_JD_DAY_OFFSET;
                let old_fraction = (epoch.nanos as f64 + correction as f64) / NANOS_PER_DAY_F64
                    + MJD_TO_JD_FRACTION_OFFSET;
                let (new_day, new_fraction) = converter.convert(old_day, old_fraction)?;
                let delta_nanos =
                    ((new_day - old_day) + (new_fraction - old_fraction)) * NANOS_PER_DAY_F64;
                Ok((delta_nanos / NANOS_PER_SECOND_F64).round() as i64 * 1_000_000_000)
            })
            .collect()
    }

    fn tt_tdb_correction(&self, positive: bool, correction: &[i64]) -> Vec<i64> {
        self.epochs
            .iter()
            .zip(correction.iter().copied())
            .map(|(epoch, correction)| {
                let days = (epoch.days - 51_545) as f64;
                let fractions = (epoch.nanos as f64 + correction as f64) / NANOS_PER_DAY_F64 + 0.5;
                let centuries = (days + fractions) / 36_525.0;
                let anomaly = (35_999.050 * centuries + 357.528).to_radians();
                let mut delta = (anomaly + 0.0167 * anomaly.sin()).sin() * 1_658_000.0;
                if !positive {
                    delta = -delta;
                }
                delta as i64
            })
            .collect()
    }

    fn add_nanos_with_scale(&self, scale: TimeScale, correction: &[i64]) -> SchemaResult<Self> {
        if correction.len() != self.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.correction".to_string(),
                expected: self.len(),
                actual: correction.len(),
            });
        }
        Self::new(
            scale,
            self.epochs
                .iter()
                .zip(correction.iter().copied())
                .map(|(epoch, correction)| Epoch::new(epoch.days, epoch.nanos + correction))
                .collect(),
        )
    }
}

fn reject_unsupported_rescale(scale: TimeScale, new_scale: TimeScale) -> SchemaResult<()> {
    if requires_time_scale_provider(scale, new_scale) {
        return unsupported_rescale(scale, new_scale);
    }
    Ok(())
}

fn requires_time_scale_provider(scale: TimeScale, new_scale: TimeScale) -> bool {
    matches!(scale, TimeScale::Ut1 | TimeScale::Gps)
        || matches!(new_scale, TimeScale::Ut1 | TimeScale::Gps)
}

fn unsupported_rescale<T>(scale: TimeScale, new_scale: TimeScale) -> SchemaResult<T> {
    Err(SchemaError::InvalidTimeScale(format!(
        "rescale from {} to {} is not supported",
        scale.as_str(),
        new_scale.as_str()
    )))
}

fn constant_correction(len: usize, value: i64) -> Vec<i64> {
    vec![value; len]
}

fn add_constant(values: &[i64], value: i64) -> Vec<i64> {
    values.iter().map(|item| item + value).collect()
}

fn add_corrections(left: &[i64], right: &[i64]) -> Vec<i64> {
    left.iter().zip(right.iter()).map(|(a, b)| a + b).collect()
}

// --- Timestamp op surface (bead personal-cmy.25) --------------------------------
//
// Array-level ports of the legacy Python `Timestamp` arithmetic, matching the
// pyarrow semantics bit-for-bit (gated by the frozen legacy fixture from bead
// personal-cmy.19). Carry/normalization is consolidated on `Epoch::new`
// (euclidean day carry), which subsumes the legacy one-day overflow branches.

impl TimeArray {
    /// Legacy `Timestamp.add_nanos`: element-wise add with day-boundary
    /// carry. When `check_range` is set, deltas must lie in
    /// `[-86400e9, 86400e9)` and violations raise the legacy
    /// "Nanoseconds out of range" error.
    pub fn add_nanos_checked(&self, delta: &[i64], check_range: bool) -> SchemaResult<Self> {
        if delta.len() != self.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.add_nanos".to_string(),
                expected: self.len(),
                actual: delta.len(),
            });
        }
        if check_range
            && delta
                .iter()
                .any(|&value| !(-NANOS_PER_DAY..NANOS_PER_DAY).contains(&value))
        {
            return Err(SchemaError::InvalidRecordBatch(
                "Nanoseconds out of range".to_string(),
            ));
        }
        self.add_nanos_with_scale(self.scale, delta)
    }

    /// Legacy `Timestamp.add_days`: day-only add, nanos untouched.
    pub fn add_days(&self, delta: &[i64]) -> SchemaResult<Self> {
        if delta.len() != self.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.add_days".to_string(),
                expected: self.len(),
                actual: delta.len(),
            });
        }
        Self::new(
            self.scale,
            self.epochs
                .iter()
                .zip(delta.iter().copied())
                .map(|(epoch, delta)| Epoch::new(epoch.days + delta, epoch.nanos))
                .collect(),
        )
    }

    /// Legacy `Timestamp.add_fractional_days`: `floor` day part, truncating
    /// float->int cast for the nano part (pyarrow `allow_float_truncate`),
    /// then day/nano adds.
    pub fn add_fractional_days(&self, fractional_days: &[f64]) -> SchemaResult<Self> {
        if fractional_days.len() != self.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.add_fractional_days".to_string(),
                expected: self.len(),
                actual: fractional_days.len(),
            });
        }
        let mut delta_days = Vec::with_capacity(fractional_days.len());
        let mut delta_nanos = Vec::with_capacity(fractional_days.len());
        for &value in fractional_days {
            let day_part = value.floor();
            let nano_part = value - day_part;
            delta_days.push(day_part as i64);
            delta_nanos.push((nano_part * NANOS_PER_DAY_F64).trunc() as i64);
        }
        self.add_days(&delta_days)?
            .add_nanos_checked(&delta_nanos, true)
    }

    /// Legacy `Timestamp.difference` / `difference_scalar`: element-wise
    /// `self - other`, normalized so nanos land in `[0, 86400e9)` (the
    /// euclidean normalization in `Epoch::new`).
    pub fn difference(&self, other: &Self) -> SchemaResult<(Vec<i64>, Vec<i64>)> {
        if other.len() != self.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.difference".to_string(),
                expected: self.len(),
                actual: other.len(),
            });
        }
        let mut days = Vec::with_capacity(self.len());
        let mut nanos = Vec::with_capacity(self.len());
        for (a, b) in self.epochs.iter().zip(&other.epochs) {
            let normalized = Epoch::new(a.days - b.days, a.nanos - b.nanos);
            days.push(normalized.days);
            nanos.push(normalized.nanos);
        }
        Ok((days, nanos))
    }

    /// Legacy `Timestamp.from_mjd`: `floor` day split and half-even rounding
    /// of the fractional part to nanoseconds (pyarrow `round` default).
    pub fn from_mjd(scale: TimeScale, mjd: &[f64]) -> SchemaResult<Self> {
        let epochs = mjd
            .iter()
            .map(|&value| {
                let day = value.floor();
                let fraction = value - day;
                Epoch::new(
                    day as i64,
                    (fraction * NANOS_PER_DAY_F64).round_ties_even() as i64,
                )
            })
            .collect();
        Self::new(scale, epochs)
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

    fn scale(value: &Value) -> TimeScale {
        TimeScale::parse(value.as_str().expect("fixture scale must be a string")).unwrap()
    }

    fn time_array(payload: &Value, scale: TimeScale) -> TimeArray {
        TimeArray::from_parts(
            scale,
            i64_array(payload.get("days").expect("payload must contain days")),
            i64_array(payload.get("nanos").expect("payload must contain nanos")),
        )
        .unwrap()
    }

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../migration/artifacts/time_scale_rescale_fixture_2026-05-15.json"
        ))
        .unwrap()
    }

    struct FixtureTimeScaleProvider {
        fixture: Value,
    }

    impl TimeScaleProvider for FixtureTimeScaleProvider {
        fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray> {
            let cases = self
                .fixture
                .get("rescale_correctness_cases")
                .and_then(Value::as_array)
                .expect("fixture must contain rescale correctness cases");
            for case in cases {
                let from_scale = scale(
                    case.get("from_scale")
                        .expect("case must contain from_scale"),
                );
                let to_scale = scale(case.get("to_scale").expect("case must contain to_scale"));
                if times.scale != from_scale || new_scale != to_scale {
                    continue;
                }

                let input = time_array(
                    case.get("input").expect("case must contain input"),
                    from_scale,
                );
                if input.epochs == times.epochs {
                    return Ok(time_array(
                        case.get("output").expect("case must contain output"),
                        to_scale,
                    ));
                }
            }
            unsupported_rescale(times.scale, new_scale)
        }
    }

    fn assert_time_array_eq(actual: &TimeArray, expected: &TimeArray) {
        assert_eq!(actual.scale, expected.scale);
        assert_eq!(actual.epochs, expected.epochs);
    }

    #[test]
    fn rescale_preserves_python_fixture() {
        let fixture = fixture();
        let cases = fixture
            .get("cases")
            .and_then(Value::as_array)
            .expect("fixture must contain rescale cases");
        for case in cases {
            let from_scale = scale(
                case.get("from_scale")
                    .expect("case must contain from_scale"),
            );
            let to_scale = scale(case.get("to_scale").expect("case must contain to_scale"));
            let input = time_array(
                case.get("input").expect("case must contain input"),
                from_scale,
            );
            let expected = time_array(
                case.get("output").expect("case must contain output"),
                to_scale,
            );
            let actual = input.rescale(to_scale).unwrap();
            assert_time_array_eq(&actual, &expected);
        }
    }

    #[test]
    fn rescale_roundtrip_preserves_midnight_regression() {
        let times =
            TimeArray::from_parts(TimeScale::Tdb, vec![59_005, 40_005, 40_000], vec![0, 0, 0])
                .unwrap();
        let round_tripped = times
            .rescale(TimeScale::Utc)
            .unwrap()
            .rescale(TimeScale::Tdb)
            .unwrap();
        assert_eq!(round_tripped.scale, TimeScale::Tdb);
        assert_eq!(round_tripped.epochs, times.epochs);
        assert_ne!(
            round_tripped.epochs,
            vec![
                Epoch {
                    days: 59_004,
                    nanos: NANOS_PER_DAY,
                },
                Epoch {
                    days: 40_004,
                    nanos: NANOS_PER_DAY,
                },
                Epoch {
                    days: 39_999,
                    nanos: NANOS_PER_DAY,
                },
            ]
        );
    }

    #[test]
    fn rescale_with_provider_preserves_python_correctness_matrix() {
        let fixture = fixture();
        let provider = FixtureTimeScaleProvider {
            fixture: fixture.clone(),
        };
        let cases = fixture
            .get("rescale_correctness_cases")
            .and_then(Value::as_array)
            .expect("fixture must contain rescale correctness cases");
        for case in cases {
            let from_scale = scale(
                case.get("from_scale")
                    .expect("case must contain from_scale"),
            );
            let to_scale = scale(case.get("to_scale").expect("case must contain to_scale"));
            let input = time_array(
                case.get("input").expect("case must contain input"),
                from_scale,
            );
            let expected = time_array(
                case.get("output").expect("case must contain output"),
                to_scale,
            );
            let actual = if requires_time_scale_provider(from_scale, to_scale) {
                input.rescale_with_provider(to_scale, &provider).unwrap()
            } else {
                input.rescale(to_scale).unwrap()
            };
            assert_time_array_eq(&actual, &expected);
        }
    }

    #[test]
    fn erfa_utc_tai_helpers_preserve_fixture_rows() {
        let fixture = fixture();
        let cases = fixture
            .get("cases")
            .and_then(Value::as_array)
            .expect("fixture must contain rescale cases");
        let utc_to_tai = cases
            .iter()
            .find(|case| case["from_scale"] == "utc" && case["to_scale"] == "tai")
            .expect("fixture must contain utc->tai case");
        let tai_to_utc = cases
            .iter()
            .find(|case| case["from_scale"] == "tai" && case["to_scale"] == "utc")
            .expect("fixture must contain tai->utc case");

        let utc = time_array(&utc_to_tai["input"], TimeScale::Utc);
        let tai = time_array(&utc_to_tai["output"], TimeScale::Tai);
        assert_time_array_eq(&utc.utc_to_tai_erfa().unwrap(), &tai);

        let tai = time_array(&tai_to_utc["input"], TimeScale::Tai);
        let utc = time_array(&tai_to_utc["output"], TimeScale::Utc);
        assert_time_array_eq(&tai.tai_to_utc_erfa().unwrap(), &utc);
    }

    #[test]
    fn rescale_rejects_ut1_and_gps_without_provider() {
        let times = TimeArray::from_parts(TimeScale::Utc, vec![60_000], vec![0]).unwrap();
        assert!(times.rescale(TimeScale::Ut1).is_err());
        assert!(times.rescale(TimeScale::Gps).is_err());

        let ut1 = TimeArray::from_parts(TimeScale::Ut1, vec![60_000], vec![0]).unwrap();
        assert!(ut1.rescale(TimeScale::Utc).is_err());
    }

    #[test]
    fn tdb_et_seconds_rejects_non_tdb_scales() {
        let times = TimeArray::from_parts(TimeScale::Utc, vec![60_000], vec![0]).unwrap();
        let error = times.tdb_et_seconds().unwrap_err();
        assert!(error.to_string().contains("requires tdb scale"));
    }

    #[test]
    fn tdb_et_seconds_preserves_python_fixture() {
        let fixture = fixture();
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
