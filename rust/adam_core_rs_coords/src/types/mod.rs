//! Canonical Rust data-model prototypes for standalone `adam-core-rs`.
//!
//! These types are intentionally small first implementations of the
//! RM-STANDALONE-002 RFC contracts. They keep Python/quivr and Arrow at the
//! adapter boundary while giving Rust-owned workflows explicit schemas,
//! validation, and error types.

pub mod arrow;
pub mod origin;
pub mod time;
pub use arrow::{
    ArrowSchemaExport, IntoNestedRecordBatch, IntoRecordBatch, TryFromNestedRecordBatch,
    TryFromRecordBatch,
};
pub use origin::{
    convert_mu_km3_s2_to_au3_day2, naif_origin_name, origin_code_mu_au3_day2, origin_mu_au3_day2,
    solar_system_barycenter_mu_au3_day2, KM_PER_AU,
};
pub use time::{TimeScaleProvider, J2000_TDB_MJD, SECONDS_PER_DAY, TAI_TT_NANOS};

use std::fmt;

pub const NANOS_PER_DAY: i64 = 86_400_000_000_000;

pub type SchemaResult<T> = Result<T, SchemaError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    LengthMismatch {
        field: String,
        expected: usize,
        actual: usize,
    },
    InvalidTimeScale(String),
    UnsupportedFrame(String),
    UnsupportedOrigin(String),
    InvalidCovarianceShape {
        rows: usize,
        dimension: usize,
        values: usize,
    },
    InvalidUnit {
        field: String,
        unit: String,
    },
    MissingRequiredField(String),
    UnsupportedNull {
        field: String,
        row: usize,
    },
    InvalidRecordBatch(String),
    Arrow(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "length mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::InvalidTimeScale(scale) => write!(f, "invalid time scale: {scale}"),
            Self::UnsupportedFrame(frame) => write!(f, "unsupported frame: {frame}"),
            Self::UnsupportedOrigin(origin) => write!(f, "unsupported origin: {origin}"),
            Self::InvalidCovarianceShape {
                rows,
                dimension,
                values,
            } => write!(
                f,
                "invalid covariance shape: rows={rows}, dimension={dimension}, values={values}"
            ),
            Self::InvalidUnit { field, unit } => {
                write!(f, "invalid unit for {field}: {unit}")
            }
            Self::MissingRequiredField(field) => write!(f, "missing required field: {field}"),
            Self::UnsupportedNull { field, row } => {
                write!(f, "unsupported null in field {field} at row {row}")
            }
            Self::InvalidRecordBatch(message) => write!(f, "invalid record batch: {message}"),
            Self::Arrow(message) => write!(f, "arrow error: {message}"),
        }
    }
}

impl std::error::Error for SchemaError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeScale {
    Tai,
    Tdb,
    Tt,
    Utc,
    Ut1,
    Gps,
}

impl TimeScale {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tai => "tai",
            Self::Tdb => "tdb",
            Self::Tt => "tt",
            Self::Utc => "utc",
            Self::Ut1 => "ut1",
            Self::Gps => "gps",
        }
    }

    pub fn parse(value: &str) -> SchemaResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "tai" => Ok(Self::Tai),
            "tdb" => Ok(Self::Tdb),
            "tt" => Ok(Self::Tt),
            "utc" => Ok(Self::Utc),
            "ut1" => Ok(Self::Ut1),
            "gps" => Ok(Self::Gps),
            other => Err(SchemaError::InvalidTimeScale(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Epoch {
    pub days: i64,
    pub nanos: i64,
}

impl Epoch {
    pub fn new(days: i64, nanos: i64) -> Self {
        let day_offset = nanos.div_euclid(NANOS_PER_DAY);
        let nanos = nanos.rem_euclid(NANOS_PER_DAY);
        Self {
            days: days + day_offset,
            nanos,
        }
    }

    pub fn validate(&self) -> SchemaResult<()> {
        if (0..NANOS_PER_DAY).contains(&self.nanos) {
            return Ok(());
        }
        Err(SchemaError::InvalidRecordBatch(format!(
            "epoch nanos must satisfy 0 <= nanos < {NANOS_PER_DAY}; got {}",
            self.nanos
        )))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeArray {
    pub scale: TimeScale,
    pub epochs: Vec<Epoch>,
}

impl TimeArray {
    pub fn new(scale: TimeScale, epochs: Vec<Epoch>) -> SchemaResult<Self> {
        let out = Self { scale, epochs };
        out.validate()?;
        Ok(out)
    }

    pub fn from_parts(scale: TimeScale, days: Vec<i64>, nanos: Vec<i64>) -> SchemaResult<Self> {
        if days.len() != nanos.len() {
            return Err(SchemaError::LengthMismatch {
                field: "time.nanos".to_string(),
                expected: days.len(),
                actual: nanos.len(),
            });
        }
        let epochs = days
            .into_iter()
            .zip(nanos)
            .map(|(days, nanos)| Epoch::new(days, nanos))
            .collect();
        Self::new(scale, epochs)
    }

    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        for epoch in &self.epochs {
            epoch.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validity {
    len: usize,
    bits: Vec<u64>,
}

impl Validity {
    pub fn all_valid(len: usize) -> Self {
        let words = len.div_ceil(64);
        let mut bits = vec![u64::MAX; words];
        if !len.is_multiple_of(64) {
            let used_bits = len % 64;
            let mask = (1_u64 << used_bits) - 1;
            if let Some(last) = bits.last_mut() {
                *last = mask;
            }
        }
        Self { len, bits }
    }

    pub fn from_bools(values: &[bool]) -> Self {
        let len = values.len();
        let mut bits = vec![0_u64; len.div_ceil(64)];
        for (index, valid) in values.iter().copied().enumerate() {
            if valid {
                bits[index / 64] |= 1_u64 << (index % 64);
            }
        }
        Self { len, bits }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_valid(&self, index: usize) -> bool {
        assert!(index < self.len, "validity index out of bounds");
        self.bits[index / 64] & (1_u64 << (index % 64)) != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbitId(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectId(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantId(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservatoryCode(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    Ecliptic,
    Equatorial,
    Itrf93,
    Unspecified,
}

impl Frame {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ecliptic => "ecliptic",
            Self::Equatorial => "equatorial",
            Self::Itrf93 => "itrf93",
            Self::Unspecified => "unspecified",
        }
    }

    pub fn parse(value: &str) -> SchemaResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "ecliptic" => Ok(Self::Ecliptic),
            "equatorial" => Ok(Self::Equatorial),
            "itrf93" => Ok(Self::Itrf93),
            "unspecified" => Ok(Self::Unspecified),
            other => Err(SchemaError::UnsupportedFrame(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateRepresentation {
    Cartesian,
    Spherical,
    Keplerian,
    Cometary,
    Geodetic,
}

impl CoordinateRepresentation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cartesian => "cartesian",
            Self::Spherical => "spherical",
            Self::Keplerian => "keplerian",
            Self::Cometary => "cometary",
            Self::Geodetic => "geodetic",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OriginId {
    SolarSystemBarycenter,
    Naif(i32),
    Named(String),
}

impl OriginId {
    pub fn from_code(code: impl Into<String>) -> Self {
        let code = code.into();
        if code == "SOLAR_SYSTEM_BARYCENTER" {
            return Self::SolarSystemBarycenter;
        }
        if let Some(rest) = code.strip_prefix("NAIF:") {
            if let Ok(id) = rest.parse::<i32>() {
                return Self::Naif(id);
            }
        }
        Self::Named(code)
    }

    pub fn code(&self) -> String {
        match self {
            Self::SolarSystemBarycenter => "SOLAR_SYSTEM_BARYCENTER".to_string(),
            Self::Naif(id) => format!("NAIF:{id}"),
            Self::Named(code) => code.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OriginArray {
    pub origins: Vec<OriginId>,
}

impl OriginArray {
    pub fn new(origins: Vec<OriginId>) -> Self {
        Self { origins }
    }

    pub fn repeat(origin: OriginId, len: usize) -> Self {
        Self {
            origins: vec![origin; len],
        }
    }

    pub fn len(&self) -> usize {
        self.origins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.origins.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovarianceUnits {
    Coordinate(CoordinateRepresentation),
    ObservationAngular2D,
    Photometry1D,
    Custom(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceBatch {
    pub rows: usize,
    pub dimension: usize,
    pub values_row_major: Vec<f64>,
    pub row_validity: Option<Validity>,
    pub units: CovarianceUnits,
}

impl CovarianceBatch {
    pub fn new(
        rows: usize,
        dimension: usize,
        values_row_major: Vec<f64>,
        units: CovarianceUnits,
    ) -> SchemaResult<Self> {
        let out = Self {
            rows,
            dimension,
            values_row_major,
            row_validity: None,
            units,
        };
        out.validate()?;
        Ok(out)
    }

    pub fn with_row_validity(mut self, row_validity: Validity) -> SchemaResult<Self> {
        if row_validity.len() != self.rows {
            return Err(SchemaError::LengthMismatch {
                field: "covariance.row_validity".to_string(),
                expected: self.rows,
                actual: row_validity.len(),
            });
        }
        self.row_validity = Some(row_validity);
        Ok(self)
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let expected = self.rows * self.dimension * self.dimension;
        if self.dimension == 0 || self.values_row_major.len() != expected {
            return Err(SchemaError::InvalidCovarianceShape {
                rows: self.rows,
                dimension: self.dimension,
                values: self.values_row_major.len(),
            });
        }
        if let Some(validity) = &self.row_validity {
            if validity.len() != self.rows {
                return Err(SchemaError::LengthMismatch {
                    field: "covariance.row_validity".to_string(),
                    expected: self.rows,
                    actual: validity.len(),
                });
            }
        }
        Ok(())
    }

    pub fn is_row_valid(&self, row: usize) -> bool {
        assert!(row < self.rows, "covariance row index out of bounds");
        self.row_validity
            .as_ref()
            .is_none_or(|validity| validity.is_valid(row))
    }

    pub fn row_values(&self, row: usize) -> &[f64] {
        assert!(row < self.rows, "covariance row index out of bounds");
        let stride = self.dimension * self.dimension;
        let start = row * stride;
        &self.values_row_major[start..start + stride]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoordinateValues {
    Cartesian(Vec<[f64; 6]>),
    Spherical(Vec<[f64; 6]>),
    Keplerian(Vec<[f64; 6]>),
    Cometary(Vec<[f64; 6]>),
    Geodetic(Vec<[f64; 6]>),
}

impl CoordinateValues {
    pub fn len(&self) -> usize {
        match self {
            Self::Cartesian(values)
            | Self::Spherical(values)
            | Self::Keplerian(values)
            | Self::Cometary(values)
            | Self::Geodetic(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn representation(&self) -> CoordinateRepresentation {
        match self {
            Self::Cartesian(_) => CoordinateRepresentation::Cartesian,
            Self::Spherical(_) => CoordinateRepresentation::Spherical,
            Self::Keplerian(_) => CoordinateRepresentation::Keplerian,
            Self::Cometary(_) => CoordinateRepresentation::Cometary,
            Self::Geodetic(_) => CoordinateRepresentation::Geodetic,
        }
    }

    pub fn cartesian(&self) -> Option<&[[f64; 6]]> {
        match self {
            Self::Cartesian(values) => Some(values),
            _ => None,
        }
    }

    pub fn spherical(&self) -> Option<&[[f64; 6]]> {
        match self {
            Self::Spherical(values) => Some(values),
            _ => None,
        }
    }

    /// The raw 6-vectors regardless of representation.
    pub fn raw_values(&self) -> &[[f64; 6]] {
        match self {
            Self::Cartesian(values)
            | Self::Spherical(values)
            | Self::Keplerian(values)
            | Self::Cometary(values)
            | Self::Geodetic(values) => values,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateBatch {
    pub values: CoordinateValues,
    pub frame: Frame,
    pub origins: OriginArray,
    pub times: Option<TimeArray>,
    pub covariance: Option<CovarianceBatch>,
}

impl CoordinateBatch {
    pub fn new(
        values: CoordinateValues,
        frame: Frame,
        origins: OriginArray,
        times: Option<TimeArray>,
        covariance: Option<CovarianceBatch>,
    ) -> SchemaResult<Self> {
        let out = Self {
            values,
            frame,
            origins,
            times,
            covariance,
        };
        out.validate()?;
        Ok(out)
    }

    pub fn cartesian(
        values: Vec<[f64; 6]>,
        frame: Frame,
        origins: OriginArray,
        times: Option<TimeArray>,
        covariance: Option<CovarianceBatch>,
    ) -> SchemaResult<Self> {
        Self::new(
            CoordinateValues::Cartesian(values),
            frame,
            origins,
            times,
            covariance,
        )
    }

    pub fn spherical(
        values: Vec<[f64; 6]>,
        frame: Frame,
        origins: OriginArray,
        times: Option<TimeArray>,
        covariance: Option<CovarianceBatch>,
    ) -> SchemaResult<Self> {
        Self::new(
            CoordinateValues::Spherical(values),
            frame,
            origins,
            times,
            covariance,
        )
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn representation(&self) -> CoordinateRepresentation {
        self.values.representation()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.len();
        if self.origins.len() != rows {
            return Err(SchemaError::LengthMismatch {
                field: "coordinates.origin".to_string(),
                expected: rows,
                actual: self.origins.len(),
            });
        }
        if let Some(times) = &self.times {
            times.validate()?;
            if times.len() != rows {
                return Err(SchemaError::LengthMismatch {
                    field: "coordinates.time".to_string(),
                    expected: rows,
                    actual: times.len(),
                });
            }
        }
        if let Some(covariance) = &self.covariance {
            covariance.validate()?;
            if covariance.rows != rows {
                return Err(SchemaError::LengthMismatch {
                    field: "coordinates.covariance".to_string(),
                    expected: rows,
                    actual: covariance.rows,
                });
            }
            if covariance.dimension != 6 {
                return Err(SchemaError::InvalidCovarianceShape {
                    rows: covariance.rows,
                    dimension: covariance.dimension,
                    values: covariance.values_row_major.len(),
                });
            }
        }
        Ok(())
    }
}

/// Optional photometric / physical parameters carried alongside an orbit, matching
/// quivr `Orbits.physical_parameters`
/// (`struct<H_v, H_v_sigma, G, G_sigma, sigma_eff, chi2_red: f64>`). Every column is
/// per-row optional. Used by the nested quivr-compatible Arrow round-trip so a full
/// `Orbits` table maps losslessly to/from a Rust `OrbitBatch` (bead personal-cmy.13).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhysicalParametersBatch {
    pub h_v: Vec<Option<f64>>,
    pub h_v_sigma: Vec<Option<f64>>,
    pub g: Vec<Option<f64>>,
    pub g_sigma: Vec<Option<f64>>,
    pub sigma_eff: Vec<Option<f64>>,
    pub chi2_red: Vec<Option<f64>>,
}

impl PhysicalParametersBatch {
    pub fn len(&self) -> usize {
        self.h_v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.h_v.is_empty()
    }

    /// True when every column is entirely null (no physical information present).
    pub fn is_all_null(&self) -> bool {
        [
            &self.h_v,
            &self.h_v_sigma,
            &self.g,
            &self.g_sigma,
            &self.sigma_eff,
            &self.chi2_red,
        ]
        .into_iter()
        .all(|column| column.iter().all(Option::is_none))
    }

    /// Gather rows by index (clones per-row optional values). Used to carry
    /// per-source-orbit physical parameters onto derived rows such as sampled
    /// variants or propagated variant outputs.
    pub fn take(&self, indices: &[usize]) -> Self {
        let gather = |column: &Vec<Option<f64>>| -> Vec<Option<f64>> {
            indices.iter().map(|&index| column[index]).collect()
        };
        Self {
            h_v: gather(&self.h_v),
            h_v_sigma: gather(&self.h_v_sigma),
            g: gather(&self.g),
            g_sigma: gather(&self.g_sigma),
            sigma_eff: gather(&self.sigma_eff),
            chi2_red: gather(&self.chi2_red),
        }
    }

    pub fn validate(&self, rows: usize) -> SchemaResult<()> {
        for (field, len) in [
            ("H_v", self.h_v.len()),
            ("H_v_sigma", self.h_v_sigma.len()),
            ("G", self.g.len()),
            ("G_sigma", self.g_sigma.len()),
            ("sigma_eff", self.sigma_eff.len()),
            ("chi2_red", self.chi2_red.len()),
        ] {
            if len != rows {
                return Err(SchemaError::LengthMismatch {
                    field: format!("physical_parameters.{field}"),
                    expected: rows,
                    actual: len,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrbitBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub coordinates: CoordinateBatch,
    pub physical_parameters: Option<PhysicalParametersBatch>,
}

impl OrbitBatch {
    pub fn new(
        orbit_id: Vec<OrbitId>,
        object_id: Vec<Option<ObjectId>>,
        coordinates: CoordinateBatch,
    ) -> SchemaResult<Self> {
        let out = Self {
            orbit_id,
            object_id,
            coordinates,
            physical_parameters: None,
        };
        out.validate()?;
        Ok(out)
    }

    /// Attach quivr-compatible physical parameters; validated against the orbit row count.
    pub fn with_physical_parameters(
        mut self,
        physical_parameters: PhysicalParametersBatch,
    ) -> SchemaResult<Self> {
        physical_parameters.validate(self.coordinates.len())?;
        self.physical_parameters = Some(physical_parameters);
        Ok(self)
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        validate_orbit_metadata(&self.orbit_id, &self.object_id, &self.coordinates)?;
        if let Some(physical_parameters) = &self.physical_parameters {
            physical_parameters.validate(self.coordinates.len())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrbitVariantBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub variant_id: Vec<Option<VariantId>>,
    pub weights: Vec<Option<f64>>,
    pub weights_cov: Vec<Option<f64>>,
    pub coordinates: CoordinateBatch,
    /// Optional per-variant physical parameters, matching quivr
    /// `VariantOrbits.physical_parameters` (bead personal-cmy.13.2). Carried
    /// through sampling and propagation so Python boundaries no longer
    /// reattach them from source-row indices.
    pub physical_parameters: Option<PhysicalParametersBatch>,
}

impl OrbitVariantBatch {
    pub fn new(
        orbit_id: Vec<OrbitId>,
        object_id: Vec<Option<ObjectId>>,
        variant_id: Vec<Option<VariantId>>,
        weights: Vec<Option<f64>>,
        weights_cov: Vec<Option<f64>>,
        coordinates: CoordinateBatch,
    ) -> SchemaResult<Self> {
        let out = Self {
            orbit_id,
            object_id,
            variant_id,
            weights,
            weights_cov,
            coordinates,
            physical_parameters: None,
        };
        out.validate()?;
        Ok(out)
    }

    /// Attach quivr-compatible physical parameters; validated against the variant row count.
    pub fn with_physical_parameters(
        mut self,
        physical_parameters: PhysicalParametersBatch,
    ) -> SchemaResult<Self> {
        physical_parameters.validate(self.coordinates.len())?;
        self.physical_parameters = Some(physical_parameters);
        Ok(self)
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    pub fn to_orbit_batch(&self) -> SchemaResult<OrbitBatch> {
        let orbits = OrbitBatch::new(
            self.orbit_id.clone(),
            self.object_id.clone(),
            self.coordinates.clone(),
        )?;
        match self.physical_parameters.clone() {
            Some(physical_parameters) => orbits.with_physical_parameters(physical_parameters),
            None => Ok(orbits),
        }
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.coordinates.len();
        validate_orbit_metadata(&self.orbit_id, &self.object_id, &self.coordinates)?;
        validate_len("variant_id", rows, self.variant_id.len())?;
        validate_len("weights", rows, self.weights.len())?;
        validate_len("weights_cov", rows, self.weights_cov.len())?;
        validate_optional_finite("weights", &self.weights)?;
        validate_optional_finite("weights_cov", &self.weights_cov)?;
        if let Some(physical_parameters) = &self.physical_parameters {
            physical_parameters.validate(rows)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserverBatch {
    pub code: Vec<ObservatoryCode>,
    pub coordinates: CoordinateBatch,
}

impl ObserverBatch {
    pub fn new(code: Vec<ObservatoryCode>, coordinates: CoordinateBatch) -> SchemaResult<Self> {
        let out = Self { code, coordinates };
        out.validate()?;
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.coordinates.len();
        if self.coordinates.representation() != CoordinateRepresentation::Cartesian {
            return Err(SchemaError::InvalidRecordBatch(
                "observer coordinates must be Cartesian".to_string(),
            ));
        }
        self.coordinates.validate()?;
        validate_len("observer.code", rows, self.code.len())?;
        for (row, code) in self.code.iter().enumerate() {
            if code.0.trim().is_empty() {
                return Err(SchemaError::InvalidRecordBatch(format!(
                    "observer code must not be empty; row {row}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub coordinates: CoordinateBatch,
    pub predicted_magnitude_v: Option<Vec<f64>>,
    pub alpha_deg: Option<Vec<f64>>,
    pub light_time_days: Vec<f64>,
    pub aberrated_coordinates: Option<CoordinateBatch>,
    pub validity: Validity,
}

impl EphemerisBatch {
    pub fn new(
        orbit_id: Vec<OrbitId>,
        object_id: Vec<Option<ObjectId>>,
        coordinates: CoordinateBatch,
        predicted_magnitude_v: Option<Vec<f64>>,
        alpha_deg: Option<Vec<f64>>,
        light_time_days: Vec<f64>,
        aberrated_coordinates: Option<CoordinateBatch>,
        validity: Validity,
    ) -> SchemaResult<Self> {
        let out = Self {
            orbit_id,
            object_id,
            coordinates,
            predicted_magnitude_v,
            alpha_deg,
            light_time_days,
            aberrated_coordinates,
            validity,
        };
        out.validate()?;
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.coordinates.len();
        if self.coordinates.representation() != CoordinateRepresentation::Spherical {
            return Err(SchemaError::InvalidRecordBatch(
                "ephemeris coordinates must be spherical".to_string(),
            ));
        }
        self.coordinates.validate()?;
        validate_len("ephemeris.orbit_id", rows, self.orbit_id.len())?;
        validate_len("ephemeris.object_id", rows, self.object_id.len())?;
        validate_len(
            "ephemeris.light_time_days",
            rows,
            self.light_time_days.len(),
        )?;
        if self.validity.len() != rows {
            return Err(SchemaError::LengthMismatch {
                field: "ephemeris.validity".to_string(),
                expected: rows,
                actual: self.validity.len(),
            });
        }
        if let Some(values) = &self.predicted_magnitude_v {
            validate_len("ephemeris.predicted_magnitude_v", rows, values.len())?;
        }
        if let Some(values) = &self.alpha_deg {
            validate_len("ephemeris.alpha_deg", rows, values.len())?;
        }
        if let Some(aberrated) = &self.aberrated_coordinates {
            if aberrated.representation() != CoordinateRepresentation::Cartesian {
                return Err(SchemaError::InvalidRecordBatch(
                    "ephemeris aberrated coordinates must be Cartesian".to_string(),
                ));
            }
            aberrated.validate()?;
            if aberrated.len() != rows {
                return Err(SchemaError::LengthMismatch {
                    field: "ephemeris.aberrated_coordinates".to_string(),
                    expected: rows,
                    actual: aberrated.len(),
                });
            }
        }
        Ok(())
    }
}

fn validate_orbit_metadata(
    orbit_id: &[OrbitId],
    object_id: &[Option<ObjectId>],
    coordinates: &CoordinateBatch,
) -> SchemaResult<()> {
    let rows = coordinates.len();
    if coordinates.representation() != CoordinateRepresentation::Cartesian {
        return Err(SchemaError::InvalidRecordBatch(
            "orbit coordinates must be Cartesian".to_string(),
        ));
    }
    coordinates.validate()?;
    validate_len("orbit_id", rows, orbit_id.len())?;
    validate_len("object_id", rows, object_id.len())?;
    Ok(())
}

fn validate_len(field: &str, expected: usize, actual: usize) -> SchemaResult<()> {
    if actual == expected {
        return Ok(());
    }
    Err(SchemaError::LengthMismatch {
        field: field.to_string(),
        expected,
        actual,
    })
}

fn validate_optional_finite(field: &str, values: &[Option<f64>]) -> SchemaResult<()> {
    for (row, value) in values.iter().enumerate() {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "{field} must be finite when present; row {row} is non-finite"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_new_normalizes_nanos() {
        let epoch = Epoch::new(10, NANOS_PER_DAY + 5);
        assert_eq!(epoch.days, 11);
        assert_eq!(epoch.nanos, 5);

        let epoch = Epoch::new(10, -5);
        assert_eq!(epoch.days, 9);
        assert_eq!(epoch.nanos, NANOS_PER_DAY - 5);
    }

    #[test]
    fn time_array_from_parts_requires_equal_lengths() {
        let err = TimeArray::from_parts(TimeScale::Tdb, vec![1, 2], vec![0]).unwrap_err();
        assert!(matches!(err, SchemaError::LengthMismatch { .. }));
    }

    #[test]
    fn coordinate_batch_validates_lengths() {
        let values = vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; 2];
        let origins = OriginArray::repeat(OriginId::SolarSystemBarycenter, 1);
        let err =
            CoordinateBatch::cartesian(values, Frame::Ecliptic, origins, None, None).unwrap_err();
        assert!(matches!(err, SchemaError::LengthMismatch { .. }));
    }

    #[test]
    fn coordinate_batch_requires_six_dimensional_covariance() {
        let values = vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; 1];
        let origins = OriginArray::repeat(OriginId::SolarSystemBarycenter, 1);
        let covariance = CovarianceBatch::new(
            1,
            2,
            vec![0.0; 4],
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let err =
            CoordinateBatch::cartesian(values, Frame::Ecliptic, origins, None, Some(covariance))
                .unwrap_err();
        assert!(matches!(err, SchemaError::InvalidCovarianceShape { .. }));
    }

    #[test]
    fn orbit_batch_requires_cartesian_coordinates() {
        let coords = CoordinateBatch::new(
            CoordinateValues::Spherical(vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; 1]),
            Frame::Equatorial,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let err = OrbitBatch::new(
            vec![OrbitId("orbit-1".to_string())],
            vec![Some(ObjectId("object-1".to_string()))],
            coords,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::InvalidRecordBatch(_)));
    }

    #[test]
    fn orbit_variant_batch_validates_metadata_lengths() {
        let coords = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; 2],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            None,
            None,
        )
        .unwrap();
        let err = OrbitVariantBatch::new(
            vec![OrbitId("orbit-1".to_string())],
            vec![Some(ObjectId("object-1".to_string())), None],
            vec![
                Some(VariantId("0".to_string())),
                Some(VariantId("1".to_string())),
            ],
            vec![Some(0.5), Some(0.5)],
            vec![Some(0.5), Some(0.5)],
            coords,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::LengthMismatch { .. }));
    }

    #[test]
    fn orbit_variant_batch_rejects_nonfinite_weights() {
        let coords = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; 1],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let err = OrbitVariantBatch::new(
            vec![OrbitId("orbit-1".to_string())],
            vec![None],
            vec![Some(VariantId("0".to_string()))],
            vec![Some(f64::NAN)],
            vec![Some(1.0)],
            coords,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::InvalidRecordBatch(_)));
    }

    #[test]
    fn orbit_variant_batch_can_export_orbit_batch_view() {
        let coords = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; 1],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let variants = OrbitVariantBatch::new(
            vec![OrbitId("orbit-1".to_string())],
            vec![Some(ObjectId("object-1".to_string()))],
            vec![Some(VariantId("0".to_string()))],
            vec![None],
            vec![None],
            coords,
        )
        .unwrap();
        let orbits = variants.to_orbit_batch().unwrap();
        assert_eq!(orbits.orbit_id, variants.orbit_id);
        assert_eq!(orbits.object_id, variants.object_id);
        assert_eq!(orbits.coordinates, variants.coordinates);
    }
}
