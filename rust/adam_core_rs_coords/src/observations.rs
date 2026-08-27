//! Rust-canonical observation data model (bead personal-cmy.20).
//!
//! Decision (user, 2026-07-05): mirror the existing quivr observation tables
//! 1:1 -- `ADESObservations`, `PointSourceDetections`, `Exposures`,
//! `Associations`, `Photometry`, and `SourceCatalog` -- with nested
//! quivr-compatible Arrow codecs in both directions, exactly like the orbit
//! data model (bead personal-cmy.13). Timestamp columns are carried as
//! `struct<days: int64, nanos: int64>` children with their scale recorded in
//! schema metadata (`adam_core_time_scale_<column>`).

use crate::types::{SchemaError, SchemaResult, TimeScale};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch, StructArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};
use std::collections::HashMap;
use std::sync::Arc;

use crate::types::{IntoNestedRecordBatch, TryFromNestedRecordBatch};

const SCHEMA_METADATA_KEY: &str = "adam_core_schema";
const SCHEMA_VERSION_KEY: &str = "adam_core_schema_version";
const TIME_SCALE_KEY_PREFIX: &str = "adam_core_time_scale_";

pub const ADES_OBSERVATION_NESTED_SCHEMA: &str = "AdesObservationBatch.nested.quivr.v1";
pub const POINT_SOURCE_DETECTION_NESTED_SCHEMA: &str = "PointSourceDetectionBatch.nested.quivr.v1";
pub const EXPOSURE_NESTED_SCHEMA: &str = "ExposureBatch.nested.quivr.v1";
pub const ASSOCIATION_NESTED_SCHEMA: &str = "AssociationBatch.nested.quivr.v1";
pub const PHOTOMETRY_NESTED_SCHEMA: &str = "PhotometryBatch.nested.quivr.v1";
pub const SOURCE_CATALOG_NESTED_SCHEMA: &str = "SourceCatalogBatch.nested.quivr.v1";

/// A quivr `Timestamp` column: `struct<days: int64, nanos: int64>` plus the
/// time scale (from schema metadata). `validity` is `Some` when the column is
/// nullable and carries nulls (for example `SourceCatalog.exposure_start_time`).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeColumn {
    pub scale: TimeScale,
    pub days: Vec<i64>,
    pub nanos: Vec<i64>,
    pub validity: Option<Vec<bool>>,
}

impl TimeColumn {
    pub fn new(scale: TimeScale, days: Vec<i64>, nanos: Vec<i64>) -> Self {
        Self {
            scale,
            days,
            nanos,
            validity: None,
        }
    }

    pub fn len(&self) -> usize {
        self.days.len()
    }

    pub fn is_empty(&self) -> bool {
        self.days.is_empty()
    }

    pub fn validate(&self, field: &str, rows: usize) -> SchemaResult<()> {
        for (name, len) in [
            ("days", self.days.len()),
            ("nanos", self.nanos.len()),
            (
                "validity",
                self.validity.as_ref().map_or(rows, |values| values.len()),
            ),
        ] {
            if len != rows {
                return Err(SchemaError::LengthMismatch {
                    field: format!("{field}.{name}"),
                    expected: rows,
                    actual: len,
                });
            }
        }
        Ok(())
    }
}

// --- column encode helpers ----------------------------------------------------

fn opt_string_array(values: &[Option<String>]) -> ArrayRef {
    Arc::new(LargeStringArray::from_iter(
        values.iter().map(|value| value.as_deref()),
    )) as ArrayRef
}

fn req_string_array(values: &[String]) -> ArrayRef {
    Arc::new(LargeStringArray::from_iter_values(
        values.iter().map(|value| value.as_str()),
    )) as ArrayRef
}

fn opt_f64_array(values: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec())) as ArrayRef
}

fn req_f64_array(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec())) as ArrayRef
}

fn time_struct_array(column: &TimeColumn) -> SchemaResult<StructArray> {
    let fields = Fields::from(vec![
        Field::new("days", DataType::Int64, true),
        Field::new("nanos", DataType::Int64, true),
    ]);
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(column.days.clone())) as ArrayRef,
        Arc::new(Int64Array::from(column.nanos.clone())) as ArrayRef,
    ];
    let nulls = column
        .validity
        .as_ref()
        .map(|validity| NullBuffer::from_iter(validity.iter().copied()));
    StructArray::try_new(fields, arrays, nulls).map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn time_field(name: &str, column: &TimeColumn) -> SchemaResult<(Field, ArrayRef)> {
    let strct = time_struct_array(column)?;
    Ok((
        Field::new(name, strct.data_type().clone(), true),
        Arc::new(strct) as ArrayRef,
    ))
}

// --- column decode helpers ----------------------------------------------------

fn column<'a>(batch: &'a RecordBatch, name: &str) -> SchemaResult<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))
}

fn decode_opt_string(batch: &RecordBatch, name: &str) -> SchemaResult<Vec<Option<String>>> {
    let array = column(batch, name)?
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| {
            SchemaError::InvalidRecordBatch(format!("column {name} must be LargeUtf8"))
        })?;
    Ok((0..array.len())
        .map(|row| {
            if array.is_null(row) {
                None
            } else {
                Some(array.value(row).to_string())
            }
        })
        .collect())
}

fn decode_req_string(batch: &RecordBatch, name: &str) -> SchemaResult<Vec<String>> {
    decode_opt_string(batch, name)?
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| {
                SchemaError::InvalidRecordBatch(format!("column {name} must not contain nulls"))
            })
        })
        .collect()
}

fn decode_opt_f64(batch: &RecordBatch, name: &str) -> SchemaResult<Vec<Option<f64>>> {
    let array = column(batch, name)?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("column {name} must be Float64")))?;
    Ok((0..array.len())
        .map(|row| {
            if array.is_null(row) {
                None
            } else {
                Some(array.value(row))
            }
        })
        .collect())
}

fn decode_req_f64(batch: &RecordBatch, name: &str) -> SchemaResult<Vec<f64>> {
    decode_opt_f64(batch, name)?
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| {
                SchemaError::InvalidRecordBatch(format!("column {name} must not contain nulls"))
            })
        })
        .collect()
}

fn decode_time(batch: &RecordBatch, name: &str) -> SchemaResult<TimeColumn> {
    let scale = metadata_time_scale(batch, name)?;
    let strct = column(batch, name)?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            SchemaError::InvalidRecordBatch(format!("column {name} must be struct<days, nanos>"))
        })?;
    let child_i64 = |child: &str| -> SchemaResult<Vec<i64>> {
        let array = strct
            .column_by_name(child)
            .ok_or_else(|| SchemaError::MissingRequiredField(format!("{name}.{child}")))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                SchemaError::InvalidRecordBatch(format!("{name}.{child} must be Int64"))
            })?
            .clone();
        Ok((0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    0
                } else {
                    array.value(row)
                }
            })
            .collect())
    };
    let days = child_i64("days")?;
    let nanos = child_i64("nanos")?;
    let validity = if strct.null_count() > 0 {
        Some((0..strct.len()).map(|row| !strct.is_null(row)).collect())
    } else {
        None
    };
    Ok(TimeColumn {
        scale,
        days,
        nanos,
        validity,
    })
}

// --- schema metadata ------------------------------------------------------------

fn observation_metadata(
    schema_name: &str,
    time_scales: &[(&str, TimeScale)],
) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert(SCHEMA_METADATA_KEY.to_string(), schema_name.to_string());
    metadata.insert(SCHEMA_VERSION_KEY.to_string(), "1".to_string());
    for (field, scale) in time_scales {
        metadata.insert(
            format!("{TIME_SCALE_KEY_PREFIX}{field}"),
            scale.as_str().to_string(),
        );
    }
    metadata
}

fn metadata_time_scale(batch: &RecordBatch, field: &str) -> SchemaResult<TimeScale> {
    let schema = batch.schema();
    let key = format!("{TIME_SCALE_KEY_PREFIX}{field}");
    let value = schema
        .metadata()
        .get(&key)
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("missing {key} schema metadata")))?;
    TimeScale::parse(value)
}

fn build_batch(
    schema_name: &str,
    time_scales: &[(&str, TimeScale)],
    columns: Vec<(Field, ArrayRef)>,
    rows: usize,
) -> SchemaResult<RecordBatch> {
    let fields = columns
        .iter()
        .map(|(field, _)| field.clone())
        .collect::<Vec<_>>();
    let arrays = columns
        .into_iter()
        .map(|(_, array)| array)
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        observation_metadata(schema_name, time_scales),
    ));
    RecordBatch::try_new_with_options(
        schema,
        arrays,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn validate_len(field: &str, rows: usize, actual: usize) -> SchemaResult<()> {
    if actual != rows {
        return Err(SchemaError::LengthMismatch {
            field: field.to_string(),
            expected: rows,
            actual,
        });
    }
    Ok(())
}

// --- ADESObservations -----------------------------------------------------------

/// Mirror of quivr `adam_core.observations.ades.ADESObservations`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdesObservationBatch {
    pub perm_id: Vec<Option<String>>,
    pub prov_id: Vec<Option<String>>,
    pub trk_sub: Vec<Option<String>>,
    pub obs_sub_id: Vec<Option<String>>,
    pub obs_time: TimeColumn,
    pub rms_time: Vec<Option<f64>>,
    pub ra: Vec<f64>,
    pub dec: Vec<f64>,
    pub rms_ra_cos_dec: Vec<Option<f64>>,
    pub rms_dec: Vec<Option<f64>>,
    pub rms_corr: Vec<Option<f64>>,
    pub mag: Vec<Option<f64>>,
    pub rms_mag: Vec<Option<f64>>,
    pub band: Vec<Option<String>>,
    pub stn: Vec<String>,
    pub mode: Vec<String>,
    pub ast_cat: Vec<String>,
    pub phot_cat: Vec<Option<String>>,
    pub log_snr: Vec<Option<f64>>,
    pub seeing: Vec<Option<f64>>,
    pub exp: Vec<Option<f64>>,
    pub remarks: Vec<Option<String>>,
}

impl AdesObservationBatch {
    pub fn len(&self) -> usize {
        self.ra.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ra.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.len();
        for (field, len) in [
            ("permID", self.perm_id.len()),
            ("provID", self.prov_id.len()),
            ("trkSub", self.trk_sub.len()),
            ("obsSubID", self.obs_sub_id.len()),
            ("rmsTime", self.rms_time.len()),
            ("dec", self.dec.len()),
            ("rmsRACosDec", self.rms_ra_cos_dec.len()),
            ("rmsDec", self.rms_dec.len()),
            ("rmsCorr", self.rms_corr.len()),
            ("mag", self.mag.len()),
            ("rmsMag", self.rms_mag.len()),
            ("band", self.band.len()),
            ("stn", self.stn.len()),
            ("mode", self.mode.len()),
            ("astCat", self.ast_cat.len()),
            ("photCat", self.phot_cat.len()),
            ("logSNR", self.log_snr.len()),
            ("seeing", self.seeing.len()),
            ("exp", self.exp.len()),
            ("remarks", self.remarks.len()),
        ] {
            validate_len(field, rows, len)?;
        }
        self.obs_time.validate("obsTime", rows)
    }
}

impl IntoNestedRecordBatch for AdesObservationBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        self.validate()?;
        let rows = self.len();
        let string = |name: &str, values: &[Option<String>]| {
            (
                Field::new(name, DataType::LargeUtf8, true),
                opt_string_array(values),
            )
        };
        let req_string = |name: &str, values: &[String]| {
            (
                Field::new(name, DataType::LargeUtf8, true),
                req_string_array(values),
            )
        };
        let float = |name: &str, values: &[Option<f64>]| {
            (
                Field::new(name, DataType::Float64, true),
                opt_f64_array(values),
            )
        };
        let req_float = |name: &str, values: &[f64]| {
            (
                Field::new(name, DataType::Float64, true),
                req_f64_array(values),
            )
        };
        let columns = vec![
            string("permID", &self.perm_id),
            string("provID", &self.prov_id),
            string("trkSub", &self.trk_sub),
            string("obsSubID", &self.obs_sub_id),
            time_field("obsTime", &self.obs_time)?,
            float("rmsTime", &self.rms_time),
            req_float("ra", &self.ra),
            req_float("dec", &self.dec),
            float("rmsRACosDec", &self.rms_ra_cos_dec),
            float("rmsDec", &self.rms_dec),
            float("rmsCorr", &self.rms_corr),
            float("mag", &self.mag),
            float("rmsMag", &self.rms_mag),
            string("band", &self.band),
            req_string("stn", &self.stn),
            req_string("mode", &self.mode),
            req_string("astCat", &self.ast_cat),
            string("photCat", &self.phot_cat),
            float("logSNR", &self.log_snr),
            float("seeing", &self.seeing),
            float("exp", &self.exp),
            string("remarks", &self.remarks),
        ];
        build_batch(
            ADES_OBSERVATION_NESTED_SCHEMA,
            &[("obsTime", self.obs_time.scale)],
            columns,
            rows,
        )
    }
}

impl TryFromNestedRecordBatch for AdesObservationBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        let out = Self {
            perm_id: decode_opt_string(batch, "permID")?,
            prov_id: decode_opt_string(batch, "provID")?,
            trk_sub: decode_opt_string(batch, "trkSub")?,
            obs_sub_id: decode_opt_string(batch, "obsSubID")?,
            obs_time: decode_time(batch, "obsTime")?,
            rms_time: decode_opt_f64(batch, "rmsTime")?,
            ra: decode_req_f64(batch, "ra")?,
            dec: decode_req_f64(batch, "dec")?,
            rms_ra_cos_dec: decode_opt_f64(batch, "rmsRACosDec")?,
            rms_dec: decode_opt_f64(batch, "rmsDec")?,
            rms_corr: decode_opt_f64(batch, "rmsCorr")?,
            mag: decode_opt_f64(batch, "mag")?,
            rms_mag: decode_opt_f64(batch, "rmsMag")?,
            band: decode_opt_string(batch, "band")?,
            stn: decode_req_string(batch, "stn")?,
            mode: decode_req_string(batch, "mode")?,
            ast_cat: decode_req_string(batch, "astCat")?,
            phot_cat: decode_opt_string(batch, "photCat")?,
            log_snr: decode_opt_f64(batch, "logSNR")?,
            seeing: decode_opt_f64(batch, "seeing")?,
            exp: decode_opt_f64(batch, "exp")?,
            remarks: decode_opt_string(batch, "remarks")?,
        };
        out.validate()?;
        Ok(out)
    }
}

// --- PointSourceDetections -------------------------------------------------------

/// Mirror of quivr `adam_core.observations.detections.PointSourceDetections`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointSourceDetectionBatch {
    pub id: Vec<String>,
    pub exposure_id: Vec<Option<String>>,
    pub time: TimeColumn,
    pub ra: Vec<f64>,
    pub ra_sigma: Vec<Option<f64>>,
    pub dec: Vec<f64>,
    pub dec_sigma: Vec<Option<f64>>,
    pub mag: Vec<Option<f64>>,
    pub mag_sigma: Vec<Option<f64>>,
}

impl PointSourceDetectionBatch {
    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.len();
        for (field, len) in [
            ("exposure_id", self.exposure_id.len()),
            ("ra", self.ra.len()),
            ("ra_sigma", self.ra_sigma.len()),
            ("dec", self.dec.len()),
            ("dec_sigma", self.dec_sigma.len()),
            ("mag", self.mag.len()),
            ("mag_sigma", self.mag_sigma.len()),
        ] {
            validate_len(field, rows, len)?;
        }
        self.time.validate("time", rows)
    }
}

impl IntoNestedRecordBatch for PointSourceDetectionBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        self.validate()?;
        let rows = self.len();
        let columns = vec![
            (
                Field::new("id", DataType::LargeUtf8, true),
                req_string_array(&self.id),
            ),
            (
                Field::new("exposure_id", DataType::LargeUtf8, true),
                opt_string_array(&self.exposure_id),
            ),
            time_field("time", &self.time)?,
            (
                Field::new("ra", DataType::Float64, true),
                req_f64_array(&self.ra),
            ),
            (
                Field::new("ra_sigma", DataType::Float64, true),
                opt_f64_array(&self.ra_sigma),
            ),
            (
                Field::new("dec", DataType::Float64, true),
                req_f64_array(&self.dec),
            ),
            (
                Field::new("dec_sigma", DataType::Float64, true),
                opt_f64_array(&self.dec_sigma),
            ),
            (
                Field::new("mag", DataType::Float64, true),
                opt_f64_array(&self.mag),
            ),
            (
                Field::new("mag_sigma", DataType::Float64, true),
                opt_f64_array(&self.mag_sigma),
            ),
        ];
        build_batch(
            POINT_SOURCE_DETECTION_NESTED_SCHEMA,
            &[("time", self.time.scale)],
            columns,
            rows,
        )
    }
}

impl TryFromNestedRecordBatch for PointSourceDetectionBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        let out = Self {
            id: decode_req_string(batch, "id")?,
            exposure_id: decode_opt_string(batch, "exposure_id")?,
            time: decode_time(batch, "time")?,
            ra: decode_req_f64(batch, "ra")?,
            ra_sigma: decode_opt_f64(batch, "ra_sigma")?,
            dec: decode_req_f64(batch, "dec")?,
            dec_sigma: decode_opt_f64(batch, "dec_sigma")?,
            mag: decode_opt_f64(batch, "mag")?,
            mag_sigma: decode_opt_f64(batch, "mag_sigma")?,
        };
        out.validate()?;
        Ok(out)
    }
}

// --- Exposures --------------------------------------------------------------------

/// Mirror of quivr `adam_core.observations.exposures.Exposures`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExposureBatch {
    pub id: Vec<String>,
    pub start_time: TimeColumn,
    pub duration: Vec<f64>,
    pub filter: Vec<String>,
    pub observatory_code: Vec<String>,
    pub seeing: Vec<Option<f64>>,
    pub depth_5sigma: Vec<Option<f64>>,
}

impl ExposureBatch {
    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.len();
        for (field, len) in [
            ("duration", self.duration.len()),
            ("filter", self.filter.len()),
            ("observatory_code", self.observatory_code.len()),
            ("seeing", self.seeing.len()),
            ("depth_5sigma", self.depth_5sigma.len()),
        ] {
            validate_len(field, rows, len)?;
        }
        self.start_time.validate("start_time", rows)
    }
}

impl IntoNestedRecordBatch for ExposureBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        self.validate()?;
        let rows = self.len();
        let columns = vec![
            (
                Field::new("id", DataType::LargeUtf8, true),
                req_string_array(&self.id),
            ),
            time_field("start_time", &self.start_time)?,
            (
                Field::new("duration", DataType::Float64, true),
                req_f64_array(&self.duration),
            ),
            (
                Field::new("filter", DataType::LargeUtf8, true),
                req_string_array(&self.filter),
            ),
            (
                Field::new("observatory_code", DataType::LargeUtf8, true),
                req_string_array(&self.observatory_code),
            ),
            (
                Field::new("seeing", DataType::Float64, true),
                opt_f64_array(&self.seeing),
            ),
            (
                Field::new("depth_5sigma", DataType::Float64, true),
                opt_f64_array(&self.depth_5sigma),
            ),
        ];
        build_batch(
            EXPOSURE_NESTED_SCHEMA,
            &[("start_time", self.start_time.scale)],
            columns,
            rows,
        )
    }
}

impl TryFromNestedRecordBatch for ExposureBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        let out = Self {
            id: decode_req_string(batch, "id")?,
            start_time: decode_time(batch, "start_time")?,
            duration: decode_req_f64(batch, "duration")?,
            filter: decode_req_string(batch, "filter")?,
            observatory_code: decode_req_string(batch, "observatory_code")?,
            seeing: decode_opt_f64(batch, "seeing")?,
            depth_5sigma: decode_opt_f64(batch, "depth_5sigma")?,
        };
        out.validate()?;
        Ok(out)
    }
}

// --- Associations -------------------------------------------------------------------

/// Mirror of quivr `adam_core.observations.associations.Associations`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssociationBatch {
    pub detection_id: Vec<String>,
    pub object_id: Vec<Option<String>>,
}

impl AssociationBatch {
    pub fn len(&self) -> usize {
        self.detection_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.detection_id.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        validate_len("object_id", self.len(), self.object_id.len())
    }
}

impl IntoNestedRecordBatch for AssociationBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        self.validate()?;
        let rows = self.len();
        let columns = vec![
            (
                Field::new("detection_id", DataType::LargeUtf8, true),
                req_string_array(&self.detection_id),
            ),
            (
                Field::new("object_id", DataType::LargeUtf8, true),
                opt_string_array(&self.object_id),
            ),
        ];
        build_batch(ASSOCIATION_NESTED_SCHEMA, &[], columns, rows)
    }
}

impl TryFromNestedRecordBatch for AssociationBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        let out = Self {
            detection_id: decode_req_string(batch, "detection_id")?,
            object_id: decode_opt_string(batch, "object_id")?,
        };
        out.validate()?;
        Ok(out)
    }
}

// --- Photometry ----------------------------------------------------------------------

/// Mirror of quivr `adam_core.observations.photometry.Photometry`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotometryBatch {
    pub time: TimeColumn,
    pub mag: Vec<Option<f64>>,
    pub mag_sigma: Vec<Option<f64>>,
    pub filter: Vec<Option<String>>,
}

impl PhotometryBatch {
    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.len();
        for (field, len) in [
            ("mag", self.mag.len()),
            ("mag_sigma", self.mag_sigma.len()),
            ("filter", self.filter.len()),
        ] {
            validate_len(field, rows, len)?;
        }
        self.time.validate("time", rows)
    }
}

impl IntoNestedRecordBatch for PhotometryBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        self.validate()?;
        let rows = self.len();
        let columns = vec![
            time_field("time", &self.time)?,
            (
                Field::new("mag", DataType::Float64, true),
                opt_f64_array(&self.mag),
            ),
            (
                Field::new("mag_sigma", DataType::Float64, true),
                opt_f64_array(&self.mag_sigma),
            ),
            (
                Field::new("filter", DataType::LargeUtf8, true),
                opt_string_array(&self.filter),
            ),
        ];
        build_batch(
            PHOTOMETRY_NESTED_SCHEMA,
            &[("time", self.time.scale)],
            columns,
            rows,
        )
    }
}

impl TryFromNestedRecordBatch for PhotometryBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        let out = Self {
            time: decode_time(batch, "time")?,
            mag: decode_opt_f64(batch, "mag")?,
            mag_sigma: decode_opt_f64(batch, "mag_sigma")?,
            filter: decode_opt_string(batch, "filter")?,
        };
        out.validate()?;
        Ok(out)
    }
}

// --- SourceCatalog ---------------------------------------------------------------------

/// Mirror of quivr `adam_core.observations.source_catalog.SourceCatalog`.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCatalogBatch {
    pub id: Vec<String>,
    pub exposure_id: Vec<Option<String>>,
    pub time: TimeColumn,
    pub ra: Vec<f64>,
    pub dec: Vec<f64>,
    pub ra_sigma: Vec<Option<f64>>,
    pub dec_sigma: Vec<Option<f64>>,
    pub radec_corr: Vec<Option<f64>>,
    pub mag: Vec<Option<f64>>,
    pub mag_sigma: Vec<Option<f64>>,
    pub fwhm: Vec<Option<f64>>,
    pub a: Vec<Option<f64>>,
    pub a_sigma: Vec<Option<f64>>,
    pub b: Vec<Option<f64>>,
    pub b_sigma: Vec<Option<f64>>,
    pub pa: Vec<Option<f64>>,
    pub pa_sigma: Vec<Option<f64>>,
    pub observatory_code: Vec<String>,
    pub filter: Vec<Option<String>>,
    pub exposure_start_time: TimeColumn,
    pub exposure_duration: Vec<Option<f64>>,
    pub exposure_seeing: Vec<Option<f64>>,
    pub exposure_depth_5sigma: Vec<Option<f64>>,
    pub object_id: Vec<Option<String>>,
    pub catalog_id: Vec<String>,
}

impl SourceCatalogBatch {
    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    pub fn validate(&self) -> SchemaResult<()> {
        let rows = self.len();
        for (field, len) in [
            ("exposure_id", self.exposure_id.len()),
            ("ra", self.ra.len()),
            ("dec", self.dec.len()),
            ("ra_sigma", self.ra_sigma.len()),
            ("dec_sigma", self.dec_sigma.len()),
            ("radec_corr", self.radec_corr.len()),
            ("mag", self.mag.len()),
            ("mag_sigma", self.mag_sigma.len()),
            ("fwhm", self.fwhm.len()),
            ("a", self.a.len()),
            ("a_sigma", self.a_sigma.len()),
            ("b", self.b.len()),
            ("b_sigma", self.b_sigma.len()),
            ("pa", self.pa.len()),
            ("pa_sigma", self.pa_sigma.len()),
            ("observatory_code", self.observatory_code.len()),
            ("filter", self.filter.len()),
            ("exposure_duration", self.exposure_duration.len()),
            ("exposure_seeing", self.exposure_seeing.len()),
            ("exposure_depth_5sigma", self.exposure_depth_5sigma.len()),
            ("object_id", self.object_id.len()),
            ("catalog_id", self.catalog_id.len()),
        ] {
            validate_len(field, rows, len)?;
        }
        self.time.validate("time", rows)?;
        self.exposure_start_time
            .validate("exposure_start_time", rows)
    }
}

impl IntoNestedRecordBatch for SourceCatalogBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        self.validate()?;
        let rows = self.len();
        let string = |name: &str, values: &[Option<String>]| {
            (
                Field::new(name, DataType::LargeUtf8, true),
                opt_string_array(values),
            )
        };
        let req_string = |name: &str, values: &[String]| {
            (
                Field::new(name, DataType::LargeUtf8, true),
                req_string_array(values),
            )
        };
        let float = |name: &str, values: &[Option<f64>]| {
            (
                Field::new(name, DataType::Float64, true),
                opt_f64_array(values),
            )
        };
        let req_float = |name: &str, values: &[f64]| {
            (
                Field::new(name, DataType::Float64, true),
                req_f64_array(values),
            )
        };
        let columns = vec![
            req_string("id", &self.id),
            string("exposure_id", &self.exposure_id),
            time_field("time", &self.time)?,
            req_float("ra", &self.ra),
            req_float("dec", &self.dec),
            float("ra_sigma", &self.ra_sigma),
            float("dec_sigma", &self.dec_sigma),
            float("radec_corr", &self.radec_corr),
            float("mag", &self.mag),
            float("mag_sigma", &self.mag_sigma),
            float("fwhm", &self.fwhm),
            float("a", &self.a),
            float("a_sigma", &self.a_sigma),
            float("b", &self.b),
            float("b_sigma", &self.b_sigma),
            float("pa", &self.pa),
            float("pa_sigma", &self.pa_sigma),
            req_string("observatory_code", &self.observatory_code),
            string("filter", &self.filter),
            time_field("exposure_start_time", &self.exposure_start_time)?,
            float("exposure_duration", &self.exposure_duration),
            float("exposure_seeing", &self.exposure_seeing),
            float("exposure_depth_5sigma", &self.exposure_depth_5sigma),
            string("object_id", &self.object_id),
            req_string("catalog_id", &self.catalog_id),
        ];
        build_batch(
            SOURCE_CATALOG_NESTED_SCHEMA,
            &[
                ("time", self.time.scale),
                ("exposure_start_time", self.exposure_start_time.scale),
            ],
            columns,
            rows,
        )
    }
}

impl TryFromNestedRecordBatch for SourceCatalogBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        let out = Self {
            id: decode_req_string(batch, "id")?,
            exposure_id: decode_opt_string(batch, "exposure_id")?,
            time: decode_time(batch, "time")?,
            ra: decode_req_f64(batch, "ra")?,
            dec: decode_req_f64(batch, "dec")?,
            ra_sigma: decode_opt_f64(batch, "ra_sigma")?,
            dec_sigma: decode_opt_f64(batch, "dec_sigma")?,
            radec_corr: decode_opt_f64(batch, "radec_corr")?,
            mag: decode_opt_f64(batch, "mag")?,
            mag_sigma: decode_opt_f64(batch, "mag_sigma")?,
            fwhm: decode_opt_f64(batch, "fwhm")?,
            a: decode_opt_f64(batch, "a")?,
            a_sigma: decode_opt_f64(batch, "a_sigma")?,
            b: decode_opt_f64(batch, "b")?,
            b_sigma: decode_opt_f64(batch, "b_sigma")?,
            pa: decode_opt_f64(batch, "pa")?,
            pa_sigma: decode_opt_f64(batch, "pa_sigma")?,
            observatory_code: decode_req_string(batch, "observatory_code")?,
            filter: decode_opt_string(batch, "filter")?,
            exposure_start_time: decode_time(batch, "exposure_start_time")?,
            exposure_duration: decode_opt_f64(batch, "exposure_duration")?,
            exposure_seeing: decode_opt_f64(batch, "exposure_seeing")?,
            exposure_depth_5sigma: decode_opt_f64(batch, "exposure_depth_5sigma")?,
            object_id: decode_opt_string(batch, "object_id")?,
            catalog_id: decode_req_string(batch, "catalog_id")?,
        };
        out.validate()?;
        Ok(out)
    }
}

// --- dispatcher ---------------------------------------------------------------------

/// Round-trip any supported observation table (nested quivr layout) through its
/// Rust-canonical batch, dispatching on the `adam_core_schema` metadata key.
pub fn round_trip_nested(batch: &RecordBatch) -> SchemaResult<RecordBatch> {
    let schema = batch.schema();
    let name = schema.metadata().get(SCHEMA_METADATA_KEY).ok_or_else(|| {
        SchemaError::InvalidRecordBatch("missing adam_core_schema metadata".to_string())
    })?;
    match name.as_str() {
        ADES_OBSERVATION_NESTED_SCHEMA => {
            AdesObservationBatch::try_from_nested_record_batch(batch)?.into_nested_record_batch()
        }
        POINT_SOURCE_DETECTION_NESTED_SCHEMA => {
            PointSourceDetectionBatch::try_from_nested_record_batch(batch)?
                .into_nested_record_batch()
        }
        EXPOSURE_NESTED_SCHEMA => {
            ExposureBatch::try_from_nested_record_batch(batch)?.into_nested_record_batch()
        }
        ASSOCIATION_NESTED_SCHEMA => {
            AssociationBatch::try_from_nested_record_batch(batch)?.into_nested_record_batch()
        }
        PHOTOMETRY_NESTED_SCHEMA => {
            PhotometryBatch::try_from_nested_record_batch(batch)?.into_nested_record_batch()
        }
        SOURCE_CATALOG_NESTED_SCHEMA => {
            SourceCatalogBatch::try_from_nested_record_batch(batch)?.into_nested_record_batch()
        }
        other => Err(SchemaError::InvalidRecordBatch(format!(
            "unsupported observation schema: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc_time(days: Vec<i64>, nanos: Vec<i64>) -> TimeColumn {
        TimeColumn::new(TimeScale::Utc, days, nanos)
    }

    fn sample_ades() -> AdesObservationBatch {
        AdesObservationBatch {
            perm_id: vec![Some("12345".to_string()), None],
            prov_id: vec![None, Some("2024 AB".to_string())],
            trk_sub: vec![None, None],
            obs_sub_id: vec![Some("sub-1".to_string()), None],
            obs_time: utc_time(vec![60000, 60001], vec![0, 43_200_000_000_000]),
            rms_time: vec![Some(1.0), None],
            ra: vec![10.5, 200.25],
            dec: vec![-5.0, 45.5],
            rms_ra_cos_dec: vec![Some(0.1), None],
            rms_dec: vec![Some(0.1), Some(0.2)],
            rms_corr: vec![None, Some(-0.5)],
            mag: vec![Some(21.2), None],
            rms_mag: vec![Some(0.1), None],
            band: vec![Some("r".to_string()), None],
            stn: vec!["X05".to_string(), "W84".to_string()],
            mode: vec!["CCD".to_string(), "CCD".to_string()],
            ast_cat: vec!["Gaia2".to_string(), "Gaia2".to_string()],
            phot_cat: vec![Some("Gaia2".to_string()), None],
            log_snr: vec![Some(1.5), None],
            seeing: vec![None, Some(0.8)],
            exp: vec![Some(30.0), Some(30.0)],
            remarks: vec![None, Some("  padded remark ".to_string())],
        }
    }

    #[test]
    fn ades_nested_round_trip_is_lossless() {
        let ades = sample_ades();
        let batch = ades.clone().into_nested_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 22);
        assert_eq!(
            batch.schema().metadata().get(SCHEMA_METADATA_KEY).unwrap(),
            ADES_OBSERVATION_NESTED_SCHEMA
        );
        let round_tripped = AdesObservationBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, ades);
    }

    #[test]
    fn ades_empty_round_trip() {
        let empty = AdesObservationBatch {
            perm_id: vec![],
            prov_id: vec![],
            trk_sub: vec![],
            obs_sub_id: vec![],
            obs_time: utc_time(vec![], vec![]),
            rms_time: vec![],
            ra: vec![],
            dec: vec![],
            rms_ra_cos_dec: vec![],
            rms_dec: vec![],
            rms_corr: vec![],
            mag: vec![],
            rms_mag: vec![],
            band: vec![],
            stn: vec![],
            mode: vec![],
            ast_cat: vec![],
            phot_cat: vec![],
            log_snr: vec![],
            seeing: vec![],
            exp: vec![],
            remarks: vec![],
        };
        let batch = empty.clone().into_nested_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 0);
        let round_tripped = AdesObservationBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, empty);
    }

    #[test]
    fn detection_round_trip_is_lossless() {
        let detections = PointSourceDetectionBatch {
            id: vec!["det-1".to_string(), "det-2".to_string()],
            exposure_id: vec![Some("exp-1".to_string()), None],
            time: TimeColumn::new(TimeScale::Tai, vec![59000, 59001], vec![1, 2]),
            ra: vec![0.0, 359.9],
            ra_sigma: vec![None, Some(0.2)],
            dec: vec![-89.9, 89.9],
            dec_sigma: vec![Some(0.1), None],
            mag: vec![Some(20.0), None],
            mag_sigma: vec![None, Some(0.05)],
        };
        let batch = detections.clone().into_nested_record_batch().unwrap();
        let round_tripped =
            PointSourceDetectionBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, detections);
    }

    #[test]
    fn exposure_round_trip_is_lossless() {
        let exposures = ExposureBatch {
            id: vec!["exp-1".to_string()],
            start_time: utc_time(vec![60200], vec![500]),
            duration: vec![30.0],
            filter: vec!["g".to_string()],
            observatory_code: vec!["X05".to_string()],
            seeing: vec![None],
            depth_5sigma: vec![Some(24.5)],
        };
        let batch = exposures.clone().into_nested_record_batch().unwrap();
        let round_tripped = ExposureBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, exposures);
    }

    #[test]
    fn association_round_trip_is_lossless() {
        let associations = AssociationBatch {
            detection_id: vec!["det-1".to_string(), "det-2".to_string()],
            object_id: vec![Some("object-1".to_string()), None],
        };
        let batch = associations.clone().into_nested_record_batch().unwrap();
        let round_tripped = AssociationBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, associations);
    }

    #[test]
    fn photometry_round_trip_is_lossless() {
        let photometry = PhotometryBatch {
            time: TimeColumn::new(TimeScale::Tdb, vec![60123], vec![987_654_321]),
            mag: vec![Some(19.5)],
            mag_sigma: vec![None],
            filter: vec![Some("i".to_string())],
        };
        let batch = photometry.clone().into_nested_record_batch().unwrap();
        let round_tripped = PhotometryBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, photometry);
    }

    #[test]
    fn source_catalog_round_trip_preserves_nullable_time() {
        let catalog = SourceCatalogBatch {
            id: vec!["src-1".to_string(), "src-2".to_string()],
            exposure_id: vec![Some("exp-1".to_string()), None],
            time: utc_time(vec![60300, 60301], vec![0, 1]),
            ra: vec![15.0, 30.0],
            dec: vec![-15.0, 30.0],
            ra_sigma: vec![Some(0.1), None],
            dec_sigma: vec![Some(0.1), None],
            radec_corr: vec![Some(0.5), None],
            mag: vec![Some(20.0), None],
            mag_sigma: vec![Some(0.1), None],
            fwhm: vec![Some(1.0), None],
            a: vec![Some(1.2), None],
            a_sigma: vec![None, None],
            b: vec![Some(0.8), None],
            b_sigma: vec![None, None],
            pa: vec![Some(45.0), None],
            pa_sigma: vec![None, None],
            observatory_code: vec!["X05".to_string(), "W84".to_string()],
            filter: vec![Some("r".to_string()), None],
            exposure_start_time: TimeColumn {
                scale: TimeScale::Utc,
                days: vec![60300, 0],
                nanos: vec![0, 0],
                validity: Some(vec![true, false]),
            },
            exposure_duration: vec![Some(30.0), None],
            exposure_seeing: vec![None, None],
            exposure_depth_5sigma: vec![Some(24.0), None],
            object_id: vec![None, Some("object-9".to_string())],
            catalog_id: vec!["cat-1".to_string(), "cat-1".to_string()],
        };
        let batch = catalog.clone().into_nested_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 25);
        let round_tripped = SourceCatalogBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped, catalog);
    }

    #[test]
    fn dispatcher_round_trips_by_schema_metadata() {
        let batch = sample_ades().into_nested_record_batch().unwrap();
        let round_tripped = round_trip_nested(&batch).unwrap();
        let decoded = AdesObservationBatch::try_from_nested_record_batch(&round_tripped).unwrap();
        assert_eq!(decoded, sample_ades());
    }
}
