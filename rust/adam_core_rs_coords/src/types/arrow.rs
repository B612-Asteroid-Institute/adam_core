//! Arrow adapters for the standalone Rust data-model prototypes.
//!
//! These adapters intentionally use a simple flat Cartesian schema as the
//! first RM-STANDALONE-003 bridge. Python/quivr compatibility wrappers can map
//! these canonical columns into nested quivr tables without making quivr own
//! the Rust schema.

use super::{
    CoordinateBatch, CoordinateRepresentation, CovarianceBatch, CovarianceUnits, Epoch, Frame,
    ObjectId, OrbitBatch, OrbitId, OrbitVariantBatch, OriginArray, OriginId, SchemaError,
    SchemaResult, TimeArray, TimeScale, Validity, VariantId,
};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

const SCHEMA_METADATA_KEY: &str = "adam_core_schema";
const SCHEMA_VERSION_KEY: &str = "adam_core_schema_version";
const REPRESENTATION_METADATA_KEY: &str = "adam_core_representation";
const FRAME_METADATA_KEY: &str = "adam_core_frame";
const TIME_SCALE_METADATA_KEY: &str = "adam_core_time_scale";
const COVARIANCE_METADATA_KEY: &str = "adam_core_covariance";
const CARTESIAN_COORDINATE_SCHEMA: &str = "CoordinateBatch.cartesian.flat.v1";
const ORBIT_SCHEMA: &str = "OrbitBatch.cartesian.flat.v1";
const ORBIT_VARIANT_SCHEMA: &str = "OrbitVariantBatch.cartesian.flat.v1";

pub trait ArrowSchemaExport {
    fn schema() -> Schema;
}

pub trait IntoRecordBatch {
    fn into_record_batch(self) -> SchemaResult<RecordBatch>;
}

pub trait TryFromRecordBatch: Sized {
    fn try_from_record_batch(batch: &RecordBatch) -> SchemaResult<Self>;
}

impl ArrowSchemaExport for CoordinateBatch {
    fn schema() -> Schema {
        coordinate_schema(Frame::Unspecified, None, false)
    }
}

impl IntoRecordBatch for CoordinateBatch {
    fn into_record_batch(self) -> SchemaResult<RecordBatch> {
        coordinate_to_record_batch(&self)
    }
}

impl TryFromRecordBatch for CoordinateBatch {
    fn try_from_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        coordinate_from_record_batch(batch)
    }
}

impl ArrowSchemaExport for OrbitBatch {
    fn schema() -> Schema {
        orbit_schema(Frame::Unspecified, None, false)
    }
}

impl IntoRecordBatch for OrbitBatch {
    fn into_record_batch(self) -> SchemaResult<RecordBatch> {
        orbit_to_record_batch(&self)
    }
}

impl TryFromRecordBatch for OrbitBatch {
    fn try_from_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        orbit_from_record_batch(batch)
    }
}

impl ArrowSchemaExport for OrbitVariantBatch {
    fn schema() -> Schema {
        orbit_variant_schema(Frame::Unspecified, None, false)
    }
}

impl IntoRecordBatch for OrbitVariantBatch {
    fn into_record_batch(self) -> SchemaResult<RecordBatch> {
        orbit_variant_to_record_batch(&self)
    }
}

impl TryFromRecordBatch for OrbitVariantBatch {
    fn try_from_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        orbit_variant_from_record_batch(batch)
    }
}

fn coordinate_fields() -> Vec<Field> {
    let mut fields = vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("vx", DataType::Float64, false),
        Field::new("vy", DataType::Float64, false),
        Field::new("vz", DataType::Float64, false),
        Field::new("time_days", DataType::Int64, true),
        Field::new("time_nanos", DataType::Int64, true),
        Field::new("origin_code", DataType::LargeUtf8, false),
    ];
    for row in 0..6 {
        for col in 0..6 {
            fields.push(Field::new(
                format!("covariance_{row}{col}"),
                DataType::Float64,
                true,
            ));
        }
    }
    fields
}

fn coordinate_metadata(
    schema_name: &str,
    frame: Frame,
    time_scale: Option<TimeScale>,
    has_covariance: bool,
) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert(SCHEMA_METADATA_KEY.to_string(), schema_name.to_string());
    metadata.insert(SCHEMA_VERSION_KEY.to_string(), "1".to_string());
    metadata.insert(
        REPRESENTATION_METADATA_KEY.to_string(),
        CoordinateRepresentation::Cartesian.as_str().to_string(),
    );
    metadata.insert(FRAME_METADATA_KEY.to_string(), frame.as_str().to_string());
    metadata.insert(
        TIME_SCALE_METADATA_KEY.to_string(),
        time_scale
            .map(TimeScale::as_str)
            .unwrap_or("none")
            .to_string(),
    );
    metadata.insert(
        COVARIANCE_METADATA_KEY.to_string(),
        if has_covariance { "present" } else { "absent" }.to_string(),
    );
    metadata
}

fn coordinate_schema(frame: Frame, time_scale: Option<TimeScale>, has_covariance: bool) -> Schema {
    Schema::new_with_metadata(
        coordinate_fields(),
        coordinate_metadata(
            CARTESIAN_COORDINATE_SCHEMA,
            frame,
            time_scale,
            has_covariance,
        ),
    )
}

fn orbit_schema(frame: Frame, time_scale: Option<TimeScale>, has_covariance: bool) -> Schema {
    let mut fields = vec![
        Field::new("orbit_id", DataType::LargeUtf8, false),
        Field::new("object_id", DataType::LargeUtf8, true),
    ];
    fields.extend(coordinate_fields());
    Schema::new_with_metadata(
        fields,
        coordinate_metadata(ORBIT_SCHEMA, frame, time_scale, has_covariance),
    )
}

fn orbit_variant_schema(
    frame: Frame,
    time_scale: Option<TimeScale>,
    has_covariance: bool,
) -> Schema {
    let mut fields = vec![
        Field::new("orbit_id", DataType::LargeUtf8, false),
        Field::new("object_id", DataType::LargeUtf8, true),
        Field::new("variant_id", DataType::LargeUtf8, true),
        Field::new("weights", DataType::Float64, true),
        Field::new("weights_cov", DataType::Float64, true),
    ];
    fields.extend(coordinate_fields());
    Schema::new_with_metadata(
        fields,
        coordinate_metadata(ORBIT_VARIANT_SCHEMA, frame, time_scale, has_covariance),
    )
}

fn coordinate_arrays(coordinates: &CoordinateBatch) -> SchemaResult<Vec<ArrayRef>> {
    coordinates.validate()?;
    let values = coordinates.values.cartesian().ok_or_else(|| {
        SchemaError::InvalidRecordBatch("expected Cartesian coordinates".to_string())
    })?;
    let rows = values.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(45);
    for column in 0..6 {
        arrays.push(Arc::new(Float64Array::from(
            values.iter().map(|row| row[column]).collect::<Vec<_>>(),
        )) as ArrayRef);
    }

    let (time_days, time_nanos) = match &coordinates.times {
        Some(times) => {
            let days = times
                .epochs
                .iter()
                .map(|epoch| Some(epoch.days))
                .collect::<Vec<_>>();
            let nanos = times
                .epochs
                .iter()
                .map(|epoch| Some(epoch.nanos))
                .collect::<Vec<_>>();
            (days, nanos)
        }
        None => (vec![None; rows], vec![None; rows]),
    };
    arrays.push(Arc::new(Int64Array::from(time_days)) as ArrayRef);
    arrays.push(Arc::new(Int64Array::from(time_nanos)) as ArrayRef);

    let origin_codes = coordinates
        .origins
        .origins
        .iter()
        .map(OriginId::code)
        .collect::<Vec<_>>();
    arrays.push(Arc::new(LargeStringArray::from_iter_values(
        origin_codes.iter().map(String::as_str),
    )) as ArrayRef);

    let covariance = coordinates.covariance.as_ref();
    if let Some(covariance) = covariance {
        if covariance.dimension != 6 {
            return Err(SchemaError::InvalidCovarianceShape {
                rows: covariance.rows,
                dimension: covariance.dimension,
                values: covariance.values_row_major.len(),
            });
        }
    }
    for element in 0..36 {
        let column = match covariance {
            Some(covariance) => (0..rows)
                .map(|row| {
                    if covariance.is_row_valid(row) {
                        Some(covariance.values_row_major[row * 36 + element])
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            None => vec![None; rows],
        };
        arrays.push(Arc::new(Float64Array::from(column)) as ArrayRef);
    }

    Ok(arrays)
}

fn coordinate_to_record_batch(coordinates: &CoordinateBatch) -> SchemaResult<RecordBatch> {
    let time_scale = coordinates.times.as_ref().map(|time| time.scale);
    let schema = coordinate_schema(
        coordinates.frame,
        time_scale,
        coordinates.covariance.is_some(),
    );
    RecordBatch::try_new(Arc::new(schema), coordinate_arrays(coordinates)?)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn orbit_to_record_batch(orbits: &OrbitBatch) -> SchemaResult<RecordBatch> {
    orbits.validate()?;
    let time_scale = orbits.coordinates.times.as_ref().map(|time| time.scale);
    let schema = orbit_schema(
        orbits.coordinates.frame,
        time_scale,
        orbits.coordinates.covariance.is_some(),
    );
    let mut arrays = orbit_metadata_arrays(&orbits.orbit_id, &orbits.object_id);
    arrays.extend(coordinate_arrays(&orbits.coordinates)?);

    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn orbit_variant_to_record_batch(variants: &OrbitVariantBatch) -> SchemaResult<RecordBatch> {
    variants.validate()?;
    let time_scale = variants.coordinates.times.as_ref().map(|time| time.scale);
    let schema = orbit_variant_schema(
        variants.coordinates.frame,
        time_scale,
        variants.coordinates.covariance.is_some(),
    );
    let mut arrays = orbit_metadata_arrays(&variants.orbit_id, &variants.object_id);
    arrays.push(Arc::new(LargeStringArray::from_iter(
        variants
            .variant_id
            .iter()
            .map(|id| id.as_ref().map(|id| id.0.as_str())),
    )) as ArrayRef);
    arrays.push(Arc::new(Float64Array::from(variants.weights.clone())) as ArrayRef);
    arrays.push(Arc::new(Float64Array::from(variants.weights_cov.clone())) as ArrayRef);
    arrays.extend(coordinate_arrays(&variants.coordinates)?);

    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn orbit_metadata_arrays(orbit_id: &[OrbitId], object_id: &[Option<ObjectId>]) -> Vec<ArrayRef> {
    vec![
        Arc::new(LargeStringArray::from_iter_values(
            orbit_id.iter().map(|id| id.0.as_str()),
        )) as ArrayRef,
        Arc::new(LargeStringArray::from_iter(
            object_id
                .iter()
                .map(|id| id.as_ref().map(|id| id.0.as_str())),
        )) as ArrayRef,
    ]
}

fn coordinate_from_record_batch(batch: &RecordBatch) -> SchemaResult<CoordinateBatch> {
    let schema = batch.schema();
    let metadata = schema.metadata();
    let representation = metadata
        .get(REPRESENTATION_METADATA_KEY)
        .map(String::as_str)
        .unwrap_or("cartesian");
    if representation != CoordinateRepresentation::Cartesian.as_str() {
        return Err(SchemaError::InvalidRecordBatch(format!(
            "expected Cartesian representation metadata, got {representation}"
        )));
    }
    let frame = metadata
        .get(FRAME_METADATA_KEY)
        .map(|value| Frame::parse(value))
        .transpose()?
        .unwrap_or(Frame::Unspecified);
    let has_covariance = metadata
        .get(COVARIANCE_METADATA_KEY)
        .is_some_and(|value| value == "present");

    let rows = batch.num_rows();
    let mut values = Vec::with_capacity(rows);
    let x = float64_column(batch, "x")?;
    let y = float64_column(batch, "y")?;
    let z = float64_column(batch, "z")?;
    let vx = float64_column(batch, "vx")?;
    let vy = float64_column(batch, "vy")?;
    let vz = float64_column(batch, "vz")?;
    for row in 0..rows {
        values.push([
            non_null_f64(x, "x", row)?,
            non_null_f64(y, "y", row)?,
            non_null_f64(z, "z", row)?,
            non_null_f64(vx, "vx", row)?,
            non_null_f64(vy, "vy", row)?,
            non_null_f64(vz, "vz", row)?,
        ]);
    }

    let time_days = int64_column(batch, "time_days")?;
    let time_nanos = int64_column(batch, "time_nanos")?;
    let times = parse_time_array(metadata, time_days, time_nanos)?;

    let origins = large_string_column(batch, "origin_code")?;
    let origins = OriginArray::new(
        (0..rows)
            .map(|row| {
                if origins.is_null(row) {
                    return Err(SchemaError::UnsupportedNull {
                        field: "origin_code".to_string(),
                        row,
                    });
                }
                Ok(OriginId::from_code(origins.value(row)))
            })
            .collect::<SchemaResult<Vec<_>>>()?,
    );

    let covariance = if has_covariance {
        Some(parse_covariance(batch, rows)?)
    } else {
        None
    };

    CoordinateBatch::cartesian(values, frame, origins, times, covariance)
}

fn orbit_from_record_batch(batch: &RecordBatch) -> SchemaResult<OrbitBatch> {
    let (orbit_id, object_id) = parse_orbit_metadata(batch)?;
    OrbitBatch::new(orbit_id, object_id, coordinate_from_record_batch(batch)?)
}

fn orbit_variant_from_record_batch(batch: &RecordBatch) -> SchemaResult<OrbitVariantBatch> {
    let rows = batch.num_rows();
    let (orbit_id, object_id) = parse_orbit_metadata(batch)?;
    let variant_id = large_string_column(batch, "variant_id")?;
    let variant_id = (0..rows)
        .map(|row| {
            if variant_id.is_null(row) {
                None
            } else {
                Some(VariantId(variant_id.value(row).to_string()))
            }
        })
        .collect::<Vec<_>>();
    let weights = optional_float64_values(float64_column(batch, "weights")?);
    let weights_cov = optional_float64_values(float64_column(batch, "weights_cov")?);

    OrbitVariantBatch::new(
        orbit_id,
        object_id,
        variant_id,
        weights,
        weights_cov,
        coordinate_from_record_batch(batch)?,
    )
}

fn parse_orbit_metadata(
    batch: &RecordBatch,
) -> SchemaResult<(Vec<OrbitId>, Vec<Option<ObjectId>>)> {
    let rows = batch.num_rows();
    let orbit_id = large_string_column(batch, "orbit_id")?;
    let orbit_id = (0..rows)
        .map(|row| {
            if orbit_id.is_null(row) {
                return Err(SchemaError::UnsupportedNull {
                    field: "orbit_id".to_string(),
                    row,
                });
            }
            Ok(OrbitId(orbit_id.value(row).to_string()))
        })
        .collect::<SchemaResult<Vec<_>>>()?;

    let object_id = large_string_column(batch, "object_id")?;
    let object_id = (0..rows)
        .map(|row| {
            if object_id.is_null(row) {
                None
            } else {
                Some(ObjectId(object_id.value(row).to_string()))
            }
        })
        .collect::<Vec<_>>();

    Ok((orbit_id, object_id))
}

fn optional_float64_values(column: &Float64Array) -> Vec<Option<f64>> {
    (0..column.len())
        .map(|row| {
            if column.is_null(row) {
                None
            } else {
                Some(column.value(row))
            }
        })
        .collect()
}

fn parse_time_array(
    metadata: &HashMap<String, String>,
    days: &Int64Array,
    nanos: &Int64Array,
) -> SchemaResult<Option<TimeArray>> {
    if days.len() != nanos.len() {
        return Err(SchemaError::LengthMismatch {
            field: "time_nanos".to_string(),
            expected: days.len(),
            actual: nanos.len(),
        });
    }
    let has_days = days.null_count() < days.len();
    let has_nanos = nanos.null_count() < nanos.len();
    if !has_days && !has_nanos {
        return Ok(None);
    }
    if days.null_count() != 0 || nanos.null_count() != 0 {
        return Err(SchemaError::InvalidRecordBatch(
            "time_days and time_nanos must be either both fully present or both fully null"
                .to_string(),
        ));
    }
    let scale = metadata
        .get(TIME_SCALE_METADATA_KEY)
        .ok_or_else(|| SchemaError::InvalidTimeScale("missing metadata".to_string()))
        .and_then(|value| TimeScale::parse(value))?;
    TimeArray::new(
        scale,
        (0..days.len())
            .map(|row| Epoch::new(days.value(row), nanos.value(row)))
            .collect(),
    )
    .map(Some)
}

fn parse_covariance(batch: &RecordBatch, rows: usize) -> SchemaResult<CovarianceBatch> {
    let columns = (0..36)
        .map(|element| {
            let row = element / 6;
            let col = element % 6;
            float64_column(batch, &format!("covariance_{row}{col}"))
        })
        .collect::<SchemaResult<Vec<_>>>()?;

    let mut values = vec![f64::NAN; rows * 36];
    let mut row_validity = vec![true; rows];
    for row in 0..rows {
        let null_count = columns.iter().filter(|column| column.is_null(row)).count();
        if null_count == 36 {
            row_validity[row] = false;
            continue;
        }
        if null_count != 0 {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "covariance row {row} must be fully present or fully null"
            )));
        }
        for element in 0..36 {
            values[row * 36 + element] = columns[element].value(row);
        }
    }

    CovarianceBatch::new(
        rows,
        6,
        values,
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )
    .and_then(|covariance| covariance.with_row_validity(Validity::from_bools(&row_validity)))
}

fn float64_column<'a>(batch: &'a RecordBatch, name: &str) -> SchemaResult<&'a Float64Array> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| SchemaError::MissingRequiredField(name.to_string()))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be Float64")))
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> SchemaResult<&'a Int64Array> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| SchemaError::MissingRequiredField(name.to_string()))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be Int64")))
}

fn large_string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> SchemaResult<&'a LargeStringArray> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| SchemaError::MissingRequiredField(name.to_string()))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be LargeUtf8")))
}

fn non_null_f64(column: &Float64Array, field: &str, row: usize) -> SchemaResult<f64> {
    if column.is_null(row) {
        return Err(SchemaError::UnsupportedNull {
            field: field.to_string(),
            row,
        });
    }
    Ok(column.value(row))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinate_batch_with_covariance() -> CoordinateBatch {
        let times =
            TimeArray::from_parts(TimeScale::Tdb, vec![60000, 60001], vec![10, 20]).unwrap();
        let mut values = vec![f64::NAN; 72];
        for (index, value) in values.iter_mut().enumerate().take(36) {
            *value = index as f64;
        }
        let covariance = CovarianceBatch::new(
            2,
            6,
            values,
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap()
        .with_row_validity(Validity::from_bools(&[true, false]))
        .unwrap();
        CoordinateBatch::cartesian(
            vec![
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            Some(times),
            Some(covariance),
        )
        .unwrap()
    }

    #[test]
    fn coordinate_arrow_round_trip_preserves_schema_metadata() {
        let coordinates = coordinate_batch_with_covariance();
        let batch = coordinates.into_record_batch().unwrap();
        let schema = batch.schema();
        let metadata = schema.metadata();
        assert_eq!(
            metadata.get(SCHEMA_METADATA_KEY).unwrap(),
            CARTESIAN_COORDINATE_SCHEMA
        );
        assert_eq!(metadata.get(FRAME_METADATA_KEY).unwrap(), "ecliptic");
        assert_eq!(metadata.get(TIME_SCALE_METADATA_KEY).unwrap(), "tdb");
        assert_eq!(metadata.get(COVARIANCE_METADATA_KEY).unwrap(), "present");

        let round_tripped = CoordinateBatch::try_from_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.frame, Frame::Ecliptic);
        assert_eq!(round_tripped.times.as_ref().unwrap().scale, TimeScale::Tdb);
        assert_eq!(
            round_tripped.origins.origins[0].code(),
            "SOLAR_SYSTEM_BARYCENTER"
        );
        let covariance = round_tripped.covariance.unwrap();
        assert_eq!(covariance.rows, 2);
        assert!(covariance.is_row_valid(0));
        assert!(!covariance.is_row_valid(1));
        assert_eq!(covariance.row_values(0)[35], 35.0);
        assert!(covariance.row_values(1)[0].is_nan());
    }

    #[test]
    fn coordinate_arrow_round_trip_distinguishes_no_covariance_from_nan_covariance() {
        let no_covariance = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Equatorial,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let batch = no_covariance.into_record_batch().unwrap();
        let schema = batch.schema();
        assert_eq!(
            schema.metadata().get(COVARIANCE_METADATA_KEY).unwrap(),
            "absent"
        );
        let round_tripped = CoordinateBatch::try_from_record_batch(&batch).unwrap();
        assert!(round_tripped.covariance.is_none());

        let nan_covariance = CovarianceBatch::new(
            1,
            6,
            vec![f64::NAN; 36],
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let with_nan_covariance = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Equatorial,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            Some(nan_covariance),
        )
        .unwrap();
        let batch = with_nan_covariance.into_record_batch().unwrap();
        let schema = batch.schema();
        assert_eq!(
            schema.metadata().get(COVARIANCE_METADATA_KEY).unwrap(),
            "present"
        );
        let round_tripped = CoordinateBatch::try_from_record_batch(&batch).unwrap();
        let covariance = round_tripped.covariance.unwrap();
        assert!(covariance.is_row_valid(0));
        assert!(covariance.row_values(0).iter().all(|value| value.is_nan()));
    }

    #[test]
    fn orbit_arrow_round_trip_preserves_ids_and_coordinates() {
        let coordinates = coordinate_batch_with_covariance();
        let orbits = OrbitBatch::new(
            vec![
                OrbitId("orbit-1".to_string()),
                OrbitId("orbit-2".to_string()),
            ],
            vec![Some(ObjectId("object-1".to_string())), None],
            coordinates,
        )
        .unwrap();
        let batch = orbits.into_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 47);
        let schema = batch.schema();
        assert_eq!(
            schema.metadata().get(SCHEMA_METADATA_KEY).unwrap(),
            ORBIT_SCHEMA
        );

        let round_tripped = OrbitBatch::try_from_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.orbit_id[0].0, "orbit-1");
        assert_eq!(round_tripped.orbit_id[1].0, "orbit-2");
        assert_eq!(round_tripped.object_id[0].as_ref().unwrap().0, "object-1");
        assert!(round_tripped.object_id[1].is_none());
        assert_eq!(round_tripped.coordinates.len(), 2);
        assert_eq!(round_tripped.coordinates.frame, Frame::Ecliptic);
    }

    #[test]
    fn orbit_variant_arrow_round_trip_preserves_variant_metadata() {
        let coordinates = coordinate_batch_with_covariance();
        let variants = OrbitVariantBatch::new(
            vec![
                OrbitId("orbit-1".to_string()),
                OrbitId("orbit-2".to_string()),
            ],
            vec![Some(ObjectId("object-1".to_string())), None],
            vec![Some(VariantId("variant-1".to_string())), None],
            vec![Some(0.75), Some(0.25)],
            vec![Some(0.5625), None],
            coordinates,
        )
        .unwrap();
        let batch = variants.into_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 50);
        let schema = batch.schema();
        assert_eq!(
            schema.metadata().get(SCHEMA_METADATA_KEY).unwrap(),
            ORBIT_VARIANT_SCHEMA
        );

        let round_tripped = OrbitVariantBatch::try_from_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.orbit_id[0].0, "orbit-1");
        assert_eq!(round_tripped.object_id[0].as_ref().unwrap().0, "object-1");
        assert_eq!(round_tripped.variant_id[0].as_ref().unwrap().0, "variant-1");
        assert!(round_tripped.variant_id[1].is_none());
        assert_eq!(round_tripped.weights, vec![Some(0.75), Some(0.25)]);
        assert_eq!(round_tripped.weights_cov, vec![Some(0.5625), None]);
        assert_eq!(round_tripped.coordinates.len(), 2);
        assert_eq!(round_tripped.coordinates.frame, Frame::Ecliptic);
    }
}
