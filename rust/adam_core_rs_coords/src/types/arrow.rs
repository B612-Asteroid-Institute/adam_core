//! Arrow adapters for the standalone Rust data-model prototypes.
//!
//! These adapters intentionally use a simple flat Cartesian schema as the
//! first RM-STANDALONE-003 bridge. Python/quivr compatibility wrappers can map
//! these canonical columns into nested quivr tables without making quivr own
//! the Rust schema.

use super::{
    CoordinateBatch, CoordinateRepresentation, CoordinateValues, CovarianceBatch, CovarianceUnits,
    EphemerisBatch, Epoch, Frame, ObjectId, ObservatoryCode, ObserverBatch, OrbitBatch, OrbitId,
    OrbitVariantBatch, OriginArray, OriginId, PhysicalParametersBatch, SchemaError, SchemaResult,
    TimeArray, TimeScale, Validity, VariantId,
};
use arrow_array::builder::{Float64Builder, LargeListBuilder, LargeStringBuilder};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, LargeListArray, LargeStringArray, RecordBatch,
    StructArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};
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
const CARTESIAN_NESTED_SCHEMA: &str = "CoordinateBatch.cartesian.nested.quivr.v1";
const ORBIT_NESTED_SCHEMA: &str = "OrbitBatch.cartesian.nested.quivr.v1";
const ORBIT_VARIANT_NESTED_SCHEMA: &str = "OrbitVariantBatch.cartesian.nested.quivr.v1";
const OBSERVER_NESTED_SCHEMA: &str = "ObserverBatch.cartesian.nested.quivr.v1";
const EPHEMERIS_NESTED_SCHEMA: &str = "EphemerisBatch.spherical.nested.quivr.v1";

pub trait ArrowSchemaExport {
    fn schema() -> Schema;
}

pub trait IntoRecordBatch {
    fn into_record_batch(self) -> SchemaResult<RecordBatch>;
}

pub trait TryFromRecordBatch: Sized {
    fn try_from_record_batch(batch: &RecordBatch) -> SchemaResult<Self>;
}

/// Nested, quivr-compatible Arrow round-trip (W1 keystone option (a), bead
/// personal-cmy.13). Emits/consumes the EXACT nested quivr layout: a `coordinates`
/// struct (for orbits) or top-level coordinate columns (for coordinate batches) with
/// nested `time`/`covariance`/`origin` structs and a `large_list` covariance, so a
/// quivr table's Arrow buffers map to a Rust batch with no reshaping and no loss.
pub trait IntoNestedRecordBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch>;
}

pub trait TryFromNestedRecordBatch: Sized {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self>;
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

impl IntoNestedRecordBatch for CoordinateBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        coordinate_to_nested_record_batch(&self)
    }
}

impl TryFromNestedRecordBatch for CoordinateBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        coordinate_from_nested_record_batch(batch)
    }
}

impl IntoNestedRecordBatch for OrbitBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        orbit_to_nested_record_batch(&self)
    }
}

impl TryFromNestedRecordBatch for OrbitBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        orbit_from_nested_record_batch(batch)
    }
}

impl IntoNestedRecordBatch for OrbitVariantBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        orbit_variant_to_nested_record_batch(&self)
    }
}

impl TryFromNestedRecordBatch for OrbitVariantBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        orbit_variant_from_nested_record_batch(batch)
    }
}

impl IntoNestedRecordBatch for ObserverBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        observer_to_nested_record_batch(&self)
    }
}

impl TryFromNestedRecordBatch for ObserverBatch {
    fn try_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<Self> {
        observer_from_nested_record_batch(batch)
    }
}

impl IntoNestedRecordBatch for EphemerisBatch {
    fn into_nested_record_batch(self) -> SchemaResult<RecordBatch> {
        ephemeris_to_nested_record_batch(&self)
    }
}

// ---- nested (quivr-compatible) builders: W1 option (a) ----

fn nested_time_struct(coordinates: &CoordinateBatch, rows: usize) -> SchemaResult<StructArray> {
    let (days, nanos) = match &coordinates.times {
        Some(times) => (
            Int64Array::from_iter_values(times.epochs.iter().map(|epoch| epoch.days)),
            Int64Array::from_iter_values(times.epochs.iter().map(|epoch| epoch.nanos)),
        ),
        None => (Int64Array::new_null(rows), Int64Array::new_null(rows)),
    };
    StructArray::try_new(
        Fields::from(vec![
            Field::new("days", DataType::Int64, true),
            Field::new("nanos", DataType::Int64, true),
        ]),
        vec![Arc::new(days) as ArrayRef, Arc::new(nanos) as ArrayRef],
        None,
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn nested_covariance_struct(
    coordinates: &CoordinateBatch,
    rows: usize,
) -> SchemaResult<StructArray> {
    if let Some(covariance) = coordinates.covariance.as_ref() {
        if covariance.dimension != 6 {
            return Err(SchemaError::InvalidCovarianceShape {
                rows: covariance.rows,
                dimension: covariance.dimension,
                values: covariance.values_row_major.len(),
            });
        }
    }
    let mut validity = vec![false; rows];
    let list = match coordinates.covariance.as_ref() {
        Some(covariance) => {
            let mut builder = LargeListBuilder::new(Float64Builder::new());
            for (row, valid) in validity.iter_mut().enumerate() {
                if covariance.is_row_valid(row) {
                    for element in 0..36 {
                        builder
                            .values()
                            .append_value(covariance.values_row_major[row * 36 + element]);
                    }
                    builder.append(true);
                    *valid = true;
                } else {
                    builder.append(false);
                }
            }
            builder.finish()
        }
        None => {
            LargeListArray::new_null(Arc::new(Field::new("item", DataType::Float64, true)), rows)
        }
    };
    // quivr marks an absent covariance as a null struct row (not just a null list),
    // so mirror that with a struct-level null buffer for lossless round-trips.
    StructArray::try_new(
        Fields::from(vec![Field::new("values", list.data_type().clone(), true)]),
        vec![Arc::new(list) as ArrayRef],
        Some(NullBuffer::from_iter(validity)),
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn nested_origin_struct(coordinates: &CoordinateBatch) -> SchemaResult<StructArray> {
    let mut builder = LargeStringBuilder::new();
    for origin in &coordinates.origins.origins {
        match origin {
            OriginId::SolarSystemBarycenter => builder.append_value("SOLAR_SYSTEM_BARYCENTER"),
            OriginId::Naif(id) => builder.append_value(format!("NAIF:{id}")),
            OriginId::Named(code) => builder.append_value(code),
        }
    }
    let codes = builder.finish();
    StructArray::try_new(
        Fields::from(vec![Field::new("code", DataType::LargeUtf8, true)]),
        vec![Arc::new(codes) as ArrayRef],
        None,
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

/// Field names for every nested-coordinate representation carried over the
/// quivr-compatible Arrow bridge.
fn nested_coordinate_field_names(
    representation: CoordinateRepresentation,
) -> SchemaResult<[&'static str; 6]> {
    match representation {
        CoordinateRepresentation::Cartesian => Ok(["x", "y", "z", "vx", "vy", "vz"]),
        CoordinateRepresentation::Spherical => Ok(["rho", "lon", "lat", "vrho", "vlon", "vlat"]),
        CoordinateRepresentation::Keplerian => Ok(["a", "e", "i", "raan", "ap", "M"]),
        CoordinateRepresentation::Cometary => Ok(["q", "e", "i", "raan", "ap", "tp"]),
        CoordinateRepresentation::Geodetic => Ok(["alt", "lon", "lat", "vup", "veast", "vnorth"]),
    }
}

/// Build the nine quivr coordinate columns (the six representation components plus
/// nested time/covariance/origin structs) as (field, array) pairs. Used both as the
/// top-level columns of a coordinates batch and as the children of the Orbits/
/// VariantOrbits `coordinates` struct.
fn nested_coordinate_named_arrays(
    coordinates: &CoordinateBatch,
) -> SchemaResult<Vec<(Arc<Field>, ArrayRef)>> {
    coordinates.validate()?;
    let field_names = nested_coordinate_field_names(coordinates.values.representation())?;
    let values = coordinates.values.raw_values();
    let rows = values.len();
    let mut out: Vec<(Arc<Field>, ArrayRef)> = Vec::with_capacity(9);
    for (column, name) in field_names.into_iter().enumerate() {
        let array = Arc::new(Float64Array::from_iter_values(
            values.iter().map(|row| row[column]),
        )) as ArrayRef;
        out.push((Arc::new(Field::new(name, DataType::Float64, true)), array));
    }
    let time = nested_time_struct(coordinates, rows)?;
    out.push((
        Arc::new(Field::new("time", time.data_type().clone(), true)),
        Arc::new(time) as ArrayRef,
    ));
    let covariance = nested_covariance_struct(coordinates, rows)?;
    out.push((
        Arc::new(Field::new(
            "covariance",
            covariance.data_type().clone(),
            true,
        )),
        Arc::new(covariance) as ArrayRef,
    ));
    let origin = nested_origin_struct(coordinates)?;
    out.push((
        Arc::new(Field::new("origin", origin.data_type().clone(), true)),
        Arc::new(origin) as ArrayRef,
    ));
    Ok(out)
}

fn coordinate_to_nested_record_batch(coordinates: &CoordinateBatch) -> SchemaResult<RecordBatch> {
    let time_scale = coordinates.times.as_ref().map(|time| time.scale);
    let columns = nested_coordinate_named_arrays(coordinates)?;
    let fields = columns
        .iter()
        .map(|(field, _)| field.as_ref().clone())
        .collect::<Vec<_>>();
    let arrays = columns
        .into_iter()
        .map(|(_, array)| array)
        .collect::<Vec<_>>();
    let mut metadata = coordinate_metadata(
        CARTESIAN_NESTED_SCHEMA,
        coordinates.frame,
        time_scale,
        coordinates.covariance.is_some(),
    );
    // The bare-coordinate nested bridge carries the actual representation (e.g.
    // Spherical for observed astrometry), unlike the always-Cartesian orbit paths.
    metadata.insert(
        REPRESENTATION_METADATA_KEY.to_string(),
        coordinates.values.representation().as_str().to_string(),
    );
    let schema = Schema::new_with_metadata(fields, metadata);
    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn orbit_to_nested_record_batch(orbits: &OrbitBatch) -> SchemaResult<RecordBatch> {
    orbits.validate()?;
    let time_scale = orbits.coordinates.times.as_ref().map(|time| time.scale);
    let columns = nested_coordinate_named_arrays(&orbits.coordinates)?;
    let coord_fields = columns
        .iter()
        .map(|(field, _)| field.clone())
        .collect::<Fields>();
    let coord_arrays = columns
        .iter()
        .map(|(_, array)| array.clone())
        .collect::<Vec<_>>();
    let coordinates = StructArray::try_new(coord_fields, coord_arrays, None)
        .map_err(|err| SchemaError::Arrow(err.to_string()))?;
    let physical_parameters =
        nested_physical_parameters_struct(orbits.physical_parameters.as_ref(), orbits.len())?;

    let fields = vec![
        Field::new("orbit_id", DataType::LargeUtf8, false),
        Field::new("object_id", DataType::LargeUtf8, true),
        Field::new("coordinates", coordinates.data_type().clone(), true),
        Field::new(
            "physical_parameters",
            physical_parameters.data_type().clone(),
            true,
        ),
    ];
    let mut arrays = orbit_metadata_arrays(&orbits.orbit_id, &orbits.object_id);
    arrays.push(Arc::new(coordinates) as ArrayRef);
    arrays.push(Arc::new(physical_parameters) as ArrayRef);
    let schema = Schema::new_with_metadata(
        fields,
        coordinate_metadata(
            ORBIT_NESTED_SCHEMA,
            orbits.coordinates.frame,
            time_scale,
            orbits.coordinates.covariance.is_some(),
        ),
    );
    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn orbit_variant_to_nested_record_batch(variants: &OrbitVariantBatch) -> SchemaResult<RecordBatch> {
    variants.validate()?;
    let rows = variants.coordinates.len();
    let time_scale = variants.coordinates.times.as_ref().map(|time| time.scale);
    let columns = nested_coordinate_named_arrays(&variants.coordinates)?;
    let coord_fields = columns
        .iter()
        .map(|(field, _)| field.clone())
        .collect::<Fields>();
    let coord_arrays = columns
        .iter()
        .map(|(_, array)| array.clone())
        .collect::<Vec<_>>();
    let coordinates = StructArray::try_new(coord_fields, coord_arrays, None)
        .map_err(|err| SchemaError::Arrow(err.to_string()))?;
    let physical_parameters =
        nested_physical_parameters_struct(variants.physical_parameters.as_ref(), rows)?;

    let fields = vec![
        Field::new("orbit_id", DataType::LargeUtf8, false),
        Field::new("object_id", DataType::LargeUtf8, true),
        Field::new("variant_id", DataType::LargeUtf8, true),
        Field::new("weights", DataType::Float64, true),
        Field::new("weights_cov", DataType::Float64, true),
        Field::new("coordinates", coordinates.data_type().clone(), true),
        Field::new(
            "physical_parameters",
            physical_parameters.data_type().clone(),
            true,
        ),
    ];
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from_iter_values(
            variants.orbit_id.iter().map(|id| id.0.as_str()),
        )) as ArrayRef,
        Arc::new(LargeStringArray::from_iter(
            variants
                .object_id
                .iter()
                .map(|id| id.as_ref().map(|id| id.0.as_str())),
        )) as ArrayRef,
        Arc::new(LargeStringArray::from_iter(
            variants
                .variant_id
                .iter()
                .map(|id| id.as_ref().map(|id| id.0.as_str())),
        )) as ArrayRef,
        Arc::new(Float64Array::from(variants.weights.clone())) as ArrayRef,
        Arc::new(Float64Array::from(variants.weights_cov.clone())) as ArrayRef,
        Arc::new(coordinates) as ArrayRef,
        Arc::new(physical_parameters) as ArrayRef,
    ];
    let schema = Schema::new_with_metadata(
        fields,
        coordinate_metadata(
            ORBIT_VARIANT_NESTED_SCHEMA,
            variants.coordinates.frame,
            time_scale,
            variants.coordinates.covariance.is_some(),
        ),
    );
    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn observer_to_nested_record_batch(observers: &ObserverBatch) -> SchemaResult<RecordBatch> {
    observers.validate()?;
    let time_scale = observers.coordinates.times.as_ref().map(|time| time.scale);
    let columns = nested_coordinate_named_arrays(&observers.coordinates)?;
    let coord_fields = columns
        .iter()
        .map(|(field, _)| field.clone())
        .collect::<Fields>();
    let coord_arrays = columns
        .iter()
        .map(|(_, array)| array.clone())
        .collect::<Vec<_>>();
    let coordinates = StructArray::try_new(coord_fields, coord_arrays, None)
        .map_err(|err| SchemaError::Arrow(err.to_string()))?;

    let fields = vec![
        Field::new("code", DataType::LargeUtf8, false),
        Field::new("coordinates", coordinates.data_type().clone(), true),
    ];
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from_iter_values(
            observers.code.iter().map(|code| code.0.as_str()),
        )) as ArrayRef,
        Arc::new(coordinates) as ArrayRef,
    ];
    let schema = Schema::new_with_metadata(
        fields,
        coordinate_metadata(
            OBSERVER_NESTED_SCHEMA,
            observers.coordinates.frame,
            time_scale,
            observers.coordinates.covariance.is_some(),
        ),
    );
    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn ephemeris_to_nested_record_batch(ephemeris: &EphemerisBatch) -> SchemaResult<RecordBatch> {
    ephemeris.validate()?;
    let rows = ephemeris.len();
    let time_scale = ephemeris.coordinates.times.as_ref().map(|time| time.scale);

    let coordinate_columns = nested_coordinate_named_arrays(&ephemeris.coordinates)?;
    let coordinate_fields = coordinate_columns
        .iter()
        .map(|(field, _)| field.clone())
        .collect::<Fields>();
    let coordinate_arrays = coordinate_columns
        .into_iter()
        .map(|(_, array)| array)
        .collect::<Vec<_>>();
    let coordinates = StructArray::try_new(coordinate_fields, coordinate_arrays, None)
        .map_err(|err| SchemaError::Arrow(err.to_string()))?;

    let (aberrated_coordinates, aberrated_scale) = match &ephemeris.aberrated_coordinates {
        Some(aberrated) => {
            let columns = nested_coordinate_named_arrays(aberrated)?;
            let fields = columns
                .iter()
                .map(|(field, _)| field.clone())
                .collect::<Fields>();
            let arrays = columns
                .into_iter()
                .map(|(_, array)| array)
                .collect::<Vec<_>>();
            let scale = aberrated.times.as_ref().map(|time| time.scale);
            (
                StructArray::try_new(fields, arrays, None)
                    .map_err(|err| SchemaError::Arrow(err.to_string()))?,
                scale,
            )
        }
        None => {
            let placeholder = CoordinateBatch::cartesian(
                vec![[f64::NAN; 6]; rows],
                Frame::Ecliptic,
                ephemeris.coordinates.origins.clone(),
                ephemeris.coordinates.times.clone(),
                None,
            )?;
            let columns = nested_coordinate_named_arrays(&placeholder)?;
            let fields = columns
                .iter()
                .map(|(field, _)| field.clone())
                .collect::<Fields>();
            let arrays = columns
                .into_iter()
                .map(|(_, array)| array)
                .collect::<Vec<_>>();
            (
                StructArray::try_new(fields, arrays, Some(NullBuffer::new_null(rows)))
                    .map_err(|err| SchemaError::Arrow(err.to_string()))?,
                time_scale,
            )
        }
    };

    let nullable_values = |values: Option<&Vec<f64>>| {
        Float64Array::from_iter((0..rows).map(|row| {
            values
                .and_then(|values| values.get(row).copied())
                .filter(|value| value.is_finite())
        }))
    };
    let light_time = Float64Array::from_iter(
        ephemeris
            .light_time_days
            .iter()
            .copied()
            .map(|value| value.is_finite().then_some(value)),
    );
    let fields = vec![
        Field::new("orbit_id", DataType::LargeUtf8, false),
        Field::new("object_id", DataType::LargeUtf8, true),
        Field::new("coordinates", coordinates.data_type().clone(), true),
        Field::new("predicted_magnitude_v", DataType::Float64, true),
        Field::new("alpha", DataType::Float64, true),
        Field::new("light_time", DataType::Float64, true),
        Field::new(
            "aberrated_coordinates",
            aberrated_coordinates.data_type().clone(),
            true,
        ),
    ];
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from_iter_values(
            ephemeris.orbit_id.iter().map(|id| id.0.as_str()),
        )),
        Arc::new(LargeStringArray::from_iter(
            ephemeris
                .object_id
                .iter()
                .map(|id| id.as_ref().map(|id| id.0.as_str())),
        )),
        Arc::new(coordinates),
        Arc::new(nullable_values(ephemeris.predicted_magnitude_v.as_ref())),
        Arc::new(nullable_values(ephemeris.alpha_deg.as_ref())),
        Arc::new(light_time),
        Arc::new(aberrated_coordinates),
    ];
    let mut metadata = coordinate_metadata(
        EPHEMERIS_NESTED_SCHEMA,
        ephemeris.coordinates.frame,
        time_scale,
        ephemeris.coordinates.covariance.is_some(),
    );
    metadata.insert(
        "adam_core_aberrated_frame".to_string(),
        Frame::Ecliptic.as_str().to_string(),
    );
    if let Some(scale) = aberrated_scale {
        metadata.insert(
            "adam_core_aberrated_time_scale".to_string(),
            scale.as_str().to_string(),
        );
    }
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, metadata)),
        arrays,
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn observer_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<ObserverBatch> {
    let rows = batch.num_rows();
    let schema = batch.schema();
    let code_array = large_string_column(batch, "code")?;
    let code = (0..rows)
        .map(|row| {
            if code_array.is_null(row) {
                return Err(SchemaError::UnsupportedNull {
                    field: "code".to_string(),
                    row,
                });
            }
            Ok(ObservatoryCode(code_array.value(row).to_string()))
        })
        .collect::<SchemaResult<Vec<_>>>()?;
    let coordinates = batch
        .column_by_name("coordinates")
        .ok_or_else(|| SchemaError::MissingRequiredField("coordinates".to_string()))?;
    let coordinates = array_as_struct(coordinates, "coordinates")?;
    let coordinates = coordinate_from_nested_columns(
        |name| coordinates.column_by_name(name),
        rows,
        coordinates.offset(),
        schema.metadata(),
    )?;
    ObserverBatch::new(code, coordinates)
}

fn coordinate_from_nested_columns<'a, F>(
    column: F,
    rows: usize,
    parent_offset: usize,
    metadata: &HashMap<String, String>,
) -> SchemaResult<CoordinateBatch>
where
    F: Fn(&str) -> Option<&'a ArrayRef>,
{
    let representation = metadata
        .get(REPRESENTATION_METADATA_KEY)
        .map(String::as_str)
        .unwrap_or("cartesian");
    let representation = match representation {
        "cartesian" => CoordinateRepresentation::Cartesian,
        "spherical" => CoordinateRepresentation::Spherical,
        "keplerian" => CoordinateRepresentation::Keplerian,
        "cometary" => CoordinateRepresentation::Cometary,
        "geodetic" => CoordinateRepresentation::Geodetic,
        other => {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "unsupported nested coordinate representation: {other}"
            )))
        }
    };
    let field_names = nested_coordinate_field_names(representation)?;
    let frame = metadata
        .get(FRAME_METADATA_KEY)
        .map(|value| Frame::parse(value))
        .transpose()?
        .unwrap_or(Frame::Unspecified);
    let require = |name: &str| -> SchemaResult<ArrayRef> {
        let array =
            column(name).ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?;
        if array.len() == rows {
            Ok(array.clone())
        } else {
            Ok(array.slice(parent_offset, rows))
        }
    };

    let value_arrays = field_names
        .iter()
        .copied()
        .map(require)
        .collect::<SchemaResult<Vec<_>>>()?;
    let columns = value_arrays
        .iter()
        .zip(field_names.iter())
        .map(|(array, name)| array_as_f64(array, name))
        .collect::<SchemaResult<Vec<_>>>()?;
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut entry = [0.0_f64; 6];
        for (index, column) in columns.iter().enumerate() {
            // quivr coordinate components are nullable. The prior NumPy bridge
            // surfaced an Arrow-null cell as NaN, so preserve that row-level
            // computation contract instead of rejecting a nullable value.
            entry[index] = if column.is_null(row) {
                f64::NAN
            } else {
                column.value(row)
            };
        }
        values.push(entry);
    }

    let time_array = require("time")?;
    let time = array_as_struct(&time_array, "time")?;
    let days_array = sliced_struct_field(time, "days")?;
    let nanos_array = sliced_struct_field(time, "nanos")?;
    let days = days_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch("days must be Int64".to_string()))?;
    let nanos = nanos_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch("nanos must be Int64".to_string()))?;
    let times = parse_time_array(metadata, days, nanos)?;

    let origin_array = require("origin")?;
    let origin = array_as_struct(&origin_array, "origin")?;
    let code_array = sliced_struct_field(origin, "code")?;
    let code = code_array
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch("code must be LargeUtf8".to_string()))?;
    let origins = OriginArray::new(
        (0..rows)
            .map(|row| {
                if code.is_null(row) {
                    return Err(SchemaError::UnsupportedNull {
                        field: "origin.code".to_string(),
                        row,
                    });
                }
                Ok(OriginId::from_code(code.value(row)))
            })
            .collect::<SchemaResult<Vec<_>>>()?,
    );

    let covariance_array = require("covariance")?;
    let covariance = array_as_struct(&covariance_array, "covariance")?;
    let covariance = parse_nested_covariance(covariance, representation, rows)?;

    let values = match representation {
        CoordinateRepresentation::Cartesian => CoordinateValues::Cartesian(values),
        CoordinateRepresentation::Spherical => CoordinateValues::Spherical(values),
        CoordinateRepresentation::Keplerian => CoordinateValues::Keplerian(values),
        CoordinateRepresentation::Cometary => CoordinateValues::Cometary(values),
        CoordinateRepresentation::Geodetic => CoordinateValues::Geodetic(values),
    };
    CoordinateBatch::new(values, frame, origins, times, covariance)
}

fn coordinate_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<CoordinateBatch> {
    let schema = batch.schema();
    coordinate_from_nested_columns(
        |name| batch.column_by_name(name),
        batch.num_rows(),
        0,
        schema.metadata(),
    )
}

/// Decode a nested quivr `VariantOrbits` record batch into the Rust-canonical
/// `OrbitVariantBatch`, including the optional `physical_parameters` child
/// (bead personal-cmy.13.2); an all-null child decodes to `None`.
fn orbit_variant_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<OrbitVariantBatch> {
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
    let schema = batch.schema();
    let coordinates = batch
        .column_by_name("coordinates")
        .ok_or_else(|| SchemaError::MissingRequiredField("coordinates".to_string()))?;
    let coordinates = array_as_struct(coordinates, "coordinates")?;
    let coordinates = coordinate_from_nested_columns(
        |name| coordinates.column_by_name(name),
        rows,
        coordinates.offset(),
        schema.metadata(),
    )?;
    let physical_parameters = batch
        .column_by_name("physical_parameters")
        .map(|column| array_as_struct(column, "physical_parameters"))
        .transpose()?
        .map(|physical_parameters| parse_nested_physical_parameters(physical_parameters, rows))
        .transpose()?
        .flatten();
    let variants = OrbitVariantBatch::new(
        orbit_id,
        object_id,
        variant_id,
        weights,
        weights_cov,
        coordinates,
    )?;
    match physical_parameters {
        Some(physical_parameters) => variants.with_physical_parameters(physical_parameters),
        None => Ok(variants),
    }
}

fn orbit_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<OrbitBatch> {
    let (orbit_id, object_id) = parse_orbit_metadata(batch)?;
    let schema = batch.schema();
    let coordinates = batch
        .column_by_name("coordinates")
        .ok_or_else(|| SchemaError::MissingRequiredField("coordinates".to_string()))?;
    let coordinates = array_as_struct(coordinates, "coordinates")?;
    let coordinates = coordinate_from_nested_columns(
        |name| coordinates.column_by_name(name),
        batch.num_rows(),
        coordinates.offset(),
        schema.metadata(),
    )?;
    let physical_parameters = batch
        .column_by_name("physical_parameters")
        .map(|column| array_as_struct(column, "physical_parameters"))
        .transpose()?
        .map(|physical_parameters| {
            parse_nested_physical_parameters(physical_parameters, batch.num_rows())
        })
        .transpose()?
        .flatten();
    let orbits = OrbitBatch::new(orbit_id, object_id, coordinates)?;
    match physical_parameters {
        Some(physical_parameters) => orbits.with_physical_parameters(physical_parameters),
        None => Ok(orbits),
    }
}

fn parse_nested_covariance(
    covariance: &StructArray,
    representation: CoordinateRepresentation,
    rows: usize,
) -> SchemaResult<Option<CovarianceBatch>> {
    if covariance.null_count() == rows {
        return Ok(None);
    }
    let values_col = sliced_struct_field(covariance, "values")?;
    let list = values_col
        .as_any()
        .downcast_ref::<LargeListArray>()
        .ok_or_else(|| {
            SchemaError::InvalidRecordBatch("covariance.values must be LargeList".to_string())
        })?;
    let mut values = vec![f64::NAN; rows * 36];
    let mut row_validity = vec![true; rows];
    let mut any_present = false;
    for row in 0..rows {
        if covariance.is_null(row) || list.is_null(row) {
            row_validity[row] = false;
            continue;
        }
        let entry = list.value(row);
        let entry = entry
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                SchemaError::InvalidRecordBatch(
                    "covariance.values items must be Float64".to_string(),
                )
            })?;
        if entry.len() != 36 {
            return Err(SchemaError::InvalidRecordBatch(format!(
                "covariance row {row} must have 36 elements, got {}",
                entry.len()
            )));
        }
        // quivr may encode an explicit all-NaN covariance as a valid list
        // whose 36 Float64 children are all null. Treat that as the same absent
        // row represented by a null struct/list, while still rejecting partial
        // element nulls below.
        if entry.null_count() == 36 {
            row_validity[row] = false;
            continue;
        }
        for element in 0..36 {
            values[row * 36 + element] = non_null_f64(entry, "covariance.values", element)?;
        }
        any_present = true;
    }
    if !any_present {
        return Ok(None);
    }
    CovarianceBatch::new(rows, 6, values, CovarianceUnits::Coordinate(representation))
        .and_then(|covariance| covariance.with_row_validity(Validity::from_bools(&row_validity)))
        .map(Some)
}

fn array_as_f64<'a>(array: &'a ArrayRef, name: &str) -> SchemaResult<&'a Float64Array> {
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be Float64")))
}

fn array_as_struct<'a>(array: &'a ArrayRef, name: &str) -> SchemaResult<&'a StructArray> {
    array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be a Struct")))
}

fn sliced_struct_field(array: &StructArray, name: &str) -> SchemaResult<ArrayRef> {
    let child = array
        .column_by_name(name)
        .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?;
    Ok(child.slice(array.offset(), array.len()))
}

fn nested_physical_parameters_struct(
    physical_parameters: Option<&PhysicalParametersBatch>,
    rows: usize,
) -> SchemaResult<StructArray> {
    let columns: [(&str, Vec<Option<f64>>); 6] = match physical_parameters {
        Some(pp) => [
            ("H_v", pp.h_v.clone()),
            ("H_v_sigma", pp.h_v_sigma.clone()),
            ("G", pp.g.clone()),
            ("G_sigma", pp.g_sigma.clone()),
            ("sigma_eff", pp.sigma_eff.clone()),
            ("chi2_red", pp.chi2_red.clone()),
        ],
        None => [
            ("H_v", vec![None; rows]),
            ("H_v_sigma", vec![None; rows]),
            ("G", vec![None; rows]),
            ("G_sigma", vec![None; rows]),
            ("sigma_eff", vec![None; rows]),
            ("chi2_red", vec![None; rows]),
        ],
    };
    // A row is a null struct (quivr `None`) iff it carries no physical information,
    // matching how quivr represents orbits without physical parameters.
    let mut validity = vec![false; rows];
    for (_, values) in &columns {
        for (row, value) in values.iter().enumerate() {
            if value.is_some() {
                validity[row] = true;
            }
        }
    }
    let mut fields = Vec::with_capacity(6);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(6);
    for (name, values) in columns {
        fields.push(Field::new(name, DataType::Float64, true));
        arrays.push(Arc::new(Float64Array::from(values)) as ArrayRef);
    }
    StructArray::try_new(
        Fields::from(fields),
        arrays,
        Some(NullBuffer::from_iter(validity)),
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

fn parse_nested_physical_parameters(
    physical_parameters: &StructArray,
    rows: usize,
) -> SchemaResult<Option<PhysicalParametersBatch>> {
    if physical_parameters.null_count() == rows {
        return Ok(None);
    }
    // Honor the struct-level null buffer: a null struct row means "no value",
    // regardless of any masked child placeholder values quivr may carry.
    let read = |name: &str| -> SchemaResult<Vec<Option<f64>>> {
        let array_ref = sliced_struct_field(physical_parameters, name)?;
        let array = array_ref
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be Float64")))?;
        Ok((0..rows)
            .map(|row| {
                if physical_parameters.is_null(row) || array.is_null(row) {
                    None
                } else {
                    Some(array.value(row))
                }
            })
            .collect())
    };
    let batch = PhysicalParametersBatch {
        h_v: read("H_v")?,
        h_v_sigma: read("H_v_sigma")?,
        g: read("G")?,
        g_sigma: read("G_sigma")?,
        sigma_eff: read("sigma_eff")?,
        chi2_red: read("chi2_red")?,
    };
    if batch.is_all_null() {
        Ok(None)
    } else {
        Ok(Some(batch))
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

    fn expected_quivr_coordinates_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("z", DataType::Float64, true),
            Field::new("vx", DataType::Float64, true),
            Field::new("vy", DataType::Float64, true),
            Field::new("vz", DataType::Float64, true),
            Field::new(
                "time",
                DataType::Struct(Fields::from(vec![
                    Field::new("days", DataType::Int64, true),
                    Field::new("nanos", DataType::Int64, true),
                ])),
                true,
            ),
            Field::new(
                "covariance",
                DataType::Struct(Fields::from(vec![Field::new(
                    "values",
                    DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
                    true,
                )])),
                true,
            ),
            Field::new(
                "origin",
                DataType::Struct(Fields::from(vec![Field::new(
                    "code",
                    DataType::LargeUtf8,
                    true,
                )])),
                true,
            ),
        ]))
    }

    #[test]
    fn orbit_nested_round_trip_matches_quivr_layout() {
        let orbits = OrbitBatch::new(
            vec![
                OrbitId("orbit-1".to_string()),
                OrbitId("orbit-2".to_string()),
            ],
            vec![Some(ObjectId("object-1".to_string())), None],
            coordinate_batch_with_covariance(),
        )
        .unwrap();
        let batch = orbits.into_nested_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 4);
        let schema = batch.schema();
        assert_eq!(
            schema.metadata().get(SCHEMA_METADATA_KEY).unwrap(),
            ORBIT_NESTED_SCHEMA
        );
        // The `coordinates` column type must match quivr's nested struct exactly
        // (W1 option (a): zero-reshape, lossless bridge).
        let coords_index = schema.index_of("coordinates").unwrap();
        assert_eq!(
            schema.field(coords_index).data_type(),
            &expected_quivr_coordinates_type()
        );

        let round_tripped = OrbitBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.orbit_id[0].0, "orbit-1");
        assert_eq!(round_tripped.orbit_id[1].0, "orbit-2");
        assert_eq!(round_tripped.object_id[0].as_ref().unwrap().0, "object-1");
        assert!(round_tripped.object_id[1].is_none());
        assert_eq!(round_tripped.coordinates.frame, Frame::Ecliptic);
        assert_eq!(
            round_tripped.coordinates.times.as_ref().unwrap().scale,
            TimeScale::Tdb
        );
        assert_eq!(
            round_tripped.coordinates.origins.origins[0].code(),
            "SOLAR_SYSTEM_BARYCENTER"
        );
        let values = round_tripped.coordinates.values.cartesian().unwrap();
        assert_eq!(values[0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        assert_eq!(values[1], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]);
        let covariance = round_tripped.coordinates.covariance.unwrap();
        assert!(covariance.is_row_valid(0));
        assert!(!covariance.is_row_valid(1));
        assert_eq!(covariance.row_values(0)[35], 35.0);
        assert!(covariance.row_values(1)[0].is_nan());
        assert!(round_tripped.physical_parameters.is_none());
    }

    #[test]
    fn orbit_nested_round_trip_preserves_physical_parameters() {
        let physical_parameters = PhysicalParametersBatch {
            h_v: vec![Some(15.5), None],
            h_v_sigma: vec![Some(0.1), None],
            g: vec![Some(0.15), None],
            g_sigma: vec![None, None],
            sigma_eff: vec![Some(0.05), None],
            chi2_red: vec![Some(1.2), None],
        };
        let orbits = OrbitBatch::new(
            vec![
                OrbitId("orbit-1".to_string()),
                OrbitId("orbit-2".to_string()),
            ],
            vec![Some(ObjectId("object-1".to_string())), None],
            coordinate_batch_with_covariance(),
        )
        .unwrap()
        .with_physical_parameters(physical_parameters.clone())
        .unwrap();
        let batch = orbits.into_nested_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 4);
        let round_tripped = OrbitBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.physical_parameters, Some(physical_parameters));
    }

    #[test]
    fn orbit_variant_nested_round_trip_preserves_physical_parameters() {
        let physical_parameters = PhysicalParametersBatch {
            h_v: vec![Some(15.5), None],
            h_v_sigma: vec![Some(0.1), None],
            g: vec![Some(0.15), None],
            g_sigma: vec![None, None],
            sigma_eff: vec![Some(0.05), None],
            chi2_red: vec![Some(1.2), None],
        };
        let variants = OrbitVariantBatch::new(
            vec![
                OrbitId("orbit-1".to_string()),
                OrbitId("orbit-2".to_string()),
            ],
            vec![Some(ObjectId("object-1".to_string())), None],
            vec![Some(VariantId("variant-1".to_string())), None],
            vec![Some(0.75), Some(0.25)],
            vec![Some(0.5625), None],
            coordinate_batch_with_covariance(),
        )
        .unwrap()
        .with_physical_parameters(physical_parameters.clone())
        .unwrap();
        let batch = variants.into_nested_record_batch().unwrap();
        assert_eq!(batch.num_columns(), 7);
        let round_tripped = OrbitVariantBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.physical_parameters, Some(physical_parameters));
        assert_eq!(round_tripped.weights, vec![Some(0.75), Some(0.25)]);

        // Without attached parameters the emitted child is all-null and decodes to None.
        let bare = OrbitVariantBatch::try_from_nested_record_batch(
            &OrbitVariantBatch::new(
                vec![
                    OrbitId("orbit-1".to_string()),
                    OrbitId("orbit-2".to_string()),
                ],
                vec![None, None],
                vec![None, None],
                vec![None, None],
                vec![None, None],
                coordinate_batch_with_covariance(),
            )
            .unwrap()
            .into_nested_record_batch()
            .unwrap(),
        )
        .unwrap();
        assert!(bare.physical_parameters.is_none());
    }

    #[test]
    fn coordinate_nested_round_trip_spherical() {
        let coordinates = CoordinateBatch::spherical(
            vec![
                [1.0, 30.0, -10.0, 0.01, 0.02, 0.03],
                [2.0, 45.0, 20.0, 0.04, 0.05, 0.06],
            ],
            Frame::Equatorial,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            Some(TimeArray::from_parts(TimeScale::Utc, vec![60000, 60001], vec![0, 0]).unwrap()),
            None,
        )
        .unwrap();
        let batch = coordinates.into_nested_record_batch().unwrap();
        let schema = batch.schema();
        assert!(schema.index_of("rho").is_ok());
        assert!(schema.index_of("vlat").is_ok());
        assert!(schema.index_of("x").is_err());
        let round_tripped = CoordinateBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.frame, Frame::Equatorial);
        assert_eq!(
            round_tripped.values.representation(),
            CoordinateRepresentation::Spherical
        );
        let values = round_tripped.values.spherical().unwrap();
        assert_eq!(values[0], [1.0, 30.0, -10.0, 0.01, 0.02, 0.03]);
        assert_eq!(values[1], [2.0, 45.0, 20.0, 0.04, 0.05, 0.06]);
    }

    #[test]
    fn coordinate_nested_decode_maps_nullable_components_to_nan() {
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::from_parts(TimeScale::Tdb, vec![60_000], vec![0]).unwrap()),
            None,
        )
        .unwrap();
        let batch = coordinates.into_nested_record_batch().unwrap();
        let mut columns = batch.columns().to_vec();
        columns[batch.schema().index_of("x").unwrap()] =
            Arc::new(Float64Array::from(vec![None])) as ArrayRef;
        let nullable_batch = RecordBatch::try_new(batch.schema(), columns).unwrap();

        let decoded = CoordinateBatch::try_from_nested_record_batch(&nullable_batch).unwrap();
        assert!(decoded.values.cartesian().unwrap()[0][0].is_nan());
    }

    #[test]
    fn orbit_nested_decode_respects_sliced_parent_struct_offsets() {
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 3),
            Some(
                TimeArray::from_parts(TimeScale::Tdb, vec![60_000, 60_001, 60_002], vec![0, 1, 2])
                    .unwrap(),
            ),
            None,
        )
        .unwrap();
        let orbits = OrbitBatch::new(
            vec![
                OrbitId("one".to_string()),
                OrbitId("two".to_string()),
                OrbitId("three".to_string()),
            ],
            vec![None, None, None],
            coordinates,
        )
        .unwrap();
        let batch = orbits.into_nested_record_batch().unwrap().slice(2, 1);

        let decoded = OrbitBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(decoded.orbit_id[0].0, "three");
        assert_eq!(decoded.coordinates.values.cartesian().unwrap()[0][0], 3.0);
        assert_eq!(
            decoded.coordinates.times.unwrap().epochs[0],
            Epoch::new(60_002, 2)
        );
    }

    #[test]
    fn coordinate_nested_decode_respects_sliced_struct_offsets() {
        let covariance = CovarianceBatch::new(
            3,
            6,
            [vec![1.0; 36], vec![2.0; 36], vec![3.0; 36]].concat(),
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            Frame::Ecliptic,
            OriginArray::new(vec![
                OriginId::Named("SUN".to_string()),
                OriginId::Named("EARTH".to_string()),
                OriginId::Named("MARS".to_string()),
            ]),
            Some(
                TimeArray::from_parts(TimeScale::Tdb, vec![60_000, 60_001, 60_002], vec![0, 1, 2])
                    .unwrap(),
            ),
            Some(covariance),
        )
        .unwrap();
        let batch = coordinates.into_nested_record_batch().unwrap().slice(2, 1);

        let decoded = CoordinateBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(decoded.values.cartesian().unwrap()[0][0], 3.0);
        assert_eq!(decoded.times.unwrap().epochs[0], Epoch::new(60_002, 2));
        assert_eq!(
            decoded.origins.origins[0],
            OriginId::Named("MARS".to_string())
        );
        assert_eq!(decoded.covariance.unwrap().row_values(0), &[3.0; 36]);
    }

    #[test]
    fn coordinate_nested_round_trip_supports_all_transform_representations() {
        let cases = [
            (
                CoordinateValues::Keplerian(vec![[1.0, 0.1, 2.0, 3.0, 4.0, 5.0]]),
                CoordinateRepresentation::Keplerian,
                ["a", "e", "i", "raan", "ap", "M"],
            ),
            (
                CoordinateValues::Cometary(vec![[0.9, 0.1, 2.0, 3.0, 4.0, 60_000.0]]),
                CoordinateRepresentation::Cometary,
                ["q", "e", "i", "raan", "ap", "tp"],
            ),
            (
                CoordinateValues::Geodetic(vec![[0.01, 2.0, 3.0, 4.0, 5.0, 6.0]]),
                CoordinateRepresentation::Geodetic,
                ["alt", "lon", "lat", "vup", "veast", "vnorth"],
            ),
        ];
        for (values, representation, fields) in cases {
            let expected = values.raw_values()[0];
            let coordinates = CoordinateBatch::new(
                values,
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
                Some(TimeArray::from_parts(TimeScale::Tdb, vec![60_000], vec![0]).unwrap()),
                None,
            )
            .unwrap();
            let batch = coordinates.into_nested_record_batch().unwrap();
            for field in fields {
                assert!(batch.schema().index_of(field).is_ok(), "missing {field}");
            }
            let round_tripped = CoordinateBatch::try_from_nested_record_batch(&batch).unwrap();
            assert_eq!(round_tripped.representation(), representation);
            assert_eq!(round_tripped.values.raw_values()[0], expected);
        }
    }

    #[test]
    fn observer_nested_round_trip_preserves_codes_and_states() {
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            Some(TimeArray::from_parts(TimeScale::Tdb, vec![60000, 60001], vec![0, 0]).unwrap()),
            None,
        )
        .unwrap();
        let observers = ObserverBatch::new(
            vec![
                ObservatoryCode("X05".to_string()),
                ObservatoryCode("500".to_string()),
            ],
            coordinates,
        )
        .unwrap();
        let batch = observers.into_nested_record_batch().unwrap();
        let round_tripped = ObserverBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.code[0].0, "X05");
        assert_eq!(round_tripped.code[1].0, "500");
        assert_eq!(round_tripped.coordinates.frame, Frame::Ecliptic);
        let values = round_tripped.coordinates.values.cartesian().unwrap();
        assert_eq!(values[0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        assert_eq!(values[1], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn coordinate_nested_round_trip_without_time_or_covariance() {
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
            Frame::Equatorial,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 1),
            None,
            None,
        )
        .unwrap();
        let batch = coordinates.into_nested_record_batch().unwrap();
        let schema = batch.schema();
        assert_eq!(
            schema.metadata().get(SCHEMA_METADATA_KEY).unwrap(),
            CARTESIAN_NESTED_SCHEMA
        );
        // Top-level columns: x..vz + nested time/covariance/origin structs = 9.
        assert_eq!(batch.num_columns(), 9);
        let round_tripped = CoordinateBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_tripped.frame, Frame::Equatorial);
        assert!(round_tripped.times.is_none());
        assert!(round_tripped.covariance.is_none());
        let values = round_tripped.values.cartesian().unwrap();
        assert_eq!(values[0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
    }
}
