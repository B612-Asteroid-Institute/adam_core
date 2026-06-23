//! Arrow adapters for the standalone Rust data-model prototypes.
//!
//! These adapters intentionally use a simple flat Cartesian schema as the
//! first RM-STANDALONE-003 bridge. Python/quivr compatibility wrappers can map
//! these canonical columns into nested quivr tables without making quivr own
//! the Rust schema.

use super::{
    CoordinateBatch, CoordinateRepresentation, CovarianceBatch, CovarianceUnits, Epoch, Frame,
    ObjectId, OrbitBatch, OrbitId, OrbitVariantBatch, OriginArray, OriginId,
    PhysicalParametersBatch, SchemaError, SchemaResult, TimeArray, TimeScale, Validity, VariantId,
};
use arrow_array::builder::{Float64Builder, LargeListBuilder};
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

// ---- nested (quivr-compatible) builders: W1 option (a) ----

fn nested_time_struct(coordinates: &CoordinateBatch, rows: usize) -> SchemaResult<StructArray> {
    let (days, nanos): (Vec<Option<i64>>, Vec<Option<i64>>) = match &coordinates.times {
        Some(times) => (
            times.epochs.iter().map(|epoch| Some(epoch.days)).collect(),
            times.epochs.iter().map(|epoch| Some(epoch.nanos)).collect(),
        ),
        None => (vec![None; rows], vec![None; rows]),
    };
    StructArray::try_new(
        Fields::from(vec![
            Field::new("days", DataType::Int64, true),
            Field::new("nanos", DataType::Int64, true),
        ]),
        vec![
            Arc::new(Int64Array::from(days)) as ArrayRef,
            Arc::new(Int64Array::from(nanos)) as ArrayRef,
        ],
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
    let mut builder = LargeListBuilder::new(Float64Builder::new());
    let mut validity = vec![false; rows];
    match coordinates.covariance.as_ref() {
        Some(covariance) => {
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
        }
        None => {
            for _ in 0..rows {
                builder.append(false);
            }
        }
    }
    let list = builder.finish();
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
    let codes =
        LargeStringArray::from_iter_values(coordinates.origins.origins.iter().map(OriginId::code));
    StructArray::try_new(
        Fields::from(vec![Field::new("code", DataType::LargeUtf8, true)]),
        vec![Arc::new(codes) as ArrayRef],
        None,
    )
    .map_err(|err| SchemaError::Arrow(err.to_string()))
}

/// Build the nine quivr coordinate columns (x..vz plus nested time/covariance/origin
/// structs) as (field, array) pairs. Used both as the top-level columns of a
/// CartesianCoordinates batch and as the children of the Orbits `coordinates` struct.
fn nested_coordinate_named_arrays(
    coordinates: &CoordinateBatch,
) -> SchemaResult<Vec<(Arc<Field>, ArrayRef)>> {
    coordinates.validate()?;
    let values = coordinates.values.cartesian().ok_or_else(|| {
        SchemaError::InvalidRecordBatch("expected Cartesian coordinates".to_string())
    })?;
    let rows = values.len();
    let mut out: Vec<(Arc<Field>, ArrayRef)> = Vec::with_capacity(9);
    for (column, name) in ["x", "y", "z", "vx", "vy", "vz"].into_iter().enumerate() {
        let array = Arc::new(Float64Array::from(
            values.iter().map(|row| row[column]).collect::<Vec<_>>(),
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
    let schema = Schema::new_with_metadata(
        fields,
        coordinate_metadata(
            CARTESIAN_NESTED_SCHEMA,
            coordinates.frame,
            time_scale,
            coordinates.covariance.is_some(),
        ),
    );
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
    // Variants carry no physical parameters; emit the quivr column as all-null.
    let physical_parameters = nested_physical_parameters_struct(None, rows)?;

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

fn coordinate_from_nested_columns<'a, F>(
    column: F,
    rows: usize,
    metadata: &HashMap<String, String>,
) -> SchemaResult<CoordinateBatch>
where
    F: Fn(&str) -> Option<&'a ArrayRef>,
{
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
    let require = |name: &str| -> SchemaResult<&'a ArrayRef> {
        column(name).ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))
    };

    let x = array_as_f64(require("x")?, "x")?;
    let y = array_as_f64(require("y")?, "y")?;
    let z = array_as_f64(require("z")?, "z")?;
    let vx = array_as_f64(require("vx")?, "vx")?;
    let vy = array_as_f64(require("vy")?, "vy")?;
    let vz = array_as_f64(require("vz")?, "vz")?;
    let mut values = Vec::with_capacity(rows);
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

    let time = array_as_struct(require("time")?, "time")?;
    let days = struct_field_i64(time, "days")?;
    let nanos = struct_field_i64(time, "nanos")?;
    let times = parse_time_array(metadata, days, nanos)?;

    let origin = array_as_struct(require("origin")?, "origin")?;
    let code = struct_field_lstr(origin, "code")?;
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

    let covariance = array_as_struct(require("covariance")?, "covariance")?;
    let covariance = parse_nested_covariance(covariance, rows)?;

    CoordinateBatch::cartesian(values, frame, origins, times, covariance)
}

fn coordinate_from_nested_record_batch(batch: &RecordBatch) -> SchemaResult<CoordinateBatch> {
    let schema = batch.schema();
    coordinate_from_nested_columns(
        |name| batch.column_by_name(name),
        batch.num_rows(),
        schema.metadata(),
    )
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
    rows: usize,
) -> SchemaResult<Option<CovarianceBatch>> {
    let values_col = covariance
        .column_by_name("values")
        .ok_or_else(|| SchemaError::MissingRequiredField("covariance.values".to_string()))?;
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
        for element in 0..36 {
            values[row * 36 + element] = non_null_f64(entry, "covariance.values", element)?;
        }
        any_present = true;
    }
    if !any_present {
        return Ok(None);
    }
    CovarianceBatch::new(
        rows,
        6,
        values,
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )
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

fn struct_field_i64<'a>(array: &'a StructArray, name: &str) -> SchemaResult<&'a Int64Array> {
    array
        .column_by_name(name)
        .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be Int64")))
}

fn struct_field_lstr<'a>(array: &'a StructArray, name: &str) -> SchemaResult<&'a LargeStringArray> {
    array
        .column_by_name(name)
        .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be LargeUtf8")))
}

fn struct_field_f64<'a>(array: &'a StructArray, name: &str) -> SchemaResult<&'a Float64Array> {
    array
        .column_by_name(name)
        .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| SchemaError::InvalidRecordBatch(format!("{name} must be Float64")))
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
    // Honor the struct-level null buffer: a null struct row means "no value",
    // regardless of any masked child placeholder values quivr may carry.
    let read = |name: &str| -> SchemaResult<Vec<Option<f64>>> {
        let array = struct_field_f64(physical_parameters, name)?;
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
