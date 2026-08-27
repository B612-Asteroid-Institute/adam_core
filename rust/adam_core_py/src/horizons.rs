use crate::http::{get_text, percent_encode};
use adam_core_rs_coords::{
    cometary_to_cartesian6, keplerian_to_cartesian6, CoordinateBatch, DataFrame, EphemerisBatch,
    IntoNestedRecordBatch, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId, TimeArray,
    TimeScale, Validity,
};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StructArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

const HORIZONS_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons.api";
const JD_MINUS_MJD: f64 = 2_400_000.5;
const MU_SUN: f64 = 0.000_295_912_208_284_119_56;

#[derive(Clone, Copy)]
enum QueryKind {
    Vectors,
    Elements,
    Ephemerides,
}

#[derive(Clone)]
struct ParsedTable {
    target_name: String,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn command(object_id: &str, id_type: Option<&str>) -> PyResult<String> {
    match id_type {
        None | Some("id") | Some("majorbody") => Ok(object_id.to_string()),
        Some("smallbody") => Ok(format!("{object_id};")),
        Some("designation") => Ok(format!("DES={object_id};")),
        Some("name") => Ok(format!("NAME={object_id};")),
        Some("asteroid_name") => Ok(format!("ASTNAM={object_id};")),
        Some("comet_name") => Ok(format!("COMNAM={object_id};")),
        Some(other) => Err(value_error(format!("invalid id_type: {other}"))),
    }
}

fn request_url(
    kind: QueryKind,
    object_id: &str,
    epochs: &[f64],
    location: &str,
    id_type: Option<&str>,
    aberrations: &str,
) -> PyResult<String> {
    let command = command(object_id, id_type)?;
    let tlist = epochs
        .iter()
        .map(|value| format!("{value:.15}"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut params: Vec<(&str, String)> = vec![
        ("format", "text".into()),
        ("COMMAND", format!("\"{command}\"")),
        ("CENTER", format!("'{location}'")),
        ("TLIST", tlist),
        ("CSV_FORMAT", "YES".into()),
        ("OBJ_DATA", "YES".into()),
    ];
    match kind {
        QueryKind::Vectors => {
            let correction = match aberrations {
                "geometric" => "NONE",
                "astrometric" => "LT",
                "apparent" => "LT+S",
                other => return Err(value_error(format!("invalid aberrations: {other}"))),
            };
            params.extend([
                ("EPHEM_TYPE", "VECTORS".into()),
                ("OUT_UNITS", "AU-D".into()),
                ("REF_PLANE", "ECLIPTIC".into()),
                ("REF_SYSTEM", "ICRF".into()),
                ("TP_TYPE", "ABSOLUTE".into()),
                ("VEC_LABELS", "YES".into()),
                ("VEC_CORR", format!("\"{correction}\"")),
                ("VEC_DELTA_T", "NO".into()),
            ]);
        }
        QueryKind::Elements => params.extend([
            ("EPHEM_TYPE", "ELEMENTS".into()),
            ("MAKE_EPHEM", "YES".into()),
            ("OUT_UNITS", "AU-D".into()),
            ("REF_SYSTEM", "J2000".into()),
            ("REF_PLANE", "ECLIPTIC".into()),
            ("TP_TYPE", "ABSOLUTE".into()),
            ("ELEM_LABELS", "YES".into()),
        ]),
        QueryKind::Ephemerides => params.extend([
            ("EPHEM_TYPE", "OBSERVER".into()),
            (
                "QUANTITIES",
                format!(
                    "'{}'",
                    (1..=43)
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
            ("SOLAR_ELONG", "\"0,180\"".into()),
            ("LHA_CUTOFF", "0".into()),
            ("CAL_FORMAT", "BOTH".into()),
            ("ANG_FORMAT", "DEG".into()),
            ("APPARENT", "AIRLESS".into()),
            ("REF_SYSTEM", "ICRF".into()),
            ("EXTRA_PREC", "YES".into()),
            ("SKIP_DAYLT", "NO".into()),
        ]),
    }
    Ok(format!(
        "{HORIZONS_URL}?{}",
        params
            .iter()
            .map(|(key, value)| format!("{key}={}", percent_encode(value)))
            .collect::<Vec<_>>()
            .join("&")
    ))
}

fn target_name(response: &str) -> String {
    let raw = response
        .lines()
        .find_map(|line| line.strip_prefix("Target body name:"))
        .unwrap_or("")
        .split("{source:")
        .next()
        .unwrap_or("")
        .trim();
    // astroquery's Horizons parser materializes targetname as a fixed-width
    // U32 column; the truncation is observable in the public object_id.
    raw.chars().take(32).collect()
}

fn parse_response(response: &str) -> PyResult<ParsedTable> {
    let lines: Vec<&str> = response.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.trim() == "$$SOE")
        .ok_or_else(|| value_error(response.to_string()))?;
    let end = lines
        .iter()
        .position(|line| line.trim() == "$$EOE")
        .ok_or_else(|| value_error("Horizons response missing $$EOE"))?;
    let header = lines[..start]
        .iter()
        .rev()
        .find(|line| line.contains(',') && !line.chars().all(|c| c == '*' || c.is_whitespace()))
        .ok_or_else(|| value_error("Horizons response missing CSV header"))?;
    let headers = header
        .split(',')
        .map(|value| value.trim().to_string())
        .collect::<Vec<_>>();
    let rows = lines[start + 1..end]
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line
                .split(',')
                .map(|value| value.trim().to_string())
                .collect::<Vec<_>>();
            while fields.len() > headers.len() && fields.last().is_some_and(String::is_empty) {
                fields.pop();
            }
            fields
        })
        .collect::<Vec<_>>();
    Ok(ParsedTable {
        target_name: target_name(response),
        headers,
        rows,
    })
}

fn column(table: &ParsedTable, name: &str) -> PyResult<usize> {
    table
        .headers
        .iter()
        .position(|header| header == name)
        .ok_or_else(|| value_error(format!("Horizons response missing column {name:?}")))
}

fn number(row: &[String], index: usize, name: &str) -> PyResult<f64> {
    row.get(index)
        .ok_or_else(|| value_error(format!("Horizons row missing {name}")))?
        .parse::<f64>()
        .map_err(|err| value_error(format!("invalid Horizons {name}: {err}")))
}

fn input_times(batch: &Bound<'_, PyAny>, scale: &str) -> PyResult<TimeArray> {
    let batch = RecordBatch::from_pyarrow_bound(batch)
        .map_err(|err| value_error(format!("invalid time RecordBatch: {err}")))?;
    let days = batch
        .column_by_name("days")
        .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| value_error("time batch missing int64 days"))?;
    let nanos = batch
        .column_by_name("nanos")
        .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| value_error("time batch missing int64 nanos"))?;
    TimeArray::from_parts(
        TimeScale::parse(scale).map_err(|err| value_error(err.to_string()))?,
        (0..batch.num_rows()).map(|row| days.value(row)).collect(),
        (0..batch.num_rows()).map(|row| nanos.value(row)).collect(),
    )
    .map_err(|err| value_error(err.to_string()))
}

fn sorted_tdb(times: TimeArray) -> PyResult<TimeArray> {
    let mut epochs = times.epochs;
    epochs.sort_by_key(|epoch| (epoch.days, epoch.nanos));
    TimeArray::new(times.scale, epochs)
        .and_then(|times| times.rescale(TimeScale::Tdb))
        .map_err(|err| value_error(err.to_string()))
}

fn time_values(times: &TimeArray, jd: bool) -> Vec<f64> {
    times
        .epochs
        .iter()
        .map(|epoch| {
            let mjd = epoch.days as f64 + epoch.nanos as f64 / 86_400_000_000_000.0;
            if jd {
                mjd + JD_MINUS_MJD
            } else {
                mjd
            }
        })
        .collect()
}

fn next_response(responses: Option<&[String]>, cursor: &mut usize, url: &str) -> PyResult<String> {
    if let Some(responses) = responses {
        let response = responses
            .get(*cursor)
            .ok_or_else(|| value_error("not enough recorded Horizons responses"))?
            .clone();
        *cursor += 1;
        Ok(response)
    } else {
        get_text(url, std::time::Duration::from_secs(120), "Horizons")
    }
}

fn query_orbits(
    object_ids: &[String],
    times: TimeArray,
    coordinate_type: &str,
    location: &str,
    aberrations: &str,
    id_type: Option<&str>,
    responses: Option<&[String]>,
) -> PyResult<OrbitBatch> {
    if times.is_empty() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(
            "Must have at least one time",
        ));
    }
    if !matches!(coordinate_type, "cartesian" | "keplerian" | "cometary") {
        return Err(value_error(
            "coordinate_type should be one of {'cartesian', 'keplerian', 'cometary'}",
        ));
    }
    let times = sorted_tdb(times)?;
    let mut cursor = 0usize;
    let mut output: Vec<(f64, usize, String, String, [f64; 6])> = Vec::new();
    for chunk in times.epochs.chunks(50) {
        let chunk_times = TimeArray::new(TimeScale::Tdb, chunk.to_vec())
            .map_err(|err| value_error(err.to_string()))?;
        let kind = if coordinate_type == "cartesian" {
            QueryKind::Vectors
        } else {
            QueryKind::Elements
        };
        let epochs = time_values(&chunk_times, coordinate_type == "cartesian");
        for (object_index, object_id) in object_ids.iter().enumerate() {
            let url = request_url(kind, object_id, &epochs, location, id_type, aberrations)?;
            let response = next_response(responses, &mut cursor, &url)?;
            let table = parse_response(&response)?;
            let jd_index = column(&table, "JDTDB")?;
            let indices = if coordinate_type == "cartesian" {
                [
                    column(&table, "X")?,
                    column(&table, "Y")?,
                    column(&table, "Z")?,
                    column(&table, "VX")?,
                    column(&table, "VY")?,
                    column(&table, "VZ")?,
                ]
            } else if coordinate_type == "keplerian" {
                [
                    column(&table, "A")?,
                    column(&table, "EC")?,
                    column(&table, "IN")?,
                    column(&table, "OM")?,
                    column(&table, "W")?,
                    column(&table, "MA")?,
                ]
            } else {
                [
                    column(&table, "QR")?,
                    column(&table, "EC")?,
                    column(&table, "IN")?,
                    column(&table, "OM")?,
                    column(&table, "W")?,
                    column(&table, "Tp")?,
                ]
            };
            for row in &table.rows {
                let jd = number(row, jd_index, "JDTDB")?;
                let mut values = [0.0; 6];
                for (slot, &index) in indices.iter().enumerate() {
                    values[slot] = number(row, index, &table.headers[index])?;
                }
                let cartesian = match coordinate_type {
                    "cartesian" => values,
                    "keplerian" => keplerian_to_cartesian6::<f64>(&values, MU_SUN, 100, 1e-15),
                    "cometary" => {
                        values[5] -= JD_MINUS_MJD;
                        cometary_to_cartesian6::<f64>(
                            &values,
                            jd - JD_MINUS_MJD,
                            MU_SUN,
                            100,
                            1e-15,
                        )
                    }
                    _ => unreachable!(),
                };
                output.push((
                    jd,
                    object_index,
                    format!("{object_index:05}"),
                    table.target_name.clone(),
                    cartesian,
                ));
            }
        }
    }
    output.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let output_times = TimeArray::from_mjd(
        TimeScale::Tdb,
        &output
            .iter()
            .map(|row| row.0 - JD_MINUS_MJD)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| value_error(err.to_string()))?;
    let coordinates = CoordinateBatch::cartesian(
        output.iter().map(|row| row.4).collect(),
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), output.len()),
        Some(output_times),
        None,
    )
    .map_err(|err| value_error(err.to_string()))?;
    OrbitBatch::new(
        output.iter().map(|row| OrbitId(row.2.clone())).collect(),
        output
            .iter()
            .map(|row| Some(ObjectId(row.3.clone())))
            .collect(),
        coordinates,
    )
    .map_err(|err| value_error(err.to_string()))
}

fn query_ephemeris(
    object_ids: &[String],
    codes: &[String],
    times: TimeArray,
    responses: Option<&[String]>,
) -> PyResult<RecordBatch> {
    if codes.len() != times.len() {
        return Err(value_error(
            "observer codes and times must have equal lengths",
        ));
    }
    let utc = times
        .rescale(TimeScale::Utc)
        .map_err(|err| value_error(err.to_string()))?;
    let mut grouped: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for (row, code) in codes.iter().enumerate() {
        grouped.entry(code).or_default().push(utc.epochs[row]);
    }
    let mut cursor = 0usize;
    let mut output: Vec<(usize, f64, String, String, f64, f64, f64, f64)> = Vec::new();
    for (code, epochs) in grouped {
        let group_times =
            TimeArray::new(TimeScale::Utc, epochs).map_err(|err| value_error(err.to_string()))?;
        let jd = time_values(&group_times, true);
        for (object_index, object_id) in object_ids.iter().enumerate() {
            let url = request_url(
                QueryKind::Ephemerides,
                object_id,
                &jd,
                code,
                Some("smallbody"),
                "geometric",
            )?;
            let response = next_response(responses, &mut cursor, &url)?;
            let table = parse_response(&response)?;
            let jd_index = column(&table, "Date_________JDUT")?;
            let ra_index = column(&table, "R.A.___(ICRF)")?;
            let dec_index = column(&table, "DEC____(ICRF)")?;
            let light_index = column(&table, "1-way_down_LT")?;
            let alpha_index = column(&table, "S-T-O")?;
            for row in &table.rows {
                output.push((
                    object_index,
                    number(row, jd_index, "datetime_jd")?,
                    code.to_string(),
                    table.target_name.clone(),
                    number(row, ra_index, "RA")?,
                    number(row, dec_index, "DEC")?,
                    number(row, light_index, "lighttime")? / 1440.0,
                    number(row, alpha_index, "alpha")?,
                ));
            }
        }
    }
    output.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let output_times = TimeArray::from_mjd(
        TimeScale::Utc,
        &output
            .iter()
            .map(|row| row.1 - JD_MINUS_MJD)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| value_error(err.to_string()))?;
    let coordinates = CoordinateBatch::spherical(
        output
            .iter()
            .map(|row| [f64::NAN, row.4, row.5, f64::NAN, f64::NAN, f64::NAN])
            .collect(),
        DataFrame::Ecliptic,
        OriginArray::new(
            output
                .iter()
                .map(|row| OriginId::from_code(&row.2))
                .collect(),
        ),
        Some(output_times),
        None,
    )
    .map_err(|err| value_error(err.to_string()))?;
    let ephemeris = EphemerisBatch::new(
        output
            .iter()
            .map(|row| OrbitId(format!("{:05}", row.0)))
            .collect(),
        output
            .iter()
            .map(|row| Some(ObjectId(row.3.clone())))
            .collect(),
        coordinates,
        None,
        Some(output.iter().map(|row| row.7).collect()),
        output.iter().map(|row| row.6).collect(),
        None,
        Validity::all_valid(output.len()),
    )
    .map_err(|err| value_error(err.to_string()))?;
    let batch = ephemeris
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?;
    null_ephemeris_distance_columns(batch)
}

fn null_ephemeris_distance_columns(batch: RecordBatch) -> PyResult<RecordBatch> {
    let coordinates = batch
        .column_by_name("coordinates")
        .and_then(|array| array.as_any().downcast_ref::<StructArray>())
        .ok_or_else(|| value_error("ephemeris coordinates must be a struct"))?;
    let mut arrays = coordinates.columns().to_vec();
    for name in ["rho", "vrho", "vlon", "vlat"] {
        let (index, _) = coordinates
            .fields()
            .find(name)
            .ok_or_else(|| value_error(format!("coordinates missing {name}")))?;
        arrays[index] = std::sync::Arc::new(arrow_array::Float64Array::new_null(batch.num_rows()));
    }
    let coordinates = StructArray::try_new(coordinates.fields().clone(), arrays, None)
        .map_err(|err| value_error(err.to_string()))?;
    let mut top = batch.columns().to_vec();
    let index = batch
        .schema()
        .index_of("coordinates")
        .map_err(|err| value_error(err.to_string()))?;
    top[index] = std::sync::Arc::new(coordinates);
    RecordBatch::try_new(batch.schema(), top).map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (object_ids, time_batch, time_scale, coordinate_type="cartesian", location="@sun", aberrations="geometric", id_type=None, recorded_responses=None))]
#[allow(clippy::too_many_arguments)]
fn query_horizons_arrow<'py>(
    py: Python<'py>,
    object_ids: Vec<String>,
    time_batch: &Bound<'py, PyAny>,
    time_scale: &str,
    coordinate_type: &str,
    location: &str,
    aberrations: &str,
    id_type: Option<String>,
    recorded_responses: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let times = input_times(time_batch, time_scale)?;
    let orbits = py.allow_threads(|| {
        query_orbits(
            &object_ids,
            times,
            coordinate_type,
            location,
            aberrations,
            id_type.as_deref(),
            recorded_responses.as_deref(),
        )
    })?;
    orbits
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (object_ids, observer_batch, time_scale, recorded_responses=None))]
fn query_horizons_ephemeris_arrow<'py>(
    py: Python<'py>,
    object_ids: Vec<String>,
    observer_batch: &Bound<'py, PyAny>,
    time_scale: &str,
    recorded_responses: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(observer_batch)
        .map_err(|err| value_error(format!("invalid observer query batch: {err}")))?;
    let codes = batch
        .column_by_name("code")
        .and_then(|array| array.as_any().downcast_ref::<LargeStringArray>())
        .ok_or_else(|| value_error("observer query batch missing large_utf8 code"))?;
    let codes: Vec<String> = (0..batch.num_rows())
        .map(|row| codes.value(row).to_string())
        .collect();
    let times = input_times(observer_batch, time_scale)?;
    let output = py.allow_threads(|| {
        query_ephemeris(&object_ids, &codes, times, recorded_responses.as_deref())
    })?;
    output
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (kind, responses, reps, trials, warmup_reps=1))]
fn benchmark_horizons_response_processing(
    kind: &str,
    responses: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(value_error("reps and trials must be >= 1"));
    }
    let process = || -> PyResult<()> {
        let times = TimeArray::from_mjd(TimeScale::Tdb, &[60_310.0])
            .map_err(|err| value_error(err.to_string()))?;
        match kind {
            "vectors" => {
                black_box(query_orbits(
                    &["101955".to_string()],
                    times,
                    "cartesian",
                    "@sun",
                    "geometric",
                    Some("smallbody"),
                    Some(&responses),
                )?);
            }
            "elements" => {
                black_box(query_orbits(
                    &["101955".to_string()],
                    times,
                    "keplerian",
                    "@sun",
                    "geometric",
                    Some("smallbody"),
                    Some(&responses),
                )?);
            }
            "ephemerides" => {
                black_box(query_ephemeris(
                    &["101955".to_string()],
                    &["500".to_string()],
                    times,
                    Some(&responses),
                )?);
            }
            other => return Err(value_error(format!("unknown Horizons kind: {other}"))),
        }
        Ok(())
    };
    let mut output = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            process()?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            process()?;
            samples.push(start.elapsed().as_secs_f64());
        }
        output.push(samples);
    }
    Ok(output)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(query_horizons_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(query_horizons_ephemeris_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(
        benchmark_horizons_response_processing,
        module
    )?)?;
    Ok(())
}
