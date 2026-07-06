//! Query-client parsers/normalizers (bead personal-cmy.29).
//!
//! Live HTTP/astroquery orchestration remains in the thin Python facade, but
//! deterministic payload parsing and row normalization for the optional network
//! clients is Rust-owned and fixture-testable here:
//!
//! * NEOCC OEF text -> structured elements/covariance/magnitude payload;
//! * SBDB JSON payloads -> cometary coordinate/covariance/physical-parameter
//!   arrays with legacy covariance label/order semantics;
//! * Scout orbit rows -> cometary VariantOrbits construction arrays;
//! * Horizons dataframe rows -> arrays consumed by existing Orbits/Ephemeris
//!   table builders.

use crate::types::{SchemaError, SchemaResult, TimeArray, TimeScale};
use serde_json::{json, Map, Value};

fn invalid(message: impl Into<String>) -> SchemaError {
    SchemaError::InvalidRecordBatch(message.into())
}

fn encode(value: Value) -> SchemaResult<String> {
    serde_json::to_string(&value).map_err(|err| invalid(format!("JSON encode failed: {err}")))
}

fn parse_json(text: &str, label: &str) -> SchemaResult<Value> {
    serde_json::from_str(text).map_err(|err| invalid(format!("invalid {label} JSON: {err}")))
}

fn as_array<'a>(value: &'a Value, label: &str) -> SchemaResult<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("{label} must be an array")))
}

fn number(value: &Value, label: &str) -> SchemaResult<f64> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| invalid(format!("{label} cannot be represented as f64"))),
        Value::String(s) => s
            .parse::<f64>()
            .map_err(|err| invalid(format!("{label} is not numeric: {err}"))),
        Value::Null => Err(invalid(format!("{label} is null"))),
        other => Err(invalid(format!("{label} must be numeric, got {other:?}"))),
    }
}

fn optional_number(value: Option<&Value>) -> Option<f64> {
    value.and_then(|v| number(v, "optional number").ok())
}

fn string_field<'a>(row: &'a Value, key: &str) -> SchemaResult<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("row field {key:?} must be a string")))
}

fn number_field(row: &Value, key: &str) -> SchemaResult<f64> {
    number(
        row.get(key)
            .ok_or_else(|| invalid(format!("row missing field {key:?}")))?,
        key,
    )
}

fn mjd_from_jd_legacy(jd: f64) -> SchemaResult<f64> {
    let time = TimeArray::from_mjd(TimeScale::Tdb, &[jd - 2_400_000.5])?;
    Ok(time.mjd_values()[0])
}

// --- NEOCC OEF ------------------------------------------------------------------

fn upper_triangular_to_full(values: &[f64], dimension: usize) -> SchemaResult<Vec<f64>> {
    let expected = dimension * (dimension + 1) / 2;
    if values.len() != expected {
        return Err(invalid(format!(
            "upper triangular matrix for dimension {dimension} must have {expected} values, got {}",
            values.len()
        )));
    }
    let mut out = vec![0.0; dimension * dimension];
    let mut index = 0;
    for row in 0..dimension {
        for col in row..dimension {
            let value = values[index];
            out[row * dimension + col] = value;
            out[col * dimension + row] = value;
            index += 1;
        }
    }
    Ok(out)
}

fn remove_last_column_of_upper_triangular_7(values: &[f64]) -> SchemaResult<Vec<f64>> {
    if values.len() != 28 {
        return Err(invalid(format!(
            "7x7 upper triangular vector must have 28 values, got {}",
            values.len()
        )));
    }
    let mut full = vec![0.0; 49];
    let mut index = 0;
    for row in 0..7 {
        for col in row..7 {
            full[row * 7 + col] = values[index];
            index += 1;
        }
    }
    let mut out = Vec::with_capacity(21);
    for row in 0..6 {
        for col in row..6 {
            out.push(full[row * 7 + col]);
        }
    }
    Ok(out)
}

pub fn neocc_parse_oef_json(data: &str) -> SchemaResult<String> {
    let lines: Vec<&str> = data.trim().split('\n').collect();
    let mut result = Map::new();

    let mut header = Map::new();
    for line in &lines {
        let stripped = line.trim();
        if stripped == "END_OF_HEADER" {
            break;
        }
        if let Some((key, value)) = stripped.split_once('=') {
            let value = value
                .split('!')
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('\'')
                .to_string();
            header.insert(key.trim().to_string(), Value::String(value));
        }
    }
    result.insert("header".to_string(), Value::Object(header));

    for line in &lines {
        if !["!", " ", "format", "rectype", "refsys", "END_OF_HEADER"]
            .iter()
            .any(|prefix| line.starts_with(prefix))
        {
            result.insert(
                "object_id".to_string(),
                Value::String(line.trim().to_string()),
            );
            break;
        }
    }

    for line in &lines {
        if line.trim().starts_with("KEP") {
            let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
            if parts.len() >= 6 {
                result.insert(
                    "elements".to_string(),
                    json!({
                        "a": parts[0].parse::<f64>().map_err(|e| invalid(format!("bad KEP a: {e}")))?,
                        "e": parts[1].parse::<f64>().map_err(|e| invalid(format!("bad KEP e: {e}")))?,
                        "i": parts[2].parse::<f64>().map_err(|e| invalid(format!("bad KEP i: {e}")))?,
                        "node": parts[3].parse::<f64>().map_err(|e| invalid(format!("bad KEP node: {e}")))?,
                        "peri": parts[4].parse::<f64>().map_err(|e| invalid(format!("bad KEP peri: {e}")))?,
                        "M": parts[5].parse::<f64>().map_err(|e| invalid(format!("bad KEP M: {e}")))?,
                    }),
                );
            }
        }
    }

    for line in &lines {
        if line.trim().starts_with("MJD") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                result.insert(
                    "epoch".to_string(),
                    json!(parts[1]
                        .parse::<f64>()
                        .map_err(|e| invalid(format!("bad MJD epoch: {e}")))?),
                );
                result.insert("time_system".to_string(), json!(parts[2]));
            }
        }
    }

    for line in &lines {
        if line.trim().starts_with("MAG") {
            let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
            if parts.len() >= 2 {
                result.insert(
                    "magnitude".to_string(),
                    json!({
                        "H": parts[0].parse::<f64>().map_err(|e| invalid(format!("bad MAG H: {e}")))?,
                        "G": parts[1].parse::<f64>().map_err(|e| invalid(format!("bad MAG G: {e}")))?,
                    }),
                );
            }
        }
    }

    let mut derived = Map::new();
    for line in &lines {
        let stripped = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if stripped.starts_with('!') && parts.len() >= 3 {
            let key = parts[1].to_lowercase();
            let value = parts[2]
                .parse::<f64>()
                .map(Value::from)
                .unwrap_or_else(|_| Value::String(parts[2].to_string()));
            derived.insert(key, value);
        }
    }
    result.insert("derived".to_string(), Value::Object(derived));

    let mut cov_values = Vec::new();
    let mut cor_values = Vec::new();
    for line in &lines {
        let stripped = line.trim();
        if stripped.starts_with("COV") {
            for token in line.split_whitespace().skip(1) {
                cov_values.push(
                    token
                        .parse::<f64>()
                        .map_err(|e| invalid(format!("bad COV value {token:?}: {e}")))?,
                );
            }
        }
        if stripped.starts_with("COR") {
            for token in line.split_whitespace().skip(1) {
                cor_values.push(
                    token
                        .parse::<f64>()
                        .map_err(|e| invalid(format!("bad COR value {token:?}: {e}")))?,
                );
            }
        }
    }
    if !cov_values.is_empty() {
        let values = if cov_values.len() == 28 {
            remove_last_column_of_upper_triangular_7(&cov_values)?
        } else {
            cov_values
        };
        result.insert(
            "covariance".to_string(),
            json!(upper_triangular_to_full(&values, 6)?),
        );
    }
    if !cor_values.is_empty() {
        let values = if cor_values.len() == 28 {
            remove_last_column_of_upper_triangular_7(&cor_values)?
        } else {
            cor_values
        };
        result.insert(
            "correlation".to_string(),
            json!(upper_triangular_to_full(&values, 6)?),
        );
    }

    encode(Value::Object(result))
}

// --- SBDB -----------------------------------------------------------------------

fn sbdb_elements_map(elements: &Value) -> SchemaResult<Map<String, Value>> {
    let elements = as_array(elements, "SBDB orbit elements")?;
    let mut out = Map::new();
    for element in elements {
        if let Value::Object(obj) = element {
            if let Some(name) = obj.get("name").and_then(Value::as_str) {
                out.insert(name.to_string(), Value::Object(obj.clone()));
            }
        }
    }
    Ok(out)
}

fn sbdb_element<'a>(elements: &'a Map<String, Value>, name: &str) -> SchemaResult<&'a Value> {
    elements
        .get(name)
        .ok_or_else(|| invalid(format!("SBDB orbit elements missing {name:?}.")))
}

fn sbdb_element_value(elements: &Map<String, Value>, name: &str) -> SchemaResult<f64> {
    let element = sbdb_element(elements, name)?;
    number(
        element
            .get("value")
            .ok_or_else(|| invalid(format!("SBDB orbit element {name:?} is missing a value.")))?,
        name,
    )
}

fn sbdb_element_sigma(elements: &Map<String, Value>, name: &str) -> SchemaResult<f64> {
    let element = sbdb_element(elements, name)?;
    match element.get("sigma") {
        Some(Value::Null) | None => Ok(f64::NAN),
        Some(value) => Ok(number(value, name).unwrap_or(f64::NAN)),
    }
}

fn sbdb_matrix6_from_json(value: &Value) -> SchemaResult<Vec<f64>> {
    let rows = as_array(value, "SBDB covariance data")?;
    if rows.len() < 6 {
        return Err(invalid("SBDB covariance data must have at least 6 rows"));
    }
    let mut out = Vec::with_capacity(36);
    for (row_index, row) in rows.iter().take(6).enumerate() {
        let cols = as_array(row, "SBDB covariance row")?;
        if cols.len() < 6 {
            return Err(invalid(format!(
                "SBDB covariance row {row_index} must have at least 6 columns"
            )));
        }
        for (col_index, value) in cols.iter().take(6).enumerate() {
            out.push(number(
                value,
                &format!("SBDB covariance[{row_index},{col_index}]"),
            )?);
        }
    }
    Ok(out)
}

fn diagonal_covariance(sigmas: &[f64; 6]) -> Vec<f64> {
    let mut out = vec![0.0; 36];
    for (i, sigma) in sigmas.iter().enumerate() {
        out[i * 6 + i] = sigma * sigma;
    }
    out
}

fn convert_sbdb_covariances(matrix: &[f64]) -> Vec<f64> {
    let get = |row: usize, col: usize| matrix[row * 6 + col];
    let mut out = vec![0.0; 36];
    let set_sym = |out: &mut Vec<f64>, row: usize, col: usize, value: f64| {
        out[row * 6 + col] = value;
        out[col * 6 + row] = value;
    };
    out[0] = get(1, 1);
    set_sym(&mut out, 1, 0, get(0, 1));
    set_sym(&mut out, 2, 0, get(5, 1));
    set_sym(&mut out, 3, 0, get(3, 1));
    set_sym(&mut out, 4, 0, get(4, 1));
    set_sym(&mut out, 5, 0, get(2, 1));

    out[7] = get(0, 0);
    set_sym(&mut out, 2, 1, get(5, 0));
    set_sym(&mut out, 3, 1, get(3, 0));
    set_sym(&mut out, 4, 1, get(4, 0));
    set_sym(&mut out, 5, 1, get(2, 0));

    out[14] = get(5, 5);
    set_sym(&mut out, 3, 2, get(3, 5));
    set_sym(&mut out, 4, 2, get(4, 5));
    set_sym(&mut out, 5, 2, get(2, 5));

    out[21] = get(3, 3);
    set_sym(&mut out, 4, 3, get(4, 3));
    set_sym(&mut out, 5, 3, get(2, 3));

    out[28] = get(4, 4);
    set_sym(&mut out, 5, 4, get(2, 4));

    out[35] = get(2, 2);
    out
}

fn phys_par_value(entry: Option<&Value>) -> Option<f64> {
    optional_number(entry.and_then(|e| e.get("value")))
}

fn phys_par_sigma(entry: Option<&Value>) -> Option<f64> {
    optional_number(entry.and_then(|e| e.get("sigma")))
}

fn py_truthy_or(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match left {
        Some(value) if value != 0.0 => Some(value),
        _ => right,
    }
}

fn phys_par_from_payload(payload: &Value) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let mut by_name: Map<String, Value> = Map::new();
    if let Some(items) = payload.get("phys_par").and_then(Value::as_array) {
        for item in items {
            if let Value::Object(obj) = item {
                if let Some(name) = obj.get("name").and_then(Value::as_str) {
                    by_name.insert(name.to_string(), Value::Object(obj.clone()));
                }
            }
        }
    }
    let h = py_truthy_or(
        phys_par_value(by_name.get("H")),
        phys_par_value(by_name.get("H_mag")),
    );
    let h_sigma = py_truthy_or(
        phys_par_sigma(by_name.get("H")),
        phys_par_sigma(by_name.get("H_mag")),
    );
    let g = phys_par_value(by_name.get("G"));
    let g_sigma = phys_par_sigma(by_name.get("G"));
    (h, h_sigma, g, g_sigma)
}

fn json_or_nan(value: Option<f64>) -> Value {
    match value {
        Some(v) if v.is_finite() => json!(v),
        Some(_) | None => Value::Null,
    }
}

pub fn sbdb_normalize_payloads_json(ids_json: &str, payloads_json: &str) -> SchemaResult<String> {
    let ids_value = parse_json(ids_json, "SBDB ids")?;
    let payloads_value = parse_json(payloads_json, "SBDB payloads")?;
    let ids = as_array(&ids_value, "SBDB ids")?;
    let payloads = as_array(&payloads_value, "SBDB payloads")?;
    if ids.len() != payloads.len() {
        return Err(invalid("ids and payloads must have the same length."));
    }

    let expected_labels = ["e", "q", "tp", "node", "peri", "i"];
    let mut orbit_ids = Vec::with_capacity(payloads.len());
    let mut object_ids = Vec::with_capacity(payloads.len());
    let mut coords = Vec::with_capacity(payloads.len());
    let mut covariances = Vec::with_capacity(payloads.len());
    let mut times_jd = Vec::with_capacity(payloads.len());
    let mut h = Vec::with_capacity(payloads.len());
    let mut h_sigma = Vec::with_capacity(payloads.len());
    let mut g = Vec::with_capacity(payloads.len());
    let mut g_sigma = Vec::with_capacity(payloads.len());

    for (i, payload) in payloads.iter().enumerate() {
        let obj_id = ids[i].as_str().unwrap_or("");
        let obj = payload
            .get("object")
            .ok_or_else(|| invalid(format!("object {obj_id} was not found")))?;
        let orbit = payload
            .get("orbit")
            .ok_or_else(|| invalid(format!("SBDB payload for {obj_id:?} missing 'orbit'.")))?;

        orbit_ids.push(format!("{i:05}"));
        object_ids.push(
            obj.get("fullname")
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_else(|| "None".to_string()),
        );

        let mut elements_list = orbit.get("elements");
        let mut epoch_jd = number(
            orbit
                .get("epoch")
                .ok_or_else(|| invalid("SBDB orbit missing epoch"))?,
            "SBDB epoch",
        )?;
        let mut cov_matrix: Option<Vec<f64>> = None;

        if let Some(cov) = orbit.get("covariance").and_then(Value::as_object) {
            if let Some(data) = cov.get("data") {
                if let Some(labels) = cov.get("labels").and_then(Value::as_array) {
                    let labels6: Vec<String> = labels
                        .iter()
                        .take(6)
                        .map(|v| v.as_str().unwrap_or("").to_string())
                        .collect();
                    if labels6 != expected_labels {
                        return Err(invalid(format!(
                            "Expected covariance matrix labels to be {:?} in the first 6 entries, got {:?}.",
                            expected_labels, labels6
                        )));
                    }
                }
                cov_matrix = Some(sbdb_matrix6_from_json(data)?);
                if let Some(cov_elements) = cov.get("elements") {
                    if !cov_elements.is_null() {
                        elements_list = Some(cov_elements);
                        if let Some(cov_epoch) = cov.get("epoch") {
                            if !cov_epoch.is_null() {
                                epoch_jd = number(cov_epoch, "SBDB covariance epoch")?;
                            }
                        }
                    }
                }
            }
        }

        let elements = sbdb_elements_map(elements_list.ok_or_else(|| {
            invalid(format!(
                "SBDB payload for {obj_id:?} missing orbit elements."
            ))
        })?)?;
        let cov_matrix = cov_matrix.unwrap_or_else(|| {
            let sigmas = [
                sbdb_element_sigma(&elements, "e").unwrap_or(f64::NAN),
                sbdb_element_sigma(&elements, "q").unwrap_or(f64::NAN),
                sbdb_element_sigma(&elements, "tp").unwrap_or(f64::NAN),
                sbdb_element_sigma(&elements, "om").unwrap_or(f64::NAN),
                sbdb_element_sigma(&elements, "w").unwrap_or(f64::NAN),
                sbdb_element_sigma(&elements, "i").unwrap_or(f64::NAN),
            ];
            diagonal_covariance(&sigmas)
        });

        let q = sbdb_element_value(&elements, "q")?;
        let e = sbdb_element_value(&elements, "e")?;
        let inc = sbdb_element_value(&elements, "i")?;
        let om = sbdb_element_value(&elements, "om")?;
        let w = sbdb_element_value(&elements, "w")?;
        let tp_mjd = mjd_from_jd_legacy(sbdb_element_value(&elements, "tp")?)?;
        coords.push(vec![q, e, inc, om, w, tp_mjd]);
        covariances.push(convert_sbdb_covariances(&cov_matrix));
        times_jd.push(epoch_jd);

        let (row_h, row_h_sigma, row_g, row_g_sigma) = phys_par_from_payload(payload);
        h.push(json_or_nan(row_h));
        h_sigma.push(json_or_nan(row_h_sigma));
        g.push(json_or_nan(row_g));
        g_sigma.push(json_or_nan(row_g_sigma));
    }

    encode(json!({
        "orbit_id": orbit_ids,
        "object_id": object_ids,
        "coords_cometary": coords,
        "covariances_cometary": covariances,
        "times_jd": times_jd,
        "physical_parameters": {
            "H_v": h,
            "H_v_sigma": h_sigma,
            "G": g,
            "G_sigma": g_sigma,
        }
    }))
}

// --- Scout ----------------------------------------------------------------------

pub fn scout_normalize_orbits_json(object_id: &str, rows_json: &str) -> SchemaResult<String> {
    let value = parse_json(rows_json, "Scout rows")?;
    let rows = as_array(&value, "Scout rows")?;
    let mut orbit_id = Vec::with_capacity(rows.len());
    let mut variant_id = Vec::with_capacity(rows.len());
    let mut object_ids = Vec::with_capacity(rows.len());
    let mut coords = Vec::with_capacity(rows.len());
    let mut times_jd = Vec::with_capacity(rows.len());
    for row in rows {
        let idx = row
            .get("idx")
            .ok_or_else(|| invalid("Scout row missing idx"))?
            .to_string()
            .trim_matches('"')
            .to_string();
        orbit_id.push(idx.clone());
        variant_id.push(idx);
        object_ids.push(object_id.to_string());
        coords.push(vec![
            number_field(row, "qr")?,
            number_field(row, "ec")?,
            number_field(row, "inc")?,
            number_field(row, "om")?,
            number_field(row, "w")?,
            number_field(row, "tp")? - 2_400_000.5,
        ]);
        times_jd.push(number_field(row, "epoch")?);
    }
    encode(json!({
        "orbit_id": orbit_id,
        "variant_id": variant_id,
        "object_id": object_ids,
        "coords_cometary": coords,
        "times_jd": times_jd,
    }))
}

// --- Horizons -------------------------------------------------------------------

pub fn horizons_vectors_normalize_json(rows_json: &str) -> SchemaResult<String> {
    let value = parse_json(rows_json, "Horizons vector rows")?;
    let rows = as_array(&value, "Horizons vector rows")?;
    let mut orbit_id = Vec::with_capacity(rows.len());
    let mut object_id = Vec::with_capacity(rows.len());
    let mut times_jd = Vec::with_capacity(rows.len());
    let mut coords = Vec::with_capacity(rows.len());
    for row in rows {
        orbit_id.push(string_field(row, "orbit_id")?.to_string());
        object_id.push(string_field(row, "targetname")?.to_string());
        times_jd.push(number_field(row, "datetime_jd")?);
        coords.push(vec![
            number_field(row, "x")?,
            number_field(row, "y")?,
            number_field(row, "z")?,
            number_field(row, "vx")?,
            number_field(row, "vy")?,
            number_field(row, "vz")?,
        ]);
    }
    encode(json!({
        "orbit_id": orbit_id,
        "object_id": object_id,
        "times_jd": times_jd,
        "coords_cartesian": coords,
    }))
}

pub fn horizons_elements_normalize_json(
    rows_json: &str,
    coordinate_type: &str,
) -> SchemaResult<String> {
    let value = parse_json(rows_json, "Horizons element rows")?;
    let rows = as_array(&value, "Horizons element rows")?;
    let mut orbit_id = Vec::with_capacity(rows.len());
    let mut object_id = Vec::with_capacity(rows.len());
    let mut times_jd = Vec::with_capacity(rows.len());
    let mut coords = Vec::with_capacity(rows.len());
    for row in rows {
        orbit_id.push(string_field(row, "orbit_id")?.to_string());
        object_id.push(string_field(row, "targetname")?.to_string());
        times_jd.push(number_field(row, "datetime_jd")?);
        match coordinate_type {
            "keplerian" => coords.push(vec![
                number_field(row, "a")?,
                number_field(row, "e")?,
                number_field(row, "incl")?,
                number_field(row, "Omega")?,
                number_field(row, "w")?,
                number_field(row, "M")?,
            ]),
            "cometary" => coords.push(vec![
                number_field(row, "q")?,
                number_field(row, "e")?,
                number_field(row, "incl")?,
                number_field(row, "Omega")?,
                number_field(row, "w")?,
                mjd_from_jd_legacy(number_field(row, "Tp_jd")?)?,
            ]),
            other => {
                return Err(invalid(format!(
                    "unsupported Horizons element type {other:?}"
                )))
            }
        }
    }
    encode(json!({
        "orbit_id": orbit_id,
        "object_id": object_id,
        "times_jd": times_jd,
        "coords": coords,
    }))
}

pub fn horizons_ephemeris_normalize_json(rows_json: &str) -> SchemaResult<String> {
    let value = parse_json(rows_json, "Horizons ephemeris rows")?;
    let rows = as_array(&value, "Horizons ephemeris rows")?;
    let mut orbit_id = Vec::with_capacity(rows.len());
    let mut object_id = Vec::with_capacity(rows.len());
    let mut light_time = Vec::with_capacity(rows.len());
    let mut alpha = Vec::with_capacity(rows.len());
    let mut times_jd = Vec::with_capacity(rows.len());
    let mut lon = Vec::with_capacity(rows.len());
    let mut lat = Vec::with_capacity(rows.len());
    let mut observatory_code = Vec::with_capacity(rows.len());
    for row in rows {
        orbit_id.push(string_field(row, "orbit_id")?.to_string());
        object_id.push(string_field(row, "targetname")?.to_string());
        light_time.push(number_field(row, "lighttime")? / 1440.0);
        alpha.push(number_field(row, "alpha")?);
        times_jd.push(number_field(row, "datetime_jd")?);
        lon.push(number_field(row, "RA")?);
        lat.push(number_field(row, "DEC")?);
        observatory_code.push(string_field(row, "observatory_code")?.to_string());
    }
    encode(json!({
        "orbit_id": orbit_id,
        "object_id": object_id,
        "light_time": light_time,
        "alpha": alpha,
        "times_jd": times_jd,
        "lon": lon,
        "lat": lat,
        "observatory_code": observatory_code,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neocc_upper_triangular_28_drops_last_column() {
        let values: Vec<f64> = (0..28).map(|x| x as f64).collect();
        let out = remove_last_column_of_upper_triangular_7(&values).unwrap();
        assert_eq!(out.len(), 21);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[5], 5.0);
        assert_eq!(out[20], 25.0);
    }

    #[test]
    fn scout_normalizer_parses_rows() {
        let out = scout_normalize_orbits_json(
            "2024AA",
            r#"[{"idx":0,"epoch":"2460000.5","ec":"0.1","qr":"1.0","tp":"2459000.5","om":"2","w":"3","inc":"4"}]"#,
        )
        .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["orbit_id"][0], "0");
        assert_eq!(v["object_id"][0], "2024AA");
        assert_eq!(v["coords_cometary"][0][5], 59000.0);
    }
}
