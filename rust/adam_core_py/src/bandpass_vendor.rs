use crate::coordinate_ops::bench_result;
use arrow::pyarrow::ToPyArrow;
use arrow_array::builder::{Float64Builder, LargeListBuilder};
use arrow_array::{Array, ArrayRef, Float64Array, LargeListArray, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn write_atomic(batch: &RecordBatch, path: &Path) -> PyResult<()> {
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| PyValueError::new_err("output path must have a UTF-8 filename"))?;
    let temp = path.with_file_name(format!(".{filename}.adam-core-{}.tmp", std::process::id()));
    let result = (|| -> Result<(), String> {
        let file = File::create(&temp)
            .map_err(|err| format!("failed to create {}: {err}", temp.display()))?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None)
            .map_err(|err| format!("failed to create parquet writer: {err}"))?;
        writer
            .write(batch)
            .map_err(|err| format!("failed to write parquet: {err}"))?;
        writer
            .close()
            .map_err(|err| format!("failed to finish parquet: {err}"))?;
        std::fs::rename(&temp, path).map_err(|err| {
            format!(
                "failed to publish {} as {}: {err}",
                temp.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result.map_err(PyRuntimeError::new_err)
}

fn output_path(out_dir: &str, filename: &str) -> PathBuf {
    Path::new(out_dir).join(filename)
}

fn band_map_batch() -> PyResult<RecordBatch> {
    let mut rows: Vec<(String, String, String)> = Vec::new();
    for band in ["u", "g", "r", "i", "z"] {
        rows.push(("W84".into(), band.into(), format!("DECam_{band}")));
    }
    for (band, filter) in [
        ("Y", "DECam_Y"),
        ("y", "DECam_Y"),
        ("VR", "DECam_VR"),
        ("vr", "DECam_VR"),
        ("z", "Mosaic3_z"),
    ] {
        let code = if filter == "Mosaic3_z" { "695" } else { "W84" };
        rows.push((code.into(), band.into(), filter.into()));
    }
    for band in ["g", "r", "i"] {
        rows.push(("I41".into(), band.into(), format!("ZTF_{band}")));
    }
    for band in ["u", "v", "g", "r", "i", "z"] {
        rows.push(("Q55".into(), band.into(), format!("SkyMapper_{band}")));
    }
    for band in ["u", "g", "r", "i", "z", "y"] {
        rows.push(("X05".into(), band.into(), format!("LSST_{band}")));
    }
    rows.push(("X05".into(), "Y".into(), "LSST_y".into()));
    for code in ["T08", "T05", "M22", "W68"] {
        rows.push((code.into(), "c".into(), "ATLAS_c".into()));
        rows.push((code.into(), "o".into(), "ATLAS_o".into()));
    }
    rows.push(("V00".into(), "g".into(), "BASS_g".into()));
    rows.push(("V00".into(), "r".into(), "BASS_r".into()));

    let codes: Vec<&str> = rows.iter().map(|row| row.0.as_str()).collect();
    let bands: Vec<&str> = rows.iter().map(|row| row.1.as_str()).collect();
    let filters: Vec<&str> = rows.iter().map(|row| row.2.as_str()).collect();
    let keys: Vec<String> = rows
        .iter()
        .map(|row| format!("{}|{}", row.0, row.1))
        .collect();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("observatory_code", DataType::LargeUtf8, false),
        Field::new("reported_band", DataType::LargeUtf8, false),
        Field::new("filter_id", DataType::LargeUtf8, false),
        Field::new("key", DataType::LargeUtf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(codes)) as ArrayRef,
            Arc::new(LargeStringArray::from(bands)) as ArrayRef,
            Arc::new(LargeStringArray::from(filters)) as ArrayRef,
            Arc::new(LargeStringArray::from(key_refs)) as ArrayRef,
        ],
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn interp_550(wavelength: &[f64], values: &[f64]) -> f64 {
    let hi = wavelength.partition_point(|value| *value <= 550.0);
    let lo = hi - 1;
    values[lo]
        + (values[hi] - values[lo]) * (550.0 - wavelength[lo]) / (wavelength[hi] - wavelength[lo])
}

fn template_batch() -> PyResult<RecordBatch> {
    let n = 4000usize;
    let step = 800.0 / (n - 1) as f64;
    let wavelength: Vec<f64> = (0..n).map(|index| 300.0 + index as f64 * step).collect();
    let mut c = Vec::with_capacity(n);
    let mut s = Vec::with_capacity(n);
    for &wl in &wavelength {
        let mut c_value = 1.0 + 0.15 * (wl - 550.0) / 550.0;
        c_value *= 1.0 - 0.03 * (-0.5 * ((wl - 700.0) / 50.0).powi(2)).exp();
        if wl < 450.0 {
            c_value *= (-((450.0 - wl) / 100.0).powi(2)).exp();
        }
        c.push(c_value);

        let mut s_value = 1.0 + 0.50 * (wl - 550.0) / 550.0;
        s_value *= 1.0 - 0.10 * (-0.5 * ((wl - 950.0) / 100.0).powi(2)).exp();
        if wl < 500.0 {
            s_value *= (-((500.0 - wl) / 120.0).powi(2)).exp();
        }
        s.push(s_value);
    }
    let c550 = interp_550(&wavelength, &c);
    let s550 = interp_550(&wavelength, &s);
    for value in &mut c {
        *value /= c550;
    }
    for value in &mut s {
        *value /= s550;
    }
    let neo: Vec<f64> = c
        .iter()
        .zip(&s)
        .map(|(&left, &right)| 0.5 * left + 0.5 * right)
        .collect();
    let mba: Vec<f64> = c
        .iter()
        .zip(&s)
        .map(|(&left, &right)| 0.7 * left + 0.3 * right)
        .collect();
    let reflectances = [c, s, neo, mba];

    let mut wl_builder = LargeListBuilder::new(Float64Builder::new());
    let mut refl_builder = LargeListBuilder::new(Float64Builder::new());
    for reflectance in &reflectances {
        wl_builder.values().append_slice(&wavelength);
        wl_builder.append(true);
        refl_builder.values().append_slice(reflectance);
        refl_builder.append(true);
    }
    let citations = [
        "Hand-built C-type reflectance template (optical/NIR). Informed by mean asteroid colors (e.g., Bowell & Lumme 1979; Erasmus et al. 2019). Normalized at 550 nm.",
        "Hand-built S-type reflectance template (optical/NIR). Informed by mean asteroid colors (e.g., Bowell & Lumme 1979; Erasmus et al. 2019). Normalized at 550 nm.",
        "Assumed NEO population mix: 50% C / 50% S (linear reflectance mix).",
        "Assumed main-belt population mix: 70% C / 30% S (linear reflectance mix).",
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("template_id", DataType::LargeUtf8, false),
        Field::new(
            "wavelength_nm",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new(
            "reflectance",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new("weight_C", DataType::Float64, false),
        Field::new("weight_S", DataType::Float64, false),
        Field::new("citation", DataType::LargeUtf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(vec!["C", "S", "NEO", "MBA"])) as ArrayRef,
            Arc::new(wl_builder.finish()) as ArrayRef,
            Arc::new(refl_builder.finish()) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 0.0, 0.5, 0.7])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 0.3])) as ArrayRef,
            Arc::new(LargeStringArray::from(citations.to_vec())) as ArrayRef,
        ],
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn parse_svo_curve(bytes: &[u8]) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let xml = std::str::from_utf8(bytes)
        .map_err(|err| PyValueError::new_err(format!("invalid SVO UTF-8: {err}")))?;
    if !xml.contains("name=\"Wavelength\"") || !xml.contains("name=\"Transmission\"") {
        return Err(PyValueError::new_err(
            "Unexpected SVO schema; required Wavelength and Transmission columns",
        ));
    }
    let mut values = Vec::new();
    let mut remaining = xml;
    while let Some(start) = remaining.find("<TD>") {
        remaining = &remaining[start + 4..];
        let end = remaining
            .find("</TD>")
            .ok_or_else(|| PyValueError::new_err("Malformed SVO TABLEDATA"))?;
        values.push(
            remaining[..end]
                .trim()
                .parse::<f64>()
                .map_err(|err| PyValueError::new_err(format!("invalid SVO value: {err}")))?,
        );
        remaining = &remaining[end + 5..];
    }
    if values.len() < 4 || values.len() % 2 != 0 {
        return Err(PyValueError::new_err("SVO curve has too few points"));
    }
    let mut rows: Vec<(f64, f64, usize)> = values
        .chunks_exact(2)
        .enumerate()
        .map(|(index, row)| (row[0] / 10.0, row[1], index))
        .collect();
    if rows
        .iter()
        .any(|row| !row.0.is_finite() || !row.1.is_finite())
    {
        return Err(PyValueError::new_err("Non-finite values in curve"));
    }
    rows.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.2.cmp(&right.2)));
    rows.dedup_by(|left, right| left.0 == right.0);
    if rows.len() < 2 {
        return Err(PyValueError::new_err(
            "Curve is degenerate after de-duplication",
        ));
    }
    let max = rows
        .iter()
        .map(|row| row.1.clamp(0.0, 1.0))
        .fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return Err(PyValueError::new_err(
            "Curve has zero throughput everywhere",
        ));
    }
    Ok((
        rows.iter().map(|row| row.0).collect(),
        rows.iter().map(|row| row.1.clamp(0.0, 1.0) / max).collect(),
    ))
}

fn curves_batch(
    specs: &[(String, String, String, String)],
    timeout_s: u64,
    payloads: Option<&[Vec<u8>]>,
) -> PyResult<RecordBatch> {
    if specs.is_empty() {
        return Err(PyValueError::new_err("No bandpass specs provided"));
    }
    if let Some(payloads) = payloads {
        if payloads.len() != specs.len() {
            return Err(PyValueError::new_err("payload count must match specs"));
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut wavelengths = Vec::with_capacity(specs.len());
    let mut throughputs = Vec::with_capacity(specs.len());
    let mut sources = Vec::with_capacity(specs.len());
    for (index, (filter_id, _, _, svo_id)) in specs.iter().enumerate() {
        if !seen.insert(filter_id.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Duplicate filter_id in specs: {filter_id}"
            )));
        }
        let url = format!("https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={svo_id}");
        let owned;
        let bytes = if let Some(payloads) = payloads {
            payloads[index].as_slice()
        } else {
            let response = ureq::get(&url)
                .timeout(std::time::Duration::from_secs(timeout_s))
                .call()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let mut reader = response.into_reader();
            let mut buffer = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut buffer)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            owned = buffer;
            owned.as_slice()
        };
        let (wl, throughput) = parse_svo_curve(bytes)?;
        wavelengths.push(wl);
        throughputs.push(throughput);
        sources.push(url);
    }
    let mut wl_builder = LargeListBuilder::new(Float64Builder::new());
    let mut throughput_builder = LargeListBuilder::new(Float64Builder::new());
    for (wl, throughput) in wavelengths.iter().zip(&throughputs) {
        wl_builder.values().append_slice(wl);
        wl_builder.append(true);
        throughput_builder.values().append_slice(throughput);
        throughput_builder.append(true);
    }
    let filters: Vec<&str> = specs.iter().map(|row| row.0.as_str()).collect();
    let instruments: Vec<&str> = specs.iter().map(|row| row.1.as_str()).collect();
    let bands: Vec<&str> = specs.iter().map(|row| row.2.as_str()).collect();
    let source_refs: Vec<&str> = sources.iter().map(String::as_str).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("filter_id", DataType::LargeUtf8, false),
        Field::new("instrument", DataType::LargeUtf8, false),
        Field::new("band", DataType::LargeUtf8, false),
        Field::new(
            "wavelength_nm",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new(
            "throughput",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new("source", DataType::LargeUtf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(filters)) as ArrayRef,
            Arc::new(LargeStringArray::from(instruments)) as ArrayRef,
            Arc::new(LargeStringArray::from(bands)) as ArrayRef,
            Arc::new(wl_builder.finish()) as ArrayRef,
            Arc::new(throughput_builder.finish()) as ArrayRef,
            Arc::new(LargeStringArray::from(source_refs)) as ArrayRef,
        ],
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn fits_card_value<'a>(header: &'a [u8], key: &str) -> Option<&'a str> {
    for card in header.chunks_exact(80) {
        if std::str::from_utf8(&card[..8]).ok()?.trim() == key {
            let text = std::str::from_utf8(&card[10..]).ok()?;
            return Some(text.split('/').next()?.trim().trim_matches('\'').trim());
        }
    }
    None
}

fn fits_field_width(form: &str) -> Option<usize> {
    let form = form.trim();
    let split = form
        .find(|value: char| !value.is_ascii_digit())
        .unwrap_or(0);
    let count = if split == 0 {
        1
    } else {
        form[..split].parse().ok()?
    };
    let kind = form.as_bytes().get(split).copied()? as char;
    let width = match kind {
        'D' | 'K' => 8,
        'E' | 'J' => 4,
        'I' => 2,
        'B' | 'L' | 'A' => 1,
        _ => return None,
    };
    Some(count * width)
}

fn parse_solar_fits(bytes: &[u8]) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let marker = b"XTENSION= 'BINTABLE'";
    let position = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or_else(|| PyValueError::new_err("solar FITS is missing a BINTABLE extension"))?;
    let header_start = position / 2880 * 2880;
    let mut card_end = header_start;
    loop {
        if card_end + 80 > bytes.len() {
            return Err(PyValueError::new_err("truncated FITS header"));
        }
        if &bytes[card_end..card_end + 8] == b"END     " {
            card_end += 80;
            break;
        }
        card_end += 80;
    }
    let data_start = card_end.div_ceil(2880) * 2880;
    let header = &bytes[header_start..card_end];
    let row_width: usize = fits_card_value(header, "NAXIS1")
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| PyValueError::new_err("solar FITS missing NAXIS1"))?;
    let rows: usize = fits_card_value(header, "NAXIS2")
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| PyValueError::new_err("solar FITS missing NAXIS2"))?;
    let fields: usize = fits_card_value(header, "TFIELDS")
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| PyValueError::new_err("solar FITS missing TFIELDS"))?;
    let mut offset = 0usize;
    let mut wavelength_offset = None;
    let mut flux_offset = None;
    for field in 1..=fields {
        let name = fits_card_value(header, &format!("TTYPE{field}"))
            .ok_or_else(|| PyValueError::new_err("solar FITS missing TTYPE"))?;
        let form = fits_card_value(header, &format!("TFORM{field}"))
            .ok_or_else(|| PyValueError::new_err("solar FITS missing TFORM"))?;
        if name == "WAVELENGTH" {
            wavelength_offset = Some(offset);
        }
        if name == "FLUX" {
            flux_offset = Some(offset);
        }
        offset += fits_field_width(form)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported FITS TFORM: {form}")))?;
    }
    if offset != row_width {
        return Err(PyValueError::new_err(
            "solar FITS row width does not match fields",
        ));
    }
    let wavelength_offset =
        wavelength_offset.ok_or_else(|| PyValueError::new_err("solar FITS missing WAVELENGTH"))?;
    let flux_offset =
        flux_offset.ok_or_else(|| PyValueError::new_err("solar FITS missing FLUX"))?;
    if data_start + rows * row_width > bytes.len() {
        return Err(PyValueError::new_err("truncated solar FITS table"));
    }
    let mut wavelength = Vec::new();
    let mut flux = Vec::new();
    for row in 0..rows {
        let base = data_start + row * row_width;
        let wl = f64::from_be_bytes(
            bytes[base + wavelength_offset..base + wavelength_offset + 8]
                .try_into()
                .unwrap(),
        );
        let value = f64::from_be_bytes(
            bytes[base + flux_offset..base + flux_offset + 8]
                .try_into()
                .unwrap(),
        );
        let wl_nm = wl / 10.0;
        if (300.0..=1200.0).contains(&wl_nm) {
            wavelength.push(wl_nm);
            flux.push(value);
        }
    }
    let max = flux.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for value in &mut flux {
        *value /= max;
    }
    Ok((wavelength, flux))
}

fn solar_batch(payload: Option<&[u8]>, timeout_s: u64) -> PyResult<RecordBatch> {
    let owned;
    let bytes = if let Some(payload) = payload {
        payload
    } else {
        let response = ureq::get(
            "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits",
        )
        .timeout(std::time::Duration::from_secs(timeout_s))
        .call()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let mut reader = response.into_reader();
        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buffer)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        owned = buffer;
        owned.as_slice()
    };
    let (wavelength, flux) = parse_solar_fits(bytes)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("wavelength_nm", DataType::Float64, false),
        Field::new("flux", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(wavelength)) as ArrayRef,
            Arc::new(Float64Array::from(flux)) as ArrayRef,
        ],
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn read_batches(path: &Path) -> PyResult<Vec<RecordBatch>> {
    let file = File::open(path).map_err(|err| {
        PyRuntimeError::new_err(format!("failed to open {}: {err}", path.display()))
    })?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(|builder| builder.build())
        .map_err(|err| {
            PyRuntimeError::new_err(format!("failed to read {}: {err}", path.display()))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| PyRuntimeError::new_err(format!("failed to read {}: {err}", path.display())))
}

fn string_values(batches: &[RecordBatch], name: &str) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for batch in batches {
        let array = batch
            .column_by_name(name)
            .and_then(|value| value.as_any().downcast_ref::<LargeStringArray>())
            .ok_or_else(|| PyValueError::new_err(format!("column {name} must be large_utf8")))?;
        out.extend((0..array.len()).map(|row| array.value(row).to_string()));
    }
    Ok(out)
}

fn f64_values(batches: &[RecordBatch], name: &str) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    for batch in batches {
        let array = batch
            .column_by_name(name)
            .and_then(|value| value.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| PyValueError::new_err(format!("column {name} must be float64")))?;
        out.extend((0..array.len()).map(|row| array.value(row)));
    }
    Ok(out)
}

fn list_f64_values(batches: &[RecordBatch], name: &str) -> PyResult<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for batch in batches {
        let array = batch
            .column_by_name(name)
            .and_then(|value| value.as_any().downcast_ref::<LargeListArray>())
            .ok_or_else(|| {
                PyValueError::new_err(format!("column {name} must be large_list<float64>"))
            })?;
        for row in 0..array.len() {
            let values = array.value(row);
            let values = values
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| PyValueError::new_err("list values must be float64"))?;
            out.push((0..values.len()).map(|index| values.value(index)).collect());
        }
    }
    Ok(out)
}

fn interp_zero(x_new: &[f64], x: &[f64], y: &[f64]) -> Vec<f64> {
    x_new
        .iter()
        .map(|&value| {
            if x.is_empty() || value < x[0] || value > x[x.len() - 1] {
                return 0.0;
            }
            let hi = x.partition_point(|item| *item <= value);
            if hi == 0 {
                return y[0];
            }
            if hi == x.len() {
                return y[y.len() - 1];
            }
            let lo = hi - 1;
            y[lo] + (y[hi] - y[lo]) * (value - x[lo]) / (x[hi] - x[lo])
        })
        .collect()
}

fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    (1..x.len())
        .map(|index| (x[index] - x[index - 1]) * (y[index] + y[index - 1]) / 2.0)
        .sum()
}

fn integral_batch(out_dir: &str) -> PyResult<RecordBatch> {
    let dir = Path::new(out_dir);
    let curve_batches = read_batches(&dir.join("bandpass_curves.parquet"))?;
    let filter_ids = string_values(&curve_batches, "filter_id")?;
    let filter_wavelength = list_f64_values(&curve_batches, "wavelength_nm")?;
    let throughputs = list_f64_values(&curve_batches, "throughput")?;
    let template_batches = read_batches(&dir.join("asteroid_templates.parquet"))?;
    let template_ids = string_values(&template_batches, "template_id")?;
    let template_wavelength = list_f64_values(&template_batches, "wavelength_nm")?;
    let reflectances = list_f64_values(&template_batches, "reflectance")?;
    let solar_batches = read_batches(&dir.join("solar_spectrum.parquet"))?;
    let mut solar_wavelength = f64_values(&solar_batches, "wavelength_nm")?;
    let mut solar_flux = f64_values(&solar_batches, "flux")?;
    if solar_wavelength.len() < 2
        || solar_wavelength
            .windows(2)
            .any(|window| window[1] <= window[0])
    {
        let mut order: Vec<usize> = (0..solar_wavelength.len()).collect();
        order.sort_by(|&left, &right| solar_wavelength[left].total_cmp(&solar_wavelength[right]));
        solar_wavelength = order.iter().map(|&row| solar_wavelength[row]).collect();
        solar_flux = order.iter().map(|&row| solar_flux[row]).collect();
    }
    let mut out_templates = Vec::new();
    let mut out_filters = Vec::new();
    let mut integrals = Vec::new();
    for (template_index, template_id) in template_ids.iter().enumerate() {
        let template_wl = &template_wavelength[template_index];
        let reflectance = &reflectances[template_index];
        for (filter_index, filter_id) in filter_ids.iter().enumerate() {
            let filter_wl = &filter_wavelength[filter_index];
            let throughput = &throughputs[filter_index];
            let wl_min = solar_wavelength[0].max(template_wl[0]).max(filter_wl[0]);
            let wl_max = solar_wavelength[solar_wavelength.len() - 1]
                .min(template_wl[template_wl.len() - 1])
                .min(filter_wl[filter_wl.len() - 1]);
            let value = if wl_max <= wl_min {
                f64::NAN
            } else {
                let rows: Vec<usize> = solar_wavelength
                    .iter()
                    .enumerate()
                    .filter_map(|(row, &wl)| (wl >= wl_min && wl <= wl_max).then_some(row))
                    .collect();
                let wl: Vec<f64> = rows.iter().map(|&row| solar_wavelength[row]).collect();
                let sun: Vec<f64> = rows.iter().map(|&row| solar_flux[row]).collect();
                let t = interp_zero(&wl, filter_wl, throughput);
                let r = interp_zero(&wl, template_wl, reflectance);
                let y: Vec<f64> = sun
                    .iter()
                    .zip(&r)
                    .zip(&t)
                    .zip(&wl)
                    .map(|(((&sun, &reflectance), &throughput), &wavelength)| {
                        sun * reflectance * throughput * wavelength
                    })
                    .collect();
                trapezoid(&y, &wl)
            };
            out_templates.push(template_id.as_str());
            out_filters.push(filter_id.as_str());
            integrals.push(value);
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("template_id", DataType::LargeUtf8, false),
        Field::new("filter_id", DataType::LargeUtf8, false),
        Field::new("integral_photon", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(out_templates)) as ArrayRef,
            Arc::new(LargeStringArray::from(out_filters)) as ArrayRef,
            Arc::new(Float64Array::from(integrals)) as ArrayRef,
        ],
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (out_dir, timeout_s=120, payload=None))]
fn build_solar_spectrum_arrow<'py>(
    py: Python<'py>,
    out_dir: &str,
    timeout_s: u64,
    payload: Option<Vec<u8>>,
) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| solar_batch(payload.as_deref(), timeout_s))?;
    write_atomic(&batch, &output_path(out_dir, "solar_spectrum.parquet"))?;
    batch
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (out_dir, specs, timeout_s, payloads=None))]
fn build_bandpass_curves_arrow<'py>(
    py: Python<'py>,
    out_dir: &str,
    specs: Vec<(String, String, String, String)>,
    timeout_s: u64,
    payloads: Option<Vec<Vec<u8>>>,
) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| curves_batch(&specs, timeout_s, payloads.as_deref()))?;
    write_atomic(&batch, &output_path(out_dir, "bandpass_curves.parquet"))?;
    batch
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn build_observatory_band_map_arrow<'py>(py: Python<'py>, out_dir: &str) -> PyResult<PyObject> {
    let batch = py.allow_threads(band_map_batch)?;
    write_atomic(
        &batch,
        &output_path(out_dir, "observatory_band_map.parquet"),
    )?;
    batch
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn build_template_bandpass_integrals_arrow<'py>(
    py: Python<'py>,
    out_dir: &str,
) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| integral_batch(out_dir))?;
    write_atomic(
        &batch,
        &output_path(out_dir, "template_bandpass_integrals.parquet"),
    )?;
    batch
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn build_asteroid_templates_arrow<'py>(py: Python<'py>, out_dir: &str) -> PyResult<PyObject> {
    let batch = py.allow_threads(template_batch)?;
    write_atomic(&batch, &output_path(out_dir, "asteroid_templates.parquet"))?;
    batch
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (out_dir, svo_payload, solar_payload, reps, trials, warmup_reps=1))]
fn benchmark_bandpass_vendor_products(
    out_dir: &str,
    svo_payload: Vec<u8>,
    solar_payload: Vec<u8>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let specs = vec![(
        "V".to_string(),
        "Bessell".to_string(),
        "V".to_string(),
        "Generic/Bessell.V".to_string(),
    )];
    let payloads = vec![svo_payload];
    bench_result(reps, trials, warmup_reps, || {
        std::hint::black_box(curves_batch(&specs, 60, Some(&payloads))?);
        std::hint::black_box(solar_batch(Some(&solar_payload), 120)?);
        std::hint::black_box(band_map_batch()?);
        std::hint::black_box(template_batch()?);
        std::hint::black_box(integral_batch(out_dir)?);
        Ok(())
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_bandpass_curves_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(
        benchmark_bandpass_vendor_products,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_solar_spectrum_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(build_observatory_band_map_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(build_asteroid_templates_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_template_bandpass_integrals_arrow,
        module
    )?)?;
    Ok(())
}
