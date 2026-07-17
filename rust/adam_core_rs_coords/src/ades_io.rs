//! ADES writer/parser (bead personal-cmy.20 slices B/C).
//!
//! Ports `adam_core.observations.ades.ADES_to_string` and the observation
//! parsing half of `ADES_string_to_tables` to Rust, replicating the legacy
//! Python/pandas/astropy behavior byte-for-byte:
//!
//! * observatory blocks in ASCII-sorted `stn` order; per-block stable sort by
//!   (provID, permID, trkSub, obsTime.days, obsTime.nanos) with nulls last;
//! * per-block `dropna(how="all", axis=1)` column pruning (NaN and null are
//!   both "missing", matching pandas);
//! * `obsTime` rendered like `astropy.time.Time(mjd, format="mjd",
//!   precision=p).utc.isot + "Z"`: the float64 MJD is split with
//!   round-half-even exactly like astropy's `day_frac`, then formatted with
//!   ERFA `d2dtf` (via the pure-Rust `erfars` port, leap-second aware);
//! * precision-formatted columns follow Python `f"{v:.Nf}"` semantics --
//!   including the legacy quirk that missing values in those columns render
//!   as the string `nan`;
//! * remaining float columns follow pandas `to_csv(float_format="%.16f",
//!   na_rep="")`; strings render empty for null; fields containing the `|`
//!   separator, quotes, or newlines get pandas QUOTE_MINIMAL quoting;
//! * the parser mirrors `_data_dict_to_table`: `|`-split blocks headed by
//!   lines containing permID/provID/trkSub, empty/whitespace cells -> null,
//!   numeric/string column coercion with stripping (remarks unstripped),
//!   trailing-character removal from obsTime (legacy `t[:-1]`), ERFA `dtf2d`
//!   plus the exact `Timestamp.from_astropy` divmod/round arithmetic, and
//!   `rmsRA` -> `rmsRACosDec` renaming; unknown columns are reported back for
//!   the caller to log.

use crate::observations::{AdesObservationBatch, TimeColumn};
use crate::types::{SchemaError, SchemaResult, TimeScale};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

const NANOS_PER_DAY_F64: f64 = 86_400e9;
const MJD_JD_OFFSET: f64 = 2_400_000.5;

/// Column order of the flattened legacy dataframe after the `obsTime` string
/// column is inserted and the `obsTime.days`/`obsTime.nanos` columns dropped.
const WRITER_COLUMNS: [&str; 22] = [
    "permID",
    "provID",
    "trkSub",
    "obsSubID",
    "obsTime",
    "rmsTime",
    "ra",
    "dec",
    "rmsRACosDec",
    "rmsDec",
    "rmsCorr",
    "mag",
    "rmsMag",
    "band",
    "stn",
    "mode",
    "astCat",
    "photCat",
    "logSNR",
    "seeing",
    "exp",
    "remarks",
];

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

// --- time formatting (astropy-equivalent) --------------------------------------

/// `astropy.time.Time(mjd, format="mjd", precision=p).utc.isot`: astropy's
/// `day_frac(mjd, 0.0)` reduces to `day = round_half_even(mjd)`,
/// `frac = mjd - day`, then ERFA `d2dtf("UTC", p, day + 2400000.5, frac)`.
fn isot_utc_from_mjd(mjd: f64, precision: i32) -> SchemaResult<String> {
    let day = mjd.round_ties_even();
    let frac = mjd - day;
    let ((iy, im, id, ih, imin, isec, ifrac), _warning) =
        erfars::timescales::D2dtf(true, precision, day + MJD_JD_OFFSET, frac)
            .map_err(|code| invalid(format!("eraD2dtf failed with ERFA status {code}")))?;
    let mut out = format!("{iy:04}-{im:02}-{id:02}T{ih:02}:{imin:02}:{isec:02}");
    if precision > 0 {
        out.push('.');
        out.push_str(&format!("{ifrac:0width$}", width = precision as usize));
    }
    Ok(out)
}

// --- float formatting (Python-equivalent) ---------------------------------------

/// Python `f"{value:.{precision}f}"` (used by the legacy precision-formatting
/// loop): NaN renders as `nan`, infinities as `inf`/`-inf`.
fn python_fixed(value: f64, precision: usize) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        }
    } else {
        format!("{value:.precision$}")
    }
}

/// pandas QUOTE_MINIMAL for `sep='|'`: quote when the field contains the
/// separator, the quote character, or a newline; double inner quotes.
fn csv_quote(field: &str) -> String {
    if field.contains('|') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn is_na(value: &Option<f64>) -> bool {
    match value {
        None => true,
        Some(v) => v.is_nan(),
    }
}

fn cmp_opt_str(a: &Option<String>, b: &Option<String>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => x.cmp(y),
    }
}

enum RenderedColumn {
    /// Pre-rendered strings (obsTime) -- always present.
    Rendered(Vec<String>),
    /// Optional strings: null -> "".
    Strings(Vec<Option<String>>),
    /// Required strings.
    ReqStrings(Vec<String>),
    /// Optional floats (missing = null or NaN).
    Floats(Vec<Option<f64>>),
    /// Required floats.
    ReqFloats(Vec<f64>),
}

impl RenderedColumn {
    fn has_any_value(&self) -> bool {
        match self {
            Self::Rendered(_) | Self::ReqStrings(_) | Self::ReqFloats(_) => true,
            Self::Strings(values) => values.iter().any(|value| value.is_some()),
            Self::Floats(values) => values.iter().any(|value| !is_na(value)),
        }
    }

    fn render_cell(&self, row: usize, precision: Option<usize>) -> String {
        match self {
            Self::Rendered(values) => values[row].clone(),
            Self::Strings(values) => values[row].clone().unwrap_or_default(),
            Self::ReqStrings(values) => values[row].clone(),
            Self::ReqFloats(values) => match precision {
                Some(precision) => python_fixed(values[row], precision),
                None => python_fixed(values[row], 16),
            },
            Self::Floats(values) => match precision {
                // Legacy quirk: the precision-formatting loop runs
                // f"{v:.Nf}" on every element, so missing values (NaN after
                // pandas conversion) render as the string "nan".
                Some(precision) => match &values[row] {
                    value if is_na(value) => "nan".to_string(),
                    Some(value) => python_fixed(*value, precision),
                    None => unreachable!(),
                },
                // Columns left as floats go through pandas to_csv
                // float_format="%.16f" with na_rep="".
                None => match &values[row] {
                    value if is_na(value) => String::new(),
                    Some(value) => python_fixed(*value, 16),
                    None => unreachable!(),
                },
            },
        }
    }
}

/// Render `ADES_to_string` byte-identically to the legacy Python writer.
///
/// `contexts` maps observatory codes to their pre-rendered `ObsContext`
/// blocks (`ObsContext.to_string()` output, newline-terminated) -- the
/// ObsContext dataclasses themselves stay Python-side.
pub fn ades_to_string(
    observations: &AdesObservationBatch,
    contexts: &HashMap<String, String>,
    seconds_precision: i32,
    columns_precision: &HashMap<String, i32>,
) -> SchemaResult<String> {
    observations.validate()?;
    if !observations.is_empty() && observations.obs_time.scale != TimeScale::Utc {
        return Err(invalid(format!(
            "ades_to_string requires obsTime in utc, got {}",
            observations.obs_time.scale.as_str()
        )));
    }
    if let Some(validity) = &observations.obs_time.validity {
        if validity.iter().any(|valid| !valid) {
            return Err(invalid("obsTime must not contain nulls".to_string()));
        }
    }

    let mut out = String::from("# version=2022\n");

    let unique_observatories: BTreeSet<&str> =
        observations.stn.iter().map(|stn| stn.as_str()).collect();

    for obs in unique_observatories {
        let context = contexts
            .get(obs)
            .ok_or_else(|| invalid(format!("Observatory {obs} not found in obs_contexts")))?;

        // Filter (original order), then stable-sort by the legacy keys with
        // nulls last -- matching pyarrow Table.sort_by defaults.
        let mut indices: Vec<usize> = (0..observations.len())
            .filter(|&row| observations.stn[row] == obs)
            .collect();
        indices.sort_by(|&a, &b| {
            cmp_opt_str(&observations.prov_id[a], &observations.prov_id[b])
                .then_with(|| cmp_opt_str(&observations.perm_id[a], &observations.perm_id[b]))
                .then_with(|| cmp_opt_str(&observations.trk_sub[a], &observations.trk_sub[b]))
                .then_with(|| observations.obs_time.days[a].cmp(&observations.obs_time.days[b]))
                .then_with(|| observations.obs_time.nanos[a].cmp(&observations.obs_time.nanos[b]))
        });

        let gather_str = |values: &[Option<String>]| -> Vec<Option<String>> {
            indices.iter().map(|&row| values[row].clone()).collect()
        };
        let gather_req_str = |values: &[String]| -> Vec<String> {
            indices.iter().map(|&row| values[row].clone()).collect()
        };
        let gather_f64 = |values: &[Option<f64>]| -> Vec<Option<f64>> {
            indices.iter().map(|&row| values[row]).collect()
        };
        let gather_req_f64 =
            |values: &[f64]| -> Vec<f64> { indices.iter().map(|&row| values[row]).collect() };

        let perm_id = gather_str(&observations.perm_id);
        let prov_id = gather_str(&observations.prov_id);
        let trk_sub = gather_str(&observations.trk_sub);

        let id_present = perm_id.iter().any(|value| value.is_some())
            || prov_id.iter().any(|value| value.is_some())
            || trk_sub.iter().any(|value| value.is_some());
        if !id_present {
            return Err(invalid(
                "At least one of permID, provID, or trkSub should\nbe present in observations."
                    .to_string(),
            ));
        }

        let obs_time: Vec<String> = indices
            .iter()
            .map(|&row| {
                let mjd = observations.obs_time.days[row] as f64
                    + observations.obs_time.nanos[row] as f64 / NANOS_PER_DAY_F64;
                isot_utc_from_mjd(mjd, seconds_precision).map(|isot| format!("{isot}Z"))
            })
            .collect::<SchemaResult<_>>()?;

        let columns: Vec<(&str, RenderedColumn)> = vec![
            ("permID", RenderedColumn::Strings(perm_id)),
            ("provID", RenderedColumn::Strings(prov_id)),
            ("trkSub", RenderedColumn::Strings(trk_sub)),
            (
                "obsSubID",
                RenderedColumn::Strings(gather_str(&observations.obs_sub_id)),
            ),
            ("obsTime", RenderedColumn::Rendered(obs_time)),
            (
                "rmsTime",
                RenderedColumn::Floats(gather_f64(&observations.rms_time)),
            ),
            (
                "ra",
                RenderedColumn::ReqFloats(gather_req_f64(&observations.ra)),
            ),
            (
                "dec",
                RenderedColumn::ReqFloats(gather_req_f64(&observations.dec)),
            ),
            (
                "rmsRACosDec",
                RenderedColumn::Floats(gather_f64(&observations.rms_ra_cos_dec)),
            ),
            (
                "rmsDec",
                RenderedColumn::Floats(gather_f64(&observations.rms_dec)),
            ),
            (
                "rmsCorr",
                RenderedColumn::Floats(gather_f64(&observations.rms_corr)),
            ),
            ("mag", RenderedColumn::Floats(gather_f64(&observations.mag))),
            (
                "rmsMag",
                RenderedColumn::Floats(gather_f64(&observations.rms_mag)),
            ),
            (
                "band",
                RenderedColumn::Strings(gather_str(&observations.band)),
            ),
            (
                "stn",
                RenderedColumn::ReqStrings(gather_req_str(&observations.stn)),
            ),
            (
                "mode",
                RenderedColumn::ReqStrings(gather_req_str(&observations.mode)),
            ),
            (
                "astCat",
                RenderedColumn::ReqStrings(gather_req_str(&observations.ast_cat)),
            ),
            (
                "photCat",
                RenderedColumn::Strings(gather_str(&observations.phot_cat)),
            ),
            (
                "logSNR",
                RenderedColumn::Floats(gather_f64(&observations.log_snr)),
            ),
            (
                "seeing",
                RenderedColumn::Floats(gather_f64(&observations.seeing)),
            ),
            ("exp", RenderedColumn::Floats(gather_f64(&observations.exp))),
            (
                "remarks",
                RenderedColumn::Strings(gather_str(&observations.remarks)),
            ),
        ];
        debug_assert_eq!(columns.len(), WRITER_COLUMNS.len());

        // dropna(how="all", axis=1)
        let kept: Vec<&(&str, RenderedColumn)> = columns
            .iter()
            .filter(|(_, column)| column.has_any_value())
            .collect();

        out.push_str(context);

        // Header row with the rmsRACosDec -> rmsRA rename.
        let header: Vec<String> = kept
            .iter()
            .map(|(name, _)| {
                let name = if *name == "rmsRACosDec" {
                    "rmsRA"
                } else {
                    name
                };
                csv_quote(name)
            })
            .collect();
        out.push_str(&header.join("|"));
        out.push('\n');

        for row in 0..indices.len() {
            let cells: Vec<String> = kept
                .iter()
                .map(|(name, column)| {
                    let precision = columns_precision
                        .get(*name)
                        .map(|precision| *precision as usize);
                    csv_quote(&column.render_cell(row, precision))
                })
                .collect();
            out.push_str(&cells.join("|"));
            out.push('\n');
        }
    }

    Ok(out)
}

// --- parser ----------------------------------------------------------------------

const KNOWN_COLUMNS: [&str; 22] = WRITER_COLUMNS;

/// `Timestamp.from_astropy` arithmetic on ERFA `dtf2d` output: exact divmod on
/// `jd1 - 2400000.5`, half-even rounding of the day fraction to nanoseconds,
/// and the 86400e9 carry fix-up.
fn timestamp_from_isot_utc(isot: &str) -> SchemaResult<(i64, i64)> {
    let bad = |what: &str| invalid(format!("invalid ISO time {isot:?}: {what}"));
    let (date, time) = isot.split_once('T').ok_or_else(|| bad("missing 'T'"))?;
    let mut date_parts = date.split('-');
    let year: i32 = date_parts
        .next()
        .ok_or_else(|| bad("missing year"))?
        .parse()
        .map_err(|_| bad("bad year"))?;
    let month: i32 = date_parts
        .next()
        .ok_or_else(|| bad("missing month"))?
        .parse()
        .map_err(|_| bad("bad month"))?;
    let day: i32 = date_parts
        .next()
        .ok_or_else(|| bad("missing day"))?
        .parse()
        .map_err(|_| bad("bad day"))?;
    let mut time_parts = time.split(':');
    let hour: i32 = time_parts
        .next()
        .ok_or_else(|| bad("missing hour"))?
        .parse()
        .map_err(|_| bad("bad hour"))?;
    let minute: i32 = time_parts
        .next()
        .ok_or_else(|| bad("missing minute"))?
        .parse()
        .map_err(|_| bad("bad minute"))?;
    let seconds: f64 = time_parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| bad("bad seconds"))?;

    let ((jd1, jd2), _warning) =
        erfars::timescales::Dtf2d(true, year, month, day, hour, minute, seconds)
            .map_err(|code| invalid(format!("eraDtf2d failed with ERFA status {code}")))?;

    let value = jd1 - MJD_JD_OFFSET;
    let mut days = value.floor();
    let mut remainder = value - days;
    remainder += jd2;
    if remainder < 0.0 {
        remainder += 1.0;
        days -= 1.0;
    }
    if remainder >= 1.0 {
        remainder -= 1.0;
        days += 1.0;
    }
    let mut nanos = (remainder * NANOS_PER_DAY_F64).round_ties_even();
    if nanos == NANOS_PER_DAY_F64 {
        days += 1.0;
        nanos = 0.0;
    }
    Ok((days as i64, nanos as i64))
}

#[derive(Debug, Default)]
struct BlockData {
    headers: Vec<String>,
    rows: Vec<Vec<Option<String>>>,
}

impl BlockData {
    fn column(&self, name: &str) -> Option<Vec<Option<String>>> {
        let index = self.headers.iter().position(|header| header == name)?;
        Some(
            self.rows
                .iter()
                .map(|row| row.get(index).cloned().flatten())
                .collect(),
        )
    }
}

/// Parse the observation blocks of an ADES string into an
/// `AdesObservationBatch` plus the list of unknown column names encountered
/// (for the caller to log, matching the legacy warning).
pub fn ades_string_to_observations(
    ades_string: &str,
) -> SchemaResult<(AdesObservationBatch, Vec<String>)> {
    let lines: Vec<&str> = ades_string
        .split('\n')
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();

    let mut blocks: Vec<BlockData> = Vec::new();
    let mut current: Option<BlockData> = None;
    for line in &lines {
        if line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        if line.contains("permID") || line.contains("provID") || line.contains("trkSub") {
            if let Some(block) = current.take() {
                blocks.push(block);
            }
            current = Some(BlockData {
                headers: line
                    .split('|')
                    .map(|header| header.trim().to_string())
                    .collect(),
                rows: Vec::new(),
            });
            continue;
        }
        if let Some(block) = &mut current {
            let values: Vec<&str> = line.split('|').collect();
            // zip semantics: pair headers with values up to the shorter length,
            // then normalize empty/whitespace-only cells to null.
            let row: Vec<Option<String>> = block
                .headers
                .iter()
                .zip(values.iter())
                .map(|(_, value)| {
                    if value.is_empty() || value.chars().all(char::is_whitespace) {
                        None
                    } else {
                        Some((*value).to_string())
                    }
                })
                .collect();
            block.rows.push(row);
        }
    }
    if let Some(block) = current.take() {
        blocks.push(block);
    }

    let mut unknown_columns: Vec<String> = Vec::new();
    let mut parts: Vec<AdesObservationBatch> = Vec::new();
    for block in &blocks {
        if block.rows.is_empty() && block.headers.is_empty() {
            continue;
        }
        for header in &block.headers {
            if !KNOWN_COLUMNS.contains(&header.as_str())
                && header != "rmsRA"
                && !unknown_columns.contains(header)
            {
                unknown_columns.push(header.clone());
            }
        }
        parts.push(block_to_batch(block)?);
    }

    let merged = concatenate(parts);
    Ok((merged, unknown_columns))
}

fn parse_float_column(
    block: &BlockData,
    name: &str,
    rows: usize,
) -> SchemaResult<Vec<Option<f64>>> {
    match block.column(name) {
        None => Ok(vec![None; rows]),
        Some(values) => values
            .into_iter()
            .map(|value| {
                value
                    .map(|value| {
                        value.trim().parse::<f64>().map_err(|_| {
                            invalid(format!("could not parse {name} value {value:?} as float"))
                        })
                    })
                    .transpose()
            })
            .collect(),
    }
}

fn parse_string_column(
    block: &BlockData,
    name: &str,
    rows: usize,
    strip: bool,
) -> Vec<Option<String>> {
    match block.column(name) {
        None => vec![None; rows],
        Some(values) => values
            .into_iter()
            .map(|value| {
                value.map(|value| {
                    if strip {
                        value.trim().to_string()
                    } else {
                        value
                    }
                })
            })
            .collect(),
    }
}

fn block_to_batch(block: &BlockData) -> SchemaResult<AdesObservationBatch> {
    let rows = block.rows.len();

    let obs_time = match block.column("obsTime") {
        None => TimeColumn::new(TimeScale::Utc, vec![0; rows], vec![0; rows]),
        Some(values) => {
            let mut days = Vec::with_capacity(rows);
            let mut nanos = Vec::with_capacity(rows);
            for value in values {
                let value = value
                    .ok_or_else(|| invalid("obsTime must not contain empty values".to_string()))?;
                // Legacy strips the last character unconditionally (`t[:-1]`).
                let mut chars = value.chars();
                chars.next_back();
                let (day, nano) = timestamp_from_isot_utc(chars.as_str())?;
                days.push(day);
                nanos.push(nano);
            }
            TimeColumn::new(TimeScale::Utc, days, nanos)
        }
    };

    // ADES uses rmsRA on the wire; the table column is rmsRACosDec.
    let rms_ra_cos_dec = parse_float_column(block, "rmsRA", rows)?;

    let req_string = |name: &str| -> SchemaResult<Vec<String>> {
        parse_string_column(block, name, rows, true)
            .into_iter()
            .map(|value| {
                value.ok_or_else(|| invalid(format!("column {name} must not contain empty values")))
            })
            .collect()
    };

    let out = AdesObservationBatch {
        perm_id: parse_string_column(block, "permID", rows, true),
        prov_id: parse_string_column(block, "provID", rows, true),
        trk_sub: parse_string_column(block, "trkSub", rows, true),
        obs_sub_id: parse_string_column(block, "obsSubID", rows, true),
        obs_time,
        rms_time: parse_float_column(block, "rmsTime", rows)?,
        ra: parse_float_column(block, "ra", rows)?
            .into_iter()
            .map(|value| value.ok_or_else(|| invalid("ra must not be empty".to_string())))
            .collect::<SchemaResult<_>>()?,
        dec: parse_float_column(block, "dec", rows)?
            .into_iter()
            .map(|value| value.ok_or_else(|| invalid("dec must not be empty".to_string())))
            .collect::<SchemaResult<_>>()?,
        rms_ra_cos_dec,
        rms_dec: parse_float_column(block, "rmsDec", rows)?,
        rms_corr: parse_float_column(block, "rmsCorr", rows)?,
        mag: parse_float_column(block, "mag", rows)?,
        rms_mag: parse_float_column(block, "rmsMag", rows)?,
        band: parse_string_column(block, "band", rows, true),
        stn: req_string("stn")?,
        mode: req_string("mode")?,
        ast_cat: req_string("astCat")?,
        phot_cat: parse_string_column(block, "photCat", rows, true),
        log_snr: parse_float_column(block, "logSNR", rows)?,
        seeing: parse_float_column(block, "seeing", rows)?,
        exp: parse_float_column(block, "exp", rows)?,
        remarks: parse_string_column(block, "remarks", rows, false),
    };
    out.validate()?;
    Ok(out)
}

fn concatenate(parts: Vec<AdesObservationBatch>) -> AdesObservationBatch {
    let mut merged = AdesObservationBatch {
        perm_id: vec![],
        prov_id: vec![],
        trk_sub: vec![],
        obs_sub_id: vec![],
        obs_time: TimeColumn::new(TimeScale::Utc, vec![], vec![]),
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
    for mut part in parts {
        merged.perm_id.append(&mut part.perm_id);
        merged.prov_id.append(&mut part.prov_id);
        merged.trk_sub.append(&mut part.trk_sub);
        merged.obs_sub_id.append(&mut part.obs_sub_id);
        merged.obs_time.days.append(&mut part.obs_time.days);
        merged.obs_time.nanos.append(&mut part.obs_time.nanos);
        merged.rms_time.append(&mut part.rms_time);
        merged.ra.append(&mut part.ra);
        merged.dec.append(&mut part.dec);
        merged.rms_ra_cos_dec.append(&mut part.rms_ra_cos_dec);
        merged.rms_dec.append(&mut part.rms_dec);
        merged.rms_corr.append(&mut part.rms_corr);
        merged.mag.append(&mut part.mag);
        merged.rms_mag.append(&mut part.rms_mag);
        merged.band.append(&mut part.band);
        merged.stn.append(&mut part.stn);
        merged.mode.append(&mut part.mode);
        merged.ast_cat.append(&mut part.ast_cat);
        merged.phot_cat.append(&mut part.phot_cat);
        merged.log_snr.append(&mut part.log_snr);
        merged.seeing.append(&mut part.seeing);
        merged.exp.append(&mut part.exp);
        merged.remarks.append(&mut part.remarks);
    }
    merged
}

// --- ObsContext rendering + metadata-section parsing (bead personal-cmy.26) ----

/// Legacy `ObsContext.to_string`: renders the `dataclasses.asdict` JSON
/// payload of an ObsContext (field order preserved) into the `# section` /
/// `! key value` header block. Floats render with shortest round-trip
/// formatting, matching Python `str()`.
pub fn obs_context_to_string(context_json: &str) -> SchemaResult<String> {
    let context: serde_json::Map<String, serde_json::Value> = serde_json::from_str(context_json)
        .map_err(|err| invalid(format!("invalid ObsContext payload: {err}")))?;
    let mut lines: Vec<String> = Vec::new();
    for (key, value) in &context {
        match value {
            serde_json::Value::Object(section) => {
                lines.push(format!("# {key}"));
                for (field, field_value) in section {
                    if !field_value.is_null() {
                        lines.push(format!("! {field} {}", json_scalar_str(field_value)));
                    }
                }
            }
            serde_json::Value::Null => {}
            _ => {
                if ["observers", "measurers", "coinvestigators", "collaborators"]
                    .contains(&key.as_str())
                {
                    lines.push(format!("# {key}"));
                    if let serde_json::Value::Array(names) = value {
                        for name in names {
                            lines.push(format!("! name {}", json_scalar_str(name)));
                        }
                    }
                } else if key == "fundingSource" {
                    lines.push(format!("# fundingSource {}", json_scalar_str(value)));
                } else if key == "comments" {
                    if let serde_json::Value::Array(comments) = value {
                        if !comments.is_empty() {
                            lines.push("# comment".to_string());
                            for comment in comments {
                                lines.push(format!("! line {}", json_scalar_str(comment)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(format!("{}\n", lines.join("\n")))
}

fn json_scalar_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Number(number) => {
            // Match Python str(): floats keep a trailing .0, ints do not.
            if let Some(int) = number.as_i64() {
                if number.is_f64() {
                    format!("{:?}", int as f64)
                } else {
                    int.to_string()
                }
            } else if let Some(float) = number.as_f64() {
                let rendered = format!("{float}");
                if rendered.contains('.') || rendered.contains('e') || rendered.contains("inf") {
                    rendered
                } else {
                    format!("{rendered}.0")
                }
            } else {
                number.to_string()
            }
        }
        serde_json::Value::Bool(flag) => if *flag { "True" } else { "False" }.to_string(),
        other => other.to_string(),
    }
}

/// Legacy `_parse_obs_contexts` metadata loop: returns a JSON array of
/// `[mpc_code_or_null, context_dict]` pairs, where `context_dict` mirrors the
/// legacy nested structure ({section: {key: value}}, list-valued sections for
/// observers/measurers/comment, top-level strings for e.g. fundingSource).
/// The caller builds the ObsContext dataclasses from the pairs.
pub fn ades_parse_obs_contexts(ades_string: &str) -> SchemaResult<String> {
    use serde_json::{Map, Value};

    let lines: Vec<&str> = ades_string
        .split('\n')
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();

    let mut contexts: Vec<(Option<String>, Map<String, Value>)> = Vec::new();
    let mut current: Map<String, Value> = Map::new();
    let mut current_code: Option<String> = None;
    let mut current_section_key: Option<String> = None;

    for line in lines {
        if let Some(rest) = line.strip_prefix('#') {
            let rest = rest.trim();
            if rest.starts_with("version") {
                continue;
            }
            // Legacy splits on single spaces and strips each piece.
            let mut parts = rest.split(' ').map(str::trim);
            let section_key = parts.next().unwrap_or("").to_string();
            let value_parts: Vec<&str> = parts.collect();
            if section_key == "observatory" {
                if !current.is_empty() {
                    contexts.push((current_code.take(), std::mem::take(&mut current)));
                }
                current_code = None;
            }
            if !value_parts.is_empty() {
                current.insert(section_key.clone(), Value::String(value_parts.join(" ")));
            }
            current_section_key = Some(section_key);
            continue;
        }
        if let Some(rest) = line.strip_prefix('!') {
            let rest = rest.trim();
            let mut parts = rest.split(' ').map(str::trim);
            let key = parts.next().unwrap_or("").to_string();
            let value = parts.collect::<Vec<&str>>().join(" ");
            if key == "mpcCode" {
                current_code = Some(value.clone());
            }
            let section_key = current_section_key.clone().unwrap_or_default();
            let section = current
                .entry(section_key.clone())
                .or_insert_with(|| Value::Object(Map::new()));
            // Legacy setdefault(current_section_key, {}) assumes a dict; a
            // scalar already stored here would raise in Python -- mirror by
            // replacing only when absent (well-formed files never hit this).
            if let Value::Object(section) = section {
                if ["observers", "measurers", "comment"].contains(&section_key.as_str()) {
                    let entry = section
                        .entry(key)
                        .or_insert_with(|| Value::Array(Vec::new()));
                    if let Value::Array(values) = entry {
                        values.push(Value::String(value));
                    }
                } else {
                    section.insert(key, Value::String(value));
                }
            }
        }
    }
    if !current.is_empty() {
        contexts.push((current_code.take(), current));
    }

    let payload: Vec<Value> = contexts
        .into_iter()
        .map(|(code, context)| {
            Value::Array(vec![
                code.map(Value::String).unwrap_or(Value::Null),
                Value::Object(context),
            ])
        })
        .collect();
    serde_json::to_string(&Value::Array(payload))
        .map_err(|err| invalid(format!("failed to encode ObsContexts: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isot_matches_astropy_examples() {
        // Time(60434.0, format="mjd", precision=3).utc.isot
        assert_eq!(
            isot_utc_from_mjd(60434.0, 3).unwrap(),
            "2024-05-04T00:00:00.000"
        );
        // Time(60435.2, format="mjd", precision=3).utc.isot (fixture row)
        assert_eq!(
            isot_utc_from_mjd(60435.2, 3).unwrap(),
            "2024-05-05T04:48:00.000"
        );
        // precision=1 fixture variant
        assert_eq!(
            isot_utc_from_mjd(60434.1, 1).unwrap(),
            "2024-05-04T02:24:00.0"
        );
    }

    #[test]
    fn python_fixed_matches_python_semantics() {
        assert_eq!(python_fixed(20.0, 4), "20.0000");
        assert_eq!(python_fixed(0.9659, 5), "0.96590");
        assert_eq!(python_fixed(f64::NAN, 4), "nan");
        assert_eq!(python_fixed(-15.05, 9), "-15.050000000");
    }

    #[test]
    fn isot_round_trips_through_parser() {
        // The legacy contract routes obsTime through a float64 MJD, so only
        // binary-representable day fractions round-trip exactly at ns
        // precision; non-representable fractions (e.g. 0.2 day) quantize
        // identically to legacy and are gated by the frozen legacy fixture.
        for &(days, nanos) in &[
            (60434_i64, 0_i64),
            (60434, 43_200_000_000_000),
            (60435, 21_600_000_000_000),
            (53735, 64_800_000_000_000),
        ] {
            let mjd = days as f64 + nanos as f64 / NANOS_PER_DAY_F64;
            let isot = isot_utc_from_mjd(mjd, 9).unwrap();
            let (rt_days, rt_nanos) = timestamp_from_isot_utc(&isot).unwrap();
            assert_eq!((rt_days, rt_nanos), (days, nanos), "isot {isot}");
        }
    }

    #[test]
    fn csv_quote_matches_pandas_minimal() {
        assert_eq!(csv_quote("plain"), "plain");
        assert_eq!(csv_quote("with|pipe"), "\"with|pipe\"");
        assert_eq!(csv_quote("with \"quote\""), "\"with \"\"quote\"\"\"");
    }
}
