//! Bandpass-driven photometry runtime (bead personal-cmy.24).
//!
//! Rust-native port of `adam_core.photometry.bandpasses.api` and the delta
//! tables in `adam_core.photometry.magnitude_common`: the vendored parquet
//! data (bandpass curves, observatory band map, template photon integrals,
//! solar spectrum) is loaded Rust-side, and all runtime compute -- solar
//! normalizations, photon-counting integrals (np.interp + trapezoid
//! semantics), template/mix integrals, V-relative delta tables, canonical
//! band mapping (including the LSST/X05 reported-band normalization and
//! SDSS/PS1 fallbacks), and the custom-template registry -- runs here with
//! legacy-exact error messages. The Python modules become thin bindings.
//!
//! Documented deviation: `np.trapezoid` uses numpy's pairwise summation;
//! this port sums sequentially, so integral-derived values are gated at
//! tight relative tolerance (<= 1e-12) rather than bit-exactness.

use crate::types::{SchemaError, SchemaResult};
use arrow_array::{Array, Float64Array, LargeListArray, LargeStringArray, ListArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

fn py_list_repr(items: &[String]) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|item| format!("'{}'", item.replace('\\', "\\\\").replace('\'', "\\'")))
        .collect();
    format!("[{}]", quoted.join(", "))
}

// --- numpy-equivalent primitives -------------------------------------------------

/// `np.interp(x_new, x, y, left=0.0, right=0.0)` for sorted `x`.
fn np_interp_zero_fill(x_new: &[f64], x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    x_new
        .iter()
        .map(|&xi| {
            if n == 0 {
                return 0.0;
            }
            if xi < x[0] || xi > x[n - 1] {
                return 0.0;
            }
            // partition_point gives the first index with x[idx] > xi.
            let hi = x.partition_point(|&v| v <= xi);
            if hi == 0 {
                return y[0];
            }
            if hi >= n {
                return y[n - 1];
            }
            let lo = hi - 1;
            if x[hi] == x[lo] {
                return y[lo];
            }
            let slope = (y[hi] - y[lo]) / (x[hi] - x[lo]);
            slope * (xi - x[lo]) + y[lo]
        })
        .collect()
}

/// `np.trapezoid(y, x)` with sequential summation (see module docs).
fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    let mut total = 0.0f64;
    for i in 1..x.len() {
        total += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }
    total
}

// --- parquet loading ---------------------------------------------------------------

fn read_parquet_batches(path: &Path) -> SchemaResult<Vec<arrow_array::RecordBatch>> {
    let file = File::open(path)
        .map_err(|err| invalid(format!("failed to open {}: {err}", path.display())))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| invalid(format!("failed to read {}: {err}", path.display())))?
        .build()
        .map_err(|err| invalid(format!("failed to read {}: {err}", path.display())))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| invalid(format!("failed to read {}: {err}", path.display())))
}

fn string_column(batches: &[arrow_array::RecordBatch], name: &str) -> SchemaResult<Vec<String>> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name(name)
            .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?;
        if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
            for row in 0..array.len() {
                out.push(array.value(row).to_string());
            }
        } else if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
            for row in 0..array.len() {
                out.push(array.value(row).to_string());
            }
        } else {
            return Err(invalid(format!("column {name} must be a string column")));
        }
    }
    Ok(out)
}

fn f64_column(batches: &[arrow_array::RecordBatch], name: &str) -> SchemaResult<Vec<f64>> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name(name)
            .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?;
        let array = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| invalid(format!("column {name} must be Float64")))?;
        for row in 0..array.len() {
            out.push(array.value(row));
        }
    }
    Ok(out)
}

fn f64_values(array: &dyn Array) -> SchemaResult<Vec<f64>> {
    let values = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| invalid("list values must be Float64".to_string()))?;
    Ok((0..values.len()).map(|row| values.value(row)).collect())
}

fn f64_list_column(
    batches: &[arrow_array::RecordBatch],
    name: &str,
) -> SchemaResult<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name(name)
            .ok_or_else(|| SchemaError::MissingRequiredField(name.to_string()))?;
        if let Some(array) = column.as_any().downcast_ref::<LargeListArray>() {
            for row in 0..array.len() {
                out.push(f64_values(array.value(row).as_ref())?);
            }
        } else if let Some(array) = column.as_any().downcast_ref::<ListArray>() {
            for row in 0..array.len() {
                out.push(f64_values(array.value(row).as_ref())?);
            }
        } else {
            return Err(invalid(format!("column {name} must be a list column")));
        }
    }
    Ok(out)
}

// --- data model ---------------------------------------------------------------------

/// Loaded vendored bandpass data plus derived solar normalizations.
pub struct BandpassData {
    pub filter_ids: Vec<String>,
    filter_index: HashMap<String, usize>,
    curves: Vec<(Vec<f64>, Vec<f64>)>,
    band_map: HashMap<String, String>,
    template_integrals: HashMap<String, f64>,
    solar_wl: Vec<f64>,
    solar_flux: Vec<f64>,
    solar_norms: Vec<f64>,
}

impl BandpassData {
    pub fn load(data_dir: &Path) -> SchemaResult<Self> {
        let curves_batches = read_parquet_batches(&data_dir.join("bandpass_curves.parquet"))?;
        let filter_ids = string_column(&curves_batches, "filter_id")?;
        let wavelengths = f64_list_column(&curves_batches, "wavelength_nm")?;
        let throughputs = f64_list_column(&curves_batches, "throughput")?;
        let curves: Vec<(Vec<f64>, Vec<f64>)> = wavelengths.into_iter().zip(throughputs).collect();

        let map_batches = read_parquet_batches(&data_dir.join("observatory_band_map.parquet"))?;
        let map_keys = string_column(&map_batches, "key")?;
        let map_filters = string_column(&map_batches, "filter_id")?;
        // pc.index_in returns the FIRST match; keep the first mapping per key.
        let mut band_map = HashMap::new();
        for (key, filter_id) in map_keys.into_iter().zip(map_filters) {
            band_map.entry(key).or_insert(filter_id);
        }

        let integral_batches =
            read_parquet_batches(&data_dir.join("template_bandpass_integrals.parquet"))?;
        let template_ids = string_column(&integral_batches, "template_id")?;
        let integral_filters = string_column(&integral_batches, "filter_id")?;
        let integral_values = f64_column(&integral_batches, "integral_photon")?;
        let mut template_integrals = HashMap::new();
        for ((template, filter), value) in template_ids
            .into_iter()
            .zip(integral_filters)
            .zip(integral_values)
        {
            template_integrals
                .entry(format!("{template}|{filter}"))
                .or_insert(value);
        }

        let solar_batches = read_parquet_batches(&data_dir.join("solar_spectrum.parquet"))?;
        let solar_wl = f64_column(&solar_batches, "wavelength_nm")?;
        let solar_flux = f64_column(&solar_batches, "flux")?;

        let mut filter_index = HashMap::new();
        for (index, filter_id) in filter_ids.iter().enumerate() {
            filter_index.entry(filter_id.clone()).or_insert(index);
        }

        let mut data = Self {
            filter_ids,
            filter_index,
            curves,
            band_map,
            template_integrals,
            solar_wl,
            solar_flux,
            solar_norms: Vec::new(),
        };
        data.solar_norms = data.compute_solar_norms()?;
        Ok(data)
    }

    /// `D = trapz(F_sun * T * lambda)` over the solar/filter overlap for every filter.
    fn compute_solar_norms(&self) -> SchemaResult<Vec<f64>> {
        let mut norms = Vec::with_capacity(self.filter_ids.len());
        for (index, filter_id) in self.filter_ids.iter().enumerate() {
            let (wl, thr) = &self.curves[index];
            let value = self.filter_solar_norm_photon(wl, thr);
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "Invalid solar normalization for filter_id '{filter_id}'"
                )));
            }
            norms.push(value);
        }
        Ok(norms)
    }

    fn filter_solar_norm_photon(&self, wl: &[f64], thr: &[f64]) -> f64 {
        if wl.len() != thr.len() || wl.len() < 2 {
            return f64::NAN;
        }
        let (Some(&solar_first), Some(&solar_last)) = (self.solar_wl.first(), self.solar_wl.last())
        else {
            return f64::NAN;
        };
        let wl_min = solar_min(wl).max(solar_first);
        let wl_max = solar_max(wl).min(solar_last);
        if wl_max <= wl_min {
            return f64::NAN;
        }
        let (grid, sun) = self.solar_window(wl_min, wl_max);
        if grid.len() < 2 {
            return f64::NAN;
        }
        let t = np_interp_zero_fill(&grid, wl, thr);
        let integrand: Vec<f64> = sun
            .iter()
            .zip(&t)
            .zip(&grid)
            .map(|((&s, &t), &lambda)| s * t * lambda)
            .collect();
        trapezoid(&integrand, &grid)
    }

    fn solar_window(&self, wl_min: f64, wl_max: f64) -> (Vec<f64>, Vec<f64>) {
        let mut grid = Vec::new();
        let mut sun = Vec::new();
        for (&wl, &flux) in self.solar_wl.iter().zip(&self.solar_flux) {
            if wl >= wl_min && wl <= wl_max {
                grid.push(wl);
                sun.push(flux);
            }
        }
        (grid, sun)
    }

    /// `I = trapz(F_sun * R_ast * T * lambda)` on the solar grid over the
    /// three-way wavelength overlap.
    fn integral_photon(
        &self,
        filter_wl: &[f64],
        filter_thr: &[f64],
        template_wl: &[f64],
        template_refl: &[f64],
    ) -> f64 {
        let (Some(&solar_first), Some(&solar_last)) = (self.solar_wl.first(), self.solar_wl.last())
        else {
            return f64::NAN;
        };
        let wl_min = solar_first
            .max(solar_min(filter_wl))
            .max(solar_min(template_wl));
        let wl_max = solar_last
            .min(solar_max(filter_wl))
            .min(solar_max(template_wl));
        if wl_max <= wl_min {
            return f64::NAN;
        }
        let (grid, sun) = self.solar_window(wl_min, wl_max);
        if grid.len() < 2 {
            return f64::NAN;
        }
        let t = np_interp_zero_fill(&grid, filter_wl, filter_thr);
        let r = np_interp_zero_fill(&grid, template_wl, template_refl);
        let integrand: Vec<f64> = sun
            .iter()
            .zip(&r)
            .zip(&t)
            .zip(&grid)
            .map(|(((&s, &r), &t), &lambda)| s * r * t * lambda)
            .collect();
        trapezoid(&integrand, &grid)
    }

    fn solar_norm_for_filter(&self, filter_id: &str) -> SchemaResult<f64> {
        match self.filter_index.get(filter_id) {
            Some(&index) => Ok(self.solar_norms[index]),
            None => Err(invalid(format!(
                "Unknown filter_id for solar normalization: {filter_id}"
            ))),
        }
    }

    fn get_integrals_precomputed(
        &self,
        template_id: &str,
        filter_ids: &[String],
    ) -> SchemaResult<Vec<f64>> {
        if filter_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut numerators = Vec::with_capacity(filter_ids.len());
        let mut missing: BTreeSet<String> = BTreeSet::new();
        for filter_id in filter_ids {
            let key = format!("{template_id}|{filter_id}");
            match self.template_integrals.get(&key) {
                Some(&value) => numerators.push(value),
                None => {
                    missing.insert(key);
                    numerators.push(f64::NAN);
                }
            }
        }
        if !missing.is_empty() {
            let missing: Vec<String> = missing.into_iter().collect();
            return Err(invalid(format!(
                "Missing precomputed integrals for template '{template_id}' and filters: {}",
                py_list_repr(&missing)
            )));
        }
        filter_ids
            .iter()
            .zip(numerators)
            .map(|(filter_id, numerator)| Ok(numerator / self.solar_norm_for_filter(filter_id)?))
            .collect()
    }

    pub fn get_integrals(
        &self,
        template_id: &str,
        filter_ids: &[String],
    ) -> SchemaResult<Vec<f64>> {
        let custom = custom_state().lock().expect("custom template lock");
        let template = custom.templates.get(template_id).cloned();
        drop(custom);

        let Some((template_wl, template_refl)) = template else {
            return self.get_integrals_precomputed(template_id, filter_ids);
        };

        let mut out = Vec::with_capacity(filter_ids.len());
        for filter_id in filter_ids {
            let denom = self.solar_norm_for_filter(filter_id)?;
            let cache_key = (template_id.to_string(), filter_id.clone());
            let cached = {
                let custom = custom_state().lock().expect("custom template lock");
                custom.integrals.get(&cache_key).copied()
            };
            let numerator = match cached {
                Some(value) => value,
                None => {
                    let Some(&index) = self.filter_index.get(filter_id) else {
                        return Err(invalid(format!(
                            "Unknown filter_id '{filter_id}' for custom template integral computation"
                        )));
                    };
                    let (filter_wl, filter_thr) = &self.curves[index];
                    let value =
                        self.integral_photon(filter_wl, filter_thr, &template_wl, &template_refl);
                    let mut custom = custom_state().lock().expect("custom template lock");
                    custom.integrals.insert(cache_key, value);
                    value
                }
            };
            out.push(numerator / denom);
        }
        Ok(out)
    }

    pub fn compute_mix_integrals(
        &self,
        weight_c: f64,
        weight_s: f64,
        filter_ids: &[String],
    ) -> SchemaResult<Vec<f64>> {
        if !weight_c.is_finite() || !weight_s.is_finite() {
            return Err(invalid("weights must be finite".to_string()));
        }
        if weight_c < 0.0 || weight_s < 0.0 {
            return Err(invalid("weights must be non-negative".to_string()));
        }
        let total = weight_c + weight_s;
        if total <= 0.0 {
            return Err(invalid("at least one weight must be > 0".to_string()));
        }
        let w_c = weight_c / total;
        let w_s = weight_s / total;
        let ints_c = self.get_integrals("C", filter_ids)?;
        let ints_s = self.get_integrals("S", filter_ids)?;
        Ok(ints_c
            .iter()
            .zip(&ints_s)
            .map(|(&c, &s)| w_c * c + w_s * s)
            .collect())
    }

    fn v_index(&self) -> SchemaResult<usize> {
        self.filter_index.get("V").copied().ok_or_else(|| {
            invalid("Bandpass curves must include a canonical 'V' filter_id.".to_string())
        })
    }

    /// Per-filter `delta = m_filter - m_V = -2.5 log10(I / I_V)` over ALL
    /// vendored filters, for a template or a normalized C/S mix.
    pub fn delta_table(
        &self,
        template_id: Option<&str>,
        mix: Option<(f64, f64)>,
    ) -> SchemaResult<Vec<f64>> {
        let v_index = self.v_index()?;
        let integrals = match (template_id, mix) {
            (Some(template_id), None) => self.get_integrals(template_id, &self.filter_ids)?,
            (None, Some((w_c, w_s))) => self.compute_mix_integrals(w_c, w_s, &self.filter_ids)?,
            _ => {
                return Err(invalid(
                    "exactly one of template_id or mix weights is required".to_string(),
                ));
            }
        };
        let i_v = integrals[v_index];
        if !i_v.is_finite() || i_v <= 0.0 {
            return Err(invalid(
                "Invalid V-band integral for bandpass conversion.".to_string(),
            ));
        }
        Ok(integrals
            .iter()
            .map(|&value| -2.5 * (value / i_v).log10())
            .collect())
    }

    pub fn delta_mag(
        &self,
        template_id: Option<&str>,
        mix: Option<(f64, f64)>,
        source_filter_id: &str,
        target_filter_id: &str,
    ) -> SchemaResult<f64> {
        if source_filter_id.is_empty() || target_filter_id.is_empty() {
            return Err(invalid(
                "source_filter_id and target_filter_id must be non-empty".to_string(),
            ));
        }
        if source_filter_id == target_filter_id {
            return Ok(0.0);
        }
        let ids = vec![source_filter_id.to_string(), target_filter_id.to_string()];
        let integrals = match (template_id, mix) {
            (Some(template_id), None) => self.get_integrals(template_id, &ids)?,
            (None, Some((w_c, w_s))) => self.compute_mix_integrals(w_c, w_s, &ids)?,
            _ => {
                return Err(invalid(
                    "exactly one of template_id or mix weights is required".to_string(),
                ));
            }
        };
        let (i_src, i_tgt) = (integrals[0], integrals[1]);
        if !i_src.is_finite() || !i_tgt.is_finite() || i_src <= 0.0 || i_tgt <= 0.0 {
            return Err(invalid(format!(
                "Invalid integrals for delta magnitude {source_filter_id} -> {target_filter_id}"
            )));
        }
        Ok(-2.5 * (i_tgt / i_src).log10())
    }

    pub fn color_terms(
        &self,
        template_id: Option<&str>,
        mix: Option<(f64, f64)>,
        source_filter_id: &str,
        target_filter_ids: Option<&[String]>,
    ) -> SchemaResult<Vec<(String, f64)>> {
        if source_filter_id.is_empty() {
            return Err(invalid("source_filter_id must be non-empty".to_string()));
        }
        let targets: Vec<String> = match target_filter_ids {
            Some(targets) => targets.to_vec(),
            None => self.filter_ids.clone(),
        }
        .into_iter()
        .filter(|target| !target.is_empty() && target != source_filter_id)
        .collect();
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::with_capacity(targets.len() + 1);
        ids.push(source_filter_id.to_string());
        ids.extend(targets.iter().cloned());
        let integrals = match (template_id, mix) {
            (Some(template_id), None) => self.get_integrals(template_id, &ids)?,
            (None, Some((w_c, w_s))) => self.compute_mix_integrals(w_c, w_s, &ids)?,
            _ => {
                return Err(invalid(
                    "exactly one of template_id or mix weights is required".to_string(),
                ));
            }
        };
        let i_src = integrals[0];
        if !i_src.is_finite() || i_src <= 0.0 {
            return Err(invalid(format!(
                "Invalid integral for source filter '{source_filter_id}'"
            )));
        }
        Ok(targets
            .into_iter()
            .zip(integrals[1..].iter())
            .map(|(target, &value)| (target, -2.5 * (value / i_src).log10()))
            .collect())
    }

    pub fn has_filter(&self, filter_id: &str) -> bool {
        self.filter_index.contains_key(filter_id)
    }

    /// Legacy `assert_filter_ids_have_curves` error (sorted-unique repr).
    pub fn assert_filter_ids_have_curves(&self, filter_ids: &[String]) -> SchemaResult<()> {
        let missing: BTreeSet<String> = filter_ids
            .iter()
            .filter(|filter_id| !self.has_filter(filter_id))
            .cloned()
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let missing: Vec<String> = missing.into_iter().collect();
        Err(invalid(format!(
            "Unknown filter_id(s) (no vendored bandpass curve): {}. Run \
             map_to_canonical_filter_bands() first to map observatory bands to \
             canonical filter_ids.",
            py_list_repr(&missing)
        )))
    }

    /// Legacy `map_to_canonical_filter_bands`.
    pub fn map_to_canonical_filter_bands(
        &self,
        observatory_codes: &[String],
        bands: &[String],
        allow_fallback_filters: bool,
    ) -> SchemaResult<Vec<String>> {
        if observatory_codes.len() != bands.len() {
            return Err(invalid(format!(
                "observatory_codes length ({}) must match bands length ({})",
                observatory_codes.len(),
                bands.len()
            )));
        }
        let n = bands.len();
        let mut out: Vec<Option<String>> = vec![None; n];
        let mut used_fallback = vec![false; n];

        for row in 0..n {
            let band = &bands[row];
            // Pass-through if already a known canonical filter_id.
            if self.has_filter(band) {
                out[row] = Some(band.clone());
                continue;
            }
            // (code, normalized band) mapping.
            let code = &observatory_codes[row];
            let normalized = if code == "X05" {
                normalize_x05_band(band)
            } else {
                band.clone()
            };
            if let Some(filter_id) = self.band_map.get(&format!("{code}|{normalized}")) {
                out[row] = Some(filter_id.clone());
                continue;
            }
            // Conservative generic-band fallbacks.
            let fallback = match band.to_lowercase().as_str() {
                "u" => Some("SDSS_u"),
                "g" => Some("SDSS_g"),
                "r" => Some("SDSS_r"),
                "i" => Some("SDSS_i"),
                "z" => Some("SDSS_z"),
                "y" => Some("PS1_y"),
                _ => None,
            };
            if let Some(fallback) = fallback {
                out[row] = Some(fallback.to_string());
                used_fallback[row] = true;
            }
        }

        let unresolved: Vec<usize> = (0..n).filter(|&row| out[row].is_none()).collect();
        if !unresolved.is_empty() {
            let pairs: Vec<String> = unresolved
                .iter()
                .map(|&row| format!("{}|{}", observatory_codes[row], bands[row]))
                .collect();
            return Err(invalid(format!(
                "Unable to suggest canonical filter_id(s) for: {}",
                pairs.join(", ")
            )));
        }

        if !allow_fallback_filters && used_fallback.iter().any(|&used| used) {
            let pairs: Vec<String> = (0..n)
                .filter(|&row| used_fallback[row])
                .map(|row| format!("{}|{}", observatory_codes[row], bands[row]))
                .collect();
            return Err(invalid(format!(
                "No non-fallback mapping found for: {}. Set \
                 allow_fallback_filters=True to allow SDSS/PS1 fallbacks.",
                pairs.join(", ")
            )));
        }

        let resolved: Vec<String> = out.into_iter().map(|value| value.unwrap()).collect();
        // Final guarantee: every output has a curve.
        let bad: BTreeSet<String> = resolved
            .iter()
            .filter(|filter_id| !self.has_filter(filter_id))
            .cloned()
            .collect();
        if !bad.is_empty() {
            let bad: Vec<String> = bad.into_iter().collect();
            return Err(invalid(format!(
                "Suggested filter_id(s) do not have vendored curves: {}",
                py_list_repr(&bad)
            )));
        }
        Ok(resolved)
    }
}

fn solar_min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn solar_max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// LSST/X05 reported-band normalization (legacy `_normalize_reported_band_for_station`).
fn normalize_x05_band(band: &str) -> String {
    let mut value = band.trim().to_string();
    if let Some(rest) = value.strip_prefix("LSST_") {
        value = rest.to_string();
    }
    let bytes = value.as_bytes();
    if bytes.len() == 2 && bytes[0] == b'L' && b"ugrizy".contains(&bytes[1]) {
        value = value[1..].to_string();
    }
    let bytes = value.as_bytes();
    if bytes.len() == 2 && bytes[0] == b'L' && b"UGRIZY".contains(&bytes[1]) {
        value = value[1..].to_lowercase();
    }
    if value == "Y" {
        value = "y".to_string();
    }
    value
}

// --- process-wide state ----------------------------------------------------------------

#[derive(Default)]
struct CustomState {
    templates: HashMap<String, (Vec<f64>, Vec<f64>)>,
    integrals: HashMap<(String, String), f64>,
}

fn custom_state() -> &'static Mutex<CustomState> {
    static CUSTOM: OnceLock<Mutex<CustomState>> = OnceLock::new();
    CUSTOM.get_or_init(|| Mutex::new(CustomState::default()))
}

/// Legacy `register_custom_template` (validation + stable sort + cache clear).
pub fn register_custom_template(
    template_id: &str,
    wavelength_nm: &[f64],
    reflectance: &[f64],
) -> SchemaResult<()> {
    if template_id.is_empty() {
        return Err(invalid("template_id must be non-empty".to_string()));
    }
    if wavelength_nm.len() != reflectance.len() {
        return Err(invalid(
            "wavelength_nm and reflectance must have the same length".to_string(),
        ));
    }
    if wavelength_nm.len() < 2 {
        return Err(invalid(
            "template arrays must have at least 2 points".to_string(),
        ));
    }
    // np.argsort default is a stable ordering for our purposes; sort pairs by
    // wavelength with a stable sort to match.
    let mut order: Vec<usize> = (0..wavelength_nm.len()).collect();
    order.sort_by(|&a, &b| {
        wavelength_nm[a]
            .partial_cmp(&wavelength_nm[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let wl: Vec<f64> = order.iter().map(|&index| wavelength_nm[index]).collect();
    let rf: Vec<f64> = order.iter().map(|&index| reflectance[index]).collect();
    if wl.iter().any(|value| !value.is_finite()) || rf.iter().any(|value| !value.is_finite()) {
        return Err(invalid("template arrays must be finite".to_string()));
    }
    if wl.windows(2).any(|pair| pair[1] - pair[0] <= 0.0) {
        return Err(invalid(
            "wavelength_nm must be strictly increasing".to_string(),
        ));
    }
    let mut custom = custom_state().lock().expect("custom template lock");
    custom.templates.insert(template_id.to_string(), (wl, rf));
    custom.integrals.clear();
    Ok(())
}

/// Test hook mirroring the legacy `_clear_custom_cache` + registry reset.
pub fn clear_custom_templates() {
    let mut custom = custom_state().lock().expect("custom template lock");
    custom.templates.clear();
    custom.integrals.clear();
}

/// Process-wide cache of loaded bandpass data keyed by data directory.
pub fn bandpass_data(data_dir: &Path) -> SchemaResult<Arc<BandpassData>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<BandpassData>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let key = data_dir.display().to_string();
    {
        let registry = registry.lock().expect("bandpass registry lock");
        if let Some(data) = registry.get(&key) {
            return Ok(data.clone());
        }
    }
    let data = Arc::new(BandpassData::load(data_dir)?);
    let mut registry = registry.lock().expect("bandpass registry lock");
    Ok(registry.entry(key).or_insert(data).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn np_interp_zero_fill_matches_numpy_semantics() {
        let x = [1.0, 2.0, 4.0];
        let y = [10.0, 20.0, 40.0];
        // Below range -> 0, at edges -> edge values, interior -> linear.
        let out = np_interp_zero_fill(&[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5], &x, &y);
        assert_eq!(out, vec![0.0, 10.0, 15.0, 20.0, 30.0, 40.0, 0.0]);
    }

    #[test]
    fn trapezoid_matches_reference() {
        let x = [0.0, 1.0, 3.0];
        let y = [0.0, 2.0, 4.0];
        // (1)*(0+2)/2 + (2)*(2+4)/2 = 1 + 6
        assert_eq!(trapezoid(&y, &x), 7.0);
    }

    #[test]
    fn normalize_x05_band_matches_legacy_rules() {
        assert_eq!(normalize_x05_band("LSST_g"), "g");
        assert_eq!(normalize_x05_band("Lg"), "g");
        assert_eq!(normalize_x05_band("LY"), "y");
        assert_eq!(normalize_x05_band("Y"), "y");
        assert_eq!(normalize_x05_band(" r "), "r");
        assert_eq!(normalize_x05_band("g"), "g");
        assert_eq!(normalize_x05_band("Lq"), "Lq");
    }

    #[test]
    fn custom_template_validation_matches_legacy_messages() {
        clear_custom_templates();
        let err = register_custom_template("", &[1.0, 2.0], &[0.5, 0.6]).unwrap_err();
        assert!(err.to_string().contains("template_id must be non-empty"));
        let err = register_custom_template("T", &[1.0], &[0.5]).unwrap_err();
        assert!(err
            .to_string()
            .contains("template arrays must have at least 2 points"));
        let err = register_custom_template("T", &[1.0, 1.0], &[0.5, 0.6]).unwrap_err();
        assert!(err
            .to_string()
            .contains("wavelength_nm must be strictly increasing"));
        clear_custom_templates();
    }
}
