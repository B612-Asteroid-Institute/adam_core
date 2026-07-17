"""Thin Python surface for the Rust backend.

Each wrapper does three things and nothing else:

1. Passes composed typed surfaces as PyArrow ``RecordBatch`` objects, or
   coerces raw-kernel inputs into contiguous ``float64`` NumPy arrays.
2. Delegates to ``adam_core._rust_native`` and trusts Rust to own all
   shape/length/value validation. Do not re-implement checks here - PyO3
   raises ``ValueError`` with the same messages.

The compiled Rust extension is mandatory for this migration branch. Importing
this module fails immediately if the extension is missing or if the installed
extension lacks any required native symbol.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

try:
    from adam_core import _rust_native as _native
except Exception as exc:  # pragma: no cover - depends on build/install state
    raise ImportError(
        "adam_core requires the compiled Rust extension "
        "`adam_core._rust_native`. Build/install the native wheel or run "
        "`pdm run rust-develop` in this checkout."
    ) from exc

_REQUIRED_NATIVE_SYMBOLS = (
    "AdamCoreSpiceBackend",
    "benchmark_transform_coordinates_arrow",
    "benchmark_calc_gauss",
    "benchmark_calc_gibbs",
    "benchmark_calc_herrick_gibbs",
    "transform_coordinates_arrow",
    "calculate_perturber_moids_native",
    "add_light_time_numpy",
    "add_stellar_aberration_numpy",
    "apply_cosine_latitude_correction_numpy",
    "apply_lagrange_coefficients_numpy",
    "bound_longitude_residual_column_in_place_numpy",
    "bound_longitude_residual_column_numpy",
    "bound_longitude_residuals_numpy",
    "calc_apoapsis_distance_numpy",
    "calc_chi_numpy",
    "calc_gauss_numpy",
    "calc_gibbs_numpy",
    "calc_herrick_gibbs_numpy",
    "calc_lagrange_coefficients_numpy",
    "calc_mean_anomaly_numpy",
    "calc_mean_motion_numpy",
    "calc_period_numpy",
    "calc_periapsis_distance_numpy",
    "calc_semi_latus_rectum_numpy",
    "calc_semi_major_axis_numpy",
    "calc_stumpff_numpy",
    "calculate_apparent_magnitude_v_and_phase_angle_numpy",
    "calculate_apparent_magnitude_v_numpy",
    "calculate_chi2_numpy",
    "compute_residuals_chi2_numpy",
    "calculate_moid_batch_numpy",
    "calculate_moid_numpy",
    "calculate_phase_angle_numpy",
    "cartesian_coordinate_schema_metadata",
    "cartesian_to_cometary_numpy",
    "cartesian_to_geodetic_numpy",
    "cartesian_to_keplerian_numpy",
    "cartesian_to_spherical_numpy",
    "classify_orbits_numpy",
    "cometary_to_cartesian_numpy",
    "fit_absolute_magnitude_grouped_numpy",
    "fit_absolute_magnitude_rows_numpy",
    "gauss_iod_fused_numpy",
    "gauss_iod_orbits_numpy",
    "generate_ephemeris_2body_numpy",
    "generate_ephemeris_2body_with_covariance_numpy",
    "izzo_lambert_numpy",
    "keplerian_to_cartesian_numpy",
    "naif_bodc2n",
    "naif_bodn2c",
    "naif_parse_text_kernel_bindings",
    "naif_pck_open",
    "naif_spk_open",
    "naif_spk_writer",
    "orbit_schema_metadata",
    "porkchop_grid_numpy",
    "propagate_orbits_arrow",
    "propagate_orbits_typed_arrow",
    "benchmark_propagate_orbits_arrow",
    "generate_ephemeris_arrow",
    "benchmark_generate_ephemeris_arrow",
    "calculate_perturber_moids_arrow",
    "generate_porkchop_data_arrow",
    "gauss_iod_orbits_arrow",
    "residuals_calculate_arrow",
    "predict_magnitudes_bandpass_numpy",
    "propagate_2body_along_arc_numpy",
    "propagate_2body_arc_batch_numpy",
    "propagate_2body_numpy",
    "propagate_2body_with_covariance_numpy",
    "rotate_cartesian_time_varying_numpy",
    "solve_barker_numpy",
    "solve_kepler_numpy",
    "spherical_to_cartesian_numpy",
    "tisserand_parameter_numpy",
    "transform_coordinates_numpy",
    "transform_coordinates_with_covariance_numpy",
    "weighted_covariance_numpy",
    "weighted_mean_numpy",
)

_missing_native_symbols = tuple(
    name for name in _REQUIRED_NATIVE_SYMBOLS if not hasattr(_native, name)
)
if _missing_native_symbols:  # pragma: no cover - depends on broken install state
    missing = ", ".join(_missing_native_symbols)
    raise ImportError(
        "adam_core._rust_native is present but incomplete; missing required "
        f"native symbol(s): {missing}"
    )

RUST_BACKEND_AVAILABLE = True
SPICEKIT_AVAILABLE = True


def _as_contiguous_f64(
    values: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.float64
        and values.flags.c_contiguous
    ):
        return values
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _as_contiguous_i32(
    values: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.int32
        and values.flags.c_contiguous
    ):
        return values
    return np.ascontiguousarray(np.asarray(values, dtype=np.int32))


def cartesian_coordinate_schema_metadata() -> tuple[list[str], dict[str, str]]:
    return _native.cartesian_coordinate_schema_metadata()


def orbit_schema_metadata() -> tuple[list[str], dict[str, str]]:
    return _native.orbit_schema_metadata()


def cartesian_to_spherical_numpy(coords: np.ndarray) -> np.ndarray:
    return _native.cartesian_to_spherical_numpy(_as_contiguous_f64(coords))


def spherical_to_cartesian_numpy(coords: np.ndarray) -> np.ndarray:
    return _native.spherical_to_cartesian_numpy(_as_contiguous_f64(coords))


def cartesian_to_geodetic_numpy(
    coords: np.ndarray,
    a: float,
    f: float,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    return _native.cartesian_to_geodetic_numpy(
        _as_contiguous_f64(coords), a, f, max_iter, tol
    )


def cartesian_to_keplerian_numpy(
    coords: np.ndarray,
    t0: np.ndarray | list[float] | tuple[float, ...],
    mu: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.cartesian_to_keplerian_numpy(
        _as_contiguous_f64(coords),
        _as_contiguous_f64(t0),
        _as_contiguous_f64(mu),
    )


def keplerian_to_cartesian_numpy(
    coords: np.ndarray,
    mu: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    return _native.keplerian_to_cartesian_numpy(
        _as_contiguous_f64(coords),
        _as_contiguous_f64(mu),
        max_iter,
        tol,
    )


def cartesian_to_cometary_numpy(
    coords: np.ndarray,
    t0: np.ndarray | list[float] | tuple[float, ...],
    mu: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.cartesian_to_cometary_numpy(
        _as_contiguous_f64(coords),
        _as_contiguous_f64(t0),
        _as_contiguous_f64(mu),
    )


def cometary_to_cartesian_numpy(
    coords: np.ndarray,
    t0: np.ndarray | list[float] | tuple[float, ...],
    mu: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    return _native.cometary_to_cartesian_numpy(
        _as_contiguous_f64(coords),
        _as_contiguous_f64(t0),
        _as_contiguous_f64(mu),
        max_iter,
        tol,
    )


def transform_coordinates_numpy(
    coords: np.ndarray,
    representation_in: str,
    representation_out: str,
    t0: np.ndarray | list[float] | tuple[float, ...] | None = None,
    mu: np.ndarray | list[float] | tuple[float, ...] | None = None,
    a: float | None = None,
    f: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-15,
    frame_in: str | None = None,
    frame_out: str | None = None,
    translation_vectors: np.ndarray | None = None,
) -> np.ndarray:
    return _native.transform_coordinates_numpy(
        _as_contiguous_f64(coords),
        representation_in,
        representation_out,
        None if t0 is None else _as_contiguous_f64(t0),
        None if mu is None else _as_contiguous_f64(mu),
        a,
        f,
        max_iter,
        tol,
        frame_in=frame_in,
        frame_out=frame_out,
        translation_vectors=(
            None
            if translation_vectors is None
            else _as_contiguous_f64(translation_vectors)
        ),
    )


def transform_coordinates_with_covariance_numpy(
    coords: np.ndarray,
    covariances: np.ndarray,
    representation_in: str,
    representation_out: str,
    t0: np.ndarray | list[float] | tuple[float, ...] | None = None,
    mu: np.ndarray | list[float] | tuple[float, ...] | None = None,
    a: float | None = None,
    f: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-15,
    frame_in: str | None = None,
    frame_out: str | None = None,
    translation_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _native.transform_coordinates_with_covariance_numpy(
        _as_contiguous_f64(coords),
        _as_contiguous_f64(covariances),
        representation_in,
        representation_out,
        None if t0 is None else _as_contiguous_f64(t0),
        None if mu is None else _as_contiguous_f64(mu),
        a,
        f,
        max_iter,
        tol,
        frame_in=frame_in,
        frame_out=frame_out,
        translation_vectors=(
            None
            if translation_vectors is None
            else _as_contiguous_f64(translation_vectors)
        ),
    )


def transform_coordinates_arrow(
    batch: pa.RecordBatch,
    representation_out: str,
    frame_out: str,
    target_origin: str | None = None,
    a: float = 0.0,
    f: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> Optional[pa.RecordBatch]:
    """Transform one coordinate RecordBatch entirely in Rust."""
    return _native.transform_coordinates_arrow(
        batch,
        representation_out,
        frame_out,
        target_origin,
        a,
        f,
        max_iter,
        tol,
    )


def propagate_orbits_arrow(
    orbit_batch: pa.RecordBatch,
    target_time_batch: pa.RecordBatch,
    is_variants: bool = False,
    covariance: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-14,
    chunk_size: int | None = None,
    thread_limit: int | None = None,
) -> pa.RecordBatch:
    """Propagate an Orbits RecordBatch to target epochs entirely in Rust."""
    return _native.propagate_orbits_arrow(
        orbit_batch,
        target_time_batch,
        is_variants,
        covariance,
        max_iter,
        tol,
        chunk_size,
        thread_limit,
    )


def generate_ephemeris_arrow(
    orbit_batch: pa.RecordBatch,
    observer_batch: pa.RecordBatch,
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    predict_magnitudes: bool = True,
    predict_phase_angle: bool = False,
    chunk_size: int | None = None,
    thread_limit: int | None = None,
) -> pa.RecordBatch:
    """Generate a finished Ephemeris RecordBatch entirely in Rust."""
    return _native.generate_ephemeris_arrow(
        orbit_batch,
        observer_batch,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        predict_magnitudes,
        predict_phase_angle,
        chunk_size,
        thread_limit,
    )


def residuals_calculate_arrow(
    observed_batch: pa.RecordBatch,
    predicted_batch: pa.RecordBatch,
    use_predicted_covariance: bool = True,
) -> tuple[pa.RecordBatch, bool]:
    """Return the finished Residuals RecordBatch and off-diagonal-NaN flag."""
    return _native.residuals_calculate_arrow(
        observed_batch,
        predicted_batch,
        use_predicted_covariance,
    )


def gauss_iod_orbits_arrow(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    obs_times_mjd: np.ndarray,
    coords_obs: np.ndarray,
    velocity_method: str = "gibbs",
    light_time: bool = True,
    mu: float = 0.00029591220828559115,
    c: float = 173.14463267424034,
) -> pa.RecordBatch:
    """Return the finished Gauss-IOD Orbits RecordBatch entirely from Rust."""
    return _native.gauss_iod_orbits_arrow(
        ra_deg,
        dec_deg,
        obs_times_mjd,
        coords_obs,
        velocity_method,
        light_time,
        mu,
        c,
    )


def calculate_perturber_moids_arrow(
    orbit_batch: pa.RecordBatch,
    perturber_codes: list[str],
    max_iter: int = 100,
    xtol: float = 1e-10,
) -> pa.RecordBatch:
    """Return the finished PerturberMOIDs RecordBatch entirely from Rust."""
    return _native.calculate_perturber_moids_arrow(
        orbit_batch,
        list(perturber_codes),
        max_iter,
        xtol,
    )


def generate_porkchop_data_arrow(
    departure_batch: pa.RecordBatch,
    arrival_batch: pa.RecordBatch,
    propagation_origin: str,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
) -> pa.RecordBatch:
    """Return the finished LambertSolutions RecordBatch entirely from Rust."""
    return _native.generate_porkchop_data_arrow(
        departure_batch,
        arrival_batch,
        propagation_origin,
        prograde,
        max_iter,
        tol,
    )


def calculate_perturber_moids_native(
    primary: np.ndarray,
    mus: np.ndarray | list[float] | tuple[float, ...],
    time_scale: str,
    time_days: np.ndarray,
    time_nanos: np.ndarray,
    perturber_codes: list[str],
    frame: str,
    origin_code: str,
    max_iter: int = 100,
    xtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fully-Rust perturber-MOID orchestrator (single Python->Rust crossing).

    Per perturber, spkez the perturber state vs ``origin_code`` and run the
    batched Rust MOID kernel against the primary Cartesian orbits, all in Rust.
    Returns ``(moids, dt_mins)`` each of length ``len(perturber_codes) * N``,
    laid out perturber-major then orbit-minor (``p * N + i``).
    """
    return _native.calculate_perturber_moids_native(
        _as_contiguous_f64(primary),
        _as_contiguous_f64(mus),
        time_scale,
        np.ascontiguousarray(np.asarray(time_days, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(time_nanos, dtype=np.int64)),
        list(perturber_codes),
        frame,
        origin_code,
        max_iter,
        xtol,
    )


def rotate_cartesian_time_varying_numpy(
    coords: np.ndarray,
    time_index: np.ndarray,
    matrices: np.ndarray,
    covariances: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    ti = np.ascontiguousarray(np.asarray(time_index, dtype=np.int64))
    mats = np.ascontiguousarray(np.asarray(matrices, dtype=np.float64))
    cov = None if covariances is None else _as_contiguous_f64(covariances)
    return _native.rotate_cartesian_time_varying_numpy(
        _as_contiguous_f64(coords),
        ti,
        mats,
        cov,
    )


def calc_mean_motion_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    mu: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_mean_motion_numpy(_as_contiguous_f64(a), _as_contiguous_f64(mu))


def calc_period_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    mu: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_period_numpy(_as_contiguous_f64(a), _as_contiguous_f64(mu))


def calc_periapsis_distance_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_periapsis_distance_numpy(
        _as_contiguous_f64(a),
        _as_contiguous_f64(e),
    )


def calc_apoapsis_distance_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_apoapsis_distance_numpy(
        _as_contiguous_f64(a),
        _as_contiguous_f64(e),
    )


def calc_semi_major_axis_numpy(
    q: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_semi_major_axis_numpy(
        _as_contiguous_f64(q),
        _as_contiguous_f64(e),
    )


def calc_semi_latus_rectum_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_semi_latus_rectum_numpy(
        _as_contiguous_f64(a),
        _as_contiguous_f64(e),
    )


def calc_mean_anomaly_numpy(
    nu: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_mean_anomaly_numpy(
        _as_contiguous_f64(nu),
        _as_contiguous_f64(e),
    )


def solve_barker_numpy(
    m: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.solve_barker_numpy(_as_contiguous_f64(m))


def solve_kepler_numpy(
    e: np.ndarray | list[float] | tuple[float, ...],
    m: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    return _native.solve_kepler_numpy(
        _as_contiguous_f64(e),
        _as_contiguous_f64(m),
        int(max_iter),
        float(tol),
    )


def calc_stumpff_numpy(
    psi: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calc_stumpff_numpy(_as_contiguous_f64(psi))


def calc_chi_numpy(
    r: np.ndarray,
    v: np.ndarray,
    dts: np.ndarray | list[float] | tuple[float, ...],
    mus: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    return _native.calc_chi_numpy(
        _as_contiguous_f64(r),
        _as_contiguous_f64(v),
        _as_contiguous_f64(dts),
        _as_contiguous_f64(mus),
        int(max_iter),
        float(tol),
    )


def calc_lagrange_coefficients_numpy(
    r: np.ndarray,
    v: np.ndarray,
    dts: np.ndarray | list[float] | tuple[float, ...],
    mus: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _native.calc_lagrange_coefficients_numpy(
        _as_contiguous_f64(r),
        _as_contiguous_f64(v),
        _as_contiguous_f64(dts),
        _as_contiguous_f64(mus),
        int(max_iter),
        float(tol),
    )


def apply_lagrange_coefficients_numpy(
    r: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _native.apply_lagrange_coefficients_numpy(
        _as_contiguous_f64(r),
        _as_contiguous_f64(v),
        _as_contiguous_f64(coeffs),
    )


def add_stellar_aberration_numpy(
    orbits: np.ndarray,
    observer_states: np.ndarray,
) -> np.ndarray:
    return _native.add_stellar_aberration_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(observer_states),
    )


def propagate_2body_numpy(
    orbits: np.ndarray,
    dts: np.ndarray | list[float] | tuple[float, ...],
    mus: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    return _native.propagate_2body_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(dts),
        _as_contiguous_f64(mus),
        max_iter,
        tol,
    )


def bound_longitude_residuals_numpy(
    observed: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Wrap longitude residuals to [-180, 180]° with sign-flip on 0/360 crossings.

    Returns a NEW array (rust kernel operates in place on a copy).
    """
    return _native.bound_longitude_residuals_numpy(
        _as_contiguous_f64(observed),
        _as_contiguous_f64(residuals),
    )


def bound_longitude_residual_column_numpy(
    observed: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Column-only longitude wrap: returns the wrapped ``residuals[:, 1]``
    column as a NEW ``(N,)`` array without copying the full ``(N, D)``
    buffers across the boundary (strided column reads Rust-side).
    """
    return _native.bound_longitude_residual_column_numpy(
        np.asarray(observed, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
    )


def bound_longitude_residual_column_in_place_numpy(
    observed: np.ndarray,
    residuals: np.ndarray,
) -> None:
    """Wrap ``residuals[:, 1]`` in place without allocating an output column."""
    _native.bound_longitude_residual_column_in_place_numpy(
        np.asarray(observed, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
    )


def apply_cosine_latitude_correction_numpy(
    lat: np.ndarray | list[float] | tuple[float, ...],
    residuals: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply cos(latitude) factor to spherical residuals (cols 1, 4) and
    covariance (rows/cols 1, 4). Returns (residuals, covariances)."""
    return _native.apply_cosine_latitude_correction_numpy(
        _as_contiguous_f64(lat),
        _as_contiguous_f64(residuals),
        _as_contiguous_f64(covariances),
    )


def fit_absolute_magnitude_rows_numpy(
    h_rows: np.ndarray | list[float] | tuple[float, ...],
    sigma_rows: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[float, float, float, float, int]:
    """Single-group H-fit: weighted mean (with σ) or arithmetic mean (without).

    Returns (H_hat, H_sigma, sigma_eff, chi2_red, n_used). NaN values
    where the metric isn't applicable; the caller should map NaN → None.
    """
    return _native.fit_absolute_magnitude_rows_numpy(
        _as_contiguous_f64(h_rows),
        _as_contiguous_f64(sigma_rows),
    )


def fit_absolute_magnitude_grouped_numpy(
    h_rows: np.ndarray | list[float] | tuple[float, ...],
    sigma_rows: np.ndarray | list[float] | tuple[float, ...],
    group_offsets: np.ndarray | list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched grouped H-fit. `group_offsets` shape (K+1,) with [0, ..., len(h_rows)].

    Returns 5-tuple of arrays each shape (K,): (H_hat, H_sigma, sigma_eff, chi2_red, n_used).
    """
    return _native.fit_absolute_magnitude_grouped_numpy(
        _as_contiguous_f64(h_rows),
        _as_contiguous_f64(sigma_rows),
        np.ascontiguousarray(np.asarray(group_offsets, dtype=np.int64)),
    )


def weighted_mean_numpy(
    samples: np.ndarray,
    w: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Weighted mean: `mean[k] = Σ_i W[i] · samples[i, k]` for k in 0..d."""
    return _native.weighted_mean_numpy(
        _as_contiguous_f64(samples),
        _as_contiguous_f64(w),
    )


def weighted_covariance_numpy(
    mean: np.ndarray | list[float] | tuple[float, ...],
    samples: np.ndarray,
    w_cov: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Weighted covariance: `cov[j, k] = Σ_i W_cov[i] · (samples[i,j]−mean[j]) · (samples[i,k]−mean[k])`."""
    return _native.weighted_covariance_numpy(
        _as_contiguous_f64(mean),
        _as_contiguous_f64(samples),
        _as_contiguous_f64(w_cov),
    )


def calculate_chi2_numpy(
    residuals: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Per-row Mahalanobis χ² = r·Σ⁻¹·rᵀ via Cholesky solve.

    Parameters
    ----------
    residuals : (N, D) float64 array
    covariances : (N, D, D) float64 array — Σ symmetric positive-definite.
        NaN off-diagonals are treated as 0.0; NaN diagonals raise ValueError.

    Returns
    -------
    chi2 : (N,) float64 array
    """
    return _native.calculate_chi2_numpy(
        _as_contiguous_f64(residuals),
        _as_contiguous_f64(covariances),
    )


def compute_residuals_chi2_numpy(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_cov: np.ndarray,
    predicted_cov: np.ndarray,
    is_spherical: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Fused residual + chi² pipeline (one PyO3 crossing).

    Replaces the legacy chain of
    ``bound_longitude_residuals_numpy`` + ``apply_cosine_latitude_correction_numpy`` (×2)
    + ``calculate_chi2_numpy`` (per batch) plus a Python-side per-batch loop.

    Parameters
    ----------
    observed : (N, D) float64 array
        Observed coordinate values.
    predicted : (N, D) float64 array
        Predicted coordinate values, already broadcast to N rows.
    observed_cov : (N, D, D) float64 array
        Observed covariance matrices.
    predicted_cov : (N, D, D) float64 array
        Predicted covariance matrices, with NaN entries already replaced
        with 0 by the caller when ``use_predicted_covariance=False`` is
        in effect.
    is_spherical : bool
        When True, applies longitude wrapping to ``residuals[:, 1]`` and
        the cos(latitude) correction to residuals (cols 1 and 4) and to
        rows/cols 1 and 4 of both covariance buffers.

    Returns
    -------
    residuals : (N, D) float64 array
    chi2 : (N,) float64 array — NaN where the active-dim covariance is
        entirely NaN for that row's NaN-pattern batch.
    dof : (N,) int64 array — ``D - count(NaN in observed[i, :])``.
    had_off_diagonal_nan : bool — True iff at least one batch had NaN
        within its active-dim covariance (caller surfaces the legacy
        UserWarning).
    """
    return _native.compute_residuals_chi2_numpy(
        _as_contiguous_f64(observed),
        _as_contiguous_f64(predicted),
        _as_contiguous_f64(observed_cov),
        _as_contiguous_f64(predicted_cov),
        bool(is_spherical),
    )


def classify_orbits_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
    q: np.ndarray | list[float] | tuple[float, ...],
    q_apo: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """PDS Small Bodies Node dynamical classification.

    Returns an int32 array of class codes (0=AST, 1=AMO, …, 13=HYA).
    Caller maps codes → string labels (see `dynamics/classification.py::CLASS_CODE_TO_NAME`).
    """
    return _native.classify_orbits_numpy(
        _as_contiguous_f64(a),
        _as_contiguous_f64(e),
        _as_contiguous_f64(q),
        _as_contiguous_f64(q_apo),
    )


def tisserand_parameter_numpy(
    a: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
    i_deg: np.ndarray | list[float] | tuple[float, ...],
    ap: float,
) -> np.ndarray:
    """Tisserand's parameter Tp = a_p/a + 2·cos(i)·sqrt((a/a_p)·(1−e²))."""
    return _native.tisserand_parameter_numpy(
        _as_contiguous_f64(a),
        _as_contiguous_f64(e),
        _as_contiguous_f64(i_deg),
        float(ap),
    )


def propagate_2body_along_arc_numpy(
    orbit: np.ndarray,
    dts: np.ndarray | list[float] | tuple[float, ...],
    mu: float,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    """Propagate a SINGLE orbit to many dt values.

    Internally caches the orbit-only constants (`r_mag`, `sqrt_mu`,
    `alpha`, `rv`) once and warm-starts the chi solver from the
    previous dt's converged chi — drops Newton iterations from ~120
    to ~5 per step. Use this in OD inner loops where one orbit is
    differenced against many observation epochs.

    Returns a `(N, 6)` array in the original `dts` order.
    """
    return _native.propagate_2body_along_arc_numpy(
        _as_contiguous_f64(orbit),
        _as_contiguous_f64(dts),
        float(mu),
        int(max_iter),
        float(tol),
    )


def propagate_2body_arc_batch_numpy(
    orbits: np.ndarray,
    dts: np.ndarray,
    mus: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    """Propagate N orbits to K dts each, with rayon parallelism across
    orbits and warm-started chi solving within each orbit.

    Inputs:
      * `orbits`: shape `(N, 6)` Cartesian states
      * `dts`: shape `(N, K)` per-orbit dt values
      * `mus`: shape `(N,)` gravitational parameters

    Returns: shape `(N*K, 6)` row-major (orbit-major: orbit 0's K rows,
    then orbit 1's K rows, etc.). Output preserves the input dt order
    within each orbit's block.
    """
    return _native.propagate_2body_arc_batch_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(dts),
        _as_contiguous_f64(mus),
        int(max_iter),
        float(tol),
    )


def propagate_2body_with_covariance_numpy(
    orbits: np.ndarray,
    covariances: np.ndarray,
    dts: np.ndarray | list[float] | tuple[float, ...],
    mus: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    return _native.propagate_2body_with_covariance_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(covariances),
        _as_contiguous_f64(dts),
        _as_contiguous_f64(mus),
        max_iter,
        tol,
    )


def generate_ephemeris_2body_numpy(
    orbits: np.ndarray,
    observer_states: np.ndarray,
    mus: np.ndarray | list[float] | tuple[float, ...],
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    max_lt_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _native.generate_ephemeris_2body_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(observer_states),
        _as_contiguous_f64(mus),
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    )


def generate_ephemeris_2body_with_covariance_numpy(
    orbits: np.ndarray,
    covariances: np.ndarray,
    observer_states: np.ndarray,
    mus: np.ndarray | list[float] | tuple[float, ...],
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    max_lt_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _native.generate_ephemeris_2body_with_covariance_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(covariances),
        _as_contiguous_f64(observer_states),
        _as_contiguous_f64(mus),
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    )


def calculate_phase_angle_numpy(
    object_pos: np.ndarray,
    observer_pos: np.ndarray,
) -> np.ndarray:
    return _native.calculate_phase_angle_numpy(
        _as_contiguous_f64(object_pos),
        _as_contiguous_f64(observer_pos),
    )


def calculate_apparent_magnitude_v_numpy(
    h_v: np.ndarray | list[float] | tuple[float, ...],
    object_pos: np.ndarray,
    observer_pos: np.ndarray,
    g: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    return _native.calculate_apparent_magnitude_v_numpy(
        _as_contiguous_f64(h_v),
        _as_contiguous_f64(object_pos),
        _as_contiguous_f64(observer_pos),
        _as_contiguous_f64(g),
    )


def calculate_apparent_magnitude_v_and_phase_angle_numpy(
    h_v: np.ndarray | list[float] | tuple[float, ...],
    object_pos: np.ndarray,
    observer_pos: np.ndarray,
    g: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return _native.calculate_apparent_magnitude_v_and_phase_angle_numpy(
        _as_contiguous_f64(h_v),
        _as_contiguous_f64(object_pos),
        _as_contiguous_f64(observer_pos),
        _as_contiguous_f64(g),
    )


def predict_magnitudes_bandpass_numpy(
    h_v: np.ndarray | list[float] | tuple[float, ...],
    object_pos: np.ndarray,
    observer_pos: np.ndarray,
    g: np.ndarray | list[float] | tuple[float, ...],
    target_ids: np.ndarray | list[int] | tuple[int, ...],
    delta_table: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """
    Predict apparent magnitudes in arbitrary target filters.

    Fused kernel: H-G V-band apparent magnitude + per-row target-filter
    delta lookup. `target_ids` indexes into `delta_table` (canonical filter
    ID table ordering). Out-of-range target_ids surface as NaN.
    """
    return _native.predict_magnitudes_bandpass_numpy(
        _as_contiguous_f64(h_v),
        _as_contiguous_f64(object_pos),
        _as_contiguous_f64(observer_pos),
        _as_contiguous_f64(g),
        _as_contiguous_i32(target_ids),
        _as_contiguous_f64(delta_table),
    )


def calc_gibbs_numpy(
    r1: np.ndarray | list[float] | tuple[float, ...],
    r2: np.ndarray | list[float] | tuple[float, ...],
    r3: np.ndarray | list[float] | tuple[float, ...],
    mu: float,
) -> np.ndarray:
    return _native.calc_gibbs_numpy(
        _as_contiguous_f64(r1),
        _as_contiguous_f64(r2),
        _as_contiguous_f64(r3),
        mu,
    )


def calc_herrick_gibbs_numpy(
    r1: np.ndarray | list[float] | tuple[float, ...],
    r2: np.ndarray | list[float] | tuple[float, ...],
    r3: np.ndarray | list[float] | tuple[float, ...],
    t1: float,
    t2: float,
    t3: float,
    mu: float,
) -> np.ndarray:
    return _native.calc_herrick_gibbs_numpy(
        _as_contiguous_f64(r1),
        _as_contiguous_f64(r2),
        _as_contiguous_f64(r3),
        t1,
        t2,
        t3,
        mu,
    )


def add_light_time_numpy(
    orbits: np.ndarray,
    observer_positions: np.ndarray,
    mus: np.ndarray | list[float] | tuple[float, ...],
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    max_lt_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched light-time correction.

    Iterates `lt = ||r_orbit - r_obs|| / C` and back-propagates the input
    orbit by `lt` (universal-Kepler 2-body) until LT stabilizes.

    Returns ``(aberrated_orbits[N, 6], light_time_days[N])``. NaN
    light-time on non-convergence within ``max_lt_iter`` iterations.
    """
    return _native.add_light_time_numpy(
        _as_contiguous_f64(orbits),
        _as_contiguous_f64(observer_positions),
        _as_contiguous_f64(mus),
        float(lt_tol),
        int(max_iter),
        float(tol),
        int(max_lt_iter),
    )


def calculate_moid_numpy(
    primary_orbit: np.ndarray,
    secondary_orbit: np.ndarray,
    mu: float,
    max_iter: int = 100,
    xtol: float = 1e-10,
) -> tuple[float, float]:
    """Returns `(moid, dt_at_min)`. Both orbits given as 6-vector Cartesian
    (x, y, z, vx, vy, vz). mu is the gravitational parameter of the central
    body in AU³/d² (consistent with the Cartesian units)."""
    return _native.calculate_moid_numpy(
        _as_contiguous_f64(primary_orbit),
        _as_contiguous_f64(secondary_orbit),
        mu,
        max_iter,
        xtol,
    )


def calculate_moid_batch_numpy(
    primary_orbits: np.ndarray,
    secondary_orbits: np.ndarray,
    mus: np.ndarray | list[float] | tuple[float, ...],
    max_iter: int = 100,
    xtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched MOID over N (primary, secondary) orbit pairs. Rayon-parallel.

    Returns `(moids[N], dt_at_min[N])`.
    """
    return _native.calculate_moid_batch_numpy(
        _as_contiguous_f64(primary_orbits),
        _as_contiguous_f64(secondary_orbits),
        _as_contiguous_f64(mus),
        max_iter,
        xtol,
    )


def porkchop_grid_numpy(
    dep_states: np.ndarray,
    dep_mjds: np.ndarray | list[float] | tuple[float, ...],
    arr_states: np.ndarray,
    arr_mjds: np.ndarray | list[float] | tuple[float, ...],
    mu: float,
    prograde: bool = True,
    maxiter: int = 35,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single Rust call: meshgrid + time-order filter + batched Lambert.

    Returns ``(dep_idx[V], arr_idx[V], v1[V, 3], v2[V, 3])`` for the V valid
    pairs (arr_mjd > dep_mjd). Rayon-parallel internally.
    """
    return _native.porkchop_grid_numpy(
        _as_contiguous_f64(dep_states),
        _as_contiguous_f64(dep_mjds),
        _as_contiguous_f64(arr_states),
        _as_contiguous_f64(arr_mjds),
        float(mu),
        bool(prograde),
        int(maxiter),
        float(atol),
        float(rtol),
    )


def izzo_lambert_numpy(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: np.ndarray | list[float] | tuple[float, ...],
    mu: float,
    m: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    maxiter: int = 35,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Izzo's Lambert solver, Rust-backed. Returns (v1, v2) arrays shape (N, 3)."""
    return _native.izzo_lambert_numpy(
        _as_contiguous_f64(r1),
        _as_contiguous_f64(r2),
        _as_contiguous_f64(tof),
        mu,
        m,
        prograde,
        low_path,
        maxiter,
        atol,
        rtol,
    )


def calc_gauss_numpy(
    r1: np.ndarray | list[float] | tuple[float, ...],
    r2: np.ndarray | list[float] | tuple[float, ...],
    r3: np.ndarray | list[float] | tuple[float, ...],
    t1: float,
    t2: float,
    t3: float,
    mu: float,
) -> np.ndarray:
    return _native.calc_gauss_numpy(
        _as_contiguous_f64(r1),
        _as_contiguous_f64(r2),
        _as_contiguous_f64(r3),
        t1,
        t2,
        t3,
        mu,
    )


def gauss_iod_fused_numpy(
    ra_deg: np.ndarray | list[float] | tuple[float, ...],
    dec_deg: np.ndarray | list[float] | tuple[float, ...],
    obs_times_mjd: np.ndarray | list[float] | tuple[float, ...],
    coords_obs: np.ndarray,
    velocity_method: str,
    light_time: bool,
    mu: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    epochs, orbits = _native.gauss_iod_fused_numpy(
        _as_contiguous_f64(ra_deg),
        _as_contiguous_f64(dec_deg),
        _as_contiguous_f64(obs_times_mjd),
        _as_contiguous_f64(coords_obs),
        velocity_method,
        light_time,
        mu,
        c,
    )
    return np.asarray(epochs, dtype=np.float64), np.asarray(orbits, dtype=np.float64)


def gauss_iod_orbits_numpy(
    r2_mags: np.ndarray | list[float] | tuple[float, ...],
    q1: np.ndarray | list[float] | tuple[float, ...],
    q2: np.ndarray | list[float] | tuple[float, ...],
    q3: np.ndarray | list[float] | tuple[float, ...],
    rho1_hat: np.ndarray | list[float] | tuple[float, ...],
    rho2_hat: np.ndarray | list[float] | tuple[float, ...],
    rho3_hat: np.ndarray | list[float] | tuple[float, ...],
    t1: float,
    t2: float,
    t3: float,
    v: float,
    velocity_method: str,
    light_time: bool,
    mu: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    epochs, orbits = _native.gauss_iod_orbits_numpy(
        _as_contiguous_f64(r2_mags),
        _as_contiguous_f64(q1),
        _as_contiguous_f64(q2),
        _as_contiguous_f64(q3),
        _as_contiguous_f64(rho1_hat),
        _as_contiguous_f64(rho2_hat),
        _as_contiguous_f64(rho3_hat),
        t1,
        t2,
        t3,
        v,
        velocity_method,
        light_time,
        mu,
        c,
    )
    return np.asarray(epochs, dtype=np.float64), np.asarray(orbits, dtype=np.float64)


def naif_spk_open(path: str):
    """Open a pure-Rust SPK reader on `path`."""
    return _native.naif_spk_open(path)


def adam_core_spice_backend():
    """Create adam-core's direct Rust-to-Rust SPICE backend."""
    return _native.AdamCoreSpiceBackend()


def naif_bodn2c(name: str) -> Optional[int]:
    """Resolve a NAIF body name to its integer ID using the pure-Rust
    built-in table.

    Returns ``None`` when the name is not in the built-in set. Callers that
    need to resolve custom-kernel names should route through
    :class:`RustBackend.bodn2c`, which also consults text-kernel bindings.
    """
    try:
        return int(_native.naif_bodn2c(name))
    except ValueError:
        return None


def naif_bodc2n(code: int) -> Optional[str]:
    """Reverse of :func:`naif_bodn2c`.

    Returns ``None`` when the code is not in the built-in table.
    """
    try:
        return str(_native.naif_bodc2n(int(code)))
    except ValueError:
        return None


def naif_parse_text_kernel_bindings(path: str) -> list[tuple[str, int]]:
    """Parse a SPICE text kernel (``.tk``/``.tf``/``.tpc``/``.ti``) and return
    the ordered list of ``NAIF_BODY_NAME`` ↔ ``NAIF_BODY_CODE`` bindings it
    declares.

    Returns an empty list if no body bindings are present. Raises
    ``ValueError`` for malformed kernels or mismatched array lengths.
    """
    return list(_native.naif_parse_text_kernel_bindings(path))


def naif_spk_writer(locifn: str = "adam-core"):
    """Create an in-memory pure-Rust SPK writer.

    The returned object exposes:
      * ``add_type3(target, center, frame_id, start_et, end_et, segment_id,
                    init, intlen, records_coeffs)`` — append a Type 3 Chebyshev
        position+velocity segment. ``records_coeffs`` is shape
        ``(n_records, 2 + 6*(degree+1))`` with each row
        ``[mid, radius, x..., y..., z..., vx..., vy..., vz...]``.
      * ``add_type9(target, center, frame_id, start_et, end_et, segment_id,
                    degree, states, epochs)`` — append a Type 9 Lagrange
        discrete-state segment. ``states`` has shape ``(N, 6)``.
      * ``write(path)`` — serialize the assembled DAF/SPK to disk via an
        atomic temp-file rename.
    """
    return _native.naif_spk_writer(locifn)


def naif_pck_open(path: str):
    """Open a pure-Rust binary PCK reader on `path`.

    The returned object exposes `sxform(from, to, et)` and
    `pxform(from, to, et)` matching CSPICE for the J2000↔ITRF93 pair,
    plus `euler_state(body_frame, et)` for raw angle access and
    `rotate_state_batch(from, to, ets, states)` for vectorized rotation.
    """
    return _native.naif_pck_open(path)
