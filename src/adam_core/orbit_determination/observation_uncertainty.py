"""
Observation uncertainty models for orbit determination.

This module defines a generic interface (`ObservationUncertaintyModel`) for
transforming the observation uncertainties of `OrbitDeterminationObservations`
prior to orbit fitting, together with a set of interpreter models that turn
an observatory bias table (see `BIAS_TABLE_SCHEMA`) into inflated observation
covariances.

The bias table itself is produced externally (e.g. by the
`adam-observatory-uncertainties` package); adam_core deliberately does not
bundle any bias numbers nor depend on that package. The two meet only at the
table schema defined here.

Frames and units
----------------
Bias-table angular quantities (biases, RMS values and residual variances /
covariances) are expressed in arcseconds (arcsec² for variances) with the RA
axis in the cos(dec)-corrected frame, following the MPC/ADES ``rmsra``
convention. `SphericalCoordinates` covariances are expressed in degrees²
with lon (RA) NOT cos(dec)-corrected. Converting a bias-table quantity onto
the lon axis therefore requires dividing by cos(dec) once per RA factor:
variances by cos²(dec), the RA×Dec cross-covariance by cos(dec) once, and
1-sigma values by cos(dec) once (in addition to the arcsec → degree scaling).

The shipped models modify ONLY the RA/Dec covariance block of the input
observations; observed positions (lon/lat) and every other column pass
through unchanged. This is a property of these implementations, not a
constraint of the interface: `apply` returns a full
`OrbitDeterminationObservations` so that subclasses defined elsewhere are
free to implement other transformations. Use `assert_positions_unchanged`
in tests to prove that a given model is position-preserving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import pyarrow as pa

from ..coordinates.covariances import CoordinateCovariances
from .evaluate import OrbitDeterminationObservations

__all__ = [
    "BIAS_TABLE_SCHEMA",
    "ObservationUncertaintyModel",
    "IdentityModel",
    "EmpiricalCovarianceModel",
    "PerformanceWeightedModel",
    "SigmaFloorModel",
    "validate_bias_table",
    "assert_positions_unchanged",
]

ARCSEC_PER_DEG = 3600.0

# Standard observatory bias-table schema: the contract between adam_core's
# interpreter models and any package reporting bias numbers. One row per
# station rollup (or per station+band), keyed by obs_code (+ optional band).
# Units: arcsec for biases / RMS, arcsec² for residual variances and the
# RA×Dec residual covariance; RA quantities are in the cos(dec)-corrected
# frame. Do not change unilaterally.
BIAS_TABLE_SCHEMA = pa.schema(
    [
        pa.field("obs_code", pa.large_string(), nullable=False),
        pa.field("band", pa.large_string(), nullable=True),
        pa.field("n_obs", pa.int64()),
        pa.field("n_objects", pa.int64()),
        pa.field("bias_ra_arcsec", pa.float64()),
        pa.field("bias_ra_ci_low", pa.float64()),
        pa.field("bias_ra_ci_high", pa.float64()),
        pa.field("bias_dec_arcsec", pa.float64()),
        pa.field("bias_dec_ci_low", pa.float64()),
        pa.field("bias_dec_ci_high", pa.float64()),
        pa.field("resid_var_ra", pa.float64()),
        pa.field("resid_var_dec", pa.float64()),
        pa.field("resid_cov_ra_dec", pa.float64()),
        pa.field("resid_cov_n", pa.int64()),
        pa.field("rms_ra_arcsec", pa.float64()),
        pa.field("rms_dec_arcsec", pa.float64()),
        pa.field("chi2_per_obs", pa.float64()),
        pa.field("bias_significant", pa.bool_()),
        pa.field("high_confidence", pa.bool_()),
        pa.field("confidence_score", pa.float64()),
    ]
)

# Columns that must be non-negative wherever they are non-null.
_NON_NEGATIVE_COLUMNS = (
    "n_obs",
    "n_objects",
    "resid_var_ra",
    "resid_var_dec",
    "resid_cov_n",
    "rms_ra_arcsec",
    "rms_dec_arcsec",
    "chi2_per_obs",
)


def validate_bias_table(table: pa.Table) -> pa.Table:
    """
    Validate a bias table against `BIAS_TABLE_SCHEMA`.

    Checks that all schema columns are present, casts them to the canonical
    types (extra columns are dropped), and checks basic value sanity
    (non-null obs_code, non-negative counts / variances / RMS values).

    Parameters
    ----------
    table : `pyarrow.Table`
        Bias table to validate.

    Returns
    -------
    table : `pyarrow.Table`
        The validated table, with columns selected and cast to
        `BIAS_TABLE_SCHEMA`.

    Raises
    ------
    ValueError
        If required columns are missing, a column cannot be cast to its
        canonical type, obs_code contains nulls, or a non-negative column
        contains negative values.
    """
    missing = [
        name for name in BIAS_TABLE_SCHEMA.names if name not in table.column_names
    ]
    if missing:
        raise ValueError(
            f"Bias table is missing required columns: {missing}. "
            f"Expected columns: {BIAS_TABLE_SCHEMA.names}"
        )

    table = table.select(BIAS_TABLE_SCHEMA.names)
    try:
        table = table.cast(BIAS_TABLE_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
        raise ValueError(
            f"Bias table columns could not be cast to the standard schema "
            f"(note: counts such as resid_cov_n must be integral, with "
            f"missing values encoded as null rather than NaN): {e}"
        ) from e

    if table["obs_code"].null_count > 0:
        raise ValueError("Bias table column 'obs_code' must not contain nulls")

    for name in _NON_NEGATIVE_COLUMNS:
        values = table[name].to_numpy(zero_copy_only=False).astype(np.float64)
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f"Bias table column '{name}' contains negative values")

    return table


def assert_positions_unchanged(
    before: OrbitDeterminationObservations,
    after: OrbitDeterminationObservations,
) -> None:
    """
    Assert that a model application did not modify observed positions.

    Checks that observation ids, times, and observed lon (RA) / lat (Dec)
    are identical between the two sets of observations. This is an opt-in
    helper for tests of position-preserving models; it is deliberately not
    enforced by `ObservationUncertaintyModel` itself.

    Parameters
    ----------
    before : `OrbitDeterminationObservations`
        Observations before the model was applied.
    after : `OrbitDeterminationObservations`
        Observations after the model was applied.

    Raises
    ------
    AssertionError
        If the number of observations, their ids, times, or observed
        positions differ.
    """
    if len(before) != len(after):
        raise AssertionError(
            f"Number of observations changed: {len(before)} -> {len(after)}"
        )
    if not before.id.equals(after.id):
        raise AssertionError("Observation ids changed")
    if not before.coordinates.time.table.equals(after.coordinates.time.table):
        raise AssertionError("Observation times changed")
    for axis in ("lon", "lat"):
        before_values = before.coordinates.table[axis].to_numpy(zero_copy_only=False)
        after_values = after.coordinates.table[axis].to_numpy(zero_copy_only=False)
        if not np.array_equal(before_values, after_values, equal_nan=True):
            raise AssertionError(f"Observed positions changed on axis '{axis}'")


class ObservationUncertaintyModel(ABC):
    """
    Abstract interface for models that transform observation uncertainties.

    A model takes `OrbitDeterminationObservations` and returns a new
    `OrbitDeterminationObservations` with transformed uncertainties, to be
    applied before orbit fitting (e.g.
    ``fitter.full_od(object_id, model.apply(observations), propagator)``).
    """

    @abstractmethod
    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        """
        Apply this model to a set of observations.

        Parameters
        ----------
        observations : `OrbitDeterminationObservations`
            Observations to transform.

        Returns
        -------
        observations : `OrbitDeterminationObservations`
            Transformed observations.
        """
        ...


class IdentityModel(ObservationUncertaintyModel):
    """
    No-op model: observations are returned unchanged.

    This is the naive baseline. A bias table may optionally be supplied
    (and is validated) so that the constructor signature is interchangeable
    with the table-driven models in variant sweeps, but its numbers are
    never used.
    """

    def __init__(self, bias_table: Optional[pa.Table] = None) -> None:
        if bias_table is not None:
            validate_bias_table(bias_table)

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        return observations


class _BiasTableModel(ObservationUncertaintyModel):
    """
    Shared machinery for bias-table-driven models.

    Rows are keyed by (obs_code, band); a station+band lookup falls back to
    the station rollup row (band null) when no band-specific row exists.
    An observation passes through with its baseline covariance unchanged
    when its station is absent from the table or the station's residual
    sample size ``resid_cov_n`` is null or below ``min_resid_cov_n``.
    """

    # Float-valued bias-table columns cached per row for fast lookup.
    _ROW_COLUMNS = (
        "bias_ra_arcsec",
        "bias_dec_arcsec",
        "resid_var_ra",
        "resid_var_dec",
        "resid_cov_ra_dec",
        "resid_cov_n",
        "chi2_per_obs",
    )

    def __init__(self, bias_table: pa.Table, min_resid_cov_n: int = 30) -> None:
        self.bias_table = validate_bias_table(bias_table)
        self.min_resid_cov_n = min_resid_cov_n

        codes = self.bias_table["obs_code"].to_pylist()
        bands = self.bias_table["band"].to_pylist()
        columns = {
            name: self.bias_table[name].to_pylist() for name in self._ROW_COLUMNS
        }
        self._rows: dict[tuple[str, Optional[str]], dict[str, float]] = {}
        for i, (code, band) in enumerate(zip(codes, bands)):
            key = (code, band)
            if key in self._rows:
                raise ValueError(
                    f"Bias table contains duplicate rows for "
                    f"(obs_code, band) = {key}"
                )
            self._rows[key] = {
                name: float(values[i]) if values[i] is not None else np.nan
                for name, values in columns.items()
            }

    def _lookup(self, code: str, band: Optional[str]) -> Optional[dict[str, float]]:
        if band is not None:
            row = self._rows.get((code, band))
            if row is not None:
                return row
        return self._rows.get((code, None))

    def _updated_block(
        self,
        row: dict[str, float],
        var_lon: float,
        var_lat: float,
        cov_lonlat: float,
        cos_dec: float,
    ) -> Optional[tuple[float, float, float]]:
        """
        Compute the updated (var_lon, var_lat, cov_lonlat) covariance block
        for one observation, in deg² with lon NOT cos(dec)-corrected.
        Return None to pass the observation through unchanged.
        """
        raise NotImplementedError  # pragma: no cover

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        if len(observations) == 0:
            return observations

        covariances = observations.coordinates.covariance.to_matrix()
        new_covariances = covariances.copy()
        lat = observations.coordinates.lat.to_numpy(zero_copy_only=False)
        cos_dec = np.cos(np.deg2rad(lat))
        codes = observations.observers.code.to_pylist()
        bands = observations.photometry.band.to_pylist()

        for i, (code, band) in enumerate(zip(codes, bands)):
            row = self._lookup(code, band)
            if row is None:
                continue
            resid_cov_n = row["resid_cov_n"]
            if not np.isfinite(resid_cov_n) or resid_cov_n < self.min_resid_cov_n:
                continue
            block = self._updated_block(
                row,
                var_lon=float(covariances[i, 1, 1]),
                var_lat=float(covariances[i, 2, 2]),
                cov_lonlat=float(covariances[i, 1, 2]),
                cos_dec=float(cos_dec[i]),
            )
            if block is None:
                continue
            var_lon, var_lat, cov_lonlat = block
            new_covariances[i, 1, 1] = var_lon
            new_covariances[i, 2, 2] = var_lat
            new_covariances[i, 1, 2] = cov_lonlat
            new_covariances[i, 2, 1] = cov_lonlat

        if np.array_equal(new_covariances, covariances, equal_nan=True):
            return observations
        return observations.set_column(
            "coordinates.covariance",
            CoordinateCovariances.from_matrix(new_covariances),
        )


class EmpiricalCovarianceModel(_BiasTableModel):
    """
    Inflate the RA/Dec covariance block with the station's measured 2×2
    residual covariance (flagship, calibrated model).

    In the bias table's cos(dec)-corrected arcsec² frame::

        C_used[ra, ra]   = C_base[ra, ra]   + resid_var_ra
        C_used[dec, dec] = C_base[dec, dec] + resid_var_dec
        C_used[ra, dec]  = C_base[ra, dec]  + resid_cov_ra_dec

    With ``mode='replace'`` the block is set to the measured covariance
    instead of added to the baseline. A non-finite baseline cross-term is
    treated as 0 when adding (matching the downstream convention that
    missing off-diagonal terms are zero); non-finite baseline variances
    propagate as NaN. Observations whose station lacks finite residual
    covariance values, or with degenerate cos(dec), pass through unchanged.
    """

    def __init__(
        self,
        bias_table: pa.Table,
        mode: Literal["add", "replace"] = "add",
        min_resid_cov_n: int = 30,
    ) -> None:
        if mode not in ("add", "replace"):
            raise ValueError(f"mode must be 'add' or 'replace', got {mode!r}")
        super().__init__(bias_table, min_resid_cov_n=min_resid_cov_n)
        self.mode = mode

    def _updated_block(
        self,
        row: dict[str, float],
        var_lon: float,
        var_lat: float,
        cov_lonlat: float,
        cos_dec: float,
    ) -> Optional[tuple[float, float, float]]:
        resid_var_ra = row["resid_var_ra"]
        resid_var_dec = row["resid_var_dec"]
        resid_cov = row["resid_cov_ra_dec"]
        if not (
            np.isfinite(resid_var_ra)
            and np.isfinite(resid_var_dec)
            and np.isfinite(resid_cov)
        ):
            return None
        if not np.isfinite(cos_dec) or cos_dec <= 0.0:
            return None

        # arcsec² (cos(dec)-corrected RA) -> deg² (lon not cos(dec)-corrected):
        # variances divide by cos²(dec) on the RA axis, the cross-term by
        # cos(dec) once, the Dec axis converts units only.
        add_var_lon = resid_var_ra / (ARCSEC_PER_DEG**2 * cos_dec**2)
        add_var_lat = resid_var_dec / ARCSEC_PER_DEG**2
        add_cov = resid_cov / (ARCSEC_PER_DEG**2 * cos_dec)

        if self.mode == "replace":
            return (add_var_lon, add_var_lat, add_cov)

        base_cov = cov_lonlat if np.isfinite(cov_lonlat) else 0.0
        return (var_lon + add_var_lon, var_lat + add_var_lat, base_cov + add_cov)


class PerformanceWeightedModel(_BiasTableModel):
    """
    Scale each axis sigma by sqrt(max(chi2_per_obs, 1.0)) for the
    observation's station (aggressive model).

    Equivalently, the RA/Dec covariance block is multiplied by
    max(chi2_per_obs, 1.0). The scaling is frame-independent, so no
    cos(dec) or unit conversion is involved. Stations with chi2_per_obs
    <= 1 or non-finite pass through unchanged.
    """

    def _updated_block(
        self,
        row: dict[str, float],
        var_lon: float,
        var_lat: float,
        cov_lonlat: float,
        cos_dec: float,
    ) -> Optional[tuple[float, float, float]]:
        chi2_per_obs = row["chi2_per_obs"]
        if not np.isfinite(chi2_per_obs) or chi2_per_obs <= 1.0:
            return None
        factor_sq = chi2_per_obs
        return (var_lon * factor_sq, var_lat * factor_sq, cov_lonlat * factor_sq)


class SigmaFloorModel(_BiasTableModel):
    """
    Floor each axis sigma at the magnitude of the station's measured bias:
    sigma_used = max(sigma_baseline, |bias|) per axis (reference model).

    The RA floor |bias_ra_arcsec| is defined in the cos(dec)-corrected
    frame and is converted onto the lon axis by dividing by cos(dec); if
    cos(dec) is degenerate the RA floor is skipped while the Dec floor
    still applies. Axes with a non-finite baseline variance or a non-finite
    bias pass through unchanged. The cross-term is left unchanged.
    """

    def _updated_block(
        self,
        row: dict[str, float],
        var_lon: float,
        var_lat: float,
        cov_lonlat: float,
        cos_dec: float,
    ) -> Optional[tuple[float, float, float]]:
        new_var_lon = var_lon
        new_var_lat = var_lat

        bias_ra = row["bias_ra_arcsec"]
        if (
            np.isfinite(bias_ra)
            and np.isfinite(var_lon)
            and np.isfinite(cos_dec)
            and cos_dec > 0.0
        ):
            floor_lon = (abs(bias_ra) / (ARCSEC_PER_DEG * cos_dec)) ** 2
            new_var_lon = max(var_lon, floor_lon)

        bias_dec = row["bias_dec_arcsec"]
        if np.isfinite(bias_dec) and np.isfinite(var_lat):
            floor_lat = (abs(bias_dec) / ARCSEC_PER_DEG) ** 2
            new_var_lat = max(var_lat, floor_lat)

        if new_var_lon == var_lon and new_var_lat == var_lat:
            return None
        return (new_var_lon, new_var_lat, cov_lonlat)
