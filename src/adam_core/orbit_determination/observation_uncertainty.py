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

The model arithmetic runs in the Rust backend
(``adam_core_rs_coords::observation_uncertainty``, one crossing per
``apply``); the classes here own the quivr tables, the bias-table schema
validation and the column plumbing.

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

The bias-table models and `NightBatchDeweightingModel` modify ONLY the
RA/Dec covariance block of the input observations; observed positions
(lon/lat) and every other column pass through unchanged. Use
`assert_positions_unchanged` in tests to prove that a given model is
position-preserving. This is a property of those implementations, not a
constraint of the interface: `apply` returns a full
`OrbitDeterminationObservations` so that models are free to implement other
transformations. `EFCC18DebiasModel` is the shipped exception: it SUBTRACTS
the EFCC18 star-catalog bias from the observed positions and leaves the
covariance untouched. That is standard astrometric debiasing (applied by
JPL, the MPC and OrbFit alike) and is distinct from the observatory
bias-table numbers, which were measured on top of catalog-debiased residuals
and are therefore only ever interpreted as covariance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from ..coordinates.covariances import CoordinateCovariances
from ..observations.efcc18 import (
    EFCC18_N_CATALOGS,
    EFCC18_N_TILES,
    load_efcc18_biases,
)
from .evaluate import OrbitDeterminationObservations

__all__ = [
    "BIAS_TABLE_SCHEMA",
    "ObservationUncertaintyModel",
    "IdentityModel",
    "EmpiricalCovarianceModel",
    "PerformanceWeightedModel",
    "SigmaFloorModel",
    "NightBatchDeweightingModel",
    "EFCC18DebiasModel",
    "CompositeModel",
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


def _float_column(table: pa.Table, name: str) -> npt.NDArray[np.float64]:
    """Bias-table column as contiguous float64 with nulls carried as NaN."""
    return np.ascontiguousarray(
        table[name].to_numpy(zero_copy_only=False), dtype=np.float64
    )


def _covariances(
    observations: OrbitDeterminationObservations,
) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(
        observations.coordinates.covariance.to_matrix(), dtype=np.float64
    )


def _latitudes(observations: OrbitDeterminationObservations) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(
        observations.coordinates.lat.to_numpy(zero_copy_only=False), dtype=np.float64
    )


def _with_covariances(
    observations: OrbitDeterminationObservations,
    covariances: Optional[npt.NDArray[np.float64]],
) -> OrbitDeterminationObservations:
    """Rebuild the covariance column from a Rust update; `None` = unchanged."""
    if covariances is None:
        return observations
    return observations.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(np.asarray(covariances, dtype=np.float64)),
    )


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
    sample size ``resid_cov_n`` is null or below ``min_resid_cov_n``. The
    per-row arithmetic is the Rust ``bias_table_model_apply`` kernel selected
    by ``_MODEL``.
    """

    _MODEL: str = ""

    # Float-valued bias-table columns handed to the Rust kernel per row.
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
        self.mode = "add"

        self._codes: list[str] = self.bias_table["obs_code"].to_pylist()
        self._bands: list[Optional[str]] = self.bias_table["band"].to_pylist()
        seen: set[tuple[str, Optional[str]]] = set()
        for key in zip(self._codes, self._bands):
            if key in seen:
                raise ValueError(
                    f"Bias table contains duplicate rows for "
                    f"(obs_code, band) = {key}"
                )
            seen.add(key)
        self._columns = {
            name: _float_column(self.bias_table, name) for name in self._ROW_COLUMNS
        }

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        if len(observations) == 0:
            return observations
        from adam_core import _rust_native

        updated = _rust_native.bias_table_model_apply_numpy(
            self._MODEL,
            self.mode,
            float(self.min_resid_cov_n),
            self._codes,
            self._bands,
            *(self._columns[name] for name in self._ROW_COLUMNS),
            _latitudes(observations),
            _covariances(observations),
            observations.observers.code.to_pylist(),
            observations.photometry.band.to_pylist(),
        )
        return _with_covariances(observations, updated)


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

    _MODEL = "empirical_covariance"

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


class PerformanceWeightedModel(_BiasTableModel):
    """
    Scale each axis sigma by sqrt(max(chi2_per_obs, 1.0)) for the
    observation's station (aggressive model).

    Equivalently, the RA/Dec covariance block is multiplied by
    max(chi2_per_obs, 1.0). The scaling is frame-independent, so no
    cos(dec) or unit conversion is involved. Stations with chi2_per_obs
    <= 1 or non-finite pass through unchanged.
    """

    _MODEL = "performance_weighted"


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

    _MODEL = "sigma_floor"


class NightBatchDeweightingModel(ObservationUncertaintyModel):
    """
    Deweight same-station, same-night batches of observations following
    Veres, Farnocchia & Chesley (2017): when a station contributes N > cap
    observations on one night, each of those observations has its RA/Dec
    covariance block scaled by N / cap, capping the effective statistical
    weight of the night-batch at ~cap independent observations. This
    accounts for error correlations within a night (shared calibration,
    atmosphere, and reduction) that would otherwise let large batches
    dominate a fit.

    A "night" is approximated by floor(UTC MJD). The MJD boundary falls at
    midnight UTC, so for stations whose local night straddles 0h UTC (e.g.
    European longitudes) a single observing night may be split into two
    batches; deweighting is then conservative (factors are underestimated,
    never overestimated).

    This model is bias-table-free and composes with the table-driven models
    via `CompositeModel`.
    """

    def __init__(self, cap: int = 4) -> None:
        if cap < 1:
            raise ValueError(f"cap must be a positive integer, got {cap}")
        self.cap = cap

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        if len(observations) == 0:
            return observations
        from adam_core import _rust_native

        mjd_utc = np.ascontiguousarray(
            observations.coordinates.time.rescale("utc")
            .mjd()
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        updated = _rust_native.night_batch_deweighting_model_apply_numpy(
            self.cap,
            _covariances(observations),
            observations.observers.code.to_pylist(),
            mjd_utc,
        )
        return _with_covariances(observations, updated)


class EFCC18DebiasModel(ObservationUncertaintyModel):
    """
    Subtract the EFCC18 star-catalog bias from observed positions.

    For each observation whose ``astcat`` is covered by EFCC18 (see
    `~adam_core.observations.efcc18.MPC_ASTCAT_TO_EFCC18`), the tabulated
    position + proper-motion correction for its HEALPix tile, catalog and
    epoch is looked up and subtracted from the observed RA and Dec::

        lon_used = lon - bias_ra_arcsec  / (3600 cos(dec))   (wrapped to [0, 360))
        lat_used = lat - bias_dec_arcsec / 3600

    THIS MODEL MODIFIES POSITIONS. That is intended: EFCC18 debiasing is the
    standard correction of catalog-induced systematic errors applied by JPL,
    the MPC and OrbFit before fitting, and the observatory bias-table numbers
    interpreted by the covariance models in this module were measured on top
    of it. The covariance block is left exactly as supplied. Do not test this
    model with `assert_positions_unchanged`; test that the covariance is
    unchanged and that positions move by the expected correction instead.

    Observations pass through unchanged when their ``astcat`` is null, not in
    the EFCC18 map, or listed in ``exclude_astcats``, and when the position is
    non-finite or at a pole (where the RA correction is undefined). If no
    observation is corrected the input object is returned as is.

    Parameters
    ----------
    bias_table : `numpy.ndarray` (49152, 26, 4), optional
        Pre-loaded EFCC18 table (see
        `~adam_core.observations.efcc18.load_efcc18_biases`). When None the
        table is loaded from ``bias_dat``, the environment, the installed
        ``jpl-debias-2018`` data package, or the cache, in that order.
    bias_dat : str or Path, optional
        Location of ``bias.dat``; only used when ``bias_table`` is None.
    exclude_astcats : iterable of str
        MPC ``astCat`` codes to leave uncorrected even though tabulated, e.g.
        `~adam_core.observations.efcc18.EFCC18_JPL_UNDEBIASED_ASTCATS` to
        follow JPL's practice of not debiasing Gaia-DR1, ACT, Tycho-2 and
        UCAC-5. Default: correct every tabulated catalog.
    """

    def __init__(
        self,
        bias_table: Optional[npt.NDArray[np.floating]] = None,
        bias_dat: Optional[Union[str, Path]] = None,
        exclude_astcats: Iterable[str] = (),
    ) -> None:
        if bias_table is None:
            bias_table = load_efcc18_biases(bias_dat)
        elif bias_dat is not None:
            raise ValueError("Pass either bias_table or bias_dat, not both")
        expected_shape = (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)
        if bias_table.shape != expected_shape:
            raise ValueError(
                f"bias_table has shape {bias_table.shape}, expected {expected_shape}"
            )
        self.bias_table = bias_table
        self.exclude_astcats = tuple(exclude_astcats)
        self._bias_table_f32 = np.ascontiguousarray(bias_table, dtype=np.float32)

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        if len(observations) == 0:
            return observations
        from adam_core import _rust_native

        jd_tdb = np.ascontiguousarray(
            observations.coordinates.time.rescale("tdb")
            .jd()
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        updated = _rust_native.efcc18_debias_model_apply_numpy(
            self._bias_table_f32,
            list(self.exclude_astcats),
            np.ascontiguousarray(
                observations.coordinates.lon.to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ),
            _latitudes(observations),
            observations.astcat.to_pylist(),
            jd_tdb,
        )
        if updated is None:
            return observations
        new_lon, new_lat = updated
        return observations.set_column(
            "coordinates.lon", pa.array(np.asarray(new_lon), type=pa.float64())
        ).set_column(
            "coordinates.lat", pa.array(np.asarray(new_lat), type=pa.float64())
        )


class CompositeModel(ObservationUncertaintyModel):
    """
    Apply a sequence of uncertainty models in order (left to right).

    Because the shipped models act multiplicatively or additively on the
    covariance block, order can matter: e.g.
    ``CompositeModel(EmpiricalCovarianceModel(table), NightBatchDeweightingModel())``
    deweights night-batches of the bias-inflated covariances.
    """

    def __init__(self, *models: ObservationUncertaintyModel) -> None:
        if not models:
            raise ValueError("CompositeModel requires at least one model")
        for model in models:
            if not isinstance(model, ObservationUncertaintyModel):
                raise TypeError(
                    f"CompositeModel components must be "
                    f"ObservationUncertaintyModel instances, got {type(model)}"
                )
        self.models = tuple(models)

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        for model in self.models:
            observations = model.apply(observations)
        return observations
