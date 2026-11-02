"""
Veres, Farnocchia & Chesley (2017) per-observation astrometric uncertainties.

Veres, Farnocchia, Chesley & Chamberlin (2017, Icarus 296, 139; "VFC2017")
derived station- and catalog-dependent astrometric uncertainties for MPC
observations from the residual statistics of well-determined orbits. Agencies
use such tables as default weights for observations that report no
uncertainty, or as floors on the reported ones. This module ships the
interpreters that apply such a table (`VeresFloorModel`, `VeresReplaceModel`,
`SigmaFillModel`, all `ObservationUncertaintyModel` subclasses) and the
generic `VeresSigmaLookup`; the interpreter arithmetic lives in the Rust
backend (``adam_core_rs_coords::veres2017``).

The table schema and `VeresSigmaLookup` are GENERIC: any station/catalog
sigma table in `VERES2017_SIGMA_TABLE_SCHEMA` can be supplied, including
station-only rows (``astcat`` null) and one global row (both keys null).

Tables
------
adam_core bundles NO sigma table. The tables are data shipped by the private
``adam-observatory-uncertainties`` package (import name
``observatory_uncertainties``), which adam_core never depends on: a model
constructed without ``sigma_table`` resolves its default table by importing
that package (`load_sigma_table`), the same soft-import pattern
`~adam_core.observations.efcc18` uses for the ``jpl_debias_2018`` data
package, and raises an `ImportError` naming the package when it is not
installed. Defaults (decision 2026-09-23): `SigmaFillModel` reads
``v2_sigma_fill`` (v2 LOOO study RMS per station x catalog for
high-confidence stations, with station / catalog / global fallbacks);
`VeresFloorModel` and `VeresReplaceModel` read ``veres2017_working`` (the
VFC2017-style working table the OD experiments used before, 34 catalog
defaults plus 14 station overrides, NOT a transcription of the paper's full
station table). Verify against the publication, or supply your own table via
``sigma_table``, before citing results that depend on the numbers.

Frames and units
----------------
Sigmas are in arcseconds with the RA axis in the cos(dec)-corrected frame
(MPC/ADES ``rmsRACosDec`` convention). `SphericalCoordinates` covariances are
in degrees² with lon NOT cos(dec)-corrected, so the RA sigma is divided by
cos(dec) once (in addition to the arcsec -> degree scaling) before squaring.
"""

from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from ..coordinates.covariances import CoordinateCovariances
from .evaluate import OrbitDeterminationObservations
from .observation_uncertainty import ObservationUncertaintyModel

__all__ = [
    "SIGMA_TABLE_PACKAGE",
    "V2_SIGMA_FILL_TABLE",
    "VERES2017_FALLBACK_SIGMA_ARCSEC",
    "VERES2017_SIGMA_TABLE_SCHEMA",
    "VERES2017_WORKING_TABLE",
    "SigmaFillModel",
    "VeresFloorModel",
    "VeresReplaceModel",
    "VeresSigmaLookup",
    "load_sigma_table",
    "validate_veres_sigma_table",
]

#: Standard schema for a station/catalog sigma table: rows are (station,
#: catalog), station-only (``astcat`` null), catalog default (``obs_code``
#: null) or one global row (both null). Sigmas in arcsec, RA in the
#: cos(dec)-corrected frame.
VERES2017_SIGMA_TABLE_SCHEMA = pa.schema(
    [
        pa.field("obs_code", pa.large_string(), nullable=True),
        pa.field("astcat", pa.large_string(), nullable=True),
        pa.field("sigma_ra_arcsec", pa.float64(), nullable=False),
        pa.field("sigma_dec_arcsec", pa.float64(), nullable=False),
    ]
)

#: Global fallback when neither a (station, catalog) nor a catalog row exists.
VERES2017_FALLBACK_SIGMA_ARCSEC = 0.75


#: Import name of the private ``adam-observatory-uncertainties`` data package
#: that ships the sigma tables. adam_core does not depend on it; it is
#: imported on demand when a model is constructed without ``sigma_table``.
SIGMA_TABLE_PACKAGE = "observatory_uncertainties"
#: Package table: the Asteroid Institute default fill-in table (v2 LOOO RMS).
V2_SIGMA_FILL_TABLE = "v2_sigma_fill"
#: Package table: the legacy VFC2017-style working table.
VERES2017_WORKING_TABLE = "veres2017_working"

_PACKAGE_INSTALL_HINT = (
    'pip install "git+ssh://git@github.com/B612-Asteroid-Institute/'
    'adam-observatory-uncertainties.git@v0.3.0"'
)


def load_sigma_table(name: str = V2_SIGMA_FILL_TABLE) -> pa.Table:
    """
    Load a per-observation sigma table from the ``adam-observatory-uncertainties``
    data package.

    Parameters
    ----------
    name : str
        Table name in the package (`V2_SIGMA_FILL_TABLE` or
        `VERES2017_WORKING_TABLE`, or any name listed by the package's
        ``list_sigma_tables``).

    Returns
    -------
    table : `pyarrow.Table`
        The table validated and cast to `VERES2017_SIGMA_TABLE_SCHEMA`
        (extra package columns such as ``level`` are dropped).

    Raises
    ------
    ImportError
        If the data package is not installed. adam_core does not depend on
        it; install it or pass an explicit ``sigma_table`` to the model.
    """
    try:
        package = importlib.import_module(SIGMA_TABLE_PACKAGE)
    except ImportError as e:
        raise ImportError(
            f"The sigma table {name!r} ships with the private "
            f"adam-observatory-uncertainties data package "
            f"(import name {SIGMA_TABLE_PACKAGE!r}), which is not installed. "
            f"Install it ({_PACKAGE_INSTALL_HINT}) or pass an explicit "
            "sigma_table."
        ) from e
    return validate_veres_sigma_table(package.load_sigma_table(name))


def validate_veres_sigma_table(table: pa.Table) -> pa.Table:
    """
    Validate and cast a sigma table to `VERES2017_SIGMA_TABLE_SCHEMA`.

    Raises
    ------
    ValueError
        If required columns are missing, cannot be cast, more than one row
        has both keys null (global row), or a sigma is not a positive finite
        number. Rows may be (station, catalog), station-only (``astcat``
        null), catalog-only (``obs_code`` null) or global (both null).
    """
    missing = [
        name
        for name in VERES2017_SIGMA_TABLE_SCHEMA.names
        if name not in table.column_names
    ]
    if missing:
        raise ValueError(
            f"Sigma table is missing required columns: {missing}. "
            f"Expected columns: {VERES2017_SIGMA_TABLE_SCHEMA.names}"
        )
    table = table.select(VERES2017_SIGMA_TABLE_SCHEMA.names)
    try:
        table = table.cast(VERES2017_SIGMA_TABLE_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
        raise ValueError(
            f"Sigma table columns could not be cast to the standard schema: {e}"
        ) from e
    both_null = sum(
        1
        for code, astcat in zip(
            table["obs_code"].to_pylist(), table["astcat"].to_pylist()
        )
        if code is None and astcat is None
    )
    if both_null > 1:
        raise ValueError(
            "Sigma table may contain at most one global row (obs_code and astcat null)"
        )
    for name in ("sigma_ra_arcsec", "sigma_dec_arcsec"):
        values = table[name].to_pylist()
        if any(v is None or not (v > 0.0) or v == float("inf") for v in values):
            raise ValueError(
                f"Sigma table column '{name}' must contain positive finite values"
            )
    return table


class VeresSigmaLookup:
    """
    Resolve (station, catalog) to (sigma_ra_arcsec, sigma_dec_arcsec).

    Lookup order: the (station, catalog) row, then the station row
    (``astcat`` null), then the catalog default row (``obs_code`` null),
    then the table's global row (both keys null) if present, then
    ``fallback_sigma_arcsec`` for both axes; None as fallback means "no sigma
    known" (the caller passes the observation through). The resolution runs
    in the Rust ``VeresSigmaLookup``; this object carries the validated table
    columns.

    Parameters
    ----------
    sigma_table : `pyarrow.Table`, optional
        Table with `VERES2017_SIGMA_TABLE_SCHEMA` columns. Default: the
        ``default_table`` of the ``adam-observatory-uncertainties`` data
        package (`load_sigma_table`).
    fallback_sigma_arcsec : float or None
        Global fallback sigma (both axes). Default 0.75".
    default_table : str
        Package table loaded when ``sigma_table`` is None. Default
        `V2_SIGMA_FILL_TABLE`.
    """

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
        default_table: str = V2_SIGMA_FILL_TABLE,
    ) -> None:
        table = (
            validate_veres_sigma_table(sigma_table)
            if sigma_table is not None
            else load_sigma_table(default_table)
        )
        self.sigma_table = table
        self.fallback_sigma_arcsec = fallback_sigma_arcsec
        self._obs_codes: list[str | None] = table["obs_code"].to_pylist()
        self._astcats: list[str | None] = table["astcat"].to_pylist()
        self._sigma_ra = np.ascontiguousarray(
            table["sigma_ra_arcsec"].to_numpy(zero_copy_only=False), dtype=np.float64
        )
        self._sigma_dec = np.ascontiguousarray(
            table["sigma_dec_arcsec"].to_numpy(zero_copy_only=False), dtype=np.float64
        )
        # Build the Rust lookup once so duplicate rows are rejected up front.
        self.sigmas_for([], [])

    def _table_arguments(self) -> tuple:
        return (
            self._obs_codes,
            self._astcats,
            self._sigma_ra,
            self._sigma_dec,
            self.fallback_sigma_arcsec,
        )

    def sigmas_for(
        self, obs_codes: list[str | None], astcats: list[str | None]
    ) -> npt.NDArray[np.float64]:
        """
        ``(N, 2)`` ``(sigma_ra_arcsec, sigma_dec_arcsec)`` per (station,
        catalog) pair, NaN where no sigma is known.
        """
        from adam_core import _rust_native

        return np.asarray(
            _rust_native.veres_sigma_lookup_numpy(
                *self._table_arguments(), list(obs_codes), list(astcats)
            ),
            dtype=np.float64,
        )

    def sigmas(
        self, obs_code: str | None, astcat: str | None
    ) -> tuple[float, float] | None:
        """(sigma_ra_arcsec, sigma_dec_arcsec) for the observation, or None."""
        sigma_ra, sigma_dec = self.sigmas_for([obs_code], [astcat])[0]
        if np.isnan(sigma_ra):
            return None
        return (float(sigma_ra), float(sigma_dec))


class _VeresSigmaModel(ObservationUncertaintyModel):
    """
    Shared machinery for the VFC2017 station/catalog sigma interpreters.

    Each observation's (station, star catalog) is resolved through a
    `~adam_core.orbit_determination.veres2017.VeresSigmaLookup` (the
    subclass's ``_DEFAULT_TABLE`` of the data package by default) to
    per-axis sigmas in arcseconds with RA in the
    cos(dec)-corrected frame; subclasses decide how the resulting variances
    combine with the reported covariance. Observations that resolve to no
    sigma (unknown station and catalog, no global row,
    ``fallback_sigma_arcsec=None``) or with a degenerate cos(dec) pass
    through unchanged. Positions are never modified.
    """

    _MODEL: str = ""
    _DEFAULT_TABLE: str = V2_SIGMA_FILL_TABLE

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
    ) -> None:
        self.lookup = VeresSigmaLookup(
            sigma_table, fallback_sigma_arcsec, default_table=self._DEFAULT_TABLE
        )
        self.fill_missing = False

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        if len(observations) == 0:
            return observations
        from adam_core import _rust_native

        updated = _rust_native.veres_model_apply_numpy(
            self._MODEL,
            self.fill_missing,
            *self.lookup._table_arguments(),
            np.ascontiguousarray(
                observations.coordinates.lat.to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                observations.coordinates.covariance.to_matrix(), dtype=np.float64
            ),
            observations.observers.code.to_pylist(),
            observations.astcat.to_pylist(),
        )
        if updated is None:
            return observations
        return observations.set_column(
            "coordinates.covariance",
            CoordinateCovariances.from_matrix(np.asarray(updated, dtype=np.float64)),
        )

    def _native_specs(self) -> list[dict]:
        obs_codes, astcats, sigma_ra, sigma_dec, fallback = (
            self.lookup._table_arguments()
        )
        return [
            {
                "model": "veres",
                "kind": self._MODEL,
                "fill_missing": bool(self.fill_missing),
                "table_obs_code": obs_codes,
                "table_astcat": astcats,
                "sigma_ra_arcsec": sigma_ra,
                "sigma_dec_arcsec": sigma_dec,
                "fallback_sigma_arcsec": fallback,
            }
        ]


class VeresFloorModel(_VeresSigmaModel):
    """
    Floor each axis sigma at the VFC2017 station/catalog sigma:
    ``sigma_used = max(sigma_reported, sigma_VFC2017)`` per axis (agency
    floor practice). The RA/Dec cross-term is left unchanged.

    A non-finite reported variance is left as is unless ``fill_missing`` is
    True, in which case it is replaced by the VFC2017 variance (the
    "default uncertainty when none is reported" use of the table).

    Parameters
    ----------
    sigma_table : `pyarrow.Table`, optional
        Sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`); default the
        ``veres2017_working`` table of the ``adam-observatory-uncertainties``
        data package (`load_sigma_table`).
    fallback_sigma_arcsec : float or None
        Sigma used for (station, catalog) pairs absent from the table;
        None passes such observations through. Default 0.75".
    fill_missing : bool
        Also assign the VFC2017 sigma where the reported one is non-finite.
        Default False.
    """

    _MODEL = "floor"
    _DEFAULT_TABLE = VERES2017_WORKING_TABLE

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
        fill_missing: bool = False,
    ) -> None:
        super().__init__(sigma_table, fallback_sigma_arcsec)
        self.fill_missing = fill_missing


class VeresReplaceModel(_VeresSigmaModel):
    """
    Replace each axis sigma by the VFC2017 station/catalog sigma outright
    (classic weighting-file practice, ignoring reported uncertainties). The
    RA/Dec cross-term is set to zero since the table carries no correlation.

    Parameters
    ----------
    sigma_table : `pyarrow.Table`, optional
        Sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`); default the
        ``veres2017_working`` table of the ``adam-observatory-uncertainties``
        data package (`load_sigma_table`).
    fallback_sigma_arcsec : float or None
        Sigma used for (station, catalog) pairs absent from the table;
        None passes such observations through. Default 0.75".
    """

    _MODEL = "replace"
    _DEFAULT_TABLE = VERES2017_WORKING_TABLE


class SigmaFillModel(_VeresSigmaModel):
    """
    Fill ONLY the per-axis variances that are missing (non-finite or
    non-positive) from the station/catalog sigma table; every reported
    sigma is left exactly as reported. This is the "default uncertainty
    when none is reported" use of such tables, isolated from the floor /
    replace semantics of `VeresFloorModel` / `VeresReplaceModel`.

    Where an axis is filled the RA/Dec cross-term is set to zero (the table
    carries no correlation and a reported correlation without a reported
    sigma is meaningless). Observations that resolve to no sigma (unknown
    station and catalog, no global row, ``fallback_sigma_arcsec=None``) pass
    through with their NaN variances, so the caller can detect them.

    The Asteroid Institute default is this model with the ``v2_sigma_fill``
    table (v2 LOOO study RMS per station x catalog, high-confidence stations,
    with station / catalog / global fallbacks) from the private
    ``adam-observatory-uncertainties`` package, which this model loads by
    default; that package's ``veres2017_working`` table is the legacy
    reference (decision 2026-09-23, adam_od_experiments bead d2f: orbits
    unchanged, prediction covariance x0.82, better-calibrated noise model at
    32 of 39 stations).

    Parameters
    ----------
    sigma_table : `pyarrow.Table`, optional
        Sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`; station-only and global
        rows allowed); default the ``v2_sigma_fill`` table of the data
        package (`load_sigma_table`).
    fallback_sigma_arcsec : float or None
        Sigma used when nothing in the table matches; None leaves the
        observation unfilled. Default 0.75".
    """

    _MODEL = "fill"
    _DEFAULT_TABLE = V2_SIGMA_FILL_TABLE
