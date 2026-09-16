"""
Veres, Farnocchia & Chesley (2017) per-observation astrometric uncertainties.

Veres, Farnocchia, Chesley & Chamberlin (2017, Icarus 296, 139; "VFC2017")
derived station- and catalog-dependent astrometric uncertainties for MPC
observations from the residual statistics of well-determined orbits. Agencies
use such tables as default weights for observations that report no
uncertainty, or as floors on the reported ones. This module ships the
per-(station, catalog) sigma table used by the Asteroid Institute's OD
experiments together with the interpreters that apply it (`VeresFloorModel`,
`VeresReplaceModel`, both `ObservationUncertaintyModel` subclasses), so that OD
runs using it are reproducible from adam_core alone. The table and the
interpreter arithmetic live in the Rust backend
(``adam_core_rs_coords::veres2017``).

Table provenance
----------------
`veres2017_sigma_table` is the working table maintained in
``adam_orbit_det_eval`` (``VERES2017_CATALOG_DEFAULTS`` +
``VERES2017_STN_CATALOG_OVERRIDES``, fallback 0.75"), transcribed verbatim.
It is a catalog-level summary in the spirit of VFC2017's Table 1 plus a few
station-specific overrides; it is NOT a transcription of the paper's full
station table. Verify against the publication (or supply your own table via
``sigma_table``) before citing results that depend on the numbers.

Frames and units
----------------
Sigmas are in arcseconds with the RA axis in the cos(dec)-corrected frame
(MPC/ADES ``rmsRACosDec`` convention). `SphericalCoordinates` covariances are
in degrees² with lon NOT cos(dec)-corrected, so the RA sigma is divided by
cos(dec) once (in addition to the arcsec -> degree scaling) before squaring.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from ..coordinates.covariances import CoordinateCovariances
from .evaluate import OrbitDeterminationObservations
from .observation_uncertainty import ObservationUncertaintyModel

__all__ = [
    "VERES2017_FALLBACK_SIGMA_ARCSEC",
    "VERES2017_SIGMA_TABLE_SCHEMA",
    "VeresFloorModel",
    "VeresReplaceModel",
    "VeresSigmaLookup",
    "validate_veres_sigma_table",
    "veres2017_sigma_table",
]

#: Standard schema for a Veres-style sigma table: one row per catalog default
#: (``obs_code`` null) or per (station, catalog) override. Sigmas in arcsec,
#: RA in the cos(dec)-corrected frame.
VERES2017_SIGMA_TABLE_SCHEMA = pa.schema(
    [
        pa.field("obs_code", pa.large_string(), nullable=True),
        pa.field("astcat", pa.large_string(), nullable=False),
        pa.field("sigma_ra_arcsec", pa.float64(), nullable=False),
        pa.field("sigma_dec_arcsec", pa.float64(), nullable=False),
    ]
)

#: Global fallback when neither a (station, catalog) nor a catalog row exists.
VERES2017_FALLBACK_SIGMA_ARCSEC = 0.75


def veres2017_sigma_table() -> pa.Table:
    """
    The bundled Veres-style sigma table (see module docstring for provenance).

    Returns
    -------
    table : `pyarrow.Table`
        Rows with ``obs_code`` null are per-catalog defaults; rows with an
        ``obs_code`` are (station, catalog) overrides. Schema
        `VERES2017_SIGMA_TABLE_SCHEMA`.
    """
    from adam_core import _rust_native

    obs_codes, astcats, sigma_ra, sigma_dec = (
        _rust_native.veres2017_sigma_table_columns()
    )
    return pa.table(
        {
            "obs_code": pa.array(obs_codes, pa.large_string()),
            "astcat": pa.array(astcats, pa.large_string()),
            "sigma_ra_arcsec": pa.array(sigma_ra, pa.float64()),
            "sigma_dec_arcsec": pa.array(sigma_dec, pa.float64()),
        },
        schema=VERES2017_SIGMA_TABLE_SCHEMA,
    )


def validate_veres_sigma_table(table: pa.Table) -> pa.Table:
    """
    Validate and cast a sigma table to `VERES2017_SIGMA_TABLE_SCHEMA`.

    Raises
    ------
    ValueError
        If required columns are missing, cannot be cast, ``astcat`` has
        nulls, or a sigma is not a positive finite number.
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
    if table["astcat"].null_count > 0:
        raise ValueError("Sigma table column 'astcat' must not contain nulls")
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

    Lookup order: the (station, catalog) override row, then the catalog
    default row (``obs_code`` null), then ``fallback_sigma_arcsec`` for both
    axes; None as fallback means "no sigma known" (the caller passes the
    observation through). The resolution runs in the Rust
    ``VeresSigmaLookup``; this object carries the validated table columns.

    Parameters
    ----------
    sigma_table : `pyarrow.Table`, optional
        Table with `VERES2017_SIGMA_TABLE_SCHEMA` columns. Default: the
        bundled `veres2017_sigma_table`.
    fallback_sigma_arcsec : float or None
        Global fallback sigma (both axes). Default 0.75".
    """

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
    ) -> None:
        table = validate_veres_sigma_table(
            sigma_table if sigma_table is not None else veres2017_sigma_table()
        )
        self.sigma_table = table
        self.fallback_sigma_arcsec = fallback_sigma_arcsec
        self._obs_codes: list[str | None] = table["obs_code"].to_pylist()
        self._astcats: list[str] = table["astcat"].to_pylist()
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
    `~adam_core.orbit_determination.veres2017.VeresSigmaLookup` (the bundled
    table by default) to per-axis sigmas in arcseconds with RA in the
    cos(dec)-corrected frame; subclasses decide how the resulting variances
    combine with the reported covariance. Observations that resolve to no
    sigma (unknown catalog with ``fallback_sigma_arcsec=None``) or with a
    degenerate cos(dec) pass through unchanged. Positions are never modified.
    """

    _MODEL: str = ""

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
    ) -> None:
        self.lookup = VeresSigmaLookup(sigma_table, fallback_sigma_arcsec)
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
        Sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`); default the bundled
        `veres2017_sigma_table`.
    fallback_sigma_arcsec : float or None
        Sigma used for (station, catalog) pairs absent from the table;
        None passes such observations through. Default 0.75".
    fill_missing : bool
        Also assign the VFC2017 sigma where the reported one is non-finite.
        Default False.
    """

    _MODEL = "floor"

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
        Sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`); default the bundled
        `veres2017_sigma_table`.
    fallback_sigma_arcsec : float or None
        Sigma used for (station, catalog) pairs absent from the table;
        None passes such observations through. Default 0.75".
    """

    _MODEL = "replace"
