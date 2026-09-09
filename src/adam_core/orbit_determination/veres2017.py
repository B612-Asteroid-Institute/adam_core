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
runs using it are reproducible from adam_core alone.

Table provenance
----------------
`VERES2017_SIGMA_TABLE` is the working table maintained in
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
import pyarrow as pa

from ..coordinates.covariances import CoordinateCovariances
from .evaluate import OrbitDeterminationObservations
from .observation_uncertainty import ARCSEC_PER_DEG, ObservationUncertaintyModel

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

# Per-catalog defaults (sigma_ra_arcsec, sigma_dec_arcsec); see module docstring
# for provenance. Keys are MPC astCat codes as carried on
# OrbitDeterminationObservations.astcat.
_CATALOG_DEFAULTS: dict[str, tuple[float, float]] = {
    # Gaia family
    "Gaia3E": (0.15, 0.15),
    "Gaia3": (0.15, 0.15),
    "Gaia2": (0.18, 0.18),
    "Gaia1": (0.25, 0.25),
    # ATLAS family
    "ATLAS2": (0.20, 0.20),
    "ATLAS": (0.25, 0.25),
    # UCAC family
    "UCAC5": (0.25, 0.25),
    "SSTRC4": (0.25, 0.25),
    "UCAC4": (0.30, 0.30),
    "UCAC3": (0.30, 0.30),
    "UCAC2": (0.40, 0.40),
    "UCAC1": (0.50, 0.50),
    # 2MASS
    "2MASS": (0.20, 0.20),
    # USNO catalogs
    "USNOB1": (0.50, 0.50),
    "USNOA2": (0.60, 0.60),
    "USNOSA2": (0.60, 0.60),
    "USNOA1": (0.80, 0.80),
    # GSC family
    "GSC": (0.50, 0.50),
    "GSC1.1": (0.50, 0.50),
    "GSC1.2": (0.50, 0.50),
    "GSC2.2": (0.40, 0.40),
    "GSC2.3": (0.35, 0.35),
    "GSCACT": (0.50, 0.50),
    # PPMXL / PPM
    "PPMXL": (0.35, 0.35),
    "PPM": (0.50, 0.50),
    # Other catalogs
    "SDSS8": (0.20, 0.20),
    "SDSS7": (0.20, 0.20),
    "NOMAD": (0.40, 0.40),
    "CMC14": (0.35, 0.35),
    "CMC15": (0.30, 0.30),
    "Tycho": (0.06, 0.06),
    "AC": (0.80, 0.80),
    "Yale": (1.00, 1.00),
    "UNK": (1.00, 1.00),
}

# Per-(station, catalog) overrides (sigma_ra_arcsec, sigma_dec_arcsec).
_STATION_CATALOG_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    ("703", "Gaia2"): (0.34, 0.34),  # Catalina Sky Survey: wider PSF
    ("703", "UCAC4"): (0.45, 0.45),
    ("703", "UCAC2"): (0.55, 0.55),
    ("G96", "Gaia2"): (0.25, 0.25),  # Mt. Lemmon Survey
    ("G96", "UCAC4"): (0.35, 0.35),
    ("704", "USNOA2"): (0.60, 0.75),  # Spacewatch: known Dec bias
    ("F51", "Gaia2"): (0.15, 0.15),  # Pan-STARRS 1
    ("F51", "Gaia1"): (0.18, 0.18),
    ("F51", "2MASS"): (0.20, 0.20),
    ("F52", "Gaia3E"): (0.15, 0.15),  # Pan-STARRS 2
    ("F52", "Gaia1"): (0.18, 0.18),
    ("T05", "Gaia2"): (0.25, 0.25),  # ATLAS Haleakala
    ("T08", "Gaia2"): (0.25, 0.25),  # ATLAS Mauna Loa
    ("W68", "Gaia2"): (0.25, 0.25),  # ATLAS Chile
}


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
    obs_codes: list[str | None] = [None] * len(_CATALOG_DEFAULTS)
    astcats: list[str] = list(_CATALOG_DEFAULTS)
    sigma_ra = [s[0] for s in _CATALOG_DEFAULTS.values()]
    sigma_dec = [s[1] for s in _CATALOG_DEFAULTS.values()]
    for (code, astcat), (ra, dec) in _STATION_CATALOG_OVERRIDES.items():
        obs_codes.append(code)
        astcats.append(astcat)
        sigma_ra.append(ra)
        sigma_dec.append(dec)
    return pa.table(
        {
            "obs_code": pa.array(obs_codes, pa.large_string()),
            "astcat": pa.array(astcats, pa.large_string()),
            "sigma_ra_arcsec": pa.array(sigma_ra, pa.float64()),
            "sigma_dec_arcsec": pa.array(sigma_dec, pa.float64()),
        },
        schema=VERES2017_SIGMA_TABLE_SCHEMA,
    )


class VeresSigmaLookup:
    """
    Resolve (station, catalog) to (sigma_ra_arcsec, sigma_dec_arcsec).

    Lookup order: the (station, catalog) override row, then the catalog
    default row (``obs_code`` null), then ``fallback_sigma_arcsec`` for both
    axes; None as fallback means "no sigma known" (the caller passes the
    observation through).

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
        self.fallback_sigma_arcsec = fallback_sigma_arcsec
        self._by_station_catalog: dict[tuple[str, str], tuple[float, float]] = {}
        self._by_catalog: dict[str, tuple[float, float]] = {}
        for code, astcat, ra, dec in zip(
            table["obs_code"].to_pylist(),
            table["astcat"].to_pylist(),
            table["sigma_ra_arcsec"].to_pylist(),
            table["sigma_dec_arcsec"].to_pylist(),
        ):
            sigmas = (float(ra), float(dec))
            if code is None:
                if astcat in self._by_catalog:
                    raise ValueError(f"Duplicate catalog default row for {astcat!r}")
                self._by_catalog[astcat] = sigmas
            else:
                key = (code, astcat)
                if key in self._by_station_catalog:
                    raise ValueError(f"Duplicate override row for {key}")
                self._by_station_catalog[key] = sigmas

    def sigmas(
        self, obs_code: str | None, astcat: str | None
    ) -> tuple[float, float] | None:
        """(sigma_ra_arcsec, sigma_dec_arcsec) for the observation, or None."""
        if obs_code is not None and astcat is not None:
            override = self._by_station_catalog.get((obs_code, astcat))
            if override is not None:
                return override
        if astcat is not None:
            default = self._by_catalog.get(astcat)
            if default is not None:
                return default
        if self.fallback_sigma_arcsec is None:
            return None
        return (self.fallback_sigma_arcsec, self.fallback_sigma_arcsec)


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

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
    ) -> None:
        self.lookup = VeresSigmaLookup(sigma_table, fallback_sigma_arcsec)

    def _updated_variances(
        self,
        var_lon: float,
        var_lat: float,
        veres_var_lon: float,
        veres_var_lat: float,
    ) -> tuple[float, float, bool] | None:
        """
        Return (var_lon, var_lat, zero_cross_term) for one observation, in
        deg² with lon NOT cos(dec)-corrected, or None to pass it through.
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
        astcats = observations.astcat.to_pylist()

        for i, (code, astcat) in enumerate(zip(codes, astcats)):
            sigmas = self.lookup.sigmas(code, astcat)
            if sigmas is None or not np.isfinite(cos_dec[i]) or cos_dec[i] <= 0.0:
                continue
            veres_var_lon = (sigmas[0] / (ARCSEC_PER_DEG * cos_dec[i])) ** 2
            veres_var_lat = (sigmas[1] / ARCSEC_PER_DEG) ** 2
            block = self._updated_variances(
                float(covariances[i, 1, 1]),
                float(covariances[i, 2, 2]),
                float(veres_var_lon),
                float(veres_var_lat),
            )
            if block is None:
                continue
            var_lon, var_lat, zero_cross_term = block
            new_covariances[i, 1, 1] = var_lon
            new_covariances[i, 2, 2] = var_lat
            if zero_cross_term:
                new_covariances[i, 1, 2] = 0.0
                new_covariances[i, 2, 1] = 0.0

        if np.array_equal(new_covariances, covariances, equal_nan=True):
            return observations
        return observations.set_column(
            "coordinates.covariance",
            CoordinateCovariances.from_matrix(new_covariances),
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

    def __init__(
        self,
        sigma_table: pa.Table | None = None,
        fallback_sigma_arcsec: float | None = VERES2017_FALLBACK_SIGMA_ARCSEC,
        fill_missing: bool = False,
    ) -> None:
        super().__init__(sigma_table, fallback_sigma_arcsec)
        self.fill_missing = fill_missing

    def _updated_variances(
        self,
        var_lon: float,
        var_lat: float,
        veres_var_lon: float,
        veres_var_lat: float,
    ) -> tuple[float, float, bool] | None:
        def floored(var: float, floor: float) -> float:
            if not np.isfinite(var):
                return floor if self.fill_missing else var
            return max(var, floor)

        new_lon = floored(var_lon, veres_var_lon)
        new_lat = floored(var_lat, veres_var_lat)
        unchanged = (
            new_lon == var_lon or (np.isnan(new_lon) and np.isnan(var_lon))
        ) and (new_lat == var_lat or (np.isnan(new_lat) and np.isnan(var_lat)))
        if unchanged:
            return None
        return (new_lon, new_lat, False)


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

    def _updated_variances(
        self,
        var_lon: float,
        var_lat: float,
        veres_var_lon: float,
        veres_var_lat: float,
    ) -> tuple[float, float, bool] | None:
        return (veres_var_lon, veres_var_lat, True)
