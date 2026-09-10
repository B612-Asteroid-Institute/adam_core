"""
EFCC18 star-catalog debiasing of astrometric observations.

Implements the per-observation correction of Eggl, Farnocchia, Chamberlin &
Chesley (2020), "Star catalog position and proper motion corrections in
asteroid astrometry II: The Gaia era", Icarus 339:113596 (EFCC18;
arXiv:1909.04558). The published table ``bias.dat``
covers 26 star catalogs over a HEALPix tessellation of the sky
(``N_side = 64``, 49152 tiles, RING ordering: the k-th data row of ``bias.dat``
is HEALPix ring pixel k, as listed in the archive's ``tiles.dat``). Each
(tile, catalog) cell
stores four numbers: the position correction in RA*cos(Dec) at J2000
[arcsec], the position correction in Dec at J2000 [arcsec], and the proper
motion corrections in RA*cos(Dec) and Dec [mas/yr]. The corrections are
expressed with respect to Gaia-DR2.

To debias one observation reduced against catalog ``c`` in tile ``k`` at
epoch ``t`` (years)::

    bias_ra_arcsec  = dRA[k, c]  + (t - 2000.0) * pmRA[k, c]  / 1000   # cos(dec) frame
    bias_dec_arcsec = dDec[k, c] + (t - 2000.0) * pmDec[k, c] / 1000
    corrected_RA  = observed_RA  - bias_ra_arcsec  / (3600 * cos(Dec))   # degrees
    corrected_Dec = observed_Dec - bias_dec_arcsec / 3600                # degrees

This module provides the table loader, the vectorised per-observation lookup
(`compute_efcc18_corrections`), the map from MPC ``astCat`` codes to the
single-character EFCC18 catalog codes, and coverage helpers. The
`~adam_core.orbit_determination.EFCC18DebiasModel` applies the correction to
`~adam_core.orbit_determination.OrbitDeterminationObservations` using their
``astcat`` column.

Reference data
--------------
``bias.dat`` is public reference data published by JPL Solar System Dynamics
in the archive ``debias_2018.tgz`` (``bias.dat``, ``tiles.dat``,
``README.txt``), linked from the EFCC18 paper:

    https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz

The file header reads ``BIAS_VERSION= 3.0 (September 21, 2018)``; the archive
was last updated on 2023-03-01 to relabel the UCAC-5 column from ``W`` to
``Y`` (``W`` is the MPC code for Gaia-DR3). Its ``bias.dat`` has SHA-256
`EFCC18_BIAS_DAT_SHA256`. At 37 MB (9 MB compressed) it is not bundled with
adam_core itself. The recommended way to obtain it is the ``jpl-debias-2018``
data package (``pip install jpl-debias-2018``), a B612 mirror of the archive in
the style of ``naif-de440`` and ``mpc-obscodes``. ``bias.dat`` is located, in
order, from:

1. an explicit ``bias_dat`` path argument;
2. the ``ADAM_CORE_EFCC18_BIAS_DAT`` environment variable;
3. the installed ``jpl_debias_2018`` package;
4. ``bias.dat`` inside the EFCC18 cache directory (``ADAM_CORE_EFCC18_DIR``,
   else ``$XDG_CACHE_HOME/adam_core/efcc18``, else
   ``~/.cache/adam_core/efcc18``).

Without the package, populate the cache once with `download_efcc18_bias_table`
(fetches the archive from JPL and verifies the checksum) or, on machines
without network access, with `install_efcc18_bias_table` pointing at a local
copy of the archive or of ``bias.dat``. A parsed ``.npy`` copy is written on
first load so subsequent loads are fast: next to ``bias.dat`` when that
directory is writable, otherwise in the cache directory.

Catalog coverage
----------------
`MPC_ASTCAT_TO_EFCC18` maps the MPC/ADES ``astCat`` codes to EFCC18 columns.
Catalogs that postdate EFCC18 (Gaia-DR2/EDR3/DR3, ATLAS, Pan-STARRS, ...) are
absent and receive zero correction, as do observations whose catalog is
unknown. The bias.dat header also states that JPL SSD does not debias four of
the tabulated catalogs (Gaia-DR1, ACT, Tycho-2 and UCAC-5); they are listed
in `EFCC18_JPL_UNDEBIASED_ASTCATS`. This module corrects them by default,
matching the behaviour of the study code it was ported from; pass
``exclude_astcats=EFCC18_JPL_UNDEBIASED_ASTCATS`` to follow JPL's practice.

healpy is required for the HEALPix tile lookup. It is a core dependency of
adam_core today; the ``adam_core[debias]`` extra also lists it so this module
keeps working should that change.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

try:
    import healpy as hp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only without healpy
    hp = None

__all__ = [
    "EFCC18_ARCHIVE_URL",
    "EFCC18_BIAS_DAT_SHA256",
    "EFCC18_BIAS_VERSION",
    "EFCC18_CATALOG_CODES",
    "EFCC18_JPL_UNDEBIASED_ASTCATS",
    "EFCC18_NSIDE",
    "EFCC18_N_CATALOGS",
    "EFCC18_N_TILES",
    "MPC_ASTCAT_TO_EFCC18",
    "compute_efcc18_corrections",
    "download_efcc18_bias_table",
    "efcc18_cache_dir",
    "install_efcc18_bias_table",
    "is_efcc18_covered",
    "load_efcc18_biases",
    "n_observations_covered",
    "ra_dec_to_healpix",
    "read_efcc18_bias_version",
    "resolve_bias_dat",
]

logger = logging.getLogger(__name__)

#: Source archive published by JPL SSD (also linked from Eggl et al. 2020).
EFCC18_ARCHIVE_URL = "https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz"
#: SHA-256 of the ``bias.dat`` inside the 2023-03-01 archive.
EFCC18_BIAS_DAT_SHA256 = (
    "ef6a50830bb83d8b1161e7acb391eed988ea9708e9fbcccc0c834da6109476e3"
)
#: ``BIAS_VERSION`` tag in the header of that ``bias.dat``.
EFCC18_BIAS_VERSION = "3.0 (September 21, 2018)"
#: Environment variable naming the cache directory.
EFCC18_CACHE_DIR_ENV = "ADAM_CORE_EFCC18_DIR"
#: Environment variable naming an explicit ``bias.dat`` path.
EFCC18_BIAS_DAT_ENV = "ADAM_CORE_EFCC18_BIAS_DAT"

#: Single-character EFCC18 catalog codes, in column order of ``bias.dat``
#: (header line "Catalogs in this file (MPC designation)").
EFCC18_CATALOG_CODES: tuple[str, ...] = tuple("abcdegijlmnopqrtuvwLNQRSUY")
EFCC18_N_CATALOGS = len(EFCC18_CATALOG_CODES)
assert EFCC18_N_CATALOGS == 26

#: HEALPix resolution used by EFCC18 and the resulting number of tiles.
EFCC18_NSIDE = 64
EFCC18_N_TILES = 12 * EFCC18_NSIDE * EFCC18_NSIDE

#: Map from MPC/ADES ``astCat`` code to the single-character EFCC18 code.
#: Catalogs that postdate EFCC18 (Gaia-DR2/EDR3/DR3, ATLAS, Pan-STARRS, ...)
#: are intentionally absent: they have no EFCC18 entry and pass through with
#: zero correction.
MPC_ASTCAT_TO_EFCC18: dict[str, str] = {
    "USNOA1": "a",
    "USNOSA1": "b",
    "USNOA2": "c",
    "USNOSA2": "d",
    "UCAC1": "e",
    "Tycho": "g",  # Tycho-2
    "GSC1.1": "i",
    "GSC1.2": "j",
    "ACT": "l",
    "GSCACT": "m",
    "SDSS8": "n",  # SDSS-DR8
    "USNOB1": "o",
    "PPM": "p",
    "UCAC4": "q",
    "UCAC2": "r",
    "PPMXL": "t",
    "UCAC3": "u",
    "NOMAD": "v",
    "CMC14": "w",
    "2MASS": "L",
    "SDSS7": "N",
    "CMC15": "Q",
    "SSTRC4": "R",
    "URAT1": "S",
    "Gaia1": "U",  # Gaia-DR1
    "UCAC5": "Y",
}

#: Catalogs tabulated in ``bias.dat`` that JPL SSD nevertheless does not
#: debias (per the file header), as MPC ``astCat`` codes.
EFCC18_JPL_UNDEBIASED_ASTCATS: tuple[str, ...] = ("Gaia1", "ACT", "Tycho", "UCAC5")

_JD_J2000 = 2451545.0
_DAYS_PER_JULIAN_YEAR = 365.25
_CODE_TO_COLUMN = {code: i for i, code in enumerate(EFCC18_CATALOG_CODES)}


def _require_healpy() -> None:
    if hp is None:
        raise ImportError(
            "healpy is required for EFCC18 debiasing (HEALPix tile lookup). "
            "Install it with `pip install healpy` or `pip install 'adam_core[debias]'`."
        )


# ---------------------------------------------------------------------------
# Locating, downloading and caching bias.dat
# ---------------------------------------------------------------------------


def efcc18_cache_dir() -> Path:
    """
    Return the directory used to cache EFCC18 reference data.

    Resolution order: ``ADAM_CORE_EFCC18_DIR``, then
    ``$XDG_CACHE_HOME/adam_core/efcc18``, then ``~/.cache/adam_core/efcc18``.
    The directory is not created by this function.
    """
    env_dir = os.environ.get(EFCC18_CACHE_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "adam_core" / "efcc18"


def resolve_bias_dat(path: Optional[Union[str, Path]] = None) -> Path:
    """
    Locate the EFCC18 ``bias.dat`` file.

    Parameters
    ----------
    path : str or Path, optional
        Explicit location. If given it must exist.

    Returns
    -------
    path : Path
        The first existing candidate among: ``path``, the
        ``ADAM_CORE_EFCC18_BIAS_DAT`` environment variable, the ``bias.dat``
        shipped by the installed ``jpl_debias_2018`` package, and ``bias.dat``
        in `efcc18_cache_dir`.

    Raises
    ------
    FileNotFoundError
        If no candidate exists. The message explains how to populate the cache.
    """
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"EFCC18 bias.dat not found at {candidate}")
        return candidate

    env_path = os.environ.get(EFCC18_BIAS_DAT_ENV)
    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"EFCC18 bias.dat not found at {candidate} "
                f"(from ${EFCC18_BIAS_DAT_ENV})"
            )
        return candidate

    packaged = _packaged_bias_dat()
    if packaged is not None:
        return packaged

    cached = efcc18_cache_dir() / "bias.dat"
    if cached.is_file():
        return cached

    raise FileNotFoundError(
        "EFCC18 bias.dat not found. Install the data package "
        "(`pip install jpl-debias-2018`), or populate the cache with "
        "adam_core.observations.efcc18.download_efcc18_bias_table() (fetches "
        f"{EFCC18_ARCHIVE_URL}) or install_efcc18_bias_table(<local debias_2018.tgz "
        "or bias.dat>), or point $"
        f"{EFCC18_BIAS_DAT_ENV} at an existing bias.dat. Cache directory: "
        f"{efcc18_cache_dir()} (override with ${EFCC18_CACHE_DIR_ENV})."
    )


def _packaged_bias_dat() -> Optional[Path]:
    """``bias.dat`` shipped by the ``jpl_debias_2018`` data package, if installed."""
    try:
        package = importlib.import_module("jpl_debias_2018")
    except ImportError:
        return None
    path = getattr(package, "bias_dat", None)
    if not path:
        return None
    candidate = Path(str(path))
    return candidate if candidate.is_file() else None


def _default_cache_path(bias_dat_path: Path) -> Path:
    """
    Location of the parsed ``.npy`` cache for ``bias_dat_path``: next to it when
    its directory is writable (e.g. a user cache), otherwise inside
    `efcc18_cache_dir` (e.g. for a read-only site-packages install).
    """
    sibling = bias_dat_path.with_suffix(".npy")
    if _is_writable(bias_dat_path.parent):
        return sibling
    return efcc18_cache_dir() / sibling.name


def _is_writable(directory: Path) -> bool:
    return os.access(directory, os.W_OK)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path) -> None:
    actual = _sha256(path)
    if actual != EFCC18_BIAS_DAT_SHA256:
        raise ValueError(
            f"EFCC18 bias.dat at {path} has SHA-256 {actual}, expected "
            f"{EFCC18_BIAS_DAT_SHA256} (BIAS_VERSION {EFCC18_BIAS_VERSION}). "
            "Pass verify_checksum=False to accept a different table version."
        )


def _extract_bias_dat_from_archive(archive: Path, destination: Path) -> None:
    """Copy the ``bias.dat`` member of ``archive`` to ``destination``."""
    with tarfile.open(archive, mode="r:*") as tar:
        member = next(
            (m for m in tar.getmembers() if Path(m.name).name == "bias.dat"), None
        )
        if member is None or not member.isfile():
            raise ValueError(f"No bias.dat member found in archive {archive}")
        source = tar.extractfile(member)
        if source is None:  # pragma: no cover - guarded by isfile above
            raise ValueError(f"Could not read bias.dat from archive {archive}")
        with source, destination.open("wb") as out:
            shutil.copyfileobj(source, out)


def install_efcc18_bias_table(
    source: Union[str, Path],
    destination_dir: Optional[Union[str, Path]] = None,
    verify_checksum: bool = True,
) -> Path:
    """
    Install ``bias.dat`` into the EFCC18 cache from a local file.

    Parameters
    ----------
    source : str or Path
        Either the ``debias_2018.tgz`` archive (``.tgz`` / ``.tar.gz`` / ``.tar``)
        or an already extracted ``bias.dat``.
    destination_dir : str or Path, optional
        Directory to install into. Default `efcc18_cache_dir`. Created if needed.
    verify_checksum : bool
        Verify the installed file against `EFCC18_BIAS_DAT_SHA256` (default True).

    Returns
    -------
    path : Path
        Path of the installed ``bias.dat``.
    """
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"EFCC18 source file not found: {source_path}")
    dest_dir = (
        Path(destination_dir).expanduser()
        if destination_dir is not None
        else efcc18_cache_dir()
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / "bias.dat"

    # Write to a temporary file in the destination directory first so a
    # failed verification never leaves a half-written or wrong bias.dat behind.
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="bias.dat.", dir=dest_dir)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        if tarfile.is_tarfile(source_path):
            _extract_bias_dat_from_archive(source_path, tmp_path)
        else:
            shutil.copyfile(source_path, tmp_path)
        if verify_checksum:
            _verify_checksum(tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Invalidate any stale parsed cache next to the new file.
    stale_npy = destination.with_suffix(".npy")
    if stale_npy.exists():
        stale_npy.unlink()
    logger.info("Installed EFCC18 bias.dat to %s", destination)
    return destination


def download_efcc18_bias_table(
    url: str = EFCC18_ARCHIVE_URL,
    destination_dir: Optional[Union[str, Path]] = None,
    verify_checksum: bool = True,
    timeout_s: float = 300.0,
) -> Path:
    """
    Download the EFCC18 archive from JPL and install ``bias.dat`` into the cache.

    Parameters
    ----------
    url : str
        Archive URL. Default `EFCC18_ARCHIVE_URL`.
    destination_dir : str or Path, optional
        Directory to install into. Default `efcc18_cache_dir`.
    verify_checksum : bool
        Verify against `EFCC18_BIAS_DAT_SHA256` (default True).
    timeout_s : float
        Request timeout in seconds.

    Returns
    -------
    path : Path
        Path of the installed ``bias.dat``.
    """
    import requests

    logger.info("Downloading EFCC18 archive from %s", url)
    with tempfile.TemporaryDirectory(prefix="efcc18_") as tmp_dir:
        archive = Path(tmp_dir) / "debias_2018.tgz"
        with requests.get(url, stream=True, timeout=timeout_s) as response:
            response.raise_for_status()
            with archive.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
        return install_efcc18_bias_table(
            archive, destination_dir=destination_dir, verify_checksum=verify_checksum
        )


# ---------------------------------------------------------------------------
# Loading the table
# ---------------------------------------------------------------------------


def read_efcc18_bias_version(bias_dat: Union[str, Path]) -> Optional[str]:
    """
    Return the ``BIAS_VERSION`` tag from the header of ``bias_dat``, if present.
    """
    with Path(bias_dat).open() as f:
        for line in f:
            if not line.startswith("!"):
                break
            if "BIAS_VERSION=" in line:
                return line.split("BIAS_VERSION=", 1)[1].strip()
    return None


def load_efcc18_biases(
    bias_dat: Optional[Union[str, Path]] = None,
    cache_npy: Optional[Union[str, Path]] = None,
) -> npt.NDArray[np.float32]:
    """
    Load EFCC18 ``bias.dat`` into a ``(EFCC18_N_TILES, 26, 4)`` float32 array.

    The last axis is ``(dRA_arcsec, dDec_arcsec, pmRA_mas_yr, pmDec_mas_yr)``;
    RA quantities are in the cos(dec)-corrected frame. The middle axis follows
    `EFCC18_CATALOG_CODES`.

    A parsed ``.npy`` copy is written on first load and reused while it is at
    least as new as ``bias.dat``: at ``cache_npy`` if given, else next to
    ``bias.dat`` when that directory is writable, else in `efcc18_cache_dir`.

    Parameters
    ----------
    bias_dat : str or Path, optional
        Location of ``bias.dat``; resolved with `resolve_bias_dat` when None.
    cache_npy : str or Path, optional
        Location of the parsed cache. Default: see above.

    Returns
    -------
    bias_table : `numpy.ndarray` (EFCC18_N_TILES, 26, 4), float32
    """
    bias_dat_path = resolve_bias_dat(bias_dat)
    cache_path = (
        Path(cache_npy).expanduser()
        if cache_npy is not None
        else _default_cache_path(bias_dat_path)
    )
    expected_shape = (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)

    if (
        cache_path.is_file()
        and cache_path.stat().st_mtime >= bias_dat_path.stat().st_mtime
    ):
        cached = np.load(cache_path)
        if cached.shape == expected_shape:
            return np.asarray(cached, dtype=np.float32)
        logger.warning(
            "EFCC18 cache at %s has shape %s, expected %s; re-parsing bias.dat",
            cache_path,
            cached.shape,
            expected_shape,
        )

    version = read_efcc18_bias_version(bias_dat_path)
    logger.info(
        "Parsing EFCC18 bias.dat from %s (BIAS_VERSION %s)", bias_dat_path, version
    )
    if version is not None and version != EFCC18_BIAS_VERSION:
        logger.warning(
            "EFCC18 bias.dat at %s reports BIAS_VERSION %r; this module was "
            "written against %r",
            bias_dat_path,
            version,
            EFCC18_BIAS_VERSION,
        )
    raw = np.loadtxt(bias_dat_path, comments="!", dtype=np.float32, ndmin=2)
    if raw.shape != (EFCC18_N_TILES, EFCC18_N_CATALOGS * 4):
        raise ValueError(
            f"Unexpected bias.dat layout {raw.shape}; expected "
            f"({EFCC18_N_TILES}, {EFCC18_N_CATALOGS * 4})"
        )
    bias_table = np.ascontiguousarray(raw.reshape(expected_shape))

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, bias_table)
        logger.info("Cached parsed EFCC18 table to %s", cache_path)
    except OSError as err:
        logger.warning("Could not write EFCC18 cache to %s: %s", cache_path, err)
    return bias_table


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def ra_dec_to_healpix(
    ra_deg: npt.ArrayLike, dec_deg: npt.ArrayLike
) -> npt.NDArray[np.int64]:
    """
    Map (RA, Dec) in degrees to EFCC18 HEALPix tile indices.

    Uses ``N_side = 64`` in RING ordering on the J2000 equatorial frame. That
    is the row order of ``bias.dat``: the JPL archive's ``tiles.dat`` lists the
    tile centres "sorted in the same way" as ``bias.dat`` (README), and those
    centres are healpy's ring-scheme pixel centres for all 49152 tiles (and
    nested-scheme centres for 1 of them). Find_Orb reads the table in ring
    order too (``bias.cpp`` / ``healpix.cpp``). An earlier version of this
    function used the nested scheme, which put every observation in an
    unrelated tile; see ``test_tile_ordering_matches_jpl_tiles_dat``.
    """
    _require_healpy()
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    theta = np.deg2rad(90.0 - dec)  # colatitude
    phi = np.deg2rad(np.mod(ra, 360.0))
    tiles = hp.ang2pix(EFCC18_NSIDE, theta, phi, nest=False)
    return np.asarray(tiles, dtype=np.int64)


def _catalog_columns(
    astcats: Sequence[Optional[str]], exclude_astcats: Iterable[str] = ()
) -> npt.NDArray[np.int64]:
    """EFCC18 column index per observation, or -1 where not covered."""
    excluded = set(exclude_astcats)
    columns = np.full(len(astcats), -1, dtype=np.int64)
    for i, astcat in enumerate(astcats):
        if astcat is None or astcat in excluded:
            continue
        code = MPC_ASTCAT_TO_EFCC18.get(astcat)
        if code is not None:
            columns[i] = _CODE_TO_COLUMN[code]
    return columns


def is_efcc18_covered(
    astcats: Sequence[Optional[str]], exclude_astcats: Iterable[str] = ()
) -> npt.NDArray[np.bool_]:
    """
    Boolean mask of which ``astcats`` receive an EFCC18 correction.

    Parameters
    ----------
    astcats : sequence of str or None (N)
        MPC ``astCat`` codes; None means unknown catalog.
    exclude_astcats : iterable of str
        Catalogs to treat as not covered (e.g. `EFCC18_JPL_UNDEBIASED_ASTCATS`).
    """
    return _catalog_columns(astcats, exclude_astcats) >= 0


def n_observations_covered(
    astcats: Sequence[Optional[str]], exclude_astcats: Iterable[str] = ()
) -> int:
    """
    Number of entries of ``astcats`` that receive an EFCC18 correction.
    """
    return int(np.count_nonzero(is_efcc18_covered(astcats, exclude_astcats)))


def compute_efcc18_corrections(
    ra_deg: npt.ArrayLike,
    dec_deg: npt.ArrayLike,
    astcats: Sequence[Optional[str]],
    jd_tdb: npt.ArrayLike,
    bias_table: Optional[npt.NDArray[np.floating]] = None,
    exclude_astcats: Iterable[str] = (),
) -> npt.NDArray[np.float64]:
    """
    Compute per-observation EFCC18 corrections.

    Parameters
    ----------
    ra_deg, dec_deg : array_like (N)
        Observed RA and Dec in degrees.
    astcats : sequence of str or None (N)
        MPC ``astCat`` codes. Entries that are None, not in
        `MPC_ASTCAT_TO_EFCC18`, or listed in ``exclude_astcats`` receive zero
        correction, as do rows whose position or epoch is not finite.
    jd_tdb : array_like (N)
        Observation epochs as Julian Dates (TDB), for the proper-motion term
        ``(JD - 2451545.0) / 365.25`` years since J2000.
    bias_table : `numpy.ndarray` (EFCC18_N_TILES, 26, 4), optional
        Pre-loaded table from `load_efcc18_biases`; loaded on demand when None.
    exclude_astcats : iterable of str
        Catalogs to leave uncorrected even though tabulated.

    Returns
    -------
    corrections : `numpy.ndarray` (N, 2), float64
        ``(bias_ra_arcsec, bias_dec_arcsec)`` in the cos(dec)-corrected
        tangent-plane frame (directly comparable to MPC ``rmsRA``). To debias,
        subtract ``bias_ra / (3600 cos(dec))`` from RA and ``bias_dec / 3600``
        from Dec, both in degrees.
    """
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    jd = np.asarray(jd_tdb, dtype=np.float64)
    n = ra.shape[0]
    if ra.ndim != 1 or dec.shape != (n,) or jd.shape != (n,) or len(astcats) != n:
        raise ValueError("ra_deg, dec_deg, astcats and jd_tdb must all have length N")

    corrections = np.zeros((n, 2), dtype=np.float64)
    columns = _catalog_columns(astcats, exclude_astcats)
    # Rows with a non-finite position cannot be placed on a tile: zero correction.
    covered = (columns >= 0) & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(jd)
    if not np.any(covered):
        return corrections

    if bias_table is None:
        bias_table = load_efcc18_biases()
    expected_shape = (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)
    if bias_table.shape != expected_shape:
        raise ValueError(
            f"bias_table has shape {bias_table.shape}, expected {expected_shape}"
        )

    tiles = ra_dec_to_healpix(ra[covered], dec[covered])
    cells = np.asarray(bias_table[tiles, columns[covered], :], dtype=np.float64)
    years_since_j2000 = (jd[covered] - _JD_J2000) / _DAYS_PER_JULIAN_YEAR
    # bias.dat stores proper motions in mas/yr; /1000 converts to arcsec/yr.
    corrections[covered, 0] = cells[:, 0] + years_since_j2000 * cells[:, 2] / 1000.0
    corrections[covered, 1] = cells[:, 1] + years_since_j2000 * cells[:, 3] / 1000.0
    return corrections
