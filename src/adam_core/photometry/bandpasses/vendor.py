"""
Tools for building (vendoring) bandpass curve and reflectance template data.

Bandpass curves are sourced from the SVO Filter Profile Service. Please see
`REFERENCES.md` for the required acknowledgement and citations when using this
service.
"""

# coverage: ignore-file

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa

from ...utils.http import _raise_compatible_http_error
from .tables import (
    AsteroidTemplates,
    BandpassCurves,
    ObservatoryBandMap,
    TemplateBandpassIntegrals,
)


@dataclass(frozen=True)
class _SvoBandpassSpec:
    filter_id: str
    instrument: str
    band: str
    svo_id: str


def _default_svo_bandpass_specs() -> tuple[_SvoBandpassSpec, ...]:
    # NOTE: These SVO IDs were validated via fps.php?ID=<id> returning VOTable with
    # columns Wavelength (Angstrom) and Transmission.
    return (
        # Standard Johnson/Cousins-like passbands (Bessell). Note: `V` is kept for
        # backwards compatibility; it is sourced from Bessell.V.
        _SvoBandpassSpec("V", "Bessell", "V", "Generic/Bessell.V"),
        _SvoBandpassSpec("Bessell_U", "Bessell", "U", "Generic/Bessell.U"),
        _SvoBandpassSpec("Bessell_B", "Bessell", "B", "Generic/Bessell.B"),
        _SvoBandpassSpec("Bessell_R", "Bessell", "R", "Generic/Bessell.R"),
        _SvoBandpassSpec("Bessell_I", "Bessell", "I", "Generic/Bessell.I"),
        # Rubin / LSST
        _SvoBandpassSpec("LSST_u", "LSST", "u", "LSST/LSST.u"),
        _SvoBandpassSpec("LSST_g", "LSST", "g", "LSST/LSST.g"),
        _SvoBandpassSpec("LSST_r", "LSST", "r", "LSST/LSST.r"),
        _SvoBandpassSpec("LSST_i", "LSST", "i", "LSST/LSST.i"),
        _SvoBandpassSpec("LSST_z", "LSST", "z", "LSST/LSST.z"),
        _SvoBandpassSpec("LSST_y", "LSST", "y", "LSST/LSST.y"),
        # Sloan Digital Sky Survey (SDSS) ugriz
        _SvoBandpassSpec("SDSS_u", "SDSS", "u", "SLOAN/SDSS.u"),
        _SvoBandpassSpec("SDSS_g", "SDSS", "g", "SLOAN/SDSS.g"),
        _SvoBandpassSpec("SDSS_r", "SDSS", "r", "SLOAN/SDSS.r"),
        _SvoBandpassSpec("SDSS_i", "SDSS", "i", "SLOAN/SDSS.i"),
        _SvoBandpassSpec("SDSS_z", "SDSS", "z", "SLOAN/SDSS.z"),
        # Pan-STARRS1 (PS1) grizy
        _SvoBandpassSpec("PS1_g", "PS1", "g", "PAN-STARRS/PS1.g"),
        _SvoBandpassSpec("PS1_r", "PS1", "r", "PAN-STARRS/PS1.r"),
        _SvoBandpassSpec("PS1_i", "PS1", "i", "PAN-STARRS/PS1.i"),
        _SvoBandpassSpec("PS1_z", "PS1", "z", "PAN-STARRS/PS1.z"),
        _SvoBandpassSpec("PS1_y", "PS1", "y", "PAN-STARRS/PS1.y"),
        # ZTF
        _SvoBandpassSpec("ZTF_g", "ZTF", "g", "Palomar/ZTF.g"),
        _SvoBandpassSpec("ZTF_r", "ZTF", "r", "Palomar/ZTF.r"),
        _SvoBandpassSpec("ZTF_i", "ZTF", "i", "Palomar/ZTF.i"),
        # DECam
        _SvoBandpassSpec("DECam_u", "DECam", "u", "CTIO/DECam.u"),
        _SvoBandpassSpec("DECam_g", "DECam", "g", "CTIO/DECam.g"),
        _SvoBandpassSpec("DECam_r", "DECam", "r", "CTIO/DECam.r"),
        _SvoBandpassSpec("DECam_i", "DECam", "i", "CTIO/DECam.i"),
        _SvoBandpassSpec("DECam_z", "DECam", "z", "CTIO/DECam.z"),
        _SvoBandpassSpec("DECam_Y", "DECam", "Y", "CTIO/DECam.Y"),
        _SvoBandpassSpec("DECam_VR", "DECam", "VR", "CTIO/DECam.VR_filter"),
        # Mosaic3 (Mayall / KPNO): SVO provides the Mosaic3 z-band passband via MzLS.
        _SvoBandpassSpec("Mosaic3_z", "Mosaic3", "z", "KPNO/MzLS.z"),
        # Bok / 90Prime (BASS) g/r used by MPC code V00 (Kitt Peak-Bok).
        _SvoBandpassSpec("BASS_g", "BASS", "g", "BOK/BASS.g"),
        _SvoBandpassSpec("BASS_r", "BASS", "r", "BOK/BASS.r"),
        # SkyMapper
        _SvoBandpassSpec("SkyMapper_u", "SkyMapper", "u", "SkyMapper/SkyMapper.u"),
        _SvoBandpassSpec("SkyMapper_v", "SkyMapper", "v", "SkyMapper/SkyMapper.v"),
        _SvoBandpassSpec("SkyMapper_g", "SkyMapper", "g", "SkyMapper/SkyMapper.g"),
        _SvoBandpassSpec("SkyMapper_r", "SkyMapper", "r", "SkyMapper/SkyMapper.r"),
        _SvoBandpassSpec("SkyMapper_i", "SkyMapper", "i", "SkyMapper/SkyMapper.i"),
        _SvoBandpassSpec("SkyMapper_z", "SkyMapper", "z", "SkyMapper/SkyMapper.z"),
        # ATLAS
        # SVO uses the names "Atlas.cyan" and "Atlas.orange" for c/o.
        _SvoBandpassSpec("ATLAS_c", "ATLAS", "c", "Misc/Atlas.cyan"),
        _SvoBandpassSpec("ATLAS_o", "ATLAS", "o", "Misc/Atlas.orange"),
    )


def build_bandpass_curves(
    out_dir: Path,
    *,
    specs: Iterable[_SvoBandpassSpec] | None = None,
    timeout_s: int = 60,
) -> BandpassCurves:
    """
    Download, normalize, and write `BandpassCurves` to `out_dir/bandpass_curves.parquet`.
    """
    if specs is None:
        specs = _default_svo_bandpass_specs()

    records = list(specs)
    if not records:
        raise ValueError("No bandpass specs provided")

    from adam_core import _rust_native
    from adam_core._rust.arrow import table_from_record_batch

    try:
        batch = _rust_native.build_bandpass_curves_arrow(
            str(out_dir),
            [
                (spec.filter_id, spec.instrument, spec.band, spec.svo_id)
                for spec in records
            ],
            int(timeout_s),
        )
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return table_from_record_batch(BandpassCurves, batch)


def build_observatory_band_map(out_dir: Path) -> ObservatoryBandMap:
    """Build and atomically publish the map in one Rust crossing."""
    from adam_core import _rust_native
    from adam_core._rust.arrow import table_from_record_batch

    batch = _rust_native.build_observatory_band_map_arrow(str(out_dir))
    return table_from_record_batch(ObservatoryBandMap, batch)


def build_asteroid_templates(out_dir: Path) -> AsteroidTemplates:
    """Build and atomically publish all template products in one Rust crossing."""
    from adam_core import _rust_native
    from adam_core._rust.arrow import table_from_record_batch

    batch = _rust_native.build_asteroid_templates_arrow(str(out_dir))
    return table_from_record_batch(AsteroidTemplates, batch)


def build_solar_spectrum(out_dir: Path) -> pa.Table:
    """Fetch, parse, normalize, and atomically publish the FITS product in Rust."""
    from adam_core import _rust_native

    try:
        batch = _rust_native.build_solar_spectrum_arrow(str(out_dir), 120)
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return pa.Table.from_batches([batch])


def build_template_bandpass_integrals(out_dir: Path) -> TemplateBandpassIntegrals:
    """Compute, assemble, and atomically publish integrals in one Rust crossing."""
    from adam_core import _rust_native
    from adam_core._rust.arrow import table_from_record_batch

    batch = _rust_native.build_template_bandpass_integrals_arrow(str(out_dir))
    return table_from_record_batch(TemplateBandpassIntegrals, batch)
