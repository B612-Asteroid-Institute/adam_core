"""
Tools for building (vendoring) bandpass curve and reflectance template data.

Bandpass curves are sourced from the SVO Filter Profile Service. Please see
`REFERENCES.md` for the required acknowledgement and citations when using this
service.
"""
# coverage: ignore-file

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.io import fits
from astropy.io.votable import parse_single_table

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


def _download_svo_votable(svo_id: str, *, timeout_s: int = 60) -> bytes:
    url = f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={svo_id}"
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content


def _parse_svo_curve(
    votable_bytes: bytes,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    table = parse_single_table(io.BytesIO(votable_bytes)).to_table()
    if "Wavelength" not in table.colnames or "Transmission" not in table.colnames:
        raise ValueError(f"Unexpected SVO schema; columns={table.colnames}")

    wl = np.asarray(table["Wavelength"], dtype=np.float64)
    trans = np.asarray(table["Transmission"], dtype=np.float64)
    if wl.ndim != 1 or trans.ndim != 1:
        raise ValueError("Unexpected SVO curve dimensions")
    if len(wl) != len(trans):
        raise ValueError("SVO Wavelength and Transmission lengths do not match")
    if len(wl) < 2:
        raise ValueError("SVO curve has too few points")

    # SVO FPS provides wavelength in Angstrom for these filters.
    wl_nm = wl / 10.0
    return wl_nm, trans


def _normalize_curve(
    wavelength_nm: npt.NDArray[np.float64], throughput: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    thr = np.asarray(throughput, dtype=np.float64)
    if np.any(~np.isfinite(wl)) or np.any(~np.isfinite(thr)):
        raise ValueError("Non-finite values in curve")

    order = np.argsort(wl)
    wl = wl[order]
    thr = thr[order]

    # Drop duplicate wavelengths by keeping the first occurrence.
    unique_mask = np.concatenate(([True], np.diff(wl) > 0))
    wl = wl[unique_mask]
    thr = thr[unique_mask]
    if len(wl) < 2:
        raise ValueError("Curve is degenerate after de-duplication")

    thr = np.clip(thr, 0.0, 1.0)
    m = float(np.max(thr))
    if m <= 0.0:
        raise ValueError("Curve has zero throughput everywhere")
    thr = thr / m
    return wl, thr


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

    filter_ids: list[str] = []
    instruments: list[str] = []
    bands: list[str] = []
    sources: list[str] = []
    wavelengths: list[npt.NDArray[np.float64]] = []
    throughputs: list[npt.NDArray[np.float64]] = []

    seen: set[str] = set()
    for spec in records:
        if spec.filter_id in seen:
            raise ValueError(f"Duplicate filter_id in specs: {spec.filter_id}")
        seen.add(spec.filter_id)

        raw = _download_svo_votable(spec.svo_id, timeout_s=timeout_s)
        wl_nm, thr = _parse_svo_curve(raw)
        wl_nm, thr = _normalize_curve(wl_nm, thr)

        filter_ids.append(spec.filter_id)
        instruments.append(spec.instrument)
        bands.append(spec.band)
        sources.append(
            f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={spec.svo_id}"
        )
        wavelengths.append(wl_nm)
        throughputs.append(thr)

    wl_arr = pa.array(wavelengths, type=pa.large_list(pa.float64()))
    thr_arr = pa.array(throughputs, type=pa.large_list(pa.float64()))

    table = BandpassCurves.from_kwargs(
        filter_id=filter_ids,
        instrument=instruments,
        band=bands,
        wavelength_nm=wl_arr,
        throughput=thr_arr,
        source=sources,
    )

    out_path = out_dir.joinpath("bandpass_curves.parquet")
    table.to_parquet(out_path)
    return table


def build_observatory_band_map(out_dir: Path) -> ObservatoryBandMap:
    """
    Write `ObservatoryBandMap` to `out_dir/observatory_band_map.parquet`.
    """
    mappings: list[tuple[str, str, str]] = []

    # DECam (W84)
    for band in ["u", "g", "r", "i", "z"]:
        mappings.append(("W84", band, f"DECam_{band}"))
    mappings.append(("W84", "Y", "DECam_Y"))
    mappings.append(("W84", "y", "DECam_Y"))  # pragmatic alias
    mappings.append(("W84", "VR", "DECam_VR"))
    mappings.append(("W84", "vr", "DECam_VR"))  # pragmatic alias

    # Mosaic3 (695): z-band only in SVO for Mosaic3/MzLS.
    mappings.append(("695", "z", "Mosaic3_z"))

    # ZTF (I41)
    for band in ["g", "r", "i"]:
        mappings.append(("I41", band, f"ZTF_{band}"))

    # SkyMapper (Q55)
    mappings.extend(
        [
            ("Q55", "u", "SkyMapper_u"),
            ("Q55", "v", "SkyMapper_v"),
            ("Q55", "g", "SkyMapper_g"),
            ("Q55", "r", "SkyMapper_r"),
            ("Q55", "i", "SkyMapper_i"),
            ("Q55", "z", "SkyMapper_z"),
        ]
    )

    # Rubin/LSST (X05)
    for band in ["u", "g", "r", "i", "z", "y"]:
        mappings.append(("X05", band, f"LSST_{band}"))
    mappings.append(("X05", "Y", "LSST_y"))  # pragmatic alias

    # ATLAS (multiple MPC codes share the same c/o passbands)
    for code in ["T08", "T05", "M22", "W68"]:
        mappings.append((code, "c", "ATLAS_c"))
        mappings.append((code, "o", "ATLAS_o"))

    # Kitt Peak-Bok (V00): BASS / Bok-90Prime g/r.
    mappings.append(("V00", "g", "BASS_g"))
    mappings.append(("V00", "r", "BASS_r"))

    obs_codes, bands, filter_ids = zip(*mappings)
    keys = [f"{c}|{b}" for c, b in zip(obs_codes, bands)]
    table = ObservatoryBandMap.from_kwargs(
        observatory_code=list(obs_codes),
        reported_band=list(bands),
        filter_id=list(filter_ids),
        key=list(keys),
    )
    out_path = out_dir.joinpath("observatory_band_map.parquet")
    table.to_parquet(out_path)
    return table


def build_asteroid_templates(out_dir: Path) -> AsteroidTemplates:
    """
    Build and vendor asteroid reflectance templates (C, S) plus fixed mixes (NEO, MBA).

    Notes
    -----
    - These are intentionally simple, hand-built reflectance templates designed for
      early synthetic photometry and are *not* a replacement for higher-fidelity
      published average spectra.
    - Reflectance is normalized to 1.0 at 550 nm.
    """
    # Shared wavelength grid (nm) spanning typical optical filters.
    wl = np.linspace(300.0, 1100.0, 4000, dtype=np.float64)

    def _normalize_at_550(
        reflectance: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r550 = float(np.interp(550.0, wl, reflectance))
        if r550 <= 0.0 or not np.isfinite(r550):
            raise ValueError("Template reflectance is invalid at 550 nm")
        return (reflectance / r550).astype(np.float64, copy=False)

    # C-type: relatively flat/slightly red with a weak 0.7 µm feature and UV drop-off.
    c = np.ones_like(wl)
    c *= 1.0 + 0.15 * (wl - 550.0) / 550.0
    c *= 1.0 - 0.03 * np.exp(-0.5 * ((wl - 700.0) / 50.0) ** 2)
    uv = wl < 450.0
    c[uv] *= np.exp(-(((450.0 - wl[uv]) / 100.0) ** 2))
    c = _normalize_at_550(c)

    # S-type: redder slope with onset of the ~1 µm silicate absorption and UV drop-off.
    s = np.ones_like(wl)
    s *= 1.0 + 0.50 * (wl - 550.0) / 550.0
    s *= 1.0 - 0.10 * np.exp(-0.5 * ((wl - 950.0) / 100.0) ** 2)
    uv = wl < 500.0
    s[uv] *= np.exp(-(((500.0 - wl[uv]) / 120.0) ** 2))
    s = _normalize_at_550(s)

    # Mixes are linear combinations in reflectance space.
    neo = 0.5 * c + 0.5 * s
    mba = 0.7 * c + 0.3 * s

    templates: list[tuple[str, float, float, npt.NDArray[np.float64]]] = [
        ("C", 1.0, 0.0, c),
        ("S", 0.0, 1.0, s),
        ("NEO", 0.5, 0.5, neo),
        ("MBA", 0.7, 0.3, mba),
    ]

    citations: dict[str, str] = {
        "C": "Hand-built C-type reflectance template (optical/NIR). Informed by mean asteroid colors (e.g., Bowell & Lumme 1979; Erasmus et al. 2019). Normalized at 550 nm.",
        "S": "Hand-built S-type reflectance template (optical/NIR). Informed by mean asteroid colors (e.g., Bowell & Lumme 1979; Erasmus et al. 2019). Normalized at 550 nm.",
        "NEO": "Assumed NEO population mix: 50% C / 50% S (linear reflectance mix).",
        "MBA": "Assumed main-belt population mix: 70% C / 30% S (linear reflectance mix).",
    }

    wl_arr = pa.array([wl] * len(templates), type=pa.large_list(pa.float64()))
    refl_arr = pa.array([t[3] for t in templates], type=pa.large_list(pa.float64()))
    table = AsteroidTemplates.from_kwargs(
        template_id=[t[0] for t in templates],
        wavelength_nm=wl_arr,
        reflectance=refl_arr,
        weight_C=[t[1] for t in templates],
        weight_S=[t[2] for t in templates],
        citation=[citations[t[0]] for t in templates],
    )

    out_path = out_dir.joinpath("asteroid_templates.parquet")
    table.to_parquet(out_path)
    return table


def build_solar_spectrum(out_dir: Path) -> pa.Table:
    """
    Download and vendor a fixed solar spectrum used for synthetic photometry integrals.

    Source
    ------
    STScI reference atlas solar spectrum FITS used widely in calibration contexts.
    """
    url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with fits.open(io.BytesIO(resp.content)) as hdul:
        data = hdul[1].data
        wavelength_angstrom = np.asarray(data["WAVELENGTH"], dtype=np.float64)
        flux = np.asarray(data["FLUX"], dtype=np.float64)

    wavelength_nm = wavelength_angstrom / 10.0

    # Restrict to a useful range for optical/NIR passbands, then normalize.
    mask = (wavelength_nm >= 300.0) & (wavelength_nm <= 1200.0)
    wavelength_nm = wavelength_nm[mask]
    flux = flux[mask]
    flux = flux / float(np.max(flux))

    table = pa.table(
        {
            "wavelength_nm": pa.array(wavelength_nm, type=pa.float64()),
            "flux": pa.array(flux, type=pa.float64()),
        }
    )
    out_path = out_dir.joinpath("solar_spectrum.parquet")
    pq.write_table(table, out_path)
    return table


def build_template_bandpass_integrals(out_dir: Path) -> TemplateBandpassIntegrals:
    """
    Compute and vendor template×filter integrals for photon-counting synthetic photometry.
    """
    curves = BandpassCurves.from_parquet(out_dir.joinpath("bandpass_curves.parquet"))
    templates = AsteroidTemplates.from_parquet(
        out_dir.joinpath("asteroid_templates.parquet")
    )
    solar_tbl = pq.read_table(out_dir.joinpath("solar_spectrum.parquet"))
    solar_wl = np.asarray(
        solar_tbl["wavelength_nm"].to_numpy(zero_copy_only=False), dtype=np.float64
    )
    solar_flux = np.asarray(
        solar_tbl["flux"].to_numpy(zero_copy_only=False), dtype=np.float64
    )

    if len(solar_wl) < 2 or np.any(np.diff(solar_wl) <= 0):
        order = np.argsort(solar_wl)
        solar_wl = solar_wl[order]
        solar_flux = solar_flux[order]

    def _interp(
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_new: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.interp(x_new, x, y, left=0.0, right=0.0).astype(
            np.float64, copy=False
        )

    template_ids: list[str] = []
    filter_ids: list[str] = []
    integrals: list[float] = []

    for tmpl_id, wl_list, refl_list in zip(
        templates.template_id.to_pylist(),
        templates.wavelength_nm.to_pylist(),
        templates.reflectance.to_pylist(),
    ):
        tmpl_wl = np.asarray(wl_list, dtype=np.float64)
        tmpl_refl = np.asarray(refl_list, dtype=np.float64)
        for filt_id, filt_wl_list, thr_list in zip(
            curves.filter_id.to_pylist(),
            curves.wavelength_nm.to_pylist(),
            curves.throughput.to_pylist(),
        ):
            filt_wl = np.asarray(filt_wl_list, dtype=np.float64)
            filt_thr = np.asarray(thr_list, dtype=np.float64)

            wl_min = max(
                float(solar_wl.min()), float(tmpl_wl.min()), float(filt_wl.min())
            )
            wl_max = min(
                float(solar_wl.max()), float(tmpl_wl.max()), float(filt_wl.max())
            )
            if wl_max <= wl_min:
                val = float("nan")
            else:
                mask = (solar_wl >= wl_min) & (solar_wl <= wl_max)
                wl = solar_wl[mask]
                sun = solar_flux[mask]
                t = _interp(filt_wl, filt_thr, wl)
                r = _interp(tmpl_wl, tmpl_refl, wl)
                val = float(np.trapz(sun * r * t * wl, wl))

            template_ids.append(str(tmpl_id))
            filter_ids.append(str(filt_id))
            integrals.append(val)

    table = TemplateBandpassIntegrals.from_kwargs(
        template_id=template_ids,
        filter_id=filter_ids,
        integral_photon=integrals,
    )
    out_path = out_dir.joinpath("template_bandpass_integrals.parquet")
    table.to_parquet(out_path)
    return table
