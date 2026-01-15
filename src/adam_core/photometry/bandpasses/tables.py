from __future__ import annotations

import pyarrow as pa
import quivr as qv


class BandpassCurves(qv.Table):
    """
    Canonical bandpass response curves.

    Each row represents a single filter response curve: wavelength (nm) and a
    normalized dimensionless throughput (0..1).
    """

    filter_id = qv.LargeStringColumn()
    instrument = qv.LargeStringColumn()
    band = qv.LargeStringColumn()
    wavelength_nm = qv.LargeListColumn(pa.float64())
    throughput = qv.LargeListColumn(pa.float64())
    source = qv.LargeStringColumn()


class ObservatoryBandMap(qv.Table):
    """
    Map (MPC observatory_code, reported band) -> canonical filter_id.

    `key` is a convenience column: f"{observatory_code}|{reported_band}".
    """

    observatory_code = qv.LargeStringColumn()
    reported_band = qv.LargeStringColumn()
    filter_id = qv.LargeStringColumn()
    key = qv.LargeStringColumn()


class AsteroidTemplates(qv.Table):
    """
    Asteroid reflectance templates, including fixed population mixes.

    `reflectance` is dimensionless and should be normalized at 550 nm.
    """

    template_id = qv.LargeStringColumn()
    wavelength_nm = qv.LargeListColumn(pa.float64())
    reflectance = qv.LargeListColumn(pa.float64())
    weight_C = qv.Float64Column()
    weight_S = qv.Float64Column()
    citation = qv.LargeStringColumn()


class TemplateBandpassIntegrals(qv.Table):
    """
    Precomputed template×filter integrals.

    The stored scalar is intended for photon-counting synthetic photometry:
        I = ∫ F_sun(λ) * R_ast(λ) * T(λ) * λ dλ
    where F_sun is the adopted solar spectrum, R_ast is reflectance, and T is throughput.
    """

    template_id = qv.LargeStringColumn()
    filter_id = qv.LargeStringColumn()
    integral_photon = qv.Float64Column()
