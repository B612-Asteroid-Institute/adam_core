"""
Bandpass response curves and asteroid spectral templates.

This subpackage vendors (as package data) a curated set of:
- instrument filter response curves (throughput vs wavelength), normalized and stored in Parquet
- asteroid reflectance templates (C, S, and mixes like NEO/MBA), also stored in Parquet
- precomputed template×filter integrals used to derive color terms efficiently

The initial implementation is data-only: it is designed so `magnitude.py` can
later resolve `(observatory_code, reported_band)` -> canonical `filter_id` and use
precomputed integrals for higher-fidelity conversions, without changing behavior yet.
"""

from .api import (
    assert_filter_ids_have_curves,
    bandpass_color_terms,
    bandpass_delta_mag,
    compute_mix_integrals,
    get_integrals,
    load_asteroid_templates,
    load_bandpass_curves,
    load_observatory_band_map,
    load_template_integrals,
    map_to_canonical_filter_bands,
    register_custom_template,
)
from .tables import (
    AsteroidTemplates,
    BandpassCurves,
    ObservatoryBandMap,
    TemplateBandpassIntegrals,
)

__all__ = [
    # Tables
    "BandpassCurves",
    "ObservatoryBandMap",
    "AsteroidTemplates",
    "TemplateBandpassIntegrals",
    # Loaders
    "load_bandpass_curves",
    "load_observatory_band_map",
    "load_asteroid_templates",
    "load_template_integrals",
    # Utilities
    "map_to_canonical_filter_bands",
    "assert_filter_ids_have_curves",
    "get_integrals",
    "compute_mix_integrals",
    "bandpass_delta_mag",
    "bandpass_color_terms",
    "register_custom_template",
]
