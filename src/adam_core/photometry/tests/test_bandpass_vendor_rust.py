import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import requests

from adam_core import _rust_native
from adam_core._rust.arrow import table_from_record_batch

from ..bandpasses.tables import (
    AsteroidTemplates,
    BandpassCurves,
    ObservatoryBandMap,
    TemplateBandpassIntegrals,
)
from ..bandpasses.vendor import (
    _SvoBandpassSpec,
    build_asteroid_templates,
    build_bandpass_curves,
    build_observatory_band_map,
    build_template_bandpass_integrals,
)

DATA = Path(__file__).parents[1] / "bandpasses" / "data"
VENDOR_FIXTURES = Path(__file__).parent / "data" / "vendor"


def test_recorded_svo_votable_is_parsed_normalized_and_published_by_rust(tmp_path):
    payload = (VENDOR_FIXTURES / "svo_bessell_v.xml").read_bytes()
    batch = _rust_native.build_bandpass_curves_arrow(
        str(tmp_path),
        [("V", "Bessell", "V", "Generic/Bessell.V")],
        60,
        [payload],
    )
    built = table_from_record_batch(BandpassCurves, batch)
    expected = BandpassCurves.from_parquet(DATA / "bandpass_curves.parquet").select(
        "filter_id", "V"
    )
    assert built.filter_id.to_pylist() == ["V"]
    np.testing.assert_allclose(
        np.asarray(built.wavelength_nm[0].as_py()),
        np.asarray(expected.wavelength_nm[0].as_py()),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(built.throughput[0].as_py()),
        np.asarray(expected.throughput[0].as_py()),
        rtol=0,
        atol=0,
    )
    assert (tmp_path / "bandpass_curves.parquet").is_file()


def test_recorded_solar_fits_is_parsed_normalized_and_published_by_rust(tmp_path):
    payload = (VENDOR_FIXTURES / "solar_spec.fits").read_bytes()
    batch = _rust_native.build_solar_spectrum_arrow(str(tmp_path), 120, payload)
    built = batch.to_pandas()
    expected = pq.read_table(DATA / "solar_spectrum.parquet").to_pandas()
    np.testing.assert_array_equal(built["wavelength_nm"], expected["wavelength_nm"])
    np.testing.assert_array_equal(built["flux"], expected["flux"])
    assert (tmp_path / "solar_spectrum.parquet").is_file()


def test_recorded_svo_errors_are_compatible_and_atomic(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "No bandpass specs provided"):
        _rust_native.build_bandpass_curves_arrow(str(tmp_path), [], 60)
    with np.testing.assert_raises_regex(ValueError, "Duplicate filter_id in specs"):
        _rust_native.build_bandpass_curves_arrow(
            str(tmp_path),
            [
                ("V", "Bessell", "V", "Generic/Bessell.V"),
                ("V", "Bessell", "V", "Generic/Bessell.V"),
            ],
            60,
            [
                (VENDOR_FIXTURES / "svo_bessell_v.xml").read_bytes(),
                (VENDOR_FIXTURES / "svo_bessell_v.xml").read_bytes(),
            ],
        )
    with np.testing.assert_raises_regex(ValueError, "Unexpected SVO schema"):
        _rust_native.build_bandpass_curves_arrow(
            str(tmp_path),
            [("V", "Bessell", "V", "Generic/Bessell.V")],
            60,
            [b"<VOTABLE/>"],
        )
    assert not (tmp_path / "bandpass_curves.parquet").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_public_http_error_translation_preserves_requests_type(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("status code 503")

    monkeypatch.setattr(_rust_native, "build_bandpass_curves_arrow", fail)
    with pytest.raises(requests.HTTPError, match="status code 503"):
        build_bandpass_curves(
            tmp_path,
            specs=[_SvoBandpassSpec("V", "Bessell", "V", "Generic/Bessell.V")],
        )


def test_build_observatory_band_map_matches_frozen_product(tmp_path):
    built = build_observatory_band_map(tmp_path)
    expected = ObservatoryBandMap.from_parquet(DATA / "observatory_band_map.parquet")
    assert built.table.equals(expected.table)
    assert (tmp_path / "observatory_band_map.parquet").is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_build_asteroid_templates_matches_frozen_product(tmp_path):
    built = build_asteroid_templates(tmp_path)
    expected = AsteroidTemplates.from_parquet(DATA / "asteroid_templates.parquet")
    assert built.template_id.to_pylist() == expected.template_id.to_pylist()
    assert built.weight_C.to_pylist() == expected.weight_C.to_pylist()
    assert built.weight_S.to_pylist() == expected.weight_S.to_pylist()
    assert built.citation.to_pylist() == expected.citation.to_pylist()
    np.testing.assert_allclose(
        np.asarray(built.wavelength_nm.to_pylist()),
        np.asarray(expected.wavelength_nm.to_pylist()),
        rtol=0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.asarray(built.reflectance.to_pylist()),
        np.asarray(expected.reflectance.to_pylist()),
        rtol=2e-15,
        atol=2e-15,
    )
    assert (tmp_path / "asteroid_templates.parquet").is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_build_template_bandpass_integrals_matches_frozen_product(tmp_path):
    for filename in [
        "bandpass_curves.parquet",
        "asteroid_templates.parquet",
        "solar_spectrum.parquet",
    ]:
        shutil.copyfile(DATA / filename, tmp_path / filename)
    built = build_template_bandpass_integrals(tmp_path)
    expected = TemplateBandpassIntegrals.from_parquet(
        DATA / "template_bandpass_integrals.parquet"
    )
    assert built.template_id.to_pylist() == expected.template_id.to_pylist()
    assert built.filter_id.to_pylist() == expected.filter_id.to_pylist()
    np.testing.assert_allclose(
        built.integral_photon.to_numpy(),
        expected.integral_photon.to_numpy(),
        rtol=2e-14,
        atol=1e-8,
    )
    assert list(tmp_path.glob("*.tmp")) == []


def test_bandpass_vendor_products_have_rust_owned_timing(tmp_path):
    for filename in [
        "bandpass_curves.parquet",
        "asteroid_templates.parquet",
        "solar_spectrum.parquet",
    ]:
        shutil.copyfile(DATA / filename, tmp_path / filename)
    samples = _rust_native.benchmark_bandpass_vendor_products(
        str(tmp_path),
        (VENDOR_FIXTURES / "svo_bessell_v.xml").read_bytes(),
        (VENDOR_FIXTURES / "solar_spec.fits").read_bytes(),
        2,
        2,
        1,
    )
    assert len(samples) == 2
    assert all(len(trial) == 2 for trial in samples)
    assert all(sample > 0 for trial in samples for sample in trial)


def test_deterministic_builders_leave_no_partial_file_on_error(tmp_path):
    missing = tmp_path / "missing"
    for builder, filename in [
        (build_observatory_band_map, "observatory_band_map.parquet"),
        (build_asteroid_templates, "asteroid_templates.parquet"),
        (build_template_bandpass_integrals, "template_bandpass_integrals.parquet"),
    ]:
        try:
            builder(missing)
        except RuntimeError:
            pass
        else:
            raise AssertionError("builder should reject a missing output directory")
        assert not (missing / filename).exists()
        assert list(tmp_path.rglob("*.tmp")) == []
