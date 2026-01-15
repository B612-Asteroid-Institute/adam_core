import numpy as np
import pyarrow as pa
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observations.detections import PointSourceDetections
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...orbits.physical_parameters import PhysicalParameters
from ...time import Timestamp
from ..absolute_magnitude import estimate_absolute_magnitude_v_from_detections
from ..bandpasses.api import find_suggested_filter_bands
from ..magnitude import predict_magnitudes


def _make_geometry(n: int) -> tuple[CartesianCoordinates, Observers]:
    time = Timestamp.from_mjd(np.full(n, 60000), scale="tdb")
    observer = Observers.from_kwargs(
        code=["500"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.full(n, 1.0),
            y=np.zeros(n),
            z=np.zeros(n),
            vx=np.zeros(n),
            vy=np.zeros(n),
            vz=np.zeros(n),
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
    )

    # Place objects at opposition-like geometry to keep expected behavior stable.
    obj = CartesianCoordinates.from_kwargs(
        x=np.full(n, 2.0),
        y=np.zeros(n),
        z=np.zeros(n),
        vx=np.zeros(n),
        vy=np.zeros(n),
        vz=np.zeros(n),
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * n),
    )
    return obj, observer


def test_estimate_absolute_magnitude_missing_sigma(monkeypatch):
    # Two exposures, three detections (with duplication) to exercise exposure_id alignment.
    exp = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=Timestamp.from_mjd([60000, 60000], scale="tdb"),
        duration=[0.0, 0.0],
        # Non-canonical reported bands (exercise find_suggested_filter_bands fallback).
        filter=["g", "r"],
        observatory_code=["X05", "X05"],
        seeing=[None, None],
        depth_5sigma=[None, None],
    )
    det = PointSourceDetections.from_kwargs(
        id=["d1", "d2", "d3"],
        exposure_id=["e1", "e2", "e1"],
        time=Timestamp.from_mjd([60000, 60000, 60000], scale="tdb"),
        ra=[0.0, 0.0, 0.0],
        dec=[0.0, 0.0, 0.0],
        mag=[None, None, None],
        mag_sigma=[None, None, None],
    )

    obj, observer = _make_geometry(n=len(det))

    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        return observer

    monkeypatch.setattr(Exposures, "observers", fake_observers)

    # Generate synthetic observed magnitudes using the same forward model.
    exp_aligned = exp.take(pa.array([0, 1, 0], type=pa.int32()))
    canonical = find_suggested_filter_bands(
        exp_aligned.observatory_code, exp_aligned.filter
    )
    exp_canon = exp_aligned.set_column(
        "filter", pa.array(canonical.tolist(), type=pa.large_string())
    )

    H_true = 18.7
    mags_true = predict_magnitudes(
        H_true, obj, exp_canon, reference_filter="V", composition="NEO"
    )

    rng = np.random.default_rng(123)
    noise_sigma = 0.12
    mags_obs = np.asarray(mags_true, dtype=np.float64) + rng.normal(
        0.0, noise_sigma, size=len(det)
    )
    det = det.set_column("mag", mags_obs)

    est = estimate_absolute_magnitude_v_from_detections(
        det, exp, obj, composition="NEO", G=0.15, strict_band_mapping=False
    )

    assert isinstance(est, PhysicalParameters)
    assert len(est) == 1
    assert float(est.H_v[0].as_py()) == pytest.approx(H_true, abs=5e-2)
    assert est.H_v_sigma[0].as_py() is not None
    assert est.sigma_eff[0].as_py() is not None
    assert est.chi2_red[0].as_py() is None


def test_estimate_absolute_magnitude_with_sigma(monkeypatch):
    exp = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=Timestamp.from_mjd([60000, 60000], scale="tdb"),
        duration=[0.0, 0.0],
        # Canonical filter IDs (no mapping needed, but mapping should be pass-through).
        filter=["LSST_r", "DECam_g"],
        observatory_code=["X05", "W84"],
        seeing=[None, None],
        depth_5sigma=[None, None],
    )
    det = PointSourceDetections.from_kwargs(
        id=["d1", "d2", "d3", "d4"],
        exposure_id=["e1", "e2", "e2", "e1"],
        time=Timestamp.from_mjd([60000, 60000, 60000, 60000], scale="tdb"),
        ra=[0.0, 0.0, 0.0, 0.0],
        dec=[0.0, 0.0, 0.0, 0.0],
        mag=[None, None, None, None],
        mag_sigma=[None, None, None, None],
    )

    obj, observer = _make_geometry(n=len(det))

    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        return observer

    monkeypatch.setattr(Exposures, "observers", fake_observers)

    exp_aligned = exp.take(pa.array([0, 1, 1, 0], type=pa.int32()))
    H_true = 21.3
    mags_true = predict_magnitudes(
        H_true, obj, exp_aligned, reference_filter="V", composition="C"
    )

    sigma = np.full(len(det), 0.1, dtype=np.float64)
    rng = np.random.default_rng(456)
    mags_obs = np.asarray(mags_true, dtype=np.float64) + rng.normal(
        0.0, sigma, size=len(det)
    )
    det = det.set_column("mag", mags_obs).set_column("mag_sigma", sigma)

    est = estimate_absolute_magnitude_v_from_detections(
        det, exp, obj, composition="C", G=0.15, strict_band_mapping=False
    )

    assert isinstance(est, PhysicalParameters)
    assert len(est) == 1
    assert float(est.H_v[0].as_py()) == pytest.approx(H_true, abs=7e-2)
    assert est.chi2_red[0].as_py() is not None
    assert est.H_v_sigma[0].as_py() is not None
