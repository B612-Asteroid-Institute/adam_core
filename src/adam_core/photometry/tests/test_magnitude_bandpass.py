import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...time import Timestamp
from ..bandpasses import get_integrals
from ..magnitude import convert_magnitude, predict_magnitudes


def _delta_mag(template_id: str, source_filter_id: str, target_filter_id: str) -> float:
    ints = get_integrals(
        template_id, np.asarray([source_filter_id, target_filter_id], dtype=object)
    )
    return float(-2.5 * np.log10(float(ints[1]) / float(ints[0])))


def test_convert_magnitude_matches_integral_ratio():
    m_v = np.asarray([20.0, 21.0], dtype=float)
    expected = m_v + _delta_mag("C", "V", "DECam_g")

    out = convert_magnitude(
        m_v,
        np.asarray(["V", "V"], dtype=object),
        np.asarray(["DECam_g", "DECam_g"], dtype=object),
        composition="C",
    )
    assert np.allclose(out, expected, rtol=0.0, atol=1e-12)


def test_convert_magnitude_mix_weights_match_named_template():
    m_v = np.asarray([20.0], dtype=float)
    out_named = convert_magnitude(
        m_v,
        np.asarray(["V"], dtype=object),
        np.asarray(["LSST_r"], dtype=object),
        composition="NEO",
    )
    out_mix = convert_magnitude(
        m_v,
        np.asarray(["V"], dtype=object),
        np.asarray(["LSST_r"], dtype=object),
        composition=(0.5, 0.5),
    )
    assert np.allclose(out_named, out_mix, rtol=0.0, atol=1e-12)


def test_predict_magnitudes_matches_geometry_plus_delta(monkeypatch):
    time = Timestamp.from_mjd([60000, 60000], scale="tdb")
    exposures = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=time,
        duration=[30.0, 30.0],
        # Bandpass predictor expects canonical filter IDs (not reported band labels).
        filter=["DECam_g", "LSST_r"],
        observatory_code=["W84", "X05"],
        seeing=[None, None],
        depth_5sigma=[None, None],
    )

    observers = Observers.from_kwargs(
        code=["500", "500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        ),
    )
    monkeypatch.setattr(exposures, "observers", lambda *args, **kwargs: observers)

    obj = CartesianCoordinates.from_kwargs(
        x=[2.0, 2.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        vx=[0.0, 0.0],
        vy=[0.0, 0.0],
        vz=[0.0, 0.0],
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
    )

    H_v = 15.0
    G = 0.15
    out = predict_magnitudes(
        H_v, obj, exposures, G=G, reference_filter="V", composition="C"
    )

    # Opposition geometry: r=2, delta=1, phase=0 => m_V = H + 5 log10(r*delta)
    m_v = H_v + 5.0 * np.log10(2.0 * 1.0)
    expected = np.asarray(
        [
            m_v + _delta_mag("C", "V", "DECam_g"),
            m_v + _delta_mag("C", "V", "LSST_r"),
        ],
        dtype=float,
    )
    assert np.allclose(out, expected, rtol=0.0, atol=1e-10)


def test_predict_magnitudes_reference_filter_conversion(monkeypatch):
    time = Timestamp.from_mjd([60000], scale="tdb")
    exposures = Exposures.from_kwargs(
        id=["e1"],
        start_time=time,
        duration=[30.0],
        filter=["DECam_g"],
        observatory_code=["W84"],
        seeing=[None],
        depth_5sigma=[None],
    )

    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ),
    )
    monkeypatch.setattr(exposures, "observers", lambda *args, **kwargs: observers)

    obj = CartesianCoordinates.from_kwargs(
        x=[2.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )

    H_v = 15.0
    # Convert H_V -> H_LSST_r using the same bandpass delta.
    H_r = H_v + _delta_mag("C", "V", "LSST_r")

    out_from_v = float(
        predict_magnitudes(H_v, obj, exposures, reference_filter="V", composition="C")[
            0
        ]
    )
    out_from_r = float(
        predict_magnitudes(
            H_r, obj, exposures, reference_filter="LSST_r", composition="C"
        )[0]
    )
    assert out_from_r == pytest.approx(out_from_v, abs=1e-10)
