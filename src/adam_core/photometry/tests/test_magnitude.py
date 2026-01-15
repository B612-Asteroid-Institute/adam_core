import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...time import Timestamp
from ..magnitude import (
    calculate_apparent_magnitude_v,
    convert_magnitude,
    predict_magnitudes,
)


def _as_scalar(x) -> float:
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr)
    if arr.size != 1:
        raise ValueError(f"Expected scalar/length-1 result, got shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def test_calculate_apparent_magnitude_v_geometry_sanity():
    """
    Sanity: farther objects are fainter; opposition is brighter than quadrature.
    """
    time = Timestamp.from_mjd([60000], scale="tdb")
    observer = Observers.from_kwargs(
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

    H = 15.0
    coords_near = CartesianCoordinates.from_kwargs(
        x=[0.0],
        y=[1.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    coords_far = CartesianCoordinates.from_kwargs(
        x=[0.0],
        y=[2.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    coords_opposition = CartesianCoordinates.from_kwargs(
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

    mag_near = _as_scalar(calculate_apparent_magnitude_v(H, coords_near, observer))
    mag_far = _as_scalar(calculate_apparent_magnitude_v(H, coords_far, observer))
    mag_opp = _as_scalar(calculate_apparent_magnitude_v(H, coords_opposition, observer))

    assert mag_far > mag_near
    assert mag_opp < mag_far


def test_convert_magnitude_requires_canonical_filter_ids():
    with pytest.raises(ValueError, match="Unknown filter_id"):
        convert_magnitude(
            np.asarray([20.0], dtype=float),
            np.asarray(["V"], dtype=object),
            np.asarray(["g"], dtype=object),
            composition="NEO",
        )


def test_predict_magnitudes_requires_composition(monkeypatch):
    time = Timestamp.from_mjd([60000], scale="tdb")
    exposures = Exposures.from_kwargs(
        id=["e1"],
        start_time=time,
        duration=[30.0],
        filter=["LSST_r"],
        observatory_code=["X05"],
        seeing=[None],
        depth_5sigma=[None],
    )

    observers = Observers.from_kwargs(
        code=["X05"],
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

    with pytest.raises(TypeError):
        # composition is required keyword-only
        predict_magnitudes(15.0, obj, exposures)  # type: ignore[misc]
