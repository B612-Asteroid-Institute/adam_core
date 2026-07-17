import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..magnitude import (
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_and_phase_angle,
    calculate_phase_angle,
    convert_magnitude,
)


def _make_benchmark_case(n: int = 2048):
    rng = np.random.default_rng(123)
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

    obj = CartesianCoordinates.from_kwargs(
        x=rng.uniform(1.2, 3.0, size=n),
        y=rng.uniform(0.1, 2.0, size=n),
        z=rng.uniform(-0.5, 0.5, size=n),
        vx=np.zeros(n),
        vy=np.zeros(n),
        vz=np.zeros(n),
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * n),
    )

    H = rng.uniform(10.0, 25.0, size=n)
    G = rng.uniform(0.0, 1.0, size=n)
    return H, obj, observer, G


def _make_convert_case(n: int = 262144):
    rng = np.random.default_rng(123)
    mags = rng.uniform(10.0, 25.0, size=n).astype(float)

    # Typical usage: many V-band magnitudes converted to per-observation filters.
    source = np.full(n, "V", dtype=object)
    targets = np.array(
        ["V", "SDSS_g", "SDSS_r", "SDSS_i", "LSST_g", "LSST_r", "DECam_r"],
        dtype=object,
    )
    target = rng.choice(targets, size=n).astype(object)
    return mags, source, target


@pytest.mark.parametrize(
    "n",
    [256, 16384, 65536, 262144, 1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_magnitude_apparent")
def test_benchmark_calculate_apparent_magnitude(benchmark, n):
    H, obj, observer, G = _make_benchmark_case(n=n)

    def run():
        out = calculate_apparent_magnitude_v(H, obj, observer, G=G)
        return out

    out = benchmark(run)
    assert len(out) == n


@pytest.mark.parametrize(
    "n",
    [256, 16384, 65536, 262144, 1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_phase_angle")
def test_benchmark_calculate_phase_angle(benchmark, n):
    _, obj, observer, _ = _make_benchmark_case(n=n)

    def run():
        out = calculate_phase_angle(obj, observer)
        return out

    out = benchmark(run)
    assert len(out) == n


@pytest.mark.parametrize(
    "n",
    [256, 16384, 65536, 262144, 1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_magnitude_apparent_and_phase_angle")
def test_benchmark_calculate_apparent_magnitude_and_phase(benchmark, n):
    H, obj, observer, G = _make_benchmark_case(n=n)

    def run():
        mags, alpha = calculate_apparent_magnitude_v_and_phase_angle(
            H, obj, observer, G=G
        )
        return mags, alpha

    mags, alpha = benchmark(run)
    mags = np.asarray(mags)
    alpha = np.asarray(alpha)
    assert len(mags) == n
    assert len(alpha) == n


@pytest.mark.parametrize(
    "n",
    [256, 16384, 65536, 262144, 1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_magnitude_convert_magnitude")
def test_benchmark_convert_magnitude_bandpass(benchmark, n):
    mags, src, tgt = _make_convert_case(n=n)

    def run():
        return convert_magnitude(mags, src, tgt, composition="NEO")

    out = benchmark(run)
    out = np.asarray(out)
    assert len(out) == n
