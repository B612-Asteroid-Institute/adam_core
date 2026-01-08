import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..simple_magnitude import (
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_jax,
    convert_magnitude,
    convert_magnitude_jax,
)


def _padded_size(n: int, pad_to: int) -> int:
    """Round n up to the next multiple of pad_to."""
    if pad_to <= 0:
        raise ValueError("pad_to must be > 0")
    return int(((n + pad_to - 1) // pad_to) * pad_to)


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
        # Keep this list restricted to filters that have a valid conversion path from "V"
        # via FILTER_CONVERSIONS.
        ["V", "g", "r", "i", "LSST_g", "LSST_r", "DECam_r"], dtype=object
    )
    target = rng.choice(targets, size=n).astype(object)
    return mags, source, target


@pytest.mark.parametrize(
    "n",
    [256,  16384,  65536,  262144,  1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_simple_magnitude_apparent")
def test_benchmark_calculate_apparent_magnitude_numpy(benchmark, n):
    pad_to = 2048
    n_padded = _padded_size(n, pad_to)
    H, obj, observer, G = _make_benchmark_case(n=n_padded)

    def run():
        return calculate_apparent_magnitude_v(H, obj, observer, G=G)

    out = benchmark(run)
    out = out[:n]  # window back to requested n
    assert len(out) == n


@pytest.mark.parametrize(
    "n",
    [256,  16384,  65536,  262144,  1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_simple_magnitude_apparent")
def test_benchmark_calculate_apparent_magnitude_jax(benchmark, n):
    pad_to = 2048
    n_padded = _padded_size(n, pad_to)
    H, obj, observer, G = _make_benchmark_case(n=n_padded)

    def run():
        out = calculate_apparent_magnitude_v_jax(H, obj, observer, G=G)
        return out

    out = benchmark(run)
    out = out[:n]  # window back to requested n
    assert len(out) == n


@pytest.mark.parametrize(
    "n",
    [256, 16384, 65536, 262144, 1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_simple_magnitude_convert_magnitude")
def test_benchmark_convert_magnitude_numpy(benchmark, n):
    pad_to = 2048
    n_padded = _padded_size(n, pad_to)
    mags, src, tgt = _make_convert_case(n=n_padded)

    def run():
        return convert_magnitude(mags, src, tgt)

    out = benchmark(run)
    out = out[:n]
    assert len(out) == n


@pytest.mark.parametrize(
    "n",
    [256, 16384, 65536, 262144, 1048576],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_simple_magnitude_convert_magnitude")
def test_benchmark_convert_magnitude_jax(benchmark, n):
    pad_to = 2048
    n_padded = _padded_size(n, pad_to)
    mags, src, tgt = _make_convert_case(n=n_padded)

    # Warm-up to exclude JIT compilation from the benchmark measurement.
    warm = convert_magnitude_jax(mags, src, tgt)
    warm.block_until_ready()

    def run():
        out = convert_magnitude_jax(mags, src, tgt)
        out.block_until_ready()
        return out

    out = benchmark(run)
    out = np.asarray(out)[:n]
    assert len(out) == n