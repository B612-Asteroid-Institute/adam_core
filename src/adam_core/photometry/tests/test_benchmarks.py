import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..simple_magnitude import calculate_apparent_magnitude, calculate_apparent_magnitude_jax


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


@pytest.mark.parametrize(
    "n",
    [256, 2048, 16384, 32768],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_simple_magnitude_apparent")
def test_benchmark_calculate_apparent_magnitude_numpy(benchmark, n):
    pad_to = 2048
    n_padded = _padded_size(n, pad_to)
    H, obj, observer, G = _make_benchmark_case(n=n_padded)

    def run():
        return calculate_apparent_magnitude(H, obj, observer, G=G, output_filter="V")

    out = benchmark(run)
    out = out[:n]  # window back to requested n
    assert len(out) == n


@pytest.mark.parametrize(
    "n",
    [256, 2048, 16384, 32768],
    ids=lambda x: f"n={x}",
)
@pytest.mark.benchmark(group="photometry_simple_magnitude_apparent")
def test_benchmark_calculate_apparent_magnitude_jax(benchmark, n):
    pad_to = 2048
    n_padded = _padded_size(n, pad_to)
    H, obj, observer, G = _make_benchmark_case(n=n_padded)

    def run():
        out = calculate_apparent_magnitude_jax(H, obj, observer, G=G, output_filter="V")
        return out

    out = benchmark(run)
    out = out[:n]  # window back to requested n
    assert len(out) == n