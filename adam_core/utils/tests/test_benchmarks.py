import numpy as np
import pytest

from ...coordinates.origin import OriginCodes
from ...time import Timestamp
from ...utils import get_perturber_state


@pytest.mark.parametrize(
    "times",
    [1, 10000],
    ids=lambda x: f"times={x},",
)
@pytest.mark.parametrize(
    "perturber",
    [OriginCodes.EARTH, OriginCodes.SUN, OriginCodes.SOLAR_SYSTEM_BARYCENTER],
    ids=lambda x: f"perturber={x.name},",
)
@pytest.mark.parametrize(
    "frame", ["equatorial", "ecliptic"], ids=lambda x: f"frame={x},"
)
@pytest.mark.parametrize(
    "origin",
    [OriginCodes.SUN, OriginCodes.SOLAR_SYSTEM_BARYCENTER],
    ids=lambda x: f"origin={x.name},",
)
@pytest.mark.benchmark(group="observer_states")
def test_benchmark_get_perturber_state(benchmark, times, perturber, frame, origin):
    # We can expect needing to get the observer states for duplicated observation
    # times in the future, so we should benchmark this case
    if times == 10000:
        # 1000 times duplicated 10 times
        repeats = 10
    else:
        # 1 time not duplicated
        repeats = 1

    times_array = np.concatenate(
        [np.arange(60000, 60000 + np.minimum(times, 1000), 1) for i in range(repeats)]
    )
    times_array = Timestamp.from_mjd(np.sort(times_array), scale="tdb")

    result = benchmark(
        get_perturber_state,
        perturber,
        times_array,
        frame=frame,
        origin=origin,
    )
    assert len(result) == len(times_array)
