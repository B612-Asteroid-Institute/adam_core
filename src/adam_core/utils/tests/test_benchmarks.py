import numpy as np
import pytest

from ...coordinates.origin import OriginCodes
from ...time import Timestamp
from ...utils import get_perturber_state
from ...utils.spice import clear_spkez_cache


@pytest.mark.parametrize(
    "times",
    [1, 10000],
    ids=lambda x: f"times={x},",
)
@pytest.mark.parametrize(
    ("perturber", "origin"),
    [
        (OriginCodes.EARTH, OriginCodes.SUN),
        (OriginCodes.EARTH, OriginCodes.SOLAR_SYSTEM_BARYCENTER),
        (OriginCodes.SUN, OriginCodes.SOLAR_SYSTEM_BARYCENTER),
        (OriginCodes.SOLAR_SYSTEM_BARYCENTER, OriginCodes.SUN),
    ],
    ids=[
        "perturber=EARTH,origin=SUN,",
        "perturber=EARTH,origin=SOLAR_SYSTEM_BARYCENTER,",
        "perturber=SUN,origin=SOLAR_SYSTEM_BARYCENTER,",
        "perturber=SOLAR_SYSTEM_BARYCENTER,origin=SUN,",
    ],
)
@pytest.mark.parametrize(
    "frame", ["equatorial", "ecliptic"], ids=lambda x: f"frame={x},"
)
@pytest.mark.benchmark(group="perturber_states_compute")
def test_benchmark_get_perturber_state_compute(
    benchmark, times, perturber, frame, origin
):
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

    result = benchmark.pedantic(
        get_perturber_state,
        args=(perturber, times_array),
        kwargs={"frame": frame, "origin": origin},
        setup=clear_spkez_cache,
        rounds=7,
        warmup_rounds=1,
        iterations=1,
    )
    assert len(result) == len(times_array)
