import numpy as np
import pytest

from ...coordinates.origin import OriginCodes
from ...observers import get_observer_state
from ...observers.state import clear_observer_state_cache
from ...time import Timestamp
from ...utils.spice import clear_spkez_cache


@pytest.mark.parametrize(
    "times",
    [1, 10000],
    ids=lambda x: f"times={x},",
)
@pytest.mark.parametrize("code", ["X05", "500"], ids=lambda x: f"code={x},")
@pytest.mark.parametrize(
    "frame", ["equatorial", "ecliptic"], ids=lambda x: f"frame={x},"
)
@pytest.mark.parametrize(
    "origin",
    [OriginCodes.SUN, OriginCodes.SOLAR_SYSTEM_BARYCENTER],
    ids=lambda x: f"origin={x.name},",
)
@pytest.mark.benchmark(group="observer_states_compute")
def test_benchmark_get_observer_state_compute(benchmark, times, code, frame, origin):
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
    times = Timestamp.from_mjd(np.sort(times_array), scale="tdb")

    def clear_result_caches() -> None:
        clear_observer_state_cache()
        clear_spkez_cache()

    result = benchmark.pedantic(
        get_observer_state,
        args=(code, times),
        kwargs={"frame": frame, "origin": origin},
        setup=clear_result_caches,
        rounds=7,
        warmup_rounds=1,
        iterations=1,
    )
    assert len(result) == len(times)
