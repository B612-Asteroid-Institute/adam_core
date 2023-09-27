from ..observations import make_observations


def test_make_observations():
    # Test that we get two tables from make observations are
    # the expected length
    exposures, observations = make_observations()
    assert len(exposures) == 2412
    assert len(observations) == 2520
