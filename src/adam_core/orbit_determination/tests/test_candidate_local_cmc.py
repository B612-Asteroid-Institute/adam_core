"""Candidate-local CMC covariance refusal regressions for corrected IOD."""

import numpy as np
import pytest

from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination import run_iod
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
)
from adam_core.time import Timestamp

from .test_differential_correction import (
    EPOCH_MJD_TDB,
    TwoBodyPropagator,
    make_truth_orbit,
)


def _arc(times: list[float], *, seed: int) -> OrbitDeterminationObservations:
    rng = np.random.default_rng(seed)
    timestamp = Timestamp.from_mjd(np.asarray(times), scale="tdb")
    observers = Observers.from_code("500", timestamp)
    ephemeris = TwoBodyPropagator().generate_ephemeris(make_truth_orbit(), observers)
    longitude = ephemeris.coordinates.lon.to_numpy(zero_copy_only=False).copy()
    latitude = ephemeris.coordinates.lat.to_numpy(zero_copy_only=False).copy()
    sigma = 0.3 / 3600.0
    longitude += rng.normal(0.0, sigma, len(longitude)) / np.cos(np.radians(latitude))
    latitude += rng.normal(0.0, sigma, len(latitude))
    covariance = np.full((len(longitude), 6, 6), np.nan)
    covariance[:, 1, 1] = sigma**2 / np.cos(np.radians(latitude)) ** 2
    covariance[:, 2, 2] = sigma**2
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"o{row:03d}" for row in range(len(longitude))],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=longitude,
            lat=latitude,
            time=timestamp,
            covariance=CoordinateCovariances.from_matrix(covariance),
            origin=Origin.from_kwargs(code=["500"] * len(longitude)),
            frame="equatorial",
        ),
        observers=observers,
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * len(longitude),
            rmsmag=[None] * len(longitude),
            band=[None] * len(longitude),
        ),
    )


def _random_case(seed: int, case: int) -> OrbitDeterminationObservations:
    rng = np.random.default_rng(seed)
    t0 = EPOCH_MJD_TDB + 5.0
    for current in range(case + 1):
        nights = int(rng.choice([3, 4, 5, 6, 8]))
        gaps = rng.uniform(1.0, 20.0, nights - 1)
        starts = np.concatenate([[0.0], np.cumsum(gaps)])
        per_night = rng.integers(2, 5, nights)
        times = sorted(
            t0 + start + row * rng.uniform(0.014, 0.028)
            for start, count in zip(starts, per_night)
            for row in range(count)
        )
        if current == case:
            return _arc(times, seed=case)
    raise AssertionError("unreachable")


@pytest.mark.parametrize(("seed", "case"), [(20261021, 43), (20261021, 70), (7, 36)])
def test_corrected_iod_continues_after_candidate_local_cmc_refusal(seed, case):
    fitted, members, report = run_iod(
        _random_case(seed, case),
        TwoBodyPropagator(),
    )
    assert report.found
    assert len(fitted) == 1
    assert len(members) > 0
    state = fitted.coordinates.values.to_numpy(zero_copy_only=False)
    assert np.isfinite(state).all()
    assert report.candidates_evaluated > 1
    assert any(
        "CMC2003 expected covariance unavailable" in str(failure)
        for failure in report.candidate_failures
    )
