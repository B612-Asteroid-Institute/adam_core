"""Collision/impact parity: Rust `adam_assist_rust` vs Python `adam_assist`.

Bead personal-cmy.9. The Rust `_detect_collisions` mirrors the legacy Python
step loop (same input transform, same TDB Julian-date horizon arithmetic, same
per-step distance checks, same overshoot-report semantics) through one Rust
crossing.

Tolerance policy: the two implementations run different libassist C builds
(pip `adam_assist` bundles its own; `libassist-sys` compiles crates.io
sources), so raw IAS15 step sequences diverge at the last-bit force level and
bit parity is not attainable. The physically meaningful contract is gated
tightly: identical impact sets, impact times within ~1 minute over multi-day
horizons, impact states within 5e-4 AU, and identical survivor sets. Survivor
states are reported at each implementation's own final (overshooting) step,
so they are compared through the step-time difference.
Measured 2026-07-03 (this fixture): impact time diff 1.5e-4 d, impact state
diff 7.7e-5 AU, survivor overshoot diff 0.18 d at dt~4 d.
"""

import numpy as np
import pytest
from adam_assist import ASSISTPropagator as PythonASSISTPropagator
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import OriginCodes
from adam_core.dynamics.impacts import (
    EARTH_RADIUS_KM,
    CollisionConditions,
    calculate_impact_probabilities,
    calculate_impacts,
)
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp
from adam_core.utils.spice import get_perturber_state

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator

EPOCH_MJD = 60000.0
NUM_DAYS = 30
IMPACT_TIME_ATOL_DAYS = 1.0e-3
IMPACT_STATE_ATOL_AU = 5.0e-4
SURVIVOR_OVERSHOOT_ATOL_DAYS = 1.0


def _earth_state() -> np.ndarray:
    epoch = Timestamp.from_mjd([EPOCH_MJD], scale="tdb")
    earth = get_perturber_state(
        OriginCodes.EARTH, epoch, frame="ecliptic", origin=OriginCodes.SUN
    )
    return earth.values[0]


def _impact_study_orbits() -> Orbits:
    """One radial-infall impactor (~60,000 km from Earth center, matching
    Earth's velocity) and one clearly safe orbit 0.2 AU away."""
    earth = _earth_state()
    impactor = earth.copy()
    impactor[0] += 4.0e-4
    safe = earth.copy()
    safe[0] += 0.2
    safe[4] += 0.005
    values = np.stack([impactor, safe])
    return Orbits.from_kwargs(
        orbit_id=["impactor", "safe"],
        object_id=["i", "s"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=Timestamp.from_mjd([EPOCH_MJD] * 2, scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )


def _earth_conditions() -> CollisionConditions:
    return CollisionConditions.from_kwargs(
        condition_id=["Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[EARTH_RADIUS_KM],
        stopping_condition=[True],
    )


@pytest.fixture(scope="module")
def python_propagator() -> PythonASSISTPropagator:
    return PythonASSISTPropagator()


@pytest.fixture(scope="module")
def rust_propagator() -> RustASSISTPropagator:
    return RustASSISTPropagator()


def test_detect_collisions_orbits_parity(python_propagator, rust_propagator):
    orbits = _impact_study_orbits()
    conditions = _earth_conditions()

    py_results, py_events = python_propagator._detect_collisions(
        orbits, NUM_DAYS, conditions
    )
    rust_results, rust_events = rust_propagator._detect_collisions(
        orbits, NUM_DAYS, conditions
    )

    # Identical impact sets and survivor sets.
    assert py_events.orbit_id.to_pylist() == ["impactor"]
    assert rust_events.orbit_id.to_pylist() == ["impactor"]
    assert sorted(py_results.orbit_id.to_pylist()) == sorted(
        rust_results.orbit_id.to_pylist()
    )

    # Impact epoch and state agree within cross-build step-sequence noise.
    py_time = py_events.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    rust_time = rust_events.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(rust_time, py_time, rtol=0, atol=IMPACT_TIME_ATOL_DAYS)
    np.testing.assert_allclose(
        rust_events.coordinates.values,
        py_events.coordinates.values,
        rtol=0,
        atol=IMPACT_STATE_ATOL_AU,
    )

    # Both implementations report the impact within the collision radius.
    for events in (py_events, rust_events):
        spherical = events.collision_coordinates
        assert (
            spherical.rho.to_numpy(zero_copy_only=False) * 149_597_870.7
            <= EARTH_RADIUS_KM
        ).all()

    # Survivors are reported at each implementation's own overshooting final
    # step; the step times must agree to within one large heliocentric step.
    py_final = py_results.select("orbit_id", "safe")
    rust_final = rust_results.select("orbit_id", "safe")
    py_final_time = py_final.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    rust_final_time = rust_final.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    assert (py_final_time >= EPOCH_MJD + NUM_DAYS).all()
    assert (rust_final_time >= EPOCH_MJD + NUM_DAYS).all()
    np.testing.assert_allclose(
        rust_final_time, py_final_time, rtol=0, atol=SURVIVOR_OVERSHOOT_ATOL_DAYS
    )


def test_detect_collisions_variants_parity(python_propagator, rust_propagator):
    """Tightly-clustered Monte Carlo variants around the impactor must all
    impact in both implementations; variants of the safe orbit must all
    survive. Variant generation is seeded so both propagators see identical
    inputs."""
    orbits = _impact_study_orbits()
    tiny = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])
    covariance = np.stack([tiny, tiny])
    coordinates = orbits.coordinates.set_column(
        "covariance", CoordinateCovariances.from_matrix(covariance)
    )
    orbits = orbits.set_column("coordinates", coordinates)
    variants = VariantOrbits.create(
        orbits, method="monte-carlo", num_samples=10, seed=42
    )
    conditions = _earth_conditions()

    py_results, py_events = python_propagator._detect_collisions(
        variants, NUM_DAYS, conditions
    )
    rust_results, rust_events = rust_propagator._detect_collisions(
        variants, NUM_DAYS, conditions
    )

    def keys(events):
        return sorted(zip(events.orbit_id.to_pylist(), events.variant_id.to_pylist()))

    assert len(py_events) == 10
    assert keys(py_events) == keys(rust_events)
    assert sorted(py_results.orbit_id.to_pylist()) == sorted(
        rust_results.orbit_id.to_pylist()
    )

    py_time = np.sort(py_events.coordinates.time.mjd().to_numpy(zero_copy_only=False))
    rust_time = np.sort(
        rust_events.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    )
    np.testing.assert_allclose(rust_time, py_time, rtol=0, atol=IMPACT_TIME_ATOL_DAYS)


def test_calculate_impacts_end_to_end(rust_propagator):
    """The Rust propagator satisfies the public ImpactMixin contract:
    calculate_impacts + calculate_impact_probabilities run end-to-end and the
    deep impactor yields probability 1, the safe orbit 0."""
    orbits = _impact_study_orbits()
    tiny = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])
    covariance = np.stack([tiny, tiny])
    coordinates = orbits.coordinates.set_column(
        "covariance", CoordinateCovariances.from_matrix(covariance)
    )
    orbits = orbits.set_column("coordinates", coordinates)

    variants, events = calculate_impacts(
        orbits,
        NUM_DAYS,
        rust_propagator,
        num_samples=10,
        seed=7,
        conditions=_earth_conditions(),
    )
    probabilities = calculate_impact_probabilities(
        variants, events, conditions=_earth_conditions()
    )
    by_orbit = {
        orbit_id: probability
        for orbit_id, probability in zip(
            probabilities.orbit_id.to_pylist(),
            probabilities.cumulative_probability.to_pylist(),
        )
    }
    assert by_orbit["impactor"] == pytest.approx(1.0)
    assert by_orbit.get("safe", 0.0) == pytest.approx(0.0)
