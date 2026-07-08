"""W12 verification: quivr<->Rust typed mapping parity on the existing boundary.

These tests close the W12 verification scope (bead personal-cmy.10) without an
Arrow transport rewrite: they confirm the dict/numpy boundary already maps quivr
tables into the Rust-canonical typed contracts (OrbitVariantBatch) with matching
row order, and that non-TDB (UTC) target rescaling rides the provider/UTC path
and matches Python adam_assist public semantics.
"""

from __future__ import annotations

import numpy as np

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp


def _covariance_orbits() -> Orbits:
    sigmas = np.array(
        [
            [1.0e-9, 2.0e-9, 3.0e-9, 1.0e-10, 2.0e-10, 3.0e-10],
            [1.5e-9, 2.5e-9, 3.5e-9, 1.5e-10, 2.5e-10, 3.5e-10],
        ],
        dtype=np.float64,
    )
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.05, 1.10],
        y=[0.0, 0.02],
        z=[0.0, -0.001],
        vx=[0.0, -0.0002],
        vy=[0.016787, 0.0159],
        vz=[0.0, 0.0001],
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        time=Timestamp.from_mjd([60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["orbit-a", "orbit-b"],
        object_id=["orbit-a", "orbit-b"],
        coordinates=coordinates,
    )


def test_variant_orbits_mapping_and_order_match_python_public(
    python_reference_propagator,
) -> None:
    variants = VariantOrbits.create(_covariance_orbits(), method="sigma-point")
    times = Timestamp.from_mjd([60000.25, 60001.0], scale="tdb")

    expected = python_reference_propagator.propagate_orbits(
        variants, times, max_processes=1, chunk_size=100
    )
    actual = RustASSISTPropagator().propagate_orbits(
        variants, times, max_processes=1, chunk_size=100
    )

    assert isinstance(actual, VariantOrbits)
    assert actual.orbit_id.to_pylist() == expected.orbit_id.to_pylist()
    assert actual.variant_id.to_pylist() == expected.variant_id.to_pylist()
    np.testing.assert_array_equal(
        actual.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        expected.coordinates.time.mjd().to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(
        actual.coordinates.values,
        expected.coordinates.values,
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(
        actual.weights.to_numpy(zero_copy_only=False),
        expected.weights.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_array_equal(
        actual.weights_cov.to_numpy(zero_copy_only=False),
        expected.weights_cov.to_numpy(zero_copy_only=False),
    )


def test_non_tdb_utc_target_rescaling_matches_python_public(
    python_reference_propagator,
) -> None:
    orbits = _covariance_orbits()
    utc_times = Timestamp.from_mjd([60000.25, 60001.0], scale="utc")

    expected = python_reference_propagator.propagate_orbits(
        orbits, utc_times, max_processes=1, chunk_size=100
    )
    actual = RustASSISTPropagator().propagate_orbits(
        orbits, utc_times, max_processes=1, chunk_size=100
    )

    assert actual.coordinates.time.scale == "utc"
    assert actual.orbit_id.to_pylist() == expected.orbit_id.to_pylist()
    np.testing.assert_array_equal(
        actual.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        expected.coordinates.time.mjd().to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(
        actual.coordinates.values,
        expected.coordinates.values,
        atol=1.0e-11,
        rtol=0,
    )
