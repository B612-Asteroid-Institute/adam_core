"""W1 data-model bridge parity (Beads personal-cmy.13, mechanism C).

Round-trips a real quivr ``Orbits`` table through the Rust-canonical
``OrbitBatch`` via Arrow IPC bytes (a single Python<->Rust crossing of the
complete nested schema) and asserts the data AND the Arrow schema (ignoring
metadata) survive exactly, then exercises a real Rust-native workflow
(frame rotation) over the same bridge. Uses the productionized bridge in
``adam_core.orbits.arrow_bridge`` rather than re-implementing the boundary.
"""

import numpy as np
import pyarrow as pa

from adam_core import _rust_native as rn
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    transform_coordinates,
)
from adam_core.orbits import Orbits
from adam_core.dynamics import propagate_2body
from adam_core.orbits.arrow_bridge import (
    orbits_to_ipc,
    propagate_orbits_2body,
    rotate_orbits_frame,
    round_trip_orbits,
    round_trip_orbits_zero_copy,
    sample_orbit_variants,
)
from adam_core.orbits.orbits import PhysicalParameters
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp


def _read_ipc_table(raw: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        return reader.read_all().combine_chunks()


def _assert_lossless(orbits: Orbits) -> None:
    tin = orbits.table.combine_chunks()
    tout = _read_ipc_table(rn.orbits_nested_ipc_round_trip(orbits_to_ipc(orbits)))
    # Full nested data survives quivr -> IPC -> Rust OrbitBatch -> IPC -> quivr.
    assert tout.to_pylist() == tin.to_pylist()
    # Arrow schema (types + nullability) is byte-identical, ignoring metadata.
    assert tout.schema.equals(tin.schema, check_metadata=False)


def _cartesian(with_covariance: bool) -> CartesianCoordinates:
    n = 3
    kwargs = dict(
        x=[1.0, 4.0, 7.0],
        y=[2.0, 5.0, 8.0],
        z=[3.0, 6.0, 9.0],
        vx=[0.1, 0.4, 0.7],
        vy=[0.2, 0.5, 0.8],
        vz=[0.3, 0.6, 0.9],
        time=Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )
    if with_covariance:
        cov = np.stack(
            [np.arange(36, dtype=float).reshape(6, 6) + i * 100 for i in range(n)]
        )
        kwargs["covariance"] = CoordinateCovariances.from_matrix(cov)
    return CartesianCoordinates.from_kwargs(**kwargs)


def _orbits(with_covariance: bool = True, physical: bool = False) -> Orbits:
    kwargs = dict(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", None, "c"],
        coordinates=_cartesian(with_covariance=with_covariance),
    )
    if physical:
        kwargs["physical_parameters"] = PhysicalParameters.from_kwargs(
            H_v=[15.5, 16.0, 17.0],
            H_v_sigma=[0.1, None, 0.3],
            G=[0.15, 0.15, 0.15],
            G_sigma=[None, None, None],
            sigma_eff=[0.05, 0.06, 0.07],
            chi2_red=[1.2, 1.1, 1.0],
        )
    return Orbits.from_kwargs(**kwargs)


def test_orbits_nested_ipc_round_trip_full_with_covariance():
    _assert_lossless(_orbits(with_covariance=True))


def test_orbits_nested_ipc_round_trip_without_covariance():
    _assert_lossless(_orbits(with_covariance=False))


def test_orbits_nested_ipc_round_trip_with_physical_parameters():
    _assert_lossless(_orbits(with_covariance=True, physical=True))


def test_round_trip_orbits_reconstructs_orbits():
    orbits = _orbits(with_covariance=True, physical=True)
    out = round_trip_orbits(orbits)
    assert out.coordinates.frame == orbits.coordinates.frame
    assert out.coordinates.time.scale == orbits.coordinates.time.scale
    assert (
        out.table.combine_chunks().to_pylist()
        == orbits.table.combine_chunks().to_pylist()
    )


def test_round_trip_orbits_zero_copy_reconstructs_orbits():
    # Arrow C Data Interface transport (no IPC copy); verifies schema metadata
    # survives the zero-copy hand-off in both directions.
    orbits = _orbits(with_covariance=True, physical=True)
    out = round_trip_orbits_zero_copy(orbits)
    assert out.coordinates.frame == orbits.coordinates.frame
    assert out.coordinates.time.scale == orbits.coordinates.time.scale
    assert (
        out.table.combine_chunks().to_pylist()
        == orbits.table.combine_chunks().to_pylist()
    )


def test_rotate_orbits_frame_matches_transform_coordinates():
    orbits = _orbits(with_covariance=True)
    rotated = rotate_orbits_frame(orbits, "equatorial")
    reference = transform_coordinates(
        orbits.coordinates, CartesianCoordinates, frame_out="equatorial"
    )
    assert rotated.coordinates.frame == "equatorial"
    np.testing.assert_allclose(
        rotated.coordinates.values, reference.values, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        rotated.coordinates.covariance.to_matrix(),
        reference.covariance.to_matrix(),
        rtol=0,
        atol=1e-12,
    )


def _orbits_with_psd_covariance() -> Orbits:
    # Sigma-point sampling requires a positive-semidefinite covariance.
    base = np.diag([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
    cov = np.stack([base * (i + 1) for i in range(3)])
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0, 4.0, 7.0],
        y=[2.0, 5.0, 8.0],
        z=[3.0, 6.0, 9.0],
        vx=[0.001, 0.004, 0.007],
        vy=[0.002, 0.005, 0.008],
        vz=[0.003, 0.006, 0.009],
        time=Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", "b", "c"],
        coordinates=coordinates,
    )


def test_sample_orbit_variants_sigma_point_matches_create():
    orbits = _orbits_with_psd_covariance()
    bridge = sample_orbit_variants(orbits, method="sigma-point")
    reference = VariantOrbits.create(orbits, method="sigma-point")
    assert bridge.orbit_id.to_pylist() == reference.orbit_id.to_pylist()
    assert bridge.variant_id.to_pylist() == reference.variant_id.to_pylist()
    np.testing.assert_allclose(
        bridge.weights.to_numpy(zero_copy_only=False),
        reference.weights.to_numpy(zero_copy_only=False),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        bridge.weights_cov.to_numpy(zero_copy_only=False),
        reference.weights_cov.to_numpy(zero_copy_only=False),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        bridge.coordinates.values, reference.coordinates.values, rtol=0, atol=1e-12
    )


def test_propagate_orbits_2body_matches_propagate_2body():
    # Physically valid near-circular heliocentric orbits (v ~ sqrt(mu_sun / r)).
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.5, 2.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        vx=[0.0, 0.0, 0.0],
        vy=[0.01720, 0.01405, 0.01216],
        vz=[0.0, 0.0, 0.0],
        time=Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", "b", "c"],
        coordinates=coordinates,
    )
    target = Timestamp.from_mjd([60010.0], scale="tdb")
    bridge = propagate_orbits_2body(orbits, target)
    reference = propagate_2body(orbits, target)
    assert bridge.coordinates.time.scale == "tdb"
    assert bridge.orbit_id.to_pylist() == reference.orbit_id.to_pylist()
    np.testing.assert_allclose(
        bridge.coordinates.values, reference.coordinates.values, rtol=0, atol=1e-11
    )


def test_propagate_orbits_2body_transports_covariance():
    # Valid near-circular orbits with a small covariance; propagation must
    # transport covariance via the STM, matching propagate_2body.
    cov = np.stack([np.diag([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12])] * 3)
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.5, 2.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        vx=[0.0, 0.0, 0.0],
        vy=[0.01720, 0.01405, 0.01216],
        vz=[0.0, 0.0, 0.0],
        time=Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", "b", "c"],
        coordinates=coordinates,
    )
    target = Timestamp.from_mjd([60010.0], scale="tdb")
    bridge = propagate_orbits_2body(orbits, target)
    reference = propagate_2body(orbits, target)
    np.testing.assert_allclose(
        bridge.coordinates.values, reference.coordinates.values, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        bridge.coordinates.covariance.to_matrix(),
        reference.coordinates.covariance.to_matrix(),
        rtol=1e-9,
        atol=1e-18,
    )
