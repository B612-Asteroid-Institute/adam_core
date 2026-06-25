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
import pyarrow.compute as pc

from adam_core import _rust_native as rn
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
    transform_coordinates,
)
from adam_core.coordinates.variants import create_coordinate_variants
from adam_core.orbits import Orbits
from adam_core.coordinates.residuals import Residuals
from adam_core.dynamics import propagate_2body
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.observers import Observers
from adam_core.orbits.arrow_bridge import (
    _rotate_orbits_frame_ipc_candidate,
    _sample_orbit_variants_ipc_candidate,
    evaluate_residuals_2body,
    fit_orbit_least_squares,
    orbits_to_ipc,
    propagate_orbits_2body,
    round_trip_observers,
    round_trip_orbits,
    round_trip_orbits_zero_copy,
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


def test_rotate_orbits_frame_candidate_matches_transform_coordinates():
    orbits = _orbits(with_covariance=True)
    rotated = _rotate_orbits_frame_ipc_candidate(orbits, "equatorial")
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


def _legacy_sigma_point_variants(orbits: Orbits) -> VariantOrbits:
    variant_coordinates = create_coordinate_variants(
        orbits.coordinates, method="sigma-point"
    )
    return VariantOrbits.from_kwargs(
        orbit_id=pc.take(orbits.orbit_id, variant_coordinates.index),
        object_id=pc.take(orbits.object_id, variant_coordinates.index),
        variant_id=np.array(
            np.arange(len(variant_coordinates)).astype(str), dtype="object"
        ),
        weights=variant_coordinates.weight,
        weights_cov=variant_coordinates.weight_cov,
        coordinates=variant_coordinates.sample,
        physical_parameters=orbits.physical_parameters.take(variant_coordinates.index),
    )


def _assert_sigma_point_variants_match(
    actual: VariantOrbits, expected: VariantOrbits
) -> None:
    assert actual.orbit_id.to_pylist() == expected.orbit_id.to_pylist()
    assert actual.object_id.to_pylist() == expected.object_id.to_pylist()
    assert actual.variant_id.to_pylist() == expected.variant_id.to_pylist()
    np.testing.assert_allclose(
        actual.weights.to_numpy(zero_copy_only=False),
        expected.weights.to_numpy(zero_copy_only=False),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        actual.weights_cov.to_numpy(zero_copy_only=False),
        expected.weights_cov.to_numpy(zero_copy_only=False),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        actual.coordinates.values, expected.coordinates.values, rtol=0, atol=1e-12
    )


def test_sample_orbit_variants_sigma_point_candidate_matches_legacy_sampler():
    orbits = _orbits_with_psd_covariance()
    bridge = _sample_orbit_variants_ipc_candidate(orbits, method="sigma-point")
    reference = _legacy_sigma_point_variants(orbits)
    _assert_sigma_point_variants_match(bridge, reference)


def test_variant_orbits_create_sigma_point_uses_rust_candidate_and_preserves_physical():
    base = _orbits_with_psd_covariance()
    orbits = Orbits.from_kwargs(
        orbit_id=base.orbit_id,
        object_id=base.object_id,
        coordinates=base.coordinates,
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.5, 16.0, 17.0],
            H_v_sigma=[0.1, None, 0.3],
            G=[0.15, 0.15, 0.15],
            G_sigma=[None, None, None],
            sigma_eff=[0.05, 0.06, 0.07],
            chi2_red=[1.2, 1.1, 1.0],
        ),
    )
    variants = VariantOrbits.create(orbits, method="sigma-point")
    reference = _legacy_sigma_point_variants(orbits)
    _assert_sigma_point_variants_match(variants, reference)
    assert (
        variants.physical_parameters.H_v.to_pylist()
        == reference.physical_parameters.H_v.to_pylist()
    )
    assert (
        variants.physical_parameters.G.to_pylist()
        == reference.physical_parameters.G.to_pylist()
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


def test_evaluate_residuals_2body_matches_generate_ephemeris_2body():
    # Orbits at the observation epoch (1:1 with observers), all TDB.
    orbit_coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.5, 2.0],
        y=[0.0, 0.1, 0.2],
        z=[0.0, 0.0, 0.05],
        vx=[0.0, 0.0, 0.0],
        vy=[0.01720, 0.01405, 0.01216],
        vz=[0.0, 0.001, 0.0],
        time=Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * 3),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(orbit_id=["o1", "o2", "o3"], coordinates=orbit_coords)
    observer_coords = CartesianCoordinates.from_kwargs(
        x=[0.0, 0.01, -0.01],
        y=[1.0, 0.99, 1.01],
        z=[0.0, 0.0, 0.0],
        vx=[-0.01720, -0.0170, -0.0173],
        vy=[0.0, 0.001, 0.0],
        vz=[0.0, 0.0, 0.0],
        time=Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * 3),
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(
        code=["X05", "X05", "X05"], coordinates=observer_coords
    )

    # Reference predicted astrometry via adam_core's 2-body ephemeris.
    ephemeris = generate_ephemeris_2body(orbits, observers)
    predicted = ephemeris.coordinates
    n = 3
    cov = np.tile(
        np.diag([1.0, (1.0 / 3600.0) ** 2, (1.0 / 3600.0) ** 2, 1.0, 1.0, 1.0]),
        (n, 1, 1),
    )
    observed = SphericalCoordinates.from_kwargs(
        rho=predicted.rho.to_numpy(zero_copy_only=False),
        lon=predicted.lon.to_numpy(zero_copy_only=False) + 1e-4,
        lat=predicted.lat.to_numpy(zero_copy_only=False) - 1e-4,
        vrho=predicted.vrho.to_numpy(zero_copy_only=False),
        vlon=predicted.vlon.to_numpy(zero_copy_only=False),
        vlat=predicted.vlat.to_numpy(zero_copy_only=False),
        time=predicted.time,
        origin=predicted.origin,
        frame=predicted.frame,
        covariance=CoordinateCovariances.from_matrix(cov),
    )
    reference = Residuals.calculate(observed, predicted)
    chi2_reference = reference.chi2.to_numpy(zero_copy_only=False)

    chi2_rust, _residuals_rust = evaluate_residuals_2body(orbits, observed, observers)
    np.testing.assert_allclose(chi2_rust, chi2_reference, rtol=1e-9, atol=1e-12)


def test_fit_orbit_least_squares_recovers_truth():
    # Generate noise-free astrometry from a truth orbit, then confirm the
    # Rust-native Gauss-Newton fit recovers it from a perturbed start. (adam_core
    # has no 2-body Propagator, so ground truth is the reference.)
    n = 8
    mjds = [60000.0 + i * 5.0 for i in range(n)]
    obs_times = Timestamp.from_mjd(mjds, scale="tdb")
    mu = 0.000_295_912_208_285_591_1
    v = mu**0.5
    thetas = np.array([v * (m - 60000.0) for m in mjds])
    observer_coords = CartesianCoordinates.from_kwargs(
        x=np.cos(thetas),
        y=np.sin(thetas),
        z=np.zeros(n),
        vx=-v * np.sin(thetas),
        vy=v * np.cos(thetas),
        vz=np.zeros(n),
        time=obs_times,
        origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * n),
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(code=["X05"] * n, coordinates=observer_coords)
    truth_state = np.array([1.2, 0.1, 0.05, -0.002, 0.016, 0.001])
    truth = Orbits.from_kwargs(
        orbit_id=["truth"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.2],
            y=[0.1],
            z=[0.05],
            vx=[-0.002],
            vy=[0.016],
            vz=[0.001],
            time=Timestamp.from_mjd([60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"]),
            frame="ecliptic",
        ),
    )
    propagated = propagate_2body(truth, obs_times)
    predicted = generate_ephemeris_2body(propagated, observers).coordinates
    arcsec = (1.0 / 3600.0) ** 2
    cov = np.tile(np.diag([1.0, arcsec, arcsec, 1.0, 1.0, 1.0]), (n, 1, 1))
    observed = SphericalCoordinates.from_kwargs(
        rho=predicted.rho.to_numpy(zero_copy_only=False),
        lon=predicted.lon.to_numpy(zero_copy_only=False),
        lat=predicted.lat.to_numpy(zero_copy_only=False),
        vrho=predicted.vrho.to_numpy(zero_copy_only=False),
        vlon=predicted.vlon.to_numpy(zero_copy_only=False),
        vlat=predicted.vlat.to_numpy(zero_copy_only=False),
        time=predicted.time,
        origin=predicted.origin,
        frame=predicted.frame,
        covariance=CoordinateCovariances.from_matrix(cov),
    )
    initial = Orbits.from_kwargs(
        orbit_id=["fit"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.2 + 1e-3],
            y=[0.1 - 1e-3],
            z=[0.05 + 5e-4],
            vx=[-0.002 + 1e-5],
            vy=[0.016 - 1e-5],
            vz=[0.001 + 1e-5],
            time=Timestamp.from_mjd([60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"]),
            frame="ecliptic",
        ),
    )
    fitted, chi2, iterations, converged = fit_orbit_least_squares(
        initial, observed, observers
    )
    assert converged
    assert chi2 < 0.1
    np.testing.assert_allclose(
        fitted.coordinates.values[0], truth_state, rtol=0, atol=1e-4
    )


def test_round_trip_observers_reconstructs_observers():
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0, 2.0],
        y=[3.0, 4.0],
        z=[5.0, 6.0],
        vx=[0.1, 0.2],
        vy=[0.3, 0.4],
        vz=[0.5, 0.6],
        time=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(code=["X05", "500"], coordinates=coordinates)
    out = round_trip_observers(observers)
    assert out.code.to_pylist() == observers.code.to_pylist()
    assert out.coordinates.frame == "ecliptic"
    assert out.coordinates.time.scale == "tdb"
    assert (
        out.table.combine_chunks().to_pylist()
        == observers.table.combine_chunks().to_pylist()
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
