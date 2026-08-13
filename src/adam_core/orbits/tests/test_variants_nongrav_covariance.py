import numpy as np
import quivr as qv

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.orbits.non_gravitational_parameters import NonGravitationalParameters
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp


def _extended_orbit() -> tuple[Orbits, np.ndarray, np.ndarray]:
    mean = np.array([1.0, 0.2, 0.1, -0.002, 0.015, 0.001, 1e-10, 2e-11, -3e-12])
    scales = np.array([1e-5, 2e-5, 3e-5, 1e-7, 2e-7, 3e-7, 1e-12, 2e-13, 3e-14])
    root = np.diag(scales)
    root[0, 6] = root[6, 0] = 2e-18
    covariance = root @ root.T
    coordinates = CartesianCoordinates.from_kwargs(
        x=[mean[0]],
        y=[mean[1]],
        z=[mean[2]],
        vx=[mean[3]],
        vy=[mean[4]],
        vz=[mean[5]],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
        covariance=CoordinateCovariances.from_matrix(covariance[None]),
    )
    nongrav = NonGravitationalParameters.from_kwargs(
        source=["test"],
        A1=[mean[6]],
        A2=[mean[7]],
        A3=[mean[8]],
        ALN=[1.0],
        NK=[0.0],
        NM=[2.0],
        NN=[5.093],
        R0=[1.0],
    )
    return (
        Orbits.from_kwargs(
            orbit_id=["o"],
            object_id=["o"],
            coordinates=coordinates,
            non_gravitational_parameters=nongrav,
        ),
        mean,
        covariance,
    )


def test_sigma_point_variants_sample_and_collapse_full_nongrav_state() -> None:
    orbits, mean, covariance = _extended_orbit()
    variants = VariantOrbits.create(orbits, method="sigma-point")

    assert len(variants) == 19
    assert variants.non_gravitational_parameters.source.to_pylist() == ["test"] * 19
    assert variants.non_gravitational_parameters.ALN.to_pylist() == [1.0] * 19
    values = np.column_stack(
        [
            variants.coordinates.values,
            variants.non_gravitational_parameters.to_array(),
        ]
    )
    weights = variants.weights.to_numpy(zero_copy_only=False)
    weights_cov = variants.weights_cov.to_numpy(zero_copy_only=False)
    reconstructed_mean = np.sum(values * weights[:, None], axis=0)
    delta = values - mean
    reconstructed_covariance = np.einsum("n,ni,nj->ij", weights_cov, delta, delta)
    np.testing.assert_allclose(reconstructed_mean, mean, rtol=0, atol=3e-16)
    np.testing.assert_allclose(
        reconstructed_covariance, covariance, rtol=2e-8, atol=2e-26
    )

    collapsed = variants.collapse(orbits)
    np.testing.assert_allclose(
        collapsed.coordinates.covariance.to_full_matrix()[0],
        covariance,
        rtol=2e-8,
        atol=2e-26,
    )
    np.testing.assert_array_equal(
        collapsed.non_gravitational_parameters.to_array(), mean[None, 6:]
    )

    collapsed_by_object = variants.collapse_by_object_id()
    assert collapsed_by_object.coordinates.covariance.has_nongrav_block()
    np.testing.assert_allclose(
        collapsed_by_object.non_gravitational_parameters.to_array()[0],
        mean[6:],
        rtol=0,
        atol=3e-26,
    )


def test_variant_nongrav_helpers_and_include_nongrav_false() -> None:
    orbits, _, _ = _extended_orbit()
    variants = VariantOrbits.create(orbits, method="sigma-point")

    assert variants.has_non_gravitational_parameters()
    stripped = variants.without_non_gravitational_parameters()
    assert not stripped.has_non_gravitational_parameters()
    assert not stripped.coordinates.covariance.has_nongrav_block()

    coordinate_only = VariantOrbits.create(
        orbits, method="sigma-point", include_nongrav=False
    )
    assert len(coordinate_only) == 13
    assert not coordinate_only.has_non_gravitational_parameters()
    assert not coordinate_only.coordinates.covariance.has_nongrav_block()


def test_mixed_covariance_rows_sample_per_source_dimension() -> None:
    extended, _, _ = _extended_orbit()
    coordinate_covariance = extended.coordinates.covariance.to_matrix()
    plain_covariance = CoordinateCovariances.from_matrix(coordinate_covariance)
    mixed_covariance = qv.concatenate(
        [plain_covariance, extended.coordinates.covariance]
    )
    plain = Orbits.from_kwargs(
        orbit_id=["plain"],
        object_id=["plain"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.5],
            y=[0.3],
            z=[0.2],
            vx=[-0.003],
            vy=[0.014],
            vz=[0.002],
            time=Timestamp.from_mjd([60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_matrix(coordinate_covariance),
        ),
    )
    mixed = qv.concatenate([plain, extended]).set_column(
        "coordinates.covariance", mixed_covariance
    )
    variants = VariantOrbits.create(mixed, method="sigma-point")

    assert len(variants.select("orbit_id", "o")) == 19
    assert len(variants.select("orbit_id", "plain")) == 13
    assert variants.orbit_id.to_pylist()[:19] == ["o"] * 19
    assert variants.variant_id.to_pylist() == [str(index) for index in range(32)]
    plain_variants = variants.select("orbit_id", "plain")
    assert not plain_variants.has_non_gravitational_parameters()


def test_coordinate_only_covariance_preserves_fixed_nongrav_parameters() -> None:
    extended, mean, covariance = _extended_orbit()
    coordinate_only_covariance = CoordinateCovariances.from_matrix(
        covariance[None, :6, :6]
    )
    semantic_six_dimensional = extended.set_column(
        "coordinates.covariance", coordinate_only_covariance
    )

    variants = VariantOrbits.create(
        semantic_six_dimensional, method="sigma-point", include_nongrav=True
    )

    assert len(variants) == 13
    np.testing.assert_array_equal(
        variants.non_gravitational_parameters.to_array(),
        np.repeat(mean[None, 6:], 13, axis=0),
    )


def test_monte_carlo_variants_sample_nongrav_dimensions_deterministically() -> None:
    orbits, _, _ = _extended_orbit()
    left = VariantOrbits.create(orbits, method="monte-carlo", num_samples=32, seed=42)
    right = VariantOrbits.create(orbits, method="monte-carlo", num_samples=32, seed=42)

    np.testing.assert_array_equal(left.coordinates.values, right.coordinates.values)
    np.testing.assert_array_equal(
        left.non_gravitational_parameters.to_array(),
        right.non_gravitational_parameters.to_array(),
    )
    assert np.ptp(left.non_gravitational_parameters.A2.to_numpy()) > 0.0
