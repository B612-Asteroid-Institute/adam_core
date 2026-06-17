import numpy as np
import pyarrow.compute as pc
import pytest
import quivr as qv

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from ...orbits.non_gravitational_parameters import NonGravitationalParameters
from ...orbits.physical_parameters import PhysicalParameters
from ...orbits.solved_state_covariances import SolvedStateCovariances
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..orbits import Orbits
from ..variants import VariantEphemeris, VariantOrbits


def _nongrav_a2_row(a2: float, a2_sigma: float) -> NonGravitationalParameters:
    return NonGravitationalParameters.from_kwargs(
        source=["SBDB"],
        model=["nongrav"],
        solution_dimension=[7],
        parameter_count=[1],
        estimated_parameter_names=["A2"],
        A1=[None],
        A1_sigma=[None],
        A2=[a2],
        A2_sigma=[a2_sigma],
        A3=[None],
        A3_sigma=[None],
        DT=[None],
        DT_sigma=[None],
        R0=[None],
        R0_sigma=[None],
        ALN=[None],
        ALN_sigma=[None],
        NK=[None],
        NK_sigma=[None],
        NM=[None],
        NM_sigma=[None],
        NN=[None],
        NN_sigma=[None],
        AMRAT=[None],
        AMRAT_sigma=[None],
        RHO=[None],
        RHO_sigma=[None],
    )


def _solved_a2_orbit(orbit_id: str, covariance_7x7: np.ndarray) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[orbit_id],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[20.0], G=[0.15]),
        non_gravitational_parameters=_nongrav_a2_row(-2.0e-13, 2.0e-13),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            covariance_7x7.reshape(1, 7, 7),
            [["x", "y", "z", "vx", "vy", "vz", "A2"]],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            covariance=CoordinateCovariances.from_matrix(
                covariance_7x7[:6, :6].reshape(1, 6, 6)
            ),
            time=Timestamp.from_mjd([60000.0]),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def test_VariantOrbits():

    # Get a sample of real orbits
    orbits = make_real_orbits(10)

    # Create a variant orbits object (expands the covariance matrix)
    # around the mean state
    variant_orbits = VariantOrbits.create(orbits)

    # For these 10 orbits this will select sigma-points so lets
    # check that the number of sigma-points is correct
    assert len(variant_orbits) == len(orbits) * 13

    # Now lets collapse the sigma-points back and see if we can reconstruct
    # the input covairance matrix
    collapsed_orbits = variant_orbits.collapse(orbits)

    # Check that the covariance matrices are close
    np.testing.assert_allclose(
        collapsed_orbits.coordinates.covariance.to_matrix(),
        orbits.coordinates.covariance.to_matrix(),
        rtol=0,
        atol=1e-14,
    )

    # Check that the orbit ids are the same
    np.testing.assert_equal(
        collapsed_orbits.orbit_id.to_numpy(zero_copy_only=False),
        orbits.orbit_id.to_numpy(zero_copy_only=False),
    )


def test_VariantOrbits_joint_sampling_uses_solved_state_covariance():
    covariance = np.diag(
        [
            1e-8,
            2e-8,
            3e-8,
            4e-10,
            5e-10,
            6e-10,
            9e-26,
            4e-26,
            1e-26,
        ]
    ).reshape(1, 9, 9)
    # Cross-covariances between orbital and non-grav dimensions: recovering
    # these is the entire point of joint sampling.
    covariance[0, 0, 6] = covariance[0, 6, 0] = 2e-17
    covariance[0, 4, 7] = covariance[0, 7, 4] = -2e-18
    covariance[0, 5, 8] = covariance[0, 8, 5] = 5e-19
    orbits = Orbits.from_kwargs(
        orbit_id=["joint"],
        object_id=["joint"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[20.0], G=[0.15]),
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"],
            model=["nongrav"],
            solution_dimension=[9],
            parameter_count=[3],
            estimated_parameter_names=["A1,A2,A3"],
            A1=[1.0e-13],
            A1_sigma=[3.0e-13],
            A2=[-2.0e-13],
            A2_sigma=[2.0e-13],
            A3=[4.0e-13],
            A3_sigma=[1.0e-13],
            DT=[None],
            DT_sigma=[None],
            R0=[None],
            R0_sigma=[None],
            ALN=[None],
            ALN_sigma=[None],
            NK=[None],
            NK_sigma=[None],
            NM=[None],
            NM_sigma=[None],
            NN=[None],
            NN_sigma=[None],
            AMRAT=[None],
            AMRAT_sigma=[None],
            RHO=[None],
            RHO_sigma=[None],
        ),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            covariance,
            [["x", "y", "z", "vx", "vy", "vz", "A1", "A2", "A3"]],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            covariance=CoordinateCovariances.from_matrix(covariance[:, :6, :6]),
            time=Timestamp.from_mjd([60000.0]),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    variants = VariantOrbits.create(orbits, method="sigma-point")

    assert len(variants) == 19
    assert variants.solved_state_covariance.dimension[0].as_py() == 9
    assert len(np.unique(variants.non_gravitational_parameters.A1.to_pylist())) > 1
    assert len(np.unique(variants.non_gravitational_parameters.A2.to_pylist())) > 1
    assert len(np.unique(variants.non_gravitational_parameters.A3.to_pylist())) > 1

    collapsed = variants.collapse(orbits)
    assert collapsed.solved_state_covariance.dimension[0].as_py() == 9
    # Compare element-wise relative to each entry's own scale: an absolute
    # tolerance would be vacuously satisfied by the ~1e-26 non-grav entries.
    recovered = collapsed.solved_state_covariance.to_matrix()[0]
    np.testing.assert_allclose(np.diag(recovered), np.diag(covariance[0]), rtol=1e-9)
    np.testing.assert_allclose(recovered[0, 6], covariance[0, 0, 6], rtol=1e-9)
    np.testing.assert_allclose(recovered[4, 7], covariance[0, 4, 7], rtol=1e-9)
    np.testing.assert_allclose(recovered[5, 8], covariance[0, 5, 8], rtol=1e-9)


def test_VariantOrbits_create_include_nongrav_false_uses_orbital_covariance_only():
    covariance = np.diag([1e-8, 2e-8, 3e-8, 4e-10, 5e-10, 6e-10, 9e-26]).reshape(
        1, 7, 7
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["joint"],
        object_id=["joint"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[20.0], G=[0.15]),
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"],
            model=["nongrav"],
            solution_dimension=[7],
            parameter_count=[1],
            estimated_parameter_names=["A2"],
            A1=[None],
            A1_sigma=[None],
            A2=[-2.0e-13],
            A2_sigma=[2.0e-13],
            A3=[None],
            A3_sigma=[None],
            DT=[None],
            DT_sigma=[None],
            R0=[None],
            R0_sigma=[None],
            ALN=[None],
            ALN_sigma=[None],
            NK=[None],
            NK_sigma=[None],
            NM=[None],
            NM_sigma=[None],
            NN=[None],
            NN_sigma=[None],
            AMRAT=[None],
            AMRAT_sigma=[None],
            RHO=[None],
            RHO_sigma=[None],
        ),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            covariance,
            [["x", "y", "z", "vx", "vy", "vz", "A2"]],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            covariance=CoordinateCovariances.from_matrix(covariance[:, :6, :6]),
            time=Timestamp.from_mjd([60000.0]),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    variants = VariantOrbits.create(orbits, method="sigma-point", include_nongrav=False)

    assert len(variants) == 13
    assert variants.non_gravitational_parameters.A2[0].as_py() is None
    assert variants.solved_state_covariance.dimension[0].as_py() is None


def test_VariantOrbits_create_mixed_solved_state_coverage():
    # One orbit with a 7x7 solved-state covariance and one with only a 6x6
    # coordinate covariance must both be sampled in a single create() call.
    covariance_7x7 = np.diag([1e-8, 2e-8, 3e-8, 4e-10, 5e-10, 6e-10, 9e-26])
    orbit_with = _solved_a2_orbit("with_nongrav", covariance_7x7)
    orbit_without = Orbits.from_kwargs(
        orbit_id=["plain"],
        object_id=["plain"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.5],
            y=[2.5],
            z=[3.5],
            vx=[0.011],
            vy=[0.021],
            vz=[0.031],
            covariance=CoordinateCovariances.from_matrix(
                covariance_7x7[:6, :6].reshape(1, 6, 6)
            ),
            time=Timestamp.from_mjd([60000.0]),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    orbits = qv.concatenate([orbit_with, orbit_without])

    variants = VariantOrbits.create(orbits, method="sigma-point")

    joint = variants.select("orbit_id", "with_nongrav")
    plain = variants.select("orbit_id", "plain")
    assert len(joint) == 15  # 2 * 7 + 1 sigma points
    assert len(plain) == 13  # 2 * 6 + 1 sigma points
    assert len(np.unique(joint.non_gravitational_parameters.A2.to_pylist())) > 1
    assert all(
        value is None for value in plain.non_gravitational_parameters.A2.to_pylist()
    )
    assert plain.solved_state_covariance.dimension[0].as_py() is None
    assert len(set(variants.variant_id.to_pylist())) == len(variants)


def test_VariantOrbits_monte_carlo_joint_sampling_recovers_nongrav_spread():
    # The non-grav variance (9e-26) is ~18 orders of magnitude below the
    # positional variances. Unwhitened sampling silently truncates it to a
    # zero eigenvalue and produces no spread in A2 at all.
    covariance_7x7 = np.diag([1e-8, 2e-8, 3e-8, 4e-10, 5e-10, 6e-10, 9e-26])
    covariance_7x7[0, 6] = covariance_7x7[6, 0] = 2e-17
    orbits = _solved_a2_orbit("mc", covariance_7x7)

    variants = VariantOrbits.create(
        orbits, method="monte-carlo", num_samples=20000, seed=42
    )

    a2 = np.array(variants.non_gravitational_parameters.A2.to_pylist())
    np.testing.assert_allclose(np.std(a2), np.sqrt(9e-26), rtol=0.05)
    x = variants.coordinates.x.to_numpy(zero_copy_only=False)
    expected_correlation = 2e-17 / np.sqrt(1e-8 * 9e-26)
    np.testing.assert_allclose(
        np.corrcoef(x, a2)[0, 1], expected_correlation, atol=0.05
    )


def test_VariantOrbits_monte_carlo_seed_is_independent_per_orbit():
    covariance_7x7 = np.diag([1e-8, 2e-8, 3e-8, 4e-10, 5e-10, 6e-10, 9e-26])
    orbits = qv.concatenate(
        [
            _solved_a2_orbit("orbit_a", covariance_7x7),
            _solved_a2_orbit("orbit_b", covariance_7x7),
        ]
    )

    variants = VariantOrbits.create(
        orbits, method="monte-carlo", num_samples=100, seed=7
    )

    # Identical means and covariances, so identical draws would mean the seed
    # was reused verbatim for both orbits.
    x_a = variants.select("orbit_id", "orbit_a").coordinates.x.to_numpy(
        zero_copy_only=False
    )
    x_b = variants.select("orbit_id", "orbit_b").coordinates.x.to_numpy(
        zero_copy_only=False
    )
    assert not np.allclose(x_a, x_b)

    # The same seed must reproduce the same variants across calls.
    variants_again = VariantOrbits.create(
        orbits, method="monte-carlo", num_samples=100, seed=7
    )
    np.testing.assert_array_equal(
        variants.coordinates.x.to_numpy(zero_copy_only=False),
        variants_again.coordinates.x.to_numpy(zero_copy_only=False),
    )


def test_VariantOrbits_collapse_by_object_id():
    """Test that VariantOrbits.collapse_by_object_id correctly collapses variants into mean orbits."""

    # Create variant orbits with multiple objects, each having multiple variants
    variant_orbits = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        object_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        variant_id=["0", "1", "2", "0", "1", "2"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0, 15.0, 17.5, 17.5, 17.5],
            G=[0.15, 0.15, 0.15, 0.25, 0.25, 0.25],
        ),
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB", "SBDB", "NEOCC", "NEOCC", "NEOCC"],
            model=[
                "nongrav",
                "nongrav",
                "nongrav",
                "yarkovsky",
                "yarkovsky",
                "yarkovsky",
            ],
            solution_dimension=[7, 7, 7, 7, 7, 7],
            parameter_count=[1, 1, 1, 1, 1, 1],
            estimated_parameter_names=["A2", "A2", "A2", "A2", "A2", "A2"],
            A1=[None] * 6,
            A1_sigma=[None] * 6,
            A2=[1e-13, 1e-13, 1e-13, -2e-14, -2e-14, -2e-14],
            A2_sigma=[None] * 6,
            A3=[None] * 6,
            A3_sigma=[None] * 6,
            DT=[None] * 6,
            DT_sigma=[None] * 6,
            R0=[None] * 6,
            R0_sigma=[None] * 6,
            ALN=[None] * 6,
            ALN_sigma=[None] * 6,
            NK=[None] * 6,
            NK_sigma=[None] * 6,
            NM=[None] * 6,
            NM_sigma=[None] * 6,
            NN=[None] * 6,
            NN_sigma=[None] * 6,
            AMRAT=[None] * 6,
            AMRAT_sigma=[None] * 6,
            RHO=[None] * 6,
            RHO_sigma=[None] * 6,
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            y=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            z=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            vx=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vy=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vz=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            time=Timestamp.from_mjd([60000] * 6),
            origin=Origin.from_kwargs(code=["SUN"] * 6),
            frame="ecliptic",
        ),
    )

    # Collapse the variants
    collapsed = variant_orbits.collapse_by_object_id()

    # Check basic properties
    assert len(collapsed) == 2  # Should have one orbit per object
    assert set(collapsed.object_id.to_pylist()) == {"obj1", "obj2"}

    # Check that means are computed correctly for each object
    obj1 = collapsed.select("object_id", "obj1")
    obj2 = collapsed.select("object_id", "obj2")

    # Check means for obj1
    np.testing.assert_allclose(
        obj1.coordinates.values[0],
        np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),  # Expected mean for obj1
        rtol=1e-14,
    )

    # Check means for obj2
    np.testing.assert_allclose(
        obj2.coordinates.values[0],
        np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2]),  # Expected mean for obj2
        rtol=1e-14,
    )

    # Physical parameters should be preserved per object_id
    assert obj1.physical_parameters.H_v[0].as_py() == 15.0
    assert obj1.physical_parameters.G[0].as_py() == 0.15
    assert obj2.physical_parameters.H_v[0].as_py() == 17.5
    assert obj2.physical_parameters.G[0].as_py() == 0.25
    assert obj1.non_gravitational_parameters.A2[0].as_py() == 1e-13
    assert obj2.non_gravitational_parameters.A2[0].as_py() == -2e-14

    # Check that covariance matrices are computed correctly
    # For obj1, the variance should be approximately 0.00667 for each component
    obj1_cov = obj1.coordinates.covariance.to_matrix()[0]
    # The variance is sum((x - mean)^2) / n where n=3
    # For positions: (1.1 - 1.0)^2 + (0.9 - 1.0)^2 + (1.0 - 1.0)^2 = 0.02
    # So variance = 0.02/3 ≈ 0.00667
    expected_variance_obj1_pos = 0.02 / 3  # Population variance
    expected_variance_obj1_vel = 0.0002 / 3  # Population variance
    np.testing.assert_allclose(
        np.diag(obj1_cov),
        [expected_variance_obj1_pos] * 3 + [expected_variance_obj1_vel] * 3,
        rtol=1e-6,
    )

    # For obj2, the variance should also be approximately 0.00667
    obj2_cov = obj2.coordinates.covariance.to_matrix()[0]
    expected_variance_obj2_pos = (
        0.02 / 3
    )  # (2.1 - 2.0)^2 + (1.9 - 2.0)^2 + (2.0 - 2.0)^2 = 0.02/3
    expected_variance_obj2_vel = (
        0.0002 / 3
    )  # (0.21 - 0.2)^2 + (0.19 - 0.2)^2 + (0.2 - 0.2)^2 = 0.0002/3
    np.testing.assert_allclose(
        np.diag(obj2_cov),
        [expected_variance_obj2_pos] * 3 + [expected_variance_obj2_vel] * 3,
        rtol=1e-6,
    )

    # Test that time and origin are preserved
    assert all(t == 60000 for t in collapsed.coordinates.time.mjd().to_pylist())
    assert all(o == "SUN" for o in collapsed.coordinates.origin.code.to_pylist())
    assert collapsed.coordinates.frame == "ecliptic"

    # Test error cases
    # Test that variants with different times raise an error
    variant_orbits_diff_times = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0], G=[0.15, 0.15]
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1],
            y=[1.0, 1.1],
            z=[1.0, 1.1],
            vx=[0.1, 0.11],
            vy=[0.1, 0.11],
            vz=[0.1, 0.11],
            time=Timestamp.from_mjd([60000, 60001]),  # Different times
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    with pytest.raises(AssertionError):
        variant_orbits_diff_times.collapse_by_object_id()

    # Test that variants with different origins raise an error
    variant_orbits_diff_origins = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0], G=[0.15, 0.15]
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1],
            y=[1.0, 1.1],
            z=[1.0, 1.1],
            vx=[0.1, 0.11],
            vy=[0.1, 0.11],
            vz=[0.1, 0.11],
            time=Timestamp.from_mjd([60000, 60000]),
            origin=Origin.from_kwargs(code=["SUN", "EARTH"]),  # Different origins
            frame="ecliptic",
        ),
    )
    with pytest.raises(AssertionError):
        variant_orbits_diff_origins.collapse_by_object_id()


def test_VariantOrbits_collapse_by_object_id_rebuilds_solved_state_covariance():
    covariance = np.diag([1e-8, 2e-8, 3e-8, 4e-10, 5e-10, 6e-10, 9e-26])
    variant_orbits = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1"],
        object_id=["obj1", "obj1", "obj1"],
        variant_id=["0", "1", "2"],
        weights=[1 / 3, 1 / 3, 1 / 3],
        weights_cov=[1 / 3, 1 / 3, 1 / 3],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0, 15.0],
            G=[0.15, 0.15, 0.15],
        ),
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"] * 3,
            model=["nongrav"] * 3,
            solution_dimension=[7] * 3,
            parameter_count=[1] * 3,
            estimated_parameter_names=["A2"] * 3,
            A1=[None] * 3,
            A1_sigma=[None] * 3,
            A2=[1e-13, 1.2e-13, 0.8e-13],
            A2_sigma=[None] * 3,
            A3=[None] * 3,
            A3_sigma=[None] * 3,
            DT=[None] * 3,
            DT_sigma=[None] * 3,
            R0=[None] * 3,
            R0_sigma=[None] * 3,
            ALN=[None] * 3,
            ALN_sigma=[None] * 3,
            NK=[None] * 3,
            NK_sigma=[None] * 3,
            NM=[None] * 3,
            NM_sigma=[None] * 3,
            NN=[None] * 3,
            NN_sigma=[None] * 3,
            AMRAT=[None] * 3,
            AMRAT_sigma=[None] * 3,
            RHO=[None] * 3,
            RHO_sigma=[None] * 3,
        ),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            [covariance, covariance, covariance],
            [["x", "y", "z", "vx", "vy", "vz", "A2"]] * 3,
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1, 0.9],
            y=[1.0, 1.1, 0.9],
            z=[1.0, 1.1, 0.9],
            vx=[0.1, 0.11, 0.09],
            vy=[0.1, 0.11, 0.09],
            vz=[0.1, 0.11, 0.09],
            time=Timestamp.from_mjd([60000] * 3),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
        ),
    )

    collapsed = variant_orbits.collapse_by_object_id()

    assert collapsed.solved_state_covariance.dimension[0].as_py() == 7
    assert collapsed.non_gravitational_parameters.A2[0].as_py() == pytest.approx(1e-13)
    np.testing.assert_allclose(
        collapsed.solved_state_covariance.to_orbital_covariances().to_matrix(),
        collapsed.coordinates.covariance.to_matrix(),
        rtol=0,
        atol=1e-12,
    )


def test_VariantEphemeris_collapse_by_object_id_single_epoch():
    """Test that VariantEphemeris.collapse_by_object_id collapses by object_id for a single epoch."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        object_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        variant_id=["0", "1", "2", "0", "1", "2"],
        predicted_magnitude_v=[20.0, 21.0, 19.0, 18.0, 18.5, 17.5],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            lon=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            lat=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            vrho=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vlon=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vlat=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            time=Timestamp.from_mjd([60000] * 6),
            origin=Origin.from_kwargs(code=["500"] * 6),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()

    assert len(collapsed) == 2
    assert set(collapsed.object_id.to_pylist()) == {"obj1", "obj2"}

    obj1 = collapsed.select("object_id", "obj1")
    obj2 = collapsed.select("object_id", "obj2")

    np.testing.assert_allclose(
        obj1.coordinates.values[0],
        np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        obj2.coordinates.values[0],
        np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2]),
        rtol=1e-14,
    )

    # Mean magnitudes should be propagated.
    assert obj1.predicted_magnitude_v[0].as_py() == pytest.approx(20.0)
    assert obj2.predicted_magnitude_v[0].as_py() == pytest.approx(18.0)

    # Check covariance diagonals (population variance with n=3).
    obj1_cov = obj1.coordinates.covariance.to_matrix()[0]
    expected_variance_pos = 0.02 / 3
    expected_variance_vel = 0.0002 / 3
    np.testing.assert_allclose(
        np.diag(obj1_cov),
        [expected_variance_pos] * 3 + [expected_variance_vel] * 3,
        rtol=1e-6,
    )

    obj2_cov = obj2.coordinates.covariance.to_matrix()[0]
    np.testing.assert_allclose(
        np.diag(obj2_cov),
        [expected_variance_pos] * 3 + [expected_variance_vel] * 3,
        rtol=1e-6,
    )

    # Time, origin, and frame should be preserved.
    assert all(t == 60000 for t in collapsed.coordinates.time.mjd().to_pylist())
    assert all(o == "500" for o in collapsed.coordinates.origin.code.to_pylist())
    assert collapsed.coordinates.frame == "equatorial"


def test_VariantEphemeris_collapse_by_object_id_groups_by_time_and_origin():
    """Collapse should be performed per (object_id, time, origin_code) group."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1"] * 6,
        object_id=["obj1"] * 6,
        variant_id=["0", "1", "0", "1", "0", "1"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.2, 10.0, 10.2, 100.0, 100.2],
            lon=[1.0, 1.2, 10.0, 10.2, 100.0, 100.2],
            lat=[1.0, 1.2, 10.0, 10.2, 100.0, 100.2],
            vrho=[0.1, 0.12, 1.0, 1.02, 10.0, 10.02],
            vlon=[0.1, 0.12, 1.0, 1.02, 10.0, 10.02],
            vlat=[0.1, 0.12, 1.0, 1.02, 10.0, 10.02],
            time=Timestamp.from_mjd([60000, 60000, 60001, 60001, 60000, 60000]),
            origin=Origin.from_kwargs(code=["500", "500", "500", "500", "X05", "X05"]),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()

    # Keys: (60000, "500"), (60001, "500"), (60000, "X05")
    assert len(collapsed) == 3
    assert set(collapsed.coordinates.origin.code.to_pylist()) == {"500", "X05"}
    assert set(collapsed.coordinates.time.mjd().to_pylist()) == {60000, 60001}

    # Validate one group: mjd=60000, origin=500 -> mean of [1.0, 1.2] => 1.1 etc.
    group = (
        collapsed.select("coordinates.origin.code", "500")
        .select("coordinates.time.days", 60000)
        .select("coordinates.time.nanos", 0)
    )
    assert len(group) == 1
    np.testing.assert_allclose(
        group.coordinates.values[0],
        np.array([1.1, 1.1, 1.1, 0.11, 0.11, 0.11]),
        rtol=0,
        atol=1e-12,
    )


def test_VariantEphemeris_collapse_by_object_id_wraps_longitude():
    """Longitude is circular in degrees; mean should respect wrap-around near 0/360."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1"],
        object_id=["obj1", "obj1", "obj1"],
        variant_id=["0", "1", "2"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.0, 1.0],
            lon=[359.0, 1.0, 0.0],
            lat=[0.0, 0.0, 0.0],
            vrho=[0.0, 0.0, 0.0],
            vlon=[0.0, 0.0, 0.0],
            vlat=[0.0, 0.0, 0.0],
            time=Timestamp.from_mjd([60000] * 3),
            origin=Origin.from_kwargs(code=["500"] * 3),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()
    assert len(collapsed) == 1
    lon = collapsed.coordinates.values[0][1]
    assert lon == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= lon < 360.0


def test_VariantEphemeris_collapse_by_object_id_uses_weights_when_present():
    """UT mean should use `weights` and covariance should use `weights_cov` when provided."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        weights=[0.25, 0.75],
        weights_cov=[0.1, 0.9],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[0.0, 10.0],
            lon=[10.0, 20.0],
            lat=[0.0, 0.0],
            vrho=[0.0, 0.0],
            vlon=[0.0, 0.0],
            vlat=[0.0, 0.0],
            time=Timestamp.from_mjd([60000.0, 60000.0]),
            origin=Origin.from_kwargs(code=["500", "500"]),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id(aberration_mode="none")
    assert len(collapsed) == 1
    # Mean rho = 0*0.25 + 10*0.75 = 7.5
    assert collapsed.coordinates.values[0][0] == pytest.approx(7.5, abs=1e-12)
    # Longitude mean uses a circular weighted mean.
    lon = np.deg2rad(np.array([10.0, 20.0]))
    w = np.array([0.25, 0.75])
    lon_mean = (
        np.degrees(np.arctan2(np.sum(w * np.sin(lon)), np.sum(w * np.cos(lon)))) + 360.0
    ) % 360.0
    assert collapsed.coordinates.values[0][1] == pytest.approx(lon_mean, abs=1e-12)

    cov = collapsed.coordinates.covariance.to_matrix()[0]
    # Only rho varies; expected var(rho) = sum(w_cov*(x-mean)^2)
    # mean_rho = 7.5, residuals = [-7.5, 2.5]
    expected = 0.1 * (-7.5) ** 2 + 0.9 * (2.5) ** 2
    assert cov[0, 0] == pytest.approx(expected, abs=1e-12)


def test_VariantEphemeris_collapse_sigma_points_orbit_major_matches_generic():
    """Fast sigma-point collapse should match generic collapse (up to ordering)."""
    rng = np.random.default_rng(0)
    n_orbits = 3
    n_variants = 13
    n_times = 5

    orbit_ids = np.array([f"obj{i}" for i in range(n_orbits)], dtype=object)
    base_orbit_ids = np.repeat(orbit_ids, n_variants)  # (O*K,)

    variant_id_base = np.tile(
        np.arange(n_variants, dtype=np.int64).astype(str), n_orbits
    ).astype(
        object
    )  # (O*K,)

    weights_base = rng.random((n_orbits, n_variants))
    weights_base /= np.sum(weights_base, axis=1, keepdims=True)
    weights_cov_base = rng.random((n_orbits, n_variants))
    weights_cov_base /= np.sum(weights_cov_base, axis=1, keepdims=True)

    # Shared time+origin grid for one base variant (length n_times), repeated for all base variants.
    times = Timestamp.from_mjd(
        np.linspace(60000.0, 60000.1, n_times, dtype=np.float64), scale="tdb"
    )
    origin_codes = np.array(["500", "X05", "500", "X05", "500"][:n_times], dtype=object)
    time_rep_mjd = np.tile(
        times.mjd().to_numpy(zero_copy_only=False), n_orbits * n_variants
    )
    origin_rep = np.tile(origin_codes, n_orbits * n_variants)

    total = n_orbits * n_variants * n_times
    orbit_id_rows = np.repeat(base_orbit_ids, n_times)
    object_id_rows = orbit_id_rows
    variant_id_rows = np.repeat(variant_id_base, n_times)
    weights_rows = np.repeat(weights_base.reshape(-1), n_times)
    weights_cov_rows = np.repeat(weights_cov_base.reshape(-1), n_times)

    # Synthetic spherical values (deg for lon/lat); ensure wrap-around cases for longitude.
    rho = rng.normal(loc=1.0, scale=0.1, size=total)
    lon = rng.uniform(low=0.0, high=360.0, size=total)
    lon[::7] = 359.5
    lon[1::7] = 0.5
    lat = rng.normal(loc=0.0, scale=1.0, size=total)
    vrho = rng.normal(loc=0.0, scale=0.01, size=total)
    vlon = rng.normal(loc=0.0, scale=0.01, size=total)
    vlat = rng.normal(loc=0.0, scale=0.01, size=total)

    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=orbit_id_rows,
        object_id=object_id_rows,
        variant_id=variant_id_rows,
        weights=weights_rows,
        weights_cov=weights_cov_rows,
        coordinates=SphericalCoordinates.from_kwargs(
            rho=rho,
            lon=lon,
            lat=lat,
            vrho=vrho,
            vlon=vlon,
            vlat=vlat,
            time=Timestamp.from_mjd(time_rep_mjd, scale="tdb"),
            origin=Origin.from_kwargs(code=origin_rep),
            frame="equatorial",
        ),
    )

    fast = variant_ephemeris.collapse_sigma_points_orbit_major(
        n_times=n_times, n_variants=n_variants
    )
    slow = variant_ephemeris.collapse_by_object_id(aberration_mode="none")

    # Compare after sorting to common order.
    sort_keys = [
        "object_id",
        "coordinates.time.days",
        "coordinates.time.nanos",
        "coordinates.origin.code",
    ]
    fast_s = fast.sort_by(sort_keys)
    slow_s = slow.sort_by(sort_keys)

    np.testing.assert_allclose(
        np.asarray(fast_s.coordinates.values, dtype=np.float64),
        np.asarray(slow_s.coordinates.values, dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fast_s.coordinates.covariance.to_matrix().astype(np.float64, copy=False),
        slow_s.coordinates.covariance.to_matrix().astype(np.float64, copy=False),
        rtol=0,
        atol=1e-12,
    )


def test_VariantEphemeris_collapse_by_object_id_aberration_mode_none_leaves_nulls():
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        weights=[0.5, 0.5],
        weights_cov=[0.5, 0.5],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1],
            lon=[1.0, 1.1],
            lat=[1.0, 1.1],
            vrho=[0.1, 0.11],
            vlon=[0.1, 0.11],
            vlat=[0.1, 0.11],
            time=Timestamp.from_mjd([60000] * 2),
            origin=Origin.from_kwargs(code=["500"] * 2),
            frame="equatorial",
        ),
    )
    collapsed = variant_ephemeris.collapse_by_object_id(aberration_mode="none")
    assert len(collapsed) == 1
    assert pc.all(pc.is_null(collapsed.aberrated_coordinates.x)).as_py()
    assert pc.all(pc.is_null(collapsed.light_time)).as_py()


def test_VariantEphemeris_collapse_by_object_id_partial_aberrated_raises():
    """Aberrated coordinates are ignored/dropped and regenerated after collapse."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1],
            lon=[1.0, 1.1],
            lat=[1.0, 1.1],
            vrho=[0.1, 0.11],
            vlon=[0.1, 0.11],
            vlat=[0.1, 0.11],
            time=Timestamp.from_mjd([60000] * 2),
            origin=Origin.from_kwargs(code=["500"] * 2),
            frame="equatorial",
        ),
        aberrated_coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, None],
            y=[1.0, None],
            z=[1.0, None],
            vx=[0.1, None],
            vy=[0.1, None],
            vz=[0.1, None],
            time=Timestamp.from_mjd([60000] * 2),
            origin=Origin.from_kwargs(code=["SUN"] * 2),
            frame="ecliptic",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()
    assert len(collapsed) == 1
    assert not pc.all(pc.is_null(collapsed.aberrated_coordinates.x)).as_py()
    assert not pc.all(pc.is_null(collapsed.light_time)).as_py()


def test_VariantEphemeris_collapse_by_object_id_aberrated_times_can_vary():
    """Collapsed aberrated emission times should be consistent with light_time."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1"],
        object_id=["obj1", "obj1", "obj1"],
        variant_id=["0", "1", "2"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1, 0.9],
            lon=[1.0, 1.1, 0.9],
            lat=[1.0, 1.1, 0.9],
            vrho=[0.1, 0.11, 0.09],
            vlon=[0.1, 0.11, 0.09],
            vlat=[0.1, 0.11, 0.09],
            time=Timestamp.from_mjd([60000.0] * 3, scale="utc"),
            origin=Origin.from_kwargs(code=["500"] * 3),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()
    assert len(collapsed) == 1

    coords_tdb = collapsed.coordinates.time.rescale("tdb")
    aberr_tdb = collapsed.aberrated_coordinates.time.rescale("tdb")
    delta_days, delta_nanos = coords_tdb.difference(aberr_tdb)
    fractional_days = pc.divide(delta_nanos, 86400 * 1e9)
    delta = pc.add(delta_days, fractional_days).to_numpy(zero_copy_only=False)
    lt = collapsed.light_time.to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(delta, lt, atol=1e-6)
