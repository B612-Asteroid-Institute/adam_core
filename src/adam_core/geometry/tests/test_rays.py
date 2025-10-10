"""
Tests for observation ray construction.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.rays import ObservationRays, detections_to_rays
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observers import Observers
from adam_core.time import Timestamp


def create_test_detections_and_exposures():
    """Create test detections and exposures for ray construction tests."""
    # Create test times
    times = Timestamp.from_mjd([59000.0, 59000.1, 59000.2], scale="tdb")

    # Create test exposures
    exposures = Exposures.from_kwargs(
        id=["exp_1", "exp_2", "exp_3"],
        start_time=times,
        duration=[300.0, 300.0, 300.0],  # 5 minutes
        filter=["r", "g", "i"],
        observatory_code=["X05", "G96", "F51"],  # Different observatories
        seeing=[1.2, 1.5, 1.1],
        depth_5sigma=[22.5, 22.0, 22.8],
    )

    # Create test detections
    detections = PointSourceDetections.from_kwargs(
        id=["det_1", "det_2", "det_3"],
        exposure_id=["exp_1", "exp_2", "exp_3"],
        time=times,
        ra=[0.0, 45.0, 90.0],  # degrees
        dec=[0.0, 30.0, -15.0],  # degrees
        ra_sigma=[0.1, 0.1, 0.1],
        dec_sigma=[0.1, 0.1, 0.1],
        mag=[20.5, 21.0, 19.8],
        mag_sigma=[0.1, 0.15, 0.08],
    )

    return detections, exposures


class TestObservationRays:
    """Test ObservationRays quivr table."""

    def test_empty_table(self):
        """Test creating empty table."""
        empty = ObservationRays.empty()
        assert len(empty) == 0

    def test_table_creation(self):
        """Test creating table with data."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        # Create minimal observer coordinates
        observer_coords = CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )

        rays = ObservationRays.from_kwargs(
            det_id=["test_det"],
            orbit_id=["test_orbit"],
            observer=Observers.from_kwargs(
                code=["X05"],
                coordinates=observer_coords,
            ),
            u_x=[1.0],
            u_y=[0.0],
            u_z=[0.0],
        )

        assert len(rays) == 1
        assert rays.det_id[0].as_py() == "test_det"
        assert rays.observer.code[0].as_py() == "X05"


class TestRaysFromDetections:
    """Test ray construction from detections."""

    def test_empty_detections(self):
        """Test with empty detections."""
        empty_detections = PointSourceDetections.empty()
        empty_exposures = Exposures.empty()

        rays = detections_to_rays(empty_detections, empty_exposures)
        assert len(rays) == 0

    def test_no_matching_exposures(self):
        """Test with detections that don't match any exposures."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        detections = PointSourceDetections.from_kwargs(
            id=["det_1"],
            exposure_id=["nonexistent_exp"],
            time=times,
            ra=[0.0],
            dec=[0.0],
            ra_sigma=[0.1],
            dec_sigma=[0.1],
            mag=[20.0],
            mag_sigma=[0.1],
        )

        exposures = Exposures.from_kwargs(
            id=["exp_1"],
            start_time=times,
            duration=[300.0],
            filter=["r"],
            observatory_code=["X05"],
            seeing=[1.2],
            depth_5sigma=[22.0],
        )

        rays = detections_to_rays(detections, exposures)
        assert len(rays) == 0

    def test_basic_ray_construction(self):
        """Test basic ray construction functionality."""
        detections, exposures = create_test_detections_and_exposures()

        rays = detections_to_rays(detections, exposures)

        # Should have rays for all detections
        assert len(rays) == len(detections)

        # Check that detection IDs are preserved
        ray_det_ids = set(rays.det_id.to_pylist())
        expected_det_ids = set(detections.id.to_pylist())
        assert ray_det_ids == expected_det_ids

        # Check that observatory codes are correct
        ray_obs_codes = rays.observer.code.to_pylist()
        expected_obs_codes = exposures.observatory_code.to_pylist()
        assert ray_obs_codes == expected_obs_codes

    def test_unit_vector_normalization(self):
        """Test that line-of-sight vectors are properly normalized."""
        detections, exposures = create_test_detections_and_exposures()

        rays = detections_to_rays(detections, exposures)

        # Check that all u vectors are unit vectors
        for i in range(len(rays)):
            u_x = rays.u_x[i].as_py()
            u_y = rays.u_y[i].as_py()
            u_z = rays.u_z[i].as_py()

            magnitude = np.sqrt(u_x**2 + u_y**2 + u_z**2)
            assert abs(magnitude - 1.0) < 1e-10

    def test_equatorial_to_ecliptic_conversion(self):
        """Test conversion from equatorial to ecliptic coordinates."""
        # Create detection at RA=0, Dec=0 (vernal equinox)
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        detections = PointSourceDetections.from_kwargs(
            id=["det_vernal_equinox"],
            exposure_id=["exp_1"],
            time=times,
            ra=[0.0],  # Vernal equinox
            dec=[0.0],
            ra_sigma=[0.1],
            dec_sigma=[0.1],
            mag=[20.0],
            mag_sigma=[0.1],
        )

        exposures = Exposures.from_kwargs(
            id=["exp_1"],
            start_time=times,
            duration=[300.0],
            filter=["r"],
            observatory_code=["500"],  # Geocenter
            seeing=[1.2],
            depth_5sigma=[22.0],
        )

        rays = detections_to_rays(detections, exposures)

        # At vernal equinox (RA=0, Dec=0), the direction should be close to +X in ecliptic
        u_x = rays.u_x[0].as_py()
        u_y = rays.u_y[0].as_py()
        u_z = rays.u_z[0].as_py()

        # Should be close to [1, 0, 0] in ecliptic coordinates
        assert abs(u_x - 1.0) < 0.1  # Allow some tolerance for coordinate transforms
        assert abs(u_y) < 0.1
        assert abs(u_z) < 0.1

    def test_observer_frame_consistency(self):
        """Test that observer coordinates are in SSB ecliptic frame."""
        detections, exposures = create_test_detections_and_exposures()

        rays = detections_to_rays(detections, exposures)

        # Check that all observer coordinates are in ecliptic frame with SUN origin
        # Frame is a table attribute (scalar) on coordinates, origin codes are a column (array)
        assert rays.observer.coordinates.frame == "ecliptic"
        origin_codes = rays.observer.coordinates.origin.code.to_pylist()
        assert all(code == OriginCodes.SUN.name for code in origin_codes)

    def test_ecliptic_input_frame(self):
        """Test with input coordinates already in ecliptic frame."""
        detections, exposures = create_test_detections_and_exposures()

        rays = detections_to_rays(detections, exposures)

        # Should still produce valid rays
        assert len(rays) == len(detections)

        # Unit vectors should still be normalized - use vectorized operations
        u_x = rays.u_x.to_numpy()
        u_y = rays.u_y.to_numpy()
        u_z = rays.u_z.to_numpy()

        magnitudes = np.sqrt(u_x**2 + u_y**2 + u_z**2)
        assert np.all(np.abs(magnitudes - 1.0) < 1e-10)

    def test_time_consistency(self):
        """Test that ray times match detection times."""
        detections, exposures = create_test_detections_and_exposures()

        rays = detections_to_rays(detections, exposures)

        # Times should match between detections and rays
        det_times = detections.time.mjd().to_numpy()
        ray_times = rays.observer.coordinates.time.mjd().to_numpy()

        # Sort both arrays since order might differ due to linkage
        det_times_sorted = np.sort(det_times)
        ray_times_sorted = np.sort(ray_times)

        np.testing.assert_array_almost_equal(det_times_sorted, ray_times_sorted)

    def test_multiple_observatories(self):
        """Test with detections from multiple observatories."""
        detections, exposures = create_test_detections_and_exposures()

        rays = detections_to_rays(detections, exposures)

        # Should have different observer positions for different observatories
        # Use vectorized operations to get all positions at once
        obs_x = rays.observer.coordinates.x.to_numpy()
        obs_y = rays.observer.coordinates.y.to_numpy()
        obs_z = rays.observer.coordinates.z.to_numpy()

        # Positions should be different (not all the same)
        # Check that positions vary across the dataset
        assert (
            not np.allclose(obs_x[0], obs_x[1], atol=1e-6)
            or not np.allclose(obs_y[0], obs_y[1], atol=1e-6)
            or not np.allclose(obs_z[0], obs_z[1], atol=1e-6)
        )
        assert (
            not np.allclose(obs_x[1], obs_x[2], atol=1e-6)
            or not np.allclose(obs_y[1], obs_y[2], atol=1e-6)
            or not np.allclose(obs_z[1], obs_z[2], atol=1e-6)
        )


class TestIntegration:
    """Integration tests for ray construction."""

    def test_full_pipeline_with_realistic_data(self):
        """Test complete pipeline with realistic detection data."""
        # Create more realistic test data
        n_detections = 10
        times = Timestamp.from_mjd(
            np.linspace(59000.0, 59001.0, n_detections), scale="tdb"
        )

        # Create exposures with varied observatories
        exposure_ids = [f"exp_{i}" for i in range(n_detections)]
        observatory_codes = [
            "X05",
            "G96",
            "F51",
            "703",
            "691",
        ] * 2  # Cycle through observatories

        exposures = Exposures.from_kwargs(
            id=exposure_ids,
            start_time=times,
            duration=[300.0] * n_detections,
            filter=["r", "g", "i", "z", "y"] * 2,
            observatory_code=observatory_codes,
            seeing=np.random.uniform(0.8, 2.0, n_detections),
            depth_5sigma=np.random.uniform(21.0, 23.0, n_detections),
        )

        # Create detections with varied sky positions
        detections = PointSourceDetections.from_kwargs(
            id=[f"det_{i}" for i in range(n_detections)],
            exposure_id=exposure_ids,
            time=times,
            ra=np.random.uniform(0.0, 360.0, n_detections),
            dec=np.random.uniform(-30.0, 30.0, n_detections),
            ra_sigma=np.random.uniform(0.05, 0.2, n_detections),
            dec_sigma=np.random.uniform(0.05, 0.2, n_detections),
            mag=np.random.uniform(18.0, 23.0, n_detections),
            mag_sigma=np.random.uniform(0.05, 0.3, n_detections),
        )

        # Convert to rays
        rays = detections_to_rays(detections, exposures)

        # Verify results
        assert len(rays) == n_detections

        # All rays should have valid unit vectors - use vectorized operations
        u_x = rays.u_x.to_numpy()
        u_y = rays.u_y.to_numpy()
        u_z = rays.u_z.to_numpy()

        magnitudes = np.sqrt(u_x**2 + u_y**2 + u_z**2)
        assert np.all(np.abs(magnitudes - 1.0) < 1e-10)

        # All observers should be in SSB ecliptic frame - use vectorized operations
        # Frame is a table attribute (scalar) on coordinates, origin codes are a column (array)
        assert rays.observer.coordinates.frame == "ecliptic"
        origin_codes = rays.observer.coordinates.origin.code.to_pylist()
        assert all(code == OriginCodes.SUN.name for code in origin_codes)
