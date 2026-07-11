"""Parity gates for the Rust HEALPix port (bead personal-cmy.37.7.1).

The public ``PointSourceDetections.healpixels`` /  ``group_by_healpixel``
surface dispatches to the Rust ``ang2pix`` port. healpy (the legacy
implementation, still installed as the test oracle) must agree exactly:
pixel indices, dtype, ascending group order, and nside validation errors.
"""

import healpy
import numpy as np
import pytest

from adam_core import _rust_native

from ..detections import PointSourceDetections

RNG = np.random.default_rng(20260711)


def _random_lonlat(n: int) -> tuple[np.ndarray, np.ndarray]:
    ra = RNG.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(RNG.uniform(-1.0, 1.0, n)))
    return ra, dec


def _structured_lonlat() -> tuple[np.ndarray, np.ndarray]:
    # Poles, equator, meridians, the z = 2/3 equatorial/polar boundary, and
    # the healpix_cxx near-pole `have_sth` branch thresholds (theta = 0.01
    # and theta = 3.14159 - 0.01).
    boundary_dec = np.degrees(np.arcsin(2.0 / 3.0))
    north_threshold = 90.0 - np.degrees(0.01)
    south_threshold = 90.0 - np.degrees(3.14159 - 0.01)
    ra_anchors = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 359.999999, 360.0])
    dec_anchors = np.array(
        [
            90.0,
            89.999999,
            north_threshold + 1e-9,
            north_threshold - 1e-9,
            boundary_dec + 1e-9,
            boundary_dec,
            boundary_dec - 1e-9,
            45.0,
            1e-12,
            0.0,
            -1e-12,
            -45.0,
            -boundary_dec,
            south_threshold + 1e-9,
            south_threshold - 1e-9,
            -89.999999,
            -90.0,
        ]
    )
    ra_grid, dec_grid = np.meshgrid(ra_anchors, dec_anchors)
    return ra_grid.ravel(), dec_grid.ravel()


def _near_pole_lonlat(n: int) -> tuple[np.ndarray, np.ndarray]:
    ra = RNG.uniform(0.0, 360.0, n)
    band = RNG.uniform(81.9, 90.0, n)
    sign = np.where(RNG.uniform(size=n) < 0.5, 1.0, -1.0)
    return ra, band * sign


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize("nside", [1, 2, 16, 64, 1024, 2**20])
def test_healpixels_match_healpy_power_of_two(nside, nest):
    for ra, dec in [
        _random_lonlat(20_000),
        _structured_lonlat(),
        _near_pole_lonlat(5_000),
    ]:
        expected = healpy.ang2pix(nside, ra, dec, nest=nest, lonlat=True)
        actual = _rust_native.detections_healpixels_numpy(ra, dec, nside, nest)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("nside", [3, 57, 1000])
def test_healpixels_match_healpy_ring_non_power_of_two(nside):
    for ra, dec in [
        _random_lonlat(20_000),
        _structured_lonlat(),
        _near_pole_lonlat(5_000),
    ]:
        expected = healpy.ang2pix(nside, ra, dec, nest=False, lonlat=True)
        actual = _rust_native.detections_healpixels_numpy(ra, dec, nside, False)
        np.testing.assert_array_equal(actual, expected)


def test_invalid_nside_error_matches_healpy():
    det = PointSourceDetections.from_kwargs(
        id=["d1"], exposure_id=["e1"], ra=[10.0], dec=[5.0], mag=[12.0]
    )
    with pytest.raises(ValueError) as ours:
        det.healpixels(nside=3, nest=True)
    with pytest.raises(ValueError) as theirs:
        healpy.ang2pix(3, np.array([10.0]), np.array([5.0]), nest=True, lonlat=True)
    assert str(ours.value) == str(theirs.value)

    with pytest.raises(ValueError):
        det.healpixels(nside=0, nest=False)
    with pytest.raises(ValueError):
        det.healpixels(nside=2**30, nest=False)


def test_grouping_and_healpix_native_timing():
    from ..arrow_bridge import observations_to_ipc
    from ..associations import Associations

    ra, dec = _random_lonlat(512)
    detections = PointSourceDetections.from_kwargs(
        id=[f"d{i}" for i in range(512)],
        exposure_id=[f"e{i % 7}" for i in range(512)],
        ra=ra,
        dec=np.clip(dec, -90.0, 90.0),
        mag=np.full(512, 20.0),
    )
    associations = Associations.from_kwargs(
        detection_id=detections.id,
        object_id=[f"o{i % 5}" if i % 11 else None for i in range(512)],
    )

    raw_detections = observations_to_ipc(detections)
    raw_associations = observations_to_ipc(associations)

    lanes = [
        _rust_native.benchmark_association_object_groups_ipc(raw_associations, 2, 2, 1),
        _rust_native.benchmark_detection_exposure_groups_ipc(raw_detections, 2, 2, 1),
        _rust_native.benchmark_detections_healpixels_numpy(
            ra, np.clip(dec, -90.0, 90.0), 64, True, 2, 2, 1
        ),
        _rust_native.benchmark_detection_healpixel_groups_ipc(
            raw_detections, 64, True, 2, 2, 1
        ),
    ]
    for samples in lanes:
        assert all(sample > 0.0 for trial in samples for sample in trial)
