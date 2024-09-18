import numpy as np
import pyarrow.compute as pc
import pytest

from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...time import Timestamp
from ..associations import Associations
from ..detections import PointSourceDetections
from ..exposures import Exposures
from ..photometry import Photometry
from ..source_catalog import SourceCatalog


@pytest.fixture
def source_catalog() -> SourceCatalog:
    source_catalog = SourceCatalog.from_kwargs(
        id=["obs_01", "obs_02", "obs_03", "obs_04", "obs_05", "obs_06"],
        exposure_id=["exp_01", "exp_01", "exp_02", "exp_02", "exp_03", "exp_04"],
        time=Timestamp.from_kwargs(
            days=[59001, 59001, 59001, 59001, 59002, 59003],
            nanos=[0, 0.5 * 1e9, 20 * 1e9, 20 * 1e9, 0, 0],
            scale="utc",
        ),
        ra=[1, 2, 3, 4, 5, 6],
        ra_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dec=[-1, -2, -3, -4, -5, -6],
        dec_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        radec_corr=[0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        mag=[10, 11, 12, 13, 14, 15],
        mag_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        fwhm=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        a=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        a_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        b=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        b_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        pa=[0, 30, 60, 90, 120, 150],
        pa_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        observatory_code=["X05", "X05", "I41", "I41", "X05", "I41"],
        filter=["g", "g", "r", "r", "i", "r"],
        exposure_start_time=Timestamp.from_kwargs(
            days=[59001, 59001, 59001, 59001, 59002, 59003],
            nanos=[0, 0, 0, 0, 0, 0],
            scale="utc",
        ),
        exposure_duration=[30, 30, 40, 40, 30, 40],
        exposure_seeing=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        exposure_depth_5sigma=[24.5, 24.5, 21.0, 21.0, 24.5, 21.5],
        object_id=["obj_01", "obj_02", "obj_03", "obj_04", "obj_05", "obj_06"],
        catalog_id=["Rubin", "Rubin", "ZTF", "ZTF", "Rubin", "ZTF"],
    )
    return source_catalog


def test_SourceCatalog_detections(source_catalog: SourceCatalog) -> None:
    # Test the detections method of the SourceCatalog class
    # returns the detections in the source catalog
    detections_actual = source_catalog.detections()
    detections_expected = PointSourceDetections.from_kwargs(
        id=["obs_01", "obs_02", "obs_03", "obs_04", "obs_05", "obs_06"],
        exposure_id=["exp_01", "exp_01", "exp_02", "exp_02", "exp_03", "exp_04"],
        time=Timestamp.from_kwargs(
            days=[59001, 59001, 59001, 59001, 59002, 59003],
            nanos=[0, 0.5 * 1e9, 20 * 1e9, 20 * 1e9, 0, 0],
            scale="utc",
        ),
        ra=[1, 2, 3, 4, 5, 6],
        ra_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dec=[-1, -2, -3, -4, -5, -6],
        dec_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        mag=[10, 11, 12, 13, 14, 15],
        mag_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )

    assert detections_actual == detections_expected


def test_SourceCatalog_exposures(source_catalog: SourceCatalog) -> None:
    # Test the exposures method of the SourceCatalog class
    # returns the unique exposures in the source catalog
    exposures_actual = source_catalog.exposures()
    exposures_expected = Exposures.from_kwargs(
        id=["exp_01", "exp_02", "exp_03", "exp_04"],
        start_time=Timestamp.from_kwargs(
            days=[59001, 59001, 59002, 59003],
            nanos=[0, 0, 0, 0],
            scale="utc",
        ),
        duration=[30, 40, 30, 40],
        filter=["g", "r", "i", "r"],
        observatory_code=["X05", "I41", "X05", "I41"],
        seeing=[1.0, 1.0, 1.0, 1.0],
        depth_5sigma=[24.5, 21.0, 24.5, 21.5],
    )
    assert exposures_actual == exposures_expected


def test_SourceCatalog_associations(source_catalog: SourceCatalog) -> None:
    # Test the associations method of the SourceCatalog class
    # returns the associations in the source catalog
    associations_actual = source_catalog.associations()
    associations_expected = Associations.from_kwargs(
        detection_id=["obs_01", "obs_02", "obs_03", "obs_04", "obs_05", "obs_06"],
        object_id=["obj_01", "obj_02", "obj_03", "obj_04", "obj_05", "obj_06"],
    )
    assert associations_actual == associations_expected


def test_SourceCatalog_photometry(source_catalog: SourceCatalog) -> None:
    # Test the photometry method of the SourceCatalog class
    # returns the photometry in the source catalog
    photometry_actual = source_catalog.photometry()
    photometry_expected = Photometry.from_kwargs(
        time=Timestamp.from_kwargs(
            days=[59001, 59001, 59001, 59001, 59002, 59003],
            nanos=[0, 0.5 * 1e9, 20 * 1e9, 20 * 1e9, 0, 0],
            scale="utc",
        ),
        mag=[10, 11, 12, 13, 14, 15],
        mag_sigma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        filter=["g", "g", "r", "r", "i", "r"],
    )
    assert photometry_actual == photometry_expected


def test_SourceCatalog_coordinates(source_catalog: SourceCatalog) -> None:
    # Test the coordinates method of the SourceCatalog class
    # returns the astrometry in the source catalog as SphericalCoordinates
    coordinates_actual = source_catalog.coordinates()

    covariances = np.empty((len(source_catalog), 6, 6))
    covariances.fill(np.nan)

    ra_sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) / 3600.0
    dec_sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) / 3600.0
    radec_corr = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])

    covariances[:, 1, 1] = ra_sigma**2
    covariances[:, 2, 2] = dec_sigma**2
    covariances[:, 1, 2] = radec_corr * ra_sigma * dec_sigma
    covariances[:, 2, 1] = covariances[:, 1, 2]

    coordinates_expected = SphericalCoordinates.from_kwargs(
        lon=[1, 2, 3, 4, 5, 6],
        lat=[-1, -2, -3, -4, -5, -6],
        time=Timestamp.from_kwargs(
            days=[59001, 59001, 59001, 59001, 59002, 59003],
            nanos=[0, 0.5 * 1e9, 20 * 1e9, 20 * 1e9, 0, 0],
            scale="utc",
        ),
        covariance=CoordinateCovariances.from_matrix(covariances),
        origin=Origin.from_kwargs(code=["X05", "X05", "I41", "I41", "X05", "I41"]),
        frame="equatorial",
    )

    assert (
        coordinates_actual.flattened_table()
        .drop_columns(["covariance.values"])
        .equals(
            coordinates_expected.flattened_table().drop_columns(["covariance.values"])
        )
    )
    np.testing.assert_almost_equal(
        coordinates_actual.covariance.to_matrix(),
        coordinates_expected.covariance.to_matrix(),
        decimal=12,
    )


def test_SourceCatalog_observers(source_catalog: SourceCatalog) -> None:
    # Test the observers method of the SourceCatalog class
    # returns the observer location at each detection time
    # correctly
    observers_actual = source_catalog.observers(exposure_midpoint=False)
    assert len(observers_actual) == 6
    assert (
        len(
            observers_actual.drop_duplicates(
                subset=["code", "coordinates.time.days", "coordinates.time.nanos"]
            )
        )
        == 5
    )
    assert pc.all(
        pc.equal(
            observers_actual.coordinates.time.days,
            [59001, 59001, 59001, 59001, 59002, 59003],
        )
    ).as_py()
    assert pc.all(
        pc.equal(
            observers_actual.coordinates.time.nanos,
            [0, 0.5 * 1e9, 20 * 1e9, 20 * 1e9, 0, 0],
        )
    ).as_py()
    assert pc.all(
        pc.equal(observers_actual.code, ["X05", "X05", "I41", "I41", "X05", "I41"])
    ).as_py()

    observers_actual = source_catalog.observers(exposure_midpoint=True)
    assert len(observers_actual) == 6
    assert (
        len(
            observers_actual.drop_duplicates(
                subset=["code", "coordinates.time.days", "coordinates.time.nanos"]
            )
        )
        == 4
    )
    assert pc.all(
        pc.equal(
            observers_actual.coordinates.time.days,
            [59001, 59001, 59001, 59001, 59002, 59003],
        )
    ).as_py()
    assert pc.all(
        pc.equal(
            observers_actual.coordinates.time.nanos,
            [15 * 1e9, 15 * 1e9, 20 * 1e9, 20 * 1e9, 15 * 1e9, 20 * 1e9],
        )
    ).as_py()
    assert pc.all(
        pc.equal(observers_actual.code, ["X05", "X05", "I41", "I41", "X05", "I41"])
    ).as_py()


def test_SourceCatalog_healpixels(source_catalog: SourceCatalog) -> None:
    # Test the healpixels method of the SourceCatalog class
    # returns the healpixels for each observation
    healpixels_actual = source_catalog.healpixels(nside=16)
    healpixels_expected = np.array([1087, 1087, 1085, 1085, 1085, 1079])
    np.testing.assert_array_equal(healpixels_actual, healpixels_expected)

    healpixels_actual = source_catalog.healpixels(nside=2)
    healpixels_expected = np.array([16, 16, 16, 16, 16, 16])
    np.testing.assert_array_equal(healpixels_actual, healpixels_expected)
