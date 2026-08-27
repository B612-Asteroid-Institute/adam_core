"""Round-trip gates for the Rust-canonical observation data model
(bead personal-cmy.20): every supported quivr observation table crosses to the
Rust batch and back losslessly (values bit-exact, time scales preserved)."""

import pytest

from ...time import Timestamp
from ..ades import ADESObservations
from ..arrow_bridge import round_trip_observations
from ..associations import Associations
from ..detections import PointSourceDetections
from ..exposures import Exposures
from ..photometry import Photometry
from ..source_catalog import SourceCatalog


def _assert_round_trip(table):
    round_tripped = round_trip_observations(table)
    assert type(round_tripped) is type(table)
    assert round_tripped.table.equals(
        table.table
    ), f"{type(table).__name__} round trip is not lossless"
    return round_tripped


def _sample_ades() -> ADESObservations:
    return ADESObservations.from_kwargs(
        permID=["12345", None],
        provID=[None, "2024 AB1"],
        trkSub=[None, None],
        obsSubID=["sub-1", None],
        obsTime=Timestamp.from_kwargs(
            days=[60000, 60001], nanos=[0, 43_200_000_000_000], scale="utc"
        ),
        rmsTime=[1.0, None],
        ra=[10.5, 200.25],
        dec=[-5.0, 45.5],
        rmsRACosDec=[0.1, None],
        rmsDec=[0.1, 0.2],
        rmsCorr=[None, -0.5],
        mag=[21.2, None],
        rmsMag=[0.1, None],
        band=["r", None],
        stn=["X05", "W84"],
        mode=["CCD", "CCD"],
        astCat=["Gaia2", "Gaia2"],
        photCat=["Gaia2", None],
        logSNR=[1.5, None],
        seeing=[None, 0.8],
        exp=[30.0, 30.0],
        remarks=[None, "  padded remark "],
    )


def test_ades_round_trip():
    round_tripped = _assert_round_trip(_sample_ades())
    assert round_tripped.obsTime.scale == "utc"


def test_ades_empty_round_trip():
    _assert_round_trip(ADESObservations.empty())


def test_detections_round_trip():
    detections = PointSourceDetections.from_kwargs(
        id=["det-1", "det-2"],
        exposure_id=["exp-1", None],
        time=Timestamp.from_kwargs(days=[59000, 59001], nanos=[1, 2], scale="tai"),
        ra=[0.0, 359.9],
        ra_sigma=[None, 0.2],
        dec=[-89.9, 89.9],
        dec_sigma=[0.1, None],
        mag=[20.0, None],
        mag_sigma=[None, 0.05],
    )
    round_tripped = _assert_round_trip(detections)
    assert round_tripped.time.scale == "tai"


def test_exposures_round_trip():
    exposures = Exposures.from_kwargs(
        id=["exp-1", "exp-2"],
        start_time=Timestamp.from_kwargs(
            days=[60200, 60201], nanos=[500, 0], scale="utc"
        ),
        duration=[30.0, 60.0],
        filter=["g", "r"],
        observatory_code=["X05", "W84"],
        seeing=[None, 1.1],
        depth_5sigma=[24.5, None],
    )
    _assert_round_trip(exposures)


def test_associations_round_trip():
    associations = Associations.from_kwargs(
        detection_id=["det-1", "det-2", "det-3"],
        object_id=["object-1", None, "object-2"],
    )
    _assert_round_trip(associations)


def test_photometry_round_trip():
    photometry = Photometry.from_kwargs(
        time=Timestamp.from_kwargs(days=[60123], nanos=[987_654_321], scale="tdb"),
        mag=[19.5],
        mag_sigma=[None],
        filter=["i"],
    )
    round_tripped = _assert_round_trip(photometry)
    assert round_tripped.time.scale == "tdb"


def test_source_catalog_round_trip_with_null_exposure_time():
    catalog = SourceCatalog.from_kwargs(
        id=["src-1", "src-2"],
        exposure_id=["exp-1", None],
        time=Timestamp.from_kwargs(days=[60300, 60301], nanos=[0, 1], scale="utc"),
        ra=[15.0, 30.0],
        dec=[-15.0, 30.0],
        ra_sigma=[0.1, None],
        dec_sigma=[0.1, None],
        radec_corr=[0.5, None],
        mag=[20.0, None],
        mag_sigma=[0.1, None],
        fwhm=[1.0, None],
        a=[1.2, None],
        a_sigma=[None, None],
        b=[0.8, None],
        b_sigma=[None, None],
        pa=[45.0, None],
        pa_sigma=[None, None],
        observatory_code=["X05", "W84"],
        filter=["r", None],
        exposure_start_time=None,
        exposure_duration=[30.0, None],
        exposure_seeing=[None, None],
        exposure_depth_5sigma=[24.0, None],
        object_id=[None, "object-9"],
        catalog_id=["cat-1", "cat-1"],
    )
    _assert_round_trip(catalog)


def test_unsupported_table_raises():
    from ...orbits import Orbits

    with pytest.raises(TypeError, match="Unsupported observation table"):
        round_trip_observations(Orbits.empty())
