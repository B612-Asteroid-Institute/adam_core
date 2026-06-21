from __future__ import annotations

import numpy as np
import pytest

from adam_assist_rust._arrow import (
    ORBIT_FLAT_FIELDS,
    SCHEMA_NAME,
    orbits_from_flat_record_batch,
    orbits_to_flat_record_batch,
)
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp


def _orbits(*, with_covariance: bool) -> Orbits:
    covariance = None
    if with_covariance:
        sigmas = np.array(
            [
                [1.0e-9, 2.0e-9, 3.0e-9, 1.0e-10, 2.0e-10, 3.0e-10],
                [1.5e-9, 2.5e-9, 3.5e-9, 1.5e-10, 2.5e-10, 3.5e-10],
            ],
            dtype=np.float64,
        )
        covariance = CoordinateCovariances.from_sigmas(sigmas)
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.05, 1.10],
        y=[0.0, 0.02],
        z=[0.0, -0.001],
        vx=[0.0, -0.0002],
        vy=[0.016787, 0.0159],
        vz=[0.0, 0.0001],
        covariance=covariance,
        time=Timestamp.from_mjd([60000.0, 60000.25], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["orbit-a", "orbit-b"],
        object_id=["obj-a", None],
        coordinates=coordinates,
    )


def test_flat_field_order_matches_rust_orbit_schema_contract() -> None:
    rust_api = pytest.importorskip("adam_core._rust.api")
    orbit_fields, orbit_metadata = rust_api.orbit_schema_metadata()
    assert tuple(orbit_fields) == ORBIT_FLAT_FIELDS
    assert orbit_metadata["adam_core_schema"] == SCHEMA_NAME
    assert len(ORBIT_FLAT_FIELDS) == 47


def test_flat_record_batch_round_trip_with_covariance() -> None:
    orbits = _orbits(with_covariance=True)
    batch = orbits_to_flat_record_batch(orbits)

    assert tuple(batch.schema.names) == ORBIT_FLAT_FIELDS
    metadata = {
        key.decode(): value.decode()
        for key, value in (batch.schema.metadata or {}).items()
    }
    assert metadata["adam_core_schema"] == SCHEMA_NAME
    assert metadata["adam_core_frame"] == "ecliptic"
    assert metadata["adam_core_time_scale"] == "tdb"
    assert metadata["adam_core_covariance"] == "present"

    restored = orbits_from_flat_record_batch(batch)
    assert restored.orbit_id.to_pylist() == orbits.orbit_id.to_pylist()
    assert restored.object_id.to_pylist() == orbits.object_id.to_pylist()
    np.testing.assert_array_equal(
        restored.coordinates.values, orbits.coordinates.values
    )
    np.testing.assert_array_equal(
        restored.coordinates.time.days.to_numpy(zero_copy_only=False),
        orbits.coordinates.time.days.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_array_equal(
        restored.coordinates.time.nanos.to_numpy(zero_copy_only=False),
        orbits.coordinates.time.nanos.to_numpy(zero_copy_only=False),
    )
    assert restored.coordinates.time.scale == orbits.coordinates.time.scale
    assert restored.coordinates.frame == orbits.coordinates.frame
    assert (
        restored.coordinates.origin.code.to_pylist()
        == orbits.coordinates.origin.code.to_pylist()
    )
    np.testing.assert_array_equal(
        restored.coordinates.covariance.to_matrix(),
        orbits.coordinates.covariance.to_matrix(),
    )


def test_flat_record_batch_round_trip_without_covariance() -> None:
    orbits = _orbits(with_covariance=False)
    batch = orbits_to_flat_record_batch(orbits)

    metadata = {
        key.decode(): value.decode()
        for key, value in (batch.schema.metadata or {}).items()
    }
    assert metadata["adam_core_covariance"] == "absent"
    assert batch.column(ORBIT_FLAT_FIELDS.index("covariance_00")).null_count == len(
        orbits
    )

    restored = orbits_from_flat_record_batch(batch)
    assert restored.orbit_id.to_pylist() == orbits.orbit_id.to_pylist()
    np.testing.assert_array_equal(
        restored.coordinates.values, orbits.coordinates.values
    )
    assert restored.coordinates.covariance.is_all_nan()
