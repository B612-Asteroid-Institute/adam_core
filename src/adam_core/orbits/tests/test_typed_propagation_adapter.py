"""W12 typed propagation adapter contract parity (bead personal-cmy.15).

The typed adapter runs quivr ``Orbits``/``VariantOrbits`` through the
Rust-canonical ``TwoBodyPropagator`` ``PropagationRequest`` pipeline in one
crossing. Parity is gated against the promoted public Rust path
(``dynamics.propagate_2body``), which itself carries randomized legacy parity
at all three speed lanes, so agreement here is transitively agreement with
baseline-main. The UTC test additionally gates the provider-owned ERFA
rescale against the fixture-gated Python ``Timestamp.rescale``.
"""

import numpy as np
import pytest

from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
from adam_core.dynamics import propagate_2body
from adam_core.orbits import Orbits
from adam_core.orbits.arrow_bridge import _propagate_orbits_typed_arrow
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp

EPOCH_MJD = 60000.0


def _orbits(
    scale: str = "tdb", with_covariance: bool = False, nan_row: bool = False
) -> Orbits:
    # Physically valid near-circular heliocentric orbits (v ~ sqrt(mu/r)).
    x = np.array([1.0, 1.5, 2.0, 2.5])
    vy = np.array([0.01720, 0.01405, 0.01216, 0.01088])
    n = len(x)
    kwargs = dict(
        x=x,
        y=[0.0] * n,
        z=[0.05] * n,
        vx=[0.0] * n,
        vy=vy,
        vz=[1e-4] * n,
        time=Timestamp.from_mjd([EPOCH_MJD] * n, scale=scale),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    )
    if nan_row:
        kwargs["x"] = np.array([np.nan, 1.5, 2.0, 2.5])
    if with_covariance:
        base = np.diag([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12])
        kwargs["covariance"] = CoordinateCovariances.from_matrix(
            np.stack([base * (i + 1) for i in range(n)])
        )
    coordinates = CartesianCoordinates.from_kwargs(**kwargs)
    return Orbits.from_kwargs(
        orbit_id=[f"o{i}" for i in range(n)],
        object_id=[f"obj{i}" for i in range(n)],
        coordinates=coordinates,
    )


def _times() -> Timestamp:
    return Timestamp.from_mjd([60010.0, 60100.0, 60365.0], scale="tdb")


def _sort(table):
    return table.sort_by(
        ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
    )


def test_typed_adapter_matches_public_propagate_2body():
    orbits = _orbits()
    times = _times()
    typed, valid = _propagate_orbits_typed_arrow(orbits, times)
    assert all(valid)
    public = propagate_2body(orbits, times)
    typed, public = _sort(typed), _sort(public)
    assert typed.orbit_id.to_pylist() == public.orbit_id.to_pylist()
    np.testing.assert_allclose(
        typed.coordinates.values, public.coordinates.values, rtol=0, atol=1e-13
    )
    np.testing.assert_array_equal(
        typed.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        public.coordinates.time.mjd().to_numpy(zero_copy_only=False),
    )


def test_typed_adapter_covariance_matches_public():
    orbits = _orbits(with_covariance=True)
    times = _times()
    typed, valid = _propagate_orbits_typed_arrow(orbits, times, covariance=True)
    assert all(valid)
    public = propagate_2body(orbits, times)
    typed, public = _sort(typed), _sort(public)
    np.testing.assert_allclose(
        typed.coordinates.values, public.coordinates.values, rtol=0, atol=1e-13
    )
    typed_cov = typed.coordinates.covariance.to_matrix()
    public_cov = public.coordinates.covariance.to_matrix()
    assert np.isfinite(typed_cov).all()
    np.testing.assert_allclose(typed_cov, public_cov, rtol=1e-9, atol=1e-20)


def test_typed_adapter_preserves_variant_metadata():
    orbits = _orbits()
    n = len(orbits)
    variants = VariantOrbits.from_kwargs(
        orbit_id=orbits.orbit_id,
        object_id=orbits.object_id,
        variant_id=[str(i) for i in range(n)],
        weights=np.linspace(0.1, 0.4, n),
        weights_cov=np.linspace(0.2, 0.5, n),
        coordinates=orbits.coordinates,
    )
    times = _times()
    typed, valid = _propagate_orbits_typed_arrow(variants, times)
    assert all(valid)
    assert isinstance(typed, VariantOrbits)
    assert len(typed) == n * len(times)
    typed = typed.sort_by(
        ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
    )
    # Variant metadata must survive per output row.
    expected_weights = np.repeat(np.linspace(0.1, 0.4, n), len(times))
    np.testing.assert_allclose(
        typed.weights.to_numpy(zero_copy_only=False), expected_weights, rtol=0, atol=0
    )
    # States equal the plain-Orbits typed path on identical coordinates.
    plain, _ = _propagate_orbits_typed_arrow(orbits, times)
    plain = _sort(plain)
    np.testing.assert_array_equal(typed.coordinates.values, plain.coordinates.values)


def test_typed_adapter_rescales_utc_epochs_via_provider():
    """UTC orbit epochs must rescale through the provider-owned ERFA service
    and agree with pre-rescaling via the fixture-gated Python
    ``Timestamp.rescale`` path."""
    orbits_utc = _orbits(scale="utc")
    coordinates_tdb = orbits_utc.coordinates.set_column(
        "time", orbits_utc.coordinates.time.rescale("tdb")
    )
    orbits_tdb = orbits_utc.set_column("coordinates", coordinates_tdb)
    times = _times()
    from_utc, valid_utc = _propagate_orbits_typed_arrow(orbits_utc, times)
    from_tdb, valid_tdb = _propagate_orbits_typed_arrow(orbits_tdb, times)
    assert all(valid_utc) and all(valid_tdb)
    np.testing.assert_array_equal(
        _sort(from_utc).coordinates.values, _sort(from_tdb).coordinates.values
    )


def test_typed_adapter_reports_invalid_rows():
    orbits = _orbits(nan_row=True)
    times = _times()
    typed, valid = _propagate_orbits_typed_arrow(orbits, times)
    valid = np.asarray(valid)
    assert valid.shape == (len(orbits) * len(times),)
    assert not valid.all()
    assert valid.sum() == (len(orbits) - 1) * len(times)


def test_typed_adapter_ut1_epochs_fail_loudly():
    orbits = _orbits(scale="ut1")
    with pytest.raises(ValueError, match="typed propagation failed|rescale"):
        _propagate_orbits_typed_arrow(orbits, _times())
