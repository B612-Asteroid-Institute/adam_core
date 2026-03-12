import os
import pickle

import pytest

try:
    from adam_core.propagator.adam_pyoorb import PYOORBPropagator
except ImportError:
    PYOORBPropagator = None

from ..native_orbit_fitter import NativeOrbitFitter
from ..orbit_fitter import OrbitFitter


def test_native_orbit_fitter_is_orbit_fitter():
    assert issubclass(NativeOrbitFitter, OrbitFitter)


@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_native_orbit_fitter_pickle_roundtrip():
    """NativeOrbitFitter must survive pickle for Ray serialization."""
    fitter = NativeOrbitFitter(
        propagator_class=PYOORBPropagator,
        min_obs=6,
        rchi2_threshold=10.0,
    )
    restored = pickle.loads(pickle.dumps(fitter))
    assert restored.propagator_class is PYOORBPropagator
    assert restored.min_obs == 6
    assert restored.rchi2_threshold == 10.0


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_native_orbit_fitter_refine_fit(pure_iod_orbit):
    """refine_fit should improve the reduced chi2 of an IOD orbit."""
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    fitter = NativeOrbitFitter(
        propagator_class=PYOORBPropagator,
        min_obs=6,
        rchi2_threshold=10.0,
        contamination_percentage=20.0,
    )

    fitted_orbit, fitted_orbit_members = fitter.refine_fit(orbit, observations, propagator)

    assert len(fitted_orbit) == 1
    assert len(fitted_orbit_members) == len(observations)
    assert fitted_orbit.reduced_chi2[0].as_py() < orbit.reduced_chi2[0].as_py()


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_native_orbit_fitter_full_od(pure_iod_orbit):
    """full_od should return a converged orbit from raw observations."""
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    # Use object_id from the orbit fixture
    object_id = orbit.object_id[0].as_py()

    fitter = NativeOrbitFitter(
        propagator_class=PYOORBPropagator,
        min_obs=6,
        rchi2_threshold=10.0,
        contamination_percentage=20.0,
    )

    fitted_orbit, fitted_orbit_members = fitter.full_od(object_id, observations, propagator)

    assert len(fitted_orbit) >= 1
    assert len(fitted_orbit_members) > 0
