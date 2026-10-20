import pytest

from ..orbit_fitter import OrbitFitter


class _InitialFitOnlyFitter(OrbitFitter):
    """A fitter written against the original ``initial_fit``-only interface,
    e.g. a plugin release predating ``OrbitFitter.refine_fit``."""

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def initial_fit(self, object_id, observations, reference_orbit=None):
        raise AssertionError("not exercised")


def test_initial_fit_only_fitter_stays_instantiable():
    """refine_fit has a NotImplementedError default rather than being abstract,
    so fitters implementing only initial_fit still construct."""
    fitter = _InitialFitOnlyFitter()

    with pytest.raises(NotImplementedError, match="_InitialFitOnlyFitter does not"):
        fitter.refine_fit(None, None, None)


def test_orbit_fitter_still_requires_initial_fit():
    class _NoInitialFit(OrbitFitter):
        pass

    with pytest.raises(TypeError, match="initial_fit"):
        _NoInitialFit()
