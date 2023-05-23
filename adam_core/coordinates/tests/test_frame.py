import numpy as np
import pytest

from ..frame import Frame


def test_frame__eq__():
    # Test equality with string
    frame = Frame.from_kwargs(name=["ecliptic", "ecliptic", "ecliptic"])
    assert frame == "ecliptic"
    frame = Frame.from_kwargs(name=["ecliptic", "ecliptic", "equatorial"])
    assert frame != "ecliptic"

    # Test equality with numpy array
    assert frame == np.array(["ecliptic", "ecliptic", "equatorial"])
    assert frame != np.array(["ecliptic", "ecliptic", "ecliptic"])

    # Test equality with Frame
    assert frame == Frame.from_kwargs(name=["ecliptic", "ecliptic", "equatorial"])
    assert frame != Frame.from_kwargs(name=["ecliptic", "ecliptic", "ecliptic"])


def test_frame__eq__raises():
    # Test that an error is raised when an unsupported type is passed
    frame = Frame.from_kwargs(name=["ecliptic", "ecliptic", "ecliptic"])
    with pytest.raises(NotImplementedError):
        frame == 1
