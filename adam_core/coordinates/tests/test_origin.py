import numpy as np
import pytest

from ..origin import Origin


def test_origin_eq__():
    # Test equality with string
    origin = Origin.from_kwargs(code=["SUN", "SUN", "SUN"])
    assert origin == "SUN"
    origin = Origin.from_kwargs(code=["SUN", "EARTH", "MARS"])
    assert origin != "ecliptic"

    # Test equality with numpy array
    assert origin == np.array(["SUN", "EARTH", "MARS"])
    assert origin != np.array(["SUN", "SUN", "MARS"])

    # Test equality with Origin
    assert origin == Origin.from_kwargs(code=["SUN", "EARTH", "MARS"])
    assert origin != Origin.from_kwargs(code=["SUN", "SUN", "MARS"])


def test_origin_eq__raises():
    # Test that an error is raised when an unsupported type is passed
    origin = Origin.from_kwargs(code=["SUN", "EARTH", "MARS"])
    with pytest.raises(TypeError):
        origin == 1
