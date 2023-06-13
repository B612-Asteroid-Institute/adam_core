import pytest

from ..orbits import make_real_orbits, make_simple_orbits


def test_make_real_orbits():
    # Test that all sample orbits are read from the CSV file
    orbits = make_real_orbits()
    assert len(orbits) == 28

    # Test that the number of orbits returned is correct
    orbits = make_real_orbits(num_orbits=10)
    assert len(orbits) == 10


def test_make_simple_orbits_raises():
    # Test that a ValueError is raised if num_orbits > 29
    with pytest.raises(ValueError):
        make_real_orbits(num_orbits=29)


def test_make_simple_orbits():
    # Test that the number of orbits returned is correct
    orbits = make_simple_orbits(num_orbits=10)
    assert len(orbits) == 10
