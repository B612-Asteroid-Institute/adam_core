import adam_core.orbits as orbits


def test_declared_orbit_exports_are_importable() -> None:
    assert set(orbits.__all__) == {
        "Ephemeris",
        "Orbits",
        "VariantOrbits",
        "VariantEphemeris",
    }
    for name in orbits.__all__:
        assert getattr(orbits, name).__name__ == name
