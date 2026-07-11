def test_gauss_iod_returns_rust_assembled_orbits(monkeypatch):
    """gaussIOD must wrap the Rust Orbits RecordBatch without from_kwargs."""
    import numpy as np

    from adam_core.orbit_determination import gauss as gauss_module
    from adam_core.orbits import Orbits

    coords = np.array(
        [
            [57.8492351997515, -0.6765348320549983],
            [57.791953931637394, -0.9959866403096104],
            [57.73672992311044, -1.3053527989774307],
        ],
        dtype=np.float64,
    )
    times = np.array([59000.0, 59004.0, 59008.0], dtype=np.float64)
    coords_obs = np.array(
        [
            [0.88, -0.45, 0.0],
            [0.91, -0.40, 0.0],
            [0.93, -0.35, 0.0],
        ],
        dtype=np.float64,
    )

    def _forbid_from_kwargs(*args, **kwargs):
        raise AssertionError("Orbits.from_kwargs must not rebuild Rust output")

    monkeypatch.setattr(Orbits, "from_kwargs", _forbid_from_kwargs)
    result = gauss_module.gaussIOD(coords, times, coords_obs)
    assert isinstance(result, Orbits)
    if len(result) > 0:
        assert result.coordinates.frame == "ecliptic"
        assert result.coordinates.time.scale == "utc"
        assert set(result.coordinates.origin.code.to_pylist()) == {"SUN"}
        orbit_ids = result.orbit_id.to_pylist()
        assert len(set(orbit_ids)) == len(orbit_ids)
        assert all(len(orbit_id) == 32 for orbit_id in orbit_ids)
