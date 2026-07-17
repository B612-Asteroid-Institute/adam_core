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


def test_gauss_iod_custom_mu_is_used_for_geometry_and_velocity():
    """A custom central-body MU must govern one coherent candidate state."""
    import numpy as np

    from adam_core.constants import Constants as c
    from adam_core.orbit_determination.gauss import gaussIOD

    coords = np.array(
        [
            [79.012188047, 26.379000215],
            [79.992351229, 26.449380113],
            [81.886684863, 26.569395368],
        ],
        dtype=np.float64,
    )
    times = np.array(
        [57927.99904508917, 57930.04075542088, 57933.99910001443],
        dtype=np.float64,
    )
    coords_obs = np.array(
        [
            [0.043443721277801516, -1.0155275923223539, 2.9767447559597342e-05],
            [0.077954648460968945, -1.0135650319924312, 3.2256895168063994e-05],
            [0.14453810606720544, -1.0063351512072007, 2.6763644933675388e-05],
        ],
        dtype=np.float64,
    )
    expected_first_velocity = {
        "gibbs": [-0.010949579745566065, 0.003991102395953751, 0.000115987699349817],
        "gauss": [-0.010954243634932293, 0.003992841939072926, 0.000116040146374998],
        "herrick+gibbs": [
            -0.010954381906015255,
            0.003992852818323928,
            0.000116038571487526,
        ],
    }

    for method, expected_velocity in expected_first_velocity.items():
        result = gaussIOD(
            coords,
            times,
            coords_obs,
            velocity_method=method,
            mu=0.5 * c.MU,
        )
        assert len(result) == 3
        assert np.isfinite(result.coordinates.values).all()
        np.testing.assert_allclose(
            result.coordinates.values[0, 3:],
            expected_velocity,
            rtol=0,
            atol=5e-13,
        )
