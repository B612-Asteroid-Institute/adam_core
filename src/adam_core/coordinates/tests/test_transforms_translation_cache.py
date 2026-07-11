import numpy as np

from ...observers.state import clear_observer_state_cache
from ...time import Timestamp
from ...utils import spice as spice_mod
from ..cartesian import CartesianCoordinates
from ..origin import Origin, OriginCodes
from ..transform import clear_translation_cache


def test_cartesian_to_origin_translation_cache_avoids_recompute(monkeypatch):
    clear_translation_cache()
    spice_mod.clear_spkez_cache()

    # Force translation cache enabled (in case environment disables it).
    # This is a module-level flag read at import time; adjust directly.
    from .. import transform as transform_mod

    monkeypatch.setattr(transform_mod, "_TRANSLATION_CACHE_ENABLED", True)

    calls = {"n": 0}
    orig = spice_mod.get_perturber_state

    def _counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(spice_mod, "get_perturber_state", _counted)

    # cartesian_to_origin now resolves origin translation in Rust (native
    # single-crossing path, served by the Rust spkez cache); the Python
    # translation cache under test lives on the retained legacy fallback
    # resolver, so exercise that path directly.
    t = Timestamp.from_mjd(np.array([60000.0, 60000.5, 60001.0, 60001.5]), scale="tdb")
    coords = CartesianCoordinates.from_kwargs(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        y=np.array([0.0, 0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0]),
        vx=np.array([0.0, 0.0, 0.0, 0.0]),
        vy=np.array([0.0, 0.0, 0.0, 0.0]),
        vz=np.array([0.0, 0.0, 0.0, 0.0]),
        time=t,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * 4),
    )

    vectors_a = transform_mod._resolve_origin_translation_vectors(
        coords, OriginCodes.SOLAR_SYSTEM_BARYCENTER
    )
    a = coords.translate(vectors_a, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name)
    n1 = int(calls["n"])
    assert n1 == 1

    vectors_b = transform_mod._resolve_origin_translation_vectors(
        coords, OriginCodes.SOLAR_SYSTEM_BARYCENTER
    )
    b = coords.translate(vectors_b, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name)
    assert int(calls["n"]) == n1
    np.testing.assert_allclose(a.values, b.values, rtol=0.0, atol=0.0)


def test_mpc_observer_state_cache_avoids_recompute(monkeypatch):
    # Observers.from_codes is served by the Rust backend since bead
    # personal-cmy.6; the Python pxform cache under test here lives on the
    # retained legacy fallback path (_from_codes_legacy, used for codes the
    # Rust ground-site table cannot serve), so exercise that path directly.
    clear_observer_state_cache()
    spice_mod.clear_spkez_cache()

    import adam_core.observers.state as state_mod

    # The pxform cache under test lives on the retained legacy composition;
    # force it so the counted batch query below is actually exercised.
    def _no_rust(*args, **kwargs):
        raise RuntimeError("forced legacy path for cache test")

    monkeypatch.setattr(state_mod, "_rust_backend_observer_states", _no_rust)

    calls = {"n": 0}
    batch_orig = state_mod._query_pxform_itrf93_batch

    def _batch_counted(*args, **kwargs):
        calls["n"] += 1
        return batch_orig(*args, **kwargs)

    monkeypatch.setattr(state_mod, "_query_pxform_itrf93_batch", _batch_counted)

    t = Timestamp.from_mjd(np.array([60000.0, 60000.5, 60001.0]), scale="tdb")
    import pyarrow as pa

    from adam_core.observers import Observers

    codes = pa.array(["X05"] * len(t))
    _ = Observers._from_codes_legacy(codes, t)
    n1 = int(calls["n"])
    assert n1 > 0
    _ = Observers._from_codes_legacy(codes, t)
    assert int(calls["n"]) == n1


def test_cartesian_to_origin_translation_cache_key_is_order_sensitive(monkeypatch):
    clear_translation_cache()
    spice_mod.clear_spkez_cache()

    from .. import transform as transform_mod

    monkeypatch.setattr(transform_mod, "_TRANSLATION_CACHE_ENABLED", True)

    calls = {"n": 0}

    def _fake_get_perturber_state(
        perturber, times, frame="ecliptic", origin=OriginCodes.SUN
    ):
        del perturber
        calls["n"] += 1
        key = times.key(scale="tdb").astype(np.float64)
        zeros = np.zeros(len(times), dtype=np.float64)
        return CartesianCoordinates.from_kwargs(
            x=key,
            y=zeros,
            z=zeros,
            vx=zeros,
            vy=zeros,
            vz=zeros,
            time=times,
            frame=frame,
            origin=Origin.from_kwargs(code=[origin.name] * len(times)),
        )

    monkeypatch.setattr(spice_mod, "get_perturber_state", _fake_get_perturber_state)

    t_a = Timestamp.from_mjd(
        np.array([60000.0, 60000.25, 60000.5, 60000.75]), scale="tdb"
    )
    t_b = Timestamp.from_mjd(
        np.array([60000.0, 60000.5, 60000.25, 60000.75]), scale="tdb"
    )
    assert t_a.signature(scale="tdb") == t_b.signature(scale="tdb")

    coords_a = CartesianCoordinates.from_kwargs(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        y=np.zeros(4),
        z=np.zeros(4),
        vx=np.zeros(4),
        vy=np.zeros(4),
        vz=np.zeros(4),
        time=t_a,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * 4),
    )
    coords_b = CartesianCoordinates.from_kwargs(
        x=np.array([10.0, 20.0, 30.0, 40.0]),
        y=np.zeros(4),
        z=np.zeros(4),
        vx=np.zeros(4),
        vy=np.zeros(4),
        vz=np.zeros(4),
        time=t_b,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * 4),
    )

    from .. import transform as transform_mod2

    vectors_a = transform_mod2._resolve_origin_translation_vectors(
        coords_a, OriginCodes.SOLAR_SYSTEM_BARYCENTER
    )
    _ = coords_a.translate(vectors_a, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name)
    vectors_b = transform_mod2._resolve_origin_translation_vectors(
        coords_b, OriginCodes.SOLAR_SYSTEM_BARYCENTER
    )
    out_b = coords_b.translate(vectors_b, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name)

    # A distinct order must miss the cache and recompute translations.
    assert int(calls["n"]) == 2
    expected_x_b = np.array([10.0, 20.0, 30.0, 40.0]) + t_b.key(scale="tdb").astype(
        np.float64
    )
    np.testing.assert_allclose(out_b.x.to_numpy(zero_copy_only=False), expected_x_b)
