from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core.time import Timestamp
from mpcq import MPCObservations
from mpcq.orbits import MPCOrbits

from ..color_detemination import ColorFit, estimate_colors

DATA_DIR = Path(__file__).parent / "data"

COLOR_FIXTURES: list[str] = sorted(
    p.name for p in DATA_DIR.glob("color_fixture_*.npz")
)
if not COLOR_FIXTURES:
    COLOR_FIXTURES = ["__NO_FIXTURES__"]

# Tolerance: compare within ±0.15 mag of the paper values.
# Greenstreet et al. 2026 reports colors from Fourier fits that include a
# rotational period term.  Our implementation sets g(t)=0 (no rotation), so
# per-band H values can be biased by ~0.1-0.15 mag when multi-band observations
# sample different rotational phases.  Adding rotation fitting later will allow
# this tolerance to be tightened.
COLOR_TOLERANCE = 0.065 # TODO 0.15


def _load_fixture_observations(fx: np.lib.npyio.NpzFile) -> MPCObservations:
    n = int(fx["mag_obs"].shape[0])
    obstime = Timestamp.from_kwargs(
        days=pa.array(fx["obstime_days"].tolist(), type=pa.int64()),
        nanos=pa.array(fx["obstime_nanos"].tolist(), type=pa.int64()),
        scale="utc",
    )
    return MPCObservations.from_kwargs(
        requested_provid=[str(fx["object_id"][0])] * n,
        primary_designation=[None] * n,
        obsid=fx["obsid"].astype(str).tolist(),
        trksub=[None] * n,
        provid=[str(fx["object_id"][0])] * n,
        permid=[None] * n,
        submission_id=[None] * n,
        obssubid=[None] * n,
        obstime=obstime,
        ra=fx["ra"].tolist(),
        dec=fx["dec"].tolist(),
        rmsra=[None] * n,
        rmsdec=[None] * n,
        rmscorr=[None] * n,
        mag=fx["mag_obs"].tolist(),
        rmsmag=fx["rmsmag"].tolist(),
        band=fx["band"].astype(str).tolist(),
        stn=[str(fx["station"][0])] * n,
        updated_at=None,
        created_at=None,
        status=[None] * n,
        astcat=[None] * n,
        mode=[None] * n,
    )


def _load_fixture_orbits(fx: np.lib.npyio.NpzFile) -> MPCOrbits:
    obj_id = str(fx["object_id"][0])
    epoch = Timestamp.from_kwargs(
        days=pa.array([int(fx["epoch_days"][0])], type=pa.int64()),
        nanos=pa.array([int(fx["epoch_nanos"][0])], type=pa.int64()),
        scale="tdb",
    )
    return MPCOrbits.from_kwargs(
        requested_provid=[obj_id],
        primary_designation=[None],
        id=[None],
        provid=[obj_id],
        epoch=epoch,
        q=fx["q"].tolist(),
        e=fx["e"].tolist(),
        i=fx["inc"].tolist(),
        node=fx["node"].tolist(),
        argperi=fx["argperi"].tolist(),
        peri_time=fx["peri_time"].tolist(),
        q_unc=[None],
        e_unc=[None],
        i_unc=[None],
        node_unc=[None],
        argperi_unc=[None],
        peri_time_unc=[None],
        a1=[None],
        a2=[None],
        a3=[None],
        h=fx["H_v_mpc"].tolist(),
        g=fx["G_mpc"].tolist(),
        created_at=None,
        updated_at=None,
    )


@pytest.mark.parametrize("fixture_name", COLOR_FIXTURES)
def test_estimate_colors_from_fixture(fixture_name: str) -> None:
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No color fixtures found on disk.")

    fixture_path = DATA_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Missing fixture {fixture_name}")

    fx = np.load(fixture_path, allow_pickle=True)
    object_id = str(fx["object_id"][0])

    observations = _load_fixture_observations(fx)
    orbits = _load_fixture_orbits(fx)

    result = estimate_colors(observations, orbits, "HG12star")

    assert isinstance(result, ColorFit)
    assert len(result) >= 1

    import pyarrow.compute as pc

    row = result.apply_mask(pc.equal(result.object_id, object_id))
    assert len(row) == 1, f"Expected 1 result row for {object_id}, got {len(row)}"

    # --- abs_mag check ---
    # Compare estimated H_V against the MPC orbit value stored in the fixture.
    # Tolerance is generous (±0.5 mag) because we assume NEO composition for
    # band-to-V conversion and the MPC value was computed with a different pipeline.
    # ABS_MAG_TOLERANCE = 0.2
    # abs_mag = row.abs_mag[0].as_py()
    # assert abs_mag is not None, f"abs_mag is None for {object_id}"
    # abs_mag = float(abs_mag)
    # mpc_h = float(fx["H_v_mpc"][0])
    # print(f"MPC {mpc_h}, computed {abs_mag}")
    # assert np.isfinite(abs_mag), f"abs_mag not finite for {object_id}"
    # assert abs(abs_mag - mpc_h) <= ABS_MAG_TOLERANCE, (
    #     f"{object_id} abs_mag: got {abs_mag:.3f}, MPC H={mpc_h:.3f}, "
    #     f"diff={abs_mag - mpc_h:+.3f} > tol={ABS_MAG_TOLERANCE}"
    # )

    # --- color checks (will pass once colors are implemented) ---
    g_r = row.g_r[0].as_py()
    g_i = row.g_i[0].as_py()
    r_i = row.r_i[0].as_py()

    if g_r is None or g_i is None or r_i is None:
        pytest.xfail("Colors not yet implemented")

    g_r = float(g_r)
    g_i = float(g_i)
    r_i = float(r_i)

    paper_g_r = float(fx["paper_g_r_fourier"][0])
    paper_g_i = float(fx["paper_g_i_fourier"][0])
    paper_r_i = float(fx["paper_r_i_fourier"][0])

    assert np.isfinite(g_r), f"g-r not finite for {object_id}"
    assert np.isfinite(g_i), f"g-i not finite for {object_id}"
    assert np.isfinite(r_i), f"r-i not finite for {object_id}"

    assert abs(g_r - paper_g_r) <= COLOR_TOLERANCE, (
        f"{object_id} g-r: got {g_r:.3f}, paper Fourier {paper_g_r:.3f}, "
        f"diff={g_r - paper_g_r:+.3f} > tol={COLOR_TOLERANCE}"
    )
    assert abs(g_i - paper_g_i) <= COLOR_TOLERANCE, (
        f"{object_id} g-i: got {g_i:.3f}, paper Fourier {paper_g_i:.3f}, "
        f"diff={g_i - paper_g_i:+.3f} > tol={COLOR_TOLERANCE}"
    )
    assert abs(r_i - paper_r_i) <= COLOR_TOLERANCE, (
        f"{object_id} r-i: got {r_i:.3f}, paper Fourier {paper_r_i:.3f}, "
        f"diff={r_i - paper_r_i:+.3f} > tol={COLOR_TOLERANCE}"
    )
