"""
Generate color determination test fixtures.

Run from the repo root:
    pdm run python src/adam_core/photometry/tests/data/generate_color_fixtures.py

Requires a BigQuery MPC replica connection (set MPCQ_PROJECT_ID).  The generated
.npz files ARE committed, so this script only needs to be re-run when adding or
refreshing fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from mpcq import MPCObservations
from mpcq.client import BigQueryMPCClient
from mpcq.orbits import MPCOrbits

from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers.observers import Observers
from adam_core.photometry.bandpasses.api import map_to_canonical_filter_bands

OUT_DIR = Path(__file__).parent

# Paper values from Greenstreet et al. 2026 (ApJL 996 L33) Table 2, for the
# subset of objects with a good Fourier/LSM color match.
PAPER_COLORS = {
    "2025 MF76": {
        "g_r_fourier": 0.58,
        "g_i_fourier": 0.75,
        "r_i_fourier": 0.17,
        "g_r_lsm": 0.56,
        "g_i_lsm": 0.75,
        "r_i_lsm": 0.19,
    },
    "2025 MN25": {
        "g_r_fourier": 0.41,
        "g_i_fourier": 0.54,
        "r_i_fourier": 0.13,
        "g_r_lsm": 0.42,
        "g_i_lsm": 0.51,
        "r_i_lsm": 0.08,
    },
    "2025 MO35": {
        "g_r_fourier": 0.56,
        "g_i_fourier": 0.71,
        "r_i_fourier": 0.15,
        "g_r_lsm": 0.57,
        "g_i_lsm": 0.71,
        "r_i_lsm": 0.14,
    },
    "2025 MS34": {
        "g_r_fourier": 0.58,
        "g_i_fourier": 0.75,
        "r_i_fourier": 0.17,
        "g_r_lsm": 0.59,
        "g_i_lsm": 0.74,
        "r_i_lsm": 0.15,
    },
    "2025 MU8": {
        "g_r_fourier": 0.46,
        "g_i_fourier": 0.58,
        "r_i_fourier": 0.12,
        "g_r_lsm": 0.47,
        "g_i_lsm": 0.58,
        "r_i_lsm": 0.12,
    },
    "2025 MV71": {
        "g_r_fourier": 0.43,
        "g_i_fourier": 0.57,
        "r_i_fourier": 0.13,
        "g_r_lsm": 0.44,
        "g_i_lsm": 0.57,
        "r_i_lsm": 0.13,
    },
}


def build_fixture(
    obj_id: str,
    mpc_obs: MPCObservations,
    mpc_orb: MPCOrbits,
    out_path: Path,
) -> None:
    mask_obs = pc.equal(mpc_obs.provid, obj_id)
    obs = mpc_obs.apply_mask(mask_obs)
    mask_orb = pc.equal(mpc_orb.provid, obj_id)
    orb = mpc_orb.apply_mask(mask_orb)
    if len(orb) == 0:
        raise ValueError(f"No orbit found for {obj_id}")

    times = obs.obstime
    stns = obs.stn.to_numpy(zero_copy_only=False).astype(str)
    bands_raw = obs.band.to_numpy(zero_copy_only=False).astype(str)

    orbits = orb.orbits()
    prop = propagate_2body(orbits, times)
    observers = Observers.from_codes(stns, times)
    filter_ids = map_to_canonical_filter_bands(stns, bands_raw, on_unknown="skip")

    obj_pos = prop.coordinates.r
    obs_pos = observers.coordinates.r

    paper = PAPER_COLORS[obj_id]

    orb0 = orb[0]
    epoch_days = int(orb0.epoch.days[0].as_py())
    epoch_nanos = int(orb0.epoch.nanos[0].as_py())

    np.savez_compressed(
        out_path,
        object_id=np.array([obj_id], dtype=object),
        station=stns,
        # Orbit parameters
        H_v_mpc=np.array([float(orb0.h[0].as_py())], dtype=np.float64),
        G_mpc=np.array([float(orb0.g[0].as_py())], dtype=np.float64),
        epoch_days=np.array([epoch_days], dtype=np.int64),
        epoch_nanos=np.array([epoch_nanos], dtype=np.int64),
        q=np.array([float(orb0.q[0].as_py())], dtype=np.float64),
        e=np.array([float(orb0.e[0].as_py())], dtype=np.float64),
        inc=np.array([float(orb0.i[0].as_py())], dtype=np.float64),
        node=np.array([float(orb0.node[0].as_py())], dtype=np.float64),
        argperi=np.array([float(orb0.argperi[0].as_py())], dtype=np.float64),
        peri_time=np.array([float(orb0.peri_time[0].as_py())], dtype=np.float64),
        # Per-observation data
        obsid=np.array(obs.obsid.to_pylist(), dtype=object),
        obstime_days=np.array(times.days.to_pylist(), dtype=np.int64),
        obstime_nanos=np.array(times.nanos.to_pylist(), dtype=np.int64),
        band=bands_raw,
        filter_id=np.array(
            [f if f is not None else "" for f in filter_ids.tolist()], dtype=object
        ),
        mag_obs=np.asarray(obs.mag.to_numpy(zero_copy_only=False), dtype=np.float64),
        rmsmag=np.asarray(obs.rmsmag.to_numpy(zero_copy_only=False), dtype=np.float64),
        ra=np.asarray(obs.ra.to_numpy(zero_copy_only=False), dtype=np.float64),
        dec=np.asarray(obs.dec.to_numpy(zero_copy_only=False), dtype=np.float64),
        object_pos=np.asarray(obj_pos, dtype=np.float64),
        observer_pos=np.asarray(obs_pos, dtype=np.float64),
        # Paper reference colors
        paper_g_r_fourier=np.array([paper["g_r_fourier"]], dtype=np.float64),
        paper_g_i_fourier=np.array([paper["g_i_fourier"]], dtype=np.float64),
        paper_r_i_fourier=np.array([paper["r_i_fourier"]], dtype=np.float64),
        paper_g_r_lsm=np.array([paper["g_r_lsm"]], dtype=np.float64),
        paper_g_i_lsm=np.array([paper["g_i_lsm"]], dtype=np.float64),
        paper_r_i_lsm=np.array([paper["r_i_lsm"]], dtype=np.float64),
    )
    print(
        f"Wrote {out_path.name}  "
        f"(n={len(obs)}, bands={dict(zip(*np.unique(bands_raw, return_counts=True)))})"
    )


def _query_mpc_data(object_ids: list[str]) -> tuple[MPCObservations, MPCOrbits]:
    client = BigQueryMPCClient(
        dataset_id="asteroid_institute_mpc_replica",
        views_dataset_id="asteroid_institute___mpc_replica_views",
        project=os.environ["MPCQ_PROJECT_ID"],
    )
    observations = client.query_observations(object_ids)
    orbits = client.query_orbits(object_ids)
    return observations, orbits


def main() -> None:
    object_ids = list(PAPER_COLORS)
    mpc_obs, mpc_orb = _query_mpc_data(object_ids)

    for obj_id in object_ids:
        slug = obj_id.replace(" ", "_")
        out_path = OUT_DIR / f"color_fixture_{slug}.npz"
        build_fixture(obj_id, mpc_obs, mpc_orb, out_path)


if __name__ == "__main__":
    main()
