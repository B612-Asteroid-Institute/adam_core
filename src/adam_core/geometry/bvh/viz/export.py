"""
Export helpers to persist single-orbit BVH view data to disk for a static viewer.

This module does not build any geometry. It consumes existing `BVHIndex`, an
`orbit_id`, and optional `ObservationRays`, filters to the requested orbit,
prepares compact arrays, and saves them as compressed .npz plus a small .json
metadata file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ...rays import ObservationRays
from .. import BVHIndex, OverlapHits
from ..viz import ViewData, prepare_view_data_single_orbit


def save_view_data_npz(
    view: ViewData,
    base_path: str | Path,
    *,
    extra_metadata: dict | None = None,
) -> None:
    """
    Save viewer arrays to compressed NPZ alongside a JSON metadata file.

    Parameters
    ----------
    view : ViewData
        Arrays prepared by `prepare_view_data_single_orbit`.
    base_path : str | Path
        Output path without extension; `.npz` and `.json` will be created.
    extra_metadata : dict, optional
        Additional metadata to merge into the JSON file.
    """
    p = Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Persist arrays
    np.savez_compressed(
        p.with_suffix(".npz"),
        segments_endpoints_f32=view["segments_endpoints_f32"],
        nodes_min_f32=view["nodes_min_f32"],
        nodes_max_f32=view["nodes_max_f32"],
        node_depth_i32=view["node_depth_i32"],
        node_indices_i32=view["node_indices_i32"],
        rays_origins_f32=view["rays_origins_f32"],
        rays_dirs_f32=view["rays_dirs_f32"],
        bounds_sphere_f32=view["bounds_sphere_f32"],
        # Optional fields if present
        rays_station_codes=np.array(view.get("rays_station_codes", []), dtype=object),
        rays_det_ids=np.array(view.get("rays_det_ids", []), dtype=object),
        rays_hit_mask=view.get("rays_hit_mask", np.zeros((0,), dtype=bool)),
    )

    # Compose metadata
    meta = {
        "type": "SingleOrbitViewData",
        "counts": {
            "segments": int(view["segments_endpoints_f32"].shape[0]),
            "nodes": int(view["nodes_min_f32"].shape[0]),
            "rays": int(view["rays_origins_f32"].shape[0]),
        },
        "shapes": {
            k: list(view[k].shape)
            for k in [
                "segments_endpoints_f32",
                "nodes_min_f32",
                "nodes_max_f32",
                "node_depth_i32",
                "node_indices_i32",
                "rays_origins_f32",
                "rays_dirs_f32",
                "bounds_sphere_f32",
            ]
        },
        "dtypes": {
            k: str(view[k].dtype)
            for k in [
                "segments_endpoints_f32",
                "nodes_min_f32",
                "nodes_max_f32",
                "node_depth_i32",
                "node_indices_i32",
                "rays_origins_f32",
                "rays_dirs_f32",
                "bounds_sphere_f32",
            ]
        },
    }
    if extra_metadata:
        meta.update(extra_metadata)

    with open(p.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def export_single_orbit_npz(
    index: BVHIndex,
    orbit_id: str,
    out_base_path: Union[str, Path],
    *,
    rays: Optional[ObservationRays] = None,
    max_rays: Optional[int] = 1000,
    tight_aabbs: bool = True,
    hits: Optional[OverlapHits] = None,
) -> None:
    """
    Prepare and save a single-orbit viewer dataset to `<out_base_path>.npz/.json`.

    Parameters
    ----------
    index : BVHIndex
        Existing BVH index over multiple orbits.
    orbit_id : str
        Requested orbit identifier to visualize.
    out_base_path : str | Path
        Output path without extension; `.npz` and `.json` will be created.
    rays : ObservationRays, optional
        Observation rays to include (capped by `max_rays`).
    max_rays : int, optional (default 1000)
        Maximum number of rays to include; set None to include all.
    tight_aabbs : bool, optional (default True)
        When True, recompute node bounds over the selected orbit's primitives.
    """
    view = prepare_view_data_single_orbit(
        index,
        orbit_id,
        rays=rays,
        max_rays=max_rays,
        tight_aabbs=tight_aabbs,
        hits=hits,
    )

    save_view_data_npz(
        view,
        out_base_path,
        extra_metadata={
            "orbit_id": orbit_id,
            "tight_aabbs": bool(tight_aabbs),
            "max_rays": None if max_rays is None else int(max_rays),
        },
    )


def save_view_data_json(view: ViewData, json_path: Union[str, Path]) -> None:
    """
    Save viewer arrays to a single JSON file (human-readable, largest size).

    This is intended for small, single-orbit datasets to enable zero-dependency
    in-browser loading. For larger data, prefer NPZ.
    """
    p = Path(json_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "segments_endpoints_f32": view["segments_endpoints_f32"].tolist(),
        "nodes_min_f32": view["nodes_min_f32"].tolist(),
        "nodes_max_f32": view["nodes_max_f32"].tolist(),
        "node_depth_i32": view["node_depth_i32"].tolist(),
        "node_indices_i32": view["node_indices_i32"].tolist(),
        "rays_origins_f32": view["rays_origins_f32"].tolist(),
        "rays_dirs_f32": view["rays_dirs_f32"].tolist(),
        "bounds_sphere_f32": view["bounds_sphere_f32"].tolist(),
    }
    # Optional ray metadata
    if "rays_station_codes" in view:
        serializable["rays_station_codes"] = view["rays_station_codes"]
    if "rays_det_ids" in view:
        serializable["rays_det_ids"] = view["rays_det_ids"]
    if "rays_hit_mask" in view:
        serializable["rays_hit_mask"] = view["rays_hit_mask"].astype(bool).tolist()

    with open(p, "w") as f:
        json.dump(serializable, f)


def export_single_orbit_json(
    index: BVHIndex,
    orbit_id: str,
    json_path: Union[str, Path],
    *,
    rays: Optional[ObservationRays] = None,
    max_rays: Optional[int] = 1000,
    tight_aabbs: bool = True,
    hits: Optional[OverlapHits] = None,
) -> None:
    """
    Prepare and save a single-orbit viewer dataset to a single JSON file.
    """
    view = prepare_view_data_single_orbit(
        index,
        orbit_id,
        rays=rays,
        max_rays=max_rays,
        tight_aabbs=tight_aabbs,
        hits=hits,
    )
    save_view_data_json(view, json_path)
