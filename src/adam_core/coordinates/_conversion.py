"""Shared one-crossing representation conversion for coordinate tables.

Replaces the legacy two-hop Python orchestration (rep_in -> Cartesian ->
rep_out, two Rust crossings plus intermediate table assembly) with a single
Rust crossing per public conversion. Covariance-bearing rows use the forward
autodiff transform; covariance-free rows use the values-only composed
transform. Legacy error messages and column contracts are preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import quivr as qv

_COMETARY_TIME_ERROR_SOURCE = (
    "To convert Cometary coordinates to Cartesian coordinates, the times\n"
    "at which the Coordinates coordinates are defined is required to give\n"
    "the time of periapsis passage context."
)
_COMETARY_TIME_ERROR_TARGET = (
    "To convert Cometary coordinates to Cartesian coordinates, the times\n"
    "at which the Cartesian coordinates are defined is required to calculate\n"
    "the time of periapsis passage."
)

_OUTPUT_COLUMNS = {
    "keplerian": ("a", "e", "i", "raan", "ap", "M"),
    "cometary": ("q", "e", "i", "raan", "ap", "tp"),
    "spherical": ("rho", "lon", "lat", "vrho", "vlon", "vlat"),
}
# The values-only Keplerian output is the 13-column extended layout; the
# autodiff covariance output is always 6 columns.
_RAW_COLUMN_INDEX = {
    "keplerian": (0, 4, 5, 6, 7, 8),
    "cometary": (0, 1, 2, 3, 4, 5),
    "spherical": (0, 1, 2, 3, 4, 5),
}


def convert_representation(
    source: "qv.Table",
    representation_in: str,
    representation_out: str,
    target_cls: type,
) -> "qv.Table":
    """Convert ``source`` to ``target_cls`` in one Rust crossing."""
    from adam_core import _rust_native

    from .covariances import CoordinateCovariances, rust_covariance_transform

    if representation_in == "cometary" and source.time is None:
        raise ValueError(_COMETARY_TIME_ERROR_SOURCE)
    if representation_out == "cometary" and source.time is None:
        raise ValueError(_COMETARY_TIME_ERROR_TARGET)

    needs_mu = "keplerian" in (representation_in, representation_out) or "cometary" in (
        representation_in,
        representation_out,
    )
    needs_t0 = representation_in == "cometary" or representation_out in (
        "keplerian",
        "cometary",
    )
    mu = (
        np.ascontiguousarray(np.asarray(source.origin.mu(), dtype=np.float64))
        if needs_mu
        else None
    )
    t0 = (
        np.ascontiguousarray(np.asarray(source.time.to_numpy(), dtype=np.float64))
        if needs_t0
        else None
    )

    values = source.values
    if not source.covariance.is_all_nan():
        coords_out, covariances_out = rust_covariance_transform(
            values,
            source.covariance.to_matrix(),
            representation_in,
            representation_out,
            t0=t0,
            mu=mu,
            frame_in=source.frame,
            frame_out=source.frame,
        )
        columns = (0, 1, 2, 3, 4, 5)  # the autodiff transform always emits 6 columns
    else:
        # Legacy anomaly-solver iteration budget: the Keplerian input leg used
        # max_iter=1000; the Cometary input leg used max_iter=100.
        max_iter = 1000 if representation_in == "keplerian" else 100
        coords_out = np.asarray(
            _rust_native.transform_coordinates_numpy(
                np.ascontiguousarray(values, dtype=np.float64),
                representation_in,
                representation_out,
                t0=t0,
                mu=mu,
                max_iter=max_iter,
                tol=1e-15,
                frame_in=source.frame,
                frame_out=source.frame,
            ),
            dtype=np.float64,
        )
        covariances_out = np.full((len(coords_out), 6, 6), np.nan)
        columns = _RAW_COLUMN_INDEX[representation_out]

    names = _OUTPUT_COLUMNS[representation_out]
    kwargs = {name: coords_out[:, index] for name, index in zip(names, columns)}
    return target_cls.from_kwargs(
        time=source.time,
        covariance=CoordinateCovariances.from_matrix(covariances_out),
        origin=source.origin,
        frame=source.frame,
        **kwargs,
    )
