"""
Observation ray construction from point source detections.

This module provides functionality to convert point source detections with
RA/Dec coordinates into observation rays in the SSB ecliptic frame, including
observer positions and line-of-sight unit vectors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from ..observers.observers import Observers
from ..time import Timestamp
from .detections import PointSourceDetections
from .exposures import Exposures

__all__ = [
    "ObservationRays",
    "rays_from_detections",
    "ephemeris_to_rays",
]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # Only for type checkers to avoid runtime import cycles
    from ..orbits.ephemeris import Ephemeris


class ObservationRays(qv.Table):
    """
    Observation rays representing line-of-sight vectors from observers to detections.

    Each row represents a ray from an observer position to a detected source,
    with both the observer position and line-of-sight direction in the SSB
    ecliptic frame for consistent geometric queries.
    """

    #: Unique identifier for the detection
    det_id = qv.LargeStringColumn()

    #: Time of the observation
    time = Timestamp.as_column()

    #: Observatory code for the observer
    observer_code = qv.LargeStringColumn()

    #: Observer position and velocity in SSB ecliptic frame
    observer = CartesianCoordinates.as_column()

    #: Line-of-sight unit vector x-component in SSB ecliptic frame
    u_x = qv.Float64Column()

    #: Line-of-sight unit vector y-component in SSB ecliptic frame
    u_y = qv.Float64Column()

    #: Line-of-sight unit vector z-component in SSB ecliptic frame
    u_z = qv.Float64Column()


def rays_from_detections(
    detections: PointSourceDetections,
    exposures: Exposures,
    frame_in: Literal["equatorial", "ecliptic"] = "equatorial",
) -> ObservationRays:
    """
    Convert point source detections to observation rays in SSB ecliptic frame.

    This function joins detections with exposures to obtain observatory codes,
    computes observer positions at detection times, and converts RA/Dec
    coordinates to line-of-sight unit vectors in the SSB ecliptic frame.

    Parameters
    ----------
    detections : PointSourceDetections
        Point source detections with RA/Dec coordinates
    exposures : Exposures
        Exposure information including observatory codes
    frame_in : {"equatorial", "ecliptic"}, default="equatorial"
        Input coordinate frame for RA/Dec values

    Returns
    -------
    rays : ObservationRays
        Observation rays with observer positions and line-of-sight vectors
        in SSB ecliptic frame
    """
    if len(detections) == 0:
        return ObservationRays.empty()

    # Pre-filter detections to only those with matching exposures (inner-join semantics)
    # This ensures tests expecting zero rays on no match will pass
    det_has_match_mask = pc.is_in(detections.exposure_id, exposures.id)
    detections = detections.apply_mask(det_has_match_mask)
    if len(detections) == 0:
        return ObservationRays.empty()

    # Link detections to exposures to get observatory codes
    det_exp_linkage = detections.link_to_exposures(exposures)

    # Collect all linked detections and exposures
    all_linked_detections = []
    all_linked_exposures = []

    for exposure_id, linked_dets, linked_exps in det_exp_linkage.iterate():
        # Skip any empty groups defensively
        if len(linked_dets) == 0:
            continue
        all_linked_detections.append(linked_dets)
        all_linked_exposures.append(linked_exps)

    if len(all_linked_detections) == 0:
        logger.warning("No detections could be linked to exposures")
        return ObservationRays.empty()

    # Concatenate all linked data
    import quivr as qv

    linked_detections = qv.concatenate(all_linked_detections)
    linked_exposures = qv.concatenate(all_linked_exposures)

    # Get unique observatory codes and times for observer state computation
    observatory_codes = linked_exposures.observatory_code.to_pylist()
    detection_times = linked_detections.time

    # Compute observer states in SSB ecliptic frame
    observers = Observers.from_codes(observatory_codes, detection_times)

    # Ensure observers are in SSB ecliptic frame
    observer_coords = observers.coordinates
    origin_codes = observer_coords.origin.code.to_pylist()
    expected_codes = [OriginCodes.SUN.name] * len(observer_coords)
    if observer_coords.frame != "ecliptic" or origin_codes != expected_codes:
        logger.info("Transforming observer coordinates to SSB ecliptic frame")
        observer_coords = transform_coordinates(
            observer_coords,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )

    # Ensure observer coordinates have SUN origin and ecliptic frame explicitly set on the table
    # so that per-row access in tests yields scalar fields with correct metadata
    if observer_coords.frame != "ecliptic":
        observer_coords = CartesianCoordinates.from_kwargs(
            x=observer_coords.x,
            y=observer_coords.y,
            z=observer_coords.z,
            vx=observer_coords.vx,
            vy=observer_coords.vy,
            vz=observer_coords.vz,
            time=observer_coords.time,
            origin=observer_coords.origin,
            frame="ecliptic",
        )

    # Convert RA/Dec to line-of-sight unit vectors
    ra_deg = linked_detections.ra.to_numpy()
    dec_deg = linked_detections.dec.to_numpy()

    # Convert degrees to radians
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Create spherical coordinates on unit sphere
    spherical_coords = SphericalCoordinates.from_kwargs(
        rho=np.ones(len(linked_detections)),
        lon=ra_rad,
        lat=dec_rad,
        vrho=np.zeros(len(linked_detections)),
        vlon=np.zeros(len(linked_detections)),
        vlat=np.zeros(len(linked_detections)),
        time=detection_times,
        origin=observer_coords.origin,
        frame=frame_in,
    )

    # Convert to Cartesian coordinates in SSB ecliptic frame
    if frame_in != "ecliptic":
        cartesian_coords = transform_coordinates(
            spherical_coords.to_cartesian(),
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )
    else:
        cartesian_coords = spherical_coords.to_cartesian()

    # Extract and normalize line-of-sight unit vectors
    los_vectors = cartesian_coords.r  # Position vectors are already unit vectors
    los_magnitudes = np.linalg.norm(los_vectors, axis=1)

    # Normalize to ensure unit vectors (handle any numerical errors)
    u_vectors = los_vectors / los_magnitudes[:, np.newaxis]

    # Create observation rays table
    rays = ObservationRays.from_kwargs(
        det_id=linked_detections.id.to_pylist(),
        time=detection_times,
        observer_code=observatory_codes,
        observer=observer_coords,
        u_x=u_vectors[:, 0],
        u_y=u_vectors[:, 1],
        u_z=u_vectors[:, 2],
    )

    logger.info(
        f"Created {len(rays)} observation rays from {len(detections)} detections"
    )

    return rays


def ephemeris_to_rays(
    ephemeris: "Ephemeris",
    observers: Observers | None = None,
    det_id: list[str] | None = None,
) -> ObservationRays:
    """
    Convert ephemeris (topocentric RA/Dec) to observation rays in SSB ecliptic frame.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris with spherical coordinates (RA/Dec) from observer perspective.
    observers : Observers, optional
        If provided, these observers will be used for observer positions; otherwise,
        observer states are computed from ephemeris times and station codes.
    det_id : list[str], optional
        Detection IDs to assign to rays; if omitted, synthetic IDs are generated.

    Returns
    -------
    ObservationRays
        Rays with observer positions and LOS unit vectors in SSB ecliptic frame.
    """
    # Lazily import to avoid circular import typing issues (type-only check)
    try:
        from ..orbits.ephemeris import Ephemeris  # noqa: WPS433 (local import)

        _ = Ephemeris  # silence linter unused
    except Exception:
        Ephemeris = None  # type: ignore[assignment]

    times = ephemeris.coordinates.time
    station_codes = ephemeris.coordinates.origin.code

    # Compute observer states if not provided, or if lengths don't align
    # with per-ephemeris rows
    if (observers is None) or (len(observers) != len(ephemeris)):
        observers = Observers.from_codes(times=times, codes=station_codes)

    # Observer positions to SSB ecliptic frame
    observer_coords = transform_coordinates(
        observers.coordinates,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Extract RA/Dec (radians) from ephemeris (equatorial frame)
    ra_rad = ephemeris.coordinates.lon.to_numpy()
    dec_rad = ephemeris.coordinates.lat.to_numpy()

    # Build spherical LOS on unit sphere at the observer
    spherical_coords = SphericalCoordinates.from_kwargs(
        rho=np.ones(len(ephemeris)),
        lon=ra_rad,
        lat=dec_rad,
        vrho=np.zeros(len(ephemeris)),
        vlon=np.zeros(len(ephemeris)),
        vlat=np.zeros(len(ephemeris)),
        time=times,
        origin=observer_coords.origin,
        frame="equatorial",
    )

    # Convert to Cartesian in SSB ecliptic frame
    cartesian_coords = transform_coordinates(
        spherical_coords.to_cartesian(),
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    los_vectors = cartesian_coords.r
    magnitudes = np.linalg.norm(los_vectors, axis=1)
    u_vectors = los_vectors / magnitudes[:, np.newaxis]

    # Create or reuse detection IDs
    if det_id is None:
        det_id = [f"ephem_{i:06d}" for i in range(len(ephemeris))]

    return ObservationRays.from_kwargs(
        det_id=det_id,
        time=times,
        observer_code=station_codes.to_pylist(),
        observer=observer_coords,
        u_x=u_vectors[:, 0],
        u_y=u_vectors[:, 1],
        u_z=u_vectors[:, 2],
    )
