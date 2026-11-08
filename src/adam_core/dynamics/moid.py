"""Global minimum-orbit-intersection distance (MOID).

The numerical kernel is a safe pure-Rust port of the error-controlled
critical-point method of Baluev & Mikryukov (2019), *Astronomy and Computing*
27, 11-22, derived from the MIT-licensed distlink 2.0 reference implementation.
The Python layer provides thin scalar and SPICE-backed adapters around the
shared Rust kernels.
"""

import numpy as np
import pyarrow as pa
import quivr as qv

from ..coordinates import Origin, OriginCodes
from ..orbits import Orbits
from ..time import Timestamp


class PerturberMOIDs(qv.Table):
    orbit_id = qv.LargeStringColumn()
    perturber = Origin.as_column()
    moid = qv.Float64Column()
    time = Timestamp.as_column()


def calculate_moid(
    primary_ellipse: Orbits, secondary_ellipse: Orbits
) -> tuple[float, Timestamp]:
    """
    Calculate the Minimum Orbit Intersection Distance (MOID) between two orbits.

    Pure-Rust global critical-point kernel with a deterministic primary-orbit
    witness epoch. This bounded API accepts elliptic conics;
    hyperbolic, parabolic, non-finite, and degenerate inputs raise ``ValueError``.
    """
    from .._rust.api import calculate_moid_numpy

    mu = float(primary_ellipse.coordinates.origin.mu()[0])
    primary_state = np.ascontiguousarray(
        primary_ellipse.coordinates.values[0], dtype=np.float64
    )
    secondary_state = np.ascontiguousarray(
        secondary_ellipse.coordinates.values[0], dtype=np.float64
    )

    result = calculate_moid_numpy(primary_state, secondary_state, mu)
    moid, dt_min = result
    moid_time = Timestamp.from_mjd(
        [primary_ellipse.coordinates.time.mjd()[0].as_py() + dt_min],
        scale=primary_ellipse.coordinates.time.scale,
    )
    return moid, moid_time


def calculate_perturber_moids(
    orbits: Orbits,
    perturber: OriginCodes | list[OriginCodes],
) -> PerturberMOIDs:
    """
    Calculate the minimum orbit intersection distance (MOID) for all orbits with respect
    to the perturbing body or bodies.

    One Arrow crossing delegates SPICE state lookup, perturber orchestration,
    and the internally parallel batch calculation to shared Rust paths.

    Parameters
    ----------
    orbits : Orbits
        The orbits to calculate the MOID for.
    perturber : OriginCodes or List[OriginCodes]
        The perturbing body or bodies to calculate the MOID for.
    Returns
    -------
    PerturberMOIDs
        A table containing the MOID and time for each orbit with respect to
        the perturbing body or bodies.

    Raises
    ------
    ValueError
        If the Rust SPICE backend has not been initialized, any input is not a
        supported finite elliptic conic, or the solver's reliability checks
        fail.
    """
    from .._rust.api import calculate_perturber_moids_arrow
    from ..orbits.arrow_bridge import orbits_to_record_batch

    assert len(orbits.orbit_id.unique()) == len(orbits)
    assert len(orbits.coordinates.origin.code.unique()) == 1

    if isinstance(perturber, OriginCodes):
        perturbers = [perturber]
    else:
        perturbers = perturber

    if len(orbits) == 0:
        return PerturberMOIDs.empty()

    # Single Arrow crossing: Rust decodes the Orbits RecordBatch, runs the
    # perturber loop + per-perturber spkez + batched MOID kernel, and returns
    # the finished perturber-major PerturberMOIDs RecordBatch.
    result = calculate_perturber_moids_arrow(
        orbits_to_record_batch(orbits),
        [perturber_i.name for perturber_i in perturbers],
    )
    table = pa.Table.from_batches([result]).replace_schema_metadata(
        {b"time.scale": orbits.coordinates.time.scale.encode()}
    )
    return PerturberMOIDs.from_pyarrow(table)
