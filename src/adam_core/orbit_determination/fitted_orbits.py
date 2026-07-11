import uuid
from typing import List, Literal, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.residuals import Residuals
from ..orbits.orbits import Orbits


def assign_duplicate_observations(
    orbits: "FittedOrbits", orbit_members: "FittedOrbitMembers"
) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
    """
    Assigns observations that have been assigned to multiple orbits to the orbit with the
    most observations, longest arc length, and lowest reduced chi2.

    Parameters
    ----------
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.

    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits with duplicate assignments removed.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members with duplicate assignments removed.
    """
    import numpy as np

    from adam_core import _rust_native

    # One Rust crossing owns the priority ordering, per-observation best-orbit
    # selection, member filtering, and surviving-orbit selection.
    orbit_take, member_keep = _rust_native.assign_duplicate_observations_numpy(
        orbits.orbit_id.to_pylist(),
        np.ascontiguousarray(
            orbits.num_obs.to_numpy(zero_copy_only=False), dtype=np.int64
        ),
        np.ascontiguousarray(
            orbits.arc_length.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        np.ascontiguousarray(
            orbits.reduced_chi2.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        orbit_members.orbit_id.to_pylist(),
        orbit_members.obs_id.to_pylist(),
    )

    filtered_orbits = orbits.take(pa.array(orbit_take, type=pa.int64()))
    orbit_members = orbit_members.apply_mask(pa.array(member_keep))

    return filtered_orbits, orbit_members


def drop_duplicate_orbits(
    orbits: "FittedOrbits",
    orbit_members: "FittedOrbitMembers",
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
    """
    Drop duplicate orbits from the fitted orbits and remove
    the corresponding orbit members.

    Parameters
    ----------
    orbits : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits.
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.
    subset : list of str, optional
        Subset of columns to consider when dropping duplicates. If not specified all the columns
        specifying unique state are used: time, x, y, z, vx, vy, vz.
    keep : {'first', 'last'}, default 'first'
        If there are duplicate rows then keep the first or last row.

    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits without duplicates.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members without duplicates.
    """
    if subset is None:
        subset = [
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.x",
            "coordinates.y",
            "coordinates.z",
            "coordinates.vx",
            "coordinates.vy",
            "coordinates.vz",
        ]

    filtered = orbits.drop_duplicates(subset=subset, keep=keep)
    filtered_orbit_members = orbit_members.apply_mask(
        pc.is_in(orbit_members.orbit_id, filtered.orbit_id)
    )
    return filtered, filtered_orbit_members


class FittedOrbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    chi2 = qv.Float64Column()
    reduced_chi2 = qv.Float64Column()
    iterations = qv.Int64Column(nullable=True)
    success = qv.BooleanColumn(nullable=True)
    status_code = qv.Int64Column(nullable=True)

    def to_orbits(self) -> Orbits:
        """
        Convert fitted orbits to orbits that can be used by
        a Propagator.

        Returns
        -------
        orbits : `~adam_core.orbits.Orbits`
            Orbits.
        """
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )


class FittedOrbitMembers(qv.Table):

    orbit_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
    residuals = Residuals.as_column(nullable=True)
    solution = qv.BooleanColumn(nullable=True)
    outlier = qv.BooleanColumn(nullable=True)
