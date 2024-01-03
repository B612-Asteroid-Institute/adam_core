import numpy as np
import pyarrow.compute as pc

from .differential_correction import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers


def remove_lowest_probability_observation(
    orbit_members: FittedOrbitMembers, observations: OrbitDeterminationObservations
) -> OrbitDeterminationObservations:
    """
    Remove the observation with the worst residual from the observations. The probability is defined to be
    the probability of drawing a more extreme residual than the one observed. If multiple observations
    have the same probability, then the observation with the highest squared residual value is removed.

    Parameters
    ----------
    orbit_members : FittedOrbitMembers
        The orbit members that contain the residuals with respect to the observations.
    observations : OrbitDeterminationObservations
        The observations to remove the worst residual from.

    Returns
    -------
    filtered_observations : OrbitDeterminationObservations
        The observations with the worst residual removed.
    """
    assert (
        len(pc.unique(orbit_members.orbit_id)) == 1
    ), "Orbit members must only contain one orbit"
    assert pc.all(
        pc.is_in(orbit_members.obs_id, observations.id)
    ).as_py(), "Observations must contain all orbit member observations"

    # Find the worst outlier (the observation that has the lowest probability of
    # drawing a more extreme residual than the one observed)
    worst_outlier = orbit_members.apply_mask(
        pc.equal(
            orbit_members.residuals.probability,
            pc.min(orbit_members.residuals.probability),
        )
    )

    if len(worst_outlier) > 1:
        # If there are multiple worst outliers (which would be quite unlikely),
        # then remove the outlier with the highest squared residual value
        index = np.nansum(worst_outlier.residuals.to_array() ** 2, axis=1).argmax()
        worst_outlier = worst_outlier.take([index])

    # Grab the surviving observation IDs
    obs_ids = orbit_members.apply_mask(
        pc.invert(pc.is_in(orbit_members.obs_id, worst_outlier.obs_id))
    ).obs_id

    return observations.apply_mask(pc.is_in(observations.id, obs_ids))
