from typing import Tuple

import numpy as np
import pyarrow.compute as pc

from .differential_correction import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers


def calculate_max_outliers(
    num_obs: int, min_obs: int, contamination_percentage: float
) -> int:
    """
    Calculate the maximum number of allowable outliers. Linkages may contain err
    oneuos observations that need to be removed. This function calculates the maximum number of
    observations that can be removed before the linkage no longer has the minimum number
    of observations required. The contamination percentage is the maximum percentage of observations
    that allowed to be erroneous.

    Parameters
    ----------
    num_obs : int
        Number of observations in the linkage.
    min_obs : int
        Minimum number of observations required for a valid linkage.
    contamination_percentage : float
        Maximum percentage of observations that allowed to be erroneous. Range is [0, 100].

    Returns
    -------
    outliers : int
        Maximum number of allowable outliers.
    """
    assert (
        num_obs >= min_obs
    ), "Number of observations must be greater than or equal to the minimum number of observations."
    assert (
        contamination_percentage >= 0 and contamination_percentage <= 100
    ), "Contamination percentage must be between 0 and 100."

    max_outliers = num_obs * (contamination_percentage / 100)
    outliers = np.min([max_outliers, num_obs - min_obs]).astype(int)
    return outliers


def remove_lowest_probability_observation(
    orbit_members: FittedOrbitMembers, observations: OrbitDeterminationObservations
) -> Tuple[str, OrbitDeterminationObservations]:
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
    obs_id : str
        The ID of the observation that was removed.
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

    # Grab the observation ID of the worst outlier
    obs_id = worst_outlier.obs_id[0].as_py()

    # Grab the surviving observation IDs
    obs_ids = orbit_members.apply_mask(
        pc.invert(pc.equal(orbit_members.obs_id, obs_id))
    ).obs_id

    return obs_id, observations.apply_mask(pc.is_in(observations.id, obs_ids))
