from .variants import VariantOrbits
import quivr as qv
from typing import Optional
from adam_core.propagator.propagator import Propagator
import pyarrow as pa
import pyarrow.compute as pc

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.constants import au
import astropy.units as u
import math
from collections import Counter
from adam_core.coordinates import CartesianCoordinates

class EarthImpacts(qv.Table):
    orbit_id = qv.StringColumn()
    # Distance from earth center in km
    distance = qv.Float64Column()
    coordinates = CartesianCoordinates.as_column()
    variant_id = qv.LargeStringColumn()


def calculate_impacts(orbits, num_days, propagator, num_samples: int = 1000):
    """
    Calculate the impacts for each variant orbit generated from the input orbits.

    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits for which to calculate impact probabilities.
    num_days : int
        Number of days to propagate the orbits.
    propagator : `~adam_core.propagator.propagator.Propagator`
        Propagator to use for orbit propagation.
    num_samples : int, optional
        Number of samples to take over the period, by default 1000.

    Returns
    -------

    """

    impact_list = []
    variants = VariantOrbits.create(
                                orbits, method='monte-carlo', num_samples=num_samples, 
                            )
    for i, var_orbit in enumerate(variants):
        results, impact = propagator._propagate_orbits_inner(var_orbit, var_orbit.coordinates.time.add_days(num_days)[0], detect_impacts=True, adaptive_mode=2, min_dt=1e-15)   
        if impact is not None:
            earth_impact = EarthImpacts.from_kwargs(
                variant_id=[var_orbit.variant_id[0]],
                orbit_id=impact.orbit_id,
                distance=impact.distance,
                coordinates=impact.coordinates,
            )
            impact_list.append(earth_impact)

    earth_impacts = qv.concatenate(impact_list)

    return variants, earth_impacts


def calculate_impact_probabilities(variants, impacts):
    """
    Calculate the impact probabilities for each variant orbit generated from the input orbits.

    Parameters
    ----------
    variants : `~adam_core.orbits.variants.VariantOrbits`
        Variant orbits for which to calculate impact probabilities.
    impacts : `~adam_core.orbits.impacts.Impacts`
        Impacts for the variant orbits.

    Returns
    -------
    impact_probabilities : `~adam_core.orbits.impact_probabilities.ImpactProbabilities`
        Impact probabilities for the variant orbits.
    """

    #Loop through the unique set of orbit_ids within variants using quivr
    unique_orbits = pc.unique(variants.orbit_id)
    print(unique_orbits)

    ip_dict = {}

    for orbit_id in unique_orbits:
        mask = pc.equal(variants.orbit_id, orbit_id)
        variant_masked = variants.table.filter(mask)
        variant_count = len(variant_masked)
        impacts_mask = pc.equal(impacts.orbit_id, orbit_id)
        impacts_maked = impacts.table.filter(impacts_mask)
        impact_count = len(impacts_mask)
        #join the variant_masked with the impacts table
        #joined = qv.join(variant_masked, impacts, 'variant_id', 'variant_id')
        ip = impact_count/variant_count

        ip_dict[orbit_id] = ip

    return ip_dict

