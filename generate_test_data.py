#!/usr/bin/env python
"""
Script to generate test data for impact visualization tests.
- Loads orbit from GCS
- Calculates impacts (which also generates variants)
- Saves all objects as parquet files
"""
import os
from pathlib import Path

import numpy as np
from adam_assist import ASSISTPropagator

from adam_core.coordinates import Origin
from adam_core.dynamics.impacts import (
    EARTH_RADIUS_KM,
    CollisionConditions,
    calculate_impacts,
)
from adam_core.dynamics.plots import MOON_RADIUS_KM
from adam_core.orbits import Orbits

# Directory to save the test data
TEST_DATA_DIR = Path("tests/adam_core/dynamics/test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Load orbit from GCS
print("Loading orbit from GCS...")
orbit = Orbits.from_parquet("gs://ntellis_scratch/impact_job_test/high_percent_IP_orbit.parquet")
print(f"Loaded orbit: {orbit}")

# Save orbit to test data directory
orbit_file = TEST_DATA_DIR / "test_orbit.parquet"
orbit.to_parquet(orbit_file)
print(f"Saved orbit to {orbit_file}")

# Initialize the propagator
print("Initializing propagator...")
propagator = ASSISTPropagator()

# Define the collision conditions including Earth and Moon
print("Setting up collision conditions...")
conditions = CollisionConditions.from_kwargs(
    condition_id=["Earth", "Moon"],
    collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
    collision_distance=[EARTH_RADIUS_KM, MOON_RADIUS_KM],
    stopping_condition=[True, True],
)

# Calculate impacts through 2035
print("Calculating impacts...")
num_variants = 100
days_to_propagate = (2036 - 2023) * 365  # Approximate days from now through 2035
variants, impacts = calculate_impacts(
    orbit,
    days_to_propagate,
    propagator,
    num_samples=num_variants,
    processes=4,  # Adjust based on your machine
    conditions=conditions,
)

# Save variant orbits to test data directory
variants_file = TEST_DATA_DIR / "test_variant_orbits.parquet"
variants.to_parquet(variants_file)
print(f"Saved variant orbits to {variants_file}")

# Save collision events to test data directory
impacts_file = TEST_DATA_DIR / "test_collision_event.parquet"
impacts.to_parquet(impacts_file)
print(f"Saved collision events to {impacts_file}")

# Print summary statistics
earth_impacts = impacts.apply_mask(impacts.collision_object.code == "EARTH")
moon_impacts = impacts.apply_mask(impacts.collision_object.code == "MOON")
print(f"Total impacts: {len(impacts)}")
print(f"Earth impacts: {len(earth_impacts)}")
print(f"Moon impacts: {len(moon_impacts)}")
print(f"Impact probability: {len(impacts) / num_variants * 100:.2f}%")

print("Test data generation complete!")