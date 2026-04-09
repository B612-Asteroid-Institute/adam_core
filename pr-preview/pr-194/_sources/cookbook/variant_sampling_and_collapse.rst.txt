Variant Sampling and Collapse
=============================

This pattern turns one uncertain orbit into an ensemble, propagates each member,
then reconstructs a mean state and covariance.

Narrative: Impact-Style Uncertainty Propagation
-----------------------------------------------

1. Start with a nominal orbit + covariance.
2. Sample variants from covariance.
3. Propagate variants to a target epoch.
4. Collapse variants back into a mean orbit and covariance.

Simple Example (Single Epoch)
-----------------------------

.. code-block:: python

   import numpy as np
   from adam_core.dynamics.propagation import propagate_2body
   from adam_core.orbits.query import query_sbdb
   from adam_core.orbits.variants import VariantOrbits
   from adam_core.time import Timestamp

   # Ceres is used here as a target object example, not as a perturber/origin.
   base = query_sbdb(["Ceres"])

   variants = VariantOrbits.create(
       base,
       method="sigma-point",  # "auto" or "monte-carlo" also supported
   )

   target_time = Timestamp.from_mjd(np.array([60220.0]), scale="tdb")
   propagated_variants = propagate_2body(variants, target_time, max_processes=1)

   reconstructed = propagated_variants.collapse_by_object_id()

Advanced Variant Controls
-------------------------

.. code-block:: python

   variants_mc = VariantOrbits.create(
       base,
       method="monte-carlo",
       num_samples=10000,
       seed=42,
   )

   variants_sp = VariantOrbits.create(
       base,
       method="sigma-point",
       alpha=1.0,
       beta=0.0,
       kappa=0.0,
   )

Choosing a Sampling Method
--------------------------

* ``sigma-point``: fast, fixed sample count (13 for 6D states).
* ``monte-carlo``: slower, robust for harder covariance geometry.
* ``auto``: tries sigma-point, falls back to Monte Carlo when reconstruction quality is poor.

From Variant Orbits to Variant Ephemerides
------------------------------------------

If your propagator returns ``VariantEphemeris`` rows, use
``VariantEphemeris.collapse_by_object_id`` to collapse by object/time/origin
into UT mean ephemerides with covariance.

.. code-block:: python

   # Example shape:
   # variant_ephem = propagator.generate_ephemeris(variants, observers, covariance=False)
   # mean_ephem = variant_ephem.collapse_by_object_id(aberration_mode="recompute")

When to Use This Pattern
------------------------

* Impact risk and uncertainty corridors.
* Observation planning under covariance.
* Any analysis where propagated uncertainty matters as much as nominal state.

Related Reference
-----------------

* :doc:`../reference/orbits`
* :doc:`../reference/dynamics`
* :doc:`../reference/propagator`

Input Types
-----------
.. code-block:: python

   # VariantOrbits.create(orbits: Orbits, method: str = "auto", num_samples: int = 10000, seed: int | None = None, alpha: float = 1.0, beta: float = 0.0, kappa: float = 0.0) -> VariantOrbits
   # propagate_2body(orbits: VariantOrbits, times: Timestamp, ...) -> VariantOrbits
   # VariantOrbits.collapse_by_object_id() -> Orbits
   # VariantEphemeris.collapse_by_object_id(aberration_mode: str = "recompute") -> Ephemeris
