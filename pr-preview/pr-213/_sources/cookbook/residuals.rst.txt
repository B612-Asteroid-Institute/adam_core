Residuals and Fit Quality
=========================

Residual analysis is the bridge between model output and observation quality.
``Residuals.calculate`` is the primary entrypoint; helper functions provide
fine-grained control for diagnostics.

Simple Residuals.calculate Example
----------------------------------

.. code-block:: python

   import numpy as np
   from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin, OriginCodes
   from adam_core.coordinates.residuals import Residuals
   from adam_core.time import Timestamp

   t = Timestamp.from_mjd(np.array([60200.0, 60200.1]), scale="tdb")
   cov = CoordinateCovariances.from_sigmas(np.full((2, 6), 1e-8))

   observed = CartesianCoordinates.from_kwargs(
       x=[1.0, 1.1], y=[0.0, 0.1], z=[0.0, 0.0],
       vx=[0.0, 0.0], vy=[0.01, 0.01], vz=[0.0, 0.0],
       time=t,
       covariance=cov,
       origin=Origin.from_OriginCodes(OriginCodes.SUN, size=2),
       frame="ecliptic",
   )

   predicted = CartesianCoordinates.from_kwargs(
       x=[1.0, 1.10001], y=[0.0, 0.09999], z=[0.0, 0.0],
       vx=[0.0, 0.0], vy=[0.01, 0.01], vz=[0.0, 0.0],
       time=t,
       covariance=CoordinateCovariances.from_sigmas(np.full((2, 6), 5e-9)),
       origin=Origin.from_OriginCodes(OriginCodes.SUN, size=2),
       frame="ecliptic",
   )

   residuals = Residuals.calculate(observed, predicted, use_predicted_covariance=True)
   residual_matrix = residuals.to_array()
   chi2 = residuals.chi2.to_pylist()
   dof = residuals.dof.to_pylist()
   p = residuals.probability.to_pylist()

Atomic Helper Functions
-----------------------

.. code-block:: python

   import numpy as np
   from adam_core.coordinates.residuals import (
       calculate_chi2,
       calculate_reduced_chi2,
       bound_longitude_residuals,
       apply_cosine_latitude_correction,
   )

   r = np.array([[0.01, -0.02, 0.0, 0.0, 0.0, 0.0]])
   c = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])[None, :, :]
   chi2 = calculate_chi2(r, c)

   reduced = calculate_reduced_chi2(residuals, parameters=6)

   # Spherical-specific helpers.
   observed_sph = np.array([[1.0, 359.0, 20.0, 0.0, 0.0, 0.0]])
   raw_resid_sph = np.array([[0.0, 358.0, 0.0, 0.0, 5.0, 0.0]])
   wrapped = bound_longitude_residuals(observed_sph, raw_resid_sph)
   corrected, corrected_cov = apply_cosine_latitude_correction(
       observed_sph[:, 2],
       wrapped,
       c.copy(),
   )

Predicted Covariance Tradeoff
-----------------------------

.. code-block:: python

   # Include both model + observation uncertainty.
   r_total = Residuals.calculate(observed, predicted, use_predicted_covariance=True)

   # Observation-only weighting.
   r_obs_only = Residuals.calculate(observed, predicted, use_predicted_covariance=False)

Use observation-only weighting for strict measurement-gating. Use combined
weighting when prediction uncertainty is a material part of the decision.

Spherical Residuals Behavior
----------------------------

For ``SphericalCoordinates``, ``Residuals.calculate`` automatically:

* wraps longitude residuals into [-180, 180] with boundary-aware sign handling
* applies cosine(latitude) scaling to longitude and longitudinal-rate terms
* applies equivalent scaling to covariance before chi-square evaluation

Failure Modes Worth Testing
---------------------------

* observed/predicted type mismatch
* frame mismatch
* origin mismatch
* missing diagonal covariance values
* length mismatch when predicted is not scalar-broadcastable

Related Reference
-----------------

* :doc:`../reference/api/adam_core.coordinates`
* :doc:`../reference/api/adam_core.orbit_determination`

