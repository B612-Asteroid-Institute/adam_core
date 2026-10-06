Coordinate Covariances
======================

``CoordinateCovariances`` stores 6x6 uncertainty matrices for coordinate tables.
This section covers the full covariance process: construction, validation,
sampling, and transformation.

Build Covariances
-----------------

From sigmas (diagonal covariance).

.. code-block:: python

   import numpy as np
   from adam_core.coordinates import CoordinateCovariances

   sigmas = np.array([
       [1e-9, 1e-9, 1e-9, 1e-11, 1e-11, 1e-11],
       [2e-9, 2e-9, 2e-9, 2e-11, 2e-11, 2e-11],
   ])
   cov_diag = CoordinateCovariances.from_sigmas(sigmas)

From full matrices.

.. code-block:: python

   cov_matrix = np.zeros((1, 6, 6), dtype=float)
   cov_matrix[0, 0, 0] = 1e-18
   cov_matrix[0, 1, 1] = 1e-18
   cov_matrix[0, 0, 1] = cov_matrix[0, 1, 0] = 1e-19

   cov_full = CoordinateCovariances.from_matrix(cov_matrix)
   round_trip = cov_full.to_matrix()

Missing or Optional Covariance
------------------------------

.. code-block:: python

   # Use when uncertainty is unavailable at ingest time.
   cov_null = CoordinateCovariances.nulls(length=5)
   all_missing = cov_null.is_all_nan()

Working with Raw Helpers
------------------------

.. code-block:: python

   from adam_core.coordinates.covariances import sigmas_to_covariances

   dense = sigmas_to_covariances(sigmas)

PSD Repair for Near-Singular Inputs
-----------------------------------

.. code-block:: python

   from adam_core.coordinates.covariances import make_positive_semidefinite

   cov_psd = make_positive_semidefinite(cov_matrix[0], semidef_tol=1e-15)

Use this when tiny negative eigenvalues are numerical artifacts, not true model
pathology.

Sampling and Reconstruction
---------------------------

Random Monte Carlo sampling.

.. code-block:: python

   from adam_core.coordinates.covariances import (
       sample_covariance_random,
       weighted_mean,
       weighted_covariance,
   )

   mean = np.array([1.0, 0.0, 0.0, 0.0, 0.01, 0.0])
   cov = np.diag([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])

   samples_mc, w_mc, w_cov_mc = sample_covariance_random(mean, cov, num_samples=5000, seed=42)
   mean_mc = weighted_mean(samples_mc, w_mc)
   cov_mc = weighted_covariance(mean_mc, samples_mc, w_cov_mc)

Sigma-point sampling.

.. code-block:: python

   from adam_core.coordinates.covariances import sample_covariance_sigma_points

   samples_sp, w_sp, w_cov_sp = sample_covariance_sigma_points(
       mean,
       cov,
       alpha=1.0,
       beta=0.0,
       kappa=0.0,
   )

Transforming Covariances Across Representations
-----------------------------------------------

.. code-block:: python

   import numpy as np
   from adam_core.coordinates.covariances import (
       transform_covariances_jacobian,
       transform_covariances_sampling,
   )
   from adam_core.coordinates.transform import (
       _cartesian_to_spherical,
       cartesian_to_spherical,
   )

   coords_cart = np.array([[1.0, 0.1, 0.2, 0.0, 0.01, 0.0]])
   cov_cart = np.diag([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])[None, :, :]

   cov_sph_jac = transform_covariances_jacobian(coords_cart, cov_cart, _cartesian_to_spherical)
   cov_sph_samp = transform_covariances_sampling(coords_cart, cov_cart, cartesian_to_spherical, num_samples=20000)

Choosing a Method
-----------------

* ``from_sigmas``: diagonal-only uncertainties from upstream catalogs.
* ``from_matrix``: full covariance from OD/fit outputs.
* Sigma-point sampling: fast uncertainty propagation.
* Monte Carlo sampling: robustness for nonlinear or poorly behaved covariance.
* Jacobian transform: fast local linear approximation.
* Sampling transform: slower but less sensitive to nonlinearity.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.coordinates`
* :doc:`../reference/api/adam_core.orbit_determination`
