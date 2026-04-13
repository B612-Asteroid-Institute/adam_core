Coordinates, Covariances, Transforms, and Residuals
===================================================

This section covers the full coordinate-analysis stack in ``adam_core``:

* coordinate representations
* covariance modeling and uncertainty propagation
* frame/origin/representation transforms
* residual and chi-square diagnostics
* variant sampling and collapse patterns

Recommended Reading Order
-------------------------

1. :doc:`coordinates_classes`
2. :doc:`coordinate_covariances`
3. :doc:`coordinate_transforms`
4. :doc:`residuals`
5. :doc:`variant_sampling_and_collapse`

Why This Ordering Works
-----------------------

* You define state representation first.
* You attach/propagate uncertainty second.
* You normalize state geometry and frame conventions third.
* You score fit quality fourth.
* You scale uncertainty propagation to ensemble operations last.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.coordinates`
* :doc:`../reference/api/adam_core.orbits`
* :doc:`../reference/api/adam_core.orbit_determination`

