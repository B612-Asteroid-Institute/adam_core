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

* :doc:`../reference/coordinates`
* :doc:`../reference/orbits`
* :doc:`../reference/orbit_determination`

Input Types
-----------
.. code-block:: python

   # Core table types used across this stack.
   # CartesianCoordinates.from_kwargs(..., time: Timestamp, covariance: CoordinateCovariances, origin: Origin, frame: str) -> CartesianCoordinates
   # transform_coordinates(coords: CoordinateType, representation_out: type[CoordinateType], frame_out: str | None = None, origin_out: OriginCodes | None = None) -> CoordinateType
   # Residuals.calculate(observed: CoordinateType, predicted: CoordinateType, use_predicted_covariance: bool = True) -> Residuals
