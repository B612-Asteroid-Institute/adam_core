Dynamics and Impact Analysis
============================

This section breaks impact and screening analysis into dedicated guides.

Dedicated Guides
----------------

* :doc:`impact_probabilities` for Monte Carlo collision probability analysis.
* :doc:`moid_analysis` for geometric screening and perturber-wise MOID tables.

Additional Dynamics Metric
--------------------------

.. code-block:: python

   from adam_core.dynamics.tisserand import calc_tisserand_parameter

   tj = calc_tisserand_parameter(a=2.5, e=0.2, i=8.0, third_body="jupiter")
   print(tj)

When to Use
-----------

* Use MOID first for fast triage.
* Use impact-probability analysis for decision-grade risk quantification.
* Use Tisserand-style diagnostics for dynamical classification context.

Related Reference
-----------------

* :doc:`../reference/dynamics`

Input Types
-----------
.. code-block:: python

   # calc_tisserand_parameter(a: float, e: float, i: float, third_body: str) -> float
