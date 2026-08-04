ADES Module: Parse, Build, and Exchange
=======================================

The ``adam_core.observations.ades`` module handles MPC ADES exchange in both
directions:

* parse ADES text into structured ``ObsContext`` + ``ADESObservations``
* serialize ``ADESObservations`` + contexts back to ADES text

Use this when ADES is your system boundary for ingest, QA, submission, or
handoff to orbit-determination pipelines.

Current Format Support
----------------------

``adam_core`` currently supports **ADES PSV** (pipe-separated values) only:

* supported: ADES PSV parse/serialize via ``ADES_string_to_tables`` and ``ADES_to_string``
* not currently implemented: ADES XML parse/serialize

Parse ADES Text into Tables
---------------------------

.. code-block:: python

   from adam_core.observations.ades import ADESObservations, ADES_string_to_tables, ObsContext

   ades_text: str = """# version=2022
   # observatory
   ! mpcCode 695
   # submitter
   ! name J. Doe
   # observers
   ! name Survey Team
   # measurers
   ! name Reduction Team
   # telescope
   ! design Reflector
   ! aperture 1.0
   ! detector CCD
   permID|obsTime|ra|dec|stn|mode|astCat
   1234|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
   """

   contexts: dict[str, ObsContext]
   observations: ADESObservations
   contexts, observations = ADES_string_to_tables(ades_text)

   print(contexts["695"].observatory.mpcCode, len(observations))

Build ADES Tables and Contexts in Python
----------------------------------------

.. code-block:: python

   import numpy as np
   from adam_core.observations.ades import (
       ADESObservations,
       ObsContext,
       ObservatoryObsContext,
       SoftwareObsContext,
       SubmitterObsContext,
       TelescopeObsContext,
   )
   from adam_core.time import Timestamp

   observations: ADESObservations = ADESObservations.from_kwargs(
       permID=["3000", "3000"],
       trkSub=["a1234b", "a1234b"],
       obsSubID=["obs01", "obs02"],
       obsTime=Timestamp.from_mjd(np.array([60434.0, 60434.1]), scale="utc"),
       ra=[240.0, 240.05],
       dec=[-15.0, -15.05],
       rmsRACosDec=[0.9659, 0.9657],
       rmsDec=[1.0, 1.0],
       mag=[20.0, 20.3],
       band=["r", "g"],
       stn=["W84", "W84"],
       mode=["CCD", "CCD"],
       astCat=["Gaia2", "Gaia2"],
   )

   context_w84: ObsContext = ObsContext(
       observatory=ObservatoryObsContext(mpcCode="W84", name="Cerro Tololo - Blanco + DECam"),
       submitter=SubmitterObsContext(name="J. Doe", institution="B612 Asteroid Institute"),
       observers=["Survey Team"],
       measurers=["Reduction Team"],
       telescope=TelescopeObsContext(
           name="Blanco 4m",
           design="Reflector",
           aperture=4.0,
           detector="CCD",
       ),
       software=SoftwareObsContext(objectDetection="ADAM::THOR"),
       comments=["TEST DATA"],
   )
   contexts: dict[str, ObsContext] = {"W84": context_w84}

Serialize ADES for Submission or Handoff
----------------------------------------

.. code-block:: python

   from adam_core.observations.ades import ADES_to_string

   ades_text: str = ADES_to_string(
       observations,
       contexts,
       seconds_precision=3,
       columns_precision={
           "ra": 9,
           "dec": 9,
           "rmsRACosDec": 5,
           "rmsDec": 5,
           "mag": 4,
           "rmsMag": 4,
       },
   )
   print(ades_text.splitlines()[0])  # "# version=2022"

Round-Trip Validation Pattern
-----------------------------

.. code-block:: python

   import numpy as np
   from adam_core.observations.ades import ADES_string_to_tables

   parsed_contexts, parsed_observations = ADES_string_to_tables(ades_text)

   # Sort before comparing because ADES is grouped by observatory block.
   original = observations.sort_by(["stn", "obsTime.days", "obsTime.nanos"])
   parsed = parsed_observations.sort_by(["stn", "obsTime.days", "obsTime.nanos"])

   np.testing.assert_allclose(
       original.obsTime.mjd().to_numpy(zero_copy_only=False),
       parsed.obsTime.mjd().to_numpy(zero_copy_only=False),
   )

Important ADES Behaviors
------------------------

* At least one of ``permID``, ``provID``, or ``trkSub`` must be present for
  serialization.
* Unknown ADES columns are ignored on parse (with a warning).
* Empty/whitespace optional fields are normalized to null values.
* ``rmsRA`` in ADES maps to ``rmsRACosDec`` in ``ADESObservations``.

When To Use
-----------

* ingesting observations from ADES exchange files
* generating ADES for MPC-style submission pipelines
* reproducible round-trip QA between internal tables and ADES text

Related Reference
-----------------

* :doc:`../reference/api/adam_core.observations`
* :doc:`observations_and_observers`
* :doc:`../use_cases/orbit_determination`
