.. meta::
   :description: Installation guide for adam_core including optional extras for plotting, OEM export, and documentation builds.

Installation
============

Base Installation
-----------------

Use ``pip`` for user/runtime installs:

.. code-block:: bash

   pip install adam_core

Optional Installation Profiles
------------------------------

Use extras when your analysis requires additional capabilities.

Pip Extras
~~~~~~~~~~

.. code-block:: bash

   # plotting and visualization helpers used in impact analysis
   pip install "adam_core[plots]"

   # OEM export/import helpers
   pip install "adam_core[oem]"

Propagator Backends
-------------------

Propagation and ephemeris generation depend on a compatible propagator implementation.
A common setup is ``adam-assist``:

.. code-block:: bash

   pip install adam-assist

Development with PDM
--------------------

For contributor/development tasks, use PDM in a cloned repository checkout:

.. code-block:: bash

   # install runtime + test + docs dependency groups
   pdm install -G test -G docs

If an existing local ``pdm.lock`` was generated before the docs group was
included, PDM may report ``Requested groups not in lockfile: docs``. Refresh the
ignored local lockfile for the requested groups, then retry the install:

.. code-block:: bash

   pdm lock -G test -G docs
   pdm install -G test -G docs

Then run the documentation builds:

.. code-block:: bash

   # run documentation builds
   pdm run docs
   pdm run docs-check

Read the Docs and Self-Hosting
------------------------------

* Canonical hosted docs target: Read the Docs.
* Local/self-hosted build command from a development environment:

.. code-block:: bash

   pdm run docs

For strict CI-style checks:

.. code-block:: bash

   pdm run docs-check
