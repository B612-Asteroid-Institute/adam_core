.. meta::
   :description: Installation guide for adam_core including optional extras for plotting, OEM export, and documentation builds.

Installation
============

Base Installation
-----------------

Use ``pip`` for user/runtime installs:

.. code-block:: bash

   pip install adam_core

Supported Native Wheels
-----------------------

The native release matrix supports CPython 3.11, 3.12, 3.13, and 3.14 on:

* manylinux 2.17+ x86-64;
* manylinux 2.17+ AArch64;
* macOS 11+ Apple silicon; and
* macOS Intel x86-64.

Windows and musllinux wheels are not current release targets. A source build
on an unlisted platform is not a substitute for an accepted release artifact.
The historical optional ``jax`` extra follows upstream JAX/JAXLIB wheel
availability and is not part of the standard CPython support or release gate.

These wheels are built once, installed and exercised by the clean-room
acceptance suite, and only those exact tested files are eligible for
publication. Installing from source requires a Rust toolchain and the platform
C build tools; release consumers should normally require wheels.

Optional Installation Profiles
------------------------------

Use extras when your analysis requires additional capabilities.

Pip Extras
~~~~~~~~~~

.. code-block:: bash

   # plotting and visualization helpers used in impact analysis
   pip install "adam_core[plots]"

   # Astropy Time objects and explicit UT1/IERS conversions
   pip install "adam_core[astropy]"

   # compatibility with code that monkeypatches astroquery.jplsbdb.SBDB.query
   pip install "adam_core[legacy-sbdb]"

   # Healpy compatibility helpers
   pip install "adam_core[healpix]"

   # historical explicit JAX rotation-fit bridge (the public solver uses Rust)
   pip install "adam_core[jax]"

   # OEM export/import helpers
   pip install "adam_core[oem]"

Propagator Backends
-------------------

Propagation and ephemeris generation depend on a separately installed
implementation of the ``adam_core.propagator.Propagator`` contract. Choose and
configure that implementation in the application; ``adam-core`` does not
bundle or select one.

``adam-core`` owns the data models, coordinate and dynamics kernels, generic
propagator contracts, and Core orbit-determination workflows. A provider owns
its propagation implementation and provider-specific data; Core does not
delegate OD orchestration or fitting to it. Consult the provider's
documentation for installation, configuration, supported physics, and license
terms.

Kernel Data Resolution
----------------------

Python wheels use the NAIF kernel files already installed in the active Python
environment; they do not copy those large files into another cache. Pure-Rust
consumers resolve kernel data in this order: an explicit path override, an
installed Python package, the adam-core kernel cache, then a checksummed wheel
download. Set ``ADAM_CORE_KERNEL_OFFLINE=1`` to prohibit downloads. A cache is
populated only when no installed package provides the requested file, avoiding
unnecessary duplication.

Development with PDM
--------------------

For contributor/development tasks, use PDM in a cloned repository checkout:

This repository does not commit ``pdm.lock``. CI resolves dependencies from
``pyproject.toml`` for each job, and local ``pdm.lock`` files are disposable
developer artifacts.

.. code-block:: bash

   # install runtime + test + docs + exact build-tool dependency groups
   pdm install -G test -G docs -G build

If an existing local ``pdm.lock`` was generated before the docs group was
included, PDM may report ``Requested groups not in lockfile: docs``. Refresh the
ignored local lockfile for the requested groups, then retry the install:

.. code-block:: bash

   pdm lock -G test -G docs -G build
   pdm install -G test -G docs -G build

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
