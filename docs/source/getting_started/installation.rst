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

The native release matrix supports CPython 3.11, 3.12, and 3.13 on:

* manylinux 2.17+ x86-64;
* manylinux 2.17+ AArch64;
* macOS 11+ Apple silicon;
* macOS Intel x86-64; and
* Windows x86-64.

These wheels are built once, installed and exercised by the clean-room
acceptance suite, and only those exact tested files are eligible for
publication. ``musllinux`` is not a supported release target: the SPICE and
ASSIST native stacks target glibc on Linux, and no native musllinux acceptance
runner is available. Installing from source requires a Rust toolchain and the
platform C build tools; release consumers should normally require wheels.

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

Propagation and ephemeris generation depend on a compatible propagator implementation.
A common setup is ``adam-assist``:

.. code-block:: bash

   pip install adam-assist

``adam-core`` owns data models, coordinate/dynamics kernels, provider clients,
and generic propagator contracts. ``adam-assist`` owns ASSIST orchestration and
exposes ``adam_assist.ASSISTPropagator``. Its Rust extension consumes the
reviewed ``libassist-sys`` and ``librebound-sys`` crates directly; there is no
separate ``assist-rs`` runtime package.

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
