Release Process
===============

Artifact policy
---------------

Release artifacts are built once and tested as artifacts. The release-candidate
workflow builds one wheel for each supported Python/platform pair, installs
those exact files in isolated binary-only environments, runs ``pip check``,
proves offline installed-package kernel discovery without duplicate cache
files, and exercises public SBDB, observer, ephemeris, and ASSIST workflows.
Publication jobs download those accepted artifacts by workflow run ID; they do
not check out a builder or rebuild wheels.

The supported matrix is CPython 3.11-3.13 on manylinux 2.17 x86-64 and AArch64
and macOS Apple silicon and Intel (12 wheels per distribution). The workflow
checks runner architecture, exact wheel count, and platform tags. Windows is
deferred because ``libassist-sys 1.2.1`` requires the upstream ASSIST POSIX
memory-mapping implementation; musllinux is also deliberately unsupported.

Preview versions and opt-in installation
----------------------------------------

The currently published migration preview is ``adam-core==0.5.6rc2`` on
PyPI. The corrected promotion candidate is ``adam-core==0.5.6rc5`` with the already
published public Rust crates ``0.1.0-rc.4`` and exact internal requirements
such as ``=0.1.0-rc.4``; it is not published until its exact artifacts complete
the separately authorized hosted matrix and post-download inspection. Pip and
Cargo exclude prereleases from ordinary resolution;
preview consumers must opt in with an exact pin. The Python wheel contains the
Python veneer and compiled ``adam_core._rust_native`` extension, so Python
consumers do not need to install the component crates from crates.io.

The current stable PyPI release remains the default for ``pip install
adam-core``. A public preview is still visible and intentionally installable by
anyone who supplies ``--pre`` or the exact version. Use a private package index
instead if public visibility is unacceptable.

Trusted publishing
------------------

``publish.yml`` uses GitHub/PyPI OIDC with protected ``testpypi-preview`` and
``pypi-preview`` environments. Its manual inputs include the successful
release-candidate run ID, exact version, and a confirmation containing the
version, commit SHA, and destination. The collector verifies the run name,
successful conclusion, exact head SHA, RC-only version, wheel metadata, and the
complete 12-wheel matrix before assembling only ``adam_core`` wheels.

The six public Rust crates ``0.1.0-rc.4`` were published through crates.io OIDC
trusted publishing from the exact hosted candidate archives. Crate publication
verified explicit confirmation, candidate provenance, checksums, archive
metadata, prerelease versions, and exact internal dependency pins, then
uploaded the accepted ``.crate`` archives in dependency order without
repackaging or compiling.

Release order
-------------

The six Rust crates are already published. After review and approval:

#. publish the exact accepted ``adam-core==0.5.6rc5`` wheel set and verify it
   from the public index;
#. resolve adam-assist against exact public RC dependencies and test the
   prepared ``adam-assist==0.4.0rc6`` candidate against
   ``adam-core==0.5.6rc5``;
#. publish the exact accepted ``adam-assist`` RC wheel set; and
#. run the precovery-v2 clean package-manager smoke test with exact pins.

``adam-assist`` owns ASSIST orchestration and consumes released
``libassist-sys`` and ``librebound-sys`` directly. Do not publish an
``assist-rs`` v2 facade. A tag, GitHub release, crates.io upload, or PyPI upload
is a human approval boundary and is never performed as part of migration
validation.
