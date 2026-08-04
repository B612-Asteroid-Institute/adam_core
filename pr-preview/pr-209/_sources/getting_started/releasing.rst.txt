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

The migration preview is ``adam-core==0.5.6rc2`` on PyPI. The public Rust
crates use ``0.1.0-rc.2`` and exact internal requirements such as
``=0.1.0-rc.2``. Pip and Cargo exclude prereleases from ordinary resolution;
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

The first release of each Rust crate uses a manually scoped token held only in
the protected ``crates-io-preview`` environment; crates.io can attach an OIDC
trusted publisher after that crate exists. Crate publication verifies the same
explicit confirmation, candidate provenance, checksums, archive metadata,
prerelease versions, and exact internal dependency pins, then uploads the exact
candidate ``.crate`` archives in dependency order without repackaging or
compiling. The bootstrap token must be revoked after owners and trusted
publishers are configured.

Release order
-------------

After review and approval:

#. publish ``adam_core_rs_autodiff`` and
   ``adam_core_rs_orbit_determination``;
#. publish ``adam_core_rs_coords``, ``adam_core_rs_spice``,
   ``adam_core_rs_kernel_data``, then the ``adam_core`` umbrella crate;
#. publish the exact accepted ``adam-core`` RC wheel set and verify it from the
   public index;
#. replace adam-assist's temporary vendored core crates with exact public RC
   dependencies and test ``adam-assist==0.4.0rc1`` against the public
   ``adam-core==0.5.6rc2`` release;
#. publish the exact accepted ``adam-assist`` RC wheel set; and
#. run the precovery-v2 clean package-manager smoke test with exact pins.

``adam-assist`` owns ASSIST orchestration and consumes released
``libassist-sys`` and ``librebound-sys`` directly. Do not publish an
``assist-rs`` v2 facade. A tag, GitHub release, crates.io upload, or PyPI upload
is a human approval boundary and is never performed as part of migration
validation.
