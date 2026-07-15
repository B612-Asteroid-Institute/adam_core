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

Trusted publishing
------------------

``publish.yml`` uses GitHub/PyPI OIDC with environment protection, not a stored
API token. Its manual input is the successful release-candidate workflow run
ID. The collector verifies the run name, successful conclusion, and exact head
SHA before assembling only ``adam_core`` wheels. TestPyPI and PyPI are separate
protected environments. The first release of each Rust crate must be made
manually before crates.io can attach an OIDC trusted publisher. Subsequent
crate publication verifies checksums and uploads the exact candidate ``.crate``
archives through the Cargo registry protocol without repackaging or compiling.
For each first release, use the same
``migration/scripts/publish_crate_archives.py --execute`` path with a manually
scoped token and the reviewed candidate artifact set; do not run ``cargo
publish`` against a fresh package.

Release order
-------------

After review and approval:

#. publish the reviewed permissive adam-core Rust crates in dependency order;
#. publish the exact accepted ``adam-core`` wheel set and verify it from the
   public index;
#. test ``adam-assist`` against that public adam-core release;
#. publish the exact accepted ``adam-assist`` wheel set; and
#. repeat the clean-room public-index smoke test.

``adam-assist`` owns ASSIST orchestration and consumes released
``libassist-sys`` and ``librebound-sys`` directly. Do not publish an
``assist-rs`` v2 facade. A tag, GitHub release, crates.io upload, or PyPI upload
is a human approval boundary and is never performed as part of migration
validation.
