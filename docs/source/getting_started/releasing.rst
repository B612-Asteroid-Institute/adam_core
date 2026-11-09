Release Process
===============

Artifact policy
---------------

Release artifacts are built once and tested as artifacts. The release-candidate
workflow builds one Core wheel for each supported Python/platform pair and
installs those exact files in isolated binary-only environments. CPython
3.11-3.13 cells pair an exact compatible Assist revision, run ``pip check``,
prove offline installed-package kernel discovery without duplicate cache
files, and exercise public SBDB, observer, ephemeris, and
propagation-provider workflows. CPython 3.14 cells instead run Core-only import,
identity, dependency, timestamp, and native numerical checks until Assist's
separately tracked 3.14 migration is complete.
Publication jobs download those accepted artifacts by workflow run ID; they do
not rebuild wheels or crates.

The supported Core matrix is CPython 3.11-3.14 on manylinux 2.17 x86-64 and
AArch64 and macOS Apple silicon and Intel (16 Core wheels). The compatible
Assist candidate is paired only on CPython 3.11-3.13 (12 paired cells). The
workflow checks runner architecture, exact wheel count, runtime/distribution version
equality, native members, and platform tags. Windows and musllinux are not
current release targets.

Stable release line
-------------------

The current published baseline is ``adam-core==0.7.0``, released on
2026-11-06 as 12 CPython 3.11-3.13 PyPI wheels and six public Rust crates built
against Arrow 54. Those files, crate versions, and the ``v0.7.0`` tag are
immutable. This CPython 3.14/native-stack candidate cannot be published as
``0.7.0``; it requires a separately authorized post-0.7 Python version. The
intended aligned release boundary is ``0.8.0``, but this candidate does not
perform or authorize that version change.

``adam_core_rs_coords`` directly exposes Arrow types, and the ``adam_core``
umbrella crate re-exports that API. Moving those public types from Arrow 54 to
Arrow 58 is semver-breaking. The project's exact internal requirements and
aligned public-crate version policy therefore require all six public Rust
crates to advance together to ``0.8.0`` before any crates.io publication. The
source manifests and unpublished ``adam_core_py`` wheel implementation crate
intentionally remain at ``0.7.0`` pending separate version-bump, review, and
publication authorization; they are not publishable from this candidate. Rust
1.87 is the authoritative MSRV; latest stable is compatibility-only. The
Python wheel contains the Python veneer and compiled
``adam_core._rust_native`` extension, so Python consumers do not install the
component crates directly.

Trusted publishing
------------------

Stable workflows target protected production environments named ``pypi`` and
``crates-io``. Those GitHub environments and matching registry trusted
publishers must be provisioned with exact stable-tag deployment policies and
required reviewers before an irreversible dispatch. Preview environments stay
separate so stable validation never weakens prerelease-only guards. TestPyPI is
not available.

Manual publisher inputs bind the successful candidate/acceptance run, exact
version, commit SHA, destination, and explicit confirmation. Collectors verify
the run name, success, exact head SHA, stable-only versions, archive hashes,
wheel metadata, and complete platform set. The six crates upload in dependency
order from exact prebuilt ``.crate`` files.

Both registry paths are safe to resume after a partial upload. Before sending
bytes, they inspect already-public filenames or crate versions and continue
only when every existing item is unyanked and checksum-identical to the
accepted artifact. PyPI receives only missing wheel files; crates.io skips only
exact existing archives. Any unexpected filename, checksum, or yanked entry
fails before continuing. A final public-registry comparison must prove the
complete release matches the accepted bytes.

Release order
-------------

Stable promotion remains separately approval-gated and proceeds in dependency
order:

#. prepare and review an exact propagation-only v2 reference-provider source
   revision for dependent acceptance. The current workflow names this required
   operational input ``adam_assist_ref`` and requires a full commit SHA; that
   revision is evidence for the Core candidate, not a Core dependency or
   publication candidate;
#. after separate version-bump authorization, align the Python distribution,
   unpublished ``adam_core_py`` implementation crate, and six public Rust
   crates on the intended ``0.8.0`` boundary without rewriting published
   ``0.7.0`` history;
#. accept the exact 16-wheel Core Python matrix for that authorized post-0.7
   version, including paired provider compatibility on CPython 3.11-3.13 and
   explicit Core-only CPython 3.14 acceptance;
#. accept and publish the aligned six-crate Rust ``0.8.0`` set in dependency
   order;
#. publish and verify the exact Core Python wheel set; and
#. run clean no-lock Cargo 1.87/latest-stable and clean pip/current-uv smoke
   tests from registry-only environments.

Provider publication and compatibility policy belong to each provider's own
release process. A tag, GitHub release, protected-environment change,
crates.io upload, or PyPI upload is a human approval boundary and is never
performed as part of Core source preparation.
