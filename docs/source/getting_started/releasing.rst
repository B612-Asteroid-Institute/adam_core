Release Process
===============

Artifact policy
---------------

Release artifacts are built once and tested as artifacts. The release-candidate
workflow builds one Core wheel for each supported Python/platform pair and
installs those exact files in fresh environments. Every cell runs the same
Core-owned acceptance: runner architecture, exact wheel count and platform tag,
``pip check``, installed distribution/Python/native identity, exact native build
commit, and one native numerical smoke. The workflow only supplies explicit
arguments to ``run_core_wheel_acceptance.py``, which invokes a copied
``core_wheel_smoke.py`` outside the source tree; it does not embed test
programs. Publication jobs download those accepted artifacts by
workflow run ID; they do not rebuild wheels or crates. Downstream projects
validate their own Core compatibility in their repositories.

The supported Core matrix is CPython 3.11-3.14 on manylinux 2.17 x86-64 and
AArch64 and macOS Apple silicon and Intel (16 Core wheels). Windows and
musllinux are not current release targets.

Stable release line
-------------------

The current published baseline is ``adam-core==0.7.0``, released on
2026-11-06 as 12 CPython 3.11-3.13 PyPI wheels and six public Rust crates built
against Arrow 54. Its files, crate versions, and ``v0.7.0`` tag are immutable
and must not be overwritten, deleted for replacement, or moved. Version 0.8.0
adds CPython 3.14 and aligns the Python package, the unpublished
``adam_core_py`` wheel implementation crate, and all six public Rust crates.
This 0.x minor release is required because public Arrow types move from Arrow 54
to Arrow 58 and the Rust OD surface has breaking changes. The 0.8.0 changelog
and expected ``v0.8.0`` tag use the release date 2026-11-10.

Once published, 0.8.0 files, versions, and its tag have the same immutable
policy. A correction after publication requires a new version and new tag; a
publisher must never replace accepted bytes under an existing filename or crate
version.

Rust 1.87 remains the authoritative MSRV; latest stable is compatibility-only.
The Python wheel contains the Python veneer and compiled
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

#. accept the exact 0.8.0 source candidate with aligned Python/Rust versions
   and stable publication guards;
#. accept the exact 16-wheel Core matrix with the same Core-owned checks on
   CPython 3.11-3.14;
#. publish and verify the aligned six-crate Rust 0.8.0 set in dependency order;
#. publish and verify the exact Core Python wheel set; and
#. run clean no-lock Cargo 1.87/latest-stable and clean pip/current-uv smoke
   tests from registry-only environments.

After Core 0.8 is public, use the manual exact-ref workflow with the reviewed
Assist candidate. Provider publication and compatibility policy belong to each
provider's own release process. A tag, GitHub release, protected-environment
change, crates.io upload, or PyPI upload is a human approval boundary and is
never performed as part of Core source preparation.
