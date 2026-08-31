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
not rebuild wheels or crates.

The supported matrix is CPython 3.11-3.13 on manylinux 2.17 x86-64 and AArch64
and macOS Apple silicon and Intel (12 wheels per distribution). The workflow
checks runner architecture, exact wheel count, runtime/distribution version
equality, native members, and platform tags. Windows is deferred because
``libassist-sys 1.2.1`` requires the upstream ASSIST POSIX memory-mapping
implementation; musllinux is deliberately unsupported.

Stable release line
-------------------

The stable migration release is ``adam-core==0.5.7`` for Python and ``0.5.7``
for all six public Rust crates. The version advances from the ``0.5.6rc6``
Python preview because the immutable historical Git tag ``v0.5.6`` already
exists at older source. The tag must never be moved or reused. Aligning the
Python and public Rust versions lets one exact ``v0.5.7`` source tag identify
the complete Core product release.

All internal public Rust dependencies use exact ``=0.5.7`` requirements.
``adam_core_py`` remains an unpublished wheel implementation crate. Rust 1.87
is the authoritative MSRV; latest stable is compatibility-only. The Python
wheel contains the Python veneer and compiled ``adam_core._rust_native``
extension, so Python consumers do not install the component crates directly.

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

#. accept the exact Core ``0.5.7`` six-crate set and paired 12-wheel matrix;
#. publish and verify the six Core crates in dependency order;
#. publish and verify the exact Core Python wheel set;
#. update ASSIST Python/Rust exact pins and frozen lock to public Core ``0.5.7``;
#. accept, publish, and verify ``adam_assist 0.4.0`` and the exact
   ``adam-assist==0.4.0`` wheel set; and
#. run clean no-lock Cargo 1.87/latest-stable and clean pip/current-uv
   propagation smoke tests from registry-only environments.

``adam-assist`` owns ASSIST orchestration and consumes released
``libassist-sys`` and ``librebound-sys`` directly. A tag, GitHub release,
protected-environment change, crates.io upload, or PyPI upload is a human
approval boundary and is never performed as part of source preparation.
