# adam_core_rs_kernel_data

Kernel/data-file resolution for pure-Rust adam-core consumers (bead `personal-3uy`).

The canonical distribution channel for adam-core's SPICE/ephemeris data remains
the published PyPI data packages (`naif-de440`, `naif-leapseconds`,
`naif-eop-*`, `naif-earth-itrf93`, `mpc_obscodes`,
`jpl-small-bodies-de441-n16`). Python-hosted adam-core keeps using those
pip-installed files directly and never touches this crate.

This crate lets Rust-only consumers resolve the exact same files through a
deterministic discovery chain, per kernel:

1. **Explicit override**: `ADAM_CORE_KERNEL_<ID>` (for example
   `ADAM_CORE_KERNEL_DE440=/path/to/de440.bsp`).
2. **Installed-Python probe**: if a Python environment is available
   (`ADAM_CORE_KERNEL_PYTHON`, `$VIRTUAL_ENV/bin/python`, or `python3` on
   `PATH`), import the data packages and reuse their already-downloaded files.
   One batched subprocess, memoized per resolver.
3. **Local cache**: `$ADAM_CORE_KERNEL_CACHE`, else
   `$XDG_CACHE_HOME/adam_core/kernels`, else `~/.cache/adam_core/kernels`.
4. **Checksummed fetch**: download the exact, immutable PyPI wheel
   (`files.pythonhosted.org` URL pinned per crate release), verify its
   published SHA-256 while streaming to disk, extract the single data member
   (wheels are zip files), and atomically publish it into the cache.

Set `ADAM_CORE_KERNEL_OFFLINE=1` to forbid tier 4 (resolution fails instead of
downloading).

Provenance: the manifest in `src/lib.rs` pins each wheel's version, URL, and
SHA-256 exactly as published on PyPI; update all three together when bumping a
data package version.
