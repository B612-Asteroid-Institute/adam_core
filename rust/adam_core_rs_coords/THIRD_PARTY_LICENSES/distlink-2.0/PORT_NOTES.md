# DISTLINK 2.0 port provenance and adam-core integration

Adam-core does not vendor the upstream DISTLINK C++ source. The upstream
DISTLINK 2.0 implementation was downloaded from the project page on 2026-11-05
and used as the source reference for the safe-Rust port in
`src/moid_distlink.rs`. The reference implementation applies the global
critical-point MOID method of:

- Kholshevnikov & Vassiliev (1999), *Celestial Mechanics and Dynamical
  Astronomy* 75, 75–83.
- Baluev & Mikryukov (2019), *Astronomy and Computing* 27, 11–22.

The upstream copyright and MIT terms are at the start of `LICENSE.txt`.
Upstream source digests before adam-core adaptation were:

- `distlink.cpp`: `cf5430a141a536e5d9c8fc040f996009d39803a9bc0b03f27cf87d57b58733fb`
- `distlink.h`: `cc1c5a9fbaddd68df4707a8941cb43a7b2025ac7c4bc0acf8f5fef221063a3e9`
- upstream README/license: `8ef481954671522eade5119641437162dd03f02f1e52ecd98f18a7b2d300542f`

`src/moid_distlink.rs` is a source-derived, safe, pure-Rust port of the elliptic
DISTLINK 2.0 MOID path. It preserves the degree-16 critical-point polynomial,
complex-root deflation and reliability diagnostics, reversed-role retry,
direct-search fallback, anomaly witnesses, and deterministic per-pair xorshift64*
stream.
The upstream C++ was used as a development oracle only and is not compiled or
shipped in this candidate. No C ABI or foreign-language boundary remains.

The adam-core public API intentionally accepts bounded elliptic conics only.
Hyperbolic, parabolic, non-finite, and degenerate inputs produce typed Rust
errors. Scalar and batch entry points share the same solver and expose no
caller-tunable iteration or tolerance controls.
