:orphan:

Orbit Determination Pipeline (Moved)
====================================

This guide was merged into :doc:`../use_cases/orbit_determination`, the single
authoritative guide for Core 0.7 canonical OD. It documents:

* Python and standalone Rust ``run_iod``, ``run_od``, ``refine``, and
  ``evaluate_orbits`` workflows;
* exact input-table/member contracts and the three result products;
* explicit configuration, profiles, resources, and propagation providers;
* deterministic overlaps, empty input, error boundaries, provenance, and
  mutable relational orbit labels; and
* the executable Rust example
  ``rust/adam_core/examples/od_canonical_workflows.rs``.

Generated Python API pages remain under
:doc:`../reference/api/adam_core.orbit_determination`.
